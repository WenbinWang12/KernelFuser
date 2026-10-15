#!/usr/bin/env python3
"""RoPE-aware isolated Q/K RMSNorm fusion sweep.

This runner is deliberately a forward-only proxy: it fuses Q/K RMSNorm and
rotary position embedding into the same fixed-launch attention kernel.  It
does not include projections, KV-cache updates, sliding-window masking, or a
full model.  The receipt records three comparison arms: a separate norm+RoPE
path using the same Triton attention anchor, torch.compile, and vendor SDPA.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, platform, random, statistics, subprocess, time, threading
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

MODELS = {
    "Qwen3-0.6B": (16, 8, 128), "Qwen3-1.7B": (32, 8, 128),
    "Qwen3-4B": (32, 8, 128), "Qwen3-8B": (32, 8, 128),
    "Qwen3-32B": (64, 8, 128), "Qwen3-30B-A3B": (32, 4, 128),
    "Gemma-3-4B": (8, 4, 256), "Gemma-3-12B": (16, 8, 256),
    "Gemma-3-27B": (32, 16, 128),
}
# Gemma-3 sets query_pre_attn_scalar=256 in its released configurations,
# including the D=128 27B geometry.  Qwen3 uses the usual head-dimension
# scaling.  Keeping this explicit prevents a D**-.5 shortcut from changing
# the isolated Gemma semantics.
MODEL_QUERY_PRE_ATTN_SCALAR = {
    'Gemma-3-4B': 256.0, 'Gemma-3-12B': 256.0, 'Gemma-3-27B': 256.0,
}
EPS = 1e-6

def model_scale(name, d):
    scalar=MODEL_QUERY_PRE_ATTN_SCALAR.get(name)
    return float(1.0/math.sqrt(scalar if scalar is not None else d))

@triton.jit
def norm_kernel(X, G, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0); j = tl.arange(0, BLOCK)
    x = tl.load(X + row*N + j, mask=j < N, other=0).to(tl.float32)
    g = tl.load(G + j, mask=j < N, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x*x, axis=0)/N + EPS)
    # Qwen/Gemma RMSNorm: rounded normalized activation, then BF16 weight
    # multiply (rather than one FP32 multiply followed by a single cast).
    xn=(x*r).to(Y.dtype.element_ty); gb=g.to(Y.dtype.element_ty)
    tl.store(Y + row*N + j, (xn*gb).to(Y.dtype.element_ty), mask=j < N)

@triton.jit
def attn_kernel(Q, K, V, GQ, GK, COS, SIN, O,
                BATCH: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                HQ: tl.constexpr, HK: tl.constexpr, SCALE: tl.constexpr, EPSILON: tl.constexpr,
                FUSED: tl.constexpr, CAUSAL: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1); b = tl.program_id(2)
    kh = h // (HQ // HK); mi = pid*BM + tl.arange(0, BM); dj = tl.arange(0, BLOCK_D)
    qoff = b*HQ*S*D + h*S*D + mi[:, None]*D + dj[None, :]
    qmask = (mi[:, None] < S) & (dj[None, :] < D)
    q = tl.load(Q + qoff, mask=qmask, other=0).to(tl.float32)
    if FUSED:
        g = tl.load(GQ + dj, mask=dj < D, other=0).to(tl.float32)
        r = tl.rsqrt(tl.sum(q*q, axis=1)/D + EPSILON)
        # Match HF's dtype boundary: norm output is rounded to the input
        # dtype before rotary multiplication.
        q = (q*r[:, None]).to(Q.dtype.element_ty)
        q = (q*g[None, :].to(Q.dtype.element_ty)).to(Q.dtype.element_ty)
        jp = tl.where(dj < D//2, dj + D//2, dj - D//2)
        qp = tl.load(Q + b*HQ*S*D + h*S*D + mi[:, None]*D + jp[None, :],
                     mask=(mi[:, None] < S) & (jp[None, :] < D), other=0).to(tl.float32)
        gp = tl.load(GQ + jp, mask=(jp >= 0) & (jp < D), other=0).to(tl.float32)
        # The partner half follows the same HF boundary as the primary half:
        # normalize in FP32, round the normalized activation to BF16, then
        # multiply the BF16 weight before rotary arithmetic.
        qp = (qp*r[:, None]).to(Q.dtype.element_ty)
        qp = (qp*gp[None, :].to(Q.dtype.element_ty)).to(Q.dtype.element_ty)
        c = tl.load(COS + mi[:, None]*D + dj[None, :], mask=qmask, other=1).to(Q.dtype.element_ty)
        s = tl.load(SIN + mi[:, None]*D + dj[None, :], mask=qmask, other=0).to(Q.dtype.element_ty)
        first = dj[None, :] < D//2
        p0=(q*c).to(Q.dtype.element_ty); p1=(qp*s).to(Q.dtype.element_ty)
        # Split-half rotate_half: the second half is q*c + partner*s.
        p2=(q*c).to(Q.dtype.element_ty); p3=(qp*s).to(Q.dtype.element_ty)
        q = tl.where(first, (p0-p1).to(Q.dtype.element_ty), (p2+p3).to(Q.dtype.element_ty)).to(tl.float32)
    else:
        q = q.to(tl.float32)
    acc = tl.zeros((BM, BLOCK_D), tl.float32); mx = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    for start in range(0, S, BN):
        nj = start + tl.arange(0, BN); koff = b*HK*S*D + kh*S*D + nj[:, None]*D + dj[None, :]
        kmask = (nj[:, None] < S) & (dj[None, :] < D)
        k = tl.load(K + koff, mask=kmask, other=0).to(tl.float32)
        if FUSED:
            g = tl.load(GK + dj, mask=dj < D, other=0).to(tl.float32)
            r = tl.rsqrt(tl.sum(k*k, axis=1)/D + EPSILON)
            k = (k*r[:, None]).to(K.dtype.element_ty)
            k = (k*g[None, :].to(K.dtype.element_ty)).to(K.dtype.element_ty)
            jp = tl.where(dj < D//2, dj + D//2, dj - D//2)
            kp = tl.load(K + b*HK*S*D + kh*S*D + nj[:, None]*D + jp[None, :],
                         mask=(nj[:, None] < S) & (jp[None, :] < D), other=0).to(tl.float32)
            gp = tl.load(GK + jp, mask=(jp >= 0) & (jp < D), other=0).to(tl.float32)
            kp = (kp*r[:, None]).to(K.dtype.element_ty)
            kp = (kp*gp[None, :].to(K.dtype.element_ty)).to(K.dtype.element_ty)
            c = tl.load(COS + nj[:, None]*D + dj[None, :], mask=kmask, other=1).to(K.dtype.element_ty)
            s = tl.load(SIN + nj[:, None]*D + dj[None, :], mask=kmask, other=0).to(K.dtype.element_ty)
            first = dj[None, :] < D//2
            p0=(k*c).to(K.dtype.element_ty); p1=(kp*s).to(K.dtype.element_ty)
            p2=(k*c).to(K.dtype.element_ty); p3=(kp*s).to(K.dtype.element_ty)
            k = tl.where(first, (p0-p1).to(K.dtype.element_ty), (p2+p3).to(K.dtype.element_ty)).to(tl.float32)
        scores = tl.dot(q, tl.trans(k))*SCALE
        scores = tl.where(nj[None, :] < S, scores, float('-inf'))
        if CAUSAL:
            scores = tl.where(mi[:, None] >= nj[None, :], scores, float('-inf'))
        mn = tl.maximum(mx, tl.max(scores, axis=1)); alpha = tl.exp(mx-mn); p = tl.exp(scores-mn[:, None])
        den = den*alpha + tl.sum(p, axis=1); acc = acc*alpha[:, None]
        v = tl.load(V + koff, mask=kmask, other=0).to(tl.float32)
        acc += tl.dot(p, v); mx = mn
    out = acc/den[:, None]
    tl.store(O + qoff, out.to(O.dtype.element_ty), mask=qmask)

@triton.jit
def transform_debug(X, G, COS, SIN, Y, BATCH: tl.constexpr, S: tl.constexpr,
                    D: tl.constexpr, BLOCK_D: tl.constexpr):
    row=tl.program_id(0); j=tl.arange(0,BLOCK_D); b=row//S; pos=row%S
    off=row*D+j; mask=j<D
    x=tl.load(X+off,mask=mask,other=0).to(tl.float32); g=tl.load(G+j,mask=mask,other=0).to(tl.float32)
    r=tl.rsqrt(tl.sum(x*x,axis=0)/D+1.e-6); x=x*r*g
    jp=tl.where(j < D//2,j+D//2,j-D//2); xp=tl.load(X+row*D+jp,mask=(jp>=0)&(jp<D),other=0).to(tl.float32)
    gp=tl.load(G+jp,mask=(jp>=0)&(jp<D),other=0).to(tl.float32); xp=xp*r*gp
    c=tl.load(COS+pos*D+j,mask=mask,other=1).to(tl.float32); s=tl.load(SIN+pos*D+j,mask=mask,other=0).to(tl.float32)
    first=j < D//2; y=tl.where(first,x*c-xp*s,x*c+xp*s)
    tl.store(Y+off,y.to(Y.dtype.element_ty),mask=mask)

def norm_triton(x, g, eps=EPS):
    y = torch.empty_like(x); d=x.shape[-1]
    norm_kernel[(x.numel()//d,)](x, g, y, d, eps, triton.next_power_of_2(d), num_warps=4)
    return y

def norm_torch(x, g, eps=EPS):
    xf=x.float(); xn=(xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
    return (xn*g.to(x.dtype)).to(x.dtype)

def rope_torch(x, cos, sin, *, fp32=False):
    """HF rotate_half (split-half) with the same dtype boundaries as HF.

    Qwen3RMSNorm returns the normalized activation in the input dtype before
    multiplying the BF16 weight.  The rotary module then performs the two
    elementwise products and the add/subtract in that dtype.  ``fp32=True``
    is reserved for the independent oracle and intentionally keeps every
    operation in float32.
    """
    d=x.shape[-1]; half=d//2
    dtype=torch.float32 if fp32 else x.dtype
    xx=x.float() if fp32 else x
    cc=cos.float() if fp32 else cos.to(dtype)
    ss=sin.float() if fp32 else sin.to(dtype)
    # Accept either [S,D] or [B,S,D] tables.  The heads dimension broadcasts.
    if cc.ndim == 2: cc=cc[None,None]
    elif cc.ndim == 3: cc=cc[:,None]
    if ss.ndim == 2: ss=ss[None,None]
    elif ss.ndim == 3: ss=ss[:,None]
    a=xx[...,:half]; b=xx[...,half:]
    c=cc[...,:half]; s=ss[...,:half]
    # Explicit casts preserve HF's BF16 rounding points.  Torch's BF16 eager
    # kernels may otherwise promote the expression to FP32 on some versions.
    p0=(a*c).to(dtype); p1=(b*s).to(dtype)
    p2=(b*c).to(dtype); p3=(a*s).to(dtype)
    y=torch.empty_like(xx)
    y[...,:half]=(p0-p1).to(dtype); y[...,half:]=(p2+p3).to(dtype)
    return y.to(x.dtype) if not fp32 else y

def expand(x, h): return x if x.shape[1]==h else x.repeat_interleave(h//x.shape[1], dim=1)

def make_inputs(cfg, b, s, seed):
    torch.manual_seed(seed); h,hk,d=cfg; shape=lambda heads:(b,heads,s,d)
    q=torch.randn(shape(h),device='cuda',dtype=torch.bfloat16)*.5
    k=torch.randn(shape(hk),device='cuda',dtype=torch.bfloat16)*.5
    v=torch.randn(shape(hk),device='cuda',dtype=torch.bfloat16)*.5
    gq=(1+torch.randn((d,),device='cuda',dtype=torch.bfloat16)*.05).contiguous()
    gk=(1+torch.randn((d,),device='cuda',dtype=torch.bfloat16)*.05).contiguous()
    pos=torch.arange(s,device='cuda',dtype=torch.float32); inv=1./(10000**(torch.arange(0,d,2,device='cuda',dtype=torch.float32)/d))
    ang=pos[:,None]*inv[None,:]
    # Duplicate the half-dimensional frequency table in the split-half layout;
    # HF rotary tables are cast to the activation dtype before multiplication.
    return q.contiguous(),k.contiguous(),v.contiguous(),gq,gk,torch.cat([torch.cos(ang),torch.cos(ang)],dim=1).to(q.dtype),torch.cat([torch.sin(ang),torch.sin(ang)],dim=1).to(q.dtype)

def fused(q,k,v,gq,gk,cos,sin,causal,eps=EPS,scale=None,norm_style='qwen3'):
    """Run the fused norm+RoPE attention proxy.

    ``eps`` and ``scale`` are explicit API parameters so model runners can
    pass the checkpoint values.  The current Triton specialization is built
    for the Qwen/Gemma epsilon (1e-6) and attention scale D**-1/2.
    """
    # Triton specialization stores one position table [S,D].  HF may expose
    # an equivalent [B,S,D] or [B,1,S,D] broadcast table; all rows in the
    # isolated test share positions, so selecting the first broadcast slice is
    # exact and avoids silently treating the batch dimension as positions.
    if cos.ndim == 4: cos=cos[:,0]
    if sin.ndim == 4: sin=sin[:,0]
    if cos.ndim == 3: cos=cos[0]
    if sin.ndim == 3: sin=sin[0]
    cos=cos.to(q.dtype).contiguous(); sin=sin.to(q.dtype).contiguous()
    b,h,s,d=q.shape; o=torch.empty_like(q); bd=triton.next_power_of_2(d)
    attn_kernel[(triton.cdiv(s,32),h,b)](q,k,v,gq,gk,cos,sin,o,b,s,d,h,k.shape[1],float(d**-.5 if scale is None else scale),float(eps),True,causal,32,32,bd,num_warps=4,num_stages=1)
    return o

def anchor(q,k,v,cos,sin,causal,scale=None):
    b,h,s,d=q.shape; o=torch.empty_like(q); bd=triton.next_power_of_2(d)
    attn_kernel[(triton.cdiv(s,32),h,b)](q,k,v,q,k,cos,sin,o,b,s,d,h,k.shape[1],float(d**-.5 if scale is None else scale),EPS,False,causal,32,32,bd,num_warps=4,num_stages=1)
    return o

def separate(q,k,v,gq,gk,cos,sin,causal,eps=EPS,scale=None,norm_style='qwen3'):
    qn=rope_torch(norm_triton(q,gq,eps),cos,sin); kn=rope_torch(norm_triton(k,gk,eps),cos,sin)
    return anchor(qn,kn,v,cos,sin,causal,scale)

def vendor(q,k,v,gq,gk,cos,sin,causal,eps=EPS,scale=None,norm_style='qwen3'):
    # Vendor SDPA arm uses the framework RMSNorm path, unlike the primary
    # same-anchor baseline (which uses native Triton RMSNorm).
    qn=rope_torch(norm_torch(q,gq,eps),cos,sin); kn=rope_torch(norm_torch(k,gk,eps),cos,sin)
    return F.scaled_dot_product_attention(qn,expand(kn,q.shape[1]),expand(v,q.shape[1]),is_causal=causal,scale=scale)

def compiled_fn(q,k,v,gq,gk,cos,sin,causal,eps=EPS,scale=None,norm_style='qwen3'):
    # TorchInductor must compile the actual torch RMSNorm + RoPE path.  Using
    # norm_triton here would silently make the compiled arm a duplicate of the
    # isolated anchor rather than the requested framework baseline.
    qn=rope_torch(norm_torch(q,gq,eps),cos,sin); kn=rope_torch(norm_torch(k,gk,eps),cos,sin)
    return F.scaled_dot_product_attention(qn,expand(kn,q.shape[1]),expand(v,q.shape[1]),is_causal=causal,scale=scale)

def stats(fn, rounds, samples):
    vals=[]
    for _ in range(rounds):
        for _ in range(10): fn()
        torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True)
        st.record();
        for _ in range(samples): fn()
        en.record(); en.synchronize(); vals.append(st.elapsed_time(en)/samples)
    return {'p50_ms':statistics.median(vals),'rounds_ms':vals,'n':len(vals)*samples}

def capture(fn, warmups=20):
    """Fresh CUDA-graph capture for one timing arm.

    The callable closes over one fixed input tuple and must not allocate
    externally visible state.  A new graph is captured for every arm and
    every round; this prevents allocator/cache state from making a paired
    comparison look like a launch-timing comparison.
    """
    for _ in range(warmups): fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=fn()
    torch.cuda.synchronize()
    return graph, out

def paired(a,b,rounds,samples,seed=1701):
    """Randomized paired CUDA-graph timing of ``a`` and ``b``.

    The caller normally passes candidate as ``b``.  We retain the generic
    names for receipt compatibility, while recording the arm order for every
    sample and capturing both graphs independently in each round.
    """
    if rounds < 1 or samples < 2: raise ValueError('paired timing requires rounds>=1 and samples>=2')
    out={'a':[],'b':[],'arm_order':[],'mode':'cuda_graph_replay_paired_interleaved','paired':True,
         'warmups_per_capture':20,'fresh_graph_captures':rounds,'samples_per_round':samples,'seed':seed}
    rng=random.Random(int(seed)+rounds*7919+samples)
    for r in range(rounds):
        ga,oa=capture(a); gb,ob=capture(b)
        for _ in range(10): ga.replay(); gb.replay()
        torch.cuda.synchronize(); orders=[]; aa=[]; bb=[]
        for _ in range(samples):
            first_a=bool(rng.getrandbits(1)); orders.append('a_first' if first_a else 'b_first')
            seq=((ga,aa,'a'),(gb,bb,'b')) if first_a else ((gb,bb,'b'),(ga,aa,'a'))
            for graph,vals,key in seq:
                st=torch.cuda.Event(enable_timing=True); en=torch.cuda.Event(enable_timing=True)
                st.record(); graph.replay(); en.record(); en.synchronize(); vals.append(float(st.elapsed_time(en)))
        out['a'].append(aa); out['b'].append(bb); out['arm_order'].append(orders)
        del ga,gb,oa,ob; torch.cuda.empty_cache()
    # p50 is retained because older receipts use it; per-round samples remain
    # the primary audit evidence.
    out['a_p50_ms']=[statistics.median(x) for x in out['a']]
    out['b_p50_ms']=[statistics.median(x) for x in out['b']]
    out['ratio_median']=statistics.median([x/y for x,y in zip(out['a_p50_ms'],out['b_p50_ms'])])
    return out

def measure_pair(baseline_fn,candidate_fn,rounds,samples,seed=1701):
    """Named API used by the Qwen E2 and A/A runners."""
    return paired(baseline_fn,candidate_fn,rounds,samples,seed)

def metric(x,y):
    a=x.float(); b=y.float(); finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff=(a-b); den=b.norm().clamp_min(1e-12)
    rl2=float(diff.norm()/den) if finite else float('inf')
    close=bool(torch.allclose(a,b,atol=.025,rtol=.025)) if finite else False
    return {'max_abs':float(diff.abs().max()) if finite else float('inf'),'relative_l2':rl2,
            'finite':finite,'pass':bool(finite and close and rl2<=.025)}

def oracle_fp32(q,k,v,gq,gk,cos,sin,causal,scale=None,eps=EPS):
    """Independent true-FP32 oracle for every arm's correctness check."""
    qf=q.float(); kf=k.float(); vf=v.float(); gqf=gq.float(); gkf=gk.float()
    qn=(qf*torch.rsqrt(qf.square().mean(-1,keepdim=True)+eps)*gqf[...,None,:] if gqf.ndim==1 else qf*torch.rsqrt(qf.square().mean(-1,keepdim=True)+eps)*gqf)
    kn=(kf*torch.rsqrt(kf.square().mean(-1,keepdim=True)+eps)*gkf[...,None,:] if gkf.ndim==1 else kf*torch.rsqrt(kf.square().mean(-1,keepdim=True)+eps)*gkf)
    # fp32=True keeps all rotation arithmetic in the oracle dtype.
    qn=rope_torch(qn,cos,sin,fp32=True); kn=rope_torch(kn,cos,sin,fp32=True)
    return F.scaled_dot_product_attention(qn,expand(kn,q.shape[1]),expand(vf,q.shape[1]),is_causal=causal,scale=scale)

def gpu_snapshot():
    """Snapshot the selected physical GPU, following the E1 QK protocol.

    ``CUDA_VISIBLE_DEVICES`` may contain logical ordinals, so the monitor first
    maps them to UUIDs and then inspects only those UUIDs.  A host-wide
    ``len(all apps) <= 1`` test is invalid on multi-GPU OSU nodes because a
    process on another card is irrelevant (and a process on the selected card
    can be missed by that shortcut).
    """
    try:
        visible=os.getenv('CUDA_VISIBLE_DEVICES','').strip()
        logical=[int(x.strip()) for x in visible.split(',') if x.strip()] if visible else [torch.cuda.current_device()]
        q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        by_index={int(x.split(',')[0].strip()):x.split(',')[1].strip() for x in q.stdout.splitlines() if ',' in x}
        selected=[by_index.get(i) for i in logical if by_index.get(i)]
        app=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10)
        own=os.getpid(); rows=[]; foreign=[]
        for line in app.stdout.splitlines():
            parts=[x.strip() for x in line.split(',')]
            if len(parts)<2 or parts[0] not in selected: continue
            try: pid=int(parts[1])
            except ValueError: continue
            item={'gpu_uuid':parts[0],'pid':pid,'process_name':parts[2] if len(parts)>2 else '', 'used_memory':parts[3] if len(parts)>3 else '', 'self':pid==own}
            rows.append(item)
            if pid!=own: foreign.append(item)
        return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'visible_gpu':visible,'logical_devices':logical,'selected_uuids':selected,'pid':own,'apps':rows,'foreign_apps':foreign,'exclusive':bool(selected) and not foreign}
    except Exception as e: return {'error':repr(e),'exclusive':False}

class ExclusiveMonitor:
    """One-second monitor for the selected physical GPU UUIDs."""
    def __init__(self, interval=1.0):
        self.interval=interval; self.stop=threading.Event(); self.records=[]
        self.thread=threading.Thread(target=self._loop,daemon=True)
    def _loop(self):
        while not self.stop.is_set():
            self.records.append(gpu_snapshot()); self.stop.wait(self.interval)
    def start(self): self.thread.start(); return self
    def finish(self):
        self.stop.set(); self.thread.join(timeout=max(5.0, self.interval*3)); return self.records

def capture_trace(fn,path,label):
    from fusion_harness import capture_gpu_trace
    return capture_gpu_trace(fn,path,label)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',default=','.join(MODELS)); ap.add_argument('--sizes',default='128,256,384,512'); ap.add_argument('--batches',default='1,8,32'); ap.add_argument('--causal-modes',default='false,true'); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--samples',type=int,default=40); ap.add_argument('--output',required=True); ap.add_argument('--trace-dir',default=''); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if a.smoke: a.models='Qwen3-8B'; a.sizes='128'; a.batches='1'; a.causal_modes='false'; a.rounds=2; a.samples=20
    sizes=[int(x) for x in a.sizes.split(',')]; batches=[int(x) for x in a.batches.split(',')]; modes=[x.strip().lower()=='true' for x in a.causal_modes.split(',')]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    trace_root=Path(a.trace_dir) if a.trace_dir else out.parent/'traces'
    rec={'schema':'fusionagent-e1-rope-v2','scope':'isolated Q/K RMSNorm+RoPE into attention; no projection/KV-cache/sliding-window/full model','candidate':'native Triton RMSNorm+RoPE fused attention','baseline':'same fixed Triton attention anchor with separate norm and RoPE','additional_arms':['torch.compile SDPA','vendor SDPA'],'environment':{'host':platform.node(),'gpu':torch.cuda.get_device_name(),'visible_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda},'protocol':{'S':sizes,'B':batches,'causal_modes':modes,'warmup':20,'rounds':a.rounds,'samples':a.samples,'paired_timing':'candidate paired independently with each baseline; randomized arm order per sample; CUDA graphs freshly captured each round','correctness':'all arms against independent true-FP32 oracle; finite and relative-L2 gate','retest':'independent fresh CUDA-graph paired retest for every candidate/baseline pair','gpu_exclusivity':'selected physical UUID only; monitor at one-second intervals; foreign PID invalidates row','traces':'separate, fused, torch.compile, and vendor arms'},'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'models':{},'status':'running','publication_eligible':False}
    for name in [x.strip() for x in a.models.split(',') if x.strip()]:
        cfg=MODELS[name]; scale=model_scale(name,cfg[2]); rec['models'][name]={'geometry':{'Hq':cfg[0],'Hkv':cfg[1],'D':cfg[2],'attention_scale':scale,'eps':EPS},'rows':[]}
        for s in sizes:
            for causal in modes:
                for b in batches:
                    row={'B':b,'S':s,'causal':causal,'timed_dtype':'bfloat16','correctness':{'seeds':[],'pass':False},'errors':[],'traces':[]}
                    row['gpu_exclusivity']={'before':gpu_snapshot()}
                    mon=ExclusiveMonitor(); mon.start()
                    try:
                        torch._dynamo.reset(); torch._dynamo.config.cache_size_limit=512; row['torchdynamo_reset_before_compile']=True
                        # Correctness uses three independent BF16 inputs and one
                        # true FP32 oracle per seed; no arm is compared only to
                        # another approximate arm.
                        for seed in (1701,2803,3907):
                            q,k,v,gq,gk,c,sn=make_inputs(cfg,b,s,seed)
                            ref=oracle_fp32(q,k,v,gq,gk,c,sn,causal,scale=scale)
                            cand=fused(q,k,v,gq,gk,c,sn,causal,scale=scale); sep=separate(q,k,v,gq,gk,c,sn,causal,scale=scale); vend=vendor(q,k,v,gq,gk,c,sn,causal,scale=scale)
                            row['correctness']['seeds'].append({'seed':seed,'separate':metric(sep,ref),'fused':metric(cand,ref),'vendor':metric(vend,ref),'finite':bool(torch.isfinite(ref).all())})
                        row['correctness']['pass']=all(x[k]['pass'] for x in row['correctness']['seeds'] for k in ('separate','fused','vendor'))
                        q,k,v,gq,gk,c,sn=make_inputs(cfg,b,s,1701+s*17+b+int(causal))
                        sep=lambda:separate(q,k,v,gq,gk,c,sn,causal,scale=scale); cand=lambda:fused(q,k,v,gq,gk,c,sn,causal,scale=scale); vend=lambda:vendor(q,k,v,gq,gk,c,sn,causal,scale=scale)
                        ext=torch.compile(compiled_fn,fullgraph=True,dynamic=False)
                        comp_records=[]
                        for seed in (1701,2803,3907):
                            cq,ck,cv,cgq,cgk,cc,csn=make_inputs(cfg,b,s,seed)
                            crep=ext(cq,ck,cv,cgq,cgk,cc,csn,causal,EPS,scale,'qwen3'); cref=oracle_fp32(cq,ck,cv,cgq,cgk,cc,csn,causal,scale=scale)
                            cm=metric(crep,cref); comp_records.append({'seed':seed,'compiled_vs_fp32':cm,'pass':cm['pass']})
                        row['torchinductor_correctness']={'seeds':comp_records,'pass':all(x['pass'] for x in comp_records)}; row['torchinductor_status']='compiled'
                        comp=lambda:ext(q,k,v,gq,gk,c,sn,causal,EPS,scale,'qwen3')
                        if not row['correctness']['pass'] or not row['torchinductor_correctness']['pass']: raise RuntimeError('FP32 correctness gate failed; timing suppressed')
                        p_sep=paired(sep,cand,a.rounds,a.samples,seed=1101); p_comp=paired(comp,cand,a.rounds,a.samples,seed=2203); p_vend=paired(vend,cand,a.rounds,a.samples,seed=3307)
                        # Every pair is independently recaptured and rerun.  The
                        # receipt keeps both initial and retest evidence.
                        r_sep=paired(sep,cand,a.rounds,a.samples,seed=101101); r_comp=paired(comp,cand,a.rounds,a.samples,seed=102203); r_vend=paired(vend,cand,a.rounds,a.samples,seed=103307)
                        row['timing']={'separate_vs_fused':{'initial':p_sep,'retest':r_sep},'compiled_vs_fused':{'initial':p_comp,'retest':r_comp},'vendor_vs_fused':{'initial':p_vend,'retest':r_vend}}
                        base_ms=[p_sep['a_p50_ms'],p_comp['a_p50_ms'],p_vend['a_p50_ms']]; fused_ms=[p_sep['b_p50_ms'],p_comp['b_p50_ms'],p_vend['b_p50_ms']]
                        row['speedup_vs_separate']=statistics.median(p_sep['a_p50_ms'])/statistics.median(p_sep['b_p50_ms']); row['speedup_vs_fastest']=min(map(statistics.median,base_ms))/max(map(statistics.median,fused_ms))
                        labels=[('separate',sep),('fused',cand),('torch_compile',comp),('vendor',vend)]
                        for label,fn in labels:
                            path=trace_root/name/f'B{b}_S{s}_C{int(causal)}_{label}.chrome_trace.json.gz'; row['traces'].append(capture_trace(fn,path,f'e1_rope.{label}'))
                    except Exception as e: row['errors'].append({'stage':'execution','error':repr(e)})
                    row['gpu_exclusivity']['observations']=mon.finish(); row['gpu_exclusivity']['after']=gpu_snapshot()
                    gate_reasons=[]
                    if row['errors']: gate_reasons.append('execution errors recorded')
                    if row['correctness'].get('pass') is not True: gate_reasons.append('all native arms did not pass true-FP32 correctness')
                    if row.get('torchinductor_correctness',{}).get('pass') is not True: gate_reasons.append('TorchInductor arm did not pass true-FP32 correctness')
                    if not row['gpu_exclusivity']['before'].get('exclusive') or not row['gpu_exclusivity']['after'].get('exclusive') or not row['gpu_exclusivity']['observations'] or not all(x.get('exclusive') for x in row['gpu_exclusivity']['observations']): gate_reasons.append('selected physical GPU exclusivity failed or was not continuously observed')
                    if len(row['traces'])!=4: gate_reasons.append('one or more arm traces are missing')
                    for pair_name,pair_rec in row.get('timing',{}).items():
                        if not isinstance(pair_rec,dict) or 'initial' not in pair_rec or 'retest' not in pair_rec: gate_reasons.append(f'{pair_name} lacks independent retest')
                    row['publication_gate']={'pass':not gate_reasons,'reasons':gate_reasons}
                    row['status']='tested_pass' if row['publication_gate']['pass'] else 'tested_invalid'
                    rec['models'][name]['rows'].append(row); out.write_text(json.dumps(rec,indent=2)); print(name,b,s,causal,row['status'],row.get('speedup_vs_fastest'),flush=True)
    rec_rows=[r for m in rec['models'].values() for r in m['rows']]; rec['status']='tested_pass' if rec_rows and all(r.get('status')=='tested_pass' for r in rec_rows) else 'tested_invalid'; rec['publication_eligible']=rec['status']=='tested_pass'; out.write_text(json.dumps(rec,indent=2)); print(json.dumps({'output':str(out),'status':rec['status']},indent=2))



# H100 refinement APIs and development-only verification.
import sys
import csv
from fusion_harness import paired as device_paired, compiler_evidence
@triton.jit  # noqa: TR001 -- explicit configurations measured by ablation runner
def bf16_attention_kernel(Q, K, V, GQ, GK, COS, SIN, O,  # noqa: TR001 -- explicit benchmark ablations
                BATCH: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                HQ: tl.constexpr, HK: tl.constexpr, SCALE: tl.constexpr, EPSILON: tl.constexpr,
                FUSED: tl.constexpr, CAUSAL: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BLOCK_D: tl.constexpr,
                BF16_DOT: tl.constexpr, CAUSAL_BOUND: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1); b = tl.program_id(2)
    kh = h // (HQ // HK); mi = pid*BM + tl.arange(0, BM); dj = tl.arange(0, BLOCK_D)
    qoff = b*HQ*S*D + h*S*D + mi[:, None]*D + dj[None, :]
    qmask = (mi[:, None] < S) & (dj[None, :] < D)
    q = tl.load(Q + qoff, mask=qmask, other=0).to(tl.float32)
    if FUSED:
        g = tl.load(GQ + dj, mask=dj < D, other=0).to(tl.float32)
        r = tl.rsqrt(tl.sum(q*q, axis=1)/D + EPSILON)
        # Match HF's dtype boundary: norm output is rounded to the input
        # dtype before rotary multiplication.
        q = (q*r[:, None]).to(Q.dtype.element_ty)
        q = (q*g[None, :].to(Q.dtype.element_ty)).to(Q.dtype.element_ty)
        jp = tl.where(dj < D//2, dj + D//2, dj - D//2)
        qp = tl.load(Q + b*HQ*S*D + h*S*D + mi[:, None]*D + jp[None, :],
                     mask=(mi[:, None] < S) & (jp[None, :] < D), other=0).to(tl.float32)
        gp = tl.load(GQ + jp, mask=(jp >= 0) & (jp < D), other=0).to(tl.float32)
        # The partner half follows the same HF boundary as the primary half:
        # normalize in FP32, round the normalized activation to BF16, then
        # multiply the BF16 weight before rotary arithmetic.
        qp = (qp*r[:, None]).to(Q.dtype.element_ty)
        qp = (qp*gp[None, :].to(Q.dtype.element_ty)).to(Q.dtype.element_ty)
        c = tl.load(COS + mi[:, None]*D + dj[None, :], mask=qmask, other=1).to(Q.dtype.element_ty)
        s = tl.load(SIN + mi[:, None]*D + dj[None, :], mask=qmask, other=0).to(Q.dtype.element_ty)
        first = dj[None, :] < D//2
        p0=(q*c).to(Q.dtype.element_ty); p1=(qp*s).to(Q.dtype.element_ty)
        # Split-half rotate_half: the second half is q*c + partner*s.
        p2=(q*c).to(Q.dtype.element_ty); p3=(qp*s).to(Q.dtype.element_ty)
        q = tl.where(first, (p0-p1).to(Q.dtype.element_ty), (p2+p3).to(Q.dtype.element_ty)).to(tl.float32)
    else:
        q = q.to(tl.float32)
    acc = tl.zeros((BM, BLOCK_D), tl.float32); mx = tl.full((BM,), float('-inf'), tl.float32)
    den = tl.zeros((BM,), tl.float32)
    end = S
    if CAUSAL and CAUSAL_BOUND:
        end = tl.minimum((pid + 1) * BM, S)
    for start in range(0, end, BN):
        nj = start + tl.arange(0, BN); koff = b*HK*S*D + kh*S*D + nj[:, None]*D + dj[None, :]
        kmask = (nj[:, None] < S) & (dj[None, :] < D)
        k = tl.load(K + koff, mask=kmask, other=0).to(tl.float32)
        if FUSED:
            key_g = tl.load(GK + dj, mask=dj < D, other=0).to(tl.float32)
            kr = tl.rsqrt(tl.sum(k*k, axis=1)/D + EPSILON)
            k = (k*kr[:, None]).to(K.dtype.element_ty)
            k = (k*key_g[None, :].to(K.dtype.element_ty)).to(K.dtype.element_ty)
            key_jp = tl.where(dj < D//2, dj + D//2, dj - D//2)
            kp = tl.load(K + b*HK*S*D + kh*S*D + nj[:, None]*D + key_jp[None, :],
                         mask=(nj[:, None] < S) & (key_jp[None, :] < D), other=0).to(tl.float32)
            key_gp = tl.load(GK + key_jp, mask=(key_jp >= 0) & (key_jp < D), other=0).to(tl.float32)
            kp = (kp*kr[:, None]).to(K.dtype.element_ty)
            kp = (kp*key_gp[None, :].to(K.dtype.element_ty)).to(K.dtype.element_ty)
            key_c = tl.load(COS + nj[:, None]*D + dj[None, :], mask=kmask, other=1).to(K.dtype.element_ty)
            key_s = tl.load(SIN + nj[:, None]*D + dj[None, :], mask=kmask, other=0).to(K.dtype.element_ty)
            key_first = dj[None, :] < D//2
            key_p0=(k*key_c).to(K.dtype.element_ty); key_p1=(kp*key_s).to(K.dtype.element_ty)
            key_p2=(k*key_c).to(K.dtype.element_ty); key_p3=(kp*key_s).to(K.dtype.element_ty)
            k = tl.where(key_first, (key_p0-key_p1).to(K.dtype.element_ty), (key_p2+key_p3).to(K.dtype.element_ty)).to(tl.float32)
        if BF16_DOT:
            scores = tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16)), input_precision="ieee")*SCALE  # noqa: TR011 -- precision explicit via modern API
        else:
            scores = tl.dot(q, tl.trans(k), input_precision="tf32")*SCALE  # noqa: TR011 -- precision explicit via modern API
        scores = tl.where(nj[None, :] < S, scores, float('-inf'))
        if CAUSAL:
            scores = tl.where(mi[:, None] >= nj[None, :], scores, float('-inf'))
        mn = tl.maximum(mx, tl.max(scores, axis=1)); alpha = tl.exp(mx-mn); p = tl.exp(scores-mn[:, None])
        den = den*alpha + tl.sum(p, axis=1); acc = acc*alpha[:, None]
        v = tl.load(V + koff, mask=kmask, other=0).to(tl.float32)
        if BF16_DOT:
            acc += tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), input_precision="ieee")  # noqa: TR011 -- precision explicit via modern API
        else:
            acc += tl.dot(p, v, input_precision="tf32")  # noqa: TR011 -- precision explicit via modern API
        mx = mn
    out = acc/den[:, None]
    tl.store(O + qoff, out.to(O.dtype.element_ty), mask=qmask)


def launch_bf16_attention(inputs, out, causal, *, bf16_dot=True, causal_bound=True, bm=32, bn=32):
    q,k,v,gq,gk,cos,sin=inputs
    b,h,s,d=q.shape
    return bf16_attention_kernel[(triton.cdiv(s,bm),h,b)](
        q,k,v,gq,gk,cos,sin,out,b,s,d,h,k.shape[1],d**-.5,1e-6,
        True,causal,bm,bn,triton.next_power_of_2(d),bf16_dot,causal_bound,
        num_warps=4,num_stages=1)




@triton.jit
def prepare_qk_kernel(Q, K, GQ, GK, COS, SIN, QN, KN,
                       QROWS: tl.constexpr, S: tl.constexpr, D: tl.constexpr):
    row=tl.program_id(0)
    is_q=row<QROWS
    local=tl.where(is_q,row,row-QROWS)
    xp=tl.where(is_q,Q,K)
    gp=tl.where(is_q,GQ,GK)
    yp=tl.where(is_q,QN,KN)
    j=tl.arange(0,D)
    x=tl.load(xp+local*D+j).to(tl.float32)  # noqa: TR003 -- arange spans exactly D, grid spans complete rows
    inverse=tl.rsqrt(tl.sum(x*x,0)/D+1e-6)
    gamma=tl.load(gp+j)  # noqa: TR003 -- arange spans exactly D, grid spans complete rows
    norm=(x*inverse).to(tl.bfloat16)
    norm=(norm*gamma).to(tl.bfloat16)
    jp=tl.where(j<D//2,j+D//2,j-D//2)
    partner=tl.load(xp+local*D+jp).to(tl.float32)  # noqa: TR003 -- arange spans exactly D, grid spans complete rows
    partner=(partner*inverse).to(tl.bfloat16)
    partner=(partner*tl.load(gp+jp)).to(tl.bfloat16)  # noqa: TR003 -- arange spans exactly D, grid spans complete rows
    c=tl.load(COS+(local%S)*D+j)  # noqa: TR003 -- arange spans exactly D, grid spans complete rows
    s=tl.load(SIN+(local%S)*D+j)  # noqa: TR003 -- arange spans exactly D, grid spans complete rows
    primary=(norm*c).to(tl.bfloat16)
    rotated=(partner*s).to(tl.bfloat16)
    out=tl.where(j<D//2,(primary-rotated).to(tl.bfloat16),(primary+rotated).to(tl.bfloat16))
    tl.store(yp+local*D+j,out)


def prepare_qk(inputs, qn, kn):
    q,k,v,gq,gk,cos,sin=inputs
    d=q.shape[-1]
    return prepare_qk_kernel[((q.numel()+k.numel())//d,)](
        q,k,gq,gk,cos,sin,qn,kn,q.numel()//d,q.shape[2],d,num_warps=4)




@triton.jit
def _rows(BASE, GAMMA, COS, SIN, POS, VALID,
           R: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
           HF_NORM: tl.constexpr, GATHER: tl.constexpr):
    tl.static_assert(D == 128 or D == 256, 'Only exact D=128/256 rows are supported')
    j=tl.arange(0,D)
    x=tl.load(BASE[:,None]+j[None,:],VALID[:,None],0).to(tl.float32)  # noqa: TR003 -- j spans exactly D; rows are masked
    g=tl.load(GAMMA[:,None]+j[None,:]).to(tl.float32)  # noqa: TR003 -- each gamma pointer addresses a complete D-element vector
    inverse=tl.rsqrt(tl.sum(x*x,1)/D+1e-6)
    if HF_NORM:
        y=(x*inverse[:,None]).to(tl.bfloat16)
        y=(y*g.to(tl.bfloat16)).to(tl.bfloat16)
    else:
        y=(x*inverse[:,None]*g).to(tl.bfloat16)
    if ROPE:
        jp=tl.where(j<D//2,j+D//2,j-D//2)
        if GATHER:
            partner=tl.gather(y,tl.broadcast_to(jp[None,:],(R,D)),axis=1)
        else:
            xp=tl.load(BASE[:,None]+jp[None,:],VALID[:,None],0).to(tl.float32)  # noqa: TR003 -- jp permutes [0,D); rows are masked
            gp=tl.load(GAMMA[:,None]+jp[None,:]).to(tl.float32)  # noqa: TR003 -- jp permutes a complete gamma vector
            if HF_NORM:
                partner=(xp*inverse[:,None]).to(tl.bfloat16)
                partner=(partner*gp.to(tl.bfloat16)).to(tl.bfloat16)
            else:
                partner=(xp*inverse[:,None]*gp).to(tl.bfloat16)
        c=tl.load(COS+POS[:,None]*D+j[None,:],VALID[:,None],1)  # noqa: TR003 -- exact D columns; position rows are masked
        s=tl.load(SIN+POS[:,None]*D+j[None,:],VALID[:,None],0)  # noqa: TR003 -- exact D columns; position rows are masked
        primary=(y*c).to(tl.bfloat16)
        rotated=(partner*s).to(tl.bfloat16)
        y=tl.where(j[None,:]<D//2,(primary-rotated).to(tl.bfloat16),
                    (primary+rotated).to(tl.bfloat16))
    return y


@triton.jit
def prepare_kernel(Q,K,GQ,GK,COS,SIN,QN,KN,
                    QROWS: tl.constexpr, KROWS: tl.constexpr, S: tl.constexpr,
                    D: tl.constexpr, R: tl.constexpr, ROPE: tl.constexpr,
                    HF_NORM: tl.constexpr, ONLY_K: tl.constexpr):
    row=tl.program_id(0)*R+tl.arange(0,R)
    if ONLY_K:
        local=row;xp=K;gp=GK;yp=KN;valid=row<KROWS
        bases=xp+local*D
        gammas=gp+tl.zeros((R,),tl.int32)
        outputs=yp+local*D
    else:
        isq=row<QROWS
        local=tl.where(isq,row,row-QROWS)
        bases=tl.where(isq,Q+local*D,K+local*D)
        gammas=tl.where(isq,GQ,GK)
        outputs=tl.where(isq,QN+local*D,KN+local*D)
        valid=row<QROWS+KROWS
    value=_rows(bases,gammas,COS,SIN,local%S,valid,R,D,ROPE,HF_NORM,True)
    j=tl.arange(0,D)
    tl.store(outputs[:,None]+j[None,:],value,valid[:,None])


@triton.jit
def attention_v2_kernel(Q,K,V,GQ,GK,COS,SIN,O,
                         S: tl.constexpr,D: tl.constexpr,HQ: tl.constexpr,HK: tl.constexpr,
                         SCALE: tl.constexpr,CAUSAL: tl.constexpr,
                         FUSE_Q: tl.constexpr,FUSE_K: tl.constexpr,
                         ROPE: tl.constexpr,HF_NORM: tl.constexpr,GATHER: tl.constexpr,
                         BM: tl.constexpr,BN: tl.constexpr):
    pid=tl.program_id(0);h=tl.program_id(1);b=tl.program_id(2)
    kh=h//(HQ//HK)
    mi=pid*BM+tl.arange(0,BM)
    j=tl.arange(0,D)
    qb=Q+(b*HQ*S+h*S+mi)*D
    if FUSE_Q:
        q=_rows(qb,GQ+tl.zeros((BM,),tl.int32),COS,SIN,mi,mi<S,BM,D,ROPE,HF_NORM,GATHER)
    else:
        q=tl.load(qb[:,None]+j[None,:],mi[:,None]<S,0)  # noqa: TR003 -- exact D columns; sequence rows are masked
    acc=tl.zeros((BM,D),tl.float32)
    maximum=tl.full((BM,),float('-inf'),tl.float32)
    denominator=tl.zeros((BM,),tl.float32)
    end=S
    if CAUSAL: end=tl.minimum((pid+1)*BM,S)
    for start in range(0,end,BN):
        ni=start+tl.arange(0,BN)
        kb=K+(b*HK*S+kh*S+ni)*D
        if FUSE_K:
            k=_rows(kb,GK+tl.zeros((BN,),tl.int32),COS,SIN,ni,ni<S,BN,D,ROPE,HF_NORM,GATHER)
        else:
            k=tl.load(kb[:,None]+j[None,:],ni[:,None]<S,0)  # noqa: TR003 -- exact D columns; sequence rows are masked
        scores=tl.dot(q,tl.trans(k),input_precision='ieee')*(SCALE*1.4426950408889634)  # noqa: TR011 -- modern precision API
        scores=tl.where(ni[None,:]<S,scores,float('-inf'))
        if CAUSAL: scores=tl.where(mi[:,None]>=ni[None,:],scores,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(scores,1))
        alpha=tl.exp2(maximum-newmax)
        probability=tl.exp2(scores-newmax[:,None])
        denominator=denominator*alpha+tl.sum(probability,1)
        acc=acc*alpha[:,None]
        vb=V+(b*HK*S+kh*S+ni)*D
        v=tl.load(vb[:,None]+j[None,:],ni[:,None]<S,0)  # noqa: TR003 -- exact D columns; sequence rows are masked
        acc=tl.dot(probability.to(tl.bfloat16),v,acc,input_precision='ieee')  # noqa: TR011 -- modern precision API
        maximum=newmax
    tl.store(O+(b*HQ*S+h*S+mi[:,None])*D+j[None,:],
             (acc/denominator[:,None]).to(O.dtype.element_ty),mi[:,None]<S)


def prepare_attention(inputs,qn,kn,*,only_k=False,rows=4,rope=True,hf_norm=True):
    q,k,v,gq,gk,cos,sin=inputs
    s,d=q.shape[-2:];qrows=q.numel()//d;krows=k.numel()//d
    count=krows if only_k else qrows+krows
    return prepare_kernel[(triton.cdiv(count,rows),)](
        q,k,gq,gk,cos,sin,qn,kn,qrows,krows,s,d,rows,rope,hf_norm,only_k,
        num_warps=4)


def launch_attention(inputs,out,*,causal=False,mode='query',bm=64,bn=64,warps=4,
            stages=1,gather=True,rope=True,hf_norm=True,qn=None,kn=None):
    """Launch full fusion, query-only fusion or a two-kernel unfused control.

    Scratch tensors must be preallocated for query/unfused modes. Timings must
    include prepare() and attention, as this function does.
    """
    if mode not in ('full','query','unfused'): raise ValueError('Invalid fusion mode')
    q,k,v,gq,gk,cos,sin=inputs
    b,h,s,d=q.shape
    if mode!='full':
        if kn is None or (mode=='unfused' and qn is None):raise ValueError('Missing scratch buffers')
        prepare_attention(inputs,qn if qn is not None else out,kn,only_k=mode=='query',rope=rope,hf_norm=hf_norm)
    return attention_v2_kernel[(triton.cdiv(s,bm),h,b)](
        q if mode!='unfused' else qn,k if mode=='full' else kn,v,gq,gk,cos,sin,out,
        s,d,h,k.shape[1],d**-.5,causal,mode!='unfused',mode=='full',rope,hf_norm,gather,
        bm,bn,num_warps=warps,num_stages=stages)


def optimized_attention(q, k, v, gq, gk, cos=None, sin=None, *, causal=False,
              mode='query', bn=128, gather=False, norm_style='hf'):
    """Forward H100 BF16 attention with query fusion or an unfused control.

    ``mode='query'`` prepares K once, then fuses Q normalization and optional
    split-half RoPE into attention. ``mode='unfused'`` prepares both Q and K
    once. Both modes launch two kernels and share the same attention schedule.
    Pass both cos/sin tables to enable RoPE. ``norm_style='hf'`` rounds the
    normalized activation before gamma multiplication; ``'legacy'`` uses one
    BF16 cast after the FP32 product. This API excludes projections, KV caches
    and backward. BF16 Tensor Core inputs accumulate in FP32.

    BN=64 or 128 is explicit: use the per-shape results to choose it. Fusion
    is not guaranteed to beat an equally tuned unfused or vendor kernel.
    """
    if mode not in ('query','unfused') or bn not in (64,128):
        raise ValueError('Expected query/unfused mode and BN=64/128')
    if norm_style not in ('hf','legacy'):
        raise ValueError('Expected hf or legacy norm_style')
    if (cos is None)!=(sin is None):
        raise ValueError('Provide both rotary tables or neither')
    rotary=cos is not None
    tensors=(q,k,v,gq,gk)+((cos,sin) if rotary else ())
    if any(t.requires_grad for t in tensors):
        raise ValueError('This API is forward-only')
    if q.ndim!=4 or k.ndim!=4 or k.shape!=v.shape:
        raise ValueError('Expected Q [B,Hq,S,D] and K/V [B,Hkv,S,D]')
    b,h,s,d=q.shape
    if (d not in (128,256) or min(b,h,s)<1 or k.shape[0]!=b or
        k.shape[2:]!=(s,d) or k.shape[1]<1 or h%k.shape[1]):
        raise ValueError('Expected D=128/256, matching batch/sequence and valid GQA')
    if gq.shape!=(d,) or gk.shape!=(d,) or (rotary and (cos.shape!=(s,d) or sin.shape!=(s,d))):
        raise ValueError('Expected gamma [D] and optional rotary tables [S,D]')
    if any(not t.is_cuda or t.device!=q.device or t.dtype!=torch.bfloat16 or
           not t.is_contiguous() for t in tensors):
        raise ValueError('Inputs must be contiguous BF16 on one CUDA device')
    if max(q.numel(),k.numel())>=2**31:
        raise ValueError('This implementation requires offsets below 2**31 elements')
    if torch.cuda.get_device_capability(q.device)[0]!=9:
        raise ValueError('This implementation was developed for SM90/Hopper')
    xs=(q,k,v,gq,gk,cos if rotary else q,sin if rotary else k)
    out=torch.empty_like(q);kn=torch.empty_like(k)
    qn=torch.empty_like(q) if mode=='unfused' else None
    with torch.cuda.device(q.device):
        launch_attention(xs,out,causal=causal,mode=mode,bm=64,bn=bn,gather=gather,
               rope=rotary,hf_norm=norm_style=='hf',qn=qn,kn=kn)
    return out


def refinement_inputs(model, b, s, seed, rotary):
    import e1_qk_publication_v3 as legacy
    if rotary:
        return make_inputs(MODELS[model], b, s, seed)
    xs, _ = legacy.inputs(legacy.MODELS[model], s, seed)
    return tuple(xs) + (xs[0], xs[1])

def refinement_oracle(xs, causal, rotary):
    import e1_qk_publication_v3 as legacy
    if rotary:
        return oracle_fp32(*xs, causal)
    q, k, v, gq, gk, _, _ = xs
    norm = lambda x, g: x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-06) * g.float()
    return torch.nn.functional.scaled_dot_product_attention(norm(q, gq), legacy.expand(norm(k, gk), q.shape[1]), legacy.expand(v.float(), q.shape[1]), is_causal=causal)

def refinement_old_arms(xs, causal, rotary, bn=64, bound=True):
    import e1_qk_publication_v3 as legacy
    q, k, v, gq, gk, c, sn = xs
    b, h, s, d = q.shape
    a, z = (torch.empty_like(q), torch.empty_like(q))
    qn, kn = (torch.empty_like(q), torch.empty_like(k))
    if rotary:
        full = lambda: launch_bf16_attention(xs, a, causal, bn=bn, causal_bound=bound)

        def unfused():
            prepare_qk(xs, qn, kn)
            return bf16_attention_kernel[triton.cdiv(s, 32), h, b](qn, kn, v, gq, gk, c, sn, z, b, s, d, h, k.shape[1], d ** (-0.5), 1e-06, False, causal, 32, bn, d, True, bound, num_warps=4, num_stages=1)
    else:
        full = lambda: legacy._flash[triton.cdiv(s, 32), h](q, k, v, gq, gk, a, s, d, h, k.shape[1], d ** (-0.5), True, causal, 32, 32, num_warps=4, num_stages=1)

        def unfused():
            for x, g, y in ((q, gq, qn), (k, gk, kn)):
                legacy._norm[x.numel() // d,](x, g, y, d, 1e-06, d, num_warps=4)
            return legacy._flash[triton.cdiv(s, 32), h](qn, kn, v, gq, gk, z, s, d, h, k.shape[1], d ** (-0.5), False, causal, 32, 32, num_warps=4, num_stages=1)
    return ({'old_full': full, 'old_unfused': unfused}, {'old_full': a, 'old_unfused': z})

def tune_refinement(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 512
    rec = {'environment': {'torch': torch.__version__, 'triton': triton.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name()}, 'rows': [], 'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(__file__).with_name('e1_qk_publication_v3.py'))}}

    def save():
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')
    mon = ExclusiveMonitor()
    rec['gpu_before'] = gpu_snapshot()
    mon.start()
    shapes = [('Qwen3-8B', 1, 128, False, True), ('Qwen3-8B', 32, 128, True, True), ('Qwen3-8B', 8, 256, False, True), ('Qwen3-8B', 1, 512, False, True), ('Qwen3-8B', 8, 512, False, True), ('Qwen3-8B', 32, 512, True, True), ('Qwen3-8B', 1, 1024, False, False), ('Qwen3-8B', 1, 1024, True, False), ('Gemma-3-4B', 1, 1024, False, False), ('Gemma-3-4B', 1, 1024, True, False), ('Qwen3-32B', 1, 2048, False, False), ('Gemma-3-12B', 1, 2048, True, False)]
    specs = [('full', 32, 64, 4, 1, False), ('full', 32, 64, 4, 1, True), ('full', 64, 64, 4, 1, True), ('full', 64, 128, 4, 1, True), ('query', 32, 64, 4, 1, True), ('query', 64, 64, 4, 1, True), ('query', 64, 128, 4, 1, True), ('query', 64, 128, 4, 2, True), ('query', 128, 64, 8, 1, True), ('unfused', 64, 64, 4, 1, True), ('unfused', 64, 128, 4, 1, True)]
    try:
        for model, b, s, causal, rotary in shapes:
            xs = refinement_inputs(model, b, s, 1909, rotary)
            ref = refinement_oracle(xs, causal, rotary)
            arms, outputs = refinement_old_arms(xs, causal, rotary)
            checks = {}
            evidence = {}
            for name, call in arms.items():
                evidence[name] = compiler_evidence(call(), args.out / 'compiler', f"{len(rec['rows'])}_{name}")
                checks[name] = metric(outputs[name], ref)
                assert checks[name]['pass'], (model, s, name, checks[name])
            for mode, bm, bn, warps, stages, gather in specs:
                name = f'{mode}_{bm}_{bn}_{warps}_{stages}_g{int(gather)}'
                out = torch.empty_like(xs[0])
                qn = torch.empty_like(xs[0])
                kn = torch.empty_like(xs[1])
                call = lambda out=out, qn=qn, kn=kn, mode=mode, bm=bm, bn=bn, warps=warps, stages=stages, gather=gather: launch_attention(xs, out, causal=causal, mode=mode, bm=bm, bn=bn, warps=warps, stages=stages, gather=gather, rope=rotary, hf_norm=rotary, qn=qn, kn=kn)
                try:
                    kernel = call()
                    checks[name] = metric(out, ref)
                    evidence[name] = compiler_evidence(kernel, args.out / 'compiler', f"{len(rec['rows'])}_{name}")
                    if checks[name]['pass']:
                        arms[name] = call
                except Exception as exc:
                    checks[name] = {'pass': False, 'error': repr(exc)}
                    print(name, repr(exc), flush=True)
            timing = device_paired(arms, rounds=3, samples=5, replays=20, seed=32109)
            row = {'model': model, 'shape': [b, s, causal], 'rotary': rotary, 'correctness': checks, 'compiler': evidence, 'timing': timing}
            rec['rows'].append(row)
            save()
            ordered = sorted(((n, round(v['median_us'], 3)) for n, v in timing['arms'].items()), key=lambda x: x[1])
            print(model, row['shape'], 'rope', rotary, ordered, flush=True)
        rec['status'] = 'passed'
    finally:
        rec['gpu_observations'] = mon.finish()
        rec['gpu_after'] = gpu_snapshot()
        save()

def verify_refinement(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rec = {'environment': {'torch': torch.__version__, 'triton': triton.__version__, 'gpu': torch.cuda.get_device_name()}, 'rows': [], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def save():
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')
    historical = list(csv.DictReader(open(Path(__file__).parent.parent / 'results/e1_qk_speedups.csv')))
    shapes = [('Qwen3-8B', b, s, c, True) for s in (128, 256, 384, 512) for c in (False, True) for b in (1, 8, 32)]
    shapes += [(r['model'], 1, int(r['sequence_length']), r['causal'] == 'true', False) for r in historical if r['device'] == 'H100']
    if getattr(args, 'models', ''):
        shapes = [x for x in shapes if x[0] in args.models.split(',')]
    if getattr(args, 'sequences', ''):
        shapes = [x for x in shapes if x[2] in [int(s) for s in args.sequences.split(',')]]
    if getattr(args, 'batches', ''):
        shapes = [x for x in shapes if x[1] in [int(b) for b in args.batches.split(',')]]
    if not shapes:
        raise ValueError('No refinement cases selected')
    specs = [('query', 64, 64, False), ('query', 64, 128, False), ('query', 64, 64, True), ('query', 64, 128, True), ('query', 32, 32, False), ('full', 32, 64, False), ('unfused', 64, 64, False), ('unfused', 64, 128, False)]
    mon = ExclusiveMonitor()
    rec['gpu_before'] = gpu_snapshot()
    mon.start()
    try:
        for model, b, s, causal, rotary in shapes:
            xs = refinement_inputs(model, b, s, 1909, rotary)
            ref = refinement_oracle(xs, causal, rotary)
            arms, outputs = refinement_old_arms(xs, causal, rotary, bn=32 if (b, s, causal) == (32, 128, False) else 64)
            checks = {}
            mapping = {}
            for mode, bm, bn, gather in specs:
                name = f'{mode}_{bm}_{bn}_g{int(gather)}'
                out = torch.empty_like(xs[0])
                qn = torch.empty_like(xs[0])
                kn = torch.empty_like(xs[1])
                call = lambda out=out, qn=qn, kn=kn, mode=mode, bm=bm, bn=bn, gather=gather: launch_attention(xs, out, causal=causal, mode=mode, bm=bm, bn=bn, gather=gather, rope=rotary, hf_norm=rotary, qn=qn, kn=kn)
                mapping[name] = (mode, bm, bn, gather)
                arms[name] = call
                outputs[name] = out
            qn = torch.empty_like(xs[0])
            kn = torch.empty_like(xs[1])

            def vendor():
                prepare_attention(xs, qn, kn, rope=rotary, hf_norm=rotary)
                outputs['vendor_unfused'] = torch.nn.functional.scaled_dot_product_attention(qn, kn, xs[2], is_causal=causal, enable_gqa=True)
            arms['vendor_unfused'] = vendor
            for name, call in list(arms.items()):
                try:
                    call()
                    checks[name] = metric(outputs[name], ref)
                    if not checks[name]['pass']:
                        del arms[name]
                except Exception as exc:
                    checks[name] = {'pass': False, 'error': repr(exc)}
                    del arms[name]
            assert all((checks[n]['pass'] for n in ('old_full', 'old_unfused'))), (model, s, checks)
            initial = device_paired(arms, rounds=3, samples=5, replays=20, seed=4109)
            latency = lambda n: initial['arms'][n]['median_us']
            fused = min((n for n in arms if n.startswith(('query_', 'full_'))), key=latency)
            unfused = min((n for n in arms if n.startswith('unfused_')), key=latency)
            selected = {n: arms[n] for n in ('old_full', 'old_unfused', fused, unfused, 'vendor_unfused') if n in arms}
            retest = device_paired(selected, rounds=5, samples=7, replays=20, seed=5109)
            seed_checks = []
            for seed in (2909, 3909):
                ys = refinement_inputs(model, b, s, seed, rotary)
                reference = refinement_oracle(ys, causal, rotary)
                for name in (fused, unfused):
                    mode, bm, bn, gather = mapping[name]
                    out = torch.empty_like(ys[0])
                    qn = torch.empty_like(ys[0])
                    kn = torch.empty_like(ys[1])
                    launch_attention(ys, out, causal=causal, mode=mode, bm=bm, bn=bn, gather=gather, rope=rotary, hf_norm=rotary, qn=qn, kn=kn)
                    check = metric(out, reference)
                    assert check['pass'], (model, s, name, seed, check)
                    seed_checks.append({'seed': seed, 'arm': name, **check})
            row = {'model': model, 'shape': [b, s, causal], 'rotary': rotary, 'correctness': checks, 'additional_seed_checks': seed_checks, 'selected_fused': fused, 'selected_unfused': unfused, 'initial': initial, 'retest': retest}
            rec['rows'].append(row)
            save()
            ts = retest['arms']
            print(len(rec['rows']), model, row['shape'], rotary, fused, 'old/new', round(ts['old_full']['median_us'] / ts[fused]['median_us'], 3), 'unfused/new', round(ts[unfused]['median_us'] / ts[fused]['median_us'], 3), flush=True)
        rec['tail_checks'] = []
        for model, rotary in [('Qwen3-8B', True), ('Gemma-3-4B', False)]:
            for s in (33, 129):
                for causal in (False, True):
                    xs = refinement_inputs(model, 1, s, 4909, rotary)
                    ref = refinement_oracle(xs, causal, rotary)
                    for mode in ('query', 'unfused', 'full'):
                        out = torch.empty_like(xs[0])
                        qn = torch.empty_like(xs[0])
                        kn = torch.empty_like(xs[1])
                        launch_attention(xs, out, causal=causal, mode=mode, bm=64, bn=64, gather=False, rope=rotary, hf_norm=rotary, qn=qn, kn=kn)
                        check = metric(out, ref)
                        assert check['pass'], (model, s, mode, check)
                        rec['tail_checks'].append({'model': model, 's': s, 'causal': causal, 'mode': mode, **check})
        rec['status'] = 'passed'
    finally:
        rec['gpu_observations'] = mon.finish()
        rec['gpu_after'] = gpu_snapshot()
        save()

if __name__ == '__main__':
    if '--fresh-processes' in sys.argv:
        from fusion_harness import fresh_process_census
        fresh_process_census(__file__, list(MODELS), rope=True)
    elif '--refine' in sys.argv or '--tune' in sys.argv:
        parser=argparse.ArgumentParser(description='H100 attention refinement')
        parser.add_argument('--refine',action='store_true')
        parser.add_argument('--tune',action='store_true')
        parser.add_argument('--out',type=Path,required=True)
        parser.add_argument('--models',default='')
        parser.add_argument('--sequences',default='')
        parser.add_argument('--batches',default='')
        args=parser.parse_args()
        (tune_refinement if args.tune else verify_refinement)(args)
    else:
        main()
