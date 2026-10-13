#!/usr/bin/env python3
"""RoPE-aware isolated Q/K RMSNorm fusion sweep.

This runner is deliberately a forward-only proxy: it fuses Q/K RMSNorm and
rotary position embedding into the same fixed-launch attention kernel.  It
does not include projections, KV-cache updates, sliding-window masking, or a
full model.  The receipt records three comparison arms: a separate norm+RoPE
path using the same Triton attention anchor, torch.compile, and vendor SDPA.
"""
from __future__ import annotations
import argparse, gzip, hashlib, json, math, os, platform, random, statistics, subprocess, time
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
EPS = 1e-6

@triton.jit
def norm_kernel(X, G, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0); j = tl.arange(0, BLOCK)
    x = tl.load(X + row*N + j, mask=j < N, other=0).to(tl.float32)
    g = tl.load(G + j, mask=j < N, other=0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x*x, axis=0)/N + EPS)
    tl.store(Y + row*N + j, (x*r*g).to(Y.dtype.element_ty), mask=j < N)

@triton.jit
def attn_kernel(Q, K, V, GQ, GK, COS, SIN, O,
                BATCH: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                HQ: tl.constexpr, HK: tl.constexpr, SCALE: tl.constexpr,
                FUSED: tl.constexpr, CAUSAL: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0); h = tl.program_id(1); b = tl.program_id(2)
    kh = h // (HQ // HK); mi = pid*BM + tl.arange(0, BM); dj = tl.arange(0, BLOCK_D)
    qoff = b*HQ*S*D + h*S*D + mi[:, None]*D + dj[None, :]
    qmask = (mi[:, None] < S) & (dj[None, :] < D)
    q = tl.load(Q + qoff, mask=qmask, other=0).to(tl.float32)
    if FUSED:
        g = tl.load(GQ + dj, mask=dj < D, other=0).to(tl.float32)
        r = tl.rsqrt(tl.sum(q*q, axis=1)/D + 1.e-6)
        q = q*r[:, None]*g[None, :]
        jp = tl.where(dj < D//2, dj + D//2, dj - D//2)
        qp = tl.load(Q + b*HQ*S*D + h*S*D + mi[:, None]*D + jp[None, :],
                     mask=(mi[:, None] < S) & (jp[None, :] < D), other=0).to(tl.float32)
        gp = tl.load(GQ + jp, mask=(jp >= 0) & (jp < D), other=0).to(tl.float32)
        qp = qp*r[:, None]*gp[None, :]
        c = tl.load(COS + mi[:, None]*D + dj[None, :], mask=qmask, other=1).to(tl.float32)
        s = tl.load(SIN + mi[:, None]*D + dj[None, :], mask=qmask, other=0).to(tl.float32)
        first = dj[None, :] < D//2
        q = tl.where(first, q*c - qp*s, q*c + qp*s)
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
            r = tl.rsqrt(tl.sum(k*k, axis=1)/D + 1.e-6)
            k = k*r[:, None]*g[None, :]
            jp = tl.where(dj < D//2, dj + D//2, dj - D//2)
            kp = tl.load(K + b*HK*S*D + kh*S*D + nj[:, None]*D + jp[None, :],
                         mask=(nj[:, None] < S) & (jp[None, :] < D), other=0).to(tl.float32)
            gp = tl.load(GK + jp, mask=(jp >= 0) & (jp < D), other=0).to(tl.float32)
            kp = kp*r[:, None]*gp[None, :]
            c = tl.load(COS + nj[:, None]*D + dj[None, :], mask=kmask, other=1).to(tl.float32)
            s = tl.load(SIN + nj[:, None]*D + dj[None, :], mask=kmask, other=0).to(tl.float32)
            first = dj[None, :] < D//2
            k = tl.where(first, k*c - kp*s, k*c + kp*s)
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

def norm_triton(x, g):
    y = torch.empty_like(x); d=x.shape[-1]
    norm_kernel[(x.numel()//d,)](x, g, y, d, EPS, triton.next_power_of_2(d), num_warps=4)
    return y

def norm_torch(x, g):
    xf=x.float(); return (xf*torch.rsqrt(xf.square().mean(-1,keepdim=True)+EPS)*g.float()).to(x.dtype)

def rope_torch(x, cos, sin):
    # HuggingFace/Qwen/Gemma use rotate_half: split-half, not interleaved.
    d=x.shape[-1]; half=d//2; a=x[...,:half].float(); b=x[...,half:].float()
    c=cos[...,:half][None,None].float(); s=sin[...,:half][None,None].float()
    y=torch.empty_like(x); y[...,:half]=(a*c-b*s).to(x.dtype); y[...,half:]=(b*c+a*s).to(x.dtype)
    return y

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
    # Duplicate the half-dimensional frequency table in the split-half layout.
    return q.contiguous(),k.contiguous(),v.contiguous(),gq,gk,torch.cat([torch.cos(ang),torch.cos(ang)],dim=1),torch.cat([torch.sin(ang),torch.sin(ang)],dim=1)

def fused(q,k,v,gq,gk,cos,sin,causal):
    b,h,s,d=q.shape; o=torch.empty_like(q); bd=triton.next_power_of_2(d)
    attn_kernel[(triton.cdiv(s,32),h,b)](q,k,v,gq,gk,cos,sin,o,b,s,d,h,k.shape[1],d**-.5,True,causal,32,32,bd,num_warps=4,num_stages=1)
    return o

def anchor(q,k,v,cos,sin,causal):
    b,h,s,d=q.shape; o=torch.empty_like(q); bd=triton.next_power_of_2(d)
    attn_kernel[(triton.cdiv(s,32),h,b)](q,k,v,q,k,cos,sin,o,b,s,d,h,k.shape[1],d**-.5,False,causal,32,32,bd,num_warps=4,num_stages=1)
    return o

def separate(q,k,v,gq,gk,cos,sin,causal):
    qn=rope_torch(norm_triton(q,gq),cos,sin); kn=rope_torch(norm_triton(k,gk),cos,sin)
    return anchor(qn,kn,v,cos,sin,causal)

def vendor(q,k,v,gq,gk,cos,sin,causal):
    qn=rope_torch(norm_triton(q,gq),cos,sin); kn=rope_torch(norm_triton(k,gk),cos,sin)
    return F.scaled_dot_product_attention(qn,expand(kn,q.shape[1]),expand(v,q.shape[1]),is_causal=causal)

def compiled_fn(q,k,v,gq,gk,cos,sin,causal):
    qn=rope_torch(norm_triton(q,gq),cos,sin); kn=rope_torch(norm_triton(k,gk),cos,sin)
    return F.scaled_dot_product_attention(qn,expand(kn,q.shape[1]),expand(v,q.shape[1]),is_causal=causal)

def stats(fn, rounds, samples):
    vals=[]
    for _ in range(rounds):
        for _ in range(10): fn()
        torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True)
        st.record();
        for _ in range(samples): fn()
        en.record(); en.synchronize(); vals.append(st.elapsed_time(en)/samples)
    return {'p50_ms':statistics.median(vals),'rounds_ms':vals,'n':len(vals)*samples}

def paired(a,b,rounds,samples):
    out={'a':[],'b':[]};
    for r in range(rounds):
        order=['a','b'] if (r%2==0) else ['b','a']
        for key in order:
            fn=a if key=='a' else b
            for _ in range(10): fn()
            torch.cuda.synchronize(); st=torch.cuda.Event(True); en=torch.cuda.Event(True); st.record()
            for _ in range(samples): fn()
            en.record(); en.synchronize(); out[key].append(st.elapsed_time(en)/samples)
    return out

def metric(x,y):
    diff=(x.float()-y.float()); den=y.float().norm().clamp_min(1e-12)
    return {'max_abs':float(diff.abs().max()),'relative_l2':float(diff.norm()/den),'pass':bool(torch.allclose(x.float(),y.float(),atol=.025,rtol=.025))}

def gpu_snapshot():
    try:
        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True).splitlines()
        return {'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'visible_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'apps':lines,'exclusive':len(lines)<=1}
    except Exception as e: return {'error':repr(e),'exclusive':False}

def capture_trace(fn,path,label):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        with torch.profiler.record_function(label): fn()
        torch.cuda.synchronize()
    raw=Path(str(path)[:-3]); p.export_chrome_trace(str(raw))
    with raw.open('rb') as f, gzip.open(path,'wb') as z: z.write(f.read())
    raw.unlink(missing_ok=True); return str(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--models',default=','.join(MODELS)); ap.add_argument('--sizes',default='128,256,384,512'); ap.add_argument('--batches',default='1,8,32'); ap.add_argument('--causal-modes',default='false,true'); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--samples',type=int,default=40); ap.add_argument('--output',required=True); ap.add_argument('--trace-dir',default=''); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if a.smoke: a.models='Qwen3-8B'; a.sizes='128'; a.batches='1'; a.causal_modes='false'; a.rounds=2; a.samples=20
    sizes=[int(x) for x in a.sizes.split(',')]; batches=[int(x) for x in a.batches.split(',')]; modes=[x.strip().lower()=='true' for x in a.causal_modes.split(',')]
    rec={'schema':'fusionagent-e1-rope-v1','scope':'isolated Q/K RMSNorm+RoPE into attention; no projection/KV-cache/sliding-window/full model','candidate':'native Triton RMSNorm+RoPE fused attention','baseline':'same fixed Triton attention anchor with separate norm and RoPE','additional_arms':['torch.compile SDPA','vendor SDPA'],'environment':{'host':platform.node(),'gpu':torch.cuda.get_device_name(),'visible_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'triton':triton.__version__,'cuda':torch.version.cuda},'protocol':{'S':sizes,'B':batches,'causal_modes':modes,'warmup':10,'rounds':a.rounds,'samples':a.samples,'paired_timing':'separate/candidate, compiled/vendor measured in alternating paired order','correctness':'candidate vs separate anchor, vendor SDPA and FP32 oracle','retest':'fresh paired retest when initial speedup exceeds 1.01','gpu_exclusivity':True,'traces':bool(a.trace_dir)},'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'models':{},'status':'running','publication_eligible':False}
    for name in [x.strip() for x in a.models.split(',')]:
        cfg=MODELS[name]; rec['models'][name]={'geometry':{'Hq':cfg[0],'Hkv':cfg[1],'D':cfg[2]},'rows':[]}
        for s in sizes:
            for causal in modes:
                for b in batches:
                    try:
                        torch._dynamo.reset(); torch._dynamo.config.cache_size_limit=512
                        before=gpu_snapshot()
                        q,k,v,gq,gk,c,sn=make_inputs(cfg,b,s,1701+s+b+int(causal)); sep=lambda:separate(q,k,v,gq,gk,c,sn,causal); cand=lambda:fused(q,k,v,gq,gk,c,sn,causal); vend=lambda:vendor(q,k,v,gq,gk,c,sn,causal); comp=torch.compile(compiled_fn,fullgraph=True,dynamic=False); compfn=lambda:comp(q,k,v,gq,gk,c,sn,causal)
                        with torch.no_grad():
                            ref=sep(); cc=cand(); vv=vend();
                        if a.smoke:
                            print('DEBUG norms',float(ref.float().norm()),float(cc.float().norm()),float(vv.float().norm()),'means',float(ref.float().mean()),float(cc.float().mean()),float(vv.float().mean()),flush=True)
                        oracle=F.scaled_dot_product_attention(rope_torch(norm_torch(q,gq),c,sn),expand(rope_torch(norm_torch(k,gk),c,sn),q.shape[1]),expand(v,q.shape[1]),is_causal=causal)
                        cor={'candidate_vs_separate':metric(cc,ref),'candidate_vs_vendor':metric(cc,vv),'candidate_vs_fp32_oracle':metric(cc,oracle)}
                        p=paired(sep,cand,a.rounds,a.samples); pc=paired(compfn,vend,a.rounds,a.samples)
                        st={'separate_norm_rope':{'p50_ms':statistics.median(p['a']),'rounds_ms':p['a']},'fused_norm_rope':{'p50_ms':statistics.median(p['b']),'rounds_ms':p['b']},'torch_compile':{'p50_ms':statistics.median(pc['a']),'rounds_ms':pc['a']},'vendor_sdpa':{'p50_ms':statistics.median(pc['b']),'rounds_ms':pc['b']}}
                        sp=st['fused_norm_rope']['p50_ms']; fast=min(st[x]['p50_ms'] for x in ['separate_norm_rope','torch_compile','vendor_sdpa']); row={'B':b,'S':s,'causal':causal,'correctness':cor,'timing':st,'speedup_vs_separate':st['separate_norm_rope']['p50_ms']/sp,'speedup_vs_fastest':fast/sp,'retest':None,'gpu_exclusivity':{'before':before,'after':gpu_snapshot()},'traces':[],'status':'pass' if all(cor[k]['pass'] for k in cor) else 'fail'}
                        if row['speedup_vs_fastest']>1.01:
                            rr=paired(sep,cand,a.rounds,a.samples); row['retest']={'separate_ms':rr['a'],'fused_ms':rr['b'],'speedup':statistics.median(rr['a'])/statistics.median(rr['b'])}
                        if a.trace_dir:
                            row['traces']=[capture_trace(sep,Path(a.trace_dir)/f'{name}_B{b}_S{s}_C{int(causal)}_separate.chrome.json.gz','e1_rope.separate'),capture_trace(cand,Path(a.trace_dir)/f'{name}_B{b}_S{s}_C{int(causal)}_fused.chrome.json.gz','e1_rope.fused')]
                    except Exception as e: row={'B':b,'S':s,'causal':causal,'status':'error','error':repr(e)}
                    rec['models'][name]['rows'].append(row); print(name,b,s,causal,row.get('status'),row.get('speedup_vs_fastest'),flush=True)
    rec['status']='tested_pass' if all(r.get('status')=='pass' for m in rec['models'].values() for r in m['rows']) else 'tested_with_errors'; rec['publication_eligible']=rec['status']=='tested_pass'; Path(a.output).write_text(json.dumps(rec,indent=2)); print(json.dumps({'output':a.output,'status':rec['status']},indent=2))

if __name__=='__main__': main()
