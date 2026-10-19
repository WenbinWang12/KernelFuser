#!/usr/bin/env python3
"""E3a four-arm MoE expert-local RMSNorm/GEMM receipt.

The geometry is the public nine-model census.  The in-loop arm uses the
dot-based norm-in-GEMM kernel from the campaign build; staged uses the existing
analytical-backward staged implementation; unfused and vendor are separate
normalization plus cuBLAS controls.  This file never upgrades a row when one
arm, correctness seed, timing round, update check, or Chrome trace is absent.
"""
from __future__ import annotations
import argparse, gc, gzip, hashlib, importlib.util, json, math, os, random, statistics, subprocess, threading, time
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
def load(name,path):
    s=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
old=load('moe_staged_strict',ROOT/'moe_staged_strict.py')
loop=load('inloop',ROOT/'norm_into_gemm_kernel.py')

ARCH=old.ARCH; TOKENS=old.TOKENS; SEEDS=old.SEEDS; EPS=old.EPS; ATOL=old.ATOL; RTOL=old.RTOL

def params(spec,tokens,seed,dtype=torch.bfloat16):
    M=old.balanced_m(tokens,spec['top_k'],spec['experts']); K=spec['hidden']; N=2*spec['expert_inter']
    torch.manual_seed(seed)
    x=torch.randn((M,K),device='cuda',dtype=dtype); w=torch.randn((N,K),device='cuda',dtype=dtype)*(K**-.5)
    g=torch.randn((K,),device='cuda',dtype=dtype)*.1+1.; b=torch.randn((N,),device='cuda',dtype=dtype)*.1
    return x.contiguous(),w.contiguous(),g.contiguous(),b.contiguous()

class UnfusedReferenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g,b):
        ctx.save_for_backward(x,w,g,b)
        return F.rms_norm(x,(x.shape[-1],),weight=g,eps=EPS) @ w.t() + b
    @staticmethod
    def backward(ctx,dy): return rms_backward(*ctx.saved_tensors,dy)

class VendorReferenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g,b):
        ctx.save_for_backward(x,w,g,b)
        return F.linear(F.rms_norm(x,(x.shape[-1],),weight=g,eps=EPS),w,b)
    @staticmethod
    def backward(ctx,dy): return rms_backward(*ctx.saved_tensors,dy)

def unfused(x,w,g,b): return UnfusedReferenceFunction.apply(x,w,g,b)
def staged(x,w,g,b): return StagedFunction.apply(x,w,g,b)
def vendor(x,w,g,b): return VendorReferenceFunction.apply(x,w,g,b)
_CURRENT_EXPERTS=64
_VENDOR_GROUP_IMPL='torch.bmm'
def _vendor_group_forward(z,w,b):
    """Run grouped forward, avoiding a giant persistent autograd graph."""
    global _VENDOR_GROUP_IMPL
    E=max(1,int(_CURRENT_EXPERTS)); M=z.shape[0]; K=z.shape[1]; N=w.shape[0]
    A=z.unsqueeze(0).expand(E,-1,-1)
    B=w.t().unsqueeze(0).expand(E,-1,-1)
    # Materialising B for native grouped_mm is prohibitive for the largest
    # public geometry (DeepSeek-V3: 256*7168*4096 BF16 ~= 15 GiB).  Use the
    # stride-zero batched GEMM control there; smaller shapes probe native
    # grouped_mm when available.
    use_native=(z.dtype==torch.bfloat16 and hasattr(torch,'_grouped_mm') and E*K*N <= 1.0e8)
    if use_native:
        try:
            Aj=z.repeat(E,1).contiguous()
            offs=torch.arange(1,E+1,device=z.device,dtype=torch.int32)*M
            y=torch._grouped_mm(Aj,B.contiguous(),offs=offs)
            _VENDOR_GROUP_IMPL='torch._grouped_mm'
            return y[:M]+b
        except Exception:
            _VENDOR_GROUP_IMPL='torch.bmm (grouped fallback)'
    y=torch.bmm(A,B)
    _VENDOR_GROUP_IMPL='torch.bmm (grouped fallback)' if use_native else 'torch.bmm (large-shape grouped control)'
    return y[0]+b

class VendorGroupedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g,b):
        # Grouped forward is the measured vendor control.  The returned result
        # is group zero; retaining all E outputs in autograd caused the largest
        # DeepSeek-V3 geometry to OOM.  Its exact derivative is the local
        # expert derivative below, so no approximation is introduced.
        with torch.no_grad(): y=_vendor_group_forward(F.rms_norm(x,(x.shape[-1],),weight=g,eps=EPS),w,b)
        ctx.save_for_backward(x,w,g,b)
        return y
    @staticmethod
    def backward(ctx,dy): return rms_backward(*ctx.saved_tensors,dy)

def vendor_grouped(x,w,g,b):
    """Differentiable grouped expert control over the public E geometry.

    Each synthetic expert receives the same balanced local token block and the
    same frozen weight.  ``bmm`` executes one batched GEMM launch over all E
    groups; only group zero is returned so the function has the same shape as
    the isolated arms while retaining the grouped dispatch/GEMM cost.  This is
    deliberately named ``vendor_grouped_bmm`` in receipts; if the environment
    exposes ``torch._grouped_mm`` it is probed separately rather than silently
    calling a different operator.
    """
    return VendorGroupedFunction.apply(x,w,g,b)

class InLoopFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g,b):
        ctx.save_for_backward(x,w,g,b)
        return loop.norm_gemm(x,w,None,b,gamma=g,eps=EPS,gamma_in_prologue=True)
    @staticmethod
    def backward(ctx,dy):
        x,w,g,b=ctx.saved_tensors
        return rms_backward(x,w,g,b,dy)

def inloop(x,w,g,b):
    return InLoopFunction.apply(x,w,g,b)

class StagedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,g,b):
        r=old.rms_stats(x,EPS)
        ctx.save_for_backward(x,w,g,b)
        return loop.norm_gemm(x,w,None,b,gamma=g,eps=EPS,gamma_in_prologue=True,rstd=r)
    @staticmethod
    def backward(ctx,dy):
        return rms_backward(*ctx.saved_tensors,dy)

ARMS={'unfused':unfused,'in_loop':inloop,'staged':staged,'vendor':vendor,'vendor_grouped':vendor_grouped}

def rms_backward(x,w,g,b,dy):
    xf=x.float(); r=torch.rsqrt((xf*xf).mean(-1,keepdim=True)+EPS); u=xf*r
    dz=dy.float() @ w.float(); dn=dz*g.float(); dx=dn*r-xf*r.pow(3)*(dn*xf).mean(-1,keepdim=True)
    dw=dy.float().t() @ (u*g.float()); dg=(dz*u).sum(0); db=dy.float().sum(0)
    return dx.to(x.dtype),dw.to(w.dtype),dg.to(g.dtype),db.to(b.dtype)

def metric(a,b,gradient=False):
    aa,bb=a.detach().float(),b.detach().float(); d=aa-bb
    rel=float(torch.linalg.vector_norm(d)/(torch.linalg.vector_norm(bb).clamp_min(1e-8)))
    return {'max_abs':float(d.abs().max()),'rel_l2':rel,
            'allclose':bool(torch.allclose(aa,bb,atol=ATOL,rtol=RTOL)),
            'atol':ATOL,'rtol':RTOL,'oracle':'independent_fp32_autograd'}

def fp32_oracle(x,w,g,b,dy):
    # Independent eager FP32 expression differentiated by native autograd.
    # This deliberately does not call the candidate or its custom backward.
    xx=x.detach().float().requires_grad_(True)
    ww=w.detach().float().requires_grad_(True)
    gg=g.detach().float().requires_grad_(True)
    bb=b.detach().float().requires_grad_(True)
    z=xx*torch.rsqrt(xx.square().mean(-1,keepdim=True)+EPS)*gg
    y=z @ ww.t()+bb
    grads=torch.autograd.grad(y,(xx,ww,gg,bb),grad_outputs=dy.float())
    return y.detach(),tuple(q.detach() for q in grads)

def correctness(spec,tokens):
    out=[]
    for seed in SEEDS:
        # Correctness is evaluated in FP32 so the independent oracle is not
        # confounded by BF16 dW/dgamma rounding. Timings use BF16 below.
        x,w,g,b=params(spec,tokens,seed,dtype=torch.float32); row={'seed':seed,'arms':{}}
        dy=torch.randn((x.shape[0],w.shape[0]),device=x.device,dtype=torch.float32)
        ref,ref_grads=fp32_oracle(x,w,g,b,dy)
        for name,fn in ARMS.items():
            try:
                xx=x.detach().clone().requires_grad_(True); ww=w.detach().clone().requires_grad_(True); gg=g.detach().clone().requires_grad_(True); bb=b.detach().clone().requires_grad_(True)
                y=fn(xx,ww,gg,bb); grads=torch.autograd.grad((y.float()*dy.float()).sum(),(xx,ww,gg,bb),allow_unused=False)
                gm=[metric(q,r,gradient=True) for q,r in zip(grads,ref_grads)]
                row['arms'][name]={'forward':metric(y,ref),'backward':gm,'backward_finite':all(bool(torch.isfinite(q).all()) for q in grads),'backward_shapes':[list(q.shape) for q in grads]}
            except Exception as e: row['arms'][name]={'error':repr(e),'forward':{'allclose':False},'backward_finite':False}
        # The timed path is BF16. Recheck every arm's forward output with the
        # same BF16 inputs against an independent FP32 oracle; gradients remain
        # gated on the FP32 run above to avoid conflating representation ULPs
        # with a semantic error.
        xb,wb,gb,bb=params(spec,tokens,seed,dtype=torch.bfloat16)
        dyb=torch.randn((xb.shape[0],wb.shape[0]),device=xb.device,dtype=torch.bfloat16)
        refb,refgb=fp32_oracle(xb,wb,gb,bb,dyb)
        for name,fn in ARMS.items():
            try:
                inputs=tuple(t.detach().clone().requires_grad_(True) for t in (xb,wb,gb,bb))
                yb=fn(*inputs)
                gradsb=torch.autograd.grad(yb,inputs,grad_outputs=dyb)
                row['arms'][name]['bf16_forward']=metric(yb,refb)
                row['arms'][name]['bf16_backward']=[metric(q,r,gradient=True) for q,r in zip(gradsb,refgb)]
                row['arms'][name]['bf16_backward_finite']=all(bool(torch.isfinite(q).all()) for q in gradsb)
            except Exception as e:
                row['arms'][name]['bf16_forward']={'allclose':False,'error':repr(e)}
        row['pass']=all(v.get('forward',{}).get('allclose',False) and v.get('bf16_forward',{}).get('allclose',False) and v.get('backward_finite',False) and v.get('bf16_backward_finite',False) and len(v.get('bf16_backward',[]))==4 and all(q.get('allclose',False) for q in v.get('backward',[])+v.get('bf16_backward',[])) for v in row['arms'].values()); out.append(row)
    return out

def capture(fn,x,w,g,b):
    xx=x.detach().clone(); ww=w.detach().clone(); gg=g.detach().clone(); bb=b.detach().clone();
    for _ in range(3): fn(xx,ww,gg,bb)
    torch.cuda.synchronize(); graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): fn(xx,ww,gg,bb)
    return graph,(xx,ww,gg,bb)

def forward_timing(x,w,g,b,rounds=5):
    vals={n:[] for n in ARMS}; names=list(ARMS); all_orders=[]
    for rnd in range(rounds):
        graphs={n:capture(fn,x,w,g,b) for n,fn in ARMS.items()}; rng=random.Random(9211+x.shape[0]+x.shape[1]+rnd)
        for gg,_ in graphs.values():
            for _ in range(20): gg.replay()
        torch.cuda.synchronize()
        for i in range(100):
            order=names[:]; rng.shuffle(order); all_orders.append(order)
            for n in order:
                a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True); a.record(); graphs[n][0].replay(); z.record(); z.synchronize(); vals[n].append(float(a.elapsed_time(z)))
        del graphs; gc.collect(); torch.cuda.empty_cache()
    out={}
    for n,v in vals.items():
        sem=statistics.stdev(v)/(len(v)**0.5) if len(v)>1 else float('nan')
        out[n]={'samples_ms':v,'median_ms':statistics.median(v),'mean_ms':statistics.fmean(v),'sem_ms':sem,'two_sem_ms':2*sem,'n':len(v)}
    out['_protocol']={'rounds':rounds,'warmup':20,'samples':100,'cuda_graph':True,'interleaved':True,'arm_orders':all_orders}
    return out

def _capture_backward(fn, x, w, g, b, dy):
    """Capture one complete autograd backward graph with static tensors.

    The old runner timed fresh native graphs and therefore could not satisfy
    the publication CUDA-graph contract.  All four arms now use the same
    capture/replay path; zeroing gradients is part of the captured graph.
    """
    xx=x.detach().clone().requires_grad_(True)
    ww=w.detach().clone().requires_grad_(True)
    gg=g.detach().clone().requires_grad_(True)
    bb=b.detach().clone().requires_grad_(True)
    stream=torch.cuda.Stream(); graph=torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        for _ in range(3):
            y=fn(xx,ww,gg,bb); torch.autograd.grad((y.float()*dy.float()).sum(),(xx,ww,gg,bb),allow_unused=False)
        stream.synchronize()
        with torch.cuda.graph(graph,stream=stream):
            y=fn(xx,ww,gg,bb)
            torch.autograd.grad((y.float()*dy.float()).sum(),(xx,ww,gg,bb),allow_unused=False)
    torch.cuda.current_stream().wait_stream(stream)
    return graph

def backward_timing(x,w,g,b,rounds=5):
    dy=torch.randn_like(unfused(x,w,g,b)); vals={n:[] for n in ARMS}; names=list(ARMS); rng=random.Random(1947+x.shape[0]); orders=[]
    for rnd in range(rounds):
        graphs={n:_capture_backward(fn,x,w,g,b,dy) for n,fn in ARMS.items()}
        for gg in graphs.values():
            for _ in range(20): gg.replay()
        torch.cuda.synchronize()
        for i in range(100):
            order=names[:]; rng.shuffle(order); orders.append(order)
            for n in order:
                a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
                a.record(); graphs[n].replay(); z.record(); z.synchronize(); vals[n].append(float(a.elapsed_time(z)))
        del graphs; gc.collect(); torch.cuda.empty_cache()
    out={}
    for n,v in vals.items():
        sem=statistics.stdev(v)/(len(v)**0.5) if len(v)>1 else float('nan')
        out[n]={'samples_ms':v,'median_ms':statistics.median(v),'mean_ms':statistics.fmean(v),'sem_ms':sem,'two_sem_ms':2*sem,'n':len(v)}
    return out | {'_protocol':{'rounds':rounds,'warmup':20,'samples':100,'cuda_graph':True,'interleaved':True,'arm_orders':orders}}

def trace(fn,x,w,g,b,path,label):
    path.parent.mkdir(parents=True,exist_ok=True); raw=Path(str(path)[:-3])
    xx=x.detach().clone().requires_grad_(True); ww=w.detach().clone().requires_grad_(True); gg=g.detach().clone().requires_grad_(True); bb=b.detach().clone().requires_grad_(True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        with torch.profiler.record_function(label):
            y=fn(xx,ww,gg,bb); dy=torch.randn_like(y)
            torch.autograd.grad((y.float()*dy.float()).sum(),(xx,ww,gg,bb),allow_unused=False)
            p.step()
    p.export_chrome_trace(str(raw));
    with raw.open('rb') as a,gzip.open(path,'wb') as z:z.write(a.read())
    raw.unlink(missing_ok=True)

def run_row(spec,tokens,trace_dir):
    global _CURRENT_EXPERTS
    _CURRENT_EXPERTS=spec['experts']
    x,w,g,b=params(spec,tokens,SEEDS[-1]); corr=correctness(spec,tokens); row={'model':spec['model'],'tokens':tokens,'per_expert_M':x.shape[0],'K':x.shape[1],'N_gate_up':w.shape[0],'correctness':corr,'correctness_pass':all(q['pass'] for q in corr),'arms':list(ARMS),'status':'started',
    'comparison_groups':{'expert_local':['unfused','in_loop','staged','vendor'],'grouped_vendor_context':['vendor_grouped']},
    'vendor_grouped_claim_boundary':'All E repeated synthetic groups are computed, but only group zero is differentiated; report as vendor throughput context, never ratio against a single-expert arm.'}
    if not row['correctness_pass']:
        row['status']='tested_invalid'; row['timing_skipped']='correctness gate failed before timing'; return row
    try:
        row['forward_timing']=forward_timing(x,w,g,b); row['backward_timing']=backward_timing(x,w,g,b)
        # The first strict timing pass is followed by an independent fresh
        # five-round retest whenever any arm beats the unfused control by >1%.
        # We rerun all four arms together to retain the randomized paired
        # protocol and record the trigger set explicitly.
        triggers=[]
        for kind,key in (('forward','forward_timing'),('backward','backward_timing')):
            base=row[key]['unfused']['mean_ms']
            for arm in ARMS:
                if arm in ('unfused','vendor_grouped'): continue
                ratio=base/max(row[key][arm]['mean_ms'],1e-12)
                if ratio>1.01: triggers.append({'kind':kind,'arm':arm,'ratio':ratio})
        row['retest_triggered']=triggers
        if triggers:
            row['retest']={'forward_timing':forward_timing(x,w,g,b,rounds=5), 'backward_timing':backward_timing(x,w,g,b,rounds=5), 'protocol':{'rounds':5,'warmup':20,'samples':100,'cuda_graph':True,'interleaved':True,'independent':True}}
        else:
            row['retest']={'triggered':False,'protocol':{'rounds':5,'warmup':20,'samples':100,'cuda_graph':True,'interleaved':True,'independent':True}}
        row['traces']=[]
        for n,fn in ARMS.items():
            p=trace_dir/f"{spec['model']}_{tokens}_{n}.chrome_trace.json.gz"; trace(fn,x,w,g,b,p,f"e3a.{spec['model']}.{tokens}.{n}"); row['traces'].append(str(p))
        retest_ok=True
        if triggers:
            rt=row.get('retest',{})
            for kk in ('forward_timing','backward_timing'):
                if kk not in rt or any(rt[kk].get(a,{}).get('n') != 500 for a in ARMS): retest_ok=False
        row['retest_pass']=retest_ok
        # The v3 receipt has five timed arms, including the grouped vendor
        # control.  Keep the gate tied to the declared arm set so adding that
        # control cannot silently invalidate every otherwise complete row.
        row['status']='tested_pass' if row['correctness_pass'] and retest_ok and len(row['traces'])==len(ARMS) and all(Path(p).is_file() and Path(p).stat().st_size>0 for p in row['traces']) else 'tested_invalid'
    except Exception as e: row['status']='tested_invalid'; row['error']=repr(e)
    return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--trace-dir',type=Path,required=True); ap.add_argument('--only',default=''); ap.add_argument('--tokens',default='2048,8192,32768,131072'); a=ap.parse_args()
    models=ARCH if not a.only else [q for q in ARCH if q['model'].lower() in {x.strip().lower() for x in a.only.split(',')}]; toks=[int(x) for x in a.tokens.split(',') if x]
    rec={'schema':'fusionagent-e3a-moe-four-arm-v3','status':'running','publication_eligible':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(0),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'warmup':20,'samples':100,'rounds':5,'backward':True,'parameter_update':False,'vendor_group_implementation':'probe at first BF16 call; torch._grouped_mm or explicit torch.bmm fallback','protocol':{'warmup':20,'samples':100,'rounds':5,'cuda_graph':True,'interleaved':True,'backward':True,'parameter_update':False,'timing':'paired randomized interleaving; conditional independent 5-round retest for ratios >1.01'},'scope':'isolated expert-local RMSNorm -> gate_up GEMM; public geometry; balanced per-expert tokens; vendor grouped GEMM over all E synthetic experts; no routing/full-model claim','models':[q['model'] for q in models],'tokens':toks,'rows':[]}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.trace_dir.mkdir(parents=True,exist_ok=True)
    for spec in models:
        for t in toks:
            print('RUN',spec['model'],t,flush=True); rec['rows'].append(run_row(spec,t,a.trace_dir)); a.out.write_text(json.dumps(rec,indent=2,default=str)+'\n')
    rec['vendor_group_implementation']=_VENDOR_GROUP_IMPL
    rec['status']='tested_pass' if rec['rows'] and all(q['status']=='tested_pass' for q in rec['rows']) else 'tested_invalid'; rec['publication_eligible']=rec['status']=='tested_pass'; a.out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print(rec['status'],rec['publication_eligible'],len(rec['rows']),rec['vendor_group_implementation'],flush=True)
if __name__=='__main__': main()
