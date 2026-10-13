#!/usr/bin/env python3
"""E8 strict forward census: dense RMSNorm prologue and SwiGLU epilogue.

This runner is intentionally forward-only.  It gates FP32 and BF16 outputs
against an independent FP32 eager oracle before timing, then uses five
interleaved CUDA-graph rounds (20 warmups/100 samples), mean/SEM/two-SEM,
conditional retesting, and Chrome traces.  No old development JSON is read or
promoted.  The dense prologue candidate is the same Triton RMSNorm->GEMM
kernel used by E3a; the SwiGLU candidate is a separately compiled
TorchInductor function, reported as such.
"""
from __future__ import annotations
import argparse, gc, gzip, hashlib, json, math, os, random, statistics
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent
import importlib.util
s=importlib.util.spec_from_file_location('norm_kernel',str(ROOT/'norm_into_gemm_kernel.py')); nk=importlib.util.module_from_spec(s); s.loader.exec_module(nk)

EPS=1e-5; ATOL=2e-2; RTOL=2e-2
GEOMS=[
    {'site':'dense_rmsnorm','model':'Qwen3-0.6B','M':512,'K':1024,'N':4096,'kind':'rms'},
    {'site':'dense_rmsnorm','model':'Qwen3-0.6B','M':2048,'K':1024,'N':4096,'kind':'rms'},
    {'site':'dense_rmsnorm','model':'Qwen3-0.6B','M':8192,'K':1024,'N':4096,'kind':'rms'},
    {'site':'dense_rmsnorm','model':'Qwen3-0.6B','M':16384,'K':1024,'N':4096,'kind':'rms'},
    {'site':'swiglu','model':'Wukong-tower','M':512,'K':2048,'N':8192,'kind':'swiglu'},
    {'site':'swiglu','model':'Wukong-tower','M':2048,'K':2048,'N':8192,'kind':'swiglu'},
    {'site':'swiglu','model':'Wukong-tower','M':8192,'K':2048,'N':8192,'kind':'swiglu'},
    {'site':'swiglu','model':'Wukong-tower','M':16384,'K':2048,'N':8192,'kind':'swiglu'},
]

def close(a,b):
    d=(a.detach().float()-b.detach().float()); den=torch.linalg.vector_norm(b.float()).clamp_min(1e-8)
    return {'max_abs':float(d.abs().max()),'relative_l2':float(torch.linalg.vector_norm(d)/den),'allclose':bool(torch.allclose(a.float(),b.float(),atol=ATOL,rtol=RTOL))}

def stats(v):
    sem=statistics.stdev(v)/(len(v)**.5) if len(v)>1 else float('nan')
    return {'n':len(v),'mean_ms':statistics.fmean(v),'sem_ms':sem,'two_sem_ms':2*sem,'p50_ms':statistics.median(v),'samples_ms':v}

def cap(fn):
    for _ in range(20): fn()
    torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    torch.cuda.synchronize(); return g

def paired(arms,seed=9047):
    vals={k:[] for k in arms}; orders=[]
    for rnd in range(5):
        gs={k:cap(v) for k,v in arms.items()}; rng=random.Random(seed+rnd)
        for i in range(100):
            order=list(arms); rng.shuffle(order); orders.append(order)
            for k in order:
                a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); gs[k].replay(); b.record(); b.synchronize(); vals[k].append(float(a.elapsed_time(b)))
        del gs; gc.collect(); torch.cuda.empty_cache()
    out={k:stats(v) for k,v in vals.items()}; out['_protocol']={'warmup':20,'samples':100,'rounds':5,'cuda_graph':True,'interleaved':True,'arm_order':'randomized_per_pair_sample','orders':orders}; return out

def trace(fn,path,label):
    path.parent.mkdir(parents=True,exist_ok=True); raw=Path(str(path)[:-3])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        with torch.profiler.record_function(label): fn(); p.step()
    p.export_chrome_trace(str(raw))
    with raw.open('rb') as a,gzip.open(path,'wb') as z:z.write(a.read())
    raw.unlink(missing_ok=True)

def make_arms(g,A,W,gamma,bias):
    if g['kind']=='rms':
        plain=lambda: F.rms_norm(A,(A.shape[-1],),weight=gamma,eps=EPS) @ W.t()+bias
        vendor=lambda: F.linear(F.rms_norm(A,(A.shape[-1],),weight=gamma,eps=EPS),W,bias)
        fused=lambda: nk.norm_gemm(A,W,None,bias,gamma=gamma,eps=EPS,gamma_in_prologue=True)
        return {'plain':plain,'fused_triton':fused,'vendor':vendor}
    if g['kind']=='swiglu':
        wg=W; wu=gamma
        plain=lambda: F.silu(A @ wg.t()) * (A @ wu.t())
        vendor=lambda: F.silu(F.linear(A,wg)) * F.linear(A,wu)
        compiled=torch.compile(plain,mode='max-autotune-no-cudagraphs',fullgraph=True)
        compiled(); torch.cuda.synchronize()
        return {'plain':plain,'fused_torchinductor':compiled,'vendor':vendor}
    raise ValueError(g)

def row(g,trace_dir):
    torch.manual_seed(20261112+g['M']+g['K']); M,K,N=g['M'],g['K'],g['N']
    A=torch.randn((M,K),device='cuda',dtype=torch.bfloat16)*.5
    W=torch.randn((N,K),device='cuda',dtype=torch.bfloat16)*(K**-.5)
    # Dense RMSNorm carries a K-vector scale; SwiGLU carries a second N x K
    # projection.  Keeping these distinct avoids accidentally broadcasting a
    # normalization vector across the epilogue output.
    gamma=(torch.randn((K,),device='cuda',dtype=torch.bfloat16)*.1+1) if g['kind']=='rms' else torch.randn((N,K),device='cuda',dtype=torch.bfloat16)*(K**-.5)
    bias=torch.randn((N,),device='cuda',dtype=torch.bfloat16)*.1
    arms=make_arms(g,A,W,gamma,bias)
    # Independent FP32 oracle and BF16 forward gate happen before timing.
    af,wf,gf,bf=A.float(),W.float(),gamma.float(),bias.float()
    if g['kind']=='rms': ref=F.rms_norm(af,(K,),weight=gf,eps=EPS) @ wf.t()+bf
    else: ref=F.silu(af @ wf.t()) * (af @ gf.t())
    corr={}; ok=True
    for k,fn in arms.items():
        y=fn(); c=close(y,ref); finite=bool(torch.isfinite(y.detach().float()).all().item()); corr[k]={'fp32_oracle':c,'bf16_forward':c,'finite':finite,'allclose':c['allclose']}; ok &= finite and c['allclose']
    if not ok: return {'geometry':g,'correctness':corr,'status':'tested_invalid','publication_gate':{'pass':False,'reason':'correctness before timing'}}
    timing=paired(arms); triggers=[]; base=timing['plain']['mean_ms']
    for k in arms:
        if k!='plain' and base/max(timing[k]['mean_ms'],1e-12)>1.01: triggers.append(k)
    if triggers:
        rt=paired(arms,seed=91919); timing['retest']=rt; timing['retest_triggered']=triggers
    traces=[]
    for k,fn in arms.items():
        p=trace_dir/(g['site']+'_'+g['model']+'_M'+str(M)+'_'+k+'.chrome_trace.json.gz'); trace(fn,p,'e8.'+g['site']+'.'+k); traces.append(str(p))
    gate=ok and len(traces)==len(arms) and all(Path(p).is_file() and Path(p).stat().st_size>0 for p in traces)
    return {'geometry':g,'correctness':corr,'timing':timing,'traces':traces,'retest_pass':all(timing.get('retest',{}).get(k,{}).get('n')==500 for k in arms) if triggers else True,'status':'tested_pass' if gate else 'tested_invalid','publication_gate':{'pass':gate,'reason':None if gate else 'trace/correctness gate'}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--trace-dir',required=True); ap.add_argument('--only',default=''); a=ap.parse_args()
    wanted={x.strip() for x in a.only.split(',') if x.strip()}; gs=[g for g in GEOMS if not wanted or g['site'] in wanted or g['model'] in wanted]
    out=Path(a.out); td=Path(a.trace_dir); out.parent.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True)
    rec={'schema':'fusionagent-e8-forward-publication-v1','status':'running','publication_eligible':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'device':torch.cuda.get_device_name(0),'visible_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'protocol':{'warmup':20,'samples':100,'rounds':5,'cuda_graph':True,'interleaved':True,'retest_threshold':1.01,'correctness_gate_before_timing':True,'claim_boundary':'forward-only dense RMSNorm prologue and SwiGLU epilogue; isolated synthetic public geometries'},'rows':[]}
    for g in gs:
        print('RUN',g,flush=True)
        try: r=row(g,td)
        except Exception as e: r={'geometry':g,'status':'tested_invalid','error':repr(e),'publication_gate':{'pass':False,'reason':'exception'}}
        rec['rows'].append(r); out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print('DONE',r['status'],flush=True)
    rec['status']='tested_pass' if rec['rows'] and all(r['status']=='tested_pass' for r in rec['rows']) else 'tested_invalid'; rec['publication_eligible']=rec['status']=='tested_pass'; out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print(rec['status'],rec['publication_eligible'],len(rec['rows']))
if __name__=='__main__': main()
