#!/usr/bin/env python3
"""SM90 fallback adapted from ADS TLX RMSNorm+matmul.

The upstream SM100 kernel relies on TMEM, CLC/cluster launch control, TMA
descriptors, and warp-specialized async_dot.  This fallback keeps the same
RMSNorm-before-GEMM math and a TLX async-task region, but uses register/SMEM
loads and ordinary tl.dot so it can compile on Hopper.  It is an ADS-derived
SM90 port, not the upstream Blackwell kernel.
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import torch, triton, triton.language as tl
import triton.language.extra.tlx as tlx

BM=32; BN=128; BK=64; EPS=1e-6

@triton.jit
def _triton_fwd(xp, wp, op, M, N, K, stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m=tl.program_id(0); pid_n=tl.program_id(1)
    rm=pid_m*BLOCK_M+tl.arange(0,BLOCK_M); rn=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
    mm=rm<M; mn=rn<N
    ss=tl.zeros((BLOCK_M,),dtype=tl.float32)
    for k0 in tl.range(0,K,BLOCK_K):
        rk=k0+tl.arange(0,BLOCK_K); mk=rk<K
        a=tl.load(xp+rm[:,None]*stride_xm+rk[None,:]*stride_xk,mask=mm[:,None]&mk[None,:],other=0).to(tl.float32)
        ss += tl.sum(a*a,axis=1)
    inv=tl.rsqrt(ss/K+eps)
    acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)
    for k0 in tl.range(0,K,BLOCK_K):
        rk=k0+tl.arange(0,BLOCK_K); mk=rk<K
        a=tl.load(xp+rm[:,None]*stride_xm+rk[None,:]*stride_xk,mask=mm[:,None]&mk[None,:],other=0).to(tl.float32)*inv[:,None]
        b=tl.load(wp+rk[:,None]*stride_wk+rn[None,:]*stride_wn,mask=mk[:,None]&mn[None,:],other=0).to(tl.float32)
        acc += tl.dot(a,b)
    tl.store(op+rm[:,None]*stride_om+rn[None,:]*stride_on,acc.to(tl.bfloat16),mask=mm[:,None]&mn[None,:])

@triton.jit
def _tlx_fwd(xp, wp, op, M, N, K, stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on, eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid_m=tl.program_id(0); pid_n=tl.program_id(1)
            rm=pid_m*BLOCK_M+tl.arange(0,BLOCK_M); rn=pid_n*BLOCK_N+tl.arange(0,BLOCK_N)
            mm=rm<M; mn=rn<N
            ss=tl.zeros((BLOCK_M,),dtype=tl.float32)
            for k0 in tl.range(0,K,BLOCK_K):
                rk=k0+tl.arange(0,BLOCK_K); mk=rk<K
                a=tl.load(xp+rm[:,None]*stride_xm+rk[None,:]*stride_xk,mask=mm[:,None]&mk[None,:],other=0).to(tl.float32)
                ss += tl.sum(a*a,axis=1)
            inv=tl.rsqrt(ss/K+eps)
            acc=tl.zeros((BLOCK_M,BLOCK_N),dtype=tl.float32)
            for k0 in tl.range(0,K,BLOCK_K):
                rk=k0+tl.arange(0,BLOCK_K); mk=rk<K
                a=tl.load(xp+rm[:,None]*stride_xm+rk[None,:]*stride_xk,mask=mm[:,None]&mk[None,:],other=0).to(tl.float32)*inv[:,None]
                b=tl.load(wp+rk[:,None]*stride_wk+rn[None,:]*stride_wn,mask=mk[:,None]&mn[None,:],other=0).to(tl.float32)
                acc += tl.dot(a,b)
            tl.store(op+rm[:,None]*stride_om+rn[None,:]*stride_on,acc.to(tl.bfloat16),mask=mm[:,None]&mn[None,:])

def launch(kind,x,w,out):
    fn=_tlx_fwd if kind=='tlx' else _triton_fwd
    fn[(triton.cdiv(x.shape[0],BM),triton.cdiv(w.shape[1],BN))](x,w,out,x.shape[0],w.shape[1],x.shape[1],x.stride(0),x.stride(1),w.stride(0),w.stride(1),out.stride(0),out.stride(1),EPS,BLOCK_M=BM,BLOCK_N=BN,BLOCK_K=BK,num_warps=4)

def stats(vals):
    xs=sorted(float(v) for v in vals); n=len(xs); q=lambda p: xs[min(n-1,max(0,int(round((n-1)*p))))]
    return {'n':n,'p50_ms':q(.5),'p90_ms':q(.9),'p99_ms':q(.99),'mean_ms':sum(xs)/n,'min_ms':xs[0],'max_ms':xs[-1],'samples_ms':xs}
def timed(fn):
    for _ in range(20): fn()
    torch.cuda.synchronize(); vals=[]
    for _ in range(100):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); vals.append(a.elapsed_time(b))
    return stats(vals)
def trace(fn,path):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize(); p.step()
    p.export_chrome_trace(str(path))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--M',type=int,default=128); ap.add_argument('--K',type=int,default=1024); ap.add_argument('--N',type=int,default=3072); args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261007); x=torch.randn((args.M,args.K),device='cuda',dtype=torch.bfloat16); w=torch.randn((args.K,args.N),device='cuda',dtype=torch.bfloat16); ref=(x.float()*torch.rsqrt((x.float()*x.float()).mean(dim=1,keepdim=True)+EPS))@w.float(); ref=ref.to(torch.bfloat16)
    rec={'schema':'kernelfuser_ads_sm90_port_v1','status':'started','implementation_family':['triton','tlx'],'implementation_origin':'ads_sm90_port','upstream_source':'https://github.com/facebookresearch/ads_model_kernel_library/blob/2889aefba0d03b3eecf779eed9f4097066ecdc5d/multi_cta_norm_fusion/src/tlx_rmsnorm_matmul.py','upstream_sha256':'26f66c287a957fa324eca24473509823b84779111acd05b07d0ea10061de4a29','removed_primitives':['storage_kind.tmem','cluster_launch_control','TMA tensor descriptors','async_dot/WGMMA-specific pipeline'],'kept_primitives':['triton.language.extra.tlx','tlx.async_tasks'],'hardware':{'name':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability()),'visible':os.environ.get('CUDA_VISIBLE_DEVICES')},'workload':{'M':args.M,'K':args.K,'N':args.N,'dtype':'bfloat16','eps':EPS},'warmup':20,'repeats':100,'claim_boundary':'kernel-level ADS-derived SM90 RMSNorm+matmul forward fallback; no module/E2E claim','implementations':{}}
    for kind in ('triton','tlx'):
        out=torch.empty((args.M,args.N),device='cuda',dtype=torch.bfloat16); launch(kind,x,w,out); torch.cuda.synchronize(); rec['implementations'][kind]={'max_abs':float((out.float()-ref.float()).abs().max()),'allclose':bool(torch.allclose(out,ref,atol=.15,rtol=.02)),'timing':timed(lambda:launch(kind,x,w,out))}; trace(lambda:launch(kind,x,w,out),args.out/f'{kind}.chrome_trace.json')
    rec['comparison']={'triton_over_tlx':rec['implementations']['triton']['timing']['p50_ms']/rec['implementations']['tlx']['timing']['p50_ms']}; rec['status']='tested_pass' if all(rec['implementations'][k]['allclose'] for k in ('triton','tlx')) else 'tested_invalid'; rec['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); (args.out/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n'); print(json.dumps({'status':rec['status'],'ratio':rec['comparison']['triton_over_tlx'],'out':str(args.out)}))
if __name__=='__main__': main()
