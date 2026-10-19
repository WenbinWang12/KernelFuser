#!/usr/bin/env python3
"""E6 Hopper-equivalent RMSNorm-to-GEMM experiment.

The original E6 candidate is a Blackwell ADS/TLX multi-CTA kernel. Hopper
does not provide its TMEM/CLC instructions, so this runner preserves the
forward contract ``rms(x) @ W`` with a fused SM90 Triton tile kernel. It is a
separate Hopper-equivalent result and never claims native Blackwell TLX.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, json, os, random, statistics, traceback
from pathlib import Path
from typing import Any, Callable
import torch
import triton
import triton.language as tl

EPS=1e-6; DTYPE=torch.bfloat16; ATOL=.15; RTOL=.02
GEOMS={
 "Qwen3-0.6B":(1024,16,8,128,3072), "Qwen3-1.7B":(2048,16,8,128,6144),
 "Qwen3-4B":(2560,32,8,128,9728), "Qwen3-8B":(4096,32,8,128,12288),
 "Qwen3-32B":(5120,64,8,128,25600), "Qwen2.5-0.5B-Instruct":(896,14,2,64,4864),
 "OLMo-1B":(2048,16,16,128,8192), "Llama-3.1-8B":(4096,32,8,128,14336),
 "Mistral-7B":(4096,32,8,128,14336)}

@triton.jit
def _fused(X,W,O,M,N,K,sxm,sxk,swk,swn,som,son,EPS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    pm=tl.program_id(0); pn=tl.program_id(1); rm=pm*BM+tl.arange(0,BM); rn=pn*BN+tl.arange(0,BN); mm=rm<M; mn=rn<N; ss=tl.zeros((BM,),tl.float32)
    for k0 in tl.range(0,K,BK):
        rk=k0+tl.arange(0,BK); mk=rk<K; a=tl.load(X+rm[:,None]*sxm+rk[None,:]*sxk,mm[:,None]&mk[None,:],other=0).to(tl.float32); ss+=tl.sum(a*a,axis=1)
    inv=tl.rsqrt(ss/K+EPS); acc=tl.zeros((BM,BN),tl.float32)
    for k0 in tl.range(0,K,BK):
        rk=k0+tl.arange(0,BK); mk=rk<K; a=tl.load(X+rm[:,None]*sxm+rk[None,:]*sxk,mm[:,None]&mk[None,:],other=0).to(tl.float32); b=tl.load(W+rk[:,None]*swk+rn[None,:]*swn,mk[:,None]&mn[None,:],other=0).to(tl.float32); acc+=tl.dot(a*inv[:,None],b,input_precision="ieee")
    tl.store(O+rm[:,None]*som+rn[None,:]*son,acc.to(tl.bfloat16),mm[:,None]&mn[None,:])

def schedule(m,n,k):
    if m>=512:return {"bm":128,"bn":128,"bk":64,"warps":4,"stages":3}
    if m>=128 and n>8192:return {"bm":128,"bn":128,"bk":64,"warps":4,"stages":3}
    if m>=128:return {"bm":64,"bn":128,"bk":64,"warps":4,"stages":3}
    return {"bm":32,"bn":128,"bk":64,"warps":4,"stages":3}

def fused(x,w,c):
    m,k=x.shape; n=w.shape[1]; o=torch.empty((m,n),device=x.device,dtype=DTYPE); _fused[(triton.cdiv(m,c["bm"]),triton.cdiv(n,c["bn"]))](x,w,o,m,n,k,x.stride(0),x.stride(1),w.stride(0),w.stride(1),o.stride(0),o.stride(1),EPS=EPS,BM=c["bm"],BN=c["bn"],BK=c["bk"],num_warps=c["warps"],num_stages=c["stages"]); return o
def vendor(x,w):
    r=torch.rsqrt((x.float()*x.float()).mean(dim=1,keepdim=True)+EPS); return ((x.float()*r)@w.float()).to(DTYPE)
def sha(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def metric(a,b):
    aa,bb=a.float(),b.float(); d=(aa-bb).abs(); return {"allclose":bool(torch.allclose(aa,bb,atol=ATOL,rtol=RTOL)),"finite":bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()),"max_abs":float(d.max()) if d.numel() else 0.,"max_rel":float((d/bb.abs().clamp_min(1e-12)).max()) if d.numel() else 0.,"atol":ATOL,"rtol":RTOL}
def timed(arms,rounds,samples,warmup,seed):
    for f in arms.values():
        for _ in range(warmup):f()
    torch.cuda.synchronize(); vals={k:[] for k in arms}; rng=random.Random(seed)
    for _ in range(rounds):
        for _ in range(samples):
            order=list(arms); rng.shuffle(order)
            for name in order:
                a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); arms[name](); b.record(); b.synchronize(); vals[name].append(float(a.elapsed_time(b)))
    return {k:{"n":len(v),"samples_ms":v,"p50_ms":statistics.median(v),"mean_ms":statistics.fmean(v),"p95_ms":sorted(v)[max(0,int(.95*len(v))-1)]} for k,v in vals.items()}
def trace(f,path,label):
    path.parent.mkdir(parents=True,exist_ok=True); raw=path.with_suffix("")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        with torch.profiler.record_function(label):f()
        torch.cuda.synchronize()
    p.export_chrome_trace(str(raw));
    with raw.open("rb") as a,gzip.open(path,"wb") as b:b.write(a.read())
    raw.unlink(missing_ok=True); return {"file":path.name,"bytes":path.stat().st_size,"sha256":sha(path)}
def dims(model,site):
    h,q,kv,hd,inter=GEOMS[model]; return h,((q+2*kv)*hd if site=="qkv" else 2*inter)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--csv",type=Path); p.add_argument("--trace-dir",type=Path); p.add_argument("--hardware",choices=("h100","h200"),required=True); p.add_argument("--physical-gpu",type=int,required=True); p.add_argument("--models",default=",".join(GEOMS)); p.add_argument("--sites",default="qkv,gate_up"); p.add_argument("--m-values",default="1,32,128,512"); p.add_argument("--smoke",action="store_true"); p.add_argument("--rounds",type=int,default=5); p.add_argument("--samples",type=int,default=7); p.add_argument("--warmup",type=int,default=20); p.add_argument("--source-commit",default="working-tree"); a=p.parse_args()
    if a.smoke:a.models,a.sites,a.m_values="Qwen3-8B","qkv","128"
    rec={"schema":"fusionagent-e6-hopper-equivalent-v1","experiment":"E6","status":"started","hardware_variant":a.hardware,"backend":"triton_sm90_hopper","native_tlx_status":"blocked_blackwell_primitives","functional_equivalence":"rmsnorm(x) @ W","publication_eligible":False,"source_commit":a.source_commit,"source_sha256":sha(Path(__file__)),"physical_gpu":a.physical_gpu,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"capability":list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,"protocol":{"dtype":"bfloat16","eps":EPS,"atol":ATOL,"rtol":RTOL,"correctness_before_timing":True,"warmup":a.warmup,"rounds":a.rounds,"samples":a.samples,"paired_randomized":True,"claim_boundary":"Hopper-equivalent forward kernel; no Blackwell instruction or E2E claim"},"rows":[]}
    try:
        allowed=set(range(4)) if a.hardware=="h100" else set(range(4,8)); vis=os.environ.get("CUDA_VISIBLE_DEVICES","");
        if not torch.cuda.is_available() or int(vis.split(",")[0]) not in allowed or int(vis.split(",")[0])!=a.physical_gpu:raise RuntimeError("physical GPU attestation failed")
        if torch.cuda.get_device_capability(0)[0]!=9:raise RuntimeError("SM90/Hopper required")
        csv_rows=[]
        for model in [x for x in a.models.split(",") if x]:
            for site in [x for x in a.sites.split(",") if x]:
                for m in [int(x) for x in a.m_values.split(",") if x]:
                    k,n=dims(model,site); c=schedule(m,n,k); g=torch.Generator(device="cuda").manual_seed(20261018+len(rec["rows"])); x=torch.randn((m,k),device="cuda",dtype=DTYPE,generator=g); w=torch.randn((k,n),device="cuda",dtype=DTYPE,generator=g)*(k**-.5); gamma=(torch.randn((k,),device="cuda",dtype=DTYPE,generator=g)*.1+1).contiguous(); wf=(w.float()*gamma.float()[:,None]).to(DTYPE).contiguous(); base=lambda:vendor(x,wf); cand=lambda:fused(x,wf,c); ref=base(); out=cand(); torch.cuda.synchronize(); row={"model":model,"site":site,"M":m,"K":k,"N":n,"schedule":c,"correctness":metric(out,ref)}
                    if row["correctness"]["allclose"] and row["correctness"]["finite"]:
                        t=timed({"hopper_fused":cand,"vendor_rms_gemm":base},a.rounds,a.samples,a.warmup,7000+len(rec["rows"])); row.update(timing=t,speedup_vs_vendor=t["vendor_rms_gemm"]["p50_ms"]/t["hopper_fused"]["p50_ms"],status="tested_pass");
                        if a.trace_dir:row["traces"]=[trace(base,a.trace_dir/f"{model}_{site}_M{m}_vendor.json.gz","e6.vendor"),trace(cand,a.trace_dir/f"{model}_{site}_M{m}_hopper.json.gz","e6.hopper")]
                        csv_rows.append({"model":model,"site":site,"M":m,"K":k,"N":n,"candidate_p50_ms":t["hopper_fused"]["p50_ms"],"vendor_p50_ms":t["vendor_rms_gemm"]["p50_ms"],"speedup":row["speedup_vs_vendor"],"status":"tested_pass"})
                    else:row["status"]="tested_invalid"
                    rec["rows"].append(row); print("DONE",model,site,m,row["status"],flush=True); del x,w,wf,gamma,ref,out; torch.cuda.empty_cache()
        rec["status"]="tested_pass" if rec["rows"] and all(r["status"]=="tested_pass" for r in rec["rows"]) else "tested_invalid"
        if a.csv and csv_rows:
            a.csv.parent.mkdir(parents=True,exist_ok=True)
            with a.csv.open("w",newline="") as f:wr=csv.DictWriter(f,fieldnames=list(csv_rows[0]));wr.writeheader();wr.writerows(csv_rows)
    except Exception as e:rec.update(status="error",error_type=type(e).__name__,error=repr(e),traceback=traceback.format_exc())
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(rec,indent=2)+"\n");print(json.dumps({"status":rec["status"],"rows":len(rec["rows"]),"out":str(a.out)}),flush=True);return 0 if rec["status"]=="tested_pass" else 1
if __name__=="__main__":raise SystemExit(main())
