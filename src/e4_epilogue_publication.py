#!/usr/bin/env python3
"""Strict forward-only E4 residual/LM-head epilogue receipt.

The kernels are imported from the campaign's open epilogue implementation.  We
run every arm with the same tensors, fresh CUDA graphs, randomized paired
replays, FP32 correctness, explicit 2SEM, and an independent retest whenever
the fused arm first exceeds the 1.01 gate.
"""
from __future__ import annotations
import argparse, gzip, hashlib, importlib.util, json, math, os, random, statistics, time
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
def load(name,p):
    s=importlib.util.spec_from_file_location(name,str(p)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ep=load("epilogue_sites", ROOT/"epilogue_sites.py")

def stat(vals,warmup=20,samples=100,rounds=5):
    vals=[float(x) for x in vals]
    if len(vals)!=samples*rounds or not all(math.isfinite(x) and x>=0 for x in vals):
        raise ValueError("non-finite or incomplete timing samples")
    mean=statistics.fmean(vals); sem=statistics.stdev(vals)/(len(vals)**.5)
    return {"n":len(vals),"mean_ms":mean,"sem_ms":sem,"two_sem_ms":2*sem,
            "p50_ms":statistics.median(vals),"p95_ms":sorted(vals)[math.ceil(.95*len(vals))-1],
            "samples_ms":vals,"warmups":warmup,"samples":samples,"rounds":rounds}

def cap(fn,warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    torch.cuda.synchronize(); return g

def paired(arms,rounds=5,samples=100,seed=7707):
    names=list(arms); allvals={n:[] for n in names}; orders=[]
    for rnd in range(rounds):
        gs={n:cap(arms[n]) for n in names}; rng=random.Random(seed+rnd)
        for i in range(samples):
            order=names[:]; rng.shuffle(order); orders.append(order[:])
            for n in order:
                a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
                a.record(); gs[n].replay(); b.record(); b.synchronize(); allvals[n].append(float(a.elapsed_time(b)))
        del gs; torch.cuda.empty_cache()
    out={n:stat(v) for n,v in allvals.items()}; out["_protocol"]={"warmup":20,"samples":samples,"rounds":rounds,"cuda_graph":True,"interleaved":True,"arm_order":"randomized_per_pair_sample","orders":orders}
    return out

def retest(arms,rounds=5,samples=100):
    return paired(arms,rounds,samples,seed=91031)

def trace(fn,path,label):
    path.parent.mkdir(parents=True,exist_ok=True); raw=Path(str(path)[:-3])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        with torch.profiler.record_function(label): fn()
        p.step()
    p.export_chrome_trace(str(raw))
    with raw.open("rb") as a,gzip.open(path,"wb") as z:z.write(a.read())
    raw.unlink(missing_ok=True)

def close(a,b,atol=0.05,rtol=0.05):
    d=(a.float()-b.float()).detach(); den=torch.linalg.vector_norm(b.float()).clamp_min(1e-8)
    return {"max_abs":float(d.abs().max()),"relative_l2":float(torch.linalg.vector_norm(d)/den),"allclose":bool(torch.allclose(a.float(),b.float(),atol=atol,rtol=rtol))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--trace-dir",required=True); ap.add_argument("--sites",default="gemm_residual"); ap.add_argument("--M",default="128,512,2048,8192"); a=ap.parse_args()
    out=Path(a.out); td=Path(a.trace_dir); out.parent.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True)
    # Keep the source geometry explicit in the receipt.  A site is a dense
    # architecture/configuration point, not a full model claim.
    rows=[]; rec={"schema":"fusionagent-e4-epilogue-forward-v1","status":"running","publication_eligible":False,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"device":torch.cuda.get_device_name(0),"visible_gpu":os.getenv("CUDA_VISIBLE_DEVICES"),"protocol":{"warmup":20,"samples":100,"rounds":5,"cuda_graph":True,"interleaved":True,"retetest_threshold":1.01,"claim_boundary":"forward-only; isolated residual epilogue and RMSNorm->LM-head GEMM"},"sites":a.sites.split(","),"rows":rows}
    for name in rec["sites"]:
        pat=ep.PATTERNS[name.strip()]()
        # Override old development sweep with the planned M points.
        for g in pat.geometries():
            if int(g[1]) not in [int(x) for x in a.M.split(",")]: continue
            label=pat.label(g); print("RUN",label,flush=True); arms=pat.arms(g); ref=arms["reference"](); corr={}
            for arm,fn in arms.items():
                if arm=="reference" or arm=="host_simb": continue
                try: corr[arm]=close(fn(),ref)
                except Exception as e: corr[arm]={"allclose":False,"error":repr(e)}
            timing={}; errors=[]; traces=[]
            try:
                timing=paired({k:arms[k] for k in arms if k!="reference"},5,100)
                fused="fused_simb"; base="plain_simb"
                ratio=timing[base]["p50_ms"]/timing[fused]["p50_ms"] if timing[fused]["p50_ms"] else 0
                timing["initial_fused_ratio"]=ratio
                if ratio>1.01:
                    rt=retest({k:arms[k] for k in arms if k!="reference"},5,100)
                    timing["retest"]=rt; timing["retest_fused_ratio"]=rt[base]["p50_ms"]/rt[fused]["p50_ms"]
                for arm,fn in arms.items():
                    if arm=="reference": continue
                    p=td/(name.strip()+"_"+str(g[0])+"_M"+str(g[1])+"_"+arm+".chrome_trace.json.gz"); trace(fn,p,"e4."+name+"."+arm); traces.append(str(p))
            except Exception as e: errors.append(repr(e))
            gate=all(v.get("allclose",False) for v in corr.values()) and not errors and len(traces)==len(arms)-1 and all(Path(p).is_file() and Path(p).stat().st_size>0 for p in traces)
            row={"site":name.strip(),"geometry":list(g),"correctness":corr,"timing":timing,"traces":traces,"errors":errors,"status":"tested_pass" if gate else "tested_invalid","publication_gate":{"pass":gate,"reasons":[] if gate else ["correctness/timing/trace gate failed"]}}
            rows.append(row); out.write_text(json.dumps(rec,indent=2,default=str)+"\n"); print("DONE",label,row["status"],flush=True)
    rec["status"]="tested_pass" if rows and all(r["status"]=="tested_pass" for r in rows) else "tested_invalid"; rec["publication_eligible"]=rec["status"]=="tested_pass"; out.write_text(json.dumps(rec,indent=2,default=str)+"\n"); print(rec["status"],rec["publication_eligible"],len(rows),flush=True)
if __name__=="__main__": main()
