#!/usr/bin/env python3
"""Publication E7 A/A noise-floor runner with provenance and independent retest."""
from __future__ import annotations
import argparse,gzip,hashlib,json,os,random,statistics,subprocess,threading,time,platform
from pathlib import Path
import torch
import e1_qk_publication_v3 as e1

ROWS=[("Qwen3-8B",512,False),("Gemma-3-12B",1024,True),("Qwen3-30B-A3B",2048,False)]
WARM,SAMPLES,ROUNDS=20,100,5
def stat(xs):
    if not xs or not all(torch.isfinite(torch.tensor(xs)).tolist()): raise RuntimeError("non-finite timing")
    m=statistics.fmean(xs); se=statistics.stdev(xs)/(len(xs)**.5)
    return {"warmup":WARM,"samples":SAMPLES,"rounds":ROUNDS,"n":len(xs),"samples_ms":xs,"mean_ms":m,"sem_ms":se,"two_sem_ms":2*se,"p50_ms":statistics.median(xs)}
def capture(fn):
    for _ in range(WARM): out=fn()
    torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    torch.cuda.synchronize(); return g,out
def timed(g1,g2,seed):
    rng=random.Random(seed); a=[];b=[];order=[]
    def m(g):
        s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True); s.record(); g.replay(); e.record(); e.synchronize(); return float(s.elapsed_time(e))
    for _ in range(ROUNDS*SAMPLES):
        if rng.random()<.5: order.append("A1_first"); a.append(m(g1)); b.append(m(g2))
        else: order.append("A2_first"); b.append(m(g2)); a.append(m(g1))
    return stat(a),stat(b),order
def trace(g,path,label):
    path.parent.mkdir(parents=True,exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        with torch.profiler.record_function(label): g.replay()
        torch.cuda.synchronize()
    raw=Path(str(path)[:-3]); p.export_chrome_trace(str(raw))
    with raw.open('rb') as f,gzip.open(path,'wb') as z: z.write(f.read())
    raw.unlink(missing_ok=True)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--trace-dir',required=True); a=ap.parse_args(); out=Path(a.out); td=Path(a.trace_dir); out.parent.mkdir(parents=True,exist_ok=True)
    rec={"schema":"fusionagent-e7-aa-v2","status":"running","publication_eligible":False,"created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"environment":{"host":platform.node(),"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"visible_gpu":os.getenv('CUDA_VISIBLE_DEVICES'),"device_count":torch.cuda.device_count(),"device_uuid_inventory":subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).splitlines()},"protocol":{"warmup":WARM,"samples":SAMPLES,"rounds":ROUNDS,"cuda_graph":True,"randomized_arm_order":True,"independent_retest":True,"retest_seed":1000003,"same_baseline_graph":True},"rows":[]}
    for name,s,causal in ROWS:
        e1.CAUSAL_MODE=causal; cfg=e1.MODELS[name]; mon=e1.ExclusiveMonitor(); before=e1.gpu_exclusivity_snapshot(); mon.start()
        try:
            if not before.get('exclusive'): raise RuntimeError(f'non-exclusive before: {before}')
            xs1,_=e1.inputs(cfg,s,1701,False); xs2=[x.detach().clone() for x in xs1]
            g1,o1=capture(lambda:e1.baseline(*xs1)); g2,o2=capture(lambda:e1.baseline(*xs2)); torch.cuda.synchronize()
            finite=bool(torch.isfinite(o1).all() and torch.isfinite(o2).all()); equal=bool(torch.allclose(o1,o2,atol=0,rtol=0));
            a1,a2,ord1=timed(g1,g2,20261112); r1,r2,ord2=timed(g1,g2,1000003)
            paths=[]
            for lab,g in [('A1',g1),('A2',g2)]: p=td/f'{name}_S{s}_causal{int(causal)}_{lab}.chrome_trace.json.gz'; trace(g,p,f'e7.{name}.{lab}'); paths.append(str(p))
            obs=mon.finish(); after=e1.gpu_exclusivity_snapshot(); gate=finite and equal and before.get('exclusive') and after.get('exclusive') and all(x.get('exclusive') for x in obs) and len(paths)==2
            rec['rows'].append({'model':name,'S':s,'causal':causal,'correctness':{'finite':finite,'same_graph_output_equal':equal,'pass':gate},'initial':{'A1':a1,'A2':a2,'ratio':a1['mean_ms']/a2['mean_ms'],'seed':20261112,'arm_order':ord1},'retest':{'A1':r1,'A2':r2,'ratio':r1['mean_ms']/r2['mean_ms'],'seed':1000003,'arm_order':ord2},'traces':paths,'gpu_exclusivity':{'before':before,'observations':obs,'after':after},'status':'tested_pass' if gate else 'tested_invalid'})
        except Exception as err:
            try: obs=mon.finish()
            except Exception: obs=[]
            rec['rows'].append({'model':name,'S':s,'causal':causal,'status':'tested_invalid','error':repr(err),'gpu_exclusivity':{'before':before,'observations':obs}})
        out.write_text(json.dumps(rec,indent=2)+'\n')
    rec['status']='tested_pass' if all(x.get('status')=='tested_pass' for x in rec['rows']) else 'tested_invalid'; rec['publication_eligible']=rec['status']=='tested_pass'; out.write_text(json.dumps(rec,indent=2)+'\n'); print(rec['status'],flush=True)
if __name__=='__main__': main()
