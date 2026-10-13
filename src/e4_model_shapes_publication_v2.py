#!/usr/bin/env python3
"""Publication E4: real model projection shapes with explicit cuBLAS addmm baseline."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,platform,random,statistics,subprocess,threading,time,traceback
from pathlib import Path
import torch
import epilogue_sites as ep

DT=torch.bfloat16
M_VALUES=(128,512,2048,8192)
GEOMETRIES=(
 ('llama3-8b','o_proj',128,4096,4096),('llama3-8b','down_proj',128,14336,4096),
 ('llama3-70b','o_proj',128,8192,8192),('llama3-70b','down_proj',128,28672,8192),
 ('qwen3-8b','o_proj',128,4096,4096),('qwen3-8b','down_proj',128,12288,4096),
 ('qwen3-32b','o_proj',128,5120,5120),('qwen3-32b','down_proj',128,25600,5120),
)
# M is replaced by the command-line sweep; the tuple's first M only keeps labels explicit.

def snapshot():
    vis=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
    try: physical=[int(x.strip()) for x in vis.split(',') if x.strip()]
    except ValueError: physical=[]
    try:
        q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        uuids={int(x.split(',')[0].strip()):x.split(',')[1].strip() for x in q.stdout.splitlines() if ',' in x}
        selected={uuids.get(i) for i in physical}; q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        own=os.getpid(); apps=[]; foreign=[]
        for line in q.stdout.splitlines():
            if ',' not in line: continue
            uid,p=line.split(',',1); uid=uid.strip(); p=int(p.strip())
            if uid in selected:
                apps.append({'gpu_uuid':uid,'pid':p,'self':p==own})
                if p!=own: foreign.append({'gpu_uuid':uid,'pid':p})
        return {'utc':time.time(),'physical_gpus':physical,'pid':own,'apps':apps,'foreign_apps':foreign,'exclusive':bool(physical) and not foreign,'host':platform.node()}
    except Exception as e:
        return {'utc':time.time(),'physical_gpus':physical,'pid':os.getpid(),'apps':[],'foreign_apps':[],'exclusive':False,'error':repr(e),'host':platform.node()}

class Monitor:
    def __init__(self): self.stop=threading.Event(); self.records=[]; self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.stop.is_set(): self.records.append(snapshot()); self.stop.wait(1.0)
    def start(self): self.thread.start()
    def finish(self): self.stop.set(); self.thread.join(timeout=20); return self.records

def arms(model,site,M,K,N):
    seed_bytes = hashlib.sha256(f'e4|{model}|{site}|{M}|{K}|{N}'.encode()).digest(); seed = int.from_bytes(seed_bytes[:8], 'little') % (2**31-1); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    A=torch.randn((M,K),device='cuda',dtype=DT)*(K**-0.5)
    W=torch.randn((N,K),device='cuda',dtype=DT)*(K**-0.5)
    R=torch.randn((M,N),device='cuda',dtype=DT)
    d=torch.zeros(N,device='cuda',dtype=DT)
    def reference(): return A.float()@W.float().t()+R.float()
    def plain_simb(): return ep.add_residual(ep.gemm_res(A,W,d,None,False),R)
    def fused_simb(): return ep.gemm_res(A,W,d,R,True)
    def host_simb(): return ep.gemm_res(A,W,d,None,False)
    # Explicit addmm is the cuBLAS/cuBLASLt vendor comparator requested by v23.
    def cublas_addmm(): return torch.addmm(R,A,W.t(),beta=1.0,alpha=1.0)
    return {'_seed': seed, 'reference':reference,'plain_simb':plain_simb,'fused_simb':fused_simb,'host_simb':host_simb,'cublas_addmm':cublas_addmm}

def corr(fn,ref):
    d=(fn().float()-ref); den=torch.linalg.vector_norm(ref).clamp_min(1e-8)
    return {'max_abs':float(d.abs().max()),'relative_l2':float(torch.linalg.vector_norm(d)/den),'allclose':bool(torch.allclose(d+ref,ref,atol=0.05,rtol=0.05))}

def stat(xs):
    xs=[float(x) for x in xs]; mean=statistics.fmean(xs); sem=statistics.stdev(xs)/(len(xs)**0.5)
    return {'n':len(xs),'mean_ms':mean,'sem_ms':sem,'two_sem_ms':2*sem,'p50_ms':statistics.median(xs),'p95_ms':sorted(xs)[math.ceil(.95*len(xs))-1],'samples_ms':xs}

def capture(fn,warm=20):
    with torch.inference_mode():
        for _ in range(warm): fn()
        torch.cuda.synchronize(); g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        torch.cuda.synchronize()
    return g

def paired(armf,rounds=5,samples=100,seed=7707):
    names=list(armf); vals={n:[] for n in names}; orders=[]
    with torch.inference_mode():
      for rnd in range(rounds):
        gs={n:capture(armf[n]) for n in names}; rng=random.Random(seed+rnd)
        for _ in range(samples):
            order=names[:]; rng.shuffle(order); orders.append(order[:])
            for n in order:
                a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record(); gs[n].replay(); b.record(); b.synchronize(); vals[n].append(float(a.elapsed_time(b)))
        del gs; torch.cuda.empty_cache()
    return {n:stat(v) for n,v in vals.items()}|{'_protocol':{'warmup':20,'samples':samples,'rounds':rounds,'randomized_arm_order':True,'orders':orders,'seed':seed}}

def trace(fn,path,label):
    path.parent.mkdir(parents=True,exist_ok=True); raw=Path(str(path)[:-3])
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as p:
        with torch.profiler.record_function(label):
            with torch.inference_mode(): fn()
            torch.cuda.synchronize(); p.step()
    p.export_chrome_trace(str(raw))
    with raw.open('rb') as a,gzip.open(path,'wb') as z:z.write(a.read())
    raw.unlink(missing_ok=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--trace-dir',required=True); ap.add_argument('--M',default=','.join(map(str,M_VALUES))); ap.add_argument('--rounds',type=int,default=5); ap.add_argument('--samples',type=int,default=100); a=ap.parse_args()
    Ms=[int(x) for x in a.M.split(',')]; out=Path(a.out); td=Path(a.trace_dir); out.parent.mkdir(parents=True,exist_ok=True); td.mkdir(parents=True,exist_ok=True)
    rec={'schema':'kernelfuser-e4-real-model-shapes-v1','status':'running','publication_eligible':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'environment':{'host':platform.node(),'gpu':torch.cuda.get_device_name(0),'visible_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'torch':torch.__version__,'cuda':torch.version.cuda,'triton':__import__('triton').__version__},'protocol':{'warmup':20,'samples':a.samples,'rounds':a.rounds,'cuda_graph':True,'interleaved':True,'randomized_arm_order':True,'correctness':'FP32 reference; BF16 arms atol/rtol 0.05','retest_threshold':1.01,'vendor_baseline':'torch.addmm (cuBLAS/cuBLASLt)','gpu_exclusivity':'nvidia-smi selected physical GPU sampled before/after and every second; foreign PID invalidates row'},'rows':[]}
    for model,site,_,K,N in GEOMETRIES:
      for M in Ms:
        label=f'{model} {site} M={M} K={K} N={N}'; print('RUN',label,flush=True); row={'model':model,'site':site,'M':M,'K':K,'N':N,'errors':[],'traces':[],'seed':None,'gpu_exclusivity':{'before':snapshot(),'poll_seconds':1.0}}
        if not row['gpu_exclusivity']['before']['exclusive']:
            row['status']='tested_invalid'; row['errors'].append('GPU not exclusive before row'); rec['rows'].append(row); out.write_text(json.dumps(rec,indent=2)); continue
        monitor=Monitor(); monitor.start(); aarms=None
        try:
            aarms=arms(model,site,M,K,N); row['seed']=aarms['_seed']; ref=aarms['reference'](); row['correctness']={k:corr(v,ref) for k,v in aarms.items() if k not in ('_seed','reference','host_simb')}
            if not all(x['allclose'] for x in row['correctness'].values()): raise RuntimeError('correctness gate failed')
            timed={k:aarms[k] for k in ('plain_simb','fused_simb','cublas_addmm')}; row['timing']=paired(timed,a.rounds,a.samples); row['timing']['fused_vs_plain_p50']=row['timing']['plain_simb']['p50_ms']/row['timing']['fused_simb']['p50_ms']; row['timing']['fused_vs_cublas_p50']=row['timing']['cublas_addmm']['p50_ms']/row['timing']['fused_simb']['p50_ms']
            if row['timing']['fused_vs_plain_p50']>1.01:
                row['timing']['retest']=paired(timed,a.rounds,a.samples,seed=91031); row['timing']['retest_fused_vs_plain_p50']=row['timing']['retest']['plain_simb']['p50_ms']/row['timing']['retest']['fused_simb']['p50_ms']
            for n in timed:
                path=td/f'{model}_{site}_M{M}_{n}.chrome_trace.json.gz'; trace(aarms[n],path,'e4.real.'+n);
                if not path.is_file() or path.stat().st_size == 0: raise RuntimeError(f'empty trace: {path}')
                row['traces'].append(str(path))
            row['status']='tested_pass'
        except Exception as e:
            row['status']='tested_invalid'; row['errors'].append(traceback.format_exc())
        row['gpu_exclusivity']['observations']=monitor.finish(); row['gpu_exclusivity']['after']=snapshot();
        if not row['gpu_exclusivity']['after']['exclusive'] or any(not x.get('exclusive',False) for x in row['gpu_exclusivity']['observations']): row['status']='tested_invalid'; row['errors'].append('GPU exclusivity failed during row')
        row['publication_gate']={'pass':row['status']=='tested_pass' and not row['errors'] and len(row['traces'])==3 and all(Path(p).is_file() and Path(p).stat().st_size>0 for p in row['traces'])}
        rec['rows'].append(row); out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print('DONE',label,row['status'],flush=True)
        del aarms; torch.cuda.empty_cache()
    rec['status']='tested_pass' if rec['rows'] and all(r['status']=='tested_pass' and r.get('publication_gate',{}).get('pass') for r in rec['rows']) else 'tested_invalid'; rec['publication_eligible']=rec['status']=='tested_pass'; out.write_text(json.dumps(rec,indent=2,default=str)+'\n'); print('COMPLETE',rec['status'],len(rec['rows']),flush=True)
if __name__=='__main__': main()
