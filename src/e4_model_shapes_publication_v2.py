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


# H100 refinement APIs and development-only verification.
import sys
import triton
import e1_rope_publication_v1 as rope
from fusion_harness import paired as device_paired, compiler_evidence
def tune_h100_residual(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rec = {'environment': {'triton': triton.__version__, 'torch': torch.__version__}, 'rows': []}
    mon = rope.ExclusiveMonitor()
    rec['gpu_before'] = rope.gpu_snapshot()
    mon.start()
    configs = [(128, 256, 64, 8, 3), (128, 128, 64, 8, 3), (64, 128, 64, 4, 3), (128, 64, 64, 4, 3), (64, 64, 64, 4, 3)]
    try:
        for m, k, n in [(128, 28672, 8192), (2048, 25600, 5120), (512, 4096, 4096)]:
            torch.manual_seed(1909)
            a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) * k ** (-0.5)
            w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * k ** (-0.5)
            r = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)
            bias = torch.zeros(n, device='cuda', dtype=torch.bfloat16)
            ref = a.float() @ w.float().t() + r.float()
            arms = {}
            checks = {}
            evidence = {}
            chosen = {}
            for flag in (False, True):
                name = 'original_auto_fused' if flag else 'original_auto_unfused'
                out = torch.empty_like(r)
                tmp = torch.empty_like(r)

                def call(out=out, tmp=tmp, flag=flag):
                    kernel = ep.gemm_res_kernel[lambda meta: (triton.cdiv(m, meta['BLOCK_M']) * triton.cdiv(n, meta['BLOCK_N']),)](a, w, r, bias, out if flag else tmp, m, n, k, k, 1, k, 1, n, 1, FUSE_RES=flag)
                    if not flag:
                        ep._add_kernel[triton.cdiv(m * n, 4096),](tmp, r, out, m * n, 4096, num_warps=8)
                    return kernel
                compiled = call()
                chosen[name] = str(ep.gemm_res_kernel.best_config)
                check = rope.metric(out, ref)
                check['pass'] = bool(torch.allclose(out.float(), ref, atol=0.05, rtol=0.05))
                checks[name] = check
                arms[name] = call
                if compiled is not None:
                    evidence[name] = compiler_evidence(compiled, args.out / 'compiler', f'{m}_{k}_{n}_{name}')
            for bm, bn, bk, warps, stages in configs:
                for flag in (False, True):
                    name = f"{('fused' if flag else 'unfused')}_{bm}_{bn}_{bk}_{warps}_{stages}"
                    out = torch.empty_like(r)
                    tmp = torch.empty_like(r)

                    def call(out=out, tmp=tmp, flag=flag, bm=bm, bn=bn, bk=bk, warps=warps, stages=stages):
                        kernel = ep.gemm_res_kernel.fn[triton.cdiv(m, bm) * triton.cdiv(n, bn),](a, w, r, bias, out if flag else tmp, m, n, k, k, 1, k, 1, n, 1, FUSE_RES=flag, GROUP_M=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)
                        if not flag:
                            ep._add_kernel[triton.cdiv(m * n, 4096),](tmp, r, out, m * n, 4096, num_warps=8)
                        return kernel
                    try:
                        compiled = call()
                        checks[name] = {'pass': bool(torch.allclose(out.float(), ref, atol=0.05, rtol=0.05))}
                        evidence[name] = compiler_evidence(compiled, args.out / 'compiler', f'{m}_{k}_{n}_{name}')
                        if checks[name]['pass']:
                            arms[name] = call
                    except Exception as exc:
                        checks[name] = {'pass': False, 'error': repr(exc)}
            vendor_out = torch.empty_like(r)
            arms['vendor_addmm'] = lambda: torch.addmm(r, a, w.t(), out=vendor_out)
            arms['vendor_addmm']()
            checks['vendor_addmm'] = {'pass': bool(torch.allclose(vendor_out.float(), ref, atol=0.05, rtol=0.05))}
            assert checks['original_auto_fused']['pass'] and checks['original_auto_unfused']['pass'] and checks['vendor_addmm']['pass']
            row = {'shape': [m, k, n], 'correctness': checks, 'compiler': evidence, 'autotune_choices': chosen, 'timing': device_paired(arms, rounds=5, samples=7, replays=20, seed=33109)}
            rec['rows'].append(row)
            (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')
            print(row['shape'], chosen, sorted([(n, round(t['median_us'], 3)) for n, t in row['timing']['arms'].items()], key=lambda x: x[1]), flush=True)
        rec['status'] = 'passed'
    finally:
        rec['gpu_observations'] = mon.finish()
        rec['gpu_after'] = rope.gpu_snapshot()
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')

def verify_h100_residual(args):
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    rec = {'environment': {'torch': torch.__version__, 'triton': triton.__version__}, 'rows': [], 'source_hashes': {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (ep, rope)}}
    mon = rope.ExclusiveMonitor()
    rec['gpu_before'] = rope.gpu_snapshot()
    mon.start()
    try:
        originals = {(128, 28672, 8192): (128, 64, 64, 4, 4), (2048, 25600, 5120): (128, 256, 64, 8, 3), (512, 4096, 4096): (128, 128, 64, 8, 4)}
        for shape in ep.H100_RESIDUAL_CONFIGS:
            m, k, n = shape
            torch.manual_seed(5909)
            a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16) * k ** (-0.5)
            w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16) * k ** (-0.5)
            r = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)
            bias = torch.zeros(n, device='cuda', dtype=torch.bfloat16)
            ref = a.float() @ w.float().t() + r.float()
            arms = {}
            outputs = {}
            evidence = {}
            for name, config in [('original_selected', originals[shape]), ('improved', ep.H100_RESIDUAL_CONFIGS[shape])]:
                for flag in (False, True):
                    label = name + ('_fused' if flag else '_unfused')
                    out = torch.empty_like(r)
                    tmp = torch.empty_like(r)
                    bm, bn, bk, warps, stages = config

                    def call(out=out, tmp=tmp, flag=flag, bm=bm, bn=bn, bk=bk, warps=warps, stages=stages):
                        kernel = ep.gemm_res_kernel.fn[triton.cdiv(m, bm) * triton.cdiv(n, bn),](a, w, r, bias, out if flag else tmp, m, n, k, k, 1, k, 1, n, 1, FUSE_RES=flag, GROUP_M=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)
                        if not flag:
                            ep._add_kernel[triton.cdiv(m * n, 4096),](tmp, r, out, m * n, 4096, num_warps=8)
                        return kernel
                    arms[label] = call
                    outputs[label] = out
                    evidence[label] = compiler_evidence(call(), args.out / 'compiler', f'{m}_{k}_{n}_{label}')
            vo = torch.empty_like(r)
            arms['vendor_addmm'] = lambda: torch.addmm(r, a, w.t(), out=vo)
            outputs['vendor_addmm'] = vo
            checks = {}
            for name, call in arms.items():
                call()
                checks[name] = bool(torch.allclose(outputs[name].float(), ref, atol=0.05, rtol=0.05))
                assert checks[name], (shape, name)
            timing = device_paired(arms, rounds=5, samples=7, replays=20, seed=6109)
            public_checks = []
            for seed in (6909, 7909, 8909):
                torch.manual_seed(seed)
                aa = torch.randn_like(a)
                ww = torch.randn_like(w) * k ** (-0.5)
                rr = torch.randn_like(r)
                reference = aa.float() @ ww.float().t() + rr.float()
                for backend in ('triton', 'vendor'):
                    yy = ep.gemm_residual(aa, ww, rr, backend=backend)
                    check = bool(torch.allclose(yy.float(), reference, atol=0.05, rtol=0.05))
                    assert check, (shape, seed, backend)
                    public_checks.append({'seed': seed, 'backend': backend, 'pass': check})
            rec['rows'].append({'shape': shape, 'correctness': checks, 'compiler': evidence, 'retest': timing, 'public_api_checks': public_checks})
            print(shape, {n: round(v['median_us'], 3) for n, v in timing['arms'].items()}, flush=True)
        rec['attention_api_checks'] = []
        for model, rotary in [('Qwen3-8B', True), ('Gemma-3-4B', False)]:
            for s in (33, 129):
                for causal in (False, True):
                    xs = rope.refinement_inputs(model, 1, s, 9909, rotary)
                    reference = rope.refinement_oracle(xs, causal, rotary)
                    for mode in ('query', 'unfused'):
                        for bn in (64, 128):
                            yy = rope.optimized_attention(*xs[:5], *(xs[5:] if rotary else ()), causal=causal, mode=mode, bn=bn, norm_style='hf' if rotary else 'legacy')
                            check = rope.metric(yy, reference)
                            assert check['pass'], (model, s, mode, bn, check)
                            rec['attention_api_checks'].append({'model': model, 's': s, 'causal': causal, 'mode': mode, 'bn': bn, **check})
        xs = rope.refinement_inputs('Qwen3-8B', 1, 33, 9909, True)
        rec['invalid_api_checks'] = []
        for kwargs in ({'mode': 'bad'}, {'bn': 31}, {'norm_style': 'bad'}):
            try:
                rope.optimized_attention(*xs, **kwargs)
            except ValueError:
                rec['invalid_api_checks'].append({'kwargs': kwargs, 'pass': True})
            else:
                raise AssertionError(kwargs)
        rec['status'] = 'passed'
    finally:
        rec['gpu_observations'] = mon.finish()
        rec['gpu_after'] = rope.gpu_snapshot()
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')

if __name__ == '__main__':
    if '--refine' in sys.argv or '--tune' in sys.argv:
        parser=argparse.ArgumentParser(description='H100 residual schedule refinement')
        parser.add_argument('--refine',action='store_true')
        parser.add_argument('--tune',action='store_true')
        parser.add_argument('--out',type=Path,required=True)
        args=parser.parse_args()
        (tune_h100_residual if args.tune else verify_h100_residual)(args)
    else:
        main()
