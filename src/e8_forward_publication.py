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


# H100 refinement APIs and development-only verification.
import sys
def verify_h100_unfused(args):
    import triton
    import e1_rope_publication_v1 as rope
    from tlx_pilot.tlx_swiglu_hopper import launch_swiglu
    from tlx_ads_sm90_port.ads_sm90_rmsnorm_matmul import launch_rms_gemm, launch_rms_gemm_tlx, rstd_kernel
    from fusion_harness import paired as device_paired, metric as kernel_metric
    torch.manual_seed(1909)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch._dynamo.config.cache_size_limit = 512
    args.out.mkdir(parents=True, exist_ok=True)
    rec = {'environment': {'torch': torch.__version__, 'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(), 'cuda': torch.version.cuda}, 'rows': []}
    before = rope.gpu_snapshot()
    mon = rope.ExclusiveMonitor()
    mon.start()

    def save():
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2) + '\n')

    def finish(row, arms):
        assert all((c['pass'] for c in row['correctness'].values())), row
        row['initial'] = device_paired(arms, rounds=5, samples=7, replays=20, seed=7109)
        row['retest'] = device_paired(arms, rounds=5, samples=7, replays=20, seed=17109)
        rec['rows'].append(row)
        save()
        print(row['operation'], row['shape'], {n: round(v['median_us'], 3) for n, v in row['retest']['arms'].items()}, flush=True)
    try:
        for n in (2048 * 3072, 2048 * 14336):
            for backward in (False, True):
                g = torch.randn(n, device='cuda', dtype=torch.bfloat16)
                u, dy = (torch.randn_like(g), torch.randn_like(g))
                y, dg, du = (torch.empty_like(g), torch.empty_like(g), torch.empty_like(g))

                def unfused():
                    gf, uf = (g.float(), u.float())
                    if not backward:
                        return (torch.nn.functional.silu(gf) * uf).bfloat16()
                    s = torch.sigmoid(gf)
                    df = dy.float()
                    return ((df * uf * s * (1 + gf * (1 - s))).bfloat16(), (df * gf * s).bfloat16())

                def fused():
                    return launch_swiglu(g, u, dy, y, dg, du, backend='tlx', block=1024, backward=backward, torch_rounding=False, fast=True)
                reference = unfused()
                fused()
                checks = [kernel_metric(o, r) for o, r in zip((dg, du) if backward else (y,), reference if backward else (reference,))]
                finish({'operation': 'swiglu_backward' if backward else 'swiglu_forward', 'shape': [n], 'baseline': 'eager unfused FP32 pointwise operations, BF16 input/output', 'correctness': {'fused': {'pass': all((x['pass'] for x in checks)), 'outputs': checks}}}, {'unfused_eager': unfused, 'fused_tlx': fused})
        for m, k, n in ((128, 1024, 3072), (512, 1024, 3072), (17, 73, 65), (33, 130, 66)):
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            w = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
            gamma = torch.ones(k, device='cuda', dtype=torch.bfloat16)
            xn = torch.empty_like(x)
            base = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            out = torch.empty_like(base)
            rstd = torch.empty(m, device='cuda', dtype=torch.float32)
            ref = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-06) @ w.float()).bfloat16()
            raw = torch.empty((m, n), device='cuda', dtype=torch.float32)
            scaled = torch.empty_like(raw)

            def unfused_fp32():
                xf = x.float()
                normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-06)
                return (normalized @ w.float()).bfloat16()

            def unfused():
                rstd_kernel[m,](x, rstd, k, 1e-06, triton.next_power_of_2(k), num_warps=4)
                torch.mm(x, w, out=raw, out_dtype=torch.float32)
                torch.mul(raw, rstd[:, None], out=scaled)
                base.copy_(scaled)
                return base

            def fused():
                if k % 2 or n % 2:
                    return launch_rms_gemm(x, w, out, bm=16, bn=64)
                return launch_rms_gemm_tlx(x, w, out, rstd)
            unfused()
            fused()
            finish({'operation': 'rmsnorm_gemm', 'shape': [m, k, n], 'baseline': 'cuBLAS raw BF16 GEMM with FP32 output + separate norm-scale multiplication and BF16 cast', 'candidate': 'TLX row-scale prepass + GEMM epilogue; Triton fallback for odd strides', 'correctness': {'unfused': kernel_metric(base, ref, atol=0.15, rtol=0.02), 'fused': kernel_metric(out, ref, atol=0.15, rtol=0.02), 'unfused_fp32': kernel_metric(unfused_fp32(), ref, atol=0.15, rtol=0.02)}}, {'unfused_norm_cublas': unfused, 'unfused_fp32_norm_before_gemm': unfused_fp32, 'fused_tlx_or_tail_fallback': fused})
        rec['status'] = 'passed'
    finally:
        rec['gpu_before'] = before
        rec['gpu_observations'] = mon.finish()
        rec['gpu_after'] = rope.gpu_snapshot()
        save()

ADS_PUBLIC_SHA256 = '26f66c287a957fa324eca24473509823b84779111acd05b07d0ea10061de4a29'
B200_ADS_SHAPES = (
    (128, 1024, 3072), (512, 1024, 3072),
    (512, 1024, 4096), (2048, 1024, 4096),
    (8192, 1024, 4096), (16384, 1024, 4096),
    (65536, 512, 512), (131072, 512, 512),
    (524288, 512, 512), (262144, 512, 2048),
)


def load_ads_blackwell(path, allow_source_version=False):
    """Load a separately supplied ADS source; never replace it with the port."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ADS_PUBLIC_SHA256 and not allow_source_version:
        raise ValueError('ADS source differs from the pinned public version; '
                         'use the pinned source or explicitly allow another version')
    os.environ['ADS_MKL_DISABLE_AUTOTUNE'] = '1'
    spec = importlib.util.spec_from_file_location('ads_blackwell_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, digest


def ads_blackwell_configs(module):
    """Bounded search using the original kernel's explicit launch interface."""
    kernel = module.matmul_kernel_tma_ws_blackwell
    kernel = getattr(kernel, 'fn', kernel)
    accepted = set(kernel.arg_names)
    base = dict(module.get_reduced_autotune_config()[0].kwargs)
    # These controls select schedules only; the supplied JIT bodies stay intact.
    choices = ((64, 64, 64, 3, 2, 1, 1, 1),
               (128, 128, 64, 4, 2, 1, 2, 1),
               (128, 256, 64, 3, 2, 1, 4, 1),
               (256, 256, 64, 3, 1, 2, 8, 1),
               (256, 128, 128, 2, 2, 2, 4, 1),
               (256, 128, 128, 2, 2, 2, 4, 2))
    configs = []
    for bm, bn, bk, smem, tmem, mma, epilogue, ctas in choices:
        config = {k: v for k, v in base.items() if k in accepted}
        config.update(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                      GROUP_SIZE_M=ctas, NUM_SMEM_BUFFERS=smem,
                      NUM_TMEM_BUFFERS=tmem, NUM_MMA_GROUPS=mma,
                      EPILOGUE_SUBTILE=epilogue, NUM_CTAS=ctas, SPLIT_K=1,
                      INTERLEAVE_EPILOGUE=0)
        configs.append({k: v for k, v in config.items() if k in accepted})
    # A finer reduction is expensive for short GEMMs: eight subtiles keeps
    # the original math and pipeline while reducing shared-load/reduce work.
    if 'NUM_X2_SUBTILES' in accepted:
        short_reduction = dict(configs[0], NUM_X2_SUBTILES=8)
        configs.append(short_reduction)
    return configs


def prepare_ads_blackwell(module, x, w, out, config, eps=1e-6):
    """Preallocate descriptors/output while launching the unchanged ADS JIT."""
    from triton.tools.tensor_descriptor import TensorDescriptor
    import triton
    m, k = x.shape
    n = w.shape[1]
    descriptors = [TensorDescriptor(t, t.shape, t.stride(), [1, 1])
                   for t in (x, w, out, out)]
    a, b, c, workspace = descriptors
    module.matmul_tma_set_block_size_hook(
        dict(a_desc=a, b_desc=b, c_desc=c, workspace_desc=workspace,
             M=m, N=n, K=k, A_ROW_MAJOR=True, B_ROW_MAJOR=True, **config))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    ctas = config.get('NUM_CTAS', 1)
    tiles_m = triton.cdiv(triton.cdiv(m, config['BLOCK_SIZE_M']), ctas) * ctas
    grid = (min(sms, tiles_m * triton.cdiv(n, config['BLOCK_SIZE_N'])),)
    kernel = module.matmul_kernel_tma_ws_blackwell
    kernel = getattr(kernel, 'fn', kernel)

    def launch():
        return kernel[grid](a, b, c, workspace, m, n, k,
                            A_ROW_MAJOR=True, B_ROW_MAJOR=True,
                            NUM_SMS=sms, FUSED_RMSNORM=True,
                            RMSNORM_EPS=eps, ENABLE_PROTON=False,
                            num_warps=4, num_stages=1,
                            ctas_per_cga=(ctas, 1, 1) if ctas > 1 else None,
                            **config)
    return launch


def verify_b200_ads(args):
    """Independent paired comparison of ADS, Triton, and separate vendor arms."""
    import csv
    import triton
    import e1_rope_publication_v1 as rope
    from fusion_harness import paired as device_paired, metric as kernel_metric, compiler_evidence, capture
    from tlx_ads_sm90_port.ads_sm90_rmsnorm_matmul import (
        launch_rms_gemm, launch_rms_gemm_prepass, rmsnorm_rows_kernel,
        rstd_kernel, scale_gemm_rows_kernel)

    if torch.cuda.get_device_capability()[0] != 10:
        raise ValueError('The original ADS Blackwell comparison requires SM100')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    args.out.mkdir(parents=True, exist_ok=True)
    module, source_hash = (None, None) if args.baselines_only else load_ads_blackwell(
        args.ads_source, args.allow_source_version)
    shapes = B200_ADS_SHAPES
    if args.shapes:
        shapes = tuple(tuple(map(int, item.split(','))) for item in args.shapes.split(';'))
    if any(len(shape) != 3 or min(shape) < 1 for shape in shapes):
        raise ValueError('Shapes must be positive M,K,N triples')
    if args.smoke:
        shapes = shapes[:1]
    rec = {'schema': 'kernelfuser-b200-ads-rmsnorm-matmul-v1',
           'status': 'running', 'publication_eligible': False,
           'environment': {'gpu': torch.cuda.get_device_name(),
                           'capability': list(torch.cuda.get_device_capability()),
                           'torch': torch.__version__, 'triton': triton.__version__,
                           'cuda': torch.version.cuda},
           'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'ads_source_sha256': source_hash,
           'ads_matches_pinned_public': source_hash == ADS_PUBLIC_SHA256,
           'protocol': {'epsilon': 1e-6, 'dtype': 'bfloat16', 'gamma': False,
                        'output_policy': 'all timed arms preallocate output and scratch',
                        'correctness': 'three seeds; FP32 norm-before-GEMM oracle; '
                                       'atol=.15, rtol=.02, relative-L2<=.025',
                        'selection': 'bounded schedule search before independent timings',
                        'baseline_norm': 'separate FP32 RMSNorm, FP32 normalized activation '
                                         'and cuBLAS, BF16 output cast; FP32 weight copy '
                                         'prepared outside timing',
                        'row_scale_control': 'separate inverse RMS, BF16 cuBLAS '
                                             'FP32 output, FP32 scaling then BF16 cast',
                        'claim_boundary': 'isolated gamma-free forward operator'},
           'rows': [], 'tail_checks': []}
    before = rope.gpu_snapshot()
    if not before['exclusive']:
        raise RuntimeError('Selected GPU is not exclusive')
    monitor = rope.ExclusiveMonitor().start()

    def save():
        (args.out / 'results.json').write_text(json.dumps(rec, indent=2, default=str) + '\n')

    def inputs(shape, seed):
        m, k, n = shape
        torch.manual_seed(seed)
        x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
        w = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
        if m >= 3:
            x[0].zero_()
            x[1].fill_(1e-3)
            x[2].fill_(1)
        return x, w

    def oracle(x, w):
        xf = x.float()
        return ((xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6))
                @ w.float()).bfloat16()

    def check(output, reference):
        info = kernel_metric(output, reference, atol=.15, rtol=.02)
        info['finite'] = bool(torch.isfinite(output).all())
        info['pass'] &= info['finite'] and info['relative_l2'] <= .025
        return info

    def make_case(shape):
        m, k, n = shape
        x, w = inputs(shape, 20261013)
        ref = oracle(x, w)
        normalized = torch.empty_like(x, dtype=torch.float32)
        normalized_bf16 = torch.empty_like(x)
        w_float = w.float()
        rstd = torch.empty(m, device='cuda', dtype=torch.float32)
        raw = torch.empty((m, n), device='cuda', dtype=torch.float32)
        outputs = {name: torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
                   for name in ('separate_norm_cublas', 'row_scale_cublas', 'improved_triton')}
        if module is not None:
            outputs['original_ads_tlx'] = torch.empty_like(outputs['improved_triton'])
        arms, selected, tuning = {}, {}, {}

        def select(name, candidates):
            results, callables = [], {}
            for index, (config, fn) in enumerate(candidates):
                try:
                    fn()
                    torch.cuda.synchronize()
                    correctness = check(outputs[name], ref)
                    if not correctness['pass']:
                        results.append({'config': config, 'correctness': correctness})
                        continue
                    measurement = device_paired({name: fn}, rounds=2, samples=3,
                                                replays=10, seed=1013 + index)
                    us = measurement['arms'][name]['median_us']
                    results.append({'config': config, 'correctness': correctness,
                                    'timing': measurement, 'median_us': us})
                    callables[index] = fn
                    results[-1]['index'] = index
                except triton.OutOfResources as error:
                    results.append({'config': config, 'error': str(error)})
            valid = [r for r in results if 'median_us' in r]
            if not valid:
                raise RuntimeError(f'No correct schedule for {name}: {results}')
            best = min(valid, key=lambda r: r['median_us'])
            arms[name] = callables[best['index']]
            selected[name], tuning[name] = best['config'], results

        norm_candidates = []
        for rows, warps in ((1, 4), (4, 4), (8, 4)):
            def norm_cublas(rows=rows, warps=warps):
                rmsnorm_rows_kernel[(triton.cdiv(m, rows),)](
                    x, normalized, m, k, 1e-6, triton.next_power_of_2(k),
                    ROWS=rows, num_warps=warps)
                torch.mm(normalized, w_float, out=raw)
                return outputs['separate_norm_cublas'].copy_(raw)
            norm_candidates.append(({'rows': rows, 'warps': warps}, norm_cublas))
        select('separate_norm_cublas', norm_candidates)

        def row_scale_cublas():
            rstd_kernel[(m,)](x, rstd, k, 1e-6, triton.next_power_of_2(k), num_warps=4)
            torch.mm(x, w, out=raw, out_dtype=torch.float32)
            return scale_gemm_rows_kernel[(triton.cdiv(m * n, 1024),)](
                raw, rstd, outputs['row_scale_cublas'], m, n, BLOCK=1024)
        arms['row_scale_cublas'] = row_scale_cublas
        selected['row_scale_cublas'] = {'implementation': 'rstd + cuBLAS + scale/cast'}
        triton_candidates = []
        for bm, bn, bk, warps, stages in (
                (32, 64, 64, 4, 3), (64, 64, 64, 4, 3),
                (64, 128, 64, 4, 3), (128, 128, 64, 8, 3),
                (64, 128, 128, 4, 3), (128, 256, 64, 8, 3)):
            for prepass in (False, True):
                def fused(bm=bm, bn=bn, bk=bk, warps=warps, stages=stages, prepass=prepass):
                    if prepass:
                        return launch_rms_gemm_prepass(x, w, outputs['improved_triton'], rstd,
                                                       bm=bm, bn=bn, bk=bk, stages=stages)
                    return launch_rms_gemm(x, w, outputs['improved_triton'], bm=bm, bn=bn,
                                           bk=bk, warps=warps, stages=stages)
                triton_candidates.append((dict(bm=bm, bn=bn, bk=bk, warps=4 if prepass else warps,
                                                stages=stages, prepass=prepass), fused))
        select('improved_triton', triton_candidates)
        if module is not None:
            ads_candidates = [(config, prepare_ads_blackwell(
                module, x, w, outputs['original_ads_tlx'], config))
                for config in ads_blackwell_configs(module)]
            select('original_ads_tlx', ads_candidates)
        checks = []
        for seed in (20261013, 20261014, 20261015):
            if seed != 20261013:
                fresh_x, fresh_w = inputs(shape, seed)
                x.copy_(fresh_x)
                w.copy_(fresh_w)
                w_float.copy_(fresh_w)
                del fresh_x, fresh_w
                ref = oracle(x, w)
            for name, fn in arms.items():
                fn()
                c = check(outputs[name], ref)
                checks.append({'seed': seed, 'arm': name, **c})
                if not c['pass']:
                    raise RuntimeError(f'Correctness gate failed: {shape}, {seed}, {name}, {c}')
            # The conventional BF16 activation boundary is a separate numerical
            # contract. Keep its FP32 gate result without promoting failed timings.
            rmsnorm_rows_kernel[(m,)](x, normalized_bf16, m, k, 1e-6,
                                     triton.next_power_of_2(k), num_warps=4)
            bf16_normalized_output = torch.mm(normalized_bf16, w)
            checks.append({'seed': seed, 'arm': 'bf16_norm_boundary_diagnostic',
                           **check(bf16_normalized_output, ref)})
        # Cross-check the preallocated adapter against the original public wrapper.
        if module is not None:
            public = module.tlx_rmsnorm_matmul(
                x, w, config=dict(selected['original_ads_tlx']), fused_rmsnorm=True,
                rmsnorm_eps=1e-6)
            if not torch.equal(public, outputs['original_ads_tlx']):
                raise RuntimeError('Preallocated ADS launcher disagrees with original wrapper')
        tuning['_compiler'] = {}
        for name in ('improved_triton', 'original_ads_tlx'):
            if name in arms:
                compiled = arms[name]()
                evidence = compiler_evidence(compiled, args.out / 'compiler',
                                             '_'.join(map(str, shape)) + '_' + name)
                ptx = compiled.asm.get('ptx', '')
                evidence['tcgen05_mma_lines'] = sum('tcgen05.mma' in line for line in ptx.splitlines())
                evidence['tma_lines'] = sum('cp.async.bulk.tensor' in line for line in ptx.splitlines())
                tuning['_compiler'][name] = evidence
        return arms, selected, tuning, checks

    try:
        for shape in shapes:
            print('RUN B200', shape, flush=True)
            arms, selected, tuning, checks = make_case(shape)
            initial = device_paired(arms, rounds=5, samples=7, replays=20, seed=20261013)
            retest = device_paired(arms, rounds=5, samples=7, replays=20, seed=20261023)
            row = {'shape': list(shape), 'configs': selected, 'tuning': tuning,
                   'correctness': checks, 'initial': initial, 'retest': retest,
                   'status': 'passed', 'traces': [], 'trace_kernel_events': {}}
            for name, fn in arms.items():
                path = args.out / 'traces' / ('_'.join(map(str, shape)) + '_' + name + '.json.gz')
                graph = capture(fn, replays=1)
                row['traces'].append(rope.capture_trace(graph.replay, path, 'b200.' + name))
                with gzip.open(path, 'rt') as handle:
                    events = json.load(handle).get('traceEvents', [])
                kernels = [event.get('name', '') for event in events if event.get('cat') == 'kernel']
                if not kernels:
                    raise RuntimeError(f'Missing GPU kernel events in {name} trace')
                row['trace_kernel_events'][name] = kernels
                del graph
            rec['rows'].append(row)
            save()
            print('DONE B200', shape,
                  {name: round(info['median_us'], 3) for name, info in retest['arms'].items()},
                  flush=True)
            del arms
            gc.collect()
            torch.cuda.empty_cache()
        for shape in ((129, 1024, 3072), (513, 1032, 3080)):
            if args.smoke:
                break
            _, selected, _, checks = make_case(shape)
            rec['tail_checks'].append({'shape': list(shape), 'configs': selected,
                                       'correctness': checks, 'status': 'passed'})
            save()
        rec['status'] = 'baselines_passed' if args.baselines_only else 'passed'
    except Exception as error:
        rec['status'] = 'failed'
        rec['error'] = repr(error)
        raise
    finally:
        rec['gpu_before'] = before
        rec['gpu_observations'] = monitor.finish()
        rec['gpu_after'] = rope.gpu_snapshot()
        exclusive = all(r.get('exclusive', False) for r in
                        [before, *rec['gpu_observations'], rec['gpu_after']])
        rec['publication_eligible'] = rec['status'] == 'passed' and exclusive and not args.smoke
        if not exclusive:
            rec['status'] = 'invalid_gpu_exclusivity'
        save()
    if rec['publication_eligible']:
        public_rows = []
        for row in rec['rows']:
            initial, retest = row['initial']['arms'], row['retest']['arms']
            for name in ('original_ads_tlx', 'improved_triton'):
                baselines = ('separate_norm_cublas', 'row_scale_cublas')
                speedup = min(initial[b]['median_us'] for b in baselines) / initial[name]['median_us']
                retest_speedup = min(retest[b]['median_us'] for b in baselines) / retest[name]['median_us']
                candidate_name = ('ads_blackwell_snapshot' if name == 'original_ads_tlx'
                                  and not rec['ads_matches_pinned_public'] else name)
                public_rows.append(dict(device='B200', M=row['shape'][0], K=row['shape'][1],
                    N=row['shape'][2], dtype='bfloat16', eps=1e-6, candidate=candidate_name,
                    candidate_us=retest[name]['median_us'],
                    separate_norm_cublas_us=retest['separate_norm_cublas']['median_us'],
                    row_scale_cublas_us=retest['row_scale_cublas']['median_us'],
                    speedup_baseline='fastest of FP32 norm-before-cuBLAS and row-scale cuBLAS',
                    initial_speedup=speedup, retest_speedup=retest_speedup,
                    separate_norm_initial_speedup=initial['separate_norm_cublas']['median_us'] / initial[name]['median_us'],
                    separate_norm_retest_speedup=retest['separate_norm_cublas']['median_us'] / retest[name]['median_us'],
                    candidate_launches=1 if name == 'original_ads_tlx' else 1 + int(row['configs'][name]['prepass']),
                    separate_norm_launches=3, row_scale_launches=3,
                    confirmed_1pct=speedup > 1.01 and retest_speedup > 1.01,
                    max_relative_l2=max(c['relative_l2'] for c in row['correctness'] if c['arm']==name),
                    candidate_config=json.dumps(row['configs'][name], sort_keys=True),
                    ads_source_sha256=source_hash,
                    ads_matches_pinned_public=rec['ads_matches_pinned_public']))
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(public_rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(public_rows)


if __name__ == '__main__':
    if '--b200-ads' in sys.argv:
        parser = argparse.ArgumentParser(description='Original ADS Blackwell RMSNorm/GEMM comparison')
        parser.add_argument('--b200-ads', action='store_true')
        parser.add_argument('--ads-source', type=Path)
        parser.add_argument('--allow-source-version', action='store_true')
        parser.add_argument('--baselines-only', action='store_true')
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--csv', type=Path, default=Path('results/b200_ads_rmsnorm_matmul.csv'))
        parser.add_argument('--shapes', default='')
        parser.add_argument('--smoke', action='store_true')
        args = parser.parse_args()
        if not args.baselines_only and args.ads_source is None:
            parser.error('--ads-source is required for the original ADS comparison')
        verify_b200_ads(args)
    elif '--h100-unfused' in sys.argv:
        parser=argparse.ArgumentParser(description='H100 SwiGLU/RMSNorm-GEMM unfused controls')
        parser.add_argument('--h100-unfused',action='store_true')
        parser.add_argument('--out',type=Path,required=True)
        verify_h100_unfused(parser.parse_args())
    else:
        main()
