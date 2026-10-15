#!/usr/bin/env python3
"""Hopper TLX-vs-ordinary-Triton pointwise SwiGLU experiment.

This deliberately isolates a pointwise activation/epilogue. GEMM, routing and model dispatch remain out
of scope; the receipt therefore labels this as a kernel-level comparison.
The TLX path uses fbtriton's Hopper-compatible ``async_tasks`` API and does
not use Blackwell-only TMEM instructions.

launch_swiglu provides the measured single-task variant and explicit
arithmetic/rounding controls; fwd/bwd retain the original pilot paths.
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, time
from pathlib import Path
import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

BLOCK = 1024

@triton.jit
def _triton_fwd(gp, up, yp, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); off = pid * BLOCK + tl.arange(0, BLOCK); m = off < n
    g = tl.load(gp + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(up + off, mask=m, other=0.0).to(tl.float32)
    y = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
    tl.store(yp + off, y, mask=m)

@triton.jit
def _tlx_fwd(gp, up, yp, n, BLOCK: tl.constexpr):
    # async_tasks is available on Hopper in fbtriton.  This is a real TLX
    # kernel; it is intentionally free of Blackwell TMEM/cluster primitives.
    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid = tl.program_id(0); off = pid * BLOCK + tl.arange(0, BLOCK // 2); m = off < n
            g = tl.load(gp + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(up + off, mask=m, other=0.0).to(tl.float32)
            y = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
            tl.store(yp + off, y, mask=m)
        with tlx.async_task(num_warps=4):
            pid = tl.program_id(0); off = pid * BLOCK + BLOCK // 2 + tl.arange(0, BLOCK // 2); m = off < n
            g = tl.load(gp + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(up + off, mask=m, other=0.0).to(tl.float32)
            y = (g * tl.sigmoid(g) * u).to(tl.bfloat16)
            tl.store(yp + off, y, mask=m)

@triton.jit
def _triton_bwd(gp, up, dyp, dgp, dup, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); off = pid * BLOCK + tl.arange(0, BLOCK); m = off < n
    g = tl.load(gp + off, mask=m, other=0.0).to(tl.float32)
    u = tl.load(up + off, mask=m, other=0.0).to(tl.float32)
    dy = tl.load(dyp + off, mask=m, other=0.0).to(tl.float32)
    s = tl.sigmoid(g); silu = g * s
    tl.store(dgp + off, (dy * u * s * (1.0 + g * (1.0 - s))).to(tl.bfloat16), mask=m)
    tl.store(dup + off, (dy * silu).to(tl.bfloat16), mask=m)

@triton.jit
def _tlx_bwd(gp, up, dyp, dgp, dup, n, BLOCK: tl.constexpr):
    with tlx.async_tasks():
        with tlx.async_task("default"):
            pid = tl.program_id(0); off = pid * BLOCK + tl.arange(0, BLOCK // 2); m = off < n
            g = tl.load(gp + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(up + off, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(dyp + off, mask=m, other=0.0).to(tl.float32)
            s = tl.sigmoid(g); silu = g * s
            tl.store(dgp + off, (dy * u * s * (1.0 + g * (1.0 - s))).to(tl.bfloat16), mask=m)
            tl.store(dup + off, (dy * silu).to(tl.bfloat16), mask=m)
        with tlx.async_task(num_warps=4):
            pid = tl.program_id(0); off = pid * BLOCK + BLOCK // 2 + tl.arange(0, BLOCK // 2); m = off < n
            g = tl.load(gp + off, mask=m, other=0.0).to(tl.float32)
            u = tl.load(up + off, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(dyp + off, mask=m, other=0.0).to(tl.float32)
            s = tl.sigmoid(g); silu = g * s
            tl.store(dgp + off, (dy * u * s * (1.0 + g * (1.0 - s))).to(tl.bfloat16), mask=m)
            tl.store(dup + off, (dy * silu).to(tl.bfloat16), mask=m)

def fwd(kind, g, u):
    out = torch.empty_like(g); n = g.numel(); grid = (triton.cdiv(n, BLOCK),)
    (_tlx_fwd if kind == "tlx" else _triton_fwd)[grid](g, u, out, n, BLOCK=BLOCK, num_warps=4)
    return out

def bwd(kind, g, u, dy):
    dg = torch.empty_like(g); du = torch.empty_like(u); n = g.numel(); grid = (triton.cdiv(n, BLOCK),)
    (_tlx_bwd if kind == "tlx" else _triton_bwd)[grid](g, u, dy, dg, du, n, BLOCK=BLOCK, num_warps=4)
    return dg, du

def stats(xs):
    xs = sorted(float(x) for x in xs)
    return {"n": len(xs), "p50_ms": statistics.median(xs),
            "p90_ms": xs[max(0, int(.90*len(xs))-1)],
            "p99_ms": xs[max(0, int(.99*len(xs))-1)],
            "mean_ms": statistics.mean(xs), "std_ms": statistics.stdev(xs) if len(xs)>1 else 0.0,
            "min_ms": xs[0], "max_ms": xs[-1], "samples_ms": xs}

def timed(fn, warmup=20, repeats=100):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); out=[]
    for _ in range(repeats):
        a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); out.append(a.elapsed_time(b))
    return stats(out)

def trace(fn, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize(); p.step()
    p.export_chrome_trace(str(path))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--seed', type=int, default=20261006)
    args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.manual_seed(args.seed); device=torch.device('cuda')
    rec={"schema":"kernelfuser_tlx_triton_kernel_v1", "implementation_family":["triton","tlx"],
         "implementation_origin":{"triton":"ordinary_triton_control","tlx":"independent_sm90_hopper_port"},
         "target_architecture":"SM90 Hopper (not the ADS Model Kernel Library Blackwell-only kernels)",
         "operation":"pointwise_swiglu_activation", "level":"kernel", "direction":["forward","backward"],
         "device":{"name":torch.cuda.get_device_name(),"capability":list(torch.cuda.get_device_capability()),
                   "visible_device":os.environ.get('CUDA_VISIBLE_DEVICES','')},
         "versions":{"torch":torch.__version__,"triton":triton.__version__,"tlx_import":"triton.language.extra.tlx"},
         "seed":args.seed,"warmup":20,"repeats":100,"cases":[],"status":"started",
         "claim_boundary":"kernel-level pointwise epilogue; no model-level or GEMM/routing claim"}
    for name,m,n in [("shape_2048x3072",2048,3072),("shape_2048x14336",2048,14336)]:
        g=torch.randn((m,n),device=device,dtype=torch.bfloat16); u=torch.randn_like(g); dy=torch.randn_like(g)
        case={"workload":{"shape_id":name,"tokens":m,"intermediate":n,"dtype":"bfloat16"},"implementations":{}}
        ref_y=(g.float()*torch.sigmoid(g.float())*u.float()).to(torch.bfloat16)
        ref_s=torch.sigmoid(g.float()); ref_dg=(dy.float()*u.float()*ref_s*(1+g.float()*(1-ref_s))).to(torch.bfloat16); ref_du=(dy.float()*g.float()*ref_s).to(torch.bfloat16)
        for kind in ("triton","tlx"):
            y=fwd(kind,g,u); dg,du=bwd(kind,g,u,dy); torch.cuda.synchronize()
            case["implementations"][kind]={"forward":{"max_abs":float((y.float()-ref_y.float()).abs().max()),"allclose":bool(torch.allclose(y,ref_y,atol=.08,rtol=.02))},"backward":{"grad_gate_max_abs":float((dg.float()-ref_dg.float()).abs().max()),"grad_up_max_abs":float((du.float()-ref_du.float()).abs().max()),"allclose":bool(torch.allclose(dg,ref_dg,atol=.08,rtol=.02) and torch.allclose(du,ref_du,atol=.08,rtol=.02))}}
            case["implementations"][kind]["timing"]={"forward":timed(lambda:fwd(kind,g,u)),"backward":timed(lambda:bwd(kind,g,u,dy))}
            trace(lambda:(fwd(kind,g,u),bwd(kind,g,u,dy)), args.out/f"{name}_{kind}.chrome_trace.json")
        case["comparison"]={"forward_speedup_triton_over_tlx":case["implementations"]["tlx"]["timing"]["forward"]["p50_ms"] and case["implementations"]["triton"]["timing"]["forward"]["p50_ms"]/case["implementations"]["tlx"]["timing"]["forward"]["p50_ms"],"backward_speedup_triton_over_tlx":case["implementations"]["triton"]["timing"]["backward"]["p50_ms"]/case["implementations"]["tlx"]["timing"]["backward"]["p50_ms"]}
        rec["cases"].append(case); del g,u,dy; torch.cuda.empty_cache()
    rec["status"]="tested_pass" if all(c["implementations"][k][d]["allclose"] for c in rec["cases"] for k in ("triton","tlx") for d in ("forward","backward")) else "tested_invalid"
    rec["source_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); (args.out/'receipt.json').write_text(json.dumps(rec,indent=2)+"\n")
    print(json.dumps({"status":rec["status"],"out":str(args.out)}))


# H100 refinement APIs and development-only verification.

@triton.jit
def _swiglu_tile(G, U, DY, Y, DG, DU, N, BASE,
                 BLOCK: tl.constexpr, BACKWARD: tl.constexpr,
                 TORCH_ROUNDING: tl.constexpr, FAST: tl.constexpr):
    i = BASE + tl.arange(0, BLOCK)
    mask = i < N
    g = tl.load(G + i, mask, 0).to(tl.float32)
    u = tl.load(U + i, mask, 0).to(tl.float32)
    if FAST:
        sigmoid = 1.0 / (1.0 + tl.exp2(-g * 1.4426950408889634))
    else:
        sigmoid = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
    silu = g * sigmoid
    if TORCH_ROUNDING:
        silu = silu.to(tl.bfloat16).to(tl.float32)
    if BACKWARD:
        dy = tl.load(DY + i, mask, 0).to(tl.float32)
        dyu = dy * u
        if TORCH_ROUNDING:
            dyu = dyu.to(tl.bfloat16).to(tl.float32)
        dg = (dyu * sigmoid) * (1.0 + g * (1.0 - sigmoid))
        du = dy * silu
        tl.store(DG + i, dg.to(tl.bfloat16), mask)
        tl.store(DU + i, du.to(tl.bfloat16), mask)
    else:
        tl.store(Y + i, (silu * u).to(tl.bfloat16), mask)


@triton.jit
def swiglu_triton_kernel(G, U, DY, Y, DG, DU, N,
                         BLOCK: tl.constexpr, BACKWARD: tl.constexpr,
                         TORCH_ROUNDING: tl.constexpr, FAST: tl.constexpr):
    _swiglu_tile(G, U, DY, Y, DG, DU, N, tl.program_id(0) * BLOCK,
                 BLOCK, BACKWARD, TORCH_ROUNDING, FAST)


@triton.jit
def swiglu_tlx_kernel(G, U, DY, Y, DG, DU, N,
                      BLOCK: tl.constexpr, BACKWARD: tl.constexpr,
                      TORCH_ROUNDING: tl.constexpr, FAST: tl.constexpr,
                      SPLIT: tl.constexpr):
    # All accesses are global and disjoint; there is no cross-CTA storage.
    with tlx.async_tasks(exclusive=True, no_ending_cluster_sync=True):
        with tlx.async_task("default"):
            if SPLIT:
                _swiglu_tile(G, U, DY, Y, DG, DU, N, tl.program_id(0) * BLOCK,
                             BLOCK // 2, BACKWARD, TORCH_ROUNDING, FAST)
            else:
                _swiglu_tile(G, U, DY, Y, DG, DU, N, tl.program_id(0) * BLOCK,
                             BLOCK, BACKWARD, TORCH_ROUNDING, FAST)
        if SPLIT:
            with tlx.async_task(num_warps=4):
                _swiglu_tile(G, U, DY, Y, DG, DU, N, tl.program_id(0) * BLOCK + BLOCK // 2,
                             BLOCK // 2, BACKWARD, TORCH_ROUNDING, FAST)


def launch_swiglu(g, u, dy, y, dg, du, *, backend="triton", block=1024,
                  warps=4, backward=False, torch_rounding=True, fast=False,
                  split=False):
    if backend not in ("triton", "tlx"):
        raise ValueError("backend must be triton or tlx")
    if block < 128 or block & (block - 1) or warps not in (4, 8):
        raise ValueError("Expected a power-of-two block >=128 and 4 or 8 warps")
    fn = swiglu_triton_kernel if backend == "triton" else swiglu_tlx_kernel
    kwargs = {"SPLIT": split} if backend == "tlx" else {}
    return fn[(triton.cdiv(g.numel(), block),)](
        g, u, dy, y, dg, du, g.numel(), BLOCK=block, BACKWARD=backward,
        TORCH_ROUNDING=torch_rounding, FAST=fast, num_warps=warps,
        enable_fp_fusion=not torch_rounding, **kwargs)


if __name__ == '__main__': main()
