#!/usr/bin/env python3
"""SM90 fallback adapted from ADS TLX RMSNorm+matmul.

The upstream SM100 kernel relies on TMEM, CLC/cluster launch control, TMA
descriptors, and warp-specialized async_dot.  This fallback keeps the same
RMSNorm-before-GEMM math and a TLX async-task region, but uses register/SMEM
loads and ordinary tl.dot so it can compile on Hopper.  It is an ADS-derived
SM90 port, not the upstream Blackwell kernel.

The original launch path remains available. The rmsnorm_gemm API below adds
a measured H100 row-scale epilogue and an optional TLX async-dot pipeline.
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


# H100 refinement APIs and development-only verification.

@triton.jit
def rms_gemm_kernel(X, W, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    EPS: tl.constexpr, TWO_PASS: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ss = tl.zeros((BM,), tl.float32)
    if TWO_PASS:
        for k0 in range(0, tl.cdiv(K, BK)):
            kk = k0 * BK + rk
            a = tl.load(X + rm[:, None] * K + kk[None, :],
                        (rm[:, None] < M) & (kk[None, :] < K), 0).to(tl.float32)
            ss += tl.sum(a * a, axis=1)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, tl.cdiv(K, BK)):
        kk = k0 * BK + rk
        a = tl.load(X + rm[:, None] * K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + kk[:, None] * N + rn[None, :],
                    (kk[:, None] < K) & (rn[None, :] < N), 0)
        if not TWO_PASS:
            af = a.to(tl.float32)
            ss += tl.sum(af * af, axis=1)
        # BF16 tensor-core inputs, FP32 accumulation; no implicit TF32 dot.
        acc = tl.dot(a, b, acc, input_precision="ieee")  # noqa: TR011 -- precision explicit via modern API
    inverse = tl.rsqrt(ss / K + EPS)
    # A row's scalar normalization commutes with the gamma-free linear map.
    out = acc * inverse[:, None]
    tl.store(O + rm[:, None] * N + rn[None, :], out.to(O.dtype.element_ty),
             (rm[:, None] < M) & (rn[None, :] < N))


def launch_rms_gemm(x, w, out, *, bm=32, bn=64, bk=64, warps=4, stages=3,
                    two_pass=False, eps=1e-6):
    m, k = x.shape
    n = w.shape[1]
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError("H100 RMSNorm/GEMM candidate requires BF16 inputs")
    if not (x.is_contiguous() and w.is_contiguous() and out.is_contiguous()):
        raise ValueError("H100 RMSNorm/GEMM candidate requires contiguous tensors")
    return rms_gemm_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, w, out, m, n, k, BM=bm, BN=bn, BK=bk, EPS=eps,
        TWO_PASS=two_pass, num_warps=warps, num_stages=stages)


@triton.jit
def rstd_kernel(X, R, K: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + k, k < K, 0).to(tl.float32)
    tl.store(R + row, tl.rsqrt(tl.sum(x * x, 0) / K + EPS))


@triton.jit
def rmsnorm_rows_kernel(X, Y, M: tl.constexpr, K: tl.constexpr,
                        EPS: tl.constexpr, BLOCK: tl.constexpr,
                        ROWS: tl.constexpr = 1):
    """Separate gamma-free RMSNorm with output precision selected by Y."""
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, BLOCK)
    mask = (row[:, None] < M) & (k[None, :] < K)
    x = tl.load(X + row[:, None] * K + k[None, :], mask, 0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, 1) / K + EPS)
    tl.store(Y + row[:, None] * K + k[None, :],
             (x * inverse[:, None]).to(Y.dtype.element_ty), mask)


@triton.jit
def scale_gemm_rows_kernel(X, R, Y, M: tl.constexpr, N: tl.constexpr,
                           BLOCK: tl.constexpr):
    """Apply FP32 row scales to a vendor GEMM, then round to BF16."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < M * N
    x = tl.load(X + i, mask, 0)
    r = tl.load(R + i // N, mask, 0)
    tl.store(Y + i, (x * r).to(Y.dtype.element_ty), mask)


@triton.jit
def rms_gemm_tlx_kernel(X, W, R, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        STAGES: tl.constexpr, BIAS=None,
                        HAS_BIAS: tl.constexpr=False, APPLY_RMS: tl.constexpr=True,
                        INLINE_RMS: tl.constexpr=False, EPS: tl.constexpr=1e-6):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    if INLINE_RMS:
        # Finish the scalar reduction before the GEMM pipeline to keep its
        # vector loads/reductions out of the WGMMA steady-state schedule.
        # Work still repeats across output-column CTAs, without global scratch.
        ss = tl.zeros((BM,), tl.float32)
        for k in tl.range(0, tl.cdiv(K, BK), num_stages=1):
            kk = k * BK + rk
            values = tl.load(X + rm[:, None] * K + kk[None, :],
                             (rm[:, None] < M) & (kk[None, :] < K), 0).to(tl.float32)
            ss += tl.sum(values * values, axis=1)
        inverse = tl.rsqrt(ss / K + EPS)
    ap = X + rm[:, None] * K + rk[None, :]
    bp = W + rk[:, None] * N + rn[None, :]
    ab = tlx.local_alloc((BM, BK), tlx.dtype_of(X), STAGES)
    bb = tlx.local_alloc((BK, BN), tlx.dtype_of(W), STAGES)
    for i in tl.range(0, STAGES - 1, loop_unroll_factor=STAGES - 1):
        ta = tlx.async_load(ap, tlx.local_view(ab, i),
                            mask=(rm[:, None] < M) & (rk[None, :] < K - i * BK), other=0)
        tb = tlx.async_load(bp, tlx.local_view(bb, i),
                            mask=(rn[None, :] < N) & (rk[:, None] < K - i * BK), other=0)
        tlx.async_load_commit_group([ta, tb])
        ap += BK
        bp += BK * N
    acc = tl.zeros((BM, BN), tl.float32)
    for k in tl.range(0, tl.cdiv(K, BK), num_stages=1):
        tlx.async_load_wait_group(STAGES - 2)
        acc = tlx.async_dot(tlx.local_view(ab, k % STAGES), tlx.local_view(bb, k % STAGES), acc)
        i = k + STAGES - 1
        acc = tlx.async_dot_wait(1, acc)
        ta = tlx.async_load(ap, tlx.local_view(ab, i % STAGES),
                            mask=(rm[:, None] < M) & (rk[None, :] < K - i * BK), other=0)
        tb = tlx.async_load(bp, tlx.local_view(bb, i % STAGES),
                            mask=(rn[None, :] < N) & (rk[:, None] < K - i * BK), other=0)
        tlx.async_load_commit_group([ta, tb])
        ap += BK
        bp += BK * N
    acc = tlx.async_dot_wait(0, acc)
    if APPLY_RMS:
        if INLINE_RMS:
            r = inverse
        else:
            r = tl.load(R + rm, rm < M, 0)
        acc = acc * r[:, None]
    if HAS_BIAS:
        bias = tl.load(BIAS + rn, rn < N, 0).to(tl.float32)
        acc = acc + bias[None, :]
    tl.store(O + rm[:, None] * N + rn[None, :], acc.to(O.dtype.element_ty),
             (rm[:, None] < M) & (rn[None, :] < N))


def launch_rms_gemm_tlx(x, w, out, rstd, *, bm=64, bn=64, bk=64, stages=3, eps=1e-6):
    m, k = x.shape
    n = w.shape[1]
    rstd_kernel[(m,)](x, rstd, k, eps, triton.next_power_of_2(k), num_warps=4)
    return rms_gemm_tlx_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, w, rstd, out, m, n, k, BM=bm, BN=bn, BK=bk, STAGES=stages,
        num_warps=4, num_stages=1)


@triton.jit
def rms_gemm_prepass_kernel(X, W, R, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        kk = k * BK + rk
        a = tl.load(X + rm[:, None] * K + kk[None, :],
                    (rm[:, None] < M) & (kk[None, :] < K), 0)
        b = tl.load(W + kk[:, None] * N + rn[None, :],
                    (rn[None, :] < N) & (kk[:, None] < K), 0)
        acc = tl.dot(a, b, acc, input_precision="ieee")  # noqa: TR011 -- precision explicit via modern API
    r = tl.load(R + rm, rm < M, 0)
    tl.store(O + rm[:, None] * N + rn[None, :], (acc * r[:, None]).to(O.dtype.element_ty),
             (rm[:, None] < M) & (rn[None, :] < N))


def launch_rms_gemm_prepass(x, w, out, rstd, *, bm=64, bn=64, bk=64, stages=3, eps=1e-6):
    m, k = x.shape
    n = w.shape[1]
    rstd_kernel[(m,)](x, rstd, k, eps, triton.next_power_of_2(k), num_warps=4)
    return rms_gemm_prepass_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, w, rstd, out, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=4, num_stages=stages)


def rmsnorm_gemm(x, w, eps=1e-6, *, backend="tlx"):
    """Gamma-free BF16 RMSNorm/GEMM forward; caller can select plain Triton.

    Normalization scaling is applied to FP32 GEMM accumulators. This is
    mathematically equivalent to FP32 normalization before GEMM, with a
    different floating-point evaluation order. Not an autograd API.
    """
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise ValueError("Expected compatible matrices [M,K] and [K,N]")
    if x.requires_grad or w.requires_grad:
        raise ValueError("This API is forward-only")
    if not (x.is_cuda and w.is_cuda and x.device == w.device):
        raise ValueError("Inputs must be on the same CUDA device")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise ValueError("Inputs must be BF16")
    if not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError("Inputs must be contiguous")
    if torch.cuda.get_device_capability(x.device)[0] != 9:
        raise ValueError("This candidate was developed for SM90/Hopper")
    if min(*x.shape, w.shape[1]) < 1 or eps <= 0:
        raise ValueError("Nonempty inputs and positive eps required")
    if backend not in ("tlx", "triton"):
        raise ValueError("backend must be tlx or triton")
    out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)
    # cp.async cannot vectorize BF16 rows with odd strides. The masked
    # synchronous path supports these tails without changing the inputs.
    with torch.cuda.device(x.device):
        if x.shape[1] % 2 or w.shape[1] % 2:
            launch_rms_gemm(x, w, out, bm=16, bn=64, eps=eps)
        else:
            rstd = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
            fn = launch_rms_gemm_tlx if backend == "tlx" else launch_rms_gemm_prepass
            fn(x, w, out, rstd, eps=eps)
    return out


def lazy_rms_projection(x, folded_weight, bias, out, inverse=None, *, eps=1e-6,
                        bm=64, bn=128, bk=64, stages=3,
                        warps=4, standalone=False, inline_rms=False):
    """BF16 projection with gamma folded once into the supplied weight.

    Compute ``inverse_rms(x) * (x @ folded_weight) + bias`` with FP32
    accumulation and one final BF16 conversion. Only one FP32 scalar per
    input row is materialized in the reduction-split variant. ``inline_rms``
    computes the inverse RMS in the GEMM kernel with no scalar prepass or
    scratch. ``standalone`` measures the identical GEMM
    schedule and bias epilogue with normalization disabled for the anchor
    gate. This changes the intermediate BF16 rounding of eager RMSNorm.
    """
    if x.dtype != torch.bfloat16 or folded_weight.dtype != torch.bfloat16:
        raise ValueError("BF16 projection inputs required")
    m, k = x.shape
    n = folded_weight.shape[1]
    if folded_weight.shape[0] != k or out.shape != (m, n) or bias.shape != (n,):
        raise ValueError("Projection shapes are incompatible")
    if out.dtype != x.dtype:
        raise ValueError("BF16 output required")
    if not standalone and not inline_rms and (inverse is None or inverse.shape != (m,) or inverse.dtype != torch.float32):
        raise ValueError("Reduction-split projection requires one FP32 inverse-RMS scalar per row")
    if min(m, n, k) < 1 or eps <= 0 or stages < 2:
        raise ValueError("Positive dimensions/epsilon and at least two pipeline stages required")
    if any(t.requires_grad for t in (x, folded_weight, bias)):
        raise ValueError("This API is forward-only")
    if not all(t.is_cuda and t.device == x.device and t.is_contiguous()
               for t in (x, folded_weight, bias, out) + (() if inverse is None else (inverse,))):
        raise ValueError("Contiguous tensors on the same CUDA device required")
    if torch.cuda.get_device_capability(x.device)[0] != 9:
        raise ValueError("This TLX pipeline targets Hopper")
    if k % 2 or n % 2:
        raise ValueError("The TLX async-load pipeline requires even strides")
    if not standalone and not inline_rms:
        rstd_kernel[(m,)](x, inverse, k, eps, triton.next_power_of_2(k), num_warps=4)
    return rms_gemm_tlx_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        x, folded_weight, inverse, out, m, n, k, BM=bm, BN=bn, BK=bk,
        STAGES=stages, BIAS=bias, HAS_BIAS=True, APPLY_RMS=not standalone,
        INLINE_RMS=inline_rms and not standalone, EPS=eps,
        num_warps=warps, num_stages=1)


if __name__ == '__main__': main()
