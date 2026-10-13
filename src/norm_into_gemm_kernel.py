#!/usr/bin/env python3
"""RMSNorm -> GEMM fused kernel used by the E3a receipt.

The previous prototype implemented LayerNorm (a mean/correction identity) and
used a dot against a ones matrix for statistics.  That was semantically wrong
for the MoE census (which is RMSNorm) and is rejected by the Hopper LLVM
backend for some tilings.  This implementation accumulates an fp32 sum of
squares from the same input tile that feeds the GEMM.  It uses one pipeline
stage and is checked against an fp32 RMSNorm oracle before timing.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_gemm_kernel(a_ptr, w_ptr, d_ptr, g_ptr, r_ptr, out_ptr,
                     M, N, K, sam, sak, swn, swk, som, son,
                     EPS: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     STAGED: tl.constexpr):
    pid = tl.program_id(0)
    n_m = tl.cdiv(M, BLOCK_M)
    n_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // n_n
    pid_n = pid % n_n
    om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mm, mn = om < M, on < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if STAGED:
        rstd_stage=tl.load(r_ptr+om,mask=mm,other=0.)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        ok = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = ok < K
        a = tl.load(a_ptr + om[:, None] * sam + ok[None, :] * sak,
                    mask=mm[:, None] & mk[None, :], other=0.0)
        w = tl.load(w_ptr + on[:, None] * swn + ok[None, :] * swk,
                    mask=mn[:, None] & mk[None, :], other=0.0)
        gamma=tl.load(g_ptr+ok,mask=mk,other=0.).to(tl.float32)
        if STAGED:
            a_in=(a.to(tl.float32)*rstd_stage[:,None]*gamma[None,:]).to(a.dtype)
        else:
            ss += tl.sum(a.to(tl.float32) * a.to(tl.float32), axis=1)
            a_in=(a.to(tl.float32)*gamma[None,:]).to(a.dtype)
        acc = tl.dot(a_in, tl.trans(w), acc)
    d = tl.load(d_ptr + on, mask=mn, other=0.0).to(tl.float32)
    if STAGED:
        y=acc+d[None,:]
    else:
        rstd=tl.rsqrt(ss/K+EPS)
        y = acc * rstd[:, None] + d[None, :]
    tl.store(out_ptr + om[:, None] * som + on[None, :] * son,
             y.to(out_ptr.dtype.element_ty), mask=mm[:, None] & mn[None, :])


def norm_gemm(A, Wp, c, d, gamma=None, eps=1.0e-5,
              gamma_in_prologue=False, welford=True, rstd=None):
    """Compute ``rms_norm(A, gamma=1) @ Wp.T + d``.

    Gamma is multiplied into the loaded A tile in-register. No full-size
    folded weights or normalized activation tensor is materialized. rstd
    selects the staged statistics path; absent rstd computes stats in-loop.
    """
    M, K = A.shape
    N = Wp.shape[0]
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    bm = 64 if M < 128 else 128
    bn = 64 if N < 128 else 128
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    _rms_gemm_kernel[grid](A, Wp, d, gamma, rstd if rstd is not None else A, out, M, N, K,
                           A.stride(0), A.stride(1), Wp.stride(0), Wp.stride(1),
                           out.stride(0), out.stride(1), EPS=eps,
                           BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=64,
                           STAGED=rstd is not None,num_warps=4, num_stages=1)
    return out


def plain_gemm(A, Wp, c, d, gamma=None, eps=1.0e-5,
               gamma_in_prologue=False, welford=True):
    return torch.matmul(A, Wp.t()) + d


def build_folded(W, b, gamma, beta=None, fold_gamma=True):
    Wf, gf = W.float(), gamma.float()
    Wp = Wf * gf[None, :] if fold_gamma else Wf
    return Wp.to(W.dtype), torch.zeros((W.shape[0],), device=W.device), b.float()
