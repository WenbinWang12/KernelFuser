#!/usr/bin/env python3
"""Fusions at the language-model head (written 2026-11-21).

Site.  The last GEMM of a causal LM maps M token states of width K to M x V
logits (V = vocabulary).  The loss then makes several passes over that M x V
tensor (cast, log-softmax, gather, and in training the gradient and its casts),
so here the memory-bound work is as large as the GEMM output itself.

Two fused kernels:

1. ce_loss_grad (memory-bound chain, forward and backward in one kernel).
   Input: BF16 logits from the vendor GEMM and the targets.  One kernel computes
   the per-row loss and writes the gradient with respect to the logits,
   (softmax(z) - onehot(y)) / n, in BF16.  The backward pass of the loss then
   launches no kernel: FusedCE.backward returns the saved gradient, and autograd
   continues into the vendor GEMMs of the linear layer.

2. lmhead_loss (reduction epilogue inside the GEMM, forward only).
   A Triton GEMM whose epilogue reduces each output tile to a per-row partial
   log-sum-exp and picks the target logit, so the M x V logits are never written.
   A row spans many thread blocks; each block writes one (max, sum) pair per row
   and a small second kernel combines them.

BACKEND = "torch" replaces both kernels with PyTorch code of the same contract;
it exists to test the surrounding code on a machine without a GPU.
"""
from __future__ import annotations

import statistics

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

IGNORE = -100
BACKEND = "triton"

# (rows per program, columns per step); 8 warps
CE_CONFIGS = [(2, 2048), (4, 1024), (2, 4096), (8, 1024)]
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages)
GEMM_CONFIGS = [(128, 256, 64, 8, 3), (256, 128, 64, 8, 3), (128, 128, 64, 8, 4), (128, 256, 64, 8, 4),
                (64, 256, 64, 8, 3), (128, 128, 64, 8, 3), (256, 256, 64, 8, 3)]
_BEST: dict = {}
TUNE_LOG: dict = {}


# ----------------------------------------------------------------------------- kernel 1
@triton.jit
def _ce_grad(Z, Y, G, L, M, inv_n, V: tl.constexpr, IGN: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    rmask = rows < M
    base = rows.to(tl.int64) * V
    y = tl.load(Y + rows, mask=rmask, other=IGN)
    valid = y != IGN
    mx = tl.full((R,), float("-inf"), tl.float32)
    sm = tl.zeros((R,), tl.float32)
    for c0 in range(0, V, BLOCK):                       # pass 1: online log-sum-exp of the row
        cols = c0 + tl.arange(0, BLOCK)
        cmask = cols < V
        z = tl.load(Z + base[:, None] + cols[None, :], mask=rmask[:, None] & cmask[None, :],
                    other=float("-inf")).to(tl.float32)
        nm = tl.maximum(mx, tl.max(z, axis=1))
        sm = sm * tl.exp(mx - nm) + tl.sum(tl.exp(z - nm[:, None]), axis=1)
        mx = nm
    lse = mx + tl.log(sm)
    yc = tl.where(valid, y, 0)
    zy = tl.load(Z + base + yc, mask=rmask, other=0.0).to(tl.float32)
    w = tl.where(valid, inv_n, 0.0)
    tl.store(L + rows, (lse - zy) * w, mask=rmask)
    for c0 in range(0, V, BLOCK):                       # pass 2: gradient with respect to the logits
        cols = c0 + tl.arange(0, BLOCK)
        cmask = cols < V
        z = tl.load(Z + base[:, None] + cols[None, :], mask=rmask[:, None] & cmask[None, :],
                    other=float("-inf")).to(tl.float32)
        p = tl.exp(z - lse[:, None])
        g = (p - tl.where(cols[None, :] == yc[:, None], 1.0, 0.0)) * w[:, None]
        tl.store(G + base[:, None] + cols[None, :], g.to(G.dtype.element_ty),
                 mask=rmask[:, None] & cmask[None, :])


def _ce_launch(z, y, g, rowloss, inv_n, cfg):
    M, V = z.shape
    r, block = cfg
    _ce_grad[(triton.cdiv(M, r),)](z, y, g, rowloss, M, inv_n, V=V, IGN=IGNORE, R=r, BLOCK=block, num_warps=8)


def _n_valid(y):
    return (y != IGNORE).sum().clamp_min(1)


def ce_loss_grad(z, y, inv_n=None):
    """Loss (scalar, FP32) and d loss / d logits (BF16, same shape as z) for mean cross-entropy."""
    if inv_n is None:
        inv_n = 1.0 / float(_n_valid(y))
    if BACKEND == "torch":
        zf = z.float()
        valid = y != IGNORE
        yc = torch.where(valid, y, torch.zeros_like(y))
        lse = torch.logsumexp(zf, dim=1)
        rowloss = (lse - zf.gather(1, yc[:, None])[:, 0]) * valid * inv_n
        g = torch.exp(zf - lse[:, None])
        g[torch.arange(z.shape[0], device=z.device), yc] -= 1.0
        g = (g * (valid * inv_n)[:, None]).to(z.dtype)
        return rowloss.sum(), g
    g = torch.empty_like(z)
    rowloss = torch.empty((z.shape[0],), device=z.device, dtype=torch.float32)
    _ce_launch(z, y, g, rowloss, inv_n, _BEST.get(("ce",) + tuple(z.shape), CE_CONFIGS[0]))
    return rowloss.sum(), g


class FusedCE(torch.autograd.Function):
    """Mean cross-entropy whose forward kernel also produces the gradient.

    The loss must be the root of the backward pass (incoming gradient 1), as in a
    training step; `scale_grad=True` multiplies by the incoming gradient instead.
    """
    scale_grad = False

    @staticmethod
    def forward(ctx, logits, target, inv_n):
        loss, g = ce_loss_grad(logits, target, inv_n)
        ctx.g = g
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        g = ctx.g * grad_out if FusedCE.scale_grad else ctx.g
        return g, None, None


def fused_cross_entropy(logits, target, inv_n=None):
    if inv_n is None:
        inv_n = 1.0 / float(_n_valid(target))
    return FusedCE.apply(logits, target, inv_n)


# ----------------------------------------------------------------------------- kernel 2
@triton.jit
def _swizzle(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_pid_n
    gid = pid // width
    first_m = gid * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % width) % gsize)
    pid_n = (pid % width) // gsize
    return pid_m, pid_n


@triton.jit
def _lmhead(a_ptr, w_ptr, y_ptr, out_ptr, pm_ptr, ps_ptr, zy_ptr, M, N, K, sam, sak, swn, swk,
            STORE: tl.constexpr, GROUP_M: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
            BLOCK_K: tl.constexpr):
    """z = A @ W^T.  STORE: write z (BF16).  Otherwise: per-row partial log-sum-exp of this tile and the target logit."""
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(a, tl.trans(w), acc)
    z16 = acc.to(tl.bfloat16)                            # the logits as the vendor path would store them
    if STORE:
        tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], z16,
                 mask=mask_m[:, None] & mask_n[None, :])
    else:
        z = tl.where(mask_n[None, :], z16.to(tl.float32), float("-inf"))
        pm = tl.max(z, axis=1)
        ps = tl.sum(tl.exp(z - pm[:, None]), axis=1)
        nt = tl.cdiv(N, BLOCK_N)
        tl.store(pm_ptr + offs_m * nt + pid_n, pm, mask=mask_m)
        tl.store(ps_ptr + offs_m * nt + pid_n, ps, mask=mask_m)
        y = tl.load(y_ptr + offs_m, mask=mask_m, other=-1)
        hit = offs_n[None, :] == y[:, None]
        zy = tl.sum(tl.where(hit, z, 0.0), axis=1)
        has = tl.sum(hit.to(tl.int32), axis=1) > 0
        tl.store(zy_ptr + offs_m, zy, mask=mask_m & has)


@triton.jit
def _lse_combine(PM, PS, ZY, Y, L, M, inv_n, NT: tl.constexpr, IGN: tl.constexpr, R: tl.constexpr,
                 BLOCK: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    rmask = rows < M
    # One loop with an online log-sum-exp (the form of _ce_grad).  A first loop that only takes the
    # row maximum, followed by a second loop that uses it, aborts the Triton 3.1.0 compiler
    # (assertion in OptimizeThreadLocality: the loop result has more than one use).
    mx = tl.full((R,), float("-inf"), tl.float32)
    sm = tl.zeros((R,), tl.float32)
    for c0 in range(0, NT, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        m2 = rmask[:, None] & (cols < NT)[None, :]
        pm = tl.load(PM + rows[:, None] * NT + cols[None, :], mask=m2, other=float("-inf"))
        ps = tl.load(PS + rows[:, None] * NT + cols[None, :], mask=m2, other=0.0)
        nm = tl.maximum(mx, tl.max(pm, axis=1))
        sm = sm * tl.exp(mx - nm) + tl.sum(ps * tl.exp(pm - nm[:, None]), axis=1)
        mx = nm
    lse = mx + tl.log(sm)
    y = tl.load(Y + rows, mask=rmask, other=IGN)
    valid = y != IGN
    zy = tl.load(ZY + rows, mask=rmask & valid, other=0.0)
    tl.store(L + rows, tl.where(valid, (lse - zy) * inv_n, 0.0), mask=rmask)


def _gemm_launch(h, W, y, cfg, store, inv_n=1.0):
    M, K = h.shape
    N = W.shape[0]
    bm, bn, bk, warps, stages = cfg
    grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
    dev = h.device
    if store:
        out = torch.empty((M, N), device=dev, dtype=h.dtype)
        dummy = torch.empty((1,), device=dev, dtype=torch.float32)
        _lmhead[grid](h, W, y, out, dummy, dummy, dummy, M, N, K, h.stride(0), h.stride(1), W.stride(0), W.stride(1),
                      STORE=True, GROUP_M=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)
        return out
    nt = triton.cdiv(N, bn)
    pm = torch.empty((M, nt), device=dev, dtype=torch.float32)
    ps = torch.empty((M, nt), device=dev, dtype=torch.float32)
    zy = torch.empty((M,), device=dev, dtype=torch.float32)
    rowloss = torch.empty((M,), device=dev, dtype=torch.float32)
    dummy = torch.empty((1,), device=dev, dtype=h.dtype)
    _lmhead[grid](h, W, y, dummy, pm, ps, zy, M, N, K, h.stride(0), h.stride(1), W.stride(0), W.stride(1),
                  STORE=False, GROUP_M=8, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps, num_stages=stages)
    _lse_combine[(triton.cdiv(M, 4),)](pm, ps, zy, y, rowloss, M, inv_n, NT=nt, IGN=IGNORE, R=4, BLOCK=256,
                                       num_warps=4)
    return rowloss.sum()


def lmhead_loss(h, W, y, inv_n=None):
    """Mean cross-entropy of (h @ W^T, y) without materializing the logits."""
    if inv_n is None:
        inv_n = 1.0 / float(_n_valid(y))
    if BACKEND == "torch":
        return F.cross_entropy(F.linear(h, W).float(), y, ignore_index=IGNORE)
    cfg = _BEST.get(("gemm", False, h.shape[0], W.shape[0], h.shape[1]), GEMM_CONFIGS[0])
    return _gemm_launch(h, W, y, cfg, False, inv_n)


def lmhead_logits(h, W, y):
    """The same Triton GEMM with a plain store epilogue (the anchor without fusion)."""
    if BACKEND == "torch":
        return F.linear(h, W)
    cfg = _BEST.get(("gemm", True, h.shape[0], W.shape[0], h.shape[1]), GEMM_CONFIGS[0])
    return _gemm_launch(h, W, y, cfg, True)


# ----------------------------------------------------------------------------- per-shape configuration choice
def _time(fn, reps=7):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        vals.append(float(a.elapsed_time(b)))
    return statistics.median(vals)


def tune_ce(z, y):
    """Pick the fastest (rows, columns) configuration for this logits shape. Call before any CUDA-graph capture."""
    if BACKEND == "torch":
        return None
    key = ("ce",) + tuple(z.shape)
    if key in _BEST:
        return _BEST[key]
    g = torch.empty_like(z)
    rl = torch.empty((z.shape[0],), device=z.device, dtype=torch.float32)
    res = {}
    for cfg in CE_CONFIGS:
        try:
            res[cfg] = _time(lambda: _ce_launch(z, y, g, rl, 1.0 / z.shape[0], cfg))
        except Exception as exc:                        # a configuration that does not compile is skipped
            res[cfg] = repr(exc)[:200]
    ok = {c: t for c, t in res.items() if isinstance(t, float)}
    if not ok:
        raise RuntimeError(f"no CE configuration ran: {res}")
    _BEST[key] = min(ok, key=ok.get)
    TUNE_LOG[str(key)] = {str(c): t for c, t in res.items()}
    return _BEST[key]


def tune_gemm(h, W, y, store):
    if BACKEND == "torch":
        return None
    key = ("gemm", bool(store), h.shape[0], W.shape[0], h.shape[1])
    if key in _BEST:
        return _BEST[key]
    res = {}
    for cfg in GEMM_CONFIGS:
        try:
            res[cfg] = _time(lambda: _gemm_launch(h, W, y, cfg, store, 1.0 / h.shape[0]), reps=5)
        except Exception as exc:
            res[cfg] = repr(exc)[:200]
    ok = {c: t for c, t in res.items() if isinstance(t, float)}
    if not ok:
        raise RuntimeError(f"no GEMM configuration ran: {res}")
    _BEST[key] = min(ok, key=ok.get)
    TUNE_LOG[str(key)] = {str(c): t for c, t in res.items()}
    return _BEST[key]
