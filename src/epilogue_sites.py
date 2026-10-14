#!/usr/bin/env python3
"""Two more fusion SITES, to complete the rule's positive/negative coverage.

The census so far probes three producer->consumer opportunities (norm->attn,
norm->GEMM, SwiGLU-epilogue). Two production sites are still missing, and they
land on OPPOSITE sides of the boundary for a reason the rule predicts:

  S4  gemm_residual   down_proj / o_proj GEMM, then the residual ADD.
      The fused thing is an EPILOGUE (like SwiGLU), and the producer -- an
      elementwise add of the residual stream -- is memory-bound and tiny next to
      the projection's MMA, so f is small. But absorbing a `+R` in the epilogue
      costs almost nothing: the accumulator is already in registers, one extra
      load and add, no reduction. So c ~ 0, the break-even c/(1+c) ~ 0, and even
      a small f clears it. PREDICTED POSITIVE. This is the epilogue half of the
      same prologue/epilogue asymmetry we see under TLX: fuse epilogues directly.

  S5  rms_lm_head      final RMSNorm -> lm_head GEMM.
      Same TRANSFORMATION as norm->GEMM (a prologue norm folded into the MMA),
      but the consumer's N is the VOCABULARY (10^5), so the norm's share f of the
      unfused baseline is diluted even harder than at an MLP projection. The
      prologue still pays c to host the reduction inside the K-loop. PREDICTED
      NEGATIVE, and more decisively so than P2 -- unless M is tiny (decode,
      M=1..8), where the GEMM turns memory-bound and f climbs back. We sweep M to
      show the boundary move, not just assert a sign.

Both reuse the SAME four-arm accounting as fusion_harness (plain / fused / host
/ reference), so f and c mean exactly what they mean everywhere else and the
identity self-check still holds to ~1e-16. RMSNorm here, not LayerNorm: every
architecture puts RMSNorm at the final-norm site, so mu=0 and the fold is
out = rstd * (A @ Wp^T) with Wp = gamma * W -- no mu*c term.

DEV-DIAGNOSTIC. publication_eligible=False, development_only=True.
"""
import os
import torch
import triton
import triton.language as tl

from fusion_harness import Pattern, DEV, DT

BW = 4.8e12          # H200 NVL HBM3e
PEAK_BF16 = 835e12
STATS_COLS = 16      # tl.dot needs N>=16; the row sum-of-squares is replicated


# ======================================================= shared GEMM tiling ===
def _configs():
    space = [
        (128, 256, 64, 8, 3), (256, 128, 64, 8, 3),
        (128, 256, 64, 8, 4), (128, 128, 64, 8, 4),
        (256, 256, 64, 8, 3), (64, 256, 64, 8, 3),
        (128, 128, 64, 8, 3), (64, 128, 64, 4, 3),
        (256, 128, 128, 8, 3), (128, 64, 64, 4, 4),
    ]
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk,
                           "GROUP_M": 8}, num_warps=w, num_stages=s)
            for bm, bn, bk, w, s in space]


@triton.jit
def _swizzle(pid, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             GROUP_M: tl.constexpr):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * num_pid_n
    gid = pid // width
    first_m = gid * GROUP_M
    gsize = min(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + ((pid % width) % gsize)
    pid_n = (pid % width) // gsize
    return pid_m, pid_n


# ============================================ S4: GEMM with residual epilogue ==
@triton.autotune(configs=_configs(), key=["M", "N", "K", "FUSE_RES"])
@triton.jit
def gemm_res_kernel(a_ptr, w_ptr, r_ptr, d_ptr, out_ptr,
                    M, N, K, sam, sak, swn, swk, som, son,
                    FUSE_RES: tl.constexpr, GROUP_M: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    """out = A@W^T (+ bias d) (+ R if FUSE_RES).

    FUSE_RES participates in the autotune key, so schedules can differ.
    Use launch_h100_residual for comparisons with an identical fixed schedule.
    """
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
    d = tl.load(d_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + d[None, :]
    if FUSE_RES:
        r = tl.load(r_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc = acc + r.to(tl.float32)
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def gemm_res(A, W, d, R, fuse_res):
    M, K = A.shape
    N = W.shape[0]
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    r = R if R is not None else out
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"])
                         * triton.cdiv(N, meta["BLOCK_N"]),)
    gemm_res_kernel[grid](A, W, r, d, out, M, N, K,
                          A.stride(0), A.stride(1), W.stride(0), W.stride(1),
                          out.stride(0), out.stride(1), FUSE_RES=fuse_res)
    return out


@triton.jit
def _add_kernel(Y, R, O, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    y = tl.load(Y + offs, mask=m, other=0.0)
    r = tl.load(R + offs, mask=m, other=0.0)
    tl.store(O + offs, y + r, mask=m)


def add_residual(Y, R):
    """The deployable default's SEPARATE add kernel (strong: coalesced, one pass
    read-read-write)."""
    o = torch.empty_like(Y)
    n = Y.numel()
    BLOCK = 4096
    _add_kernel[(triton.cdiv(n, BLOCK),)](Y.view(-1), R.view(-1), o.view(-1),
                                          n, BLOCK=BLOCK, num_warps=8)
    return o


# ================================================ S5: RMSNorm -> GEMM (lm_head) =
@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def rms_gemm_kernel(a_ptr, wp_ptr, out_ptr, M, N, K,
                    sam, sak, swn, swk, som, son, EPS: tl.constexpr,
                    STATS_COLS: tl.constexpr, GROUP_M: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    """out = rstd * (A @ Wp^T), rstd = 1/sqrt(mean(a^2)+eps), Wp = gamma*W.

    RMSNorm has mu=0, so only sum(a^2) is needed -- taken as (a*a) @ ones inside
    the K-loop (dot-ones, not tl.sum: the tl.sum-feeding-tl.dot path silently
    corrupts the accumulator on Triton 3.4.0, see norm_into_gemm_kernel.py). No
    mu*c term, no beta. Each CTA sees the full K, so its rstd closes with zero
    extra global traffic; CTAs in a row band recompute it (ALU only)."""
    pid = tl.program_id(0)
    pid_m, pid_n = _swizzle(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Hopper Triton rejects the previous dot-against-ones encoding on H200.
    # Accumulate the already-loaded tile with a native reduction instead; this
    # remains inside the GEMM K-loop and does not add a global-memory pass.
    p2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    cnt = 0.0
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        p2 += tl.sum((a * a).to(tl.float32), axis=1)
        cnt += tl.sum(mask_k.to(tl.float32))
        w = tl.load(wp_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(a, tl.trans(w), acc)
    nrow = tl.maximum(cnt, 1.0)
    ms = p2 / nrow
    rstd = 1.0 / tl.sqrt(ms + EPS)
    out = rstd[:, None] * acc
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             out.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(configs=_configs(), key=["M", "N", "K"])
@triton.jit
def plain_gemm_kernel(a_ptr, wp_ptr, out_ptr, M, N, K,
                      sam, sak, swn, swk, som, son, EPS: tl.constexpr,
                      STATS_COLS: tl.constexpr, GROUP_M: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    """Arm-G control: identical tiling/swizzle to rms_gemm_kernel, norm removed,
    so (fused - plain) attributes purely to the in-kernel rstd work."""
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
        w = tl.load(wp_ptr + offs_n[:, None] * swn + offs_k[None, :] * swk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc = tl.dot(a, tl.trans(w), acc)
    tl.store(out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
             acc.to(out_ptr.dtype.element_ty),
             mask=mask_m[:, None] & mask_n[None, :])


def _gemm(kernel, A, Wp, eps):
    M, K = A.shape
    N = Wp.shape[0]
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"])
                         * triton.cdiv(N, meta["BLOCK_N"]),)
    kernel[grid](A, Wp, out, M, N, K, A.stride(0), A.stride(1),
                 Wp.stride(0), Wp.stride(1), out.stride(0), out.stride(1),
                 EPS=eps, STATS_COLS=STATS_COLS)
    return out


@triton.jit
def _rms_std_kernel(X, O, n_rows, EPS_: tl.constexpr, K: tl.constexpr,
                    BLOCK_K: tl.constexpr, R: tl.constexpr):
    """RMS standardization a*rstd with NO gamma (gamma is folded into Wp). R rows
    per program to fill the memory pipe, matching the strong-baseline discipline
    used for the LayerNorm producer. K is looped in BLOCK_K chunks so a non-power-
    of-two hidden size (e.g. Gemma-3's 3840) compiles -- tl.arange requires a
    power-of-two length, so a single full-width arange would reject those."""
    pid = tl.program_id(0)
    rows = pid * R + tl.arange(0, R)
    rm = rows < n_rows
    ms = tl.zeros((R,), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        cm = cols < K
        x = tl.load(X + rows[:, None] * K + cols[None, :],
                    mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        ms += tl.sum(x * x, axis=1)
    rstd = 1.0 / tl.sqrt(ms / K + EPS_)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        cols = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        cm = cols < K
        x = tl.load(X + rows[:, None] * K + cols[None, :],
                    mask=rm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        y = x * rstd[:, None]
        tl.store(O + rows[:, None] * K + cols[None, :],
                 y.to(O.dtype.element_ty), mask=rm[:, None] & cm[None, :])


def rms_standardize(x, eps, R=8):
    o = torch.empty_like(x)
    M, K = x.shape
    BLOCK_K = 2048 if K % 2048 == 0 else 1024 if K % 1024 == 0 else 256
    _rms_std_kernel[(triton.cdiv(M, R),)](x, o, M, EPS_=eps, K=K,
                                          BLOCK_K=BLOCK_K, R=R, num_warps=8)
    return o


def build_wp(W, gamma):
    """Wp = gamma * W  (RMSNorm gamma folded into the lm_head weights, offline)."""
    return (W.float() * gamma.float()[None, :]).to(W.dtype)


# ================================================================ patterns ====
class GemmResidual(Pattern):
    """down_proj / o_proj GEMM with the residual add fused into the epilogue.

    Predicted POSITIVE: the add is cheap to host (c ~ 0), so its small share f
    still clears c/(1+c). Host arm returns the un-added projection by
    construction, so it is gate-exempt exactly like SwiGLU's host cell."""

    name = "gemm_residual"
    backends = ("simb", "vendor")
    geom_keys = ("site", "M", "K", "N")
    gate_exempt = ("host_simb",)

    def geometries(self):
        # (site, M, K, N). down_proj: K=intermediate -> N=hidden. o_proj:
        # K=hidden -> N=hidden. M sweeps prefill token counts.
        base = [
            ("down_proj", 11008, 4096), ("down_proj", 14336, 4096),
            ("o_proj", 4096, 4096), ("o_proj", 8192, 8192),
        ]
        Ms = [int(m) for m in os.environ.get("S4_M", "128,512,2048,8192").split(",")]
        return [(site, M, K, N) for (site, K, N) in base for M in Ms]

    def label(self, g):
        return f"{g[0]} M={g[1]} K={g[2]} N={g[3]}"

    def arms(self, g):
        _, M, K, N = g
        torch.manual_seed(0)
        A = torch.randn(M, K, device=DEV, dtype=DT) * (K ** -0.5)
        W = torch.randn(N, K, device=DEV, dtype=DT) * (K ** -0.5)
        d = torch.zeros(N, device=DEV, dtype=DT)
        R = torch.randn(M, N, device=DEV, dtype=DT)

        def ref():
            return (A.float() @ W.float().t() + R.float())

        return {
            "reference": ref,
            # plain: GEMM (no residual), then the separate add kernel.
            "plain_simb": lambda: add_residual(gemm_res(A, W, d, None, False), R),
            # fused: residual added in the GEMM epilogue.
            "fused_simb": lambda: gemm_res(A, W, d, R, True),
            # host: GEMM alone, residual absent -> defines c, gate-exempt.
            "host_simb": lambda: gemm_res(A, W, d, None, False),
            # vendor: cuBLAS GEMM then a torch add -- the deployable baseline.
            "plain_vendor": lambda: torch.nn.functional.linear(A, W) + R,
        }

    def roofline_ms(self, g):
        _, M, _, N = g
        # the residual add alone: read Y, read R, write Y
        return 3 * 2 * M * N / BW * 1e3


class RmsLmHead(Pattern):
    """Final RMSNorm folded into the lm_head GEMM prologue.

    Predicted NEGATIVE at prefill/training M (N=vocab dilutes f below break-even)
    and only recovering as M -> 1 (decode turns the GEMM memory-bound)."""

    name = "rms_lm_head"
    backends = ("simb", "vendor")
    geom_keys = ("model", "M", "K", "N")

    def __init__(self, eps=1e-6):
        self.eps = eps

    def geometries(self):
        # (model, K=hidden, N=vocab). Real (hidden, vocab) pairs from shipped
        # models; M sweeps decode (1) through prefill (many).
        base = [("llama3-8b", 4096, 128256), ("gemma3-12b", 3840, 262144),
                ("qwen3-8b", 4096, 151936)]
        Ms = [int(m) for m in os.environ.get("S5_M", "1,64,2048,8192").split(",")]
        return [(nm, M, K, N) for (nm, K, N) in base for M in Ms]

    def label(self, g):
        return f"{g[0]} M={g[1]} K={g[2]} N={g[3]}"

    def arms(self, g):
        _, M, K, N = g
        eps = self.eps
        torch.manual_seed(0)
        A = torch.randn(M, K, device=DEV, dtype=DT)
        W = torch.randn(N, K, device=DEV, dtype=DT) * (K ** -0.5)
        gamma = torch.randn(K, device=DEV, dtype=DT) * 0.1 + 1.0
        Wp = build_wp(W, gamma)
        An = rms_standardize(A, eps)   # precomputed producer output for host arm

        def ref():
            Af = A.float()
            rstd = torch.rsqrt((Af * Af).mean(dim=1, keepdim=True) + eps)
            return (Af * rstd) @ (W.float() * gamma.float()[None, :]).t()

        return {
            "reference": ref,
            # plain: strong tiled RMS standardize, then the SAME GEMM, no norm.
            "plain_simb": lambda: _gemm(plain_gemm_kernel,
                                        rms_standardize(A, eps), Wp, eps),
            # fused: rstd computed inside the GEMM's K-loop.
            "fused_simb": lambda: _gemm(rms_gemm_kernel, A, Wp, eps),
            # host: GEMM alone on pre-standardized input -> defines c.
            "host_simb": lambda: _gemm(plain_gemm_kernel, An, Wp, eps),
            # vendor: torch RMSNorm-equivalent standardize then cuBLAS.
            "plain_vendor": lambda: torch.nn.functional.linear(
                A * torch.rsqrt((A.float() * A.float()).mean(-1, keepdim=True)
                                + eps).to(A.dtype), Wp),
        }

    def roofline_ms(self, g):
        _, M, K, _ = g
        # the RMS standardize alone: read A, write A
        return 2 * 2 * M * K / BW * 1e3


PATTERNS = {"gemm_residual": GemmResidual, "rms_lm_head": RmsLmHead}


# Explicit H100 schedules; vendor addmm is the measured default.
H100_RESIDUAL_CONFIGS = {(128,28672,8192):(128,64,64,4,4),
                        (2048,25600,5120):(128,128,64,8,3),
                        (512,4096,4096):(128,128,64,8,4)}

def launch_h100_residual(a,w,r,bias,out,*,fused=True):
    """Preallocated direct Triton launch; callers own input validation.

    Fused and unfused launches use the identical per-shape schedule. The
    unfused launch writes only GEMM+bias; callers must include residual add.
    """
    m,k=a.shape;n=w.shape[0]
    bm,bn,bk,warps,stages=H100_RESIDUAL_CONFIGS[(m,k,n)]
    return gemm_res_kernel.fn[(triton.cdiv(m,bm)*triton.cdiv(n,bn),)](
        a,w,r,bias,out,m,n,k,k,1,k,1,n,1,FUSE_RES=fused,GROUP_M=8,
        BLOCK_M=bm,BLOCK_N=bn,BLOCK_K=bk,num_warps=warps,num_stages=stages)


def gemm_residual(a,w,r,*,backend='vendor'):
    """Forward BF16 ``A @ W.T + R`` on H100.

    Vendor addmm is the fastest measured default. ``backend='triton'`` uses
    an explicit schedule for the three measured shapes in CONFIGS; it is a
    useful fusion experiment, and currently slower than vendor addmm.
    """
    if backend not in ('vendor','triton'):
        raise ValueError('Expected vendor or triton backend')
    if a.ndim!=2 or w.ndim!=2 or r.ndim!=2 or a.shape[1]!=w.shape[1] or r.shape!=(a.shape[0],w.shape[0]):
        raise ValueError('Expected A [M,K], W [N,K], R [M,N]')
    if any(t.requires_grad for t in (a,w,r)):
        raise ValueError('This API is forward-only')
    if any(not t.is_cuda or t.device!=a.device or t.dtype!=torch.bfloat16 or not t.is_contiguous() for t in (a,w,r)):
        raise ValueError('Inputs must be contiguous BF16 on one CUDA device')
    if min(*a.shape,w.shape[0])<1:
        raise ValueError('Expected positive dimensions')
    if torch.cuda.get_device_capability(a.device)[0]!=9:
        raise ValueError('These schedules were measured on SM90/Hopper')
    with torch.cuda.device(a.device):
        if backend=='vendor':
            return torch.addmm(r,a,w.t())
        if (a.shape[0],a.shape[1],w.shape[0]) not in H100_RESIDUAL_CONFIGS:
            raise ValueError('No measured Triton schedule for this shape')
        out=torch.empty_like(r);bias=torch.zeros(w.shape[0],device=a.device,dtype=a.dtype)
        launch_h100_residual(a,w,r,bias,out)
        return out
