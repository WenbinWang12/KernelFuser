#!/usr/bin/env python3
"""Native Hopper TLX SwiGLU backend.

This module replaces only the pointwise SwiGLU activation with a TLX kernel;
projections, routing and all other model operators remain the caller's
implementation.

The first-order forward and backward paths are both native TLX kernels.  The
two async task regions write disjoint halves of each flat tile, so no eager
fallback or cross-task synchronization is needed.  The source is SM90/Hopper
only and expects the fbtriton build that provides ``triton.language.extra.tlx``.
Higher-order gradients are rejected explicitly rather than silently returning a
non-differentiable first-order result.
"""
from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


BLOCK = 1024


@triton.jit
def _fwd(
    gp,
    up,
    yp,
    n,
    BLOCK: tl.constexpr,
):
    """TLX forward SwiGLU with BF16 intermediate rounding."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    # Each task owns a disjoint range.  Keep the intermediate BF16 round after
    # SiLU exactly as in the matching Triton implementation.
    with tlx.async_tasks():
        with tlx.async_task("default"):
            i = base + tl.arange(0, BLOCK // 2)
            m = i < n
            g = tl.load(gp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            u = tl.load(up + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
            silu = (g * s).to(tl.bfloat16).to(tl.float32)
            tl.store(yp + i, (silu * u).to(tl.bfloat16), mask=m)
        with tlx.async_task(num_warps=4):
            i = base + BLOCK // 2 + tl.arange(0, BLOCK // 2)
            m = i < n
            g = tl.load(gp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            u = tl.load(up + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
            silu = (g * s).to(tl.bfloat16).to(tl.float32)
            tl.store(yp + i, (silu * u).to(tl.bfloat16), mask=m)


@triton.jit
def _bwd(
    gp,
    up,
    dyp,
    dgp,
    dup,
    n,
    BLOCK: tl.constexpr,
):
    """Native TLX backward for gate and up tensors."""
    pid = tl.program_id(0)
    base = pid * BLOCK
    with tlx.async_tasks():
        with tlx.async_task("default"):
            i = base + tl.arange(0, BLOCK // 2)
            m = i < n
            g = tl.load(gp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            u = tl.load(up + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            dy = tl.load(dyp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
            silu = (g * s).to(tl.bfloat16).to(tl.float32)
            # PyTorch's product backward materializes dy*u in BF16 before the
            # SiLU backward multiply.
            dyu = (dy * u).to(tl.bfloat16).to(tl.float32)
            dyus = dyu * s
            dg = (dyus * (1.0 + g * (1.0 - s))).to(tl.bfloat16)
            du = (dy * silu).to(tl.bfloat16)
            tl.store(dgp + i, dg, mask=m)
            tl.store(dup + i, du, mask=m)
        with tlx.async_task(num_warps=4):
            i = base + BLOCK // 2 + tl.arange(0, BLOCK // 2)
            m = i < n
            g = tl.load(gp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            u = tl.load(up + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            dy = tl.load(dyp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
            s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
            silu = (g * s).to(tl.bfloat16).to(tl.float32)
            dyu = (dy * u).to(tl.bfloat16).to(tl.float32)
            dyus = dyu * s
            dg = (dyus * (1.0 + g * (1.0 - s))).to(tl.bfloat16)
            du = (dy * silu).to(tl.bfloat16)
            tl.store(dgp + i, dg, mask=m)
            tl.store(dup + i, du, mask=m)


def _check(g: torch.Tensor, u: torch.Tensor, name: str = "native_swiglu_tlx") -> None:
    if not isinstance(g, torch.Tensor) or not isinstance(u, torch.Tensor):
        raise TypeError(f"{name} requires tensors")
    if not (g.is_cuda and u.is_cuda):
        raise ValueError(f"{name} requires CUDA tensors")
    if g.device != u.device:
        raise ValueError(f"{name} requires tensors on one CUDA device")
    if g.dtype != torch.bfloat16 or u.dtype != torch.bfloat16:
        raise ValueError(f"{name} requires BF16 tensors")
    if g.shape != u.shape:
        raise ValueError(f"{name} requires same-shape gate/up tensors")
    major, _minor = torch.cuda.get_device_capability(g.device)
    if major != 9:
        raise RuntimeError(f"{name} is an SM90/Hopper backend; got capability {major}.x")


def forward(g: torch.Tensor, u: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Run native TLX SwiGLU forward with PyTorch-compatible rounding."""
    _check(g, u)
    if block <= 0 or block % 2:
        raise ValueError("block must be a positive even integer")
    shape = g.shape
    gg = g.contiguous().reshape(-1)
    uu = u.contiguous().reshape(-1)
    if gg.numel() == 0:
        return torch.empty_like(g)
    yy = torch.empty_like(gg)
    with torch.cuda.device(g.device):
        _fwd[(triton.cdiv(gg.numel(), block),)](
            gg,
            uu,
            yy,
            gg.numel(),
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return yy.reshape(shape)


def backward(
    g: torch.Tensor,
    u: torch.Tensor,
    dy: torch.Tensor,
    block: int = BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run native TLX backward, returning dgate and dup in input shape."""
    _check(g, u)
    _check(g, dy, "native_swiglu_tlx.backward")
    if block <= 0 or block % 2:
        raise ValueError("block must be a positive even integer")
    shape = g.shape
    gg = g.contiguous().reshape(-1)
    uu = u.contiguous().reshape(-1)
    dd = dy.contiguous().reshape(-1)
    if gg.numel() == 0:
        return torch.empty_like(g), torch.empty_like(u)
    dg = torch.empty_like(gg)
    du = torch.empty_like(uu)
    with torch.cuda.device(g.device):
        _bwd[(triton.cdiv(gg.numel(), block),)](
            gg,
            uu,
            dd,
            dg,
            du,
            gg.numel(),
            BLOCK=block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return dg.reshape(shape), du.reshape(shape)


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(g, u)
        return forward(g, u)

    @staticmethod
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError(
                "native_swiglu_tlx does not provide higher-order gradients; "
                "use the first-order F/B contract"
            )
        g, u = ctx.saved_tensors
        return backward(g, u, dy)


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Autograd-compatible native TLX SwiGLU."""
    return _Fn.apply(g, u)


def fused(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Projection wrapper matching the strict native backend API.

    Only the SwiGLU activation is TLX; the two projections intentionally remain
    PyTorch `F.linear`, matching the existing Triton runner's claim boundary.
    """
    import torch.nn.functional as F

    return swiglu(F.linear(x, wg), F.linear(x, wu))


__all__ = ["swiglu", "forward", "backward", "fused"]
