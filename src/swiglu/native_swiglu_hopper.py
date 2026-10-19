#!/usr/bin/env python3
"""Hopper-safe Triton SwiGLU backend used by the E9 equivalent runner.

It has the same BF16 intermediate-rounding and first-order backward contract
as ``native_swiglu_tlx`` but uses only Triton primitives available in the
H100/H200 software stacks.  This is deliberately named separately from TLX.
"""
from __future__ import annotations
from typing import Any
import torch
import triton
import triton.language as tl

BLOCK = 1024

@triton.jit
def _fwd(gp, up, yp, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    g = tl.load(gp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    u = tl.load(up + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g)); silu = (g * s).to(tl.bfloat16).to(tl.float32)
    tl.store(yp + i, (silu * u).to(tl.bfloat16), mask=m)

@triton.jit
def _bwd(gp, up, dyp, dgp, dup, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    g = tl.load(gp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    u = tl.load(up + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    dy = tl.load(dyp + i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-g)); silu = (g * s).to(tl.bfloat16).to(tl.float32)
    dyu = (dy * u).to(tl.bfloat16).to(tl.float32)
    tl.store(dgp + i, (dyu * s * (1.0 + g * (1.0 - s))).to(tl.bfloat16), mask=m)
    tl.store(dup + i, (dy * silu).to(tl.bfloat16), mask=m)

def _check(g, u, name="native_swiglu_hopper"):
    if not (isinstance(g, torch.Tensor) and isinstance(u, torch.Tensor)): raise TypeError(f"{name} requires tensors")
    if not (g.is_cuda and u.is_cuda and g.device == u.device): raise ValueError(f"{name} requires same CUDA device")
    if g.dtype != torch.bfloat16 or u.dtype != torch.bfloat16 or g.shape != u.shape: raise ValueError(f"{name} requires same-shape BF16 tensors")
    if torch.cuda.get_device_capability(g.device)[0] != 9: raise RuntimeError("SM90/Hopper required")

def forward(g, u, block=BLOCK):
    _check(g, u); gg=g.contiguous().reshape(-1); uu=u.contiguous().reshape(-1); y=torch.empty_like(gg)
    _fwd[(triton.cdiv(gg.numel(), block),)](gg, uu, y, gg.numel(), BLOCK=block, num_warps=4)
    return y.reshape(g.shape)

def backward(g, u, dy, block=BLOCK):
    _check(g, u); _check(g, dy, "native_swiglu_hopper.backward"); gg=g.contiguous().reshape(-1); uu=u.contiguous().reshape(-1); dd=dy.contiguous().reshape(-1); dg=torch.empty_like(gg); du=torch.empty_like(uu)
    _bwd[(triton.cdiv(gg.numel(), block),)](gg, uu, dd, dg, du, gg.numel(), BLOCK=block, num_warps=4)
    return dg.reshape(g.shape), du.reshape(u.shape)

class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, g, u): ctx.save_for_backward(g, u); return forward(g, u)
    @staticmethod
    def backward(ctx: Any, dy):
        g, u = ctx.saved_tensors; return backward(g, u, dy)

def swiglu(g, u): return _Fn.apply(g, u)
def fused(x, wg, wu, **kwargs):
    import torch.nn.functional as F
    return swiglu(F.linear(x, wg), F.linear(x, wu))
__all__ = ["swiglu", "forward", "backward", "fused"]
