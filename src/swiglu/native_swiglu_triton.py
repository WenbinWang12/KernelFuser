#!/usr/bin/env python3
"""Triton SwiGLU kernel with PyTorch-compatible BF16 rounding.

For ``y = silu(g) * u``, autograd first forms ``dy_u = dy * u`` for the
product node, then the SiLU node evaluates ``(dy_u * sigmoid(g)) *
(1 + g * (1 - sigmoid(g)))``.  The two BF16 stores mirror the BF16 tensor
results of those graph nodes before the final BF16 gradient store.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl

BLOCK = 1024

@triton.jit
def _fwd(gp, up, yp, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    g = tl.load(gp+i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    u = tl.load(up+i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
    silu = (g*s).to(tl.bfloat16).to(tl.float32)
    tl.store(yp+i, (silu*u).to(tl.bfloat16), mask=m)

@triton.jit
def _bwd(gp, up, dyp, dgp, dup, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = i < n
    g = tl.load(gp+i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    u = tl.load(up+i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    dy = tl.load(dyp+i, mask=m, other=0).to(tl.bfloat16).to(tl.float32)
    s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-g))
    silu = (g*s).to(tl.bfloat16).to(tl.float32)
    # Match y = silu(g) * u's autograd node order, retaining BF16 tensor
    # results between product and SiLU backward operations.
    dyu = (dy*u).to(tl.bfloat16).to(tl.float32)
    dyus = dyu*s  # PyTorch SiLU backward keeps this multiply in accumulator dtype
    factor = 1.0 + g*(1.0-s)
    dg = (dyus*factor).to(tl.bfloat16)
    du = (dy*silu).to(tl.bfloat16)
    tl.store(dgp+i, dg, mask=m); tl.store(dup+i, du, mask=m)

def _check(g,u):
    if not (g.is_cuda and u.is_cuda and g.device==u.device and g.dtype==u.dtype==torch.bfloat16 and g.shape==u.shape):
        raise ValueError('native_swiglu_triton requires same-shape BF16 CUDA tensors')

def forward(g,u,block=BLOCK):
    _check(g,u); sh=g.shape; gg=g.contiguous().reshape(-1); uu=u.contiguous().reshape(-1); y=torch.empty_like(gg)
    _fwd[(triton.cdiv(gg.numel(),block),)](gg,uu,y,gg.numel(),BLOCK=block,num_warps=4,enable_fp_fusion=False); return y.reshape(sh)

def backward(g,u,dy,block=BLOCK):
    _check(g,u); _check(g,dy); sh=g.shape; gg=g.contiguous().reshape(-1); uu=u.contiguous().reshape(-1); dd=dy.contiguous().reshape(-1); dg=torch.empty_like(gg); du=torch.empty_like(uu)
    _bwd[(triton.cdiv(gg.numel(),block),)](gg,uu,dd,dg,du,gg.numel(),BLOCK=block,num_warps=4,enable_fp_fusion=False); return dg.reshape(sh),du.reshape(sh)

class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx,g,u): ctx.save_for_backward(g,u); return forward(g,u)
    @staticmethod
    def backward(ctx,dy): return backward(*ctx.saved_tensors,dy)

def swiglu(g,u): return _Fn.apply(g,u)
def fused(x,wg,wu,**kwargs):
    import torch.nn.functional as F
    return swiglu(F.linear(x,wg),F.linear(x,wu))
__all__=['swiglu','forward','backward','fused']
