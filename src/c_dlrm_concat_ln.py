#!/usr/bin/env python3
"""Public jagged DLRM-style concat + segment LayerNorm candidate.

This is the implementation stub for the pre-registered Track C lane.  The
input is a flat embedding-bag tensor ``values`` and an ``offsets`` array.  One
offset interval represents one (batch, feature) segment.  The kernel
normalizes each interval over all of its ``length * embedding_dim`` values;
because the intervals are already flat, no padded dense tensor is materialized.

This file deliberately contains only the candidate and an independent eager
reference.  It does not contain timing code or a synthetic result.  The Track
C smoke remains dependency-blocked until the public TorchRec/FBGEMM stack is
available on the target software environment.  The checked-in manifest caps
the largest interval at 64 * 128 = 8192 elements, which is the bound used by
the fixed-size Triton loop below.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK = 1024
MAX_BLOCKS = 8
MAX_SEGMENT_ELEMENTS = BLOCK * MAX_BLOCKS


@triton.jit
def _jagged_segment_ln(
    values_ptr,
    offsets_ptr,
    out_ptr,
    embedding_dim,
    epsilon,
    BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Normalize one variable-length segment without padding."""

    segment = tl.program_id(0)
    start = tl.load(offsets_ptr + segment).to(tl.int32)
    end = tl.load(offsets_ptr + segment + 1).to(tl.int32)
    length = end - start
    element_count = length * embedding_dim

    total = tl.zeros((), dtype=tl.float32)
    total_sq = tl.zeros((), dtype=tl.float32)
    for block_id in tl.static_range(0, MAX_BLOCKS):
        element = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = element < element_count
        ptr = values_ptr + start * embedding_dim + element
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(tl.where(mask, x, 0.0), axis=0)
        total_sq += tl.sum(tl.where(mask, x * x, 0.0), axis=0)

    count = tl.maximum(element_count, 1)
    mean = total / count
    variance = tl.maximum(total_sq / count - mean * mean, 0.0)
    inv_std = tl.rsqrt(variance + epsilon)

    for block_id in tl.static_range(0, MAX_BLOCKS):
        element = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = element < element_count
        ptr = values_ptr + start * embedding_dim + element
        out = (tl.load(ptr, mask=mask, other=0.0).to(tl.float32) - mean) * inv_std
        tl.store(out_ptr + start * embedding_dim + element, out, mask=mask)


def jagged_concat_layernorm(
    values: torch.Tensor,
    offsets: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Apply segment LayerNorm to a flat jagged embedding tensor.

    ``values`` has shape ``[total_tokens, embedding_dim]`` and ``offsets`` has
    shape ``[segments + 1]``.  The operation is intentionally scoped to the
    public C manifest (BF16 values, embedding dimensions up to 128, and a
    segment of at most 8192 scalar elements).  A future benchmark must compare
    this candidate with eager PyTorch, ``torch.compile``, and the vendor path
    using the same offsets; this function does not time or report speedups.
    """

    if not values.is_cuda:
        raise ValueError("jagged_concat_layernorm requires a CUDA values tensor")
    if values.dtype != torch.bfloat16:
        raise ValueError("the pre-registered C lane uses BF16 values")
    if values.ndim != 2 or not values.is_contiguous():
        raise ValueError("values must be a contiguous [total_tokens, embedding_dim] tensor")
    if offsets.ndim != 1 or not offsets.is_cuda or not offsets.is_contiguous():
        raise ValueError("offsets must be a contiguous CUDA vector")
    if offsets.dtype not in (torch.int32, torch.int64):
        raise ValueError("offsets must use int32 or int64 indices")
    if offsets.numel() < 2:
        return values.clone()
    if values.shape[1] > 128:
        raise ValueError("embedding_dim exceeds the public C manifest bound (128)")
    lengths = offsets[1:] - offsets[:-1]
    if bool(torch.any(lengths < 0).item()):
        raise ValueError("offsets must be nondecreasing")
    if bool(torch.any(lengths * values.shape[1] > MAX_SEGMENT_ELEMENTS).item()):
        raise ValueError("a jagged segment exceeds the fixed public C bound")
    if int(offsets[-1].item()) > values.shape[0]:
        raise ValueError("offsets refer past the end of values")

    output = torch.empty_like(values)
    _jagged_segment_ln[(offsets.numel() - 1,)](
        values,
        offsets,
        output,
        values.shape[1],
        eps,
        BLOCK=BLOCK,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=4,
        num_stages=2,
    )
    return output


def reference_jagged_concat_layernorm(
    values: torch.Tensor,
    offsets: torch.Tensor,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Independent FP32 reference used by a future correctness harness."""

    output = torch.empty_like(values)
    for segment in range(offsets.numel() - 1):
        start = int(offsets[segment].item())
        end = int(offsets[segment + 1].item())
        if end <= start:
            continue
        x = values[start:end].float()
        mean = x.mean()
        variance = (x - mean).square().mean()
        output[start:end] = ((x - mean) / torch.sqrt(variance + eps)).to(values.dtype)
    return output


__all__ = ["jagged_concat_layernorm", "reference_jagged_concat_layernorm"]
