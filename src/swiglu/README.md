# SwiGLU backends

These first-party modules replace only the pointwise SwiGLU activation
`silu(gate) * up`. They do not fuse either projection, routing, or the rest of
a model.

- `native_swiglu_tlx.py`: Hopper/SM90 TLX forward and backward kernels plus a
  first-order PyTorch autograd wrapper.
- `native_swiglu_triton.py`: ordinary Triton forward/backward comparator with
  the same BF16 intermediate-rounding contract.
- `../../tests/test_swiglu_sm90.py`: optional correctness checks for both
  backends.

## Requirements

Use a Linux environment with an NVIDIA Hopper (SM90) GPU, a compatible CUDA
PyTorch build, `pytest`, and the `fbtriton` build that provides
`triton.language.extra.tlx`. The recorded isolated probes used `fbtriton
3.7.4+fb`; compiler APIs and kernel compilation can vary between builds.
Ordinary Triton is also required for the comparator.

## Correctness check

From the repository root, run:

```bash
python -m pytest tests/test_swiglu_sm90.py
```

The tests skip when CUDA or the TLX extension is unavailable, and skip the
native TLX case on non-SM90 devices. They compare the fused forward result and
first-order gate/up gradients against PyTorch autograd for BF16 tensors,
including a non-contiguous input and a partial final tile.

The `fused(x, wg, wu)` helper in each module intentionally uses PyTorch
`F.linear` for both projections and applies the custom kernel only to the
activation. These are operator integrations, not end-to-end model runners.
To import a backend from a project that uses this repository directly, add
`src/` to `PYTHONPATH` and import `swiglu` from the desired module.
