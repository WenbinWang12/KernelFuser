# ABC smoke receipts

These are dependency and implementation-gate receipts, not performance data.
They were collected on the allowed H200 physical cards 5, 6, and 7 from the
`abc-vendor-anchor-plan` branch at commit `37a98e7`.

* `a_smoke_h200_gpu5.json` is `blocked_dependency`: the formal H200 stack
  (PyTorch 2.5.1, Triton 3.1.0) has no `triton.language.extra.tlx` namespace.
  The A full 126-row sweep was therefore not started, and the plain Triton
  fallback was not promoted.
* `b_smoke_h200_gpu6.json` is `ready_for_implementation`: the public Qwen3-8B
  shape manifest and vendor SDPA/compiler imports are present, but this is not
  a B performance result. The B kernel still has to be implemented and pass
  the standalone cuBLAS anchor gate.
* `c_smoke_h200_gpu7.json` is `blocked_dependency`: the fixed jagged manifest
  is present, while TorchRec/FBGEMM and the C implementation are absent. No C
  timing row was emitted.

The H100 counterparts are queued on physical cards 0, 1, and 2 and will be
added only after those cards become idle. H200 cards 0--3 were not used.
See [`experiments/ABC_VENDOR_ANCHOR_PLAN.md`](../../experiments/ABC_VENDOR_ANCHOR_PLAN.md)
for the full pre-registered scope and promotion rules.
