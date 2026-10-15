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

The H100 B and C counterparts completed on physical cards 1 and 2 and are
included beside the H200 receipts. The H100 A TLX smoke remains queued on
physical card 0 because that card is busy. H200 cards 0--3 were not used.
See [`experiments/ABC_VENDOR_ANCHOR_PLAN.md`](../../experiments/ABC_VENDOR_ANCHOR_PLAN.md)
for the full pre-registered scope and promotion rules.

* `a_smoke_h100_gpu1_fbtriton.json` is a fresh H100 TLX-toolchain smoke.  The
  full RMSNorm->GEMM output is numerically correct, but the standalone TLX
  anchor is 1.481x the cuBLAS anchor and the row is `anchor_gate_failed`.
* `a_smoke_h200_gpu5_torch28_patched.json` is the corresponding H200 user-space
  fbtriton/Torch 2.8 smoke on allowed physical card 5.  Correctness passes,
  while the TLX/cuBLAS anchor ratio is 1.853x; it is also
  `publication_eligible=false`.  This alternate environment is retained as a
  toolchain diagnostic and is not pooled with the formal H200 release stack.

These two receipts explain why the A census was not promoted: installing TLX
solves the import problem, but the SM90 port still fails the pre-registered
vendor-anchor gate.  No A number from these files belongs in `Experiment.tex`.

* `c_import_h100.json` and `c_import_h200.json` rerun the C smoke after the
  candidate source was added.  The implementation and fixed manifest are
  syntax/manifest-valid on both stacks, but TorchRec and FBGEMM are absent;
  both receipts remain `blocked_dependency` with no timing claim.

* `b_smoke_h200_gpu5_final4.json` is the first executable B smoke on allowed
  H200 card 5.  Q/K/V, attention, synthetic logits, top-1 and NLL checks pass,
  but the fused custom projection is 41.16x slower than `torch.mm` (0.01997 ms
  vendor versus 0.82182 ms custom).  It is therefore `tested_invalid` and not
  a paper result.  The smoke intentionally skips Inductor; a full B row must
  run the compiler audit before promotion.
* `e2_smoke_h200_gpu5_contig.json` is a sanitized Qwen3-8B prefill diagnostic
  after the harness fixes (frozen inference weights, contiguous Q/K/V, shared
  batched-RoPE validation, single-thread Inductor).  At B=1/S=128 the compiled
  baseline and the candidate both fail the strict full-logit/top-1 gate; no
  timing was recorded.  The maximum candidate-versus-vendor relative L2 is
  0.01251 and top-1 agreement is 0.96875.  It is not publication-eligible.
