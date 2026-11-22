# KernelFuser experiment TODO

This directory is the handoff plan for the remaining measurements behind the
paper tables and figures.  It deliberately separates **operator evidence**
from **complete-model deployment**: an isolated kernel speedup must not be
reported as an end-to-end model speedup.

Start with [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md).  Every run should return
the receipt fields and acceptance gates in [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md).
The existing `experiments/RUN_REQUESTS.md` remains authoritative for the active
OSU P0/P1 handoff and GPU allocation constraints.

## Priority queue

- [ ] **P0 — matched main matrix:** complete-model decode and training on
  A800/H100/H200 for the selected dense and MoE checkpoints, with both Triton
  and TLX where the implementation exists.  Fill Table 2 only from complete
  model receipts.
- [ ] **P0 — production-model integration:** rerun the B200 production shapes
  after the final kernel/source pin.  Record all generated-kernel correctness,
  integrated count, end-to-end training QPS, and the eager/compiled baselines
  used for Table 1.
- [ ] **P0 — fusion-site census:** one matched baseline/fused receipt for every
  fusion family in the taxonomy, across the required shape sweep.  These are
  the data for Table 4 and the main speedup figure.
- [ ] **P0 — baseline comparison:** eager PyTorch, `torch.compile`, and the
  strongest vendor/library arm must be paired with each candidate.  Record
  candidate-vs-baseline and fusion-only-vs-anchor ratios separately.
- [ ] **P1 — scaling curves:** sequence-length and batch-size sweeps, with
  fixed model/GPU/backend and an explicit causal/non-causal split.
- [ ] **P1 — controller ablations:** model-only, harness-only, and full
  controller under the same proposal/template/timing budget.
- [ ] **P1 — training ablations:** AdamW and Muon, forward/backward/optimizer
  step timings, peak memory, and complete-step throughput.
- [ ] **P2 — diagnostics:** compiler/resource counters, traces, rejected
  shapes, failure cases, and source/provenance audits.  These support the
  explanation but do not become speedup claims by themselves.

## Before starting a run

1. Pin the repository commit, Python/PyTorch/CUDA/Triton/TLX versions, GPU
   model, clock/power settings, dtype, seed list, and exact shape list.
2. Run correctness and baseline-agreement gates before any timing.  A failed
   gate is `tested_invalid`, with no speedup value.
3. Alternate candidate and baseline arms in randomized order, use warm-up and
   synchronized CUDA timing, and perform an independent fresh retest for every
   claimed win.
4. Check GPU exclusivity before and during timing.  Keep the Chrome trace and
   compiler/source fingerprints for promoted rows.
5. Return JSON receipt, sanitized CSV, trace manifest, and a short run note.
   Do not commit model weights, credentials, hostnames, GPU UUIDs, raw traces,
   or private paths.

## Current known boundaries

- The public E1/E1-RoPE/E4/E7 files are operator-level studies.  They do not
  establish full-model decode or training gains.
- The Qwen3-8B model-prefill runner currently stops at the compiled/eager
  agreement gate; do not invent timing rows from that smoke test.
- B200 ADS CSVs are snapshot diagnostics with
  `publication_eligible=False`; they require the source-version audit before
  any paper claim.
- Withdrawn Gemma receipts and non-paired historical ratios are not valid
  substitutes for the P0 matrix.
