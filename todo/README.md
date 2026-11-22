# KernelFuser experiment TODO

This directory is the handoff plan for the remaining measurements behind the
paper tables and figures.  It deliberately separates **operator evidence**
from **complete-model deployment**: an isolated kernel speedup must not be
reported as an end-to-end model speedup.

Start with [`CLAUDE_CODEX_RUNBOOK.md`](CLAUDE_CODEX_RUNBOOK.md) when you are
Claude or Codex executing a task.  It is the operational contract: task cards,
real runner commands, output directories, correctness/timing gates, failure
statuses, and the final handoff format.  Then read the concrete model and input
contract in [`MODEL_AND_DATA_MANIFEST.md`](MODEL_AND_DATA_MANIFEST.md), the
broader experiment/table plan in [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md),
and the field-level receipt contract in [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md).
The existing `experiments/RUN_REQUESTS.md` remains authoritative for the active
OSU P0/P1 handoff and GPU allocation constraints.

Do not start a GPU job until the owner has supplied any private checkpoint,
tokenizer, WikiText-2 preprocessing manifest, production-model data, or
complete-model harness required by the task.  The public checkout contains
operator runners and sanitized receipts, not model weights or private data.

## Current execution snapshot

The canonical EIT A800 matrix is receipt-complete (41/41 stages: 25
`tested_pass`, 16 `tested_invalid`). Five explicitly named Muon follow-ups are
queued separately; see [`EIT_A800_STATUS.md`](EIT_A800_STATUS.md). The OSU
H100/H200 handoff remains governed by `experiments/RUN_REQUESTS.md`.

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

## Files in this folder

- [`CLAUDE_CODEX_RUNBOOK.md`](CLAUDE_CODEX_RUNBOOK.md): executable task cards
  and commands for an experiment worker.
- [`RUNNER_INTERFACE_AUDIT.md`](RUNNER_INTERFACE_AUDIT.md): per-runner flags,
  input/output layouts, protocol boundaries, and common failure causes.
- [`MODEL_AND_DATA_MANIFEST.md`](MODEL_AND_DATA_MANIFEST.md): model geometries,
  synthetic inputs, checkpoint/data contracts, shape grids, and baselines.
- [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md): paper table/figure design and
  complete experiment matrix.
- [`RESULT_SCHEMA.md`](RESULT_SCHEMA.md): required JSON/CSV fields for a
  promoted or failed receipt.
- [`EIT_A800_STATUS.md`](EIT_A800_STATUS.md): sanitized EIT A800 queue
  snapshot, follow-up cards, and blocked boundaries.

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
