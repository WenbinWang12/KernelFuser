# EIT A800 campaign status

This is the sanitized handoff for the EIT A800 container. It is intentionally
separate from private host credentials, model weights, raw logs, and the complete
model harness. A Claude/Codex worker must read this file together with
`CLAUDE_CODEX_RUNBOOK.md`, `MODEL_AND_DATA_MANIFEST.md`, and the active task card
before changing the queue.

## Snapshot

The canonical A800 matrix described by `experiments/osdi_20261120/analysis/campaign.js`
is complete: 41/41 stages have a receipt (T-AdamW 12/12, T-Muon 5/5, T-SGD 6/6,
decode 11/11, long decode 3/3, long attention 3/3, and LM-head 1/1). The
receipt audit contains 25 `tested_pass` and 16 `tested_invalid` stages. Invalid
rows are completed gate outcomes and remain in the denominator; they are not
missing measurements and must not be silently retried.

The larger EIT queue also contains earlier protocol repeats and controller or
distributed probes. Before the follow-up cards below, it had 255 completed rows,
with eight terminal invalid/failed controller or multi-GPU rows and no unclaimed
canonical matrix row.
The distributed probes are compatibility diagnostics: they do not establish a
model-parallel or communication-optimization claim.

## Follow-up queue

Five separate A800 follow-up cards were appended to fill currently empty Muon
cells in the open-weight table. They use the same WikiText-2 training protocol,
`B=8,S=512`, BF16, 100 timed steps plus 10 warm-up steps, paired eager/compiled
baselines, correctness retest, and the existing `train_step_ce.py` harness:

```text
table2_muon_SmolLM2-135M
table2_muon_Qwen2.5-0.5B
table2_muon_Qwen2.5-1.5B
table2_muon_Gemma-3-1B
table2_muon_SmolLM2-1.7B
```

These are follow-up measurements, not part of the 41-stage canonical matrix. A
receipt may be promoted to the paper only after the one-step and multi-step gates,
paired retest, and provenance checks pass. Until then the corresponding Table 2
cells remain pending. The queue worker must record `tested_pass`, `tested_invalid`,
or a terminal execution failure and preserve the raw reason.

Sixteen additional figure-only cards are now queued for the A800s: attention
sequence/batch sweeps for Qwen3-0.6B and Qwen3-8B, decode cache/batch sweeps,
prefill grids, LM-head token sweeps, and Qwen3-0.6B AdamW/Muon shape sweeps.
Their complete factor definitions and output names are in
[`A800_ABLATION_QUEUE.md`](A800_ABLATION_QUEUE.md). They are diagnostic inputs
for scaling plots and must not replace the canonical 100-step results.

## Blocked A800 work

Do not queue a fake placeholder for a blocked architecture:

- TLX is a Hopper-oriented backend and has no validated A800 implementation in
  this campaign; use `N/A (arch.)` only when the receipt states the feature and
  architecture reason.
- Whole-model decode/training for Qwen3.5 hybrid attention and Qwen3.8 hybrid
  attention needs a functional decoder that is not in the current harness.
- Mistral-7B weights are not present in the EIT model manifest.
- Complete Qwen3-30B-A3B and Qwen3.5-35B-A3B training is outside the one-GPU
  memory/protocol envelope; their LM-head measurements are separate evidence.
- The production ranking model and its B200 harness are private and are not an
  A800 task. Public ADS snapshots are diagnostics with
  `publication_eligible=false`.

## Promotion rules

Use only sanitized receipts in the paper. Keep model revisions, input protocol,
GPU UUID/exclusivity snapshot, source hash, timing arms, correctness metrics,
failure reason, and retest status. Do not commit credentials, host paths,
weights, raw prompts, or unredacted logs. Re-run a `tested_invalid` row only as a
new, explicitly approved task after fixing the recorded source or model issue;
changing the shape or optimizer silently would invalidate the comparison.


The public checkout contains only the sanitized receipt snapshot; the EIT plan.tsv, state/, and raw queue logs stay on the host. The stale e10_steps directory-output entry is excluded from the canonical matrix and is not a missing row.

