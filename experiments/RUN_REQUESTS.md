# OSU GPU handoff: revised P0/P1 runs

Run these on the OSU H100 and H200 systems only. H200 physical cards 0--3 are
reserved and must never be used. Any H200 run must use an available card in
4--7. Do not use withdrawn or
non-canonical receipts as substitutes. A promoted row must record source hash,
host/GPU/software versions, fixed seeds, correctness before timing, randomized
paired timing, fresh retest, GPU-exclusivity snapshots, and non-empty Chrome
traces. Any failed gate is `tested_invalid` and carries no speedup claim.

## P0-1: RoPE-aware isolated attention replacement for E1

Use `src/e1_rope_publication_v1.py`. The candidate fuses Q/K RMSNorm and
split-half RoPE into the fixed attention anchor. Sweep all nine geometries,
`S=128,256,384,512`, causal and non-causal, and `B=1,8,32` on both GPUs.
Compare (a) the same anchor with separate norm and RoPE, (b) `torch.compile`,
and (c) vendor SDPA. Compiled and vendor arms must be paired and alternated
against the candidate; retest rows whose initial ratio exceeds 1.01. The
runner's `--trace-dir` must be supplied.

Example (adjust Python and output paths for each host):

```bash
python src/e1_rope_publication_v1.py \
  --output outputs/e1_rope_h100.json \
  --trace-dir outputs/e1-rope-h100-traces
```

The v2 implementation passes the split-half RoPE correctness smoke on both
devices. The H100 and H200 Qwen3-8B 24-row receipts are complete. The earlier
non-paired full sweep remains diagnostic.

## P0-2: Qwen3-8B prefill with the same RoPE kernel

Run the model-level prefill using the same candidate and shape dispatch for
Qwen3-8B at `S=128,256` and `B=1,8,32`. Use compiled unfused and compiled fused
models, and validate full logits, top-1 predictions, NLL, and parameter/state
agreement before timing. The first smoke on both GPUs stopped at the
compiled-versus-eager baseline agreement gate and recorded no timing. Gemma
sliding-window prefill is out of this scope; do not reuse the withdrawn Gemma
v2 receipts.

## P0-3: A/A noise floor extension

Use `src/e7_rope_aa.py`. The six Qwen3-8B rows (B=1,8,32 and S=128,256,
causal) have passed on both GPUs with independent retests, exclusivity
monitoring and four traces per row. The older seven-row controls remain
supplementary.

## P1: offline and audit follow-ups

* Recompute E5 from canonical E1 receipts with `f > c/(1+c)`; the existing
  E5 JSON is diagnostic-only and must not enter the paper.
* E10 is not an optimization ablation. Either instrument one on the new RoPE
  kernel with one receipt per optimization step, or remove step-level claims.
* Correctness study: use the H200 13-checkpoint receipt in the main text; add
  H100 only after its audit passes.
* E8 Appendix C uses only the six H200 model-shaped rows for Llama3-8B and
  Qwen2.5-3B. The four Wukong rows are development provenance and must not be
  used in the paper claim.

## No run requested

Do not spend GPU time on residual E4 (move to Appendix C), E3c, E6, E9,
Qwen3-30B-A3B E3b, LM-head supplement, or SwiGLU-vs-cuBLAS: the existing
receipts are sufficient for their current appendix/status decisions.

## Return bundle

Return JSON receipts, source hashes, software versions, physical GPU and
exclusivity evidence, trace archive/manifest, and sanitized CSV. Do not return
weights, credentials, private host paths, or stale withdrawn E2 receipts.

## 2026-10-14 active OSU queue

The private OSU run directory uses the clean `src/e1_rope_publication_v1.py`
source from `origin/main`.  H100 runs the full nine-geometry census on its
assigned GPU.  H200 is split between physical GPUs 4 and 7, four geometries
on GPU 4 and five on GPU 7, for 216 rows in total.  Physical H200 GPUs 0--3
are prohibited.  E1 is not promoted until its JSON is complete and its row
count, status, source hash, software versions, correctness gates, paired
timing, retest records, traces, and exclusivity evidence are checked.

Three strict E2 waiters are active (H100 GPU 0, H200 GPU 4, and H200 GPU 7).
They start only after their corresponding E1 receipt is complete and the
assigned GPU is free.  E2 timing is accepted only when the compiled
candidate/reference gate and the eager full-logit correctness gate pass; a
failed gate remains diagnostic and carries no speedup claim.

### Observed status (2026-10-15)

The H100 E1 receipt completed with 216/216 rows and `tested_pass`.  Its source
hash is `b7fcae1f8fba39c920d1d5f8f7c1382f39163ea02e0c59d28472c9f443ad2272`;
the environment is H100 NVL, PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8.
All rows passed the correctness and exclusivity checks and contain four traces.
The H100 E2 receipt completed with six rows, all
`fail_compiled_baseline_correctness`; it contains no timing and is diagnostic
only.  The H200 E1 shards subsequently completed with 96 rows on physical GPU
4 and 120 rows on physical GPU 7 (216 total), both `tested_pass`, using
PyTorch 2.5.1, Triton 3.1.0 and CUDA 12.4.  The H200 E2 waiters then produced
six diagnostic rows per GPU; all rows failed the compiled-baseline correctness
gate before timing.  No H200 E2 speedup is promoted.

## 2026-10-15 ABC vendor-anchor follow-up

The proposed high-yield open-model follow-up is specified in
[`ABC_VENDOR_ANCHOR_PLAN.md`](ABC_VENDOR_ANCHOR_PLAN.md).  That file is the
source of truth for the pre-registered A/B/C scopes, baselines, compiler and
vendor-anchor gates, OSU queue commands, and receipt schema.  The current
repository has no B or C implementation or result: do not relabel the E1
attention-prologue rows as B, and do not use the Blackwell-only `tlx_jfa`
sources as an H100/H200 C result.

The ABC dependency smoke was run on H200 cards 5, 6, and 7. The A receipt is
`blocked_dependency` because the formal H200 Triton stack lacks the TLX
namespace; B is `ready_for_implementation` after the public Qwen3-8B shape
check; C is `blocked_dependency` because TorchRec/FBGEMM and the C kernel are
not installed. The corresponding H100 smoke scripts are queued on physical
cards 0, 1, and 2 and wait for exclusivity. Do not launch the 126-row A census
until its TLX smoke passes, and do not write any ABC number into
`Experiment.tex` before the receipt promotion gates in
`ABC_VENDOR_ANCHOR_PLAN.md` pass.
