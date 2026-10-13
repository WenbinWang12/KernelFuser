# OSU GPU handoff: revised P0/P1 runs

Run these on the OSU H100 and H200 systems only. Do not use withdrawn or
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

The initial implementation has passed a H100 B=1, S=128 split-half RoPE
correctness smoke. Full receipts remain unpromoted until all rows pass.

## P0-2: Qwen3-8B prefill with the same RoPE kernel

Run the model-level prefill using the same candidate and shape dispatch for
Qwen3-8B at `S=128,256` and `B=1,8,32`. Use compiled unfused and compiled fused
models, and validate full logits, top-1 predictions, NLL, and parameter/state
agreement before timing. Gemma sliding-window prefill is out of this scope;
do not reuse the withdrawn Gemma v2 receipts.

## P0-3: A/A noise floor extension

Use `src/e7_aa_noise_floor_osu_v3.py` after the source update. Collect at least
5--6 rows per GPU including S=128 and S=256 and the Qwen3-8B prefill shapes.
The current source has seven rows (Qwen3-8B/32B short sequences plus the prior
long-sequence controls). Keep the independent retest and trace archive.

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
