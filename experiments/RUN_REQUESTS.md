# OSU GPU run requests

This file is the handoff list for experiments that cannot be completed in the
current Windows workspace. Run only on the requested OSU GPU, with no foreign
compute process on the selected physical GPU. Do not use the old BABEL H100
receipts as a substitute.

Every returned receipt must include the source hash, host/GPU/software
environment, fixed seeds, correctness results before timing, randomized paired
timing, independent retest when the ratio exceeds 1.01, GPU-exclusivity
snapshots, and non-empty Chrome traces. A row that fails correctness or
exclusivity is `tested_invalid` and must not have a speedup claim.

## 1. E2 Gemma prefill: H100 rerun

**Target:** OSU H100, one GPU at a time. Use Gemma-3 4B, 12B, and 27B local
text-decoder checkpoints at `S=512,1024,2048,4096`.

The candidate is the replay-first source in `src/e2_gemma_forward_v5_replay.py`
and `src/e2_gemma_forward.py`. The runner takes a local checkpoint path and
does not download weights:

```bash
python src/e2_gemma_forward_v5_replay.py \
  --model /path/to/Gemma-3-4B \
  --model-name Gemma-3-4B \
  --out outputs/e2_gemma4_h100.json \
  --seqlens 512 1024 2048 4096
```

Repeat for 12B and 27B. Promote only if every row passes finite logits,
full-logit agreement, top-1 agreement, NLL agreement, compiled-arm timing,
retest, trace, and GPU-exclusivity gates. If the model-level gate fails, keep
the receipt and stop timing that row.

**Expected output:** one JSON receipt per model plus its generated trace
directory. The paper's E2E table must remain blank if any model-level gate
fails.

## 2. E3a MoE expert-local H100 audit

**Target:** OSU H100. Run the nine public geometries and all four token counts
unless a resource limit is documented:

```bash
python src/e3a_moe_publication.py \
  --out outputs/e3a_h100.json \
  --trace-dir outputs/e3a-h100-traces \
  --tokens 2048,8192,32768,131072
```

This is an expert-local forward/backward study. It does not establish routing,
full-model, or parameter-update performance. The existing H200 receipt is
already the promoted architecture comparison; H100 must pass the same audit.

## 3. E8 H100 failure-case census

**Target:** OSU H100. Run the forward-only dense RMSNorm and SwiGLU failure
cases with the public runner:

```bash
python src/e8_forward_publication.py \
  --out outputs/e8_h100.json \
  --trace-dir outputs/e8-h100-traces \
  --only dense_rmsnorm,swiglu
```

The runner checks an independent FP32 oracle before timing, uses randomized
20/100/5 CUDA-graph pairs, conditionally retests, and records plain/fused/
vendor arms. This result is isolated forward-only evidence. It does not fill
the LM-head rows.

## 4. E8 LM-head and cuBLAS supplement

Run the final normalization-to-LM-head rows on OSU H100 with the same strict
gate and an explicit `torch.addmm` vendor arm. Also report SwiGLU against the
vendor arm, rather than only against a slower hand-written anchor. The model
checkpoint loader is intentionally not included in this public repository;
the returned receipt must be sanitized before adding a public CSV.

## 5. E10 independent optimization ablation

The current `e10_h100.json` and `e10_h200.json` are exact forward-only rows
from the E1 runner and are **not** an independent optimization ablation. Do
not use them to claim separate effects for hoisting, fold gamma, key scaling,
or reduction changes. A valid replacement must time the same fixed shape and
inputs after each optimization step, retain correctness and traces for every
step, and include a baseline plus one row per step.

## 6. Offline E5 rescore after receipt return

This step needs no GPU. Recompute the cost-aware decision from the canonical
`e1_h100_osu_v3.json` and `e1_h200_v3.json` receipts using

```text
f = (t_F_unfused - t_G) / t_F_unfused
c = (t_F_fused - t_G) / t_G
fuse iff f > c / (1 + c)
```

The existing `e5_break_even_rescore_v2.json` is diagnostic-only because it
uses an old merged H100 input and reconstructs `f` and `c` from the same timing
triplet as the observed ratio. It must not be used for a paper claim.

## Deliberately out of scope for this handoff

Do not spend GPU time on E3c full-model MoE training, E6 TLX selection, or E9
TLX end-to-end. E3c's previous attempts failed with OOM/logit mismatch, and
E6/E9 depend on Blackwell-only `tmem` instructions that do not compile on
Hopper SM90. The current project scope also does not request another
E3b/Qwen3-30B-A3B run.

## Return checklist

Return the JSON receipt, source/implementation hashes, software versions,
selected physical GPU and exclusivity snapshots, trace archive or trace
manifest, and a short sanitized CSV. Do not return model weights, credentials,
private local paths, or raw host identifiers in the public repository.
