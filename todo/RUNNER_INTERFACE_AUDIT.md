# Public runner interface and output audit

This is a command-line reference for the public synthetic and model-prefill
runners.  It records the interfaces in the checked-out source; run `--help`
on the Linux worker after pinning the commit because the Windows checkout does
not provide the CUDA Python environment.  All paths below are examples.  Use
a unique `<RUN_ID>` and never overwrite a receipt or trace directory.

The public runners are not a single interchangeable benchmark.  They use
different input protocols and output layouts, and most are operator-level
evidence.  A row may enter a paper table only after the claim boundary,
correctness gate, paired timing, retest, trace, and GPU-exclusivity checks in
`CLAUDE_CODEX_RUNBOOK.md` pass.

## Attention and model-prefill runners

| Runner | Required arguments | Important optional arguments | Output shape | Common failure / boundary |
|---|---|---|---|---|
| `src/e1_rope_publication_v1.py` | `--output FILE`, `--trace-dir DIR` for promoted rows | `--models` comma list; `--sizes` comma list; `--batches` comma list; `--causal-modes false,true`; `--rounds` (5); `--samples` (40); `--smoke` | One JSON receipt at `--output`; four gzipped Chrome traces per row under `--trace-dir` | Requires CUDA, Triton and `nvidia-smi`; nonfinite/FP32-oracle mismatch, foreign GPU PID, compile/OOR, missing trace, or failed retest yields `tested_invalid`; isolated Q/K norm+RoPE only |
| `src/e1_rope_refined_census.py` | `--output FILE`, `--trace-dir DIR` | Same comma-list selectors; `--bn 64|128`; `--smoke` | One JSON receipt containing per-row records, plus traces | Refined follow-up, not the active OSU canonical runner; `--bn` is accepted here but **not** by `e1_rope_publication_v1.py` |
| `src/e1_qk_publication.py` | `--output FILE` | `--models`, `--sizes`, `--causal-modes` are comma lists; `--rounds`, `--samples` | JSON at `--output`; traces are created beside it under `<output-parent>/traces` | Legacy no-RoPE Q/K norm forward protocol, normally B=1; no `--trace-dir` selector; do not merge with RoPE or complete-model rows |
| `src/e1_qk_publication_v3.py` | `--output FILE` | `--models`, `--sizes` (default 128..2048), `--causal-modes`, `--rounds`, `--samples`; `--refine` additionally needs `--out` and `--census` | JSON receipt; refinement writes its own output | Legacy no-RoPE census, normally B=1; do not merge with RoPE or complete-model rows |
| `src/e1_rope_refined_impl.py` | Main mode: `--output FILE`; refinement/tuning mode: `--out FILE` plus `--refine` or `--tune` | Main mode accepts comma-list `--models`, `--sizes`, `--batches`, `--causal-modes`; refinement accepts `--models`, `--sequences`, `--batches`, `--causal-modes`, `--resource-safe` | JSON receipt; refinement may include compiler evidence and fallback choices | Contains legacy/refinement paths with different schemas; record the mode and do not combine receipts |
| `src/e2e_prefill_refined.py` | `--model-dir CHECKPOINT_DIR`; `--output FILE` is required in practice | `--shapes BxS,...` (default `1x128,1x512,8x256,32x128`), `--bn`, `--rounds-initial`, `--rounds-retest`, `--samples`, `--cpu-check` | JSON receipt at `--output`; writes partial receipt on gate failure | Needs a local HF checkpoint and tokenizer/config; CPU/compiled full-logit agreement is checked before timing; a failed gate means no timing claim |
| `src/e2_qwen3_rope_prefill.py` | `--model CHECKPOINT_DIR`, `--out FILE` | `--sizes` and `--batches` are **space-separated** integers; `--warmup`, `--repeats`, `--rounds`, `--smoke` | One JSON receipt at `--out` | Legacy monkey-patch prefill; causal Qwen3 only; keep separate from `e2e_prefill_refined.py` and Table 2 |
| `src/e2_gemma_forward.py` | `--model CHECKPOINT_DIR`, `--model-name NAME`, `--out FILE` | `--seqlens` is a space-separated list (default 512,1024,2048,4096); `--warmup`, `--repeats`, `--rounds` | JSON at `--out`; traces under `<out-stem>_traces` | Gemma text-decoder forward-only runner; local checkpoint only; Gemma sliding-window/withdrawn v2 rows are outside the active OSU handoff |
| `src/e2_gemma_forward_v5_replay.py` | `--model CHECKPOINT_DIR`, `--model-name NAME`, `--out FILE` | Same space-separated `--seqlens`; `--warmup`, `--repeats`, `--rounds` | JSON at `--out`; replay traces under `<out-stem>_traces` | Compiled replay variant with eager correctness; audit-only unless the owner opens a new Gemma task card |

Canonical E1 command (active OSU handoff):

```bash
python src/e1_rope_publication_v1.py \
  --output outputs/<RUN_ID>/e1_rope.json \
  --trace-dir outputs/<RUN_ID>/e1-traces
```

The comma syntax above is intentional.  In contrast, the legacy E2 runner
expects `--sizes 128 256 --batches 1 8 32`.

## MoE, epilogue, noise-floor, and E8 runners

| Runner | Required arguments | Output layout | Input/defaults and boundary |
|---|---|---|---|
| `src/e3a_moe_publication.py` | `--out FILE`, `--trace-dir DIR` | JSON at `--out`; five-arm traces under `--trace-dir` | `--tokens 2048,8192,32768,131072`; `--only` is a comma list.  Synthetic balanced expert-local tensors, backward included, no router/all-to-all/shared/full model.  Large DeepSeek grouped-MM materialization may OOM and invalidates that row. |
| `src/moe_staged_strict.py` | `--out FILE`, `--trace-dir DIR` | JSON at `--out`; baseline/candidate traces under `--trace-dir` | Same token protocol; `--tokens` defaults to the built-in tuple when omitted or `,`.  Staged expert-local evidence only. |
| `src/e4_model_shapes_publication_v2.py` | `--out DIR`, `--trace-dir DIR` | `DIR/results.json`, `DIR/compiler/*`, and traces | `--M 128,512,2048,8192`; optional `--rounds`, `--samples`; `--refine`/`--tune` select another mode.  Do not launch residual E4 when the active `RUN_REQUESTS.md` marks it no-run. |
| `src/e4_epilogue_publication.py` | `--out FILE`, `--trace-dir DIR` | JSON at `--out`; site traces under `--trace-dir` | `--sites gemm_residual`; `--M` comma list.  Isolated epilogue site, not complete training. |
| `src/e7_rope_aa.py` | `--out FILE` (or `--output FILE`), `--trace-dir DIR` | JSON at `--out`; A/A traces under `--trace-dir` | `--batches`/`--sizes` comma lists; defaults B=1,8,32 and S=128,256; no fused-vs-baseline claim, only noise floor. |
| `src/e7_aa_noise_floor_osu_v3.py` | `--out FILE`, `--trace-dir DIR` | JSON at `--out`; traces under `--trace-dir` | Fixed seven-control A/A suite; no shape selectors. |
| `src/e8_forward_publication.py --h100-unfused` | `--out DIR` | `DIR/results.json`, compiler evidence, and `DIR/traces` | H100 synthetic controls; forward-only.  Development Wukong geometry must not be relabelled as a Qwen/Gemma result. |
| `src/e8_forward_publication.py --b200-ads` | `--out DIR`; `--ads-source PATH` unless `--baselines-only` | `DIR/results.json`, optional `--csv FILE`, traces | Private ADS source/version gate.  `--allow-source-version` requires owner-approved diff audit; mismatch sets `publication_eligible=false`. |
| `src/tlx_pilot/tlx_swiglu_hopper.py` | `--out DIR` | TLX-specific receipt/artifacts under `DIR` | `--seed` optional.  Requires SM90/Hopper and matching `fbtriton`; unavailable architecture is N/A, not a Triton substitute. |
| `src/tlx_ads_sm90_port/ads_sm90_rmsnorm_matmul.py` | `--out DIR` | TLX receipt/artifacts under `DIR` | `--M`, `--K`, `--N` optional; isolated RMSNorm/GEMM port only. |

## Receipt and handoff checklist

For every invocation, preserve the exact command, commit/source hash, Python,
PyTorch, CUDA, Triton/TLX versions, `CUDA_VISIBLE_DEVICES`, GPU snapshot,
input seed/shape list, and stdout/stderr.  The receipt should finish with an
explicit `tested_pass`, `tested_invalid`, `compile_fail`, `timeout`, `N/A`, or
`blocked` status.  A missing model, tokenizer, dataset, private ADS source,
CUDA-capable worker, or exclusive GPU is `blocked`; do not substitute a nearby
checkpoint or silently change the shape list.

Before copying a result into `results/`, verify that every compared arm has a
non-empty trace, correctness was checked against the declared oracle, the
candidate and baseline were paired, and an independent retest supports any
claimed ratio above 1.01.  Sanitize hostnames, UUIDs, PIDs, private paths,
weights, credentials, and raw traces from the committed copy.  Keep invalid
or rejected shapes in the coverage summary with no speedup claim.

`src/epilogue_sites.py`, `src/fusion_harness.py`,
`src/norm_into_gemm_kernel.py`, and the files under `src/swiglu/` are library
modules rather than standalone experiment entry points; they have no
independent `--help` contract.  Exercise them only through the runner named in
the corresponding task card and record the importing runner's source hash.
