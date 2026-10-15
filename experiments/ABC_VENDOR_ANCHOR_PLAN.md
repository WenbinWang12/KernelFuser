# ABC follow-up: vendor-anchor fusion census

This document is the handoff for the next GPU campaign.  It is intentionally
an experiment specification, not a result file.  A row is not a paper result
until the receipt passes the correctness, timing, source, and GPU-exclusivity
gates below.  Do not fill an absent number with an estimate.

## Why this campaign exists

The existing open-model rows mostly use a hand-written Triton attention anchor.
That anchor is sometimes slower than the compiled or vendor implementation,
so a fusion can lose even when it removes a real kernel.  The new campaign
restricts the claim to two pre-registered conditions:

1. the compiler does not already produce the requested fusion; and
2. the unfused anchor is within 1% of the fastest vendor/compiler implementation
   on the same shape.

The paper reports every row in the pre-registered scope, including rejected
rows.  Rows outside this scope remain diagnostics and cannot be used to raise
the win rate.

## Common receipt and acceptance contract

Every row must record the source commit and SHA256, GPU model and physical GPU
identifier, PyTorch/Triton/CUDA versions, dtype, shape, seed, warmups, sample
count, paired arm order, correctness metrics, timing samples, retest, four
Chrome traces (candidate, unfused, compiled, vendor), and one-second
exclusivity snapshots.  Remove hostnames, UUIDs, PIDs, weights, and local paths
from the public CSV; retain them only in the private audit bundle.

Use CUDA-event timings after warmup.  Alternate candidate and each baseline in
the same process and repeat the pair when the initial speedup exceeds 1.01x.
The fastest baseline for a row is the minimum of the three baseline medians.
Report `speedup = fastest_baseline_ms / candidate_ms`; `speedup > 1` is a win.
Correctness is checked against an independent FP32 reference before any timing.
Any failed gate is `tested_invalid` and has no speedup claim.

The two OSU environments must be reported separately.  Do not pool H100 and
H200 ratios or attribute a stack difference to hardware.  H200 physical GPUs
0--3 are reserved; use an available H200 card 4--7.  H100 cards may be used
when the scheduler grants them.  A job must wait for an exclusive card rather
than sharing a busy device.

## A -- lazy RMSNorm to projection GEMM (priority 1)

The A implementation is the existing RMSNorm-to-GEMM family in
`src/norm_into_gemm_kernel.py` and `src/tlx_ads_sm90_port/`.  Before launching
the census, pin the exact public commit and verify that the kernel computes

```
  y = (x * rsqrt(mean(x*x) + eps) * gamma) @ W.T + bias
```

without materializing the normalized input.  If gamma is folded into `W`, the
receipt must state that this is the lazy-norm variant and include a separate
unfused reference with the same folded weights.

The OSU release environments currently used by the paper (H100 PyTorch
2.11/Triton 3.6 and H200 PyTorch 2.5.1/Triton 3.1) do not expose
`triton.language.extra.tlx`.  Therefore the first A job on each GPU must be a
one-shape import/compile/correctness smoke.  If TLX is unavailable, return
`blocked_dependency` with the exact environment and do not relabel the plain
Triton fallback as TLX or as a vendor-anchor result.  A fallback may be kept
as a separately named diagnostic only.  A publication-eligible A receipt must
identify the installed TLX package/source and pass the same smoke before the
126-row census is started.

Use these nine public model families, with hidden/intermediate/head dimensions
resolved from a pinned public config before the first timed row: Qwen3-0.6B,
Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-32B, Qwen2.5-0.5B-Instruct,
OLMo-1B, Llama-3.1-8B and Mistral-7B.  For each family test both the fused
QKV projection (`N = Q + K + V`, using the actual query/KV head counts) and
the gate/up projection (`N = 2 * intermediate_size`).  Emit the resolved
model/site/K/N manifest and its SHA256; this makes the shape-only run
reproducible without downloading weights.  Sweep
`M in {1, 8, 32, 128, 512, 2048, 8192}` where the allocation fits; record an
explicit `oom` row instead of dropping a shape.  The declared scope is 9 x 2 x
7 = 126 rows per GPU.  Run H100 and H200 with the same public release software
stack where possible.  Baselines are, in the same process:

* compiled RMSNorm followed by cuBLAS GEMM;
* row-scale cuBLAS GEMM; and
* `torch.compile(mode="max-autotune")` for the unfused reference.

First measure the standalone GEMM anchor.  Promote an A row only when the
unfused vendor/TLX anchor is no slower than the fastest baseline by more than
1%.  Then compare the fused result with all three baselines.  Include an
`M=128` optimization-step ablation (old schedule, reduction split, final
schedule) on one QKV and one gate/up shape per device.

Suggested OSU command after the implementation audit (the runner does not
exist on `origin/main` yet):

```bash
python src/a_tlx_rmsnorm_gemm_model_census.py \
  --device-tag h100 --sites qkv,gate_up --m 1,8,32,128,512,2048,8192 \
  --out outputs/a_rmsnorm_gemm_h100.json
```

Do not report the existing ADS/B200 snapshot as A: it is explicitly marked
`snapshot_diagnostic` and uses a different source commit.

## B -- QKV-output Q/K norm+RoPE with vendor attention (priority 2)

No B runner or kernel is present in this repository.  The existing E1/E2 code
fuses norm+RoPE in an attention prologue and must not be relabeled as B.

Implement B as a new two-stage comparison for the Qwen3-8B geometry first,
then expand to the same nine geometries only after the single-model smoke
passes.  The candidate must fuse the QKV projection output write with Q/K
RMSNorm and split-half RoPE while leaving V unchanged, and must call the same
vendor SDPA/FlashAttention implementation as the baseline.  A custom GEMM is
allowed only if its standalone QKV GEMM is within 1% of cuBLAS on that shape;
otherwise the row is rejected as a non-vendor anchor.  A Triton post-GEMM
kernel that merely follows cuBLAS is an unfused baseline, not the candidate.

Pre-register `B in {1, 8, 32}`, `S in {128, 256}`, both causal and non-causal
attention, BF16, and all nine model geometries.  The declared scope is
9 x 3 x 2 x 2 = 108 rows per GPU.  Baselines are:

1. vendor QKV GEMM + separate Q/K norm+RoPE + vendor attention;
2. the corresponding `torch.compile` path; and
3. the same vendor attention with a separately timed QKV path.

Correctness must compare Q, K, V, attention output, logits, top-1, and NLL to
an eager FP32 reference for both mask modes.  The receipt must expose the
compiler graph or a
kernel-count report showing whether the compiler already fused the operation.
If it did, mark `compiler_already_fused` and retain the row as a boundary
diagnostic; it does not enter the vendor-anchor win rate.

Suggested command after the new implementation is reviewed:

```bash
python src/b_qkv_output_norm_rope.py \
  --models qwen3-8b --batch 1,8,32 --seq 128,256 \
  --out outputs/b_qkv_qwen3_8b.json --trace-dir outputs/b-qkv-traces
```

No B speedup, row count, or expected win rate is currently available.

## C -- public jagged/recommendation fusion (priority 3)

No C runner is present.  The checked-in FuxiCTR DLRM/DCNv2 sources are model
reference code, not a fused kernel benchmark.  The `external/tlx_jfa` source
targets Blackwell and cannot be used as an H100/H200 result.  Do not reuse
private GEM production receipts for this open-model experiment.

Use one public workload and freeze it before implementing the candidate.  The
recommended first lane is TorchRec DLRM-style jagged feature interaction:

* input: BF16 embedding bags represented by `values` and `offsets`, with a
  fixed seed and a checked-in length histogram.  The fixed matrix is feature
  count in `{16, 32}`, embedding dimension in `{64, 128}`, per-feature length
  in `{1, 4, 16, 64}`, and batch in `{1, 8, 32, 128}`; every Cartesian row is
  retained in the CSV;
* candidate: one Triton kernel for jagged concat + per-row LayerNorm (or the
  exact DLRM interaction operation if the reference does not contain LN);
* baselines: eager PyTorch, `torch.compile`, and the vendor embedding/linear
  path; and
* shape sweep: the full pre-registered length histogram plus a small uniform
  control set, with no padding hidden from the timing.

If the DLRM lane cannot be made compiler-independent, use a second, separately
identified DCNv2 lane for two narrow GEMMs whose intermediate is not exposed
to the compiler.  HSTU is a possible follow-up, but it requires an explicit
public commit and dependency lock before any GPU allocation.  C must not mix
lanes in one win-rate denominator.

Each lane needs an independent reference, a jagged-layout checksum, a compiler
fusion report, and a standalone vendor-anchor check.  Check forward outputs and
gradients before timing and perform one real optimizer update where the lane
has trainable parameters.  Start with a CPU/import
smoke and one tiny CUDA shape; only then enqueue the full H100/H200 sweep.

Suggested command once the lane is implemented and pinned:

```bash
python src/c_jagged_recsys.py --lane dlrm_concat_ln \
  --lengths data/dlrm_lengths.json --out outputs/c_dlrm_h100.json \
  --trace-dir outputs/c-dlrm-traces
```

Until that source, histogram, and receipt schema exist, C is `blocked_on_implementation`
and no C number may be written in the paper.

## Public output contract

Every track returns a JSON receipt and a sanitized CSV under `results/abc/`.
The receipt top level must include `schema`, `track`, `status`,
`publication_eligible`, `source_commit`, `source_sha256`,
`config_manifest_sha256`, the GPU/software environment, and the timing
protocol.  Each CSV row includes model/site/shape, seed, correctness metrics,
all arm p50/p95 values, the selected fastest baseline, `speedup_vs_fastest`,
`anchor_ratio`, retest outcome, compiler status, trace hashes, and the
exclusivity verdict.  Include invalid, OOM, unsupported, and
`compiler_already_fused` rows; do not delete them.  Sanitize hostnames, PIDs,
GPU UUIDs, credentials, local paths, and weights before committing.

The source/config hashes in the receipt must match the commit used by the job.
Only receipts with every declared row present, all mandatory gates passed, and
`publication_eligible=true` can update `Experiment.tex`.  The plan contains
no projected speedups and no paper claim is allowed before the receipts are
audited.

## Queue and merge checklist

1. Add the runner and a unit-level correctness smoke to `src/`.
2. Record the source commit in this file and in the receipt before submitting
   an OSU job.
3. Submit one smoke job per lane on an idle H100 and an allowed H200 card;
   enqueue the full sweep only after the smoke passes.
4. Return sanitized JSON/CSV plus traces and private audit evidence.  Merge
   only source, public configs, and sanitized results to `origin/main`.
5. Update `Experiment.tex` only from promoted rows.  Keep rejected and
   compiler-already-fused rows in the appendix or diagnostic archive.

Current status at handoff: A has a reusable kernel but needs the model-shaped
census runner; B and C require new implementations.  There are no B/C results
in this repository.

## Exact manifests and row accounting

The A and B model manifest is fixed before the first full run.  Unless a public
configuration is unavailable in the pinned release, use these nine families:

```
Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-32B,
Qwen2.5-0.5B-Instruct, OLMo-1B, Llama-3.1-8B, Mistral-7B
```

Resolve `hidden_size`, `num_attention_heads`, `num_key_value_heads`,
`head_dim`, `intermediate_size`, and the norm epsilon from the public config.
Write the resolved values and the config-file SHA256 to a manifest before
allocating a GPU.  Do not substitute a convenient shape for a model whose
public config is missing; record `blocked_dependency` for that model instead.

For A, each model has two sites (QKV and gate/up) and seven token counts
`M={1,8,32,128,512,2048,8192}`: 126 attempted rows per GPU before OOM or
dependency statuses.  The QKV output width is
`(num_attention_heads + 2*num_key_value_heads)*head_dim`; gate/up is twice the
public intermediate size.  An OOM is a row with `status=oom`, not a row that
may be removed from the denominator.

For B, each model has `B={1,8,32}`, `S={128,256}`, and both causal and
non-causal masks: 108 attempted rows per GPU.  A row is
`compiler_already_fused`, `blocked_dependency`, or `tested_invalid` when its
gate fails, but it remains in the CSV.  The promoted win count is computed
only over rows with `publication_eligible=true` and an unfused compiler graph.

For C, the first lane is a shape-only public DLRM-style workload, not a private
production trace.  Freeze this matrix in `data/c_dlrm_manifest.json`:

```
feature_count in {16,32}
embedding_dim in {64,128}
per_feature_length in {1,4,16,64}
batch in {1,8,32,128}
```

This gives 64 rows per C fusion site before unsupported/dependency statuses.
The manifest stores deterministic flattened indices, offsets, and a histogram
checksum.  The two pre-registered sites are (i) jagged concat plus row
LayerNorm and (ii) the adjacent narrow-MLP GEMM pair.  If the chosen public
module does not contain a legal second site, every such row is
`unsupported_shape`; do not replace it with a dense synthetic GEMM.

## Arm names and promotion formulas

Every receipt uses the same arm names so that an offline script can recompute
the claim:

```
candidate
unfused_vendor_anchor
torch_compile_default
torch_compile_max_autotune
cublas_or_cublaslt
```

The last arm is required for A and for every GEMM-containing B/C row.  Let
`anchor_ratio = vendor_anchor_p50 / cublas_p50`; an anchor is vendor-quality
only when it is no slower than cuBLAS by more than one percent, or when a
documented vendor library arm is faster and becomes the declared gate.  Let
`fastest_baseline_p50` be the minimum of all valid baseline medians, and define

```
speedup_vs_fastest = fastest_baseline_p50 / candidate_p50
win_vs_fastest = speedup_vs_fastest > 1 + noise_margin
```

`noise_margin` comes from an A/A receipt on the same physical GPU, dtype,
shapes, and batch/token sizes.  A fixed one-percent margin must not be used
without that A/A measurement.  All initial candidate ratios above 1.01 are
repeated in a fresh paired retest; the retest, not the optimistic initial
sample, is used for the claim.

The compiler gate is a graph report or kernel-count report captured from the
same released software stack.  If `torch.compile` has already fused the
operation, mark the row `compiler_already_fused`; it can illustrate a boundary
case but cannot increase the vendor-anchor win rate.  A candidate that merely
beats eager Python while losing to compiled or vendor code is a regression.

## Implementation notes for the sister's Claude session

* Start by reading this file, `README.md`, and the relevant public config
  manifest.  Do not search private session transcripts for replacement
  numbers.
* B is not the existing `e1_rope_*` attention-prologue kernel.  Implement a
  QKV GEMM output tile that writes Q/K/V, applies Q/K RMSNorm and split-half
  RoPE to Q/K, then calls vendor SDPA.  The custom GEMM must pass the standalone
  cuBLAS anchor gate before any B row can be promoted.
* C is not the Blackwell-only `external/tlx_jfa` implementation.  First run a
  CPU/import smoke for the pinned public DLRM/DCNv2 dependencies and one tiny
  CUDA shape.  If the dependency or compiler graph is unavailable, return a
  complete `blocked_dependency` manifest rather than changing the workload.
* A must distinguish lazy RMSNorm from LayerNorm.  Fold gamma into the weight
  once, compute the per-row inverse RMS factor in the fused kernel, and never
  silently add a mean/bias term.
* Do not upload weights, private paths, hostnames, PIDs, GPU UUIDs, raw traces,
  or company logs.  Keep those in the private audit bundle and commit only
  sanitized receipts, manifests, summaries, and source.

No estimated speedup, projected win count, or expected percentage belongs in
`results/` or `Experiment.tex` until the complete receipt passes these gates.
