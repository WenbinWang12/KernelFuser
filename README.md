# KernelFuser

Scoped benchmark code, reusable Triton/TLX kernels, and sanitized result data
for the kernel-fusion study. The repository intentionally excludes manuscript
working files, model weights, raw receipts, profiler traces, credentials,
session transcripts, and private paths.

## Contents

- `src/e1_rope_publication_v1.py`: RoPE-aware Q/K RMSNorm+RoPE isolated runner
  with FP32 oracle checks, paired compiled/vendor arms, fresh retests, GPU
  snapshots, and compressed Chrome traces.
- `src/e1_qk_publication_v3.py`: legacy B=1, no-RoPE Q/K RMSNorm census.
- `src/e3a_moe_publication.py`, `src/moe_staged_strict.py`, and
  `src/norm_into_gemm_kernel.py`: expert-local MoE studies.
- `src/e4_model_shapes_publication_v2.py`: model-shaped residual epilogue
  runner with explicit `torch.addmm` comparator.
- `src/e7_rope_aa.py`: six-shape Qwen3-8B RoPE A/A noise-floor runner.
- `src/e7_aa_noise_floor_osu_v3.py`: supplementary seven-row A/A controls.
- `src/e2_qwen3_rope_prefill.py`: Qwen3-8B model-prefill correctness/timing
  runner; no timing row is emitted when the baseline agreement gate fails.
- `src/e2_gemma_forward_v5_replay.py`: replay-first Gemma candidate retained
  for audit; its withdrawn v2 receipts are not results.
- `src/swiglu/`, `src/tlx_pilot/`, and `src/tlx_ads_sm90_port/`: reusable
  pointwise/TLX experiments with their own scope notes.
- `src/a_tlx_rmsnorm_gemm_model_census.py`: pinned public-model lazy
  RMSNorm/projection census with TLX dependency, vendor, compiler and trace gates.
- `results/`: sanitized CSV summaries and ABC JSON receipts.
- `experiments/RUN_REQUESTS.md`: current OSU P0/P1 handoff and fail-closed
  acceptance gates.
- `experiments/ABC_VENDOR_ANCHOR_PLAN.md`: pre-registered A/B/C follow-up
  campaign.  B and C are implementation tasks only until their receipts pass
  the vendor-anchor and correctness gates; this file contains no results.

## Reproduction

The runners require Linux, CUDA, PyTorch, Triton, and a supported NVIDIA GPU.
The legacy E1 H100/H200 receipts use different toolchains (H100 PyTorch
2.11.0+cu128/Triton 3.6.0; H200 PyTorch 2.5.1/CUDA 12.4/Triton 3.1.0), so
ratios are interpreted within each device. The updated Qwen3-8B RoPE runner
passes the strict semantic smoke on both devices and the full 24-row isolation
receipt on both devices. The Qwen3-8B model
prefill smoke currently stops at the compiled-versus-eager agreement gate and
has no timing claim.

```bash
mkdir -p outputs
CUDA_VISIBLE_DEVICES=1 python src/e1_rope_publication_v1.py --fresh-processes \
  --output outputs/e1-rope.json --trace-dir outputs/e1-rope-traces
CUDA_VISIBLE_DEVICES=1 python src/e1_qk_publication_v3.py --fresh-processes \
  --output outputs/e1-legacy.json
CUDA_VISIBLE_DEVICES=1 python src/e7_rope_aa.py \
  --output outputs/e7-rope-aa.json --trace-dir outputs/e7-rope-aa-traces
CUDA_VISIBLE_DEVICES=1 python src/e7_aa_noise_floor_osu_v3.py --fresh-processes \
  --out outputs/e7-aa.json --trace-dir outputs/e7-aa-traces
```

Select an idle physical card instead of assuming index 1 is free. E1's
`--fresh-processes` mode uses a new Python/CUDA process for each shape and
aggregates the original per-case protocol. This avoids the CPU-only profiler
captures observed after repeated shape runs in one process. Without the flag,
the original single-process census mode remains available.

Generated receipts and traces can contain host identifiers and local paths;
they are ignored by the public artifact policy and should be sanitized before
sharing. A passing correctness check alone is not a speedup claim; the timing,
retest, trace, and exclusivity gates must also pass.

## ABC H100 follow-up

The requested baseline is commit `84b0adc`. That commit adds the ABC plan,
with reusable A kernels but no model census runner and no B/C implementation.
The follow-up adds the A runner and public configs. Its release dependency
check retains all 126 rows when TLX is unavailable; no plain Triton result is
relabeled as TLX. B/C readiness receipts retain their full planned matrices.

Run the A dependency/correctness smoke before the full census:

```bash
python src/a_tlx_rmsnorm_gemm_model_census.py \
  --models Qwen3-8B --sites qkv,gate_up --m 128 --smoke \
  --out outputs/a-smoke.json
python src/a_tlx_rmsnorm_gemm_model_census.py \
  --device-tag h100 --out outputs/a-full.json
```

`--allow-development` explicitly selects diagnostic results on a development
stack. Those results cannot update the release-stack or paper win rate.
Operator timings include dispatch and fresh output allocation. The candidate
reduces inverse RMS and performs GEMM in one GPU kernel, with no normalized
activation or global scalar prepass. The standalone
GEMM gate uses CUDA events around identical graph blocks for both arms and
also retains their normal operator timings. Gamma is folded into synthetic
weights once; model weights are not downloaded.

E1 trace capture now checks for GPU kernel events and retains failed captures
before retrying. A CPU-only trace is rejected even when its file is nonempty.
The requested E1 kernel math and per-case timing protocol are unchanged.
The dated rerun uses the default README attention proxies at `84b0adc`,
separately from the earlier `--refine` results. These proxies do not implement
the QKV-GEMM output fusion required by ABC B.

See [the dated H100 rerun](results/RESULTS.md#h100-abc-follow-up-2026-10-15)
for coverage, blockers and diagnostic comparisons.

## Claim boundaries

The legacy E1 and new E1-RoPE runners are isolated forward-only operator
studies. The Qwen3-8B E1-RoPE receipt is not an end-to-end model result. E3a
is expert-local and has no routing or parameter update. E4 remains an appendix
diagnostic against the vendor gate. The active model-level request is the
Qwen3-8B short-sequence prefill described in `RUN_REQUESTS.md`.

## H100 fusion refinements

The existing implementations now include the H100 refinements. See
[results and limitations](results/RESULTS.md#h100-fusion-refinement-2026-10-13)
and the sanitized CSVs in `results/`.

- `src/e1_rope_publication_v1.py`: `optimized_attention` prepares K once
  and fuses Q normalization/RoPE into attention. Explicit query/unfused
  modes use BM64 with BN64/BN128; dtype, GQA and sequence tails are checked.
- `src/e1_qk_publication_v3.py`: `launch_factored_attention` folds Q gamma
  into K and Q's inverse RMS into scores for the no-RoPE proxy. This reduces
  Q scratch by 128x for D256, with different intermediate BF16 rounding.
- `src/tlx_ads_sm90_port/ads_sm90_rmsnorm_matmul.py`: `rmsnorm_gemm`
  adds row scaling in the GEMM epilogue, with a TLX async-load/async-dot
  pipeline and a masked Triton fallback for odd strides.
- `src/tlx_pilot/tlx_swiglu_hopper.py`: `launch_swiglu` adds the measured
  single-task candidate and explicit arithmetic/rounding controls.
- `src/epilogue_sites.py`: `gemm_residual` offers measured H100 schedules;
  vendor `addmm` remains the fastest measured default.
- `src/fusion_harness.py`: shared randomized device timing and compiler
  evidence utilities used by the existing runners.

Use the new refinement flags for tuning and independent verification:

```bash
python src/e1_rope_publication_v1.py --tune --out outputs/attention-tune
python src/e1_rope_publication_v1.py --refine --out outputs/attention-refine
python src/e1_qk_publication_v3.py --refine \
  --census outputs/attention-refine/results.json --out outputs/attention-factored
python src/e4_model_shapes_publication_v2.py --tune --out outputs/residual-tune
python src/e4_model_shapes_publication_v2.py --refine --out outputs/residual-refine
python src/e8_forward_publication.py --h100-unfused --out outputs/pointwise-gemm
```

The plain attention/residual refinement modes do not require TLX. The
SwiGLU/GEMM experiments require a TLX-enabled Triton build. These are
isolated operator tests; original runner modes and their publication gates
remain available. Raw receipts and profiler artifacts stay in ignored
local output directories. The CSVs contain only sanitized performance
fields and repository-relative source names.

## B200 ADS Blackwell snapshot diagnostics

The checked-in B200 CSVs are **snapshot diagnostics**: all record
`ads_matches_pinned_public=False`. Their timings, repeatability counts and
M128 ablation apply only to the measured snapshot. Public-revision claims
require an exact rerun of commit `2889aefba0d03b3eecf779eed9f4097066ecdc5d`
or an attached explicit source-diff audit. Neither is currently attached.

The existing E8 runner can compare the original ADS Blackwell JIT with the
improved plain Triton implementation and separate vendor controls. Supply
the ADS source separately; the runner verifies the source fingerprint from
the pinned public revision cited in the ADS port's `NOTICE`.

```bash
python src/e8_forward_publication.py --b200-ads \
  --ads-source /path/to/ads_model_kernel_library/multi_cta_norm_fusion/src/tlx_rmsnorm_matmul.py \
  --out outputs/b200-ads --csv results/b200_ads_rmsnorm_matmul.csv
```

For another ADS revision, explicitly use `--allow-source-version` and a
distinct CSV filename. The CSV records the source fingerprint and whether
it matches the pinned version. No ADS source is copied into this repository.
CSV exports include `evidence_status` and `publication_eligible`; a passing
snapshot measurement exports as `snapshot_diagnostic` with
`publication_eligible=False`. Correctness and timing gates alone do not
validate public-source provenance.

The bounded schedule search keeps the supplied ADS JIT bodies unchanged.
It includes an eight-subtile RMSNorm reduction schedule to avoid excessive
shared loads and reductions on short GEMMs. The M128 diagnosis, controlled
ablations and follow-up results are in [results/RESULTS.md](results/RESULTS.md).
All timed arms preallocate outputs and scratch. Each case receives three
correctness seeds, randomized CUDA-graph timings and an independent retest;
sequence tails and selected-GPU exclusivity are checked before CSV export.

The strict norm-before-cuBLAS control normalizes in FP32, uses FP32 cuBLAS,
then casts the output to BF16. Its FP32 weight copy is prepared outside
timing. A second control uses BF16 cuBLAS with FP32 output and separate row
scaling/casting. The main speedup columns compare against the faster of
these two controls. Normalizing to BF16 before cuBLAS introduces a different
rounding boundary; its FP32 correctness result is retained as a diagnostic.

## License

First-party code is MIT unless a directory states otherwise. The ADS-derived
SM90 fallback carries its Apache-2.0 attribution notice. PyTorch, Triton,
`fbtriton`, CUDA, and NVIDIA tooling remain separate dependencies.
