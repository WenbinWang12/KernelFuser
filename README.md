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
- `results/`: sanitized CSV summaries only.
- `experiments/RUN_REQUESTS.md`: current OSU P0/P1 handoff and fail-closed
  acceptance gates.

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
python src/e1_rope_publication_v1.py \
  --output outputs/e1-rope.json --trace-dir outputs/e1-rope-traces
python src/e1_qk_publication_v3.py --output outputs/e1-legacy.json
python src/e7_aa_noise_floor_osu_v3.py \
  --out outputs/e7-aa.json --trace-dir outputs/e7-aa-traces
```

Generated receipts and traces can contain host identifiers and local paths;
they are ignored by the public artifact policy and should be sanitized before
sharing. A passing correctness check alone is not a speedup claim; the timing,
retest, trace, and exclusivity gates must also pass.

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

## License

First-party code is MIT unless a directory states otherwise. The ADS-derived
SM90 fallback carries its Apache-2.0 attribution notice. PyTorch, Triton,
`fbtriton`, CUDA, and NVIDIA tooling remain separate dependencies.


