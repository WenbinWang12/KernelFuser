# Sanitized result data

The CSV files contain only fields needed to inspect the reported ratios. They
contain no weights, raw receipts, profiler traces, hostnames, GPU UUIDs,
process IDs, or private paths.

## Files

- `e1_qk_speedups.csv`: legacy isolated Q/K RMSNorm rows (144 per device,
  batch 1, no RoPE).
- `e1_rope_qwen8_short.csv`: first-pass diagnostic RoPE-aware Qwen3-8B rows.
- `e1_rope_qwen8_v2.csv`: sanitized strict v2 Qwen3-8B isolation rows (24/24 rows pass per GPU; only two batch-one S=128 rows beat the strongest arm on each GPU).
- `e7_rope_aa_h100_h200.csv`: sanitized six-row-per-device short-shape A/A controls.
- `e3a_h200_speedups.csv`: expert-local MoE forward/backward ratios.
- `e4_residual_speedups.csv`: model-shaped residual-epilogue rows with an
  explicit `torch.addmm` vendor comparator.
- `e7_aa_noise_floor.csv`: seven paired A/A rows per device, including short
  sequences and prefill-shaped controls.
- `e8_failure_cases.csv`: H200 dense RMSNorm and model-shaped Llama3-8B /
  Qwen2.5-3B SwiGLU rows with plain/fused/vendor ratios.

Values above one mean the fused arm is faster. E1, E1-RoPE, E4 and E7 are
operator-level studies and do not imply end-to-end acceleration. The legacy
Gemma v2 receipts were withdrawn; the active model-level follow-up is the
Qwen3-8B short-sequence RoPE path listed in `experiments/RUN_REQUESTS.md`.

## Current gates

The first E1-RoPE Qwen3-8B sweep is diagnostic because its compiled/vendor arms were
not independently candidate-paired. The v2 runner pairs the candidate with each
baseline, retimes every row, records fresh retests, monitors the selected GPU, and
retains four traces per row. H100 and H200 v2 are complete.

The Qwen3-8B model-prefill smoke stopped before timing because the compiled and
eager vendor baselines failed the full-logit/top-1 agreement gate on both GPUs.

The residual epilogue is faster than its Triton anchor in 30/32 H100 and
32/32 H200 rows, but it beats explicit cuBLAS `addmm` in only 0/32 and 2/32;
it remains an Appendix C diagnostic.


