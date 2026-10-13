# Sanitized result data

The CSV files contain only the fields needed to inspect the published ratios.
They do not contain model weights, raw receipts, profiler traces, hostnames,
GPU UUIDs, process IDs, or local filesystem paths. The authoritative raw
receipts remain in the authors' private campaign storage.

## Files

- `e1_qk_speedups.csv`: 144 paired Q/K-normalization configurations for each
  OSU device. `speedup` is the initial unfused/fused median ratio and
  `retest_speedup` is populated when the independent retest ran.
- `e3a_h200_speedups.csv`: H200 expert-local MoE forward/backward ratios. It is
  an isolated module study and does not include routing or a parameter update.
- `e4_residual_speedups.csv`: 32 corrected model-shaped residual-epilogue
  configurations for each OSU device. The vendor comparator is explicit
  `torch.addmm` (cuBLAS/cuBLASLt).
- `e7_aa_noise_floor.csv`: three paired A/A rows per device. These rows are a
  noise-floor sample, not a GPU-wide confidence interval.
- `e8_failure_cases.csv`: H200 dense-RMSNorm and SwiGLU failure-case rows with
  plain/fused/vendor p50 ratios. These are isolated forward-only studies.

All promoted OSU E1, E4, and E7 rows pass their stated correctness, trace, and
GPU-exclusivity gates. Values above one mean the fused arm is faster. The
operator-level results do not imply model-level or end-to-end acceleration.

## Aggregates

Counts use ratios strictly greater than one. E1 geometric means are over all
144 configurations. E4 means and ranges in parentheses are over the faster
configurations, matching the manuscript table.

| Experiment / comparison | Device | Faster | Geometric mean | Speedup range |
|---|---|---:|---:|---:|
| E1 Q/K RMSNorm fusion | OSU H100 | 74/144 | 1.0103x (all) | 0.5566–1.5757x |
| E1 Q/K RMSNorm fusion | OSU H200 | 67/144 | 0.9710x (all) | 0.6843–1.3365x |
| E4 residual epilogue vs plain anchor | OSU H100 | 30/32 | 1.0722x (wins) | 1.0028–1.1270x |
| E4 residual epilogue vs plain anchor | OSU H200 | 32/32 | 1.0433x (wins) | 1.0085–1.1007x |
| E4 residual epilogue vs cuBLAS `addmm` | OSU H100 | 0/32 | 0.7921x (all) | 0.5545–0.9863x |
| E4 residual epilogue vs cuBLAS `addmm` | OSU H200 | 2/32 | 0.8220x (all) | 0.6003–1.0455x |

The old BABEL measurements are excluded from this repository and from the
current claim boundary. The Gemma end-to-end prefill candidate is withheld
because its model-level correctness gate did not pass.
