# Sanitized result data

The CSV files are derived from the strict JSON receipts retained by the
authors. Raw receipts, trace archives, hostnames, GPU UUIDs, process IDs, and
local filesystem paths are deliberately excluded from this public repository.

## CSV schemas

- `e1_qk_speedups.csv`: one row per device/model/sequence-length/causal-mode
  configuration. `speedup` is the initial five-round paired median ratio
  (unfused divided by fused); `retest_speedup` is populated only when the
  receipt required an independent retest.
- `e3a_h200_speedups.csv`: one row per H200 model/token-count configuration.
  Forward and backward columns contain mean latency in milliseconds and
  baseline-to-candidate ratios. `vendor_grouped` is included as throughput
  context only and must not be compared to a single-expert arm.
- `e4_residual_speedups.csv`: one row per device/projection geometry/M.
  `speedup` uses the independent retest ratio when one was recorded, otherwise
  the initial ratio. Geometry dimensions are synthetic and do not identify a
  complete model.

## Coverage

- E1: 144 configurations per device on H100 and H200; all recorded speedups
  are preserved, including values below 1.
- E3a: H200 only (36 configurations). H100 rows are omitted because the
  current local audit does not promote that device's run.
- E4: 16 residual-epilogue configurations per device on H100 and H200.

The values are operator-level measurements. They do not imply model-level or
end-to-end acceleration.

The repository also publishes first-party Hopper TLX/Triton SwiGLU backends
and an ADS-attributed SM90 RMSNorm+GEMM fallback. Those kernel sources are not
included in the E1, E3a, or E4 result tables above; the isolated pilot scripts
are provided for independent correctness and timing runs, not as additional
published aggregate claims.

## Aggregates

Geometric means are computed over the listed per-configuration speedup ratios;
“wins” counts ratios strictly greater than 1. The E1 values below use the
initial paired ratios; independent retest ratios remain in the CSV.

| Experiment / comparison | Device | Wins | Geometric mean | Speedup range |
|---|---|---:|---:|---:|
| E1 Q/K RMSNorm fusion | H100 | 58/144 | 0.9361x | 0.6662–1.2927x |
| E1 Q/K RMSNorm fusion | H200 | 67/144 | 0.9710x | 0.6843–1.3365x |
| E3a in-loop / unfused, forward | H200 | 3/36 | 0.5631x | 0.2990–1.0501x |
| E3a staged / unfused, forward | H200 | 3/36 | 0.5708x | 0.3004–1.0790x |
| E3a in-loop / unfused, backward | H200 | 0/36 | 0.2555x | 0.1392–0.6050x |
| E3a staged / unfused, backward | H200 | 0/36 | 0.2556x | 0.1391–0.6071x |
| E4 residual epilogue | H100 | 16/16 | 1.0630x | 1.0010–1.1312x |
| E4 residual epilogue | H200 | 16/16 | 1.0437x | 1.0153–1.0945x |
