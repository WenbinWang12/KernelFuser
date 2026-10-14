# Sanitized result data

The CSV files contain only fields needed to inspect the reported ratios. They
contain no weights, raw receipts, profiler traces, hostnames, GPU UUIDs,
process IDs, or private paths.

## Files

- `h100_attention_refined.csv`: 168 final H100 attention selections and
  fingerprints of the relocated kernel definitions.
- `h100_pointwise_gemm_refined.csv`: eight H100 SwiGLU/RMSNorm-GEMM comparisons.
- `h100_residual_refined.csv`: three H100 residual-GEMM independent retests.
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

## H100 fusion refinement (2026-10-13)

Measured on H100 on 2026-10-13 with Triton 3.8 development build, PyTorch
2.15 development build and CUDA 12.8. The requested power cap was 500 W;
actual clocks could be power limited. Compare timings within each paired experiment.
Historical ratios identify earlier regressions; historical latencies are not
mixed with new measurements.

### Results

| Attention proxy | Cases | Repeated gains >1% vs improved unfused | Previous losing cases recovered |
|---|---:|---:|---:|
| Qwen3 norm + RoPE | 24 | 22 | 15/15 |
| Legacy no-RoPE | 144 | 143 | 70/70 |
| Total | 168 | 165 | 85/85 |

The tables call a win `confirmed_1pct` only when the speedup exceeds 1.01
in both tuning and an independent retest. This is a repeatability threshold,
not a statistical confidence interval. Both categories received a tile
search; their selected tiles can differ. This compares the best tested
implementations rather than attributing every gain exclusively to fusion.

The two B=1, S=128 RoPE cases favor the improved unfused implementation.
Gemma-3-4B noncausal S=128 has a marginal 1.16% retest gain but only 0.98%
during tuning, so it is not counted as confirmed. All 85 previously losing
attention cases exceed the threshold after the changes.

The earlier four SwiGLU and four gamma-free RMSNorm/GEMM cases retain their
verified gains. Together with RoPE attention, the current 32-case subset
has 30/32 repeated gains over 1%. This is distinct from the broader
168-case attention census.

### Causes and implementation changes

#### Attention: repeated work and Hopper scheduling

Full Q/K fusion recomputes K normalization and RoPE in every query tile and
every grouped query head. At Qwen3-8B noncausal S=512, BM=32 and Hq/Hkv=4,
each K row is transformed 64 times. `src/e1_rope_publication_v1.py` prepares K once,
then fuses Q normalization and optional RoPE into attention. Both query
fusion and its unfused control use two kernels.

BM=64 enables WGMMA in the measured compiler. BF16 dot inputs with FP32
accumulation, exp2 online softmax, and bounded causal loops improve the
schedule. The original legacy causal kernel scanned the entire sequence.

For B=8, S=512 noncausal RoPE attention, the reproduced old fused kernel
took 534.378 us; query fusion took 146.715 us and improved unfused took
176.547 us in the same paired retest. That is 3.642x versus old fused and
1.203x versus improved unfused.

#### D256 without RoPE: factor Q normalization

The boundary-preserving query prologue still loses in many D256 cases.
With gamma vectors shared across heads, the mathematical operation can
instead be written as:

```text
Qnorm = Q * inverse_rms(Q) * gamma_Q
Knorm = K * inverse_rms(K) * gamma_K
Qnorm @ Knorm.T = inverse_rms(Q) * (Q @ (Knorm * gamma_Q).T)
```

`src/e1_qk_publication_v3.py` prepares K with both gamma vectors, computes
one FP32 inverse-RMS value per Q row, and applies that scalar to attention
scores. Q scratch drops from 512 bytes to 4 bytes per D256 row, a 128x
reduction. Q's reduction is removed from the attention prologue.

All 32 D256 retests favor factoring; 31/32 exceed 1% in both phases.
Speedups versus improved unfused range from 1.012x to 1.121x. Three-seed
relative-L2 error stays below 0.00284 versus the original 0.025 gate.

This preserves the mathematical FP32 operation but changes intermediate
BF16 rounding. It applies to the tested no-RoPE proxy. The original
boundary-preserving path remains available when those intermediate dtype
boundaries are required. The low-level launcher requires preallocated
BF16 output/K scratch and FP32 inverse-RMS scratch `[B,Hq,S]`.

#### Residual GEMM: a better tile, no repeated fusion-only win

For M2048/K25600/N5120, reducing BN from 256 to 128 with BM128/BK64,
8 warps and 3 stages reduces registers from 206 to 116 and shared memory
from 147,456 to 98,304 bytes. Independent retest latency improves from
1296.483 to 1173.226 us, a 1.105x gain.

Equally tuned unfused is 1169.699 us: the fusion-only difference is within
0.3%, and the tuning gain did not repeat. Vendor `addmm` is 823.502 us,
1.425x faster than improved custom fused. The other historical residual
loss, 0.29% at M128/K28672/N8192, also did not reproduce as a meaningful
regression. The `gemm_residual` API in `src/epilogue_sites.py` therefore
defaults to vendor `addmm` and offers explicit Triton schedules for only
the three verified shapes.

### Profiling and unsuccessful ablations

| Direct NCU profile | Registers/thread | Shared memory/block | Theoretical occupancy | Achieved occupancy |
|---|---:|---:|---:|---:|
| Old attention B8/S512 | 238 | 28,672 B | 12.50% | 12.27% |
| Query fusion BM64/BN128 | 255 | 49,152 B | 12.50% | 12.07% |
| Original residual tile | 206 | 147,456 B | 12.50% | 12.48% |
| Smaller residual tile | 116 | 98,304 B | 25.00% | 22.80% |

Attention improves with essentially unchanged occupancy. Removing repeated
K work and changing the schedule explain the gain; higher occupancy is not
its mechanism. The smaller residual tile permits two resident CTAs instead
of one, but twice the occupancy does not imply twice the speed.

Additional stages and gathers can regress performance. At query BM64/BN128,
D128, a second stage triples shared memory and slows B8/S512 noncausal from
146.773 to 180.685 us in tuning. For D256 it exceeds the per-CTA shared-memory
limit. Gathering the rotary partner can increase layout conversions and
static barriers; a BM32 full-fusion ablation grew from 18 to 41 barriers.
Both reload and gather remain available for experiments.

The attention changes use plain Triton. The gamma-free GEMM implementation
also includes a TLX async-load/async-dot pipeline. TLX provides scheduling
controls; it does not guarantee that repeated work or a resource-heavy
schedule becomes faster. These NCU sections establish occupancy and launch
resources, not Tensor Core utilization.

### Tables and verification

- [Attention: all 168 final selections](h100_attention_refined.csv)
- [SwiGLU and RMSNorm/GEMM: eight cases](h100_pointwise_gemm_refined.csv)
- [Residual GEMM: three independent retests](h100_residual_refined.csv)

Attention retests use five fresh CUDA-graph rounds, seven randomized samples
per round and twenty launches per sample. Preparation kernels are included.
Numerical verification covers three seeds, sequence tails and public APIs.
Attention retains allclose atol/rtol=.025 and relative-L2<=.025; residual
retains atol/rtol=.05, including additional seeds where GEMM and residual
have comparable magnitude. SwiGLU/GEMM retain their earlier verification
gates. GPU exclusivity and source hashes are recorded in local receipts.

Custom attention uses preallocated output buffers. Supplementary native
SDPA controls allocate graph outputs per captured call, so they have a
different output-buffer policy and are not the basis of the counts above.
The SwiGLU baseline is eager FP32 pointwise arithmetic with BF16 I/O;
the GEMM baseline uses cuBLAS with FP32 output, separate row scaling and
BF16 conversion. These are not universal claims against every vendor or
compiled implementation.

These are isolated forward attention/GEMM proxies, plus isolated SwiGLU
forward/backward kernels. They do not establish full-model speedups,
KV-cache behavior, backward attention, or the publication trace gate.
The no-RoPE study retains the historical D**-.5 scaling, including its
Gemma-3-27B proxy; it does not implement Gemma's complete model semantics.
Raw receipts, profiles and host identifiers remain outside the repository.

### Integration into the existing source files

The optimized kernel bodies are unchanged after relocation. The attention
CSV's `kernel_ast_sha256` fingerprints the relevant JIT definitions, so
moving code between files does not invalidate the implementation identity.
The numerical/performance tables were measured before relocation; focused
integration checks verify the relocated APIs and existing runner modes.
Only the three result CSVs are new files; the implementations, verification
entry points, README and this results document update existing files.
