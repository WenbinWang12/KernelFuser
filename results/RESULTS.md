# Sanitized result data

The CSV files contain only fields needed to inspect the reported ratios. They
contain no weights, raw receipts, profiler traces, hostnames, GPU UUIDs,
process IDs, or private paths.

## Files

- `b200_ads_blackwell_snapshot.csv`: snapshot diagnostics for ten B200
  shapes, improved Triton, and separate vendor controls.
- `b200_ads_blackwell_tuned.csv`: snapshot diagnostics for a fresh ten-shape
  comparison after adding the eight-subtile reduction candidate.
- `b200_ads_m128_ablation.csv`: snapshot diagnostics for sixteen controlled
  M128 variants, paired timings, gates and sanitized compiler/profile metrics.
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

## B200 ADS Blackwell snapshot diagnostics (2026-10-13)

**Evidence status: snapshot diagnostics; public-source validation pending.**
All three B200 CSVs record `ads_matches_pinned_public=False`,
`evidence_status=snapshot_diagnostic` and `publication_eligible=False`.
The repeatability counts and causal ablation below apply to the measured
snapshot. An exact public-commit rerun or an explicit source-diff audit must
be attached before using them to support claims about the pinned public
implementation. Neither artifact is currently attached.

A rerun must record the exact public commit and source hash alongside the
existing correctness gates and paired timing protocol. A source-diff audit
must identify both source versions, compare the JIT bodies, numerical
semantics and launch/configuration changes, and state which performance
claims the differences support. A fingerprint alone is not a source-diff audit.

The measured ADS Blackwell source snapshot has SHA-256
`d23832b4d5b964848da5482079f03a3102b84a31175bde4f8eab9654b6af7f79`.
It differs from the pinned public source cited in the SM90 port's `NOTICE`.
These measurements establish results for this snapshot, not for that public
revision. The CSV explicitly records `ads_matches_pinned_public=False`.
The pinned revision is `2889aefba0d03b3eecf779eed9f4097066ecdc5d`, with
source SHA-256
`26f66c287a957fa324eca24473509823b84779111acd05b07d0ea10061de4a29`.
Hash mismatch alone does not establish which source changes affect timing.
No ADS source is added to this repository. Its JIT bodies are unchanged;
the runner selects explicit schedules and preallocates the descriptors and
output, then checks agreement with the original ADS wrapper.

Measured on NVIDIA B200, 148 SMs, with PyTorch 2.15 development build,
Triton 3.9 development build and CUDA 12.8. The selected GPU used an 850 W
power cap and requested 1965 MHz graphics clock. Clock limits were released
after the run. Compare ratios within this run; the H100 compiler version
differs. Inputs are synthetic unscaled Gaussian BF16 matrices, with zero,
small-constant and unit-constant rows added to correctness checks.

### Original six-configuration search: independent retest results

| M | K | N | ADS TLX (us) | Improved Triton (us) | Fastest vendor control (us) | ADS/vendor |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1024 | 3072 | 7.894 | 7.200 | 7.309 | 0.926x |
| 512 | 1024 | 3072 | 9.222 | 9.472 | 9.867 | 1.070x |
| 512 | 1024 | 4096 | 9.490 | 9.678 | 10.878 | 1.146x |
| 2048 | 1024 | 4096 | 16.613 | 23.606 | 25.163 | 1.515x |
| 8192 | 1024 | 4096 | 61.098 | 87.706 | 96.685 | 1.582x |
| 16384 | 1024 | 4096 | 116.286 | 163.939 | 190.470 | 1.638x |
| 65536 | 512 | 512 | 39.952 | 67.350 | 112.029 | 2.804x |
| 131072 | 512 | 512 | 74.875 | 140.870 | 218.826 | 2.923x |
| 524288 | 512 | 512 | 309.518 | 531.584 | 838.754 | 2.710x |
| 262144 | 512 | 2048 | 563.843 | 1107.363 | 1252.854 | 2.222x |

The tested ADS snapshot has gains over 1% in both phases for 9/10 shapes,
and is faster than improved Triton for those same nine shapes. Improved
Triton has 9/10 confirmed gains over the fastest vendor control. Its M128
gain is 0.77% initially and 1.51% in retest, so it is not confirmed.

The fastest vendor control for every row is separate inverse RMS, BF16
cuBLAS with FP32 output, then FP32 row scaling and a BF16 cast: three GPU
kernels. The explicit norm-before-cuBLAS control also uses three kernels:
FP32 RMSNorm, FP32 cuBLAS and BF16 output conversion. Its preconverted FP32
weight copy is outside timing. That control ranges from 33.131 to
11779.289 us; its per-row timings and speedups are included in the CSV.
Reported diagnostic ratios use the faster vendor control. `confirmed_1pct`
denotes repeated timing gains only; it does not confirm public-source identity.

### Explicit unfused norm-before-GEMM comparison

This baseline materializes FP32 RMSNorm output, runs FP32 cuBLAS, then
converts the GEMM output to BF16. The timings include all three kernels.

| M / K / N | Unfused RMSNorm + cuBLAS (us) | ADS/unfused | Triton/unfused |
|---|---:|---:|---:|
| 128 / 1024 / 3072 | 33.131 | 4.197x | 4.602x |
| 512 / 1024 / 3072 | 79.192 | 8.587x | 8.361x |
| 512 / 1024 / 4096 | 90.949 | 9.584x | 9.397x |
| 2048 / 1024 / 4096 | 349.534 | 21.040x | 14.807x |
| 8192 / 1024 / 4096 | 1301.157 | 21.296x | 14.836x |
| 16384 / 1024 / 4096 | 2529.078 | 21.749x | 15.427x |
| 65536 / 512 / 512 | 734.944 | 18.396x | 10.912x |
| 131072 / 512 / 512 | 1489.472 | 19.893x | 10.573x |
| 524288 / 512 / 512 | 6432.751 | 20.783x | 12.101x |
| 262144 / 512 / 2048 | 11779.289 | 20.891x | 10.637x |

Both candidates exceed 1% in both independent phases for all ten shapes
against this explicit unfused baseline. These larger ratios also include
the difference between FP32 cuBLAS arithmetic and BF16 Tensor Core GEMM.
They do not isolate fusion alone. The faster, accuracy-compatible unfused
row-scale/cuBLAS baseline remains the basis of the snapshot's 9/10 counts.
Both baseline timings and candidate ratios are in
`b200_ads_blackwell_snapshot.csv`.

### Correctness and measurement protocol

All 144 selected-arm numerical checks pass: three seeds, four arms, ten
timed shapes and two tail shapes (M129/K1024/N3072 and M513/K1032/N3080).
The FP32 norm-before-GEMM oracle is rounded to the BF16 output contract.
Existing atol=.15, rtol=.02 and relative-L2<=.025 gates are retained; maximum
observed relative-L2 is 0.0000825. The conventional BF16-normalized-activation
control fails the FP32 allclose gate in all 36 additional checks. Those
failed checks remain diagnostic and are not timed or counted as comparable
implementations. Intermediate rounding cannot silently change the contract.

Schedules are selected from six ADS configurations, twelve Triton choices
and three separate-normalization configurations. This is a bounded search,
not an exhaustive optimum. Every timed arm uses fixed inputs and
preallocated output/scratch. Each phase uses five fresh CUDA-graph rounds,
seven randomized block samples per round, twenty launches per sample and
independent ordering seeds. All forty GPU traces contain kernel events.
ADS uses one kernel per call; selected Triton uses one or two. The selected
GPU remains exclusive throughout. Raw receipts, IR and traces remain local;
only the sanitized twenty-row CSV is exported.

### What the snapshot diagnostics show

Plain Triton already emits Blackwell `tcgen05.mma` instructions in this
compiler. ADS additionally uses TMA and TMEM, a persistent schedule and
separate producer/MMA/epilogue tasks. Its normalization reduction shares
loaded A tiles with GEMM and runs in the epilogue task. The larger selected
ADS tile is BM256/BN256/BK64 with two MMA groups; these pipeline and tile
differences accompany the larger-shape gains.

At M128/K1024/N3072, both selected kernels use BM64/BN64/BK64. ADS has 256
threads and 65924 bytes of shared memory, versus 128 threads and 49168
bytes for Triton; its static PTX has 52 barrier lines versus 27. The controlled
follow-up below identifies excessive subdivision of the normalization
reduction as a recoverable cause of the measured snapshot's small-shape
regression.

### M128 snapshot root cause and schedule improvement

The selected schedule divided each BK64 activation tile into sixteen
four-element reduction subtiles. Across sixteen K iterations, this creates
256 reduction subtiles per output tile. Its gamma-free normalization is
computed alongside GEMM and applied to the output. With BN64, the same input
row reduction is repeated across 48 output tiles; the unfused row-scale
control computes one inverse-RMS value per row. Sharing the activation load
does not eliminate that repeated reduction and synchronization work.

Changing only `NUM_X2_SUBTILES` from 16 to 8 reduces the subdivision to 128
subtiles per output tile. In a randomized paired ablation with three
correctness seeds and an independent retest:

| Variant | Retest (us) | Interpretation |
|---|---:|---|
| Original ADS, 16 reduction subtiles | 7.917 | Original regression reproduced |
| ADS, 8 reduction subtiles | 6.267 | 20.8% less latency; 1.180x vs unfused |
| ADS, 1 reduction subtile | 6.888 | Fewer subtiles is not monotonically faster |
| ADS, two shared-memory buffers | 7.922 | Buffer count alone does not help |
| ADS, one tensor-memory buffer | 7.888 | Buffer count alone does not help |
| ADS, BK128 and two shared-memory buffers | 6.549 | Fewer K iterations also help |
| ADS, BN32 | 11.800 | More output blocks increases total work |
| Improved Triton | 7.213 | Same paired experiment |
| Unfused inverse RMS + cuBLAS + scale | 7.394 | Same paired experiment |
| ADS GEMM only, normalization disabled | 5.970 | Different operation; diagnostic only |

Disabling normalization also removes its shared-buffer participation, so
the 1.947 us difference measures its combined reduction/pipeline cost, not
just arithmetic. This diagnostic is checked against an unnormalized GEMM
oracle and is excluded from RMSNorm speedup claims. All sixteen variants
pass their respective three-seed numerical gates; both timing phases use
five rounds, seven samples and twenty graph launches per sample.

The eight-subtile schedule keeps 96 blocks, 256 threads per block, 65924
shared-memory bytes and the same static barrier and Tensor Core instruction
counts. Static PTX shared-load lines fall from 19 to 11 and shuffle lines
from 19 to 11. These are static counts, not dynamic instruction totals.
Nsight Compute reports achieved occupancy of 12.31% originally and 12.09%
afterward, with theoretical occupancy of 12.50% for both. DRAM throughput
is only 0.17% and 0.23%, respectively, under the measured cache-reuse policy.
The unchanged resources and improved latency support reduction scheduling
as the cause of this recoverable regression, rather than increased occupancy
or lower HBM traffic. The 96-block grid still cannot cover all 148 SMs, but
the BN32 ablation shows that increasing block count alone is insufficient.

The existing runner now searches the six original ADS schedules plus the
eight-subtile candidate. The supplied ADS JIT bodies remain unchanged.
The complete ablation is in [b200_ads_m128_ablation.csv](b200_ads_m128_ablation.csv).

### Snapshot follow-up seven-configuration search: independent retest results

The full ten-shape run was repeated after adding the reduction candidate.
The table uses only timings from this fresh run. The original measurements
remain in `b200_ads_blackwell_snapshot.csv`.

| M | K | N | ADS TLX (us) | Improved Triton (us) | Fastest unfused (us) | ADS/unfused |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1024 | 3072 | 6.245 | 7.216 | 7.310 | 1.171x |
| 512 | 1024 | 3072 | 9.208 | 9.466 | 9.883 | 1.073x |
| 512 | 1024 | 4096 | 9.443 | 9.659 | 10.898 | 1.154x |
| 2048 | 1024 | 4096 | 16.554 | 23.571 | 25.155 | 1.520x |
| 8192 | 1024 | 4096 | 61.299 | 87.901 | 97.322 | 1.588x |
| 16384 | 1024 | 4096 | 116.672 | 164.739 | 191.611 | 1.642x |
| 65536 | 512 | 512 | 40.173 | 66.968 | 112.648 | 2.804x |
| 131072 | 512 | 512 | 75.573 | 140.326 | 219.331 | 2.902x |
| 524288 | 512 | 512 | 311.238 | 534.960 | 838.864 | 2.695x |
| 262144 | 512 | 2048 | 556.742 | 1103.573 | 1252.379 | 2.249x |

The tested ADS snapshot exceeds 1% speedup against the faster unfused control
in both phases for **10/10 shapes**, with M128 gains of 1.166x initially and 1.171x in
retest. It is faster than Triton on all ten shapes. Triton also exceeds
the 1% threshold on all ten shapes in this fresh run; its M128 margin is
only 1.31% in retest and was not confirmed in the earlier run, so that
particular Triton comparison remains sensitive to measurement variation.
The new ADS candidate is selected for M128; the other nine ADS schedules
remain the original selections. Small latency shifts elsewhere are fresh
measurements, not improvements attributed to this reduction change.

All 144 selected-arm numerical checks and forty kernel traces pass again,
with maximum relative-L2 of 0.0000825 and selected-GPU exclusivity retained.
The fixed eight-subtile schedule additionally passes nine checks across
three seeds and three M/K/N tail shapes, including M129/K1032/N3080, with
exact agreement against the original ADS wrapper. Gates and comparison
contracts are unchanged. Both unfused baseline timings are exported in
[b200_ads_blackwell_tuned.csv](b200_ads_blackwell_tuned.csv).

The ratios include scheduling, tile and implementation differences; they
do not isolate the contribution of normalization fusion alone. This is a
gamma-free forward operator study, with no full-model or training claim.
The exact pinned public-source comparison or an explicit source-diff audit
remains outstanding. No conclusions about the pinned public implementation
are established by these snapshot diagnostics.
