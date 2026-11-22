# Experiment plan and figure/table inputs

## 1. Fusion taxonomy for Table 4

The fusion-site list is finite at the operator-graph level.  The canonical
graph taxonomy has four vertical types plus one horizontal type:

1. **Prologue:** apply a producer-side operator after an anchor loads its input
   tile, such as Q/K normalization before attention or normalization before a
   GEMM.
2. **Epilogue:** apply a consumer-side operator to an anchor output tile before
   it is written, such as residual addition or output normalization.
3. **Multi-GEMM chain:** keep the intermediate of two or three dependent GEMMs
   on chip.
4. **Memory-bound chain:** merge consecutive pointwise or reduction operators,
   such as activation and normalization.
5. **Horizontal group:** combine independent sibling operators in one launch,
   such as several jagged index selections.  This is distinct from vertical
   producer-consumer fusion.

Table 4 should include a `Type` column with these names and a `Fusion site`
column for the concrete graph pattern.  A row is a **site family**, while GPU,
model, shape, backend, and schedule are measured factors.  Do not create a new
site for every tile configuration.

| Type | Site family | Fused graph pattern | Typical anchor/baseline | Required factors | Primary metrics | Claim supported |
|---|---|---|---|---|---|---|
| Prologue | Q/K norm + RoPE attention | RMSNorm → split-half RoPE → QK attention | separate norm/RoPE + fixed attention, `torch.compile`, vendor SDPA | B, S, head dim, causal, GQA, H100/H200/A800, Triton/TLX | p50 latency, speedup vs strongest baseline, correctness, K-preparation launches | attention prologue fusion removes intermediate traffic/repeated K work |
| Prologue | Q/K norm without RoPE | Q/K RMSNorm → QK attention | boundary-preserving unfused attention | B, S, D=128/256, dtype, causal | same as above plus max relative L2 and scratch bytes | factoring/reuse helps when the operation permits it |
| Prologue | Norm → GEMM | RMSNorm → GEMM | norm kernel + cuBLAS/cuBLASLt or compiled GEMM | M/K/N, norm axis, dtype, GPU, backend | latency, speedup, registers/shared memory, correctness | normalization intermediate stays on chip before GEMM |
| Epilogue | GEMM → norm | GEMM → RMSNorm/LayerNorm | GEMM followed by separate norm; vendor addmm where applicable | M/K/N, norm type, output dtype, GPU/backend | latency, fusion-only ratio, vendor ratio, memory traffic | epilogue fusion; report vendor comparison separately |
| Epilogue | GEMM → residual | GEMM + residual add (optionally norm) | `torch.addmm`/cuBLAS plus add | model-shaped M/K/N and residual stride/layout | latency, speedup vs compiled and vendor, peak memory | residual write/read elimination |
| Multi-GEMM chain | Multi-GEMM | GEMM → activation/reshape → GEMM | two independent GEMMs and materialized intermediate | M/K/N, activation, sequence/batch, GPU/backend | end-to-end chain latency, intermediate bytes, speedup | intermediate remains on chip across GEMMs |
| Epilogue / memory-bound chain | SwiGLU/pointwise | gate/up GEMMs → SiLU × gate → down GEMM (or pointwise subchain) | eager pointwise + vendor GEMMs | hidden/intermediate sizes, B/S, forward/backward, optimizer | forward/backward latency, complete-step impact, correctness | pointwise and GEMM boundary fusion |
| Prologue | MoE expert-local | routed expert GEMM → activation → expert GEMM | staged expert implementation and vendor GEMM | experts, tokens/expert, capacity, top-k, forward/backward | per-expert and routed end-to-end latency, load balance, correctness | expert-local fusion only; never imply router/routing fusion |
| Prologue | LM-head/loss training | logits projection → loss → backward/update | eager/compiled/vendor projection and loss | vocab, B/S, AdamW/Muon, forward/backward | loss/gradient agreement, step throughput, peak memory | training loss fusion; keep separate from attention claims |
| Horizontal / memory-bound chain | Jagged concat + LayerNorm | independent index-select/concat siblings plus row normalization | separate index-selects and LayerNorm | feature count, embedding dim, length distribution, batch | latency, tail sensitivity, memory traffic, correctness | horizontal grouping and memory-bound chain |

For each family report both `fusion_vs_anchor` and `fusion_vs_strongest`
(the fastest valid standard baseline).  A fusion can beat its custom anchor
while still losing to cuBLAS/cuDNN/SDPA; Table 4 must preserve that distinction.

Use this mapping when grouping the concrete rows: Q/K norm (with or without
RoPE), norm-to-GEMM, MoE expert normalization, and LM-head normalization are
**prologues**; residual and SwiGLU projection fusions are **epilogues**;
dependent GEMM pairs are **multi-GEMM chains**; pointwise activation/loss
chains are **memory-bound chains**; and jagged concatenation/index selection is
a **horizontal group** (often also a memory-bound chain).  Native MoE routing
must remain a separate diagnostic because it is not fused by an expert-local
kernel.

## 2. Main complete-model matrix (Table 2)

Run the same model, shape, precision, fusion plan, and optimizer for both
backends whenever both implementations are available.  The minimum matrix is:

| Factor | Required values |
|---|---|
| GPU | A800, H100, H200 |
| Backend | Triton, TLX; use `N/A (arch.)` only for a documented unsupported feature |
| Mode | Decode and Train |
| Train optimizer | AdamW and Muon |
| Dense checkpoints | SmolLM2-135M/360M, Qwen2.5-0.5B, Qwen3-0.6B/1.7B/4B/8B/14B, Gemma-3-4B, Llama-3.1-8B, Mistral-7B, Qwen3.5-9B, Qwen3.8-27B |
| MoE checkpoints | Qwen3-30B-A3B and Qwen3.5-35B-A3B (or the exact available revisions) |
| Decode shapes | fixed prompt length and generated-token protocol; record batch, prompt length, decode length, KV-cache state |
| Train shapes | at least B×S = 1×512, 8×512, and the largest memory-valid shape per GPU |

Per cell record deployed geometric mean, valid/kept count, p50 latency, tokens/s
or samples/s, peak memory, and the baseline used.  A rejected shape contributes
`1.000x` to a deployed aggregate and remains visible as rejected.  Never merge
isolated census or LM-head rows into this table.

## 2.1 Production-model integration (Table 1)

On the production ranking model and B200, rerun the exact production-shape
workload after pinning the final source revision.  Record the eight generated
kernel sites, correctness count, integrated/deployed count, complete training
step QPS, eager PyTorch and compiled baselines, peak memory, and per-site
forward/backward evidence.  Keep this run separate from open-weight LLM rows:
the production model uses its own jagged layout and deployment policy, and its
QPS result must not be inferred from isolated operator ratios.  If the ADS
source fingerprint is not the pinned public revision, label the receipt as a
snapshot diagnostic and exclude it from a production claim.

## 3. Figures to prepare as placeholders

Create the plotting scripts and empty CSV schemas before measurements.  Each
figure should be reproducible from sanitized CSVs and should show the number of
valid rows; no silently dropped failures.

### Figure A — fusion-site speedup bars (main result)

- x-axis: fusion site family from the Table 4 taxonomy;
- y-axis: geometric-mean speedup over the strongest valid baseline;
- grouped bars: A800/H100/H200, or separate panels if labels become crowded;
- hatch/line style: Triton versus TLX; do not rely on color alone;
- overlay a thin point or second bar for `fusion_vs_anchor` to expose the
  fusion-only effect;
- annotate `n`, kept fraction, and 95% bootstrap CI (or min/median/max when
  rows are too few).  Report per-shape points in an accompanying panel.

This figure answers “how much acceleration does each fusion family provide?”
and keeps “better than the strongest SOTA baseline” distinct from “better than
the unfused implementation.”

### Figure B — baseline comparison bars

For a fixed model/shape/GPU, group Eager PyTorch, `torch.compile`, vendor
library, Triton, and TLX.  Plot normalized latency (lower is better) and write
the absolute p50 in a small label.  Use the same correctness gate and timing
protocol for every arm.  Include only valid, paired rows; list rejected rows in
the caption or a small coverage strip.

### Figure C — sequence-length scaling

- x-axis: S = 128, 256, 512, 1024, 2048, 4096 (extend only when memory-valid);
- y-axis: speedup versus strongest baseline, with a second optional panel for
  absolute latency;
- fixed: model, B, head dimension, causal flag, dtype, GPU, backend;
- lines: fusion and each baseline, with markers at every measured shape.

Use separate causal and non-causal panels.  Do not connect missing or failed
rows as if they were measurements.

### Figure D — batch-size scaling

Sweep B = 1, 8, 32 (and 64 where valid) at fixed S and model.  Show speedup,
latency, and peak memory.  For decode, report both per-token latency and
tokens/s; for training, report examples/s and complete-step time.

### Figure E — controller ablation

Grouped bars for model-only, harness-only, and full controller.  Show candidate
proposals, compiled trials, search iterations, back-offs, correctness pass rate,
kept fraction, and deployed speedup.  Keep the checkpoint/shape/template set
and timing budget identical.  This is an admission/controller experiment, not a
model-size speedup comparison.

### Figure F — coverage and failure outcomes

A stacked bar per GPU/backend/model showing valid-and-kept, valid-but-rejected,
correctness failure, timeout/compile failure, and unsupported (`N/A`).  This
prevents a high geometric mean from hiding poor coverage and gives Table 2/4 a
clear denominator.

## 4. Required ablations

1. **Backend:** Triton versus TLX with matched kernels and schedules where
   possible; separately mark backend-specific features.
2. **Fusion:** unfused, fusion-only, fusion plus schedule tuning, and the
   selected production plan.
3. **Search/controller:** model-only, harness-only, full controller; fixed
   proposal count and timing budget.
4. **Shape:** sequence length, batch size, head dimension, causal flag, hidden
   size, number of experts/tokens per expert, and tail/non-multiple dimensions.
5. **Precision:** BF16 primary; FP16/FP32 only when the baseline and numerical
   contract are unchanged and the run is labelled separately.
6. **Training:** AdamW and Muon; report forward, backward, optimizer update,
   and complete step rather than loss-kernel time alone.
7. **Resource mechanism:** registers/thread, shared memory, occupancy,
   barriers, launch count, DRAM bytes/throughput, and achieved Tensor Core
   utilization when a profiler is available.  These explain a result but are
   not speedup substitutes.

## 5. Run acceptance and naming

Use a stable name such as
`<site>__<model>__<gpu>__<backend>__<mode>__B<B>_S<S>__<commit>`.
Every receipt must include:

- exact source commit and kernel fingerprint;
- GPU name, selected device, clock/power settings, and software versions;
- input shape, dtype, causal/layout flags, optimizer and seed list;
- correctness metrics before timing (max abs, max rel, relative L2, logits/top-1
  or loss/gradient checks as applicable);
- warm-up count, synchronized timing method, p50/p90/p99 or sample list,
  randomized arm order, and independent retest;
- baseline latency for every comparison, speedup definition, and whether the
  row is `tested_pass`, `tested_invalid`, `tested_fail`, or `N/A`;
- GPU exclusivity snapshots, trace count/manifest, and compile time if measured.

Promote a row to the paper only when the required correctness, paired timing,
fresh retest, and provenance gates pass.  Keep failed rows in the CSV with an
explicit status and reason.

## 6. Runner map

Use the checked-in runners for isolated evidence and keep complete-model
deployment receipts in the private experiment harness.  These commands are
templates; every worker must set an output directory unique to the run.

```bash
# Q/K norm + RoPE census (H100/H200; include traces)
python src/e1_rope_publication_v1.py \
  --output outputs/<gpu>-e1-rope.json \
  --trace-dir outputs/<gpu>-e1-rope-traces

# Refined K-once census and model-prefill follow-up
python src/e1_rope_refined_census.py \
  --output outputs/<gpu>-e1-rope-refined.json \
  --trace-dir outputs/<gpu>-e1-rope-refined-traces
python src/e2e_prefill_refined.py --help

# Residual epilogue and pointwise/GEMM diagnostics
python src/e4_model_shapes_publication_v2.py --refine \
  --out outputs/<gpu>-residual-refine
python src/e8_forward_publication.py --h100-unfused \
  --out outputs/<gpu>-pointwise-gemm

# A/A noise-floor controls
python src/e7_aa_noise_floor_osu_v3.py \
  --out outputs/<gpu>-e7-aa.json \
  --trace-dir outputs/<gpu>-e7-aa-traces
```

For each command, preserve stdout/stderr, the exact CLI, source hash, and the
sanitized CSV next to the JSON receipt.  Do not use the legacy E1 diagnostic as
a replacement for the paired v2/refined runner.  The public repo does not
contain the complete-model deployment harness or model weights; those rows
must be generated by the private runner and copied back only as sanitized
receipts.

## 7. Allocation notes

The existing OSU handoff reserves completed H200 cards 0–3; future H200 reruns
must use an available card in 4–7.  Follow the current `RUN_REQUESTS.md` for
the exact host allocation.  Do not launch new work on a card already listed as
reserved, and do not replace a missing receipt with a withdrawn historical run.
