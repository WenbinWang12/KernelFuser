# Claude/Codex experiment runbook

This file is the execution contract for an agent or human who is asked to run
the KernelFuser experiments.  Read it before launching a GPU job.  The goal is
to produce an auditable receipt that can be placed into Table 1, Table 2,
Table 4, or a figure; the goal is not merely to obtain a favorable timing.

The public repository contains isolated runners and sanitized results.  It does
**not** contain model weights, the complete-model deployment harness, private
production data, or raw traces.  A worker must have access to those inputs from
the project owner before attempting the corresponding task.

## 0. Agent operating rules

When Claude or Codex is assigned an experiment, follow this order:

1. Read `todo/README.md`, `todo/EXPERIMENT_PLAN.md`, this file,
   `todo/RESULT_SCHEMA.md`, and the relevant `experiments/RUN_REQUESTS.md`.
2. Inspect the current checkout and record `git rev-parse HEAD`.  Do not mix
   files from another checkout or silently use a locally modified kernel.
3. Run the smallest smoke test first.  Confirm CUDA, the model path, the
   runner arguments, and the correctness gate before starting a full sweep.
4. Use a unique output directory.  Never overwrite a receipt from another
   GPU, software stack, source commit, or experiment task.
5. Do not change a kernel, baseline, tolerance, seed, timing protocol, or
   shape list to make a run pass.  If a change is needed, stop and report it as
   a new experiment with a new source hash.
6. Correctness is checked before timing.  A correctness, compile, timeout,
   trace, or GPU-exclusivity failure is retained with an explicit status and
   has no speedup claim.
7. Compare paired arms on identical tensors and shapes.  Report both
   `speedup_vs_anchor` (the fusion-only effect) and
   `speedup_vs_strongest` (the claim against the fastest valid standard arm).
8. Preserve stdout/stderr, the exact command, source hashes, environment
   versions, GPU snapshots, trace manifest, JSON receipt, and sanitized CSV.
9. Do not commit weights, credentials, hostnames, GPU UUIDs, raw traces, or
   private paths.  Copy only sanitized receipts to `results/` after review.
10. If an input or requirement is missing, mark the task `blocked` and list the
    exact missing item.  Do not substitute a nearby model or an old receipt.

An agent must finish with one of these outcomes: `tested_pass`,
`tested_invalid`, `not_run`, or `blocked`.  `tested_invalid` is a useful result
when the candidate loses or fails correctness; it is not permission to delete
the row.

## 1. Environment and run directory

Run on Linux with an NVIDIA GPU, CUDA, PyTorch, Triton, and (for TLX tests)
the matching `fbtriton`/TLX build.  The recorded public H100 refinement used a
development PyTorch 2.15/Triton 3.8 stack; the public H200 receipts used the
PyTorch 2.5.1/Triton 3.1 stack.  Do not combine ratios across stacks without
recording the stack and explaining the comparison.

Before a run, save the following to `run_meta.txt` or the JSON receipt:

```bash
git rev-parse HEAD
git status --short
nvidia-smi -L
nvidia-smi --query-gpu=index,name,uuid,memory.total,temperature.gpu,power.limit,clocks.current.graphics --format=csv
python --version
python -c "import torch, triton; print('torch',torch.__version__,'cuda',torch.version.cuda,'triton',triton.__version__); print(torch.cuda.get_device_name(0))"
```

The public OSU handoff reserves completed H200 cards 0--3.  Future H200 runs
must use an available card in 4--7 and must check the current allocation before
starting.  Use one exclusive GPU.  A foreign process, clock change, thermal
throttle, or power-cap change during a promoted row invalidates that row unless
the receipt records and explains the condition.

Use this naming convention:

```text
<task>__<site>__<model>__<gpu>__<backend>__<mode>__B<B>_S<S>__<commit>
```

Recommended private layout:

```text
outputs/<task>/<GPU>/<model>/<backend>/<shape>/<commit>/
  command.txt
  run_meta.txt
  receipt.json
  rows.csv
  traces/
  stdout.log
  stderr.log
```

Do not put `outputs/` into the public repository.  The public `results/`
directory is for sanitized CSV summaries only.

## 2. Data and model contracts

### 2.1 Isolated attention geometry

The E1 and refined E1 runners use seeded random Q/K/V and RMSNorm weights;
they do not load Hugging Face checkpoints.  Use the geometry encoded in
`src/e1_rope_publication_v1.py` and `src/e1_rope_refined_impl.py`:

| Model label | Query heads | KV heads | Head dim | Special rule |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 16 | 8 | 128 | Qwen RMSNorm, scale $D^{-1/2}$ |
| Qwen3-1.7B | 32 | 8 | 128 | same Qwen rule |
| Qwen3-4B | 32 | 8 | 128 | same Qwen rule |
| Qwen3-8B | 32 | 8 | 128 | main short-prefill follow-up |
| Qwen3-32B | 64 | 8 | 128 | same Qwen rule |
| Qwen3-30B-A3B | 32 | 4 | 128 | isolated attention geometry only |
| Gemma-3-4B | 8 | 4 | 256 | `query_pre_attn_scalar=256` |
| Gemma-3-12B | 16 | 8 | 256 | `query_pre_attn_scalar=256` |
| Gemma-3-27B | 32 | 16 | 128 | `query_pre_attn_scalar=256` |

The normal E1 sweep is `S=128,256,384,512,768,1024,1536,2048`,
`B=1`, and causal/non-causal.  The RoPE sweep is `S=128,256,384,512`,
`B=1,8,32`, and causal/non-causal.  The refined RoPE census uses the same
four sequence lengths and batches.  A `--smoke` run is only a protocol check;
it is never a paper result.

### 2.2 MoE expert-local geometry

`src/moe_staged_strict.py` and `src/e3a_moe_publication.py` use synthetic,
balanced expert-local tensors.  They do not execute a router, all-to-all,
shared experts, or a complete checkpoint.  The public nine-architecture list
is:

| Model | Hidden | Top-k | Experts | Expert intermediate | Shared experts |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 2048 | 8 | 128 | 768 | 0 |
| Qwen3-235B-A22B | 4096 | 8 | 128 | 1536 | 0 |
| DeepSeek-V3 | 7168 | 8 | 256 | 2048 | 1 |
| DeepSeek-V2 | 5120 | 6 | 160 | 1536 | 2 |
| OLMoE-1B-7B | 2048 | 8 | 64 | 1024 | 0 |
| GPT-OSS-120B | 2880 | 4 | 128 | 2880 | 0 |
| Qwen2-57B-A14B | 3584 | 8 | 64 | 2560 | 1 |
| Qwen3-Next-80B-A3B | 2048 | 10 | 512 | 512 | 1 |
| Llama-4-Maverick | 5120 | 1 | 128 | 8192 | 1 |

Use `tokens=2048,8192,32768,131072`.  The local expert row count is
`M=max(16, round(tokens * top_k / experts))`.  Record this calculated `M` in
each row; do not call it a routed-token result.

### 2.3 Dense and epilogue geometry

`src/e8_forward_publication.py` contains the public synthetic E8 geometries:

- Qwen3-0.6B dense RMSNorm prologue: `K=1024, N=4096,
  M=512,2048,8192,16384`.
- Wukong-tower SwiGLU development geometry: `K=2048, N=8192,
  M=512,2048,8192,16384`.

The Wukong rows are development provenance and must not be presented as a
Qwen/Gemma complete-model result.  `src/e4_model_shapes_publication_v2.py`
is the model-shaped residual experiment and defaults to `M=128,512,2048,8192`;
the exact `(M,K,N)` values are emitted in its receipt.  Its vendor comparator
is `torch.addmm`/cuBLAS.  `src/e4_epilogue_publication.py` is a separate
forward-only site runner; inspect `src/epilogue_sites.py` before selecting a
site and do not infer full-model training speed from it.

### 2.4 Complete-model Table 2 matrix

The complete-model deployment harness and weights are private.  Do not try to
reconstruct Table 2 from isolated CSVs.  The target checkpoint list is:

```text
SmolLM2-135M, SmolLM2-360M,
Qwen2.5-0.5B,
Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B,
Gemma-3-4B, Llama-3.1-8B, Mistral-7B,
Qwen3.5-9B, Qwen3.8-27B,
Qwen3-30B-A3B, Qwen3.5-35B-A3B
```

The intended factors are GPU `A800/H100/H200`, backend `Triton/TLX`, mode
`Decode/Train`, and optimizer `AdamW/Muon` for training.  Use the exact
checkpoint revision and local model directory supplied by the owner; record
the revision, config hash, tokenizer hash, dtype, and weight file manifest.
Do not replace a missing checkpoint with a similarly named model.

For decode, hold the prompt/cache protocol fixed and record `B`, prompt/cache
length `T`, generated-token count, KV-cache dtype/layout, and CUDA-graph
policy.  For training, record `B`, sequence length `S`, microbatching,
gradient accumulation, optimizer hyperparameters, and whether the timing is a
forward-only, forward+backward, optimizer-update, or complete-step number.
Minimum planned training shapes are `B x S = 1 x 512` and `8 x 512`, plus the
largest memory-valid shape per GPU.  A rejected shape contributes `1.000x` to
the deployed aggregate and remains visible in the coverage counts.

### 2.5 Production Table 1 data

The production ranking model, B200 shapes, ADS source, and training data are
private.  Run only with the pinned private harness.  Record the eight generated
kernel sites, per-site correctness, integrated count, complete training-step
QPS, eager and compiled baselines, peak memory, and source fingerprint.  A
B200 snapshot whose source does not match the pinned public revision is a
diagnostic only (`publication_eligible=false`).

## 3. Task cards and exact public commands

Run the relevant smoke command first, then the full command.  All commands are
from the repository root.  Replace `<GPU>`, `<RUN_ID>`, and output paths with
actual values; do not leave angle-bracket placeholders in a receipt.

### P0-A — RoPE-aware isolated attention

Purpose: measure Q/K RMSNorm + split-half RoPE inside the fixed attention
anchor.  Arms are same-anchor with separate norm/RoPE, `torch.compile`/SDPA,
and vendor SDPA.  This is forward-only operator evidence; it is not full-model
decode or training.

```bash
python src/e1_rope_refined_census.py \
  --output outputs/<RUN_ID>/e1_rope_refined.json \
  --trace-dir outputs/<RUN_ID>/traces
```

Optional controlled subsets use `--models`, `--sizes`, `--batches`,
`--causal-modes`, `--bn 64|128`; `--smoke` means Qwen3-8B, B=1, S=128,
non-causal, and is not publishable.  For the older full protocol use:

```bash
python src/e1_rope_publication_v1.py \
  --output outputs/<RUN_ID>/e1_rope.json \
  --trace-dir outputs/<RUN_ID>/traces \
  --models Qwen3-0.6B,Qwen3-1.7B,Qwen3-4B,Qwen3-8B,Qwen3-32B,Qwen3-30B-A3B,Gemma-3-4B,Gemma-3-12B,Gemma-3-27B \
  --sizes 128,256,384,512 --batches 1,8,32 --causal-modes false,true
```

Do not call the old non-paired CSV a final result.  Every promoted row needs
correctness against an independent FP32 oracle, fresh retest when the initial
ratio exceeds 1.01, four arm traces, and exclusivity evidence.

The refined census derives each row's input seed as
`1701 + model_index*10000 + B*101 + S*7 + int(causal)`.  Its paired timing is
three rounds of ten samples for the initial check followed by five rounds of
fourteen samples for the fresh retest, for each of the unfused, compiled, and
vendor baselines.  It captures four traces (candidate, unfused, compiled,
vendor) per row and records the physical-GPU exclusivity observations.  Do not
replace this with a single average or with a timing collected during lazy
TorchInductor compilation.

### P0-B — Qwen3-8B model prefill

Purpose: determine whether the isolated attention improvement survives a real
model step.  Supply a local Qwen3 checkpoint directory; the runner does not
download weights.

```bash
python src/e2e_prefill_refined.py \
  --model-dir /abs/path/Qwen3-8B \
  --shapes 1x128,1x512,8x256,32x128 \
  --output outputs/<RUN_ID>/qwen3_prefill.json \
  --rounds-initial 2 --rounds-retest 3 --samples 12 \
  --cpu-check
```

The CPU check compares the functional model implementation against the local
Hugging Face model before GPU timing.  Full-logit/top-1 agreement between eager
and compiled baselines is a prerequisite; if it fails, retain the receipt with
no timing rows.  Record model revision, tokenizer/config hashes, BF16 dtype,
causal mask, and the exact shape list.

The older monkey-patch runner is available for audit only:

```bash
python src/e2_qwen3_rope_prefill.py --model /abs/path/Qwen3-8B \
  --out outputs/<RUN_ID>/legacy_prefill.json --sizes 128 256 \
  --batches 1 8 32 --smoke
```

Do not merge its rows with the refined runner or with complete-model Table 2.

### P0-C — A/A noise floor

Purpose: bound run-to-run timing noise at the same short prefill shapes.

```bash
python src/e7_rope_aa.py --out outputs/<RUN_ID>/e7_aa.json \
  --trace-dir outputs/<RUN_ID>/traces --batches 1,8,32 --sizes 128,256
python src/e7_aa_noise_floor_osu_v3.py \
  --out outputs/<RUN_ID>/e7_aa_noise.json \
  --trace-dir outputs/<RUN_ID>/noise-traces
```

The A/A runner has no fused-vs-baseline claim.  Use it to report timing spread,
not to inflate the fusion speedup.

### P0-D — Residual and dense epilogue/prologue sites

Purpose: compare the fused operator with the same Triton anchor and with the
strongest vendor comparator.

```bash
python src/e4_model_shapes_publication_v2.py \
  --out outputs/<RUN_ID>/e4_residual \
  --trace-dir outputs/<RUN_ID>/e4-traces \
  --M 128,512,2048,8192
python src/e4_epilogue_publication.py \
  --out outputs/<RUN_ID>/e4_sites.json \
  --trace-dir outputs/<RUN_ID>/site-traces \
  --sites gemm_residual --M 128,512,2048,8192
```

The vendor comparator is `torch.addmm`; do not report a win over the same
custom Triton GEMM as a win over cuBLAS.  Record registers, shared memory,
occupancy, launch count, p50/p95 latency, correctness, and fresh retest.

### P0-E — MoE expert-local normalization

Purpose: test RMSNorm → gate/up GEMM with balanced synthetic expert-local
tokens.  This does not test routing or a complete MoE layer.

```bash
python src/moe_staged_strict.py \
  --out outputs/<RUN_ID>/moe_staged \
  --trace-dir outputs/<RUN_ID>/moe-traces \
  --tokens 2048,8192,32768,131072
python src/e3a_moe_publication.py \
  --out outputs/<RUN_ID>/e3a_moe.json \
  --trace-dir outputs/<RUN_ID>/e3a-traces \
  --tokens 2048,8192,32768,131072
```

Use `--only Qwen3-30B-A3B,DeepSeek-V3` for a smoke subset.  Before a full run,
confirm that the vendor grouped implementation does not materialize the
largest DeepSeek weight tensor.  Record implementation (`torch._grouped_mm` or
the documented `torch.bmm` fallback), balanced local `M`, expert count, top-k,
correctness of input/weight/scale/bias gradients, and backward timing.  Do not
describe this as router or end-to-end MoE acceleration.

### P0-F — E8 dense RMSNorm and SwiGLU diagnostics

Purpose: fill the remaining isolated site families with the public synthetic
geometries.  The runner itself is forward-only.

```bash
python src/e8_forward_publication.py \
  --h100-unfused --out outputs/<RUN_ID>/e8_h100
```

The `Wukong-tower` rows are development geometry and must be labelled as such.
Do not use them as a model-level claim.  If a worker runs the B200 ADS path,
use the source gate explicitly:

```bash
python src/e8_forward_publication.py --b200-ads \
  --ads-source /private/pinned/ads/source \
  --out outputs/<RUN_ID>/e8_b200 \
  --csv outputs/<RUN_ID>/b200.csv
```

Never pass `--allow-source-version` unless the owner has supplied a written
source-diff audit.  A source mismatch remains `publication_eligible=false`.

### P1 — TLX/SwiGLU correctness and backend comparison

The TLX modules require Hopper/SM90 and the matching `fbtriton` build.  First
run the tests without changing tolerances:

```bash
python -m pytest tests/test_swiglu_sm90.py
```

The test covers BF16 forward/backward, non-contiguous input, and a partial final
tile.  The modules fuse only `silu(gate) * up`; they do not fuse projections,
routing, or a complete model.  If a TLX feature is unavailable, write
`N/A (arch.)` with the exception and software version rather than silently
using Triton as if it were TLX.

### P0-G — Complete-model Table 2 (private harness)

The public runners above cannot populate Table 2.  The owner must provide the
private complete-model harness and local checkpoint paths.  For every
`model × GPU × backend × mode × optimizer` cell:

1. Use the same model revision, dtype, shape list, fusion plan, and baseline
   policy for Triton and TLX.
2. Run correctness before timing.  Decode checks full logits/top-1/NLL; train
   checks loss, gradients, and parameter/update agreement.
3. Time eager, compiled, vendor/SDPA, same-anchor, and candidate arms with
   randomized paired order, CUDA graphs where supported, and fresh retests.
4. Record p50/p95, tokens/s or samples/s, step time, peak memory, valid count,
   kept count, rejection reason, and the strongest baseline used.
5. Aggregate a deployed geometric mean with rejected/invalid shapes at
   `1.000x`; never drop rejected rows from the denominator.

The output must identify `backend` as `Triton` or `TLX` (not an unlabeled
"fused" arm), and must distinguish Decode from Train/AdamW/Muon.  A complete
model row is not valid if it only contains an isolated attention or LM-head
receipt.

### P0-H — Production Table 1 (private B200 harness)

Run the pinned production ranking model at its production shapes.  Return the
eight generated kernel sites, correctness count, integrated count, end-to-end
training QPS, eager/compiled baselines, peak memory, and source fingerprint.
Keep this receipt separate from open-weight language-model rows.  If the ADS
source is a snapshot, label the entire receipt diagnostic and do not promote it
to the production claim.

## 4. Required metrics and acceptance gates

Every row needs the fields in `todo/RESULT_SCHEMA.md`.  The following fields
are the minimum decision record:

### Correctness

- FP32 oracle output and BF16 candidate output;
- max absolute error, max relative error, relative L2, finite check,
  `allclose` result and tolerance;
- for attention/full model: logits L2, top-1 agreement, NLL/KL when available;
- for training: loss agreement, gradient agreement, parameter/update agreement;
- exact seed list and whether checks cover tails, causal masks, GQA, and
  non-contiguous layouts.

### Timing

- warmups (normally 20), samples (normally 100), rounds (normally 5), and
  launches per sample;
- p50, p95/p99 or full sanitized sample list, mean, SEM/two-SEM;
- candidate, anchor, strongest standard baseline, and vendor p50;
- `speedup_vs_anchor = anchor_p50 / candidate_p50`;
- `speedup_vs_strongest = strongest_baseline_p50 / candidate_p50`;
- initial ratio, retest ratio, and a boolean `confirmed_1pct` when the fresh
  retest also exceeds 1.01.

### System and provenance

- physical GPU name/index/UUID privately, visible-device setting, clocks,
  power limit, temperature and exclusivity observations;
- Python, PyTorch, CUDA, Triton, TLX/fbtriton versions;
- source commit, source hash, kernel fingerprint, exact command and model
  revision/config/tokenizer hashes;
- non-empty trace for every compared arm, trace manifest, and compile errors;
- `publication_eligible` and a reason whenever it is false.

Do not promote a ratio because it is numerically high when coverage is poor,
the baseline was not paired, correctness was skipped, the trace is missing, or
the source version is unknown.

## 5. Output, review, and handoff

At the end of each task, produce a short `RUN_NOTE.md` containing:

```text
Task and claim boundary:
Git commit / source hashes:
GPU and software stack:
Model/checkpoint/data revision:
Exact command:
Shape matrix and number of rows:
Correctness status and tolerances:
Timing protocol and arm order:
Initial and retest ratios:
Coverage / kept / rejected / invalid:
Trace manifest:
Publication eligibility:
Known deviations or blocked inputs:
```

Then run a consistency check:

```bash
node -e "const fs=require('fs'); JSON.parse(fs.readFileSync('outputs/<RUN_ID>/receipt.json')); console.log('JSON ok')"
```

Sanitize a copy into CSV without hostnames, paths, UUIDs, PIDs, credentials,
weights, or raw traces.  The owner reviews the receipt before copying it to
`results/` and before changing any paper table.  A worker should report the
Git commit, output path, row count, status counts, and any blocked item in its
final message; it should not claim that a pending table cell is complete.
