# KernelFuser model and data manifest

This file is the first input contract for a Claude/Codex worker.  Read it
before downloading a checkpoint, preparing a dataset, or launching a GPU job.
The public repository contains the isolated synthetic runners and sanitized
receipts, but it intentionally does not contain model weights, private
production data, or the complete-model deployment harness.  A worker must ask
the owner for the missing manifest instead of guessing a path or substituting
a nearby checkpoint.

## 1. Authority and provenance

Use the following order when two descriptions disagree:

1. the task-specific manifest supplied by the owner (checkpoint revision,
   private model path, dataset hash, and GPU allocation);
2. the exact checked-out source commit and the runner defaults in `src/`;
3. `experiments/RUN_REQUESTS.md` for the active OSU H100/H200 handoff;
4. this file and the broader planning document in `EXPERIMENT_PLAN.md`.

Record all four references in `RUN_NOTE.md`.  A model name alone is not an
identity.  Record the Hugging Face/config revision, tokenizer revision, local
directory hash or weight-file manifest, dtype, and any quantization or
sharding choice.

## 2. Public isolated attention geometries

The E1 runners do **not** load a checkpoint.  They create seeded random Q/K/V
and RMSNorm tensors with the published attention geometry.  These rows are
operator evidence only; they do not establish whole-model decode, prefill, or
training speedups.

| Label | Query heads | KV heads | Head dim | Query scale | Notes |
|---|---:|---:|---:|---|---|
| Qwen3-0.6B | 16 | 8 | 128 | $D^{-1/2}$ | Qwen RMSNorm |
| Qwen3-1.7B | 32 | 8 | 128 | $D^{-1/2}$ | Qwen RMSNorm |
| Qwen3-4B | 32 | 8 | 128 | $D^{-1/2}$ | Qwen RMSNorm |
| Qwen3-8B | 32 | 8 | 128 | $D^{-1/2}$ | main short-prefill follow-up |
| Qwen3-32B | 64 | 8 | 128 | $D^{-1/2}$ | isolated geometry only |
| Qwen3-30B-A3B | 32 | 4 | 128 | $D^{-1/2}$ | isolated attention geometry only |
| Gemma-3-4B | 8 | 4 | 256 | `query_pre_attn_scalar=256` | Gemma scaling rule |
| Gemma-3-12B | 16 | 8 | 256 | `query_pre_attn_scalar=256` | Gemma scaling rule |
| Gemma-3-27B | 32 | 16 | 128 | `query_pre_attn_scalar=256` | Gemma scaling rule |

Use BF16 inputs and an independent FP32 oracle unless a task card explicitly
changes the precision.  The legacy no-RoPE census is normally `B=1`,
`S=128,256,384,512,768,1024,1536,2048`, causal and non-causal.  The RoPE
census is `B=1,8,32`, `S=128,256,384,512`, causal and non-causal.  A smoke
run is only an environment/protocol check and must not be promoted.

The fixed correctness seeds used by the public E1 paths are `1701`, `2803`,
and `3907`; refinement and retest code may add seeds that are recorded in the
receipt.  Never replace the geometry with a real checkpoint without changing
the `site_family` and claim boundary.

## 3. Public MoE expert-local geometries

E3a and `moe_staged_strict.py` use balanced synthetic expert-local tensors.
They do not execute a router, all-to-all dispatch, shared experts, capacity
overflow, or a complete MoE checkpoint.  For token count $T$, the local row
count is `M=max(16, round(T * top_k / experts))`.

| Label | Hidden $K$ | Top-$k$ | Experts | Expert intermediate $I$ | Shared experts |
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

Required token scales are `2048,8192,32768,131072`; correctness seeds are
`20261009,20261010,20261011`; the primary dtype is BF16 with an FP32 oracle.
Record the calculated local `M`, the grouped implementation selected by the
runner (`torch._grouped_mm` or the documented `torch.bmm` fallback), and the
four gradient checks.  Do not label this result as router or end-to-end MoE
acceleration.

## 4. Public dense and epilogue geometries

`src/e8_forward_publication.py` contains synthetic site checks:

- Qwen3-0.6B dense RMSNorm prologue: `K=1024`, `N=4096`,
  `M=512,2048,8192,16384`.
- Wukong-tower development SwiGLU geometry: `K=2048`, `N=8192`,
  `M=512,2048,8192,16384`.

The Wukong rows are development provenance, not Qwen/Gemma model results.
`e4_model_shapes_publication_v2.py` defaults to residual GEMM `M=128,512,2048,8192`; it emits the exact model/site `(M,K,N)` in the receipt and compares with
the vendor `torch.addmm`/cuBLAS path.  `e4_epilogue_publication.py` is a
separate forward-only site runner.  Read `src/epilogue_sites.py` before adding
a site; do not infer a complete training-step gain from an isolated ratio.

## 5. Complete-model Table 2 checkpoint contract

Table 2 requires complete-model receipts.  The owner must provide a local
checkpoint directory and revision for every row.  The planned labels are:

```text
SmolLM2-135M, SmolLM2-360M,
Qwen2.5-0.5B,
Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B,
Gemma-3-4B, Llama-3.1-8B, Mistral-7B,
Qwen3.5-9B, Qwen3.8-27B,
Qwen3-30B-A3B, Qwen3.5-35B-A3B
```

The target factors are:

| Factor | Required values |
|---|---|
| GPU | A800, H100, H200 |
| Backend | Triton and TLX when implemented; `N/A (arch.)` only with a documented reason |
| Mode | Decode and Train |
| Optimizer | AdamW and Muon for Train |
| Dtype | BF16 primary; other dtypes are separate experiments |
| Decode | fixed prompt/cache protocol, batch $B$, cache/prompt length $T$, generated-token count, KV-cache dtype/layout |
| Train | microbatch $B$, sequence length $S$, gradient accumulation, optimizer hyperparameters, timed-step definition |

Do not silently mix the two valid input protocols.  The public decode runner
uses seeded random token IDs and an eager BF16 prefill to construct the KV
cache; the WikiText-2 prompt variant is a separate protocol.  A Table 2 task
must state which protocol it uses and keep it fixed for all backends and
baselines.  For training, the owner must supply the exact WikiText-2 split,
tokenizer revision, tokenization/preprocessing hash, sample order or seed,
sequence packing rule, and number of warm-up/timed steps.  Current planning
uses at least $B\times S=1\times512$ and $8\times512$, plus the largest
memory-valid shape per GPU; the receipt must state whether the number is
forward-only, forward+backward, optimizer update, or a complete step.

For every cell, record eager, compiled, vendor/SDPA, same-anchor, and candidate
arms when applicable.  Decode correctness includes full logits, top-1 and NLL
(or the task's declared equivalent); training correctness includes loss,
gradients, and parameter/update agreement.  A rejected or invalid shape stays
in coverage and contributes `1.000x` to the deployed geometric mean.

## 6. Private production Table 1 contract

The production ranking model, B200 shape list, ADS source, training data, and
complete deployment harness are private.  The owner must provide:

- checkpoint/config/tokenizer revision and a model/data manifest;
- the exact production shape list and batch/sequence semantics;
- the eight generated kernel sites and their source fingerprints;
- eager and compiled baseline commands;
- correctness oracle and parameter/update agreement checks;
- timing definition for complete training-step QPS and peak memory;
- ADS source fingerprint and whether it matches the pinned public revision.

If the ADS source is a snapshot or the source audit is incomplete, set
`publication_eligible=false`; do not use its numbers as a production claim.

## 7. Dataset and input checklist

Before timing a complete-model task, the worker must answer these questions in
the run note:

1. Which checkpoint revision and tokenizer revision are loaded?
2. Which dataset and split are used (synthetic IDs, WikiText-2, or private
   production data)?
3. What is the dataset/preprocessing hash and sample/seed order?
4. For decode, how was the KV cache created and what are $B$, $T$, generated
   tokens, mask, GQA layout, and cache dtype?
5. For training, what are $B$, $S$, packing, accumulation, optimizer settings,
   number of steps, and loss/gradient checks?
6. Are all compared arms using identical inputs, weights, cache, precision,
   and CUDA-graph policy?

An unanswered item is `blocked`, not an invitation to use a convenient local
dataset.  Never commit weights, private data, credentials, raw host paths, or
tokens to this repository.
