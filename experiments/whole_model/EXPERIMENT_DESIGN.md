# Experiment design of the open-model campaign

Written 2026-11-22. The stage list of the A800 queue is generated from `analysis/campaign.js`, which
encodes the matrix below. `node analysis/campaign.js status results_a800` prints what is finished.

## What the experiments test

The paper's model of a fusion is `speedup = 1 / ((1 - f)(1 + c))`, bounded by the headroom of the anchor:
`f` is the share of unfused time that the fusion removes, `c` the overhead it adds to the anchor.
Three predictions follow, and each group of experiments tests one of them.

| Prediction | Tested by |
|---|---|
| P1. Fusion pays where memory-bound work is a large share of the step and the anchor is small: token-by-token decode, the loss at the LM head, short or batched attention. | D, D-long, H, T (small tier), A |
| P2. The gain shrinks as GEMMs or attention take over: wider models, longer sequences, larger models in training. | D (by width), T (by tier), A-long |
| P3. A fused kernel can lose even when the fusion itself wins, if its anchor trails the vendor kernel. | A and A-long (three baselines: same anchor, `torch.compile`, eager SDPA), prefill |

## Factors

| Factor | Levels |
|---|---|
| Workload | decode step, training step, prefill, isolated kernel |
| Model tier | small (<1B): SmolLM2-135M/360M, Gemma-3-270M, Qwen2.5-0.5B, Qwen3-0.6B. Medium (1-8B): Gemma-3-1B, Qwen2.5-1.5B, SmolLM2-1.7B, Qwen3-1.7B, Qwen3-4B, Gemma-3-4B. Large (>=8B): Llama-3.1-8B, Qwen3-8B, Qwen3.5-9B, Qwen3-14B |
| Inputs | WikiText-2 (training windows and decode prompts). Earlier runs with seeded token ids are kept as a second input set |
| Optimizer (training) | AdamW; Muon (hidden matrices, AdamW for embedding, head and gains); SGD where optimizer state does not fit on one 80 GB GPU |
| Software stack | PyTorch 2.5.1 / Triton 3.1 (main), 2.11 / 3.6, 2.15 / 3.9 |
| GPU | A800 now. H100/H200: same matrix, when the hosts are available |

## Groups (the matrix)

| Group | Experiment | Cells | Baselines | Stage names |
|---|---|---|---|---|
| T-AdamW | Training step with the fused loss, 100 timed steps | 11 models, 8x512 (Qwen3-0.6B at three shapes; 4B at 4x512); 8B at 1x512 | eager, `torch.compile` | `train3_<model>` |
| T-Muon | Same, Muon | Qwen3-0.6B, SmolLM2-360M, Qwen3-1.7B, Qwen3-4B, Qwen3-8B | same | `train3_<model>_muon` |
| T-SGD | Same, SGD | Qwen3-8B, Llama-3.1-8B, Qwen3.5-9B, Qwen3-14B; bridges Qwen3-0.6B and 4B | same | `train3_<model>_sgd` |
| D | Whole-model decode step, five fused kernels per layer | 11 models x {1, 4, 16 sequences} x {256, 1024 cached tokens} | eager, compile default, compile max-autotune, same kernels unfused | `dec5_<model>` |
| D-long | Decode over a cache of 2048 and 4096 tokens | Qwen3-0.6B, 4B, 8B | same | `dec5L_<model>` |
| A | Q/K norm + RoPE in attention, optimized kernel | 9 models x batch {1, 8, 32} x {128..512 tokens} x {causal, not}: complete | same anchor, `torch.compile`, eager SDPA | `census_rope_<model>` |
| A-long | Same at 1024, 2048, 4096 tokens | Qwen3-8B, Qwen3-0.6B, Gemma-3-27B x batch {1, 8} | same | `census_rope_long_<model>` |
| H | Fused loss and gradient at the LM head | 15 heads complete; + Qwen3.5-9B, Qwen3.8-27B, Qwen3.5-35B-A3B | eager, compiled | `lmhead_train*` |
| S | Repeats of D, A, H, T on PyTorch 2.11 and 2.15 | about 70 stages from the earlier plan | as above | `*_t211`, `*_t215` |

Complete before this design and not repeated: prefill with fused attention (five Qwen3 models), decode-shape MLP chain,
rejected sites (loss in the GEMM epilogue, residual epilogue), cost-model evaluation, controller ablation and policy
episodes, replicas on 1/2/4 GPUs.

## Verdict rules

- **Decode, kernels, prefill:** the candidate's error against an FP32 forward may not exceed 1.5 times the worse
  PyTorch arm's (with stated floors and caps); kept if it beats every baseline by more than 1% in the initial
  measurement and in a retest with fresh graph captures.
- **Training step** (rule chosen on 2026-11-22 at the authors' request to pick a fair one; it replaces the earlier
  per-parameter rule that used the eager reference alone as yardstick, under which `torch.compile` itself fails):
  1. one step: for every parameter, the distance of the candidate's gradient to the FP32 oracle is at most 1.5 times
     the larger distance of the two PyTorch arms (eager, compiled), floor 5e-3; loss within 1e-3;
  2. 110 steps: the largest loss difference from the reference is at most 1.5 times the larger of (reference rerun,
     compiled reference) differences; the held-out loss difference at most 1.5 times that of the compiled reference;
  3. speed: the median of the per-step ratio reference/candidate exceeds 1.01 against eager and against compiled, in
     the initial measurement and in the retest.
  The earlier rule's result is still recorded in every receipt.

## Not covered, and why

- Fused decode for the hybrid-attention (Qwen3.5, Qwen3.8) and MoE checkpoints: needs new functional decoders.
- Whole-model training steps of Qwen3.8-27B and Qwen3.5-35B-A3B: need four GPUs jointly; the fused loss gains
  0.1% at 14B, so the expected result is parity. Their heads are measured in H.
- A vendor-grade attention anchor (TLX on Hopper) for long sequences: not possible on A800; first experiment for
  H200 when it is available.
