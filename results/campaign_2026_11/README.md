# Open-model campaign, November 2026

Generated 2026-11-22 by `experiments/whole_model/analysis/gen_results_md.js` from the measurement receipts.
The CSV files in this folder hold every measured row; receipts themselves are not published because they carry host identifiers.
A configuration is *kept* only if it passes the correctness gate and beats every unfused baseline by more than 1% in an initial measurement and in a retest.

## Whole-model decode step (`decode_step_a800.csv`)

One new token per sequence over a key/value cache, five fused kernels per layer, against eager PyTorch, `torch.compile` (default and max-autotune) and the same kernels without fusion. A800.

| Inputs | Models | Rows | Valid | Kept | Kept rows vs torch.compile | Fusion alone (vs same kernels) |
|---|---:|---:|---:|---:|---|---|
| seeded token ids | 10 | 50 | 47 | 44 | 1.09-1.45x | 1.01-1.22x |
| WikiText-2 prompts | 4 | 20 | 19 | 19 | 1.06-1.26x | 1.01-1.16x |

| Model (seeded token ids) | Layers | Width | Kept | vs torch.compile | Fusion alone | vs eager |
|---|---:|---:|---|---|---|---|
| SmolLM2-135M | 30 | 576 | 5/5 | 1.21-1.42x | 1.16-1.22x | 2.55-3.67x |
| SmolLM2-360M | 32 | 960 | 5/5 | 1.32-1.45x | 1.17-1.21x | 2.55-3.23x |
| Qwen2.5-0.5B | 24 | 896 | 4/5 | 1.15-1.27x | 1.15-1.18x | 2.25-2.60x |
| Qwen3-0.6B | 28 | 1024 | 4/5 | 1.17-1.27x | 1.10-1.16x | 2.55-3.38x |
| Qwen2.5-1.5B | 28 | 1536 | 5/5 | 1.09-1.18x | 1.10-1.11x | 1.75-1.96x |
| SmolLM2-1.7B | 24 | 2048 | 4/5 | 1.16-1.30x | 1.06-1.08x | 1.71-2.00x |
| Qwen3-1.7B | 28 | 2048 | 5/5 | 1.16-1.25x | 1.08-1.10x | 2.13-2.53x |
| Qwen3-4B | 36 | 2560 | 5/5 | 1.12-1.15x | 1.06-1.06x | 1.76-1.98x |
| Llama-3.1-8B | 32 | 4096 | 2/5 | 1.05-1.15x | 0.99-1.01x | 1.36-1.48x |
| Qwen3-8B | 36 | 4096 | 5/5 | 1.09-1.12x | 1.03-1.04x | 1.51-1.62x |

## Attention census with the optimized fused kernel (`attention_census_optimized_kernel.csv`)

Fused kernel against the best of: the same attention kernel after separate kernels, `torch.compile`, eager SDPA.

| Fusion | GPU | PyTorch | Models | Configurations | Kept | Kept rows vs best unfused |
|---|---|---|---:|---:|---:|---|
| Q/K norm | H200 | 2.5.1 | 1 | 16 | 13 | 1.03-1.08x |
| Q/K norm | H200 | 2.11.0 | 1 | 16 | 7 | 1.01-1.09x |
| Q/K norm | H200 | 2.15.0 | 1 | 16 | 9 | 1.02-1.15x |
| Q/K norm + RoPE | H200 | 2.5.1 | 6 | 144 | 118 | 1.01-1.18x |
| Q/K norm + RoPE | H200 | 2.11.0 | 1 | 24 | 18 | 1.02-1.17x |
| Q/K norm + RoPE | H200 | 2.15.0 | 1 | 24 | 22 | 1.01-1.47x |
| Q/K norm | H100 | 2.5.1 | 1 | 16 | 14 | 1.02-1.08x |
| Q/K norm + RoPE | H100 | 2.5.1 | 1 | 24 | 20 | 1.04-1.15x |
| Q/K norm | A800 | 2.5.1 | 9 | 144 | 35 | 1.01-1.04x |
| Q/K norm + RoPE | A800 | 2.5.1 | 9 | 216 | 80 | 1.01-1.19x |

## Training step on WikiText-2 with the fused loss (`training_step_wikitext2_a800.csv`)

100 timed steps per arm after 10 warm-up steps. Speedups are medians of the per-step ratio reference/fused, initial / retest.
Correct: one-step gradients within 1.5x of the worse PyTorch arm's distance to an FP32 oracle, loss trajectory and held-out loss within 1.5x of the spread between equivalent PyTorch runs.

| Model | Params | Shape | Optimizer | vs torch.compile | vs eager | Held-out perplexity (before, reference, fused) | Correct | Kept |
|---|---:|---|---|---|---|---|---|---|
| SmolLM2-135M | 0.135B | 8x512 | AdamW | 0.995 / 0.996 | 1.001 / 1.003 | 29.382, 28.496, 28.493 | yes | no |
| Qwen3-0.6B | 0.596B | 16x256 | AdamW | 1.014 / 1.013 | 1.076 / 1.077 | 43.986, 22.644, 22.66 | yes | yes |
| Qwen3-0.6B | 0.596B | 8x512 | AdamW | 1.014 / 1.021 | 1.076 / 1.075 | 33.393, 19.414, 19.408 | yes | yes |
| Qwen3-0.6B | 0.596B | 4x1024 | AdamW | 1.015 / 1.014 | 1.073 / 1.073 | 27.38, 17.204, 17.203 | yes | yes |
| Qwen3-1.7B | 1.721B | 8x512 | AdamW | 1.005 / 1.006 | 1.039 / 1.039 | 25.634, 14.224, 14.223 | yes | no |
| Qwen3-4B | 4.022B | 4x512 | AdamW | 1.000 / 1.000 | 1.015 / 1.015 | 24.194, 13.344, 13.341 | yes | no |
| Qwen3-8B | 8.191B | 2x512 | SGD | 1.002 / 1.002 | 1.010 / 1.010 | 14.237, 13.686, 13.672 | yes | no |
| Qwen3-14B | 14.768B | 1x512 | SGD | 1.001 / 1.001 | 1.004 / 1.005 | 11.67, 11.325, 11.327 | yes | no |

## Fused loss and gradient at the LM head (`fused_loss_lm_head.csv`)

| GPU | PyTorch | Set | Heads x token counts | Kept | vs compiled head | vs eager head |
|---|---|---|---:|---:|---|---|
| H200 | 2.5.1 | lmhead_train | 20 | 20 | 1.01-1.09x | 1.36-2.75x |
| H200 | 2.11.0 | lmhead_train_t211 | 20 | 20 | 1.04-1.21x | 1.30-2.65x |
| H200 | 2.15.0 | lmhead_train_t215 | 20 | 20 | 1.02-1.09x | 1.26-2.43x |
| A800 | 2.5.1 | lmhead_train | 20 | 17 | 1.01-1.08x | 1.18-2.39x |
| A800 | 2.5.1 | lmhead_train_more | 10 | 9 | 1.01-1.08x | 1.14-2.47x |

## Whole-model prefill with the fused attention kernel (`prefill_whole_model.csv`)

| GPU | Rows | Kept | Fused vs fastest unfused model |
|---|---:|---:|---|
| H200 | 24 | 2 | 0.98-1.02x |
| A800 | 23 | 0 | 0.99-1.01x |

Attention is a small share of prefill, so the kernel-level gain does not carry over; the gate keeps the unfused model.
