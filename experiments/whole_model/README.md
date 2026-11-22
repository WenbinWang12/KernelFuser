# Whole-model and optimized-kernel experiments

Runners and analysis scripts of the open-model campaign of November 2026. The design (what each group of
experiments tests, the factor levels and the verdict rules) is in [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md);
the measured rows are in [`results/campaign_2026_11`](../../results/campaign_2026_11).

## Runners

| File | What it measures |
|---|---|
| `e2e_decode2.py` (with `e2e_decode.py`, `decode_mlp.py`) | One decode step of a whole model (model types llama, qwen2, qwen3) with five fused kernels per layer, against eager PyTorch, `torch.compile` in its default and max-autotune modes, and the same kernels without fusion. `--data FILE` takes the prompts from a text file; `--cpu-check` compares the functional step with transformers. |
| `train_step_ce.py` (with `lmhead_kernels.py`) | A whole-model training step in which the loss and its backward pass run as one fused kernel. One-step gradient check against an FP32 oracle, per-step loss trajectories, held-out perplexity, paired step timing against eager and compiled references, retest, self-comparison. Optimizers: AdamW, SGD, Muon. |
| `lmhead_bench.py` | The LM head with its loss and backward pass in isolation, over the (hidden, vocabulary) shapes of released models. |
| `recensus_driver.py` (with `e1_rope_refined_census.py`, `e1_rope_refined_impl.py`, `e1_qk_publication_v3_refined.py`) | Q/K normalization, with or without RoPE, fused into attention, using the kernel after the optimization steps. Baselines: the same attention kernel after separate kernels, `torch.compile`, eager SDPA. |
| `e2e_prefill.py` | Whole-model prefill with the fused attention kernel, every arm a CUDA-graph replay. |
| `queue_v3.sh` | Stage queue with one worker per GPU. A stage starts only on an idle GPU and stops if another process appears on it. |

Every runner writes a JSON receipt with its protocol, source hashes, correctness numbers, all timing arms and the
verdict. `slim.py` drops the raw sample arrays; `summarize.js` prints one block per receipt.

## Running

Requirements: Linux, an NVIDIA GPU, PyTorch with Triton, `transformers` and `safetensors`, and local copies of the
checkpoints (`<models>/<name>/config.json` and weights).

```bash
python e2e_decode2.py --model-dir $MODELS/Qwen3-0.6B --autotune --data wikitext2_train.txt --output out/dec.json
python train_step_ce.py --model-dir $MODELS/Qwen3-0.6B --shapes 8x512 --steps 100 --warmup 10 \
    --data wikitext2_train.txt --retest --aa --compile --lean --output out/train.json
python recensus_driver.py --mode rope --models Qwen3-8B --sizes 128,256,384,512 --batches 1,8,32 \
    --causal-modes false,true --bn 128 --output out/census.json
```

The whole matrix as a queue plan (paths come from the environment):

```bash
KF_MODELS=/path/to/models KF_TEXT=/path/to/wikitext2_train.txt node analysis/campaign.js plan plan.tsv
node analysis/campaign.js status <directory with receipts>
```

## Analysis

| Script | Output |
|---|---|
| `analysis/build_public_data.js` | The CSV tables of `results/campaign_2026_11` and `docs/assets/data.js` of the project page. |
| `analysis/gen_results_md.js` | The summary `README.md` of the results folder. |
| `analysis/train3_numbers.js`, `analysis/recensus_numbers.js` | Per-row printouts of the training and census receipts. |
| `analysis/longseq_diag.js` | Splits the result at each sequence length into what the fusion adds over its own anchor and how the anchor compares with the PyTorch attention paths. |

## Scope

The decode and loss kernels in this folder were written with a coding agent directed by an author; they are
measurements of fused kernels under the gates, not outputs of the four-stage synthesis pipeline. Inputs are
WikiText-2 text or seeded token ids, as recorded in each row. Results hold for the GPU and software stack named
in the row.
