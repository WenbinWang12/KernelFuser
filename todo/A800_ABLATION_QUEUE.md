# A800 figure-only ablation queue

These cards are diagnostic sweeps for the batch-size, sequence-length,
cache-length, token-count, and optimizer-shape figures. They reuse the EIT A800
runners and their correctness/timing gates, but they are not headline Table 2
rows. Keep their receipts separate from the canonical 100-step training rows.

## Fixed protocol

- One exclusive A800 per stage; BF16; PyTorch 2.5.1/Triton 3.1 (`formal`).
- Attention uses `recensus_driver.py`, causal and non-causal arms, an FP32
  oracle, paired compiled/vendor baselines, and traces.
- Decode uses `e2e_decode2.py` with WikiText-2 prompt windows and `--autotune`.
- Prefill uses `e2e_prefill.py`'s seeded-token protocol; it does not accept
  `--data`.
- LM-head `train` includes backward; `loss` is forward-only.
- Training cards use WikiText-2, paired eager/compiled arms, `--retest --aa
  --compile --lean`, 30 steps and 5 warm-ups. They are diagnostic curves, not
  100-step headline results.

Every attempted point keeps `tested_pass` or `tested_invalid`, correctness,
initial/retest timing, GPU exclusivity, source hash, and exact shape. Invalid
points are missing points in plots, never zero-speedup measurements.

## Queued cards

| Cards | Fixed factor | Sweep |
|---|---|---|
| `plot_attn_q06_seq`, `plot_attn_q08_seq` | B={1,8}, causal={false,true} | S={64,128,256,384,512,768,1024,1536,2048} |
| `plot_attn_q06_batch`, `plot_attn_q08_batch` | S=256, causal={false,true} | B={1,2,4,8,16,32}; q06 also 64 |
| `plot_decode_q06_seq`, `plot_decode_q08_seq` | B=1, WikiText-2 | cached T={64,128,256,512,1024,2048}; q06 also 4096 |
| `plot_decode_q06_batch`, `plot_decode_q08_batch` | T=256, WikiText-2 | B={1,2,4,8,16,32}; q06 also 64 |
| `plot_prefill_q06`, `plot_prefill_q08` | seeded-token prefill | q06: 1x64,1x128,1x256,1x512,1x1024,4/8/16/32x128; q08: 1x64,1x128,1x256,1x512,4/8x128 |
| `plot_lmhead_q06_train`, `plot_lmhead_q08_train` | LM-head backward | M={256,512,1024,2048,4096,8192} |
| `plot_lmhead_q06_loss`, `plot_lmhead_q08_loss` | LM-head forward | M={256,512,1024,2048,4096,8192} |
| `plot_train_q06_adamw_shapes`, `plot_train_q06_muon_shapes` | Qwen3-0.6B; BxS diagnostic grid | 1/2/4/8/16/32x128, 1/2/4/8x256, 1x1024, 2x512, 4x512 |
| `module_mlp_q06`, `module_mlp_q08` | One checkpoint MLP module; one decoder layer | M={1,2,4,8,16,32,64} rows, `--layers 1` |

The five canonical Table 2 Muon follow-ups remain separate. Do not change their
optimizer, shape, steps, or output name while these cards run.

## Plot extraction

- Attention: use `B`, `S`, `causal`, `best`, `same_anchor`, `kept`, `status`,
  and retest ratios. Plot fusion-vs-anchor and fusion-vs-strongest-baseline as
  separate curves.
- Decode: use `B`, `T`, `status`, `kept`, `speedup_vs_fastest`,
  `speedup_vs_unfused`, and per-arm p50 latency.
- Prefill: use `B`, `S`, correctness status, p50 latency, and candidate ratios.
- LM-head: use `M`, mode, candidate/compiled/eager latency, speedup, and gate.
- Training: use `B`, `S`, optimizer, correctness, trajectory/held-out checks,
  and paired ratios. Never call a 30-step point a headline speedup.

Export one sanitized CSV per figure with one row per attempted shape and a
status column. Do not connect missing or invalid points. Keep model, causal flag,
optimizer, backend, and GPU as separate facets and report valid/kept counts.
Only sanitized receipts/CSVs may be copied to the local snapshot; do not commit
weights, credentials, raw prompts, host paths, or unredacted logs.
