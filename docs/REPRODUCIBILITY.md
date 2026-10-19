# Reproducibility protocol

Run commands from the repository root on a clean Linux checkout. Use a fresh
environment for each GPU family and record the versions of Python, PyTorch,
Triton, CUDA, and the GPU driver in the receipt. Set `CUDA_VISIBLE_DEVICES`
to a free device; the runners fail closed when their physical-GPU checks or
correctness gates fail.

The main measured entry points are:

```bash
python src/e1_rope_publication_v1.py --output outputs/e1_rope.json \
  --trace-dir outputs/e1_rope_traces
python src/e1_qk_publication_v3.py --output outputs/e1_qk.json
python src/e7_aa_noise_floor_osu_v3.py --out outputs/e7_aa.json \
  --trace-dir outputs/e7_aa_traces
python src/e3a_moe_strict.py --out outputs/e3a.json --trace-dir outputs/e3a_traces
python src/e3b_moe_strict.py --model-path /path/to/checkpoint \
  --model-id model-id --out outputs/e3b.json --trace-dir outputs/e3b_traces
```

Replace `/path/to/checkpoint` with a locally acquired, licensed checkpoint.
No checkpoint, cluster path, scheduler command, or SSH configuration is part
of this artifact. The published tables are the sanitized CSVs in `results/`.

Correctness must be checked before timing. A row is retained as a publication
result only when its receipt records the declared correctness, paired timing,
independent retest, provenance, and exclusivity gates. Hopper-equivalent and
forward-only rows remain labeled as such.

For a new run, write outputs below `outputs/` and do not commit them. Before
sharing a receipt, remove hostnames, process IDs, GPU UUIDs, absolute paths,
checkpoint paths, and trace filenames; keep only the metrics needed to audit
the claim.
