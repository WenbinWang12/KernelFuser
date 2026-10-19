# KernelFuser: reproducible kernel-fusion artifact

This is an anonymous review snapshot for a kernel-fusion study. It contains
the implementation, shape-only public inputs, sanitized summary tables, and
commands for the smoke checks. It intentionally excludes model weights, raw
profiler traces, cluster launch scripts, credentials, and Git history.

The artifact reports several distinct scopes:

| Track | Scope in this snapshot |
|---|---|
| E1 / E1-RoPE | Isolated forward attention and Q/K RMSNorm+RoPE studies. |
| E3a / E3b | Expert-local MoE kernel studies; no routing or full-model speedup claim. |
| E3c | Checkpoint E2E evidence only where the corresponding receipt says so. |
| E6 / E9 | Hopper-equivalent implementations; not native Blackwell/TLX claims. |
| E7 / E8 / E10 | A/A controls, forward kernel studies, and resource ablations with the claim boundaries in the tables. |
| A / B / C / E2 | The summary tables retain passing, diagnostic, and blocked outcomes; a diagnostic result is not promoted to a publication claim. |

## Reproduce

The runners require Linux, Python 3.10 or newer, a supported NVIDIA GPU,
CUDA, PyTorch, and Triton. Install versions appropriate for the target GPU,
then run the smoke commands from the repository root:

```bash
python -m pip install torch triton
mkdir -p outputs
python src/e1_rope_publication_v1.py --smoke \
  --output outputs/e1_rope_smoke.json
python src/b_qkv_output_norm_rope.py --smoke \
  --out outputs/b_qkv_smoke.json
python src/e7_rope_aa.py --batches 1 --sizes 128 --rounds 1 --samples 3 \
  --out outputs/e7_smoke.json
```

The smoke commands are functional checks, not a substitute for the measured
publication protocol. Full sweep commands and protocol details are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). They write receipts to
`outputs/`, which is ignored and may contain machine-specific metadata.

## Reading the results

The CSV files under `results/` are sanitized summaries. `tested_pass` means
that the gates declared by that runner passed. `publication_eligible` and the
claim boundary must also be checked before using a row in a paper. The source
and results deliberately preserve blocked and diagnostic outcomes instead of
silently turning them into positive claims.

The anonymous-submission procedure and final audit are in
[`docs/ANONYMOUS_SUBMISSION.md`](docs/ANONYMOUS_SUBMISSION.md). The anonymous
review URL, once created, belongs in [`CODE_LINK.txt`](CODE_LINK.txt); this
file is intentionally a placeholder until the exact frozen branch is uploaded
to the anonymous mirror.

## License

First-party code is MIT unless a subdirectory states otherwise. The ADS-derived
SM90 port carries its own Apache-2.0 notice. PyTorch, Triton, CUDA, and NVIDIA
tooling remain separate dependencies.
