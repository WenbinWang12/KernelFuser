# KernelFuser

First-party benchmark code and sanitized per-configuration results for the
paper's isolated kernel-fusion experiments E1, E3a, and E4.

This repository is a scoped research artifact, not a dump of the working
directory. It contains no manuscript draft, model weights, private datasets,
raw execution receipts, profiler traces, credentials, session transcripts, or
third-party source trees. The manuscript fragment in the working directory
contains internal production results and is intentionally not included. The
CSV files in `results/` retain only the fields needed to reproduce the
reported ratios and figures.

## Contents

- `src/e1_qk_publication.py`: Q/K RMSNorm-to-attention forward census.
- `src/e3a_moe_publication.py`: expert-local MoE RMSNorm/GEMM forward and
  backward census.
- `src/moe_staged_strict.py` and `src/norm_into_gemm_kernel.py`: E3a reference,
  staged-statistics, and fused-kernel implementations.
- `src/e4_epilogue_publication.py`, `src/epilogue_sites.py`, and
  `src/fusion_harness.py`: residual-add GEMM epilogue benchmark.
- `results/`: sanitized, row-level E1, E3a, and E4 speedup data.

E4 evaluates four projection geometries, not four named models:
two `down_proj` geometries and two `o_proj` geometries. Model-level claims
should not be inferred from these synthetic geometry tests.

## Reproduction

The benchmark runners require Linux, a supported NVIDIA GPU, CUDA, PyTorch,
and Triton. E1's recorded environment is PyTorch 2.6.0+cu126 and Triton 3.2.0;
reproduce the matching CUDA-enabled environment for comparable measurements.
The E1/E4 timing receipts were collected on H100 and H200 GPUs. E3a's
publication-eligible receipt is H200; the H100 run remains pending audit and
is intentionally not included in `results/`.

Install a PyTorch build compatible with the installed NVIDIA driver/CUDA stack,
then install a compatible Triton release. From the repository root:

```bash
mkdir -p outputs
python src/e1_qk_publication.py --output outputs/e1.json
python src/e3a_moe_publication.py \
  --out outputs/e3a.json --trace-dir outputs/e3a-traces
python src/e4_epilogue_publication.py \
  --out outputs/e4.json --trace-dir outputs/e4-traces \
  --sites gemm_residual --M 128,512,2048,8192
```

These commands perform long-running GPU benchmarks and write full receipts and
Chrome traces under `outputs/`. Those generated files can contain machine
identifiers and local paths; they are ignored by Git and should not be
published without a separate review. Do not commit raw receipts or traces.

The scripts preserve correctness gates and record negative results; a run is
not evidence of a speedup unless its correctness and timing checks pass. GPU
architecture, compiler versions, power state, and contention affect timing.
Use the source receipt protocol and rerun on the intended hardware before
making new claims.

## Results and claim boundaries

All ratios use `baseline / fused`; values above 1 indicate a faster fused arm.
See [`results/RESULTS.md`](results/RESULTS.md) for aggregation rules and
coverage caveats.

- E1 is an isolated, forward-only operator census; it is not an end-to-end
  model result.
- E3a is an expert-local forward/backward kernel study without routing or a
  parameter update.
- E4 is an isolated forward-only residual epilogue study.

## License

First-party code in this repository is released under the MIT License. The
paper's publication materials, model names, and external software remain
subject to their respective copyrights and licenses. PyTorch, Triton, CUDA,
and NVIDIA tooling are separate dependencies and are not relicensed here.
