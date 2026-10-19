# Reproduction helpers

The review artifact keeps only generic, host-independent helpers. Cluster
waiters, SSH deployment scripts, scheduler commands, and local account paths
are intentionally excluded. Set `CUDA_VISIBLE_DEVICES` and the environment
appropriate for the machine before running a benchmark.

Use the commands in the root README and `docs/REPRODUCIBILITY.md`; every
output path is relative to the checkout.
