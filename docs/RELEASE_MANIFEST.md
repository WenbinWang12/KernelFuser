# Release manifest

This tree is the frozen anonymous review snapshot. It is intentionally a
single-root artifact with no parent history, private deployment scripts,
credentials, checkpoints, or raw profiler traces.

Included:

- source runners under `src/`;
- small public shape/configuration inputs under `data/`;
- sanitized CSV summaries under `results/`;
- smoke and reproduction instructions under `README.md`, `docs/`, and
  `experiments/`.

Excluded:

- model weights and licensed checkpoints;
- raw traces, large archives, and machine-local receipts;
- scheduler, SSH, cluster-wait, and deployment scripts;
- hostnames, account names, GPU UUIDs, absolute paths, and Git history.

The result tables preserve diagnostic and blocked rows. A row is a paper claim
only when its receipt and the corresponding protocol satisfy the stated gates.
The anonymous mirror URL is supplied separately in the review submission.
