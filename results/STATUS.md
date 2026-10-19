# Review snapshot status

The CSV summaries in this directory are the shareable result surface. The
following boundaries are part of the claims and should be carried into the
paper or artifact appendix.

| Track | Status in the recorded campaign | Claim boundary |
|---|---|---|
| E1 / E1-RoPE | Strict isolated rows pass on both GPU families | Forward operator study; no full-model claim. |
| E3a | Strict expert-local rows pass | No routing, checkpoint, or update claim. |
| E3b | Strict real-checkpoint expert-layer rows pass | Declared expert-layer scope only. |
| E3c | Checkpoint E2E evidence exists for the recorded H200 protocols | H100 checkpoint E2E is not represented as complete. |
| E6 | Hopper-equivalent forward rows pass | Not a native Blackwell/TLX result. |
| E7 | A/A noise-floor rows pass | Control study, not an optimization claim by itself. |
| E8 | Isolated forward rows pass; model-head outcomes remain separately labeled | Forward-only and geometry-specific. |
| E9 | Hopper-equivalent OLMoE E2E evidence passes | Backend is not a native Blackwell/TLX implementation. |
| E10 | Resource-safe forward ablation passes | No backward, update, or full-model claim. |
| A / B / C / E2 | Passing, diagnostic, and blocked outcomes are represented in the summaries | Do not promote a diagnostic or blocked row to a speedup claim. |

The complete private campaign archive contains raw receipts and traces. This
anonymous snapshot intentionally includes only sanitized CSV summaries and
does not expose hostnames, account names, GPU UUIDs, process IDs, checkpoints,
or cluster paths.
