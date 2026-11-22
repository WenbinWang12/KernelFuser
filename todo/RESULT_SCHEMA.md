# Required result schema

This is the minimum schema for a promoted measurement.  JSON may contain richer
fields, but the sanitized CSV must retain these columns or an equivalent
explicit field mapping.

## Identity and provenance

`run_id`, `site_family`, `model`, `gpu`, `backend`, `mode`, `optimizer`,
`dtype`, `B`, `S`, `M`, `K`, `N`, `head_dim`, `causal`, `commit`,
`kernel_source`, `kernel_sha256`, `torch_version`, `cuda_version`,
`triton_version`, `tlx_version`, `seed_list`.

## Correctness and status

`status`, `failure_reason`, `correctness_pass`, `max_abs_err`, `max_rel_err`,
`relative_l2`, `logits_top1_agreement`, `loss_agreement`, `grad_agreement`,
`unsupported_reason`.

## Timing and speedup

`warmup`, `timing_rounds`, `launches_per_sample`, `candidate_p50_us`,
`baseline_p50_us`, `anchor_p50_us`, `vendor_p50_us`, `retest_candidate_p50_us`,
`retest_baseline_p50_us`, `speedup_vs_strongest`, `speedup_vs_anchor`,
`retest_speedup`, `confirmed_1pct`, `tokens_per_s`, `samples_per_s`,
`step_time_ms`, `peak_memory_mb`.

## Coverage and evidence

`valid_shapes`, `kept_shapes`, `rejected_shapes`, `compile_failures`,
`trace_count`, `trace_manifest`, `gpu_exclusive`, `clock_setting`,
`power_setting`, `profiler_metrics`, `publication_eligible`.

Values above one mean the candidate is faster.  A rejected or invalid shape
must retain its status and must not be silently removed from coverage or a
deployed geometric mean.  `publication_eligible=false` is mandatory for
snapshot diagnostics, source-version mismatches, or incomplete gates.
