# Required result schema

This is the minimum schema for a promoted measurement.  JSON may contain richer
fields, but the sanitized CSV must retain these columns or an equivalent
explicit field mapping.

## Identity and provenance

`run_id`, `site_family`, `model`, `model_id`, `model_revision`,
`tokenizer_revision`, `config_sha256`, `gpu`, `backend`, `mode`, `optimizer`,
`dtype`, `input_protocol`, `dataset_revision`, `dataset_sha256`, `B`, `S`,
`T`, `M`, `K`, `N`, `head_dim`, `causal`, `layout`, `commit`,
`kernel_source`, `kernel_sha256`, `torch_version`, `cuda_version`,
`triton_version`, `tlx_version`, `seed_list`.

## Correctness and status

`status`, `failure_reason`, `correctness_pass`, `oracle_type`, `finite_check`,
`allclose_atol`, `allclose_rtol`, `max_abs_err`, `max_rel_err`, `relative_l2`,
`logits_top1_agreement`, `nll_agreement`, `loss_agreement`, `grad_agreement`,
`parameter_update_agreement`, `unsupported_reason`.

## Timing and speedup

`warmup`, `timing_rounds`, `timing_samples`, `launches_per_sample`,
`candidate_p50_us`, `candidate_p95_us`, `baseline_p50_us`, `anchor_p50_us`,
`vendor_p50_us`, `retest_candidate_p50_us`, `retest_baseline_p50_us`,
`speedup_vs_strongest`, `speedup_vs_anchor`, `retest_speedup`,
`confirmed_1pct`, `gate_verdict`, `kept`, `deployed_speedup`, `tokens_per_s`,
`samples_per_s`, `step_time_ms`, `forward_time_ms`, `backward_time_ms`,
`optimizer_time_ms`, `peak_memory_mb`.

## Coverage and evidence

`valid_shapes`, `kept_shapes`, `rejected_shapes`, `compile_failures`,
`timeout_count`, `unsupported_count`, `trace_count`, `trace_manifest`,
`gpu_exclusive`, `gpu_name`, `gpu_index_private`, `clock_setting`,
`power_setting`, `profiler_metrics`, `publication_eligible`.

Values above one mean the candidate is faster.  A rejected or invalid shape
must retain its status and must not be silently removed from coverage or a
deployed geometric mean.  `gate_verdict` must say `keep` or `reject`, even when
the correctness check passed.  `deployed_speedup` is `1.000` for a rejected
shape.  Use `tested_pass`, `tested_invalid`, `tested_fail`, `compile_fail`,
`timeout`, `N/A`, or `blocked` for `status`; do not encode these states as a
blank speedup.  `publication_eligible=false` is mandatory for snapshot
diagnostics, source-version mismatches, incomplete gates, or non-paired runs.
