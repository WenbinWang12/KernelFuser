#!/usr/bin/env python3
"""A: pinned model-shaped lazy RMSNorm/projection census.

Weights are synthetic BF16 tensors at public, immutable config geometries.
Gamma is folded into the weights once, outside all timings. Raw receipts,
compiler code and traces are private output artifacts; public CSVs contain
only an allowlist of fields. Development stacks remain diagnostic.
"""
from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["candidate", "unfused_vendor_anchor", "torch_compile_default",
        "torch_compile_max_autotune", "cublas_or_cublaslt"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    files = [Path(__file__), ROOT / "src/fusion_harness.py",
             ROOT / "src/tlx_ads_sm90_port/ads_sm90_rmsnorm_matmul.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return commit, combined, hashes


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values)-1, math.ceil(p * len(values))-1)]


def summarize(values):
    return {"p50_ms": statistics.median(values) / 1000,
            "p95_ms": percentile(values, .95) / 1000,
            "samples_us": values}


def event_paired(functions, rounds, samples, replays, seed):
    """CUDA-event operator latency, with the requested compiler modes intact.

    Max-autotune can own an internal CUDA graph. Calls are not nested inside
    an external capture. All arms include their normal dispatch path and
    receive identical repeated-call/event timing and warmup policies.
    """
    import torch
    rng = random.Random(seed)
    values = {name: [] for name in functions}
    orders = []
    for _ in range(rounds):
        for call in functions.values():
            for _ in range(20):
                call()
        torch.cuda.synchronize()
        for _ in range(samples):
            order = list(functions)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(replays):
                    functions[name]()
                end.record()
                end.synchronize()
                values[name].append(start.elapsed_time(end) * 1000 / replays)
    return {"arms": {name: {"median_us": statistics.median(times), "samples_us": times}
                     for name, times in values.items()},
            "protocol": {"mode": "cuda_event_repeated_operator_calls", "rounds": rounds,
                         "samples_per_round": samples, "calls_per_sample": replays,
                         "warmups_per_round": 20, "arm_order": orders, "seed": seed,
                         "includes_dispatch": True, "external_cuda_graph_capture": False,
                         "candidate_output_ownership": "fresh_output_and_scalar_scratch_per_operator_call"}}


def write_public(rec, path):
    fields = ["device", "stack", "model", "site", "M", "K", "N", "seed", "status",
              "publication_eligible", "reason", "correctness_pass", "max_relative_l2",
              "anchor_ratio", "noise_margin", "speedup_vs_fastest", "retest_speedup",
              "fastest_baseline", "compiler_status", "exclusivity_pass", "source_commit",
              "source_sha256", "config_manifest_sha256", "trace_sha256"]
    fields += [f"{arm}_{stat}_ms" for arm in ARMS for stat in ["p50", "p95"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rec["rows"]:
            public = {key: row.get(key) for key in fields}
            public.update(device="H100", stack=rec.get("stack"), source_commit=rec["source_commit"],
                          source_sha256=rec["source_sha256"], config_manifest_sha256=rec["config_manifest_sha256"])
            public["reason"] = ";".join(row.get("gate_reasons", []))
            public["trace_sha256"] = json.dumps({k:v["sha256"] for k,v in row.get("traces", {}).items()}, sort_keys=True)
            for arm, result in row.get("arm_statistics", {}).items():
                for stat in ["p50", "p95"]:
                    public[f"{arm}_{stat}_ms"] = result[f"{stat}_ms"]
            writer.writerow(public)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-tag", default="h100")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/abc_model_manifest.json")
    parser.add_argument("--models", default="")
    parser.add_argument("--sites", default="qkv,gate_up")
    parser.add_argument("--m", default="1,8,32,128,512,2048,8192")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--allow-development", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    if args.device_tag.lower() != "h100":
        parser.error("This implementation is pre-registered for H100")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    configs = manifest["models"]
    if args.models:
        requested = args.models.split(",")
        if set(requested) - {x["model"] for x in configs}:
            parser.error("Unknown model name")
        configs = [x for x in configs if x["model"] in requested]
    sites = args.sites.split(",")
    if not set(sites) <= {"qkv", "gate_up"}:
        parser.error("Sites must be qkv and/or gate_up")
    sizes = [int(x) for x in args.m.split(",")]
    if any(x <= 0 for x in sizes) or args.rounds < 2 or args.samples < 2 or args.replays < 1:
        parser.error("Positive shapes and at least two rounds/samples required")
    scope = [(cfg, site, m) for cfg in configs for site in sites for m in sizes]
    source_commit, source_sha256, files = source_identity()
    public_source = bool(subprocess.check_output(["git", "branch", "-r", "--contains", "HEAD"], cwd=ROOT, text=True).strip())
    rec = {"schema": "abc_a_lazy_rmsnorm_projection_v1", "track": "A", "status": "running",
           "publication_eligible": False, "source_commit": source_commit, "source_sha256": source_sha256,
           "source_files": files, "source_public_remote_ref_present": public_source,
           "config_manifest_sha256": sha(args.manifest), "expected_rows": len(scope),
           "scope": "synthetic BF16 weights at pinned public model projection geometries; isolated forward only",
           "contract": "lazy RMSNorm: gamma folded once into BF16 weight; FP32 accumulators and bias; no normalized input materialized",
           "timing_protocol": {"warmups": 20, "rounds": args.rounds, "samples": args.samples,
                               "replays": args.replays, "candidate_paired_with_each_baseline": True,
                               "fresh_retest": True, "mode": "cuda_event_repeated_operator_calls",
                               "includes_dispatch": True, "torch_compile_max_autotune_mode": "max-autotune",
                               "noise": "same-shape A/A, p95 absolute paired ratio deviation"},
           "rows": []}
    trace_root = args.trace_dir or args.out.parent / "traces"

    def save():
        args.out.write_text(json.dumps(rec, indent=2) + "\n")
        write_public(rec, ROOT / "results/abc/h100_a_development_20261015.csv" if args.allow_development else ROOT / "results/abc/h100_a_release_20261015.csv")

    try:
        import torch
        import triton
        import triton.language.extra.tlx as tlx
    except ImportError as exc:
        rec.update(status="blocked_dependency", stack="release", dependency_error=f"{type(exc).__name__}: {exc}")
        rec["environment"] = {"torch": torch.__version__ if "torch" in locals() else None,
                              "triton": triton.__version__ if "triton" in locals() else None,
                              "cuda": torch.version.cuda if "torch" in locals() else None,
                              "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "gpu": "H100; dependency check does not initialize CUDA"}
        for cfg, site, m in scope:
            rec["rows"].append({"model": cfg["model"], "site": site, "M": m,
                                "K": cfg.get("hidden_size"), "N": cfg.get(site + "_width"),
                                "status": "blocked_dependency", "publication_eligible": False,
                                "gate_reasons": ["tlx_unavailable"]})
        save()
        print("blocked_dependency: TLX unavailable;", len(scope), "rows retained", flush=True)
        return

    paired = event_paired
    from fusion_harness import paired as device_paired
    import e1_rope_publication_v1 as provenance
    from tlx_ads_sm90_port import ads_sm90_rmsnorm_matmul as kernel
    from torch._inductor.utils import run_and_get_code

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rec["environment"] = {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
                          "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "tlx_source": tlx.__file__,
                          "tlx_source_sha256": sha(tlx.__file__)}
    def development_version(version):
        local = version.partition("+")[2]
        return "dev" in version or "a0" in version or bool(local and not local.startswith(("cu", "rocm")))
    development = development_version(torch.__version__) or development_version(triton.__version__)
    rec["stack"] = "development" if development else "release"
    if development and not args.allow_development:
        raise RuntimeError("Development stack requires --allow-development and stays diagnostic")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Hopper required")

    def metric(value, reference):
        delta = value.float() - reference
        relative = float(delta.norm() / reference.norm().clamp_min(1e-12))
        finite = bool(torch.isfinite(value).all() and torch.isfinite(reference).all())
        return {"pass": finite and bool(torch.allclose(value.float(), reference, atol=.025, rtol=.025)) and relative <= .025,
                "relative_l2": relative, "max_abs": float(delta.abs().max()), "finite": finite}

    def inputs(cfg, m, n, seed):
        k = cfg["hidden_size"]
        torch.manual_seed(seed)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * k**-.5
        gamma = (1 + .1 * torch.randn(k, device="cuda", dtype=torch.bfloat16)).contiguous()
        bias = (.1 * torch.randn(n, device="cuda", dtype=torch.bfloat16)).contiguous()
        folded = (w.float() * gamma.float()[None]).bfloat16().t().contiguous()
        inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + cfg["norm_epsilon"])
        reference = (x.float() * inv * gamma.float()[None]) @ w.float().t() + bias.float()
        lazy_reference = (x.float() @ folded.float()) * inv + bias.float()
        return x, folded, bias, reference, lazy_reference

    def norm(x, eps):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)).to(x.dtype)

    def unfused(x, w, b, eps):
        return torch.addmm(b, norm(x, eps), w)

    def row_scale(x, w, b, eps):
        product = torch.mm(x, w, out_dtype=torch.float32)
        return (product * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) + b.float()).to(x.dtype)

    def capture_trace(call, path, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts = []
        for attempt in range(3):
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                with torch.profiler.record_function(label):
                    call()
                torch.cuda.synchronize()
            raw = path.with_suffix("")
            profile.export_chrome_trace(str(raw))
            payload = raw.read_bytes()
            events = json.loads(payload)["traceEvents"]
            kernels = [e for e in events if e.get("cat") == "kernel"]
            artifact = path if kernels or attempt == 2 else path.with_name(path.name + f".attempt{attempt}.gz")
            with gzip.open(artifact, "wb") as handle:
                handle.write(payload)
            raw.unlink()
            attempts.append({"path": str(artifact), "sha256": sha(artifact), "kernel_count": len(kernels)})
            if kernels:
                break
        return {"path": str(artifact), "sha256": sha(artifact), "kernel_count": len(kernels),
                "kernel_names": [e.get("name") for e in kernels], "nonempty_gpu_trace": bool(kernels),
                "capture_attempts": attempts}

    for cfg, site, m in scope:
        row = {"model": cfg["model"], "site": site, "M": m, "K": cfg.get("hidden_size"),
               "N": cfg.get(site + "_width"), "seed": 1701, "status": "running", "publication_eligible": False,
               "gate_reasons": [], "traces": {}}
        if cfg["status"] != "resolved":
            row.update(status="blocked_dependency", gate_reasons=["public_config_unavailable"])
            rec["rows"].append(row); save(); continue
        if cfg.get("a_semantic_status") == "unsupported_norm_contract":
            row.update(status="unsupported_shape", gate_reasons=["model_normalization_is_layernorm_not_rmsnorm"])
            rec["rows"].append(row); save(); continue
        before = provenance.gpu_snapshot()
        row["gpu_exclusivity"] = {"before": before}
        if not before["exclusive"]:
            raise RuntimeError("Assigned GPU is occupied; defer job until exclusive")
        mon = provenance.ExclusiveMonitor().start()
        try:
            torch._dynamo.reset()
            k, n, eps = row["K"], row["N"], cfg["norm_epsilon"]
            bm, bn, stages, warps = (64, 128, 3, 4) if m < 512 else (128, 128, 3, 4)
            bk = 64
            if m == 128:
                if site == "qkv":
                    bm, bn, bk, stages = 64, 128, 128, 4
                else:
                    bm, bn, bk, stages = 128, 128, 64, 3
            row["schedule"] = {"bm": bm, "bn": bn, "bk": bk, "stages": stages, "warps": warps}
            default = torch.compile(unfused, fullgraph=True, dynamic=False)
            maximum = torch.compile(unfused, fullgraph=True, dynamic=False, mode="max-autotune")
            compiled_norm = torch.compile(norm, fullgraph=True, dynamic=False)
            x, w, bias, reference, lazy_reference = inputs(cfg, m, n, 1701)
            compiled_code = {}
            for label, compiled in [("torch_compile_default", default), ("torch_compile_max_autotune", maximum)]:
                _, code = run_and_get_code(compiled, x, w, bias, eps)
                compiled_code[label] = "\n".join(code)
                path = args.out.parent / "compiler" / f"{cfg['model']}_{site}_{m}_{label}.py"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(compiled_code[label])
            outputs = {}

            def build(xx, ww, bb):
                inverse = torch.empty(m, device="cuda", dtype=torch.float32)
                out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
                anchor_out = torch.empty_like(out)
                def candidate():
                    # Match normal operator output ownership: compiled/vendor
                    # calls return fresh outputs, so the candidate allocates
                    # its output and required scalar scratch on each call.
                    result = torch.empty((m, n), device=xx.device, dtype=xx.dtype)
                    scalar = torch.empty(m, device=xx.device, dtype=torch.float32)
                    kernel.lazy_rms_projection(xx, ww, bb, result, scalar, eps=eps,
                                               bm=bm, bn=bn, bk=bk, stages=stages, warps=warps)
                    return result
                def anchor():
                    kernel.lazy_rms_projection(xx, ww, bb, anchor_out, inverse, eps=eps,
                                               bm=bm, bn=bn, bk=bk, stages=stages, warps=warps, standalone=True)
                    return anchor_out
                return {
                    "candidate": candidate,
                    "unfused_vendor_anchor": lambda: torch.addmm(bb, compiled_norm(xx, eps), ww),
                    "torch_compile_default": lambda: default(xx, ww, bb, eps),
                    "torch_compile_max_autotune": lambda: maximum(xx, ww, bb, eps),
                    "cublas_or_cublaslt": lambda: row_scale(xx, ww, bb, eps),
                    "standalone_tlx": anchor,
                    "standalone_cublas": lambda: torch.addmm(bb, xx, ww),
                }

            calls = build(x, w, bias)
            checks = []
            for seed in [1701, 2803, 3907]:
                if seed == 1701:
                    xx, ww, bb, ref, lazy = x, w, bias, reference, lazy_reference
                    arms = calls
                else:
                    xx, ww, bb, ref, lazy = inputs(cfg, m, n, seed)
                    arms = build(xx, ww, bb)
                raw_ref = xx.float() @ ww.float() + bb.float()
                for name, call in arms.items():
                    value = call()
                    original = metric(value, raw_ref if name.startswith("standalone") else ref)
                    folded = metric(value, raw_ref if name.startswith("standalone") else lazy)
                    checks.append({"seed": seed, "arm": name, "original_fp32": original, "folded_fp32": folded})
                    if not original["pass"] or not folded["pass"]:
                        raise RuntimeError(f"Independent FP32 correctness failed for {name} seed {seed}")
                del arms, xx, ww, bb, ref, lazy
            row["correctness"] = checks
            row["correctness_pass"] = True
            row["max_relative_l2"] = max(check["original_fp32"]["relative_l2"] for check in checks)
            measured = {name: [] for name in ARMS}
            pairs = {}
            for i, baseline in enumerate(ARMS[1:]):
                result = paired({"candidate": calls["candidate"], baseline: calls[baseline]},
                                rounds=args.rounds, samples=args.samples, replays=args.replays, seed=4109+i)
                pairs[baseline] = result
                for name, result_arm in result["arms"].items():
                    measured[name].extend(result_arm["samples_us"])
            row["paired_timing"] = pairs
            row["arm_statistics"] = {name: summarize(values) for name, values in measured.items()}
            fastest = min(ARMS[1:], key=lambda name: row["arm_statistics"][name]["p50_ms"])
            row["fastest_baseline"] = fastest
            row["speedup_vs_fastest"] = row["arm_statistics"][fastest]["p50_ms"] / row["arm_statistics"]["candidate"]["p50_ms"]
            retest = paired({"candidate": calls["candidate"], fastest: calls[fastest]},
                            rounds=args.rounds, samples=args.samples, replays=args.replays, seed=105109)
            row["retest"] = retest
            row["retest_speedup"] = retest["arms"][fastest]["median_us"] / retest["arms"]["candidate"]["median_us"]
            row["anchor_operator_timing"] = paired(
                {name: calls[name] for name in ["standalone_tlx", "standalone_cublas"]},
                rounds=args.rounds, samples=args.samples, replays=args.replays, seed=7109)
            # The standalone GEMM gate measures device kernel latency. Both
            # standalone arms use identical fresh graph blocks and CUDA events;
            # their normal Python dispatch timings are retained separately.
            anchor = device_paired({name: calls[name] for name in ["standalone_tlx", "standalone_cublas"]},
                                   rounds=args.rounds, samples=args.samples, replays=args.replays, seed=7109)
            row["anchor_timing"] = anchor
            row["anchor_ratio"] = anchor["arms"]["standalone_tlx"]["median_us"] / anchor["arms"]["standalone_cublas"]["median_us"]
            aa = paired({"A1": calls[fastest], "A2": calls[fastest]}, rounds=args.rounds,
                        samples=args.samples, replays=args.replays, seed=8109)
            deviations = [abs(a/b - 1) for a,b in zip(aa["arms"]["A1"]["samples_us"], aa["arms"]["A2"]["samples_us"])]
            row["aa_receipt"] = aa
            row["noise_margin"] = percentile(deviations, .95)
            labels = ["candidate", "unfused_vendor_anchor", "torch_compile_default", "torch_compile_max_autotune", "cublas_or_cublaslt"]
            for label in labels:
                path = trace_root / cfg["model"] / f"{site}_M{m}_{label}.chrome_trace.json.gz"
                row["traces"][label] = capture_trace(calls[label], path, "abc_a." + label)
            statuses = []
            reports = {}
            for label in ["torch_compile_default", "torch_compile_max_autotune"]:
                code = compiled_code[label]
                count = row["traces"][label]["kernel_count"]
                # A separate pointwise normalization followed by an external
                # addmm is positive evidence that this operation is unfused.
                external_gemm = "extern_kernels.addmm" in code or "extern_kernels.mm" in code
                separate_norm = "rsqrt" in code and ("tl.store" in code or "triton_poi" in code)
                # Max-autotune may choose a Triton GEMM template rather than
                # extern_kernels. Prove that a BF16 normalized-input buffer
                # is written by the reduction and consumed by a distinct GEMM.
                norm_buffer = re.search(r"triton_red[^\n]*\.run\([^\n]*\b(buf\d+)", code)
                materialized_norm = bool(norm_buffer)
                template_calls = [line for line in code.splitlines() if "triton_tem" in line and ".run(" in line]
                separate_template = bool(norm_buffer) and "addmm" in code and any(
                    re.search(r"\b" + norm_buffer.group(1) + r"\b", line) for line in template_calls)
                status = "not_fused" if separate_norm and (external_gemm or separate_template) and count >= 2 else "compiler_already_fused" if count == 1 and "rsqrt" in code else "unknown"
                statuses.append(status)
                reports[label] = {"status": status, "kernel_count": count,
                                  "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                                  "external_gemm": external_gemm, "separate_norm": separate_norm,
                                  "materialized_normalized_input": materialized_norm, "separate_gemm_template": separate_template}
            row["compiler_report"] = reports
            row["compiler_status"] = "compiler_already_fused" if "compiler_already_fused" in statuses else "not_fused" if all(x == "not_fused" for x in statuses) else "unknown"
            if m == 128 and cfg["model"] == "Qwen3-8B":
                old_out = torch.empty_like(calls["candidate"]())
                split_out = torch.empty_like(old_out)
                split_inv = torch.empty(m, device="cuda", dtype=torch.float32)
                def old():
                    kernel.launch("tlx", x, w, old_out)
                    return old_out + bias
                def reduction_split():
                    kernel.lazy_rms_projection(x, w, bias, split_out, split_inv, eps=eps,
                                               bm=64, bn=64, stages=3, warps=4)
                    return split_out
                for name, call in [("old_schedule", old), ("reduction_split", reduction_split)]:
                    check = metric(call(), reference)
                    if not check["pass"]:
                        raise RuntimeError("Ablation correctness failed: " + name)
                row["optimization_steps"] = paired({"old_schedule": old, "reduction_split": reduction_split,
                                                   "final_schedule": calls["candidate"]}, rounds=args.rounds,
                                                  samples=args.samples, replays=args.replays, seed=9109)
            if row["anchor_ratio"] > 1.01:
                row["gate_reasons"].append("standalone_anchor_slower_than_cublas_by_more_than_one_percent")
            if row["compiler_status"] != "not_fused":
                row["gate_reasons"].append(row["compiler_status"])
            if not all(trace["nonempty_gpu_trace"] for trace in row["traces"].values()):
                row["gate_reasons"].append("missing_gpu_trace_events")
            row["status"] = "compiler_already_fused" if row["compiler_status"] == "compiler_already_fused" else "tested_pass" if not row["gate_reasons"] else "tested_invalid"
        except torch.OutOfMemoryError:
            row.update(status="oom", gate_reasons=["allocation_does_not_fit"])
        except Exception as exc:
            row.update(status="tested_invalid", execution_error=f"{type(exc).__name__}: {exc}")
            row["gate_reasons"].append("execution_error")
        finally:
            observations = mon.finish()
            after = provenance.gpu_snapshot()
            row["gpu_exclusivity"].update(observations=observations, after=after)
            row["exclusivity_pass"] = before["exclusive"] and after["exclusive"] and bool(observations) and all(x["exclusive"] for x in observations)
            if not row["exclusivity_pass"]:
                row["status"] = "tested_invalid"
                row["gate_reasons"].append("gpu_exclusivity_failed")
            row["publication_eligible"] = row["status"] == "tested_pass" and public_source and not development and not args.smoke
            row["win_vs_fastest"] = row["publication_eligible"] and row.get("retest_speedup", 0) > 1 + row.get("noise_margin", 0)
            rec["rows"].append(row)
            save()
            print(len(rec["rows"]), cfg["model"], site, m, row["status"],
                  "ratio", row.get("retest_speedup"), "anchor", row.get("anchor_ratio"), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    assert len(rec["rows"]) == rec["expected_rows"]
    rec["status"] = "complete"
    rec["publication_eligible"] = public_source and not development and not args.smoke and all(row["publication_eligible"] for row in rec["rows"]) and len(scope) == 126
    save()


if __name__ == "__main__":
    main()
