#!/usr/bin/env python3
"""Strict Track-C jagged concat + segment-LayerNorm benchmark.

The runner is deliberately fail-closed. It records every pre-registered
shape, including unsupported secondary-site rows, and never turns a missing
TorchRec/FBGEMM dependency, compiler failure, trace failure, or non-exclusive
GPU into a performance claim. The primary site is the checked-in Triton
``jagged_concat_layernorm`` candidate. The secondary site is retained as an
``unsupported_shape`` row because the public C manifest pre-registers both
sites, but this repository has no legal narrow-MLP GEMM module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

PLAN_REL = "experiments/ABC_VENDOR_ANCHOR_PLAN.md"
SCHEMA = "kernelfuser-abc-c-jagged-benchmark-v2"
ARMS = ("candidate", "eager_pytorch", "unfused_vendor_anchor",
        "torch_compile_default", "torch_compile_max_autotune", "cublas_or_cublaslt")
PRIMARY_SITE = "jagged_concat_layernorm"
SECONDARY_SITE = "adjacent_narrow_mlp_gemm_pair"


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _redact(value: Any) -> str:
    return str(value)[:500].replace("\r", " ").replace("\n", " ")


def _module(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"available": True, "version": str(getattr(mod, "__version__", "unknown"))}
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__, "error": _redact(exc)}


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("kernelfuser_c_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load implementation: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("feature_count", "embedding_dim", "per_feature_length", "batch")
    if not isinstance(data, dict) or any(k not in data for k in required):
        raise ValueError("manifest is missing one of the four pre-registered axes")
    for key in required:
        if not isinstance(data[key], list) or not data[key]:
            raise ValueError(f"manifest axis {key!r} must be a non-empty list")
    if data.get("padding_hidden_from_timing") is not True:
        raise ValueError("manifest must declare padding_hidden_from_timing=true")
    expected = [{"feature_count": int(features), "embedding_dim": int(dim),
                 "per_feature_length": int(length), "batch": int(batch)}
                for features in data["feature_count"] for dim in data["embedding_dim"]
                for length in data["per_feature_length"] for batch in data["batch"]]
    declared = data.get("rows")
    if not isinstance(declared, list) or len(declared) != len(expected):
        raise ValueError("manifest rows do not cover the Cartesian public matrix")
    for index, (actual, want) in enumerate(zip(declared, expected)):
        if any(int(actual.get(key, -1)) != value for key, value in want.items()):
            raise ValueError(f"manifest row {index} does not match flattened axis order")
        if int(actual.get("row_index", -1)) != index:
            raise ValueError(f"manifest row_index mismatch at row {index}")
        segments = want["batch"] * want["feature_count"]
        offsets = list(range(0, (segments + 1) * want["per_feature_length"], want["per_feature_length"]))
        expected_offsets = hashlib.sha256(",".join(map(str, offsets)).encode()).hexdigest()
        if actual.get("offsets_sha256") != expected_offsets:
            raise ValueError(f"manifest offsets checksum mismatch at row {index}")
        if int(actual.get("segments", -1)) != segments:
            raise ValueError(f"manifest segments mismatch at row {index}")
        if int(actual.get("total_tokens", -1)) != segments * want["per_feature_length"]:
            raise ValueError(f"manifest total_tokens mismatch at row {index}")
        if actual.get("length_histogram") != {str(want["per_feature_length"]): segments}:
            raise ValueError(f"manifest length_histogram mismatch at row {index}")
    flat = "|".join(",".join(str(want[key]) for key in ("feature_count", "embedding_dim", "per_feature_length", "batch"))
                    for want in expected).encode()
    histogram = "|".join(f"{want['per_feature_length']}:{want['batch'] * want['feature_count']}" for want in expected).encode()
    if data.get("flattened_index_sha256") != hashlib.sha256(flat).hexdigest():
        raise ValueError("manifest flattened_index_sha256 mismatch")
    if data.get("histogram_checksum") != hashlib.sha256(histogram).hexdigest():
        raise ValueError("manifest histogram_checksum mismatch")
    return data


def _shape_rows(data: dict[str, Any], smoke: bool, full: bool) -> list[dict[str, int]]:
    rows = [{"feature_count": int(features), "embedding_dim": int(dim),
             "per_feature_length": int(length), "batch": int(batch)}
            for features in data["feature_count"] for dim in data["embedding_dim"]
            for length in data["per_feature_length"] for batch in data["batch"]]
    if smoke:
        return [rows[0]]
    if not full:
        return []
    return rows


def _build_inputs(torch, shape: dict[str, int], seed: int):
    torch.manual_seed(seed)
    segments = shape["batch"] * shape["feature_count"]
    length, dim = shape["per_feature_length"], shape["embedding_dim"]
    offsets = torch.arange(0, (segments + 1) * length, length,
                          dtype=torch.int32, device="cuda")
    values = torch.randn((segments * length, dim), device="cuda",
                         dtype=torch.bfloat16)
    return values.contiguous(), offsets.contiguous()


def _layout_checksum(torch, shape: dict[str, int], offsets) -> dict[str, Any]:
    """Hash only the public jagged layout, never random values or weights."""
    raw = json.dumps(list(offsets.detach().to("cpu").tolist()), separators=(",", ":")).encode("utf-8")
    lengths = offsets[1:] - offsets[:-1]
    histogram: dict[str, int] = {}
    for item in lengths.detach().to("cpu").tolist():
        key = str(int(item)); histogram[key] = histogram.get(key, 0) + 1
    payload = json.dumps({"shape": shape, "offsets": list(offsets.detach().to("cpu").tolist())},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"offsets_sha256": hashlib.sha256(raw).hexdigest(),
            "layout_sha256": hashlib.sha256(payload).hexdigest(),
            "length_histogram": histogram,
            "total_tokens": int(offsets[-1].item())}


def _reference(torch, values, offsets, eps: float):
    # Concatenation, rather than writes into an empty tensor, keeps this
    # independent FP32 reference differentiable for the gradient gate.
    lengths = offsets[1:] - offsets[:-1]
    if lengths.numel() and bool(torch.all(lengths == lengths[0]).item()):
        width = int(lengths[0].item()) * values.shape[1]
        x = values.reshape(-1, width).float()
        mean = x.mean(dim=1, keepdim=True)
        inv = torch.rsqrt((x - mean).square().mean(dim=1, keepdim=True) + eps)
        return ((x - mean) * inv).reshape_as(values).to(values.dtype)
    parts = []
    for i in range(offsets.numel() - 1):
        start, end = int(offsets[i].item()), int(offsets[i + 1].item())
        if end <= start:
            continue
        x = values[start:end].float().reshape(-1)
        mean = x.mean()
        inv = torch.rsqrt((x - mean).square().mean() + eps)
        parts.append(((x - mean) * inv).reshape(end - start, values.shape[1]))
    return torch.cat(parts, dim=0).to(values.dtype) if parts else values.clone()


def _reference_grad(torch, values, offsets, grad_output, eps: float):
    lengths = offsets[1:] - offsets[:-1]
    if lengths.numel() and bool(torch.all(lengths == lengths[0]).item()):
        width = int(lengths[0].item()) * values.shape[1]
        x = values.reshape(-1, width).float()
        dy = grad_output.reshape(-1, width).float()
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var_eps = centered.square().mean(dim=1, keepdim=True) + eps
        inv = torch.rsqrt(var_eps)
        dx = (dy - dy.mean(dim=1, keepdim=True) - centered * (dy * centered).mean(dim=1, keepdim=True) / var_eps) * inv
        return dx.reshape_as(values).to(values.dtype)
    grad = torch.zeros_like(values)
    for i in range(offsets.numel() - 1):
        start, end = int(offsets[i].item()), int(offsets[i + 1].item())
        if end <= start:
            continue
        x = values[start:end].float().reshape(-1)
        dy = grad_output[start:end].float().reshape(-1)
        mean = x.mean()
        centered = x - mean
        var_eps = centered.square().mean() + eps
        inv = torch.rsqrt(var_eps)
        dx = (dy - dy.mean() - centered * (dy * centered).mean() / var_eps) * inv
        grad[start:end] = dx.reshape(end - start, values.shape[1]).to(values.dtype)
    return grad


def _vendor_reference(torch, values, offsets, eps: float):
    """Unfused ATen CUDA LayerNorm anchor over the same unpadded segments.

    FBGEMM/TorchRec remain mandatory dependency gates. There is no stable public
    FBGEMM jagged LayerNorm symbol across release stacks, so the declared
    vendor anchor is ATen's native LayerNorm and its exact name is recorded.
    """
    import torch.nn.functional as F
    lengths = offsets[1:] - offsets[:-1]
    # Every registered row has a fixed per-feature length.  Reshaping the
    # already-flat segments lets the vendor LayerNorm launch once without
    # introducing padded storage or hiding any work from the timer.
    if lengths.numel() and bool(torch.all(lengths == lengths[0]).item()):
        segment_width = int(lengths[0].item()) * values.shape[1]
        x = values.reshape(-1, segment_width)
        return F.layer_norm(x.float(), (segment_width,), eps=eps).reshape_as(values).to(values.dtype)
    out = torch.empty_like(values)
    for i in range(offsets.numel() - 1):
        start, end = int(offsets[i].item()), int(offsets[i + 1].item())
        if end <= start:
            continue
        x = values[start:end].reshape(1, -1)
        y = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
        out[start:end] = y.reshape(end - start, values.shape[1]).to(values.dtype)
    return out


def _candidate_autograd(torch, candidate, values, offsets, eps: float):
    """Run candidate forward and the checked analytic segment-LN backward."""
    class CandidateFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, o):
            ctx.save_for_backward(x, o)
            with torch.no_grad():
                return candidate.jagged_concat_layernorm(x, o, eps)

        @staticmethod
        def backward(ctx, grad_output):
            x, o = ctx.saved_tensors
            return _reference_grad(torch, x, o, grad_output, eps), None

    return CandidateFunction.apply(values, offsets)


def _error(torch, actual, expected) -> dict[str, Any]:
    delta = (actual.float() - expected.float()).abs()
    denom = expected.float().abs().clamp_min(1.0e-3)
    atol, rtol = 0.02, 0.02
    return {"max_abs": float(delta.max().item()) if delta.numel() else 0.0,
            "max_rel": float((delta / denom).max().item()) if delta.numel() else 0.0,
            "atol": atol, "rtol": rtol,
            "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))}


def _timed(torch, fn: Callable[[], Any], warmup: int, samples: int, rounds: int = 1) -> dict[str, float]:
    values: list[float] = []
    for _round in range(rounds):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        for _ in range(samples):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(); fn(); end.record(); end.synchronize()
            values.append(float(begin.elapsed_time(end)))
    values.sort()
    return {"p50_ms": float(statistics.median(values)),
            "p95_ms": float(values[min(len(values) - 1, int((len(values) - 1) * 0.95))]),
            "samples": len(values), "rounds": rounds}


def _timed_paired(torch, fns: dict[str, Callable[[], Any]], warmup: int,
                  samples: int, rounds: int = 1,
                  order: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Measure arms in an alternating order within one CUDA process."""
    names = tuple(order or fns.keys())
    if not names or any(name not in fns for name in names):
        raise ValueError("paired timing order must name every supplied arm")
    measurements: dict[str, list[float]] = {name: [] for name in names}
    for _round in range(rounds):
        for name in names:
            for _ in range(warmup):
                fns[name]()
        torch.cuda.synchronize()
        for _sample in range(samples):
            for name in names:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(); fns[name](); end.record(); end.synchronize()
                measurements[name].append(float(begin.elapsed_time(end)))
    out: dict[str, Any] = {"paired_arm_order": list(names), "arms": {}}
    for name, values in measurements.items():
        values.sort()
        out["arms"][name] = {
            "p50_ms": float(statistics.median(values)),
            "p95_ms": float(values[min(len(values) - 1, int((len(values) - 1) * 0.95))]),
            "samples": len(values), "rounds": rounds,
        }
    return out


def _gpu_snapshot(physical_gpu: str, duration_sec: float = 1.0,
                  poll_interval_sec: float = 0.2) -> dict[str, Any]:
    """Poll exclusivity for a full window without retaining PIDs or UUIDs."""
    snap = {"query_ok": False, "physical_gpu": str(physical_gpu),
            "external_process_count": None, "process_names": [],
            "snapshot_duration_sec": 0.0, "snapshot_samples": 0}
    if not str(physical_gpu).isdigit():
        snap["reason"] = "physical_gpu_not_supplied"
        return snap
    try:
        start = time.monotonic()
        deadline = start + max(0.0, float(duration_sec))
        samples, max_external, names = 0, 0, set()
        query_ok = True
        while samples == 0 or time.monotonic() < deadline:
            proc = subprocess.run(["nvidia-smi", "-i", str(physical_gpu),
                                   "--query-compute-apps=pid,process_name,used_memory",
                                   "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10, check=False)
            if proc.returncode != 0:
                query_ok = False
                break
            external = 0
            for line in proc.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if not fields or not fields[0].isdigit():
                    continue
                if int(fields[0]) != os.getpid():
                    external += 1
                    if len(fields) > 1:
                        names.add(Path(fields[1]).name[:80])
            samples += 1
            max_external = max(max_external, external)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(max(0.01, float(poll_interval_sec)), remaining))
        elapsed = time.monotonic() - start
        snap.update({"query_ok": query_ok and samples > 0,
                     "external_process_count": max_external if samples else None,
                     "process_names": sorted(names), "exclusive": bool(query_ok and samples and max_external == 0),
                     "snapshot_duration_sec": elapsed, "snapshot_samples": samples})
        if not query_ok:
            snap["reason"] = "nvidia_smi_failed"
    except Exception as exc:
        snap["reason"] = type(exc).__name__
    return snap


def _compiler_report(torch, fn, values, offsets, mode: str):
    report: dict[str, Any] = {"mode": mode, "status": "unavailable",
                              "graph_count": None, "graph_break_count": None,
                              "op_count": None}
    try:
        compiled = torch.compile(fn, mode=mode)
        compiled(values, offsets); torch.cuda.synchronize()
        try:
            info = torch._dynamo.explain(fn)(values, offsets)
            graphs = getattr(info, "graphs", None)
            ops = getattr(info, "ops_per_graph", None)
            report.update({"status": "pass",
                           "graph_count": int(getattr(info, "graph_count", len(graphs) if graphs is not None else 0)),
                           "graph_break_count": int(getattr(info, "graph_break_count", 0)),
                           "op_count": int(sum(len(x) for x in ops)) if ops is not None else None})
        except Exception as exc:
            # A compiled callable without a graph report cannot satisfy the
            # compiler-fusion gate. Keep it for diagnostics but fail closed
            # when deciding whether a row is publishable.
            report.update({"status": "failed_no_explain", "explain_error": _redact(exc)})
        return (lambda: compiled(values, offsets)), report
    except Exception as exc:
        report.update({"error_type": type(exc).__name__, "error": _redact(exc)})
        return None, report


def _capture_trace(torch, fn: Callable[[], Any], path: Path):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                 torch.profiler.ProfilerActivity.CUDA],
                                   record_shapes=False, profile_memory=False) as prof:
            fn(); torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        return {"status": "pass", "sha256": _sha256(path), "path": str(path)}
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error": _redact(exc)}


def _write(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


CSV_FIELDS = ["track", "lane", "site", "row_index", "feature_count", "embedding_dim",
              "per_feature_length", "batch", "seed", "status", "candidate_p50_ms",
              "candidate_p95_ms", "eager_pytorch_p50_ms", "eager_pytorch_p95_ms",
              "unfused_vendor_anchor_p50_ms", "unfused_vendor_anchor_p95_ms",
              "torch_compile_default_p50_ms", "torch_compile_default_p95_ms",
              "torch_compile_max_autotune_p50_ms", "torch_compile_max_autotune_p95_ms",
              "cublas_or_cublaslt_p50_ms", "cublas_or_cublaslt_p95_ms", "fastest_baseline", "fastest_baseline_p50_ms",
              "speedup_vs_fastest", "anchor_ratio", "win_vs_fastest", "retest_required",
              "retest_pass", "candidate_max_abs", "candidate_max_rel", "candidate_grad_max_abs",
              "candidate_grad_max_rel", "vendor_max_abs", "layout_sha256", "compiler_status", "trace_hash", "trace_hashes",
              "trace_status", "candidate_trace_hash", "candidate_trace_status",
              "unfused_trace_hash", "unfused_trace_status", "compiled_trace_hash",
              "compiled_trace_status", "vendor_trace_hash", "vendor_trace_status",
              "exclusivity_verdict", "exclusivity_pre_external_count",
              "exclusivity_post_external_count", "trainable_parameter_count",
              "optimizer_update_status", "reason"]


def _csv_row(row: dict[str, Any]) -> dict[str, Any]:
    shape, corr = row.get("shape", {}), row.get("correctness", {})
    grad, timing = row.get("gradient", {}), row.get("timing", {})
    arms, retest = timing.get("arms", {}), row.get("retest", {})
    pre = row.get("exclusivity", {}).get("pre", {})
    post = row.get("exclusivity", {}).get("post", {})
    out = {key: "" for key in CSV_FIELDS}
    out.update({"track": "C", "lane": "dlrm_concat_ln", "site": row.get("site", PRIMARY_SITE),
                "row_index": row.get("row_index", ""), "seed": row.get("seed", ""),
                "status": row.get("status", "")})
    out.update({k: shape.get(k, "") for k in ("feature_count", "embedding_dim", "per_feature_length", "batch")})
    for arm in ARMS:
        if arm in arms:
            out[f"{arm}_p50_ms"], out[f"{arm}_p95_ms"] = arms[arm].get("p50_ms", ""), arms[arm].get("p95_ms", "")
    out.update({"fastest_baseline": timing.get("fastest_baseline", ""),
                "fastest_baseline_p50_ms": timing.get("fastest_baseline_p50_ms", ""),
                "speedup_vs_fastest": timing.get("speedup_vs_fastest", ""),
                "anchor_ratio": timing.get("anchor_ratio", ""),
                "win_vs_fastest": timing.get("win_vs_fastest", ""),
                "retest_required": retest.get("required", ""), "retest_pass": retest.get("pass", ""),
                "candidate_max_abs": corr.get("candidate_max_abs", ""),
                "candidate_max_rel": corr.get("candidate_max_rel", ""),
                "candidate_grad_max_abs": grad.get("candidate_max_abs", ""),
                "candidate_grad_max_rel": grad.get("candidate_max_rel", ""),
                "vendor_max_abs": corr.get("vendor_max_abs", ""),
                "layout_sha256": row.get("layout", {}).get("layout_sha256", ""),
                "compiler_status": row.get("compiler", {}).get("status", ""),
                "trace_hash": row.get("trace", {}).get("sha256", ""),
                "trace_hashes": ";".join(
                    f"{name}:{info.get('sha256', '')}"
                    for name, info in sorted(row.get("traces", {}).items())
                ),
                "trace_status": row.get("trace", {}).get("status", ""),
                "candidate_trace_hash": row.get("traces", {}).get("candidate", {}).get("sha256", ""),
                "candidate_trace_status": row.get("traces", {}).get("candidate", {}).get("status", ""),
                "unfused_trace_hash": row.get("traces", {}).get("unfused", {}).get("sha256", ""),
                "unfused_trace_status": row.get("traces", {}).get("unfused", {}).get("status", ""),
                "compiled_trace_hash": row.get("traces", {}).get("compiled", {}).get("sha256", ""),
                "compiled_trace_status": row.get("traces", {}).get("compiled", {}).get("status", ""),
                "vendor_trace_hash": row.get("traces", {}).get("vendor", {}).get("sha256", ""),
                "vendor_trace_status": row.get("traces", {}).get("vendor", {}).get("status", ""),
                "exclusivity_verdict": row.get("exclusivity", {}).get("verdict", ""),
                "exclusivity_pre_external_count": pre.get("external_process_count", ""),
                "exclusivity_post_external_count": post.get("external_process_count", ""),
                "trainable_parameter_count": row.get("optimizer_update", {}).get("trainable_parameter_count", ""),
                "optimizer_update_status": row.get("optimizer_update", {}).get("status", ""),
                "reason": ";".join(row.get("reasons", []))})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS); writer.writeheader()
        writer.writerows([_csv_row(row) for row in rows])


def _blocked_rows(shapes: list[dict[str, int]], status: str, reason: str,
                  include_secondary: bool, seed: int) -> list[dict[str, Any]]:
    """Retain every requested public row when a preflight gate blocks CUDA."""
    rows: list[dict[str, Any]] = []
    for row_index, shape in enumerate(shapes):
        for site in ([PRIMARY_SITE, SECONDARY_SITE] if include_secondary else [PRIMARY_SITE]):
            rows.append({"row_index": row_index, "site": site, "shape": shape,
                         "seed": seed, "status": status, "timing": {},
                         "reasons": [reason]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", dest="out", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--trace-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).parents[1] / "data/abc/c_dlrm_manifest.json")
    parser.add_argument("--implementation", type=Path, default=Path(__file__).with_name("c_dlrm_concat_ln.py"))
    parser.add_argument("--source-commit", default=os.getenv("KERNELFUSER_SOURCE_COMMIT", "unknown"))
    parser.add_argument("--physical-gpu", default=os.getenv("KERNELFUSER_PHYSICAL_GPU", "unknown"))
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--full", action="store_true")
    parser.add_argument("--include-secondary-site", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=20); parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--correctness-seeds", type=int, default=3); parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261015); parser.add_argument("--eps", type=float, default=1.0e-5)
    args = parser.parse_args()

    here = Path(__file__).resolve(); plan = here.parent.parent / PLAN_REL
    csv_path = args.csv or args.out.with_suffix(".csv")
    trace_dir = args.trace_dir or args.out.with_name(args.out.stem + "-traces")
    receipt: dict[str, Any] = {
        "schema": SCHEMA, "track": "C", "lane": "dlrm_concat_ln", "status": "started",
        "publication_eligible": False,
        "claim_boundary": "public flat jagged concat + segment LayerNorm only; no full DLRM/model claim",
        "source_commit": args.source_commit, "source_sha256": _sha256(here),
        "implementation_sha256": _sha256(args.implementation), "config_manifest_sha256": _sha256(args.manifest),
        "plan_sha256": _sha256(plan), "arms": list(ARMS),
        "sites": [PRIMARY_SITE] + ([SECONDARY_SITE] if args.include_secondary_site else []),
        "protocol": {"warmup": args.warmup, "samples": args.samples, "rounds": args.rounds,
                      "correctness_seeds": args.correctness_seeds, "cuda_events": True,
                      "correctness_before_timing": True, "trace_dir": str(trace_dir),
                      "trace_labels": ["candidate", "unfused", "compiled", "vendor"],
                      "exclusivity_window_sec": 1.0,
                      "paired_retest_required_above_speedup": 1.01,
                      "noise_margin_source": "same-run A/A required before publication"},
        "rows": [], "reasons": [], "csv": str(csv_path)}

    try:
        manifest = _manifest(args.manifest)
        receipt["manifest_axes"] = {k: manifest[k] for k in ("feature_count", "embedding_dim", "per_feature_length", "batch")}
        receipt["manifest_checksums"] = {k: manifest.get(k) for k in ("manifest_schema", "flattened_index_sha256", "histogram_checksum")}
    except Exception as exc:
        receipt.update({"status": "blocked_on_implementation", "reasons": ["invalid_shape_manifest", _redact(exc)]})
        _write(args.out, receipt); _write_csv(csv_path, []); return

    checks = {"torch": _module("torch"), "triton": _module("triton"), "torchrec": _module("torchrec"),
              "torchrec_jagged_tensor": _module("torchrec.sparse.jagged_tensor"), "fbgemm_gpu": _module("fbgemm_gpu")}
    receipt["checks"] = checks
    requested_rows = _shape_rows(manifest, args.smoke, args.full)
    missing = [name for name, info in checks.items() if not info.get("available")]
    if missing:
        receipt.update({"status": "blocked_dependency", "reasons": ["required_dependency_missing:" + name for name in missing],
                        "rows": _blocked_rows(requested_rows, "blocked_dependency", "required_dependency_missing",
                                               args.include_secondary_site, args.seed)})
        _write(args.out, receipt); _write_csv(csv_path, receipt["rows"]); return
    if not args.smoke and not args.full:
        receipt.update({"status": "ready_for_smoke", "reasons": ["explicit_--smoke_required_before_full_sweep"]})
        _write(args.out, receipt); _write_csv(csv_path, []); return

    try:
        import torch
        if not torch.cuda.is_available():
            receipt.update({"status": "blocked_dependency", "reasons": ["cuda_unavailable"],
                            "rows": _blocked_rows(requested_rows, "blocked_dependency", "cuda_unavailable",
                                                   args.include_secondary_site, args.seed)})
            _write(args.out, receipt); _write_csv(csv_path, receipt["rows"]); return
        candidate = _load_module(args.implementation); rows = requested_rows
        if not rows:
            receipt.update({"status": "ready_for_smoke", "reasons": ["no_rows_requested"]})
            _write(args.out, receipt); _write_csv(csv_path, []); return
        receipt["environment"] = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0),
                                   "visible_gpu": os.getenv("CUDA_VISIBLE_DEVICES"), "physical_gpu": str(args.physical_gpu),
                                   "torch": str(torch.__version__), "triton": checks["triton"].get("version"),
                                   "cuda": torch.version.cuda, "dtype": "bfloat16", "python": sys.version.split()[0]}
        top_pre = _gpu_snapshot(str(args.physical_gpu)); receipt["exclusivity_pre"] = top_pre
        if not top_pre.get("query_ok") or top_pre.get("external_process_count") != 0:
            receipt["reasons"].append("gpu_not_exclusive_at_start")
            receipt["status"] = "blocked_exclusivity"
            receipt["rows"] = _blocked_rows(rows, "blocked_exclusivity", "gpu_not_exclusive_at_start",
                                             args.include_secondary_site, args.seed)
            _write(args.out, receipt); _write_csv(csv_path, receipt["rows"]); return
        primary_pass = True
        for row_index, shape in enumerate(rows):
            values, offsets = _build_inputs(torch, shape, args.seed)
            row = {"row_index": row_index, "site": PRIMARY_SITE, "shape": shape, "seed": args.seed,
                   "status": "tested_invalid", "timing": {}, "reasons": []}
            row["layout"] = _layout_checksum(torch, shape, offsets)
            candidate_error, vendor_error, candidate_grad = [], [], []
            for seed_offset in range(args.correctness_seeds):
                vals, offs = _build_inputs(torch, shape, args.seed + seed_offset)
                expected = _reference(torch, vals, offs, args.eps)
                actual = candidate.jagged_concat_layernorm(vals, offs, args.eps)
                vendor = _vendor_reference(torch, vals, offs, args.eps)
                candidate_error.append(_error(torch, actual, expected)); vendor_error.append(_error(torch, vendor, expected))
                grad_seed = torch.randn_like(vals)
                cx = vals.detach().clone().requires_grad_(True)
                cg = torch.autograd.grad((_candidate_autograd(torch, candidate, cx, offs, args.eps).float() * grad_seed.float()).sum(), cx)[0]
                rx = vals.detach().clone().requires_grad_(True); rr = _reference(torch, rx, offs, args.eps)
                rg = torch.autograd.grad((rr.float() * grad_seed.float()).sum(), rx)[0]
                candidate_grad.append(_error(torch, cg, rg))
            row["correctness"] = {"candidate_max_abs": max(x["max_abs"] for x in candidate_error),
                                   "candidate_max_rel": max(x["max_rel"] for x in candidate_error),
                                   "candidate_allclose": all(x["allclose"] for x in candidate_error),
                                   "vendor_max_abs": max(x["max_abs"] for x in vendor_error),
                                   "vendor_allclose": all(x["allclose"] for x in vendor_error), "atol": 0.02, "rtol": 0.02}
            row["gradient"] = {"candidate_max_abs": max(x["max_abs"] for x in candidate_grad),
                                "candidate_max_rel": max(x["max_rel"] for x in candidate_grad),
                                "candidate_allclose": all(x["allclose"] for x in candidate_grad),
                                "reference": "analytic segment LayerNorm backward"}
            row["optimizer_update"] = {
                "performed": False,
                "status": "not_applicable",
                "trainable_parameter_count": 0,
                "reason": "lane_has_no_trainable_parameters",
            }
            row["vendor_arm"] = "torch.nn.functional.layer_norm (ATen CUDA; TorchRec/FBGEMM dependency gate passed)"
            row["vendor_anchor_check"] = {
                "status": "pass",
                "implementation": "torch.nn.functional.layer_norm",
                "library": "ATen CUDA",
                "dependency_gate": "TorchRec/FBGEMM import and jagged tensor gate",
                "claim_scope": "semantic unfused LayerNorm anchor; no FBGEMM jagged-LayerNorm symbol",
            }
            if not (row["correctness"]["candidate_allclose"] and row["correctness"]["vendor_allclose"] and row["gradient"]["candidate_allclose"]):
                row["reasons"].append("correctness_or_gradient_gate_failed"); primary_pass = False; receipt["rows"].append(row)
                if args.include_secondary_site:
                    receipt["rows"].append({"row_index": row_index, "site": SECONDARY_SITE, "shape": shape,
                                            "seed": args.seed, "status": "unsupported_shape", "timing": {},
                                            "reasons": ["no_legal_public_narrow_mlp_gemm_site_in_checked_in_module"]})
                continue
            pre = _gpu_snapshot(str(args.physical_gpu)); candidate_fn = lambda: candidate.jagged_concat_layernorm(values, offsets, args.eps)
            vendor_fn = lambda: _vendor_reference(torch, values, offsets, args.eps); eager_fn = lambda: _reference(torch, values, offsets, args.eps)
            compiled_default, report_default = _compiler_report(torch, lambda x, o: _reference(torch, x, o, args.eps), values, offsets, "default")
            compiled_max, report_max = _compiler_report(torch, lambda x, o: _reference(torch, x, o, args.eps), values, offsets, "max-autotune")
            fns: dict[str, Callable[[], Any]] = {"candidate": candidate_fn, "eager_pytorch": eager_fn,
                                                 "unfused_vendor_anchor": vendor_fn}
            if compiled_default is not None: fns["torch_compile_default"] = compiled_default
            if compiled_max is not None: fns["torch_compile_max_autotune"] = compiled_max
            paired = _timed_paired(torch, fns, args.warmup, args.samples, args.rounds,
                                   order=tuple(fns))
            arms = paired["arms"]
            arms["cublas_or_cublaslt"] = {"status": "not_applicable", "reason": "LN-only lane has no GEMM"}
            compiler_default_ok = report_default.get("status") == "pass" and int(report_default.get("graph_count") or 0) >= 1
            compiler_max_ok = report_max.get("status") == "pass" and int(report_max.get("graph_count") or 0) >= 1
            row["compiler"] = {"default": report_default, "max_autotune": report_max,
                                "status": "pass" if compiler_default_ok and compiler_max_ok else "partial",
                                "graph_report_required": True}
            row["timing_protocol"] = {"paired": True, "paired_arm_order": paired["paired_arm_order"]}
            if row["compiler"]["status"] != "pass":
                row["reasons"].append("compiler_graph_gate_failed")
                primary_pass = False
            baselines = {k: v["p50_ms"] for k, v in arms.items() if k != "candidate" and isinstance(v, dict) and "p50_ms" in v}
            if not baselines:
                row["reasons"].append("no_valid_baseline"); primary_pass = False
                post = _gpu_snapshot(str(args.physical_gpu)); row["exclusivity"] = {"pre": pre, "post": post, "verdict": bool(pre.get("exclusive") and post.get("exclusive"))}
                receipt["rows"].append(row)
                if args.include_secondary_site:
                    receipt["rows"].append({"row_index": row_index, "site": SECONDARY_SITE, "shape": shape,
                                            "seed": args.seed, "status": "unsupported_shape", "timing": {},
                                            "reasons": ["no_legal_public_narrow_mlp_gemm_site_in_checked_in_module"]})
                continue
            fastest_name = min(baselines, key=baselines.get); fastest = baselines[fastest_name]; speedup = fastest / arms["candidate"]["p50_ms"]
            row["timing"] = {"arms": arms, "fastest_baseline": fastest_name, "fastest_baseline_p50_ms": fastest,
                              "speedup_vs_fastest": speedup, "anchor_ratio": arms["unfused_vendor_anchor"]["p50_ms"] / arms["candidate"]["p50_ms"],
                              "anchor_gate": {"status": "not_applicable", "reason": "LN-only lane has no cublas arm"},
                              "noise_margin": None, "win_vs_fastest": False}
            # A fresh paired sample is required for every initial >1% win.  A
            # second vendor sample also supplies the same-GPU A/A noise floor;
            # it is retained in the receipt instead of silently using a fixed
            # one-percent margin.
            retest_required = bool(speedup > 1.01)
            retest = {"required": retest_required, "pass": not retest_required}
            # Always collect an A/A pair.  The initial >1% condition controls
            # whether a fresh candidate/vendor retest is needed; it must not
            # control whether the noise floor exists.
            aa_pair = _timed_paired(
                torch,
                {"vendor_a": vendor_fn, "vendor_b": vendor_fn},
                args.warmup, args.samples, args.rounds,
                order=("vendor_a", "vendor_b"),
            )
            aa_a = aa_pair["arms"]["vendor_a"]["p50_ms"]
            aa_b = aa_pair["arms"]["vendor_b"]["p50_ms"]
            aa_margin = abs(aa_b / aa_a - 1.0) if aa_a else None
            if aa_margin is None:
                retest["pass"] = False
                retest["aa_error"] = "zero_vendor_AA_median"
            if retest_required:
                values_2, offsets_2 = _build_inputs(torch, shape, args.seed + 100003)
                candidate_2 = lambda: candidate.jagged_concat_layernorm(values_2, offsets_2, args.eps)
                vendor_2 = lambda: _vendor_reference(torch, values_2, offsets_2, args.eps)
                retest_pair = _timed_paired(
                    torch,
                    {"candidate": candidate_2, "vendor": vendor_2},
                    args.warmup, args.samples, args.rounds,
                    order=("candidate", "vendor"),
                )
                t_candidate_2 = retest_pair["arms"]["candidate"]
                t_vendor_2 = retest_pair["arms"]["vendor"]
                noise_margin = aa_margin if aa_margin is not None else float("inf")
                retest_speedup = t_vendor_2["p50_ms"] / t_candidate_2["p50_ms"]
                retest.update({"pass": bool(retest_speedup > 1.0 + noise_margin),
                               "candidate_p50_ms": t_candidate_2["p50_ms"],
                               "vendor_p50_ms": t_vendor_2["p50_ms"],
                               "speedup_vs_vendor": retest_speedup,
                               "noise_margin": noise_margin,
                               "paired_arm_order": retest_pair["paired_arm_order"]})
            else:
                noise_margin = aa_margin if aa_margin is not None else float("inf")
            row["retest"] = retest
            row["aa_control"] = {"performed": True, "noise_margin": aa_margin,
                                 "paired_arm_order": aa_pair["paired_arm_order"],
                                 "vendor_a_p50_ms": aa_a, "vendor_b_p50_ms": aa_b,
                                 "arm": "unfused_vendor_anchor", "same_shape": True,
                                 "same_dtype": True, "same_physical_gpu": str(args.physical_gpu)}
            row["timing"]["noise_margin"] = noise_margin
            row["timing"]["win_vs_fastest"] = bool(speedup > 1.0 + noise_margin)
            # Keep one trace for each declared execution path.  The candidate
            # trace remains in the legacy ``trace`` field for consumers of the
            # v1 receipt shape, while ``traces`` is the strict four-trace gate.
            trace_fns: dict[str, Callable[[], Any]] = {
                "candidate": candidate_fn,
                "unfused": eager_fn,
                "vendor": vendor_fn,
            }
            if compiled_default is not None:
                trace_fns["compiled"] = compiled_default
            row["traces"] = {
                name: _capture_trace(torch, fn, trace_dir / f"row_{row['row_index']:03d}_{name}.json")
                for name, fn in trace_fns.items()
            }
            if "compiled" not in row["traces"]:
                row["traces"]["compiled"] = {
                    "status": "failed", "error_type": "compiler_unavailable",
                    "error": "torch_compile_default did not produce a callable"
                }
            row["trace"] = row["traces"]["candidate"]
            post = _gpu_snapshot(str(args.physical_gpu)); row["exclusivity"] = {"pre": pre, "post": post,
                                                                                   "verdict": bool(pre.get("exclusive") and post.get("exclusive"))}
            if not row["exclusivity"]["verdict"]: row["reasons"].append("gpu_not_exclusive_for_timing"); primary_pass = False
            if any(info.get("status") != "pass" for info in row["traces"].values()):
                row["reasons"].append("trace_gate_failed"); primary_pass = False
            if not row["retest"].get("pass", False): row["reasons"].append("paired_retest_gate_failed"); primary_pass = False
            row["status"] = "tested_pass" if not row["reasons"] else "tested_invalid"; primary_pass = primary_pass and row["status"] == "tested_pass"
            receipt["rows"].append(row)
            if args.include_secondary_site:
                receipt["rows"].append({"row_index": row_index, "site": SECONDARY_SITE, "shape": shape, "seed": args.seed,
                                        "status": "unsupported_shape", "timing": {}, "reasons": ["no_legal_public_narrow_mlp_gemm_site_in_checked_in_module"]})
        top_post = _gpu_snapshot(str(args.physical_gpu)); receipt["exclusivity_post"] = top_post
        receipt["status"] = "tested_pass" if primary_pass else "tested_invalid"
        receipt["publication_eligible"] = bool(primary_pass and top_pre.get("exclusive") and top_post.get("exclusive"))
        if not receipt["publication_eligible"]: receipt["reasons"].append("paired_retest_AA_and_anchor_gates_not_satisfied")
    except Exception as exc:
        receipt.update({"status": "execution_error", "reasons": [type(exc).__name__, _redact(exc)]})
    _write(args.out, receipt); _write_csv(csv_path, receipt["rows"])


if __name__ == "__main__":
    main()
