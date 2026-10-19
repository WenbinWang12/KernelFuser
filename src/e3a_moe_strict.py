#!/usr/bin/env python3
"""Fail-closed strict receipt runner for the isolated E3a MoE census.

The historical implementation owns the independent FP32/BF16 correctness
checks, paired CUDA-graph timing, conditional retest, and Chrome traces. This
entry point adds source/command/environment provenance, continuous physical
GPU exclusivity sampling, trace hashes, CSV projection, and complete-row
validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
IMPL_PATH = HERE / "e3a_moe_publication.py"
_spec = importlib.util.spec_from_file_location("e3a_moe_publication_strict", IMPL_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load {IMPL_PATH}")
impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(impl)


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def gpu_snapshot(physical_gpu: str, *, window_seconds: float = 1.0,
                 poll_seconds: float = 0.2) -> dict[str, Any]:
    """Sample compute applications on one physical GPU for a real window."""
    result: dict[str, Any] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physical_gpu": str(physical_gpu), "query_ok": False,
        "external_process_count": None, "process_names": [],
        "snapshot_duration_sec": 0.0, "snapshot_samples": 0,
        "exclusive": False,
    }
    if not str(physical_gpu).isdigit():
        result["reason"] = "physical_gpu_not_supplied"
        return result
    started = time.monotonic()
    deadline = started + max(0.0, float(window_seconds))
    max_external = 0
    names: set[str] = set()
    samples = 0
    query_ok = True
    try:
        while samples == 0 or time.monotonic() < deadline:
            q = subprocess.run(
                ["nvidia-smi", "-i", str(physical_gpu),
                 "--query-compute-apps=pid,process_name,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False, timeout=10,
            )
            if q.returncode != 0:
                query_ok = False
                result["reason"] = "nvidia_smi_failed"
                break
            external = 0
            for line in q.stdout.splitlines():
                fields = [x.strip() for x in line.split(",")]
                if not fields or not fields[0].isdigit():
                    continue
                if int(fields[0]) != os.getpid():
                    external += 1
                    if len(fields) > 1:
                        names.add(Path(fields[1]).name[:120])
            max_external = max(max_external, external)
            samples += 1
            remain = deadline - time.monotonic()
            if remain > 0:
                time.sleep(min(max(0.01, float(poll_seconds)), remain))
    except Exception as exc:
        query_ok = False
        result["reason"] = f"{type(exc).__name__}: {exc}"
    elapsed = time.monotonic() - started
    result.update({
        "query_ok": bool(query_ok and samples > 0),
        "external_process_count": max_external if samples else None,
        "process_names": sorted(names),
        "snapshot_duration_sec": elapsed,
        "snapshot_samples": samples,
        "exclusive": bool(query_ok and samples > 0 and max_external == 0),
    })
    return result


class ExclusivityWindow:
    """Poll nvidia-smi throughout a row, including the interval of execution."""

    def __init__(self, physical_gpu: str, minimum_seconds: float = 1.0):
        self.physical_gpu = str(physical_gpu)
        self.minimum_seconds = float(minimum_seconds)
        self.records: list[dict[str, Any]] = []
        self.started = 0.0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> dict[str, Any]:
        self.started = time.monotonic()
        first = gpu_snapshot(self.physical_gpu, window_seconds=0.0)
        self.records.append(first)

        def poll() -> None:
            while not self.stop.wait(1.0):
                self.records.append(gpu_snapshot(self.physical_gpu, window_seconds=0.0))

        self.thread = threading.Thread(target=poll, name="e3a-gpu-exclusivity", daemon=True)
        self.thread.start()
        return first

    def finish(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        if elapsed < self.minimum_seconds:
            time.sleep(self.minimum_seconds - elapsed)
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=15.0)
        self.records.append(gpu_snapshot(self.physical_gpu, window_seconds=0.0))
        duration = time.monotonic() - self.started
        passed = bool(duration >= self.minimum_seconds and self.records and
                      all(x.get("query_ok") and x.get("exclusive") for x in self.records))
        return {
            "poll_interval_seconds": 1.0,
            "minimum_window_seconds": self.minimum_seconds,
            "window_seconds": duration,
            "before": self.records[0] if self.records else {},
            "observations": self.records,
            "after": self.records[-1] if self.records else {},
            "passed": passed,
        }


def _blocked_row(spec: dict[str, Any], tokens: int, status: str, reason: str) -> dict[str, Any]:
    return {
        "model": spec["model"], "tokens": int(tokens),
        "per_expert_M": impl.old.balanced_m(tokens, spec["top_k"], spec["experts"]),
        "K": spec["hidden"], "N_gate_up": 2 * spec["expert_inter"],
        "top_k": spec["top_k"], "n_experts": spec["experts"],
        "expert_inter": spec["expert_inter"], "status": status,
        "reasons": [reason], "timing": {},
    }


def _trace_manifest(row: dict[str, Any], trace_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for raw in row.get("traces", []):
        declared = Path(str(raw))
        candidates = [declared, trace_dir / declared, trace_dir / declared.name,
                      Path.cwd() / declared]
        path = next((p for p in candidates if p.is_file()), declared)
        item: dict[str, Any] = {"declared_path": str(declared), "path": str(path),
                                "status": "failed"}
        try:
            size = path.stat().st_size
            if size > 0:
                item.update({"status": "pass", "size_bytes": size, "sha256": _sha256(path)})
            else:
                item["error"] = "trace_empty"
        except OSError:
            item["error"] = "trace_missing"
        manifest.append(item)
    return manifest


def _csv_row(row: dict[str, Any], commit: str, physical_gpu: str) -> dict[str, Any]:
    fwd, bwd = row.get("forward_timing", {}), row.get("backward_timing", {})
    traces = row.get("trace_manifest", [])
    ex = row.get("exclusivity", {})
    return {
        "model": row.get("model", ""), "tokens": row.get("tokens", ""),
        "per_expert_M": row.get("per_expert_M", ""), "K": row.get("K", ""),
        "N_gate_up": row.get("N_gate_up", ""), "status": row.get("status", ""),
        "correctness_pass": row.get("correctness_pass", False),
        "forward_unfused_ms": fwd.get("unfused", {}).get("median_ms", ""),
        "forward_in_loop_ms": fwd.get("in_loop", {}).get("median_ms", ""),
        "backward_unfused_ms": bwd.get("unfused", {}).get("median_ms", ""),
        "backward_in_loop_ms": bwd.get("in_loop", {}).get("median_ms", ""),
        "retest_pass": row.get("retest_pass", False), "trace_count": len(traces),
        "trace_all_pass": bool(traces) and all(x.get("status") == "pass" for x in traces),
        "exclusivity_pass": ex.get("passed", ex.get("verdict", False)),
        "source_commit": commit, "physical_gpu": physical_gpu,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], commit: str, physical_gpu: str) -> None:
    fields = list(_csv_row(rows[0], commit, physical_gpu)) if rows else ["status", "source_commit"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_csv_row(row, commit, physical_gpu) for row in rows)


def _requested(only: str, tokens: str, smoke: bool) -> list[tuple[dict[str, Any], int]]:
    models = list(impl.ARCH)
    if only:
        wanted = {x.strip().lower() for x in only.split(",") if x.strip()}
        models = [x for x in models if x["model"].lower() in wanted]
    toks = [int(x.strip()) for x in tokens.split(",") if x.strip()] if tokens else list(impl.TOKENS)
    requested = [(spec, tok) for spec in models for tok in toks]
    return requested[:1] if smoke else requested


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--physical-gpu", required=True)
    ap.add_argument("--source-commit", required=True)
    args = ap.parse_args(argv)
    requested = _requested(args.only, args.tokens, args.smoke)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, Any] = {"torch": {"available": False}, "triton": {"available": False},
                               "cuda": {"available": False}}
    try:
        import torch  # type: ignore
        checks["torch"] = {"available": True, "version": str(torch.__version__)}
        checks["cuda"] = {"available": bool(torch.cuda.is_available()),
                           "version": str(torch.version.cuda),
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        import triton  # type: ignore
        checks["triton"] = {"available": True, "version": str(triton.__version__)}
    except Exception as exc:
        checks["error"] = f"{type(exc).__name__}: {exc}"

    receipt: dict[str, Any] = {
        "schema": "kernelfuser-e3a-strict-v2", "status": "running",
        "publication_eligible": False, "source_commit": args.source_commit,
        "strict_runner_sha256": _sha256(Path(__file__)),
        "implementation_sha256": _sha256(IMPL_PATH),
        "command": [sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        "environment": {"host": platform.node(), "pid": os.getpid(),
                         "python": sys.version, "python_executable": sys.executable,
                         "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                         "physical_gpu": str(args.physical_gpu)},
        "checks": checks,
        "protocol": {"warmup": 20, "samples": 100, "rounds": 5, "cuda_graph": True,
                     "paired_interleaved": True, "correctness_before_timing": True,
                     "correctness_seeds": list(impl.SEEDS), "exclusivity_poll_seconds": 1.0,
                     "retest_trigger": ">1.01x against unfused arm",
                     "optimizer_update": "not_applicable: frozen synthetic expert-local scope"},
        "scope": {"level": "isolated expert-local", "operation": "RMSNorm -> gate_up GEMM",
                  "routing": "balanced synthetic per-expert geometry; excluded",
                  "full_model": False, "claim_boundary": "no routing/full-model speedup claim"},
        "requested_rows": len(requested), "rows": [], "reasons": [],
    }

    def write_partial() -> None:
        receipt["csv"] = str(args.csv.name)
        args.out.write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
        _write_csv(args.csv, receipt["rows"], args.source_commit, str(args.physical_gpu))

    missing = [x for x in ("torch", "triton", "cuda") if not checks.get(x, {}).get("available")]
    if missing:
        reason = "required_dependency_missing:" + ",".join(missing)
        receipt.update({"status": "blocked_dependency", "reasons": [reason],
                        "rows": [_blocked_row(s, t, "blocked_dependency", reason) for s, t in requested]})
        write_partial()
        return 1

    top_pre = gpu_snapshot(str(args.physical_gpu), window_seconds=1.0)
    receipt["exclusivity_pre"] = top_pre
    if not top_pre.get("exclusive"):
        reason = "gpu_not_exclusive_at_start"
        receipt.update({"status": "blocked_exclusivity", "reasons": [reason],
                        "rows": [_blocked_row(s, t, "blocked_exclusivity", reason) for s, t in requested]})
        write_partial()
        return 1

    primary_pass = True
    for spec, tok in requested:
        print(f"RUN {spec['model']} {tok}", flush=True)
        row_pre = gpu_snapshot(str(args.physical_gpu), window_seconds=1.0)
        if not row_pre.get("exclusive"):
            row = _blocked_row(spec, tok, "blocked_exclusivity", "gpu_not_exclusive_before_row")
            row["exclusivity"] = {"pre": row_pre, "passed": False}
            primary_pass = False
        else:
            window = ExclusivityWindow(str(args.physical_gpu), minimum_seconds=1.0)
            window.start()
            try:
                row = impl.run_row(spec, tok, args.trace_dir)
                row["exclusivity"] = window.finish()
                row["trace_manifest"] = _trace_manifest(row, args.trace_dir)
                if not row["exclusivity"].get("passed"):
                    row.setdefault("reasons", []).append("gpu_not_exclusive_during_row")
                    row["status"] = "tested_invalid"
                if len(row["trace_manifest"]) != len(impl.ARMS) or any(
                        x.get("status") != "pass" for x in row["trace_manifest"]):
                    row.setdefault("reasons", []).append("trace_gate_failed")
                    row["status"] = "tested_invalid"
                if row.get("status") != "tested_pass":
                    primary_pass = False
            except Exception as exc:
                row = _blocked_row(spec, tok, "execution_error", type(exc).__name__)
                row["error"] = repr(exc)
                row["exclusivity"] = window.finish()
                row["exclusivity"]["passed"] = False
                primary_pass = False
        row["source_commit"] = args.source_commit
        row["physical_gpu"] = str(args.physical_gpu)
        row["optimizer_update"] = {"status": "not_applicable", "trainable_parameter_count": 0}
        receipt["rows"].append(row)
        receipt["completed_rows"] = len(receipt["rows"])
        receipt["status"] = "partial"
        write_partial()

    top_post = gpu_snapshot(str(args.physical_gpu), window_seconds=1.0)
    receipt["exclusivity_post"] = top_post
    complete = len(receipt["rows"]) == receipt["requested_rows"]
    rows_pass = all(row.get("status") == "tested_pass" for row in receipt["rows"])
    receipt["status"] = "tested_pass" if complete and primary_pass and rows_pass and top_post.get("exclusive") and not args.smoke else "tested_invalid"
    receipt["publication_eligible"] = receipt["status"] == "tested_pass"
    if args.smoke:
        receipt["reasons"].append("smoke_run_not_publication_eligible")
    if not complete:
        receipt["reasons"].append("row_completeness_failed")
    if not receipt["publication_eligible"] and "strict_gate_failed" not in receipt["reasons"]:
        receipt["reasons"].append("strict_gate_failed")
    write_partial()
    print(json.dumps({"status": receipt["status"], "publication_eligible": receipt["publication_eligible"],
                      "rows": len(receipt["rows"]), "out": str(args.out)}, indent=2), flush=True)
    return 0 if receipt["publication_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
