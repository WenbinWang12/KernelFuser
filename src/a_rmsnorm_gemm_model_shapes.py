#!/usr/bin/env python3
"""Auditable model-shaped RMSNorm -> GEMM campaign (experiment A).

This runner deliberately fails closed.  The candidate is the TLX-backed
``launch(kind, x, w, out)`` implementation from ``tlx_ads_sm90_port``; if CUDA,
the TLX namespace, or an SM90 candidate compile is unavailable, the receipt is
``blocked_dependency`` and no Triton fallback is silently promoted.  The
default campaign uses public model geometries and random BF16 tensors.  It
does not load checkpoints or embed host paths, UUIDs, PIDs, or raw traces in
the receipt (optional private traces are represented by hashes only).

The candidate folds the RMSNorm weight into the GEMM weight once, outside the
timed region (``W_fold = W * gamma[:, None]``).  The gate/up site uses a
concatenated gate-and-up width (``2 * intermediate_size``).  Timed arms are:

* ``separate_norm_cublas``: PyTorch RMSNorm followed by a vendor matmul;
* ``row_scale_cublas``: raw BF16 vendor matmul with FP32 row scaling; and
* ``torch_compile_max_autotune``: the same RMSNorm + matmul graph compiled by
  TorchInductor's ``max-autotune`` mode.

All correctness checks run before timing.  Ratios use the fastest successful
baseline for each row.  This is a forward-only operator study; it makes no
full-model, training, or checkpoint-quality claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import random
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F


EPS = 1e-6
DTYPE = torch.bfloat16
DEFAULT_M = (1, 8, 32, 128, 512, 2048, 8192)

# These are the nine public text-model configurations pre-registered for A.
# ``qkv_n`` is derived from GQA (Q + K + V).  ``gate_up_n`` is the concatenated
# gate/up projection (2 * intermediate_size), matching the handoff protocol.
# Config references are public and weights are never loaded.
MODEL_GEOMETRIES: dict[str, dict[str, int | str]] = {
    "Qwen3-0.6B": {"hidden": 1024, "q_heads": 16, "kv_heads": 8,
                    "head_dim": 128, "intermediate": 3072,
                    "config_ref": "Qwen/Qwen3-0.6B:config.json"},
    "Qwen3-1.7B": {"hidden": 2048, "q_heads": 16, "kv_heads": 8,
                    "head_dim": 128, "intermediate": 6144,
                    "config_ref": "Qwen/Qwen3-1.7B:config.json"},
    "Qwen3-4B": {"hidden": 2560, "q_heads": 32, "kv_heads": 8,
                  "head_dim": 128, "intermediate": 9728,
                  "config_ref": "Qwen/Qwen3-4B:config.json"},
    "Qwen3-8B": {"hidden": 4096, "q_heads": 32, "kv_heads": 8,
                  "head_dim": 128, "intermediate": 12288,
                  "config_ref": "Qwen/Qwen3-8B:config.json"},
    "Qwen3-32B": {"hidden": 5120, "q_heads": 64, "kv_heads": 8,
                   "head_dim": 128, "intermediate": 25600,
                   "config_ref": "Qwen/Qwen3-32B:config.json"},
    "Qwen2.5-0.5B-Instruct": {"hidden": 896, "q_heads": 14, "kv_heads": 2,
                               "head_dim": 64, "intermediate": 4864,
                               "config_ref": "Qwen/Qwen2.5-0.5B-Instruct:config.json"},
    "OLMo-1B": {"hidden": 2048, "q_heads": 16, "kv_heads": 16,
                 "head_dim": 128, "intermediate": 8192,
                 "config_ref": "allenai/OLMo-1B:config.json"},
    "Llama-3.1-8B": {"hidden": 4096, "q_heads": 32, "kv_heads": 8,
                      "head_dim": 128, "intermediate": 14336,
                      "config_ref": "meta-llama/Llama-3.1-8B:config.json"},
    "Mistral-7B": {"hidden": 4096, "q_heads": 32, "kv_heads": 8,
                    "head_dim": 128, "intermediate": 14336,
                    "config_ref": "mistralai/Mistral-7B-v0.3:config.json"},
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


GEOMETRY_CONFIG_HASH = hashlib.sha256(_canonical(MODEL_GEOMETRIES)).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_environment() -> dict[str, Any]:
    """Return reproducibility fields without host, PID, UUID, or paths."""
    env: dict[str, Any] = {
        "torch": getattr(torch, "__version__", "unknown"),
        "python": platform.python_version(),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import triton

        env["triton"] = getattr(triton, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - dependency diagnostic
        env["triton"] = None
        env["triton_import_error"] = type(exc).__name__
    if torch.cuda.is_available():
        try:
            env["device"] = torch.cuda.get_device_name()
            env["capability"] = list(torch.cuda.get_device_capability())
        except Exception as exc:  # pragma: no cover - driver diagnostic
            env["device_error"] = type(exc).__name__
    return env


def _load_candidate() -> tuple[Any | None, dict[str, Any] | None, str | None]:
    """Import the TLX candidate and return (module, metadata, blocker)."""
    if not torch.cuda.is_available():
        return None, None, "cuda_unavailable"
    try:
        module = importlib.import_module("tlx_ads_sm90_port.ads_sm90_rmsnorm_matmul")
    except Exception as exc:
        return None, None, f"tlx_import_failed:{type(exc).__name__}"
    try:
        tlx = importlib.import_module("triton.language.extra.tlx")
        del tlx
    except Exception as exc:
        return None, None, f"tlx_namespace_unavailable:{type(exc).__name__}"
    capability = tuple(torch.cuda.get_device_capability())
    if not capability or capability[0] != 9:
        return None, None, f"candidate_requires_sm90:capability={capability}"
    source = Path(getattr(module, "__file__", ""))
    if not source.is_file():
        return None, None, "candidate_source_path_unavailable"
    return module, {
        "candidate_source_sha256": _sha256_file(source),
        "candidate_implementation": "tlx_ads_sm90_port.launch(kind='tlx', x, w, out)",
        "candidate_scope": "SM90 ADS-derived TLX port; forward-only; not upstream SM100 TLX",
    }, None


def _geometry(model: str, site: str) -> tuple[int, int, dict[str, Any]]:
    cfg = MODEL_GEOMETRIES[model]
    k = int(cfg["hidden"])
    qkv_n = (int(cfg["q_heads"]) + 2 * int(cfg["kv_heads"])) * int(cfg["head_dim"])
    if site == "qkv":
        return k, qkv_n, {"projection": "qkv", **cfg, "N": qkv_n}
    if site == "gate_up":
        n = 2 * int(cfg["intermediate"])
        return k, n, {"projection": "gate_or_up", **cfg, "N": n,
                      "gate_up_width": "2*intermediate_size"}
    raise ValueError(f"unknown site: {site}")


def _make_case(module: Any, model: str, site: str, m: int, seed: int) -> dict[str, Any]:
    k, n, geom = _geometry(model, site)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=DTYPE, generator=g)
    w = torch.randn((n, k), device="cuda", dtype=DTYPE, generator=g) * (k ** -0.5)
    gamma = (torch.randn((k,), device="cuda", dtype=DTYPE, generator=g) * 0.1 + 1).contiguous()
    # W is [N,K] for torch.linear; the TLX wrapper consumes [K,N].
    w_fold = (w.float() * gamma.float()[None, :]).to(DTYPE).contiguous()
    w_tlx = w_fold.t().contiguous()
    rstd = torch.empty((m,), device="cuda", dtype=torch.float32)
    raw = torch.empty((m, n), device="cuda", dtype=torch.float32)
    out = torch.empty((m, n), device="cuda", dtype=DTYPE)

    def separate_norm_cublas() -> torch.Tensor:
        y = F.rms_norm(x, (k,), weight=gamma, eps=EPS)
        return torch.matmul(y, w.t())

    def row_scale_cublas() -> torch.Tensor:
        xf = x.float()
        r = torch.rsqrt((xf * xf).mean(dim=1) + EPS)
        rstd.copy_(r)
        # CUDA supports BF16 inputs with FP32 output for torch.mm. Keep a
        # fallback for older releases, while recording the selected path.
        try:
            torch.mm(x, w_fold.t(), out=raw, out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            raw.copy_(torch.matmul(x.float(), w_fold.t().float()))
        return out.copy_(raw * r[:, None]).to(DTYPE)

    def torch_compile_max_autotune() -> torch.Tensor:
        return compiled(x, w.t(), gamma)

    def unfused_vendor_anchor() -> torch.Tensor:
        """Standalone vendor GEMM used for the anchor gate."""
        return torch.matmul(x, w_fold.t())

    def cublas_reference() -> torch.Tensor:
        """Explicit cuBLAS reference API for the standalone anchor check."""
        return torch.mm(x, w_fold.t())

    def tlx_fused() -> torch.Tensor:
        # The checked-in SM90 port exposes ``launch(kind, x, w, out)`` and
        # performs the RMS reduction inside the TLX kernel.  ``w_tlx`` already
        # contains the folded gamma, so this is the lazy-norm contract.
        module.launch("tlx", x, w_tlx, out)
        return out

    try:
        compiled = torch.compile(
            lambda a, wt, gg: F.rms_norm(a, (k,), weight=gg, eps=EPS) @ wt,
            mode="max-autotune", fullgraph=True,
        )
        # Force compilation before correctness/timing accounting.
        compiled(x, w.t(), gamma)
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"torch_compile_unavailable:{type(exc).__name__}") from exc

    reference = (x.float() * torch.rsqrt((x.float() * x.float()).mean(dim=1, keepdim=True) + EPS)
                 * gamma.float()) @ w.float().t()
    reference = reference.to(DTYPE)
    arms: dict[str, Callable[[], torch.Tensor]] = {
        "separate_norm_cublas": separate_norm_cublas,
        "row_scale_cublas": row_scale_cublas,
        "torch_compile_max_autotune": torch_compile_max_autotune,
        "tlx_fused": tlx_fused,
    }
    anchor_arms: dict[str, Callable[[], torch.Tensor]] = {
        "unfused_vendor_anchor": unfused_vendor_anchor,
        "cublas_reference": cublas_reference,
    }
    # Compile the candidate before any correctness or timing arm.  A smoke
    # run therefore distinguishes a missing/unsupported TLX backend from an
    # ordinary numerical failure and can emit ``blocked_dependency``.
    try:
        tlx_fused()
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError(f"tlx_candidate_compile:{type(exc).__name__}") from exc
    correctness: dict[str, dict[str, Any]] = {}
    for name, fn in arms.items():
        y = fn()
        torch.cuda.synchronize()
        diff = y.float() - reference.float()
        den = torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)
        correctness[name] = {
            "max_abs": float(diff.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(diff) / den),
            "finite": bool(torch.isfinite(y).all().item()),
            "allclose": bool(torch.allclose(y, reference, atol=0.15, rtol=0.02)),
        }
    if not all(v["finite"] and v["allclose"] for v in correctness.values()):
        return {"geometry": geom, "correctness": correctness, "status": "tested_invalid",
                "error": "correctness_gate_failed"}
    return {"geometry": geom, "arms": arms, "anchor_arms": anchor_arms,
            "correctness": correctness,
            "reference": reference, "status": "ready"}


def _time_paired(arms: dict[str, Callable[[], torch.Tensor]], rounds: int,
                 samples: int, warmup: int, seed: int) -> dict[str, dict[str, float]]:
    for fn in arms.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    values: dict[str, list[float]] = {name: [] for name in arms}
    rng = random.Random(seed)
    for _round in range(rounds):
        names = list(arms)
        for _sample in range(samples):
            order = names[:]
            rng.shuffle(order)
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                arms[name]()
                end.record()
                end.synchronize()
                values[name].append(float(start.elapsed_time(end) * 1000.0))
    summary: dict[str, dict[str, float]] = {}
    for name, vals in values.items():
        summary[name] = {
            "n": float(len(vals)),
            "median_us": float(statistics.median(vals)),
            "mean_us": float(statistics.fmean(vals)),
            "min_us": float(min(vals)),
            "max_us": float(max(vals)),
        }
    return summary


def _capture_traces(arms: dict[str, Callable[[], torch.Tensor]], trace_dir: Path,
                    stem: str) -> dict[str, dict[str, Any]]:
    """Capture private traces and return only hashes/counts for the receipt."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    traces: dict[str, dict[str, Any]] = {}
    for name, fn in arms.items():
        path = trace_dir / f"{stem}_{name}.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            fn()
            torch.cuda.synchronize()
            prof.step()
        prof.export_chrome_trace(str(path))
        traces[name] = {"sha256": _sha256_file(path),
                        "bytes": path.stat().st_size}
    return traces


def _blocked_receipt(out: Path, reason: str, args: argparse.Namespace) -> None:
    rec = {
        "schema": "kernelfuser-a-rmsnorm-gemm-v1",
        "status": "blocked_dependency",
        "publication_eligible": False,
        "blocker": reason,
        "source_commit": args.source_commit,
        "source_sha256": _sha256_file(Path(__file__)),
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "environment": _safe_environment(),
        "geometry_config_sha256": GEOMETRY_CONFIG_HASH,
        "config_manifest_sha256": GEOMETRY_CONFIG_HASH,
        "protocol": {"candidate": "TLX only; no fallback promotion",
                     "dtype": "bfloat16", "epsilon": EPS,
                     "correctness_before_timing": True,
                     "fastest_baseline_rule": "min successful baseline median"},
        "requested": {"models": args.models, "sites": args.sites,
                       "m_values": args.m_values, "smoke": bool(args.smoke)},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    # Keep parser construction explicit: this file is also copied into remote
    # handoff directories, where argparse monkey patches are undesirable.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--models", default=",".join(MODEL_GEOMETRIES))
    ap.add_argument("--sites", default="qkv,gate_up")
    ap.add_argument("--m-values", default=",".join(map(str, DEFAULT_M)))
    ap.add_argument("--smoke", action="store_true",
                    help="one Qwen3-8B QKV M=128 compile/correctness row")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--trace-dir", type=Path, default=None,
                    help="private trace output directory; only hashes enter JSON")
    ap.add_argument("--source-commit", default="working-tree",
                    help="commit used for the run (required for promotion)")
    args = ap.parse_args()
    if args.smoke:
        args.models, args.sites, args.m_values = "Qwen3-8B", "qkv", "128"
    out = args.out
    module, candidate_meta, blocker = _load_candidate()
    if blocker:
        _blocked_receipt(out, blocker, args)
        return 2
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    sites = [x.strip() for x in args.sites.split(",") if x.strip()]
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    unknown = [m for m in models if m not in MODEL_GEOMETRIES]
    if unknown or any(s not in ("qkv", "gate_up") for s in sites):
        _blocked_receipt(out, "invalid_geometry_arguments", args)
        return 2
    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for model in models:
        for site in sites:
            for m in m_values:
                print(f"RUN A {model} {site} M={m}", flush=True)
                row_seed = 20261015 + len(rows)
                try:
                    case = _make_case(module, model, site, m,
                                      row_seed)
                    row: dict[str, Any] = {
                        "model": model, "site": site, "M": m, "seed": row_seed,
                        "geometry": case["geometry"],
                        "correctness": case["correctness"],
                        "status": case.get("status", "tested_invalid"),
                    }
                    if row["status"] != "ready":
                        row["error"] = case.get("error", "unknown")
                    else:
                        timed_arms = {**case["arms"], **case["anchor_arms"]}
                        timing = _time_paired(timed_arms, args.rounds,
                                              args.samples, args.warmup,
                                              7109 + len(rows))
                        row["timing_us"] = timing
                        if args.trace_dir is not None:
                            row["traces"] = _capture_traces(
                                case["arms"], args.trace_dir,
                                f"{model}_{site}_M{m}")
                        baselines = [timing[x]["median_us"] for x in (
                            "separate_norm_cublas", "row_scale_cublas",
                            "torch_compile_max_autotune")]
                        fastest = min(baselines)
                        cand = timing["tlx_fused"]["median_us"]
                        initial_speedup = fastest / cand
                        selected_timing = timing
                        if initial_speedup > 1.01:
                            retest = _time_paired(
                                timed_arms, args.rounds, args.samples,
                                args.warmup, 91031 + len(rows))
                            row["retest_timing_us"] = retest
                            fastest_retest = min(retest[x]["median_us"] for x in (
                                "separate_norm_cublas", "row_scale_cublas",
                                "torch_compile_max_autotune"))
                            cand_retest = retest["tlx_fused"]["median_us"]
                            row["retest_speedup_vs_fastest"] = fastest_retest / cand_retest
                            fastest, cand = fastest_retest, cand_retest
                            selected_timing = retest
                        anchor_ratio = (selected_timing["unfused_vendor_anchor"]["median_us"] /
                                        selected_timing["cublas_reference"]["median_us"])
                        row.update({"fastest_baseline_us": fastest,
                                    "initial_speedup_vs_fastest": initial_speedup,
                                    "speedup_vs_fastest": fastest / cand,
                                    "anchor_ratio": anchor_ratio,
                                    "anchor_gate": anchor_ratio <= 1.01,
                                    "fastest_baseline": min(
                                        ("separate_norm_cublas",
                                         "row_scale_cublas",
                                         "torch_compile_max_autotune"),
                                        key=lambda x: selected_timing[x]["median_us"]),
                                    "status": "tested_pass" if anchor_ratio <= 1.01
                                    else "anchor_gate_failed"})
                        csv_rows.append({
                            "model": model, "site": site, "M": m,
                            "seed": row_seed,
                            "K": case["geometry"]["hidden"],
                            "N": case["geometry"]["N"],
                            "candidate_us": cand,
                            "fastest_baseline_us": fastest,
                            "speedup_vs_fastest": fastest / cand,
                            "retest": "yes" if initial_speedup > 1.01 else "no",
                            "fastest_baseline": row["fastest_baseline"],
                            "correctness_pass": True,
                            "anchor_ratio": anchor_ratio,
                            "anchor_gate": anchor_ratio <= 1.01,
                            "status": row["status"],
                        })
                except torch.cuda.OutOfMemoryError:  # retain declared shape
                    row = {"model": model, "site": site, "M": m,
                           "seed": row_seed, "status": "oom",
                           "error": "OutOfMemoryError"}
                    torch.cuda.empty_cache()
                except Exception as exc:  # fail closed per row
                    if args.smoke and "tlx_candidate_compile" in str(exc):
                        _blocked_receipt(out, str(exc), args)
                        return 2
                    row = {"model": model, "site": site, "M": m,
                           "seed": row_seed,
                           "status": "tested_invalid",
                           # Keep receipts shareable: compiler exceptions can
                           # embed local source paths, so retain only a stable
                           # exception class/code.
                           "error": type(exc).__name__}
                if row["status"] in ("tested_invalid", "oom"):
                    k_bad, n_bad, _ = _geometry(model, site)
                    csv_rows.append({
                        "model": model, "site": site, "M": m,
                        "seed": row_seed, "K": k_bad, "N": n_bad,
                        "candidate_us": "", "fastest_baseline_us": "",
                        "speedup_vs_fastest": "", "fastest_baseline": "",
                        "retest": "no", "correctness_pass": False,
                        "status": row["status"],
                    })
                rows.append(row)
                print("DONE", row["status"], flush=True)
    rec = {
        "schema": "kernelfuser-a-rmsnorm-gemm-v1",
        "status": "tested_pass" if rows and all(r["status"] == "tested_pass" for r in rows) else "tested_invalid",
        "publication_eligible": bool(rows) and all(r["status"] == "tested_pass" for r in rows),
        "source_commit": args.source_commit,
        "environment": _safe_environment(),
        "candidate": candidate_meta,
        "source_sha256": _sha256_file(Path(__file__)),
        "runner_source_sha256": _sha256_file(Path(__file__)),
        "geometry_config_sha256": GEOMETRY_CONFIG_HASH,
        "config_manifest_sha256": GEOMETRY_CONFIG_HASH,
        "protocol": {"dtype": "bfloat16", "epsilon": EPS,
                     "rounds": args.rounds, "samples": args.samples,
                     "warmup": args.warmup, "paired_randomized": True,
                     "traces": "private Chrome traces; receipt stores hashes only"
                     if args.trace_dir is not None else "disabled",
                     "correctness_before_timing": True,
                     "fastest_baseline_rule": "minimum of separate_norm_cublas, row_scale_cublas, torch_compile_max_autotune",
                     "claim_boundary": "forward-only operator; public geometry; no checkpoint/full-model claim"},
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
    if args.csv is not None and csv_rows:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            fields = list(csv_rows[0])
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(csv_rows)
    return 0 if rec["publication_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
