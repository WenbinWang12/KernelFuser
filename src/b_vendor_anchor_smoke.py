#!/usr/bin/env python3
"""Fail-closed import/config smoke for ABC track B.

This is deliberately not a benchmark.  It never loads model weights, allocates
model-sized tensors, launches a kernel, or records timing.  It only answers
whether an the target GPU host node has the dependencies needed to implement the pre-registered
QKV-output Q/K RMSNorm+RoPE experiment.  A ``ready`` receipt is still smoke
evidence only and always has ``publication_eligible=false``.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any


PLAN_REL = "experiments/ABC_VENDOR_ANCHOR_PLAN.md"
REQUIRED_CONFIG_KEYS = ("hidden_size", "num_attention_heads", "num_key_value_heads")


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _redact(value: str) -> str:
    # Exception strings from imports can contain a local checkout or cache path.
    value = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "<path>", value)
    value = re.sub(r"/(?:home|Users|mnt|tmp)/[^\s'\"]+", "<path>", value)
    return value[:400]


def _module(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"available": True, "version": str(getattr(mod, "__version__", "unknown"))}
    except Exception as exc:  # dependency smoke; do not hide the status
        return {"available": False, "error_type": type(exc).__name__, "error": _redact(str(exc))}


def _config_summary(path: Path | None, model_id: str) -> tuple[dict[str, Any], list[str]]:
    summary: dict[str, Any] = {"model_id": model_id, "source": "none"}
    reasons: list[str] = []
    if path is None:
        summary["status"] = "missing"
        reasons.append("model_config_missing_no_network_download")
        return summary, reasons
    summary["source"] = "local_json"
    summary["file_sha256"] = _sha256(path)
    summary["file_name"] = path.name
    if not path.is_file():
        summary["status"] = "missing"
        reasons.append("model_config_file_missing")
        return summary, reasons
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        summary["status"] = "invalid_json"
        summary["error_type"] = type(exc).__name__
        summary["error"] = _redact(str(exc))
        reasons.append("model_config_invalid_json")
        return summary, reasons
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in data]
    if missing:
        summary["status"] = "missing_required_fields"
        summary["missing_fields"] = missing
        reasons.append("model_config_missing_required_fields")
        return summary, reasons
    head_dim = data.get("head_dim")
    if head_dim is None:
        hidden = data.get("hidden_size")
        heads = data.get("num_attention_heads")
        head_dim = hidden // heads if isinstance(hidden, int) and isinstance(heads, int) and heads else None
    summary["geometry"] = {
        "hidden_size": data.get("hidden_size"),
        "num_attention_heads": data.get("num_attention_heads"),
        "num_key_value_heads": data.get("num_key_value_heads"),
        "head_dim": head_dim,
        "rms_norm_eps": data.get("rms_norm_eps", data.get("rms_norm_eps")),
        "rope_theta": data.get("rope_theta"),
    }
    if any(not isinstance(summary["geometry"][key], int) or summary["geometry"][key] <= 0
           for key in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")):
        summary["status"] = "invalid_geometry"
        reasons.append("model_config_invalid_geometry")
    else:
        summary["status"] = "valid"
    return summary, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--config-json", type=Path, default=None,
                        help="local public config.json; never downloads a model")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    plan = here.parent.parent / PLAN_REL
    rec: dict[str, Any] = {
        "schema": "kernelfuser-abc-b-import-smoke-v1",
        "track": "B",
        "status": "started",
        "publication_eligible": False,
        "claim_boundary": "CPU/import/config smoke only; no timing or performance claim",
        "source_sha256": _sha256(here),
        "plan": {"path": PLAN_REL, "sha256": _sha256(plan)},
        "model": _config_summary(args.config_json, args.model_id)[0],
        "checks": {},
        "reasons": [],
    }
    rec["model"], config_reasons = _config_summary(args.config_json, args.model_id)
    rec["reasons"].extend(config_reasons)

    torch_info = _module("torch")
    triton_info = _module("triton")
    transformers_info = _module("transformers")
    flash_info = _module("flash_attn")
    rec["checks"]["torch"] = torch_info
    rec["checks"]["triton"] = triton_info
    rec["checks"]["transformers"] = transformers_info
    rec["checks"]["flash_attn"] = flash_info

    cuda_available = False
    cuda_built = False
    sdpa_available = False
    compile_available = False
    if torch_info["available"]:
        try:
            torch = importlib.import_module("torch")
            import torch.nn.functional as F
            cuda_available = bool(torch.cuda.is_available())
            cuda_built = bool(torch.backends.cuda.is_built())
            sdpa_available = hasattr(F, "scaled_dot_product_attention")
            compile_available = callable(getattr(torch, "compile", None))
            rec["environment"] = {
                "torch": str(torch.__version__),
                "cuda_runtime": torch.version.cuda,
                "cuda_built": cuda_built,
                "cuda_available": cuda_available,
                "gpu_name": str(torch.cuda.get_device_name()) if cuda_available else None,
                "triton": triton_info.get("version"),
            }
        except Exception as exc:
            rec["reasons"].append("torch_runtime_probe_failed")
            rec["checks"]["torch_runtime_probe"] = {
                "available": False, "error_type": type(exc).__name__, "error": _redact(str(exc))
            }
    else:
        rec["reasons"].append("torch_import_failed")

    rec["checks"]["vendor_sdpa"] = {
        "torch_scaled_dot_product_attention": sdpa_available,
        "cuda_backend_built": cuda_built,
        "cuda_available": cuda_available,
        "flash_attn_import": flash_info.get("available", False),
    }
    rec["checks"]["torch_compile"] = {"available": compile_available}
    if not transformers_info.get("available"):
        rec["reasons"].append("transformers_import_failed")
    if not triton_info.get("available"):
        rec["reasons"].append("triton_import_failed")
    if not sdpa_available and not flash_info.get("available", False):
        rec["reasons"].append("vendor_attention_missing")
    if not cuda_available:
        rec["reasons"].append("cuda_unavailable_for_vendor_attention")

    # B is ready only to proceed to a reviewed implementation; it is never a
    # performance result.  A local config is mandatory so no network download
    # or hidden model revision can enter the campaign.
    if not rec["reasons"]:
        rec["status"] = "ready_for_implementation"
    else:
        rec["status"] = "blocked_dependency"
    rec["reasons"] = sorted(set(rec["reasons"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
