#!/usr/bin/env python3
"""Fail-closed CPU/import smoke for ABC track C.

Track C currently has no public fused DLRM/DCN/HSTU implementation in this
repository.  This script records whether the public TorchRec dependency and a
reviewed lane implementation/shape manifest are present.  It never downloads
weights, launches CUDA, creates a jagged tensor, or times a kernel.  Missing
pieces are explicit ``blocked_dependency`` or ``blocked_on_implementation``
statuses; no synthetic performance result is emitted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import re
from pathlib import Path
from typing import Any


PLAN_REL = "experiments/ABC_VENDOR_ANCHOR_PLAN.md"


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _redact(value: str) -> str:
    value = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "<path>", value)
    value = re.sub(r"/(?:home|Users|mnt|tmp)/[^\s'\"]+", "<path>", value)
    return value[:400]


def _module(name: str) -> dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"available": True, "version": str(getattr(mod, "__version__", "unknown"))}
    except Exception as exc:
        return {"available": False, "error_type": type(exc).__name__, "error": _redact(str(exc))}


def _source_check(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {"status": "missing", "source": "none"}, "implementation_missing"
    out: dict[str, Any] = {"status": "missing", "file_name": path.name, "file_sha256": _sha256(path)}
    if not path.is_file():
        return out, "implementation_file_missing"
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    except Exception as exc:
        out.update({"status": "invalid_python", "error_type": type(exc).__name__, "error": _redact(str(exc))})
        return out, "implementation_invalid_python"
    out["status"] = "syntax_valid"
    return out, None


def _manifest_check(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {"status": "missing", "source": "none"}, "shape_manifest_missing"
    out: dict[str, Any] = {"status": "missing", "file_name": path.name, "file_sha256": _sha256(path)}
    if not path.is_file():
        return out, "shape_manifest_file_missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        out.update({"status": "invalid_json", "error_type": type(exc).__name__, "error": _redact(str(exc))})
        return out, "shape_manifest_invalid_json"
    required = {"feature_count", "embedding_dim", "per_feature_length", "batch"}
    missing = sorted(required - set(data)) if isinstance(data, dict) else sorted(required)
    if missing:
        out.update({"status": "missing_required_fields", "missing_fields": missing})
        return out, "shape_manifest_missing_required_fields"
    out["status"] = "valid"
    out["keys"] = sorted(str(key) for key in data)
    return out, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", default="dlrm_concat_ln")
    parser.add_argument("--implementation", type=Path, default=None,
                        help="reviewed public C runner; no code is downloaded")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="checked-in public shape manifest JSON")
    args = parser.parse_args()

    here = Path(__file__).resolve()
    plan = here.parent.parent / PLAN_REL
    rec: dict[str, Any] = {
        "schema": "kernelfuser-abc-c-import-smoke-v1",
        "track": "C",
        "lane": args.lane,
        "status": "started",
        "publication_eligible": False,
        "claim_boundary": "CPU/import/source-manifest smoke only; no timing or performance claim",
        "source_sha256": _sha256(here),
        "plan": {"path": PLAN_REL, "sha256": _sha256(plan)},
        "checks": {},
        "reasons": [],
    }

    torch_info = _module("torch")
    triton_info = _module("triton")
    torchrec_info = _module("torchrec")
    jagged_info = _module("torchrec.sparse.jagged_tensor")
    fbgemm_info = _module("fbgemm_gpu")
    rec["checks"].update({
        "torch": torch_info,
        "triton": triton_info,
        "torchrec": torchrec_info,
        "torchrec_jagged_tensor": jagged_info,
        "fbgemm_gpu": fbgemm_info,
    })

    source, source_reason = _source_check(args.implementation)
    manifest, manifest_reason = _manifest_check(args.manifest)
    rec["implementation"] = source
    rec["shape_manifest"] = manifest
    if source_reason:
        rec["reasons"].append(source_reason)
    if manifest_reason:
        rec["reasons"].append(manifest_reason)
    if not torch_info.get("available"):
        rec["reasons"].append("torch_import_failed")
    if not torchrec_info.get("available") or not jagged_info.get("available"):
        rec["reasons"].append("torchrec_jagged_dependency_missing")
    if not triton_info.get("available"):
        rec["reasons"].append("triton_import_failed")

    rec["status"] = "ready_for_implementation" if not rec["reasons"] else (
        "blocked_dependency" if "torchrec_jagged_dependency_missing" in rec["reasons"]
        else "blocked_on_implementation"
    )
    rec["reasons"] = sorted(set(rec["reasons"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
