#!/usr/bin/env python3
"""Untimed CUDA correctness smoke for the public Track C candidate.

This command intentionally does not allocate a benchmark sweep or collect
timings.  It exercises one fixed manifest shape and compares the Triton
candidate with the independent FP32 eager reference.  A missing CUDA runtime
or a failed check is recorded as a non-publication status rather than turned
into a performance row.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("kernelfuser_c_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load candidate module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", type=Path,
                        default=Path(__file__).with_name("c_dlrm_concat_ln.py"))
    parser.add_argument("--manifest", type=Path,
                        default=Path(__file__).parents[1] / "data/abc/c_dlrm_manifest.json")
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--physical-gpu", default="unrecorded")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rec: dict[str, Any] = {
        "schema": "kernelfuser-abc-c-correctness-smoke-v1",
        "track": "C",
        "lane": "dlrm_concat_ln",
        "status": "started",
        "publication_eligible": False,
        "claim_boundary": "One untimed BF16 CUDA correctness smoke only; no performance claim",
        "source_commit": args.source_commit,
        "implementation_sha256": _sha256(args.implementation),
        "config_manifest_sha256": _sha256(args.manifest),
        "shape": {
            "embedding_dim": 64,
            "segment_lengths": [1, 4, 16, 64],
            "segments": 4,
            "seed": 20261016,
        },
        "timing": {"performed": False, "warmups": 0, "samples": 0},
        "exclusivity": {"physical_gpu": str(args.physical_gpu)},
    }

    try:
        import torch
        import triton

        rec["environment"] = {
            "gpu_model": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"),
            "physical_gpu": str(args.physical_gpu),
            "torch": str(torch.__version__),
            "triton": str(triton.__version__),
            "dtype": "bfloat16",
        }
        if args.device != "cuda" or not torch.cuda.is_available():
            rec["status"] = "blocked_dependency"
            rec["reason"] = "cuda_unavailable"
        else:
            candidate = _load_candidate(args.implementation)
            torch.manual_seed(rec["shape"]["seed"])
            dim = rec["shape"]["embedding_dim"]
            lengths = rec["shape"]["segment_lengths"]
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            offsets_t = torch.tensor(offsets, dtype=torch.int32, device=args.device)
            values = torch.randn((offsets[-1], dim), dtype=torch.bfloat16,
                                 device=args.device).contiguous()
            actual = candidate.jagged_concat_layernorm(values, offsets_t)
            expected = candidate.reference_jagged_concat_layernorm(values, offsets_t)
            error = (actual.float() - expected.float()).abs()
            max_abs = float(error.max().item()) if error.numel() else 0.0
            max_rel = float((error / (expected.float().abs() + 1.0e-3)).max().item()) if error.numel() else 0.0
            allclose = bool(torch.allclose(actual, expected, atol=0.02, rtol=0.02))
            rec["correctness"] = {
                "reference": "reference_jagged_concat_layernorm (FP32 eager)",
                "max_abs_error": max_abs,
                "max_relative_error": max_rel,
                "atol": 0.02,
                "rtol": 0.02,
                "allclose": allclose,
            }
            rec["status"] = "correctness_pass" if allclose else "correctness_fail"
    except Exception as exc:  # fail closed; the receipt remains auditable
        rec["status"] = "execution_error"
        rec["error_type"] = type(exc).__name__
        rec["error"] = str(exc)[:400]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
