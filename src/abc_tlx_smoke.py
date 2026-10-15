#!/usr/bin/env python3
"""One-shape TLX provenance/correctness smoke for the ABC campaign.

This intentionally fails closed when the installed Triton does not expose the
TLX extension.  It is a dependency check, not a performance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--N", type=int, default=3072)
    args = ap.parse_args()
    rec = {
        "schema": "kernelfuser-abc-tlx-smoke-v1",
        "track": "A",
        "status": "started",
        "publication_eligible": False,
        "source_sha256": source_hash(),
        "shape": {"M": args.M, "K": args.K, "N": args.N, "dtype": "bfloat16"},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    try:
        import torch
        import triton

        rec["environment"] = {
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
        }
        try:
            from tlx_ads_sm90_port import ads_sm90_rmsnorm_matmul as impl
        except Exception as exc:  # dependency gate: do not substitute a fallback
            rec.update({
                "status": "blocked_dependency",
                "reason": "tlx_import_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(rec, indent=2) + "\n")
            return

        torch.manual_seed(20261015)
        x = torch.randn((args.M, args.K), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((args.K, args.N), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((args.M, args.N), device="cuda", dtype=torch.bfloat16)
        ref = (x.float() * torch.rsqrt((x.float() * x.float()).mean(1, keepdim=True) + 1e-6)) @ w.float()
        impl.launch("tlx", x, w, out)
        torch.cuda.synchronize()
        rec["correctness"] = {
            "max_abs": float((out.float() - ref.float()).abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(out.float() - ref.float()) / torch.linalg.vector_norm(ref.float())),
            "allclose": bool(torch.allclose(out, ref.to(torch.bfloat16), atol=0.15, rtol=0.02)),
        }
        if not rec["correctness"]["allclose"]:
            rec["status"] = "tested_invalid"
            rec["reason"] = "correctness_gate"
        else:
            rec["status"] = "tested_pass"
            rec["publication_eligible"] = False
            rec["reason"] = "smoke_only"
    except Exception as exc:
        rec.update({"status": "tested_invalid", "error_type": type(exc).__name__, "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=2) + "\n")


if __name__ == "__main__":
    main()
