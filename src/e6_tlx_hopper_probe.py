#!/usr/bin/env python3
"""Record the TLX compatibility gate for E6 on Hopper."""
from __future__ import annotations
import hashlib, json, os, sys, traceback
from pathlib import Path

def main() -> None:
    import torch
    rec = {
        "schema": "fusionagent-e6-tlx-compatibility-v1",
        "experiment": "E6", "status": "blocked", "publication_eligible": False,
        "development_only": False,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "python": sys.version,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tlx_import": None, "attempts": [],
    }
    try:
        import triton
        rec["triton"] = getattr(triton, "__version__", None)
        import triton.language.extra.tlx as tlx
        rec["tlx_import"] = str(tlx)
    except Exception as exc:
        rec["reason"] = "TLX import failed"; rec["exception"] = repr(exc)
        Path(sys.argv[1]).write_text(json.dumps(rec, indent=2) + "\n"); return
    try:
        import tlx_rmsnorm_matmul as m
    except Exception as exc:
        rec["reason"] = "TLX source import failed before Hopper compile"
        rec["source_import_exception"] = repr(exc)
        rec["source_import_traceback"] = traceback.format_exc(limit=12)
        Path(sys.argv[1]).write_text(json.dumps(rec, indent=2) + "\n"); return
    a = torch.randn((128, 256), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    for fused in (False, True):
        item = {"fused_rmsnorm": fused, "shape": [128, 256, 512]}
        try:
            cfg = m.get_reduced_autotune_config()[0].kwargs.copy()
            y = m.tlx_rmsnorm_matmul(a, b, config=cfg, fused_rmsnorm=fused)
            torch.cuda.synchronize(); item.update(status="compiled", output_shape=list(y.shape))
        except Exception as exc:
            item.update(status="compile_failed", exception_type=type(exc).__name__, exception=repr(exc), traceback=traceback.format_exc(limit=8))
        rec["attempts"].append(item)
    rec["reason"] = ("OSS TLX kernel requires Blackwell-only tmem; Hopper SM90 compilation failed"
                     if any(x.get("status") == "compile_failed" for x in rec["attempts"]) else "unexpectedly compiled")
    Path(sys.argv[1]).write_text(json.dumps(rec, indent=2) + "\n")

if __name__ == "__main__": main()
