#!/usr/bin/env python3
"""OSDI re-census driver (written 2026-11-20).

Runs the checked-in refined census runner (e1_rope_refined_census.py) with the
checked-in kernels (e1_rope_refined_impl.py).  Nothing in those two files is
edited.  This driver adds three things, all recorded in the receipt:

1. Qwen3-1.7B geometry is corrected to (Hq, Hkv, D) = (16, 8, 128), the
   released model configuration.  The checked-in table still lists 32 heads.
2. Models whose attention scale differs from D**-0.5 (Gemma-3-27B: D=128 with
   query_pre_attn_scalar=256), and every model on a GPU that is not Hopper,
   launch the same two kernels directly.  impl.optimized_attention() hard-codes D**-0.5, so for those models
   this driver repeats the body of impl.launch_attention() with one argument
   changed.  All other models go through impl.optimized_attention() itself.
3. --mode norope measures Q/K RMSNorm without RoPE (the controlled census of
   the paper) by launching the same kernels with ROPE=False and by making the
   reference, vendor and compiled arms skip the rotation.

Arms (unchanged from the checked-in runner): candidate = query fusion (K is
prepared once, Q normalization [+RoPE] fused into attention); unfused = same
attention kernel after a separate Q/K preparation kernel; compiled =
torch.compile(default) of norm[+RoPE]+SDPA; vendor = eager norm[+RoPE]+SDPA.
"""
from __future__ import annotations

import argparse, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
import triton
import e1_rope_refined_impl as impl
import e1_rope_refined_census as census

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("rope", "norope"), required=True)
ap.add_argument("--list", action="store_true", help="print the planned rows and exit (no GPU work)")
ap.add_argument("--bm", type=int, default=64, choices=(32, 64, 128),
                help="query tile of the attention kernel; values other than 64 use the direct launch")
known, rest = ap.parse_known_args()
MODE = known.mode
ROTARY = MODE == "rope"
BM = known.bm

# 1. geometry correction
impl.MODELS = dict(impl.MODELS)
impl.MODELS["Qwen3-1.7B"] = (16, 8, 128)

# 3. no-RoPE mode: reference/vendor/compiled arms skip the rotation
if not ROTARY:
    def _identity_rope(x, cos, sin, *, fp32=False):
        return x.float() if fp32 else x
    impl.rope_torch = _identity_rope

SCALE_PATHS = {}


def launch(xs, *, causal, mode, bn, scale):
    q, k, v, gq, gk, cos, sin = xs
    b, h, s, d = q.shape
    cap = torch.cuda.get_device_capability(q.device)
    if abs(scale - d ** -0.5) < 1e-12 and cap[0] == 9 and BM == 64:
        SCALE_PATHS[(h, k.shape[1], d)] = "impl.optimized_attention"
        return impl.optimized_attention(
            q, k, v, gq, gk, cos if ROTARY else None, sin if ROTARY else None,
            causal=causal, mode=mode, bn=bn, gather=False, norm_style="hf")
    # 2. same kernels and schedule as impl.launch_attention(mode, bm=64, bn=bn,
    #    warps=4, stages=1, gather=False, hf_norm=True); only SCALE differs.
    # The same path serves GPUs that are not Hopper (compute capability 9): impl.optimized_attention()
    # refuses them, although the two kernels are plain Triton and compile for other architectures.
    SCALE_PATHS[(h, k.shape[1], d)] = "driver launch, scale %.6f, compute capability %d.%d" % (scale, cap[0], cap[1])
    out = torch.empty_like(q)
    kn = torch.empty_like(k)
    qn = torch.empty_like(q) if mode == "unfused" else None
    inputs = (q, k, v, gq, gk, cos if ROTARY else q, sin if ROTARY else k)
    with torch.cuda.device(q.device):
        impl.prepare_attention(inputs, qn if qn is not None else out, kn,
                               only_k=(mode == "query"), rope=ROTARY, hf_norm=True)
        impl.attention_v2_kernel[(triton.cdiv(s, BM), h, b)](
            q if mode != "unfused" else qn, kn, v, gq, gk, inputs[5], inputs[6], out,
            s, d, h, k.shape[1], scale, causal, mode != "unfused", False,
            ROTARY, True, False, BM, bn, num_warps=4, num_stages=1)
    return out


def make_arms(xs, causal, scale, bn=128):
    candidate = lambda: launch(xs, causal=causal, mode="query", bn=bn, scale=scale)
    unfused = lambda: launch(xs, causal=causal, mode="unfused", bn=bn, scale=scale)
    vendor = census.vendor_fn(xs, causal, scale)
    compiled = torch.compile(vendor, fullgraph=True, dynamic=False, mode="default")
    return {"candidate": candidate, "unfused": unfused, "compiled": compiled, "vendor": vendor}


census.make_arms = make_arms


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _arg(name, default=None):
    return rest[rest.index(name) + 1] if name in rest else default


if known.list:
    models = _arg("--models", ",".join(impl.MODELS)).split(",")
    sizes = _arg("--sizes", "128,256,384,512").split(",")
    batches = _arg("--batches", "1,8,32").split(",")
    masks = _arg("--causal-modes", "false,true").split(",")
    n = len(models) * len(sizes) * len(batches) * len(masks)
    print(json.dumps({"mode": MODE, "models": {m: impl.MODELS[m] for m in models},
                      "scales": {m: round(impl.model_scale(m, impl.MODELS[m][2]), 6) for m in models},
                      "sizes": sizes, "batches": batches, "masks": masks, "rows": n}))
    sys.exit(0)

sys.argv = [sys.argv[0]] + rest
census.main()

out = Path(_arg("--output"))
rec = json.loads(out.read_text(encoding="utf-8"))
rec["driver"] = {
    "file": "recensus_driver.py", "sha256": _sha(__file__), "mode": MODE,
    "rotary": ROTARY,
    "geometry_correction": {"Qwen3-1.7B": [16, 8, 128]},
    "compute_capability": list(torch.cuda.get_device_capability()),
    "bm": BM,
    "scale_paths": {"%d/%d/%d" % k: v for k, v in SCALE_PATHS.items()},
    "note": "kernels and census runner are the checked-in files; see driver docstring",
}
if not ROTARY:
    rec["scope"] = "isolated Q/K RMSNorm (no RoPE); query fusion prepares K once"
out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"driver": rec["driver"]["mode"], "status": rec["status"], "rows": len(rec["rows"])}))
