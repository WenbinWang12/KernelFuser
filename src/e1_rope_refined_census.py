#!/usr/bin/env python3
"""Publication census for the refined K-once RoPE attention kernel.

This runner deliberately lives beside the original E1 runner.  It imports the
source-audited implementation from ``e1_rope_refined_impl.py`` and records a
fresh receipt for every model/shape.  The candidate is the query-fused mode
(K is prepared once); the unfused, compiled, and vendor paths are all timed
as paired baselines.  No timing is retained when the true-FP32 correctness or
GPU-exclusivity gate fails.
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, statistics, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import e1_rope_refined_impl as impl  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def vendor_fn(xs, causal, scale):
    q, k, v, gq, gk, cos, sin = xs

    def run():
        qn = impl.rope_torch(impl.norm_torch(q, gq), cos, sin)
        kn = impl.rope_torch(impl.norm_torch(k, gk), cos, sin)
        return F.scaled_dot_product_attention(
            qn, impl.expand(kn, q.shape[1]), impl.expand(v, q.shape[1]),
            is_causal=causal, scale=scale,
        )
    return run


def make_arms(xs, causal, scale, bn=128):
    q, k, v, gq, gk, cos, sin = xs
    candidate = lambda: impl.optimized_attention(
        q, k, v, gq, gk, cos, sin, causal=causal, mode="query", bn=bn,
        gather=False, norm_style="hf",
    )
    unfused = lambda: impl.optimized_attention(
        q, k, v, gq, gk, cos, sin, causal=causal, mode="unfused", bn=bn,
        gather=False, norm_style="hf",
    )
    vendor = vendor_fn(xs, causal, scale)
    # Default Inductor avoids a nested cudagraph capture when the outer
    # publication harness captures each arm for paired replay.
    compiled = torch.compile(vendor, fullgraph=True, dynamic=False,
                             mode="default")
    return {"candidate": candidate, "unfused": unfused,
            "compiled": compiled, "vendor": vendor}


def check(fn, ref):
    return impl.metric(fn(), ref)


def timed_pair(base, candidate, seed):
    initial = impl.paired(base, candidate, rounds=3, samples=10, seed=seed)
    retest = impl.paired(base, candidate, rounds=5, samples=14, seed=seed + 100000)
    return {"initial": initial, "retest": retest,
            "baseline_p50_ms": statistics.median(retest["a_p50_ms"]),
            "candidate_p50_ms": statistics.median(retest["b_p50_ms"]),
            "speedup": statistics.median(retest["a_p50_ms"]) /
                       statistics.median(retest["b_p50_ms"])}


def one_row(model, b, s, causal, args, model_no):
    cfg = impl.MODELS[model]
    scale = impl.model_scale(model, cfg[2])
    seed = 1701 + model_no * 10000 + b * 101 + s * 7 + int(causal)
    xs = impl.make_inputs(cfg, b, s, seed)
    ref = impl.oracle_fp32(*xs, causal, scale=scale)
    arms = make_arms(xs, causal, scale, bn=args.bn)
    row = {"model": model, "B": b, "S": s, "causal": causal, "seed": seed,
           "status": "running", "correctness": {}, "timing": {},
           "traces": [], "gpu_exclusivity": {"before": impl.gpu_snapshot()}}
    monitor = impl.ExclusiveMonitor()
    monitor.start()
    try:
        # Compile and warm the compiled reference before correctness/timing.
        for name, fn in arms.items():
            row["correctness"][name] = check(fn, ref)
        row["correctness_pass"] = all(x.get("pass") for x in row["correctness"].values())
        if not row["correctness_pass"]:
            row["status"] = "tested_invalid"
            row["gate_reasons"] = ["one or more arms failed true-FP32 correctness"]
            return row
        # TorchInductor may finish a lazy specialization on the first few
        # calls even after the numerical check.  Complete that work before a
        # CUDA-graph capture; compiling during capture invalidates the row.
        for _ in range(5):
            arms["compiled"]()
        torch.cuda.synchronize()
        for name in ("unfused", "compiled", "vendor"):
            row["timing"][name] = timed_pair(arms[name], arms["candidate"],
                                               seed + 1000 + len(row["timing"]))
        row["fastest_baseline"] = min(
            row["timing"], key=lambda n: row["timing"][n]["candidate_p50_ms"] *
            row["timing"][n]["speedup"]
        )
        for name, fn in arms.items():
            path = (Path(args.trace_dir) / model /
                    f"B{b}_S{s}_C{int(causal)}_{name}.chrome_trace.json.gz")
            row["traces"].append(impl.capture_trace(fn, path,
                                                     f"e1_refined.{name}"))
    except Exception as exc:
        row["status"] = "tested_invalid"
        row["gate_reasons"] = [repr(exc)]
        return row
    finally:
        row["gpu_exclusivity"]["observations"] = monitor.finish()
        row["gpu_exclusivity"]["after"] = impl.gpu_snapshot()
    if not row["gpu_exclusivity"]["before"].get("exclusive") or \
       not row["gpu_exclusivity"]["after"].get("exclusive") or \
       not row["gpu_exclusivity"]["observations"] or \
       not all(x.get("exclusive") for x in row["gpu_exclusivity"]["observations"]):
        row["status"] = "tested_invalid"
        row["gate_reasons"] = ["selected physical GPU was not exclusive"]
    elif len(row["traces"]) != 4:
        row["status"] = "tested_invalid"
        row["gate_reasons"] = ["trace gate failed"]
    else:
        row["status"] = "tested_pass"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(impl.MODELS))
    ap.add_argument("--sizes", default="128,256,384,512")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--causal-modes", default="false,true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--bn", type=int, default=128, choices=(64, 128))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.models, args.sizes, args.batches, args.causal_modes = "Qwen3-8B", "128", "1", "false"
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    sizes = [int(x) for x in args.sizes.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    causal_modes = [x.strip().lower() == "true" for x in args.causal_modes.split(",")]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    Path(args.trace_dir).mkdir(parents=True, exist_ok=True)
    rec = {
        "schema": "fusionagent-e1-rope-refined-census-v1",
        "scope": "isolated Q/K RMSNorm+RoPE; query fusion prepares K once",
        "implementation": "e1_rope_refined_impl.py optimized_attention(mode=query)",
        "baseline_arms": ["unfused", "compiled", "vendor"],
        "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "torch": torch.__version__, "triton": impl.triton.__version__,
                        "cuda": torch.version.cuda},
        "source_hashes": {p.name: sha(p) for p in
                          (HERE / "e1_rope_refined_census.py",
                           HERE / "e1_rope_refined_impl.py",
                           HERE / "fusion_harness.py")},
        "protocol": {"models": models, "S": sizes, "B": batches,
                     "causal": causal_modes, "bn": args.bn,
                     "correctness": "independent true-FP32 oracle before timing",
                     "timing": "randomized paired CUDA-graph initial+fresh retest",
                     "dispatch": "fastest baseline selected per shape",
                     "traces": "candidate, unfused, compiled, vendor"},
        "rows": [], "status": "running", "publication_eligible": False,
    }
    # One process compiles every (model, B, S) specialization.  The default
    # Dynamo cache limit of eight would invalidate the rest of the census.
    torch._dynamo.config.cache_size_limit = 512
    for mi, model in enumerate(models):
        for s in sizes:
            for causal in causal_modes:
                for b in batches:
                    print(f"{model} B={b} S={s} causal={causal}", flush=True)
                    row = one_row(model, b, s, causal, args, mi)
                    rec["rows"].append(row)
                    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({k: row.get(k) for k in
                                      ("model", "B", "S", "causal", "status",
                                       "fastest_baseline", "gate_reasons")},
                                     default=str), flush=True)
    rec["status"] = ("tested_pass" if rec["rows"] and
                      all(r.get("status") == "tested_pass" for r in rec["rows"])
                      else "tested_invalid")
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    out.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "status": rec["status"],
                      "rows": len(rec["rows"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
