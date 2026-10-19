#!/usr/bin/env python3
"""A/A CUDA-graph noise controls for Qwen3 short-prefill attention shapes.

Both arms execute the same complete separate RMSNorm -> RoPE -> vendor SDPA
path imported from the RoPE isolation runner.  This is an isolated attention
control at the six Qwen3-8B prefill shapes, not a full-model noise estimate.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

import torch
import triton

import e1_qk_publication_v3 as harness
import e1_rope_publication_v1 as rope

MODEL = "Qwen3-8B"
WARMUPS = 20
RETEST_SEED_OFFSET = 1000003
EXPECTED_SHAPES = {(b, s) for b in (1, 8, 32) for s in (128, 256)}


def make_inputs(cfg, batch, seq, seed):
    """Call the corrected RoPE runner while tolerating its pre-API form."""
    try:
        return rope.make_inputs(cfg, batch, seq, seed, norm_style="qwen3")
    except TypeError:
        return rope.make_inputs(cfg, batch, seq, seed)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_check(left, right):
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(left).all().item() and torch.isfinite(right).all().item())
    equal = bool(torch.equal(left, right))
    return {
        "finite": finite,
        "exact_output_equal": equal,
        "max_abs": float((left.float() - right.float()).abs().max().item()) if finite else None,
        "pass": finite and equal,
    }


def capture(fn):
    """Capture one independent graph and execute it before inspecting output."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUPS):
            output = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, output


def event_sample(graph):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    value = float(start.elapsed_time(end))
    if not math.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid CUDA-event sample: {value}")
    return value


def trace_graph(graph, path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        with torch.profiler.record_function(label):
            graph.replay()
        torch.cuda.synchronize()
    raw = path.with_suffix("")
    profiler.export_chrome_trace(str(raw))
    data = raw.read_bytes()
    events = json.loads(data).get("traceEvents", [])
    kernels = sum("kernel" in str(event.get("cat", "")).lower() for event in events)
    with gzip.open(path, "wb") as compressed:
        compressed.write(data)
    raw.unlink()
    if not kernels or path.stat().st_size == 0:
        raise RuntimeError(f"trace contains no CUDA kernels: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "cuda_kernel_events": kernels,
        "label": label,
    }


def measure_pair(inputs_a, inputs_b, reference, rounds, samples, seed, trace_dir, prefix):
    """Fresh A1/A2 captures in every round; randomized order for every pair."""
    fn_a = lambda: rope.vendor(*inputs_a, True, norm_style="qwen3")
    fn_b = lambda: rope.vendor(*inputs_b, True, norm_style="qwen3")
    rng = random.Random(seed)
    round_records = []
    traces = []
    for index in range(rounds):
        ga, oa = capture(fn_a)
        gb, ob = capture(fn_b)
        checks = {
            "A1_vs_A2": exact_check(oa, ob),
            "A1_vs_eager": exact_check(oa, reference),
            "A2_vs_eager": exact_check(ob, reference),
        }
        if not all(check["pass"] for check in checks.values()):
            raise RuntimeError(f"A/A output check failed before timing: {checks}")
        for _ in range(WARMUPS):
            ga.replay()
            gb.replay()
        torch.cuda.synchronize()
        a_times, b_times, orders = [], [], []
        for _ in range(samples):
            if rng.random() < 0.5:
                orders.append("A1_first")
                a_times.append(event_sample(ga))
                b_times.append(event_sample(gb))
            else:
                orders.append("A2_first")
                b_times.append(event_sample(gb))
                a_times.append(event_sample(ga))
        final_check = exact_check(oa, ob)
        if not final_check["pass"]:
            raise RuntimeError(f"A/A output changed during replay: {final_check}")
        deltas = [a - b for a, b in zip(a_times, b_times)]
        round_records.append({
            "round": index,
            "A1": harness.stats(a_times),
            "A2": harness.stats(b_times),
            "arm_order": orders,
            "ratio": statistics.median(a_times) / statistics.median(b_times),
            "paired_delta_mean_ms": statistics.fmean(deltas),
            "paired_delta_ci95_ms": 1.96 * statistics.stdev(deltas) / math.sqrt(samples),
            "correctness_before": checks,
            "correctness_after": final_check,
        })
        if index == 0:
            for arm, graph in (("A1", ga), ("A2", gb)):
                path = trace_dir / f"{prefix}_{arm}.chrome_trace.json.gz"
                traces.append(trace_graph(graph, path, f"e7_rope.{prefix}.{arm}"))
        del ga, gb, oa, ob
        gc.collect()
        torch.cuda.empty_cache()
    a_all = [v for record in round_records for v in record["A1"]["samples_ms"]]
    b_all = [v for record in round_records for v in record["A2"]["samples_ms"]]
    ratios = [record["ratio"] for record in round_records]
    means = [record["paired_delta_mean_ms"] for record in round_records]
    return {
        "mode": "cuda_graph_replay_paired_interleaved",
        "paired": True,
        "arm_order": "randomized_per_pair_sample",
        "fresh_graph_captures_per_arm": rounds,
        "warmups_per_capture": WARMUPS,
        "samples_per_round": samples,
        "seed": seed,
        "rounds": round_records,
        "A1": harness.stats(a_all),
        "A2": harness.stats(b_all),
        "ratio": statistics.fmean(a_all) / statistics.fmean(b_all),
        "ratio_median_round": statistics.median(ratios),
        "ratio_range_round": [min(ratios), max(ratios)],
        "delta_mean_across_rounds_ms": statistics.fmean(means),
        "delta_ci95_across_rounds_ms": (
            1.96 * statistics.stdev(means) / math.sqrt(rounds) if rounds > 1 else None
        ),
    }, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--output", dest="out", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--batches", default="1,8,32")
    parser.add_argument("--sizes", default="128,256")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    batches = [int(x) for x in args.batches.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    if args.rounds < 1 or args.samples < 2 or not all(x > 0 for x in batches + sizes):
        parser.error("positive shapes, at least one round, and at least two samples are required")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir)
    protocol_reasons = []
    if set((b, s) for b in batches for s in sizes) != EXPECTED_SHAPES or len(batches) * len(sizes) != 6:
        protocol_reasons.append("full P0 noise-floor protocol requires B=1,8,32 and S=128,256 exactly once")
    if args.rounds < 5 or args.samples < 100:
        protocol_reasons.append("full P0 noise-floor protocol requires at least 5 rounds and 100 samples per round")
    cfg = rope.MODELS[MODEL]
    rec = {
        "schema": "fusionagent-e7-rope-aa-v1",
        "experiment": "E7",
        "status": "running",
        "publication_eligible": False,
        "scope": "isolated Qwen3-8B norm+RoPE+vendor-SDPA A/A at short-prefill shapes; no full-model A/A claim",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_sha256": sha256(__file__),
        "implementation_sha256": sha256(rope.__file__),
        "harness_sha256": sha256(harness.__file__),
        "environment": {
            "host": platform.node(), "python": platform.python_version(),
            "torch": torch.__version__, "triton": triton.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "visible_gpu": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
        "protocol": {
            "model": MODEL, "B": batches, "S": sizes, "causal": True,
            "dtype": "bfloat16", "normalization": "qwen3",
            "baseline": "e1_rope_publication_v1.vendor: separate norm + RoPE + vendor SDPA",
            "warmup": WARMUPS, "rounds": args.rounds, "samples": args.samples,
            "cuda_graph": True, "randomized_arm_order": "per_pair_sample",
            "independent_retest": "new graph captures per arm and per round; different order seed",
            "retest_seed_offset": RETEST_SEED_OFFSET,
            "correctness": "finite and bitwise-equal A1/A2/eager outputs after graph replay, before timing",
            "gpu_exclusivity": "selected physical GPU before/after and every second during each row",
            "trace_count_per_row": 4,
        },
        "publication_gate_reasons": protocol_reasons,
        "rows": [],
    }
    for b in batches:
        for s in sizes:
            seed = 1701 + b + s + 1
            before = harness.gpu_exclusivity_snapshot()
            monitor = harness.ExclusiveMonitor()
            monitor.start()
            row = {"model": MODEL, "B": b, "S": s, "causal": True,
                   "geometry": {"Hq": cfg[0], "Hkv": cfg[1], "D": cfg[2]},
                   "input_seed": seed, "status": "tested_invalid"}
            reasons = []
            try:
                if not before.get("exclusive"):
                    raise RuntimeError("selected physical GPU is not exclusive before the row")
                with torch.inference_mode():
                    inputs_a = make_inputs(cfg, b, s, seed)
                    inputs_b = tuple(x.detach().clone() for x in inputs_a)
                    reference = rope.vendor(*inputs_a, True, norm_style="qwen3")
                    eager_b = rope.vendor(*inputs_b, True, norm_style="qwen3")
                    row["correctness"] = exact_check(reference, eager_b)
                    if not row["correctness"]["pass"]:
                        raise RuntimeError("baseline eager arms differ or contain non-finite values")
                    prefix = f"{MODEL}_B{b}_S{s}_C1"
                    row["initial"], initial_traces = measure_pair(
                        inputs_a, inputs_b, reference, args.rounds, args.samples,
                        98431 + b + s, trace_dir, prefix + "_initial")
                    row["retest"], retest_traces = measure_pair(
                        inputs_a, inputs_b, reference, args.rounds, args.samples,
                        98431 + b + s + RETEST_SEED_OFFSET, trace_dir, prefix + "_retest")
                    row["trace_manifest"] = initial_traces + retest_traces
                    row["traces"] = [item["path"] for item in row["trace_manifest"]]
                    del inputs_a, inputs_b, reference, eager_b
            except Exception as error:
                row["error"] = repr(error)
                reasons.append("execution, output, or trace validation failed")
            finally:
                observations = monitor.finish()
                after = harness.gpu_exclusivity_snapshot()
                row["gpu_exclusivity"] = {"before": before, "observations": observations, "after": after}
            if not (before.get("exclusive") and after.get("exclusive") and observations
                    and all(x.get("exclusive") for x in observations)):
                reasons.append("selected-GPU exclusivity failed or monitoring observations are missing")
            if len(row.get("trace_manifest", [])) != 4:
                reasons.append("four validated CUDA traces are required")
            if "initial" not in row or "retest" not in row:
                reasons.append("both initial and independently recaptured retest records are required")
            row["gate_reasons"] = reasons
            row["status"] = "tested_pass" if not reasons else "tested_invalid"
            rec["rows"].append(row)
            out.write_text(json.dumps(rec, indent=2) + "\n")
            print(MODEL, f"B={b}", f"S={s}", row["status"],
                  "initial=", row.get("initial", {}).get("ratio"),
                  "retest=", row.get("retest", {}).get("ratio"), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    passed = bool(rec["rows"]) and all(row["status"] == "tested_pass" for row in rec["rows"])
    rec["status"] = "tested_pass" if passed else "tested_invalid"
    rec["publication_eligible"] = passed and not protocol_reasons
    if passed:
        rec["summary"] = {}
        for phase in ("initial", "retest"):
            ratios = [row[phase]["ratio"] for row in rec["rows"]]
            medians = [row[phase]["ratio_median_round"] for row in rec["rows"]]
            rec["summary"][phase] = {
                "row_mean_ratio_range": [min(ratios), max(ratios)],
                "row_median_ratio_range": [min(medians), max(medians)],
                "max_symmetric_deviation": max(max(r, 1 / r) - 1 for r in ratios),
            }
    rec["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({"out": str(out), "status": rec["status"],
                      "publication_eligible": rec["publication_eligible"],
                      "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
