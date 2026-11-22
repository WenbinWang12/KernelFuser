"""Small, deterministic execution primitives shared by the public runners.

The harness deliberately owns timing and evidence collection rather than letting
individual fusion definitions choose their own protocol.  Callers still decide
the model and shapes, but invalid timing parameters, non-CUDA execution, path
traversal in evidence labels, and non-finite correctness values fail explicitly.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Callable, Mapping

import torch

DEV, DT = "cuda", torch.bfloat16


class Pattern:
    name = "unnamed"
    backends = ("simb",)
    gate_exempt = ()

    def geometries(self):
        raise NotImplementedError

    def arms(self, geom):
        raise NotImplementedError

    def roofline_ms(self, geom):
        return None

    def label(self, geom):
        return str(geom)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("fusion_harness requires a CUDA device for graph timing")


def capture(fn: Callable[[], object], replays: int = 50, warmups: int = 10):
    """Capture a callable in a private stream after a fixed warm-up sequence."""
    if not callable(fn):
        raise TypeError("fn must be callable")
    _positive_int("replays", replays)
    _positive_int("warmups", warmups)
    _require_cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmups):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(replays):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def paired(functions: Mapping[str, Callable[[], object]], rounds: int = 5,
           samples: int = 10, replays: int = 50, seed: int = 1701):
    """Measure named arms with fresh CUDA graphs and randomized paired order."""
    if not isinstance(functions, Mapping) or not functions:
        raise ValueError("functions must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in functions):
        raise ValueError("every timing arm needs a non-empty string name")
    if any(not callable(fn) for fn in functions.values()):
        raise TypeError("every timing arm must be callable")
    _positive_int("rounds", rounds)
    _positive_int("samples", samples)
    _positive_int("replays", replays)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    _require_cuda()
    rng = random.Random(seed)
    values = {name: [] for name in functions}
    orders = []
    for _ in range(rounds):
        graphs = {name: capture(fn, replays) for name, fn in functions.items()}
        for graph in graphs.values():
            for _ in range(3):
                graph.replay()
        torch.cuda.synchronize()
        for _ in range(samples):
            order = list(functions)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                values[name].append(start.elapsed_time(end) * 1000 / replays)
        del graphs
    result = {
        "arms": {name: {"median_us": statistics.median(times),
                        "p10_us": sorted(times)[max(0, int(len(times) * .1))],
                        "p90_us": sorted(times)[min(len(times)-1, int(len(times) * .9))],
                        "samples_us": times} for name, times in values.items()},
        "protocol": {"rounds": rounds, "samples_per_round": samples,
                     "graph_launches_per_sample": replays, "arm_order": orders,
                     "seed": seed, "fresh_graphs_each_round": True,
                     "cache_policy": "fixed inputs and outputs; steady-state reuse"},
    }
    if any(not times for times in values.values()):
        raise RuntimeError("timing produced no samples for at least one arm")
    return result


def _atomic_write(path: Path, data: str) -> None:
    """Write evidence atomically so an interrupted run cannot leave JSON half-written."""
    tmp = path.with_name(f".{path.name}.tmp-{random.randrange(1 << 30):08x}")
    try:
        tmp.write_text(data, encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _safe_label(label: str) -> str:
    if not isinstance(label, str) or not label:
        raise ValueError("evidence label must be a non-empty string")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    if safe in {"", ".", ".."}:
        raise ValueError(f"invalid evidence label: {label!r}")
    return safe


def compiler_evidence(kernel, directory, label):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    label = _safe_label(label)
    asm = getattr(kernel, "asm", None)
    if not isinstance(asm, Mapping):
        raise TypeError("kernel.asm must be a mapping of compiler artifacts")
    info = {"n_regs": getattr(kernel, "n_regs", None), "n_spills": getattr(kernel, "n_spills", None),
            "metadata": str(getattr(kernel, "metadata", ""))}
    for kind in ("ptx", "ttgir", "llir"):
        code = asm.get(kind)
        if code is not None:
            path = directory / f"{label}.{kind}"
            _atomic_write(path, code)
            info[kind + "_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    ptx = asm.get("ptx", "") or ""
    info["ptx_instruction_lines"] = {
        "wgmma": sum("wgmma.mma" in line for line in ptx.splitlines()),
        "mma_bf16": sum("mma" in line and ".bf16" in line for line in ptx.splitlines()),
        "mma_tf32": sum("mma" in line and ".tf32" in line for line in ptx.splitlines()),
        "barriers": sum("bar." in line or "mbarrier" in line for line in ptx.splitlines()),
    }
    info["maxntid"] = next((line.strip() for line in ptx.splitlines() if ".maxntid" in line or ".reqntid" in line), None)
    _atomic_write(directory / f"{label}.json", json.dumps(info, indent=2) + "\n")
    return info


def metric(candidate, reference, atol=.025, rtol=.025):
    if atol < 0 or rtol < 0 or not math.isfinite(float(atol)) or not math.isfinite(float(rtol)):
        raise ValueError("atol and rtol must be finite non-negative numbers")
    if not isinstance(candidate, torch.Tensor) or not isinstance(reference, torch.Tensor):
        raise TypeError("candidate and reference must be tensors")
    candidate_f, reference_f = candidate.detach().float(), reference.detach().float()
    shape_match = tuple(candidate_f.shape) == tuple(reference_f.shape)
    finite = bool(torch.isfinite(candidate_f).all() and torch.isfinite(reference_f).all())
    if shape_match:
        delta = candidate_f - reference_f
        max_abs = float(delta.abs().max().item()) if delta.numel() else 0.0
        relative_l2 = float((delta.norm() / reference_f.norm().clamp_min(1e-12)).item())
        passed = finite and bool(torch.allclose(candidate_f, reference_f, atol=atol, rtol=rtol))
    else:
        max_abs, relative_l2, passed = float("inf"), float("inf"), False
    return {"pass": passed, "shape_match": shape_match, "finite": finite,
            "max_abs": max_abs, "relative_l2": relative_l2,
            "atol": float(atol), "rtol": float(rtol)}
