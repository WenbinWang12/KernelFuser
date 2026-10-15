"""Shared interface and defaults used by the public E4 pattern definition."""
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


# Shared randomized device timing and compiler evidence.
import hashlib
import json
import random
import statistics
from pathlib import Path

def capture(fn, replays=50):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            fn()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(replays):
                fn()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def paired(functions, rounds=5, samples=10, replays=50, seed=1701):
    """Randomized graph blocks, fresh captures each round, per-launch latency."""
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
    return {
        "arms": {name: {"median_us": statistics.median(times),
                        "p10_us": sorted(times)[max(0, int(len(times) * .1))],
                        "p90_us": sorted(times)[min(len(times)-1, int(len(times) * .9))],
                        "samples_us": times} for name, times in values.items()},
        "protocol": {"rounds": rounds, "samples_per_round": samples,
                     "graph_launches_per_sample": replays, "arm_order": orders,
                     "seed": seed, "fresh_graphs_each_round": True,
                     "cache_policy": "fixed inputs and outputs; steady-state reuse"},
    }


def compiler_evidence(kernel, directory, label):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info = {"n_regs": getattr(kernel, "n_regs", None), "n_spills": getattr(kernel, "n_spills", None),
            "metadata": str(kernel.metadata)}
    for kind in ("ptx", "ttgir", "llir"):
        code = kernel.asm.get(kind)
        if code is not None:
            path = directory / f"{label}.{kind}"
            path.write_text(code)
            info[kind + "_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    ptx = kernel.asm.get("ptx", "")
    info["ptx_instruction_lines"] = {
        "wgmma": sum("wgmma.mma" in line for line in ptx.splitlines()),
        "mma_bf16": sum("mma" in line and ".bf16" in line for line in ptx.splitlines()),
        "mma_tf32": sum("mma" in line and ".tf32" in line for line in ptx.splitlines()),
        "barriers": sum("bar." in line or "mbarrier" in line for line in ptx.splitlines()),
    }
    info["maxntid"] = next((line.strip() for line in ptx.splitlines() if ".maxntid" in line or ".reqntid" in line), None)
    (directory / f"{label}.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def metric(candidate, reference, atol=.025, rtol=.025):
    delta = candidate.float() - reference.float()
    return {"pass": bool(torch.allclose(candidate.float(), reference.float(), atol=atol, rtol=rtol)),
            "max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / reference.float().norm().clamp_min(1e-12)).item()}


def capture_gpu_trace(fn, path, label, attempts=3):
    """Capture GPU events, retaining failed captures for private audit.

    A nonempty Chrome file can contain CPU events only. Retry profiler setup
    without timing the operation again, and reject an arm with no GPU kernels.
    """
    import gzip
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(attempts):
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function(label):
                fn()
            torch.cuda.synchronize()
        raw = path.with_suffix("")
        profile.export_chrome_trace(str(raw))
        payload = raw.read_bytes()
        events = json.loads(payload).get("traceEvents", [])
        kernels = [event for event in events if event.get("cat") == "kernel"]
        target = path if kernels else path.with_name(path.name + f".failed{attempt}.gz")
        with gzip.open(target, "wb") as handle:
            handle.write(payload)
        raw.unlink()
        if kernels:
            return str(path)
    raise RuntimeError("Profiler captured no GPU kernel events after retries")
