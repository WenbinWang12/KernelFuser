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
    """Serialize profiler sessions and reject captures without GPU events.

    Failed captures stay private. Releasing each profiler before the next
    session avoids overlapping collector teardown in repeated captures.
    Profiler locking and cleanup are outside the benchmark timing region.
    """
    import fcntl
    import gc
    import gzip
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(tempfile.gettempdir()) / f"kernelfuser_gpu_profiler_{os.getuid()}.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(attempts):
            gc.collect()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                with torch.profiler.record_function(label):
                    fn()
                torch.cuda.synchronize()
            raw = path.with_suffix("")
            profile.export_chrome_trace(str(raw))
            del profile
            gc.collect()
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


def fresh_process_census(script, models, *, rope):
    """Run each pre-registered shape in a fresh Python/CUDA process.

    This resets profiler lifecycle state between shapes. The child uses the
    original correctness, warmup, timing and retest protocol without changes.
    """
    import argparse
    import itertools
    import os
    import subprocess
    import sys
    import time
    parser = argparse.ArgumentParser(description="Fresh-process E1 census")
    parser.add_argument("--fresh-processes", action="store_true")
    parser.add_argument("--models", default=",".join(models))
    parser.add_argument("--sizes", default="128,256,384,512" if rope else "128,256,384,512,768,1024,1536,2048")
    parser.add_argument("--causal-modes", default="false,true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40 if rope else 100)
    parser.add_argument("--output", type=Path, required=True)
    if rope:
        parser.add_argument("--batches", default="1,8,32")
        parser.add_argument("--trace-dir", type=Path)
        parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    names = args.models.split(",")
    sizes = [int(value) for value in args.sizes.split(",")]
    batches = [int(value) for value in args.batches.split(",")] if rope else [1]
    masks = args.causal_modes.split(",")
    if set(names) - set(models) or not set(masks) <= {"false", "true"}:
        parser.error("Unknown model or causal mode")
    if min(sizes + batches) < 1 or args.rounds < 1 or args.samples < 2:
        parser.error("Positive dimensions, rounds and at least two samples required")
    if rope and args.smoke:
        names, sizes, batches, masks = ["Qwen3-8B"], [128], [1], ["false"]
        args.rounds, args.samples = 2, 20
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdigit():
        parser.error("Set CUDA_VISIBLE_DEVICES to one physical GPU index")
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = {line.split(",")[0].strip(): line.split(",")[1].strip() for line in inventory.splitlines()}
    if visible not in mapping:
        parser.error("Selected GPU index is unavailable")
    selected_uuid = mapping[visible]
    def wait_idle():
        while True:
            apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
            if not any(line.split(",")[0].strip() == selected_uuid for line in apps.splitlines() if "," in line):
                return
            time.sleep(5)
    script = Path(script).resolve()
    source_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
    support_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    scope = list(itertools.product(names, batches, sizes, masks))
    cases = args.output.parent / (args.output.stem + "_cases")
    cases.mkdir(parents=True, exist_ok=True)
    aggregate = None
    env = os.environ.copy()
    # Preserve package resolution when launched by a packaged Python runner.
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sys.path if path)
    for index, (name, batch, size, mask) in enumerate(scope):
        wait_idle()
        directory = cases / f"{index:04d}_{name}_B{batch}_S{size}_C{mask}"
        directory.mkdir(exist_ok=True)
        receipt = directory / "receipt.json"
        command = [sys.executable, str(script), "--models", name, "--sizes", str(size),
                   "--causal-modes", mask, "--rounds", str(args.rounds), "--samples", str(args.samples),
                   "--output", str(receipt)]
        if rope:
            command += ["--batches", str(batch), "--trace-dir", str(directory / "traces")]
        with (directory / "run.log").open("w") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode or not receipt.is_file():
            raise RuntimeError("Child census failed; inspect its retained private log")
        child = json.loads(receipt.read_text())
        assert child["source_hash"] == source_sha256
        if aggregate is None:
            aggregate = {key: value for key, value in child.items() if key != "models"}
            aggregate.update(models={}, status="running", publication_eligible=False,
                             expected_rows=len(scope), support_source_sha256=support_sha256)
            aggregate["protocol"].update(S=sizes, causal_modes=[value == "true" for value in masks],
                                          collection_mode="fresh_child_process_per_shape", child_startup_in_timing=False)
            if rope:
                aggregate["protocol"]["B"] = batches
        for model, data in child["models"].items():
            aggregate["models"].setdefault(model, {"geometry": data["geometry"], "rows": []})["rows"].extend(data["rows"])
        args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
        print(f"Fresh-process census: {index + 1}/{len(scope)} rows", flush=True)
    rows = [row for model in aggregate["models"].values() for row in model["rows"]]
    assert len(rows) == len(scope)
    aggregate["status"] = "tested_pass" if all(row["status"] == "tested_pass" for row in rows) else "tested_invalid"
    aggregate["publication_eligible"] = aggregate["status"] == "tested_pass" and not (rope and args.smoke)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n")
