#!/usr/bin/env python3
"""Publication E2 receipt: one Gemma model, two paired CUDA graphs.

The baseline graph is captured before the attention methods are patched.  The
candidate graph is then captured after patching the *same* model.  CUDA graph
replay no longer consults Python dispatch, so the baseline graph remains the
unfused graph while both arms share one set of weights.  This avoids loading
two copies of Gemma-3-27B while preserving paired, interleaved timing.
"""
from __future__ import annotations
import argparse, gc, gzip, hashlib, json, statistics, traceback, os, subprocess, threading, time, platform
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import e2_gemma_forward as impl


DT = impl.DT
ATOL = impl.ATOL
RTOL = impl.RTOL


def gpu_exclusivity_snapshot():
    """Return selected physical GPU compute-app provenance for the receipt."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        physical = [int(x.strip()) for x in visible.split(",") if x.strip()]
    except ValueError:
        physical = []
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=10)
        uuids = {int(x.split(",")[0].strip()): x.split(",")[1].strip()
                 for x in q.stdout.splitlines() if "," in x}
        selected = {uuids.get(i) for i in physical}
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=10)
        own = os.getpid(); apps=[]; foreign=[]
        for line in q.stdout.splitlines():
            if "," not in line: continue
            uid,pid = line.split(",",1); pid=int(pid.strip()); uid=uid.strip()
            if uid in selected:
                apps.append({"gpu_uuid":uid,"pid":pid,"self":pid==own})
                if pid != own: foreign.append({"gpu_uuid":uid,"pid":pid})
        return {"physical_gpus":physical,"pid":own,"apps":apps,"foreign_apps":foreign,
                "exclusive":bool(physical) and not foreign,
                "host":platform.node(),"utc":time.time()}
    except Exception as exc:
        return {"physical_gpus":physical,"pid":os.getpid(),"apps":[],"foreign_apps":[],
                "exclusive":False,"error":repr(exc),"host":platform.node(),"utc":time.time()}


class ExclusiveMonitor:
    def __init__(self):
        self.stop = threading.Event(); self.observations=[]
        self.thread = threading.Thread(target=self._loop, daemon=True)
    def _loop(self):
        while not self.stop.is_set():
            self.observations.append(gpu_exclusivity_snapshot())
            self.stop.wait(1.0)
    def start(self): self.thread.start()
    def finish(self):
        self.stop.set(); self.thread.join(timeout=20)
        return self.observations


def _patch_with_restore(model):
    """Install the E2 candidate and return the original attention methods."""
    import types
    saved = []
    for mod in model.modules():
        if type(mod).__name__ != "Gemma3Attention":
            continue
        old = mod.forward

        def candidate(self, hidden_states, position_embeddings=None,
                      attention_mask=None, past_key_values=None, **kw):
            shape = hidden_states.shape[:-1]
            hs = (*shape, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(hs).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hs).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hs).transpose(1, 2)
            window = self.sliding_window if self.sliding_window else 0
            o = impl.fused_attn(q, k, v, self, position_embeddings, window)
            o = o.transpose(1, 2).reshape(*shape, -1).contiguous()
            return self.o_proj(o), None

        mod.forward = types.MethodType(candidate, mod)
        saved.append((mod, old))
    return saved


def _restore(saved):
    for mod, old in saved:
        mod.forward = old


def _trace(fn, path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.profiler.record_function(label):
            with torch.inference_mode():
                fn()
        prof.step()
    raw = Path(str(path)[:-3])
    prof.export_chrome_trace(str(raw))
    with raw.open("rb") as src, gzip.open(path, "wb") as dst:
        dst.write(src.read())
    raw.unlink(missing_ok=True)


def _stat(xs, warm, reps, rounds):
    return {
        "warmup": warm,
        "repeats": reps,
        "rounds": rounds,
        "cuda_graph": True,
        "interleaved": True,
        "samples_ms": xs,
        "mean_ms": statistics.fmean(xs),
        "median_ms": statistics.median(xs),
        "p95_ms": sorted(xs)[int(.95 * len(xs)) - 1],
        "sem2_ms": 2 * statistics.stdev(xs) / (len(xs) ** .5),
    }


def _capture(compiled, ids, attention_mask, position_ids, cache_position):
    """Warm and capture a graph; return graph and its static logits."""
    fn = lambda: compiled(input_ids=ids, attention_mask=attention_mask,
                          position_ids=position_ids,
                          cache_position=cache_position,
                          last_cache_position=ids.shape[-1], use_cache=False).logits
    # Compilation and allocator warmup happen outside capture.
    with torch.inference_mode():
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = fn()
        # Capture records GPU work; it does not execute the recorded graph.
        # Reading graph outputs before a replay reads uninitialized storage.
        graph.replay()
        torch.cuda.synchronize()
    return graph, static_logits


def _paired(gb, gc, warm, reps, rounds, seed=0):
    # Graphs have already had the required warmups during capture.  Replays are
    # alternated per sample so both arms observe the same allocator/clock state.
    with torch.inference_mode():
        for _ in range(warm):
            gb.replay(); gc.replay()
    torch.cuda.synchronize()
    bs, cs, order = [], [], []
    rng = random.Random(seed)
    for _ in range(rounds):
        for _ in range(reps):
            first_base = rng.random() < 0.5
            order.append("baseline,candidate" if first_base else "candidate,baseline")
            def measure(graph):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # Captured outputs are inference tensors; replay must stay in
                # InferenceMode or PyTorch rejects the graph's in-place writes.
                with torch.inference_mode():
                    a.record(); graph.replay(); b.record(); b.synchronize()
                return float(a.elapsed_time(b))
            if first_base:
                bs.append(measure(gb)); cs.append(measure(gc))
            else:
                cs.append(measure(gc)); bs.append(measure(gb))
    bstat, cstat = _stat(bs, warm, reps, rounds), _stat(cs, warm, reps, rounds)
    bstat.update({"arm_order": order, "random_seed": seed})
    cstat.update({"arm_order": order, "random_seed": seed})
    return bstat, cstat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seqlens", nargs="+", type=int,
                    default=[512, 1024, 2048, 4096])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = out.with_name(out.stem + "_traces")
    rec = {
        "schema": "fusionagent-e2-gemma-forward-v5-replay",
        "model_name": args.model_name,
        "model_path": args.model,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "implementation_hash": hashlib.sha256(Path(impl.__file__).read_bytes()).hexdigest(),
        "scope": "Gemma text decoder prefill; forward-only Q/K RMSNorm+RoPE attention prologue; no vision branch",
        "strict_tolerance": {"atol": ATOL, "rtol": RTOL},
        "protocol": {
            "warmup": args.warmup, "samples": args.repeats, "rounds": args.rounds,
            "cuda_graph": True, "interleaved": True, "paired_same_weight_model": True,
            "compiled_arms": True, "backward": False,
            "consistency": ["logits", "top1", "NLL"],
            "required_arms": ["compiled_unfused", "compiled_fused"],
            "gpu_exclusivity": "selected physical GPU has no foreign compute applications before, during, or after each row",
        },
        "rows": [], "status": "running", "publication_eligible": False,
    }
    model = None
    try:
        torch.cuda.set_device(0)
        # The publication protocol owns the outer CUDA graphs.  Inductor's
        # optional internal cudagraph tree must be disabled, otherwise an
        # internal replay is attempted while our graph is being captured.
        try:
            import torch._inductor.config as _inductor_config
            _inductor_config.triton.cudagraphs = False
            _inductor_config.triton.cudagraph_trees = False
        except Exception:
            pass
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=DT, local_files_only=True,
            low_cpu_mem_usage=True
        ).cuda().eval()
        model.config._attn_implementation = "eager"
        # Materialize the CUDA context before taking the first provenance
        # snapshot.  The selected physical GPU is CUDA_VISIBLE_DEVICES[0].
        torch.cuda.synchronize()
        rec["gpu"] = gpu_exclusivity_snapshot()
        if not rec["gpu"]["exclusive"]:
            raise RuntimeError(f"selected GPU is not exclusive before timing: {rec['gpu']}")
        vocab = int(getattr(getattr(model.config, "text_config", model.config),
                            "vocab_size", getattr(model.config, "vocab_size", 32000)))
        rec["vocab_size"] = vocab
        # Compile the unmodified model before any candidate method is installed.
        # Default Inductor mode still supplies a compiled arm but leaves outer
        # CUDA-graph ownership to this receipt (max-autotune enables an inner
        # cudagraph tree that cannot be nested during capture).
        for S in args.seqlens:
            torch._dynamo.reset()
            base_compiled = torch.compile(model, fullgraph=True, mode="default")
            before_exclusive = gpu_exclusivity_snapshot()
            if not before_exclusive["exclusive"]:
                raise RuntimeError(f"selected GPU is not exclusive before S={S}: {before_exclusive}")
            monitor = ExclusiveMonitor()
            monitor.start()
            ids = torch.arange(S, device="cuda").reshape(1, S).remainder(vocab)
            # A fixed 2-D mask avoids the Gemma implementation's
            # cache_position[-1].item() branch (which is unsupported by
            # fullgraph capture); its causal 4-D mask is then built from
            # static shape tensors during the warmup/capture.
            mask = torch.ones((1, S), dtype=DT, device="cuda")
            pos = torch.arange(S, device="cuda", dtype=torch.long).unsqueeze(0)
            gb, base_logits = _capture(base_compiled, ids, mask, pos,
                                       pos.reshape(-1))
            # A CUDA-graph output is backed by graph-owned storage.  Copy the
            # baseline logits *before* candidate capture: allocator reuse can
            # otherwise make the candidate graph overwrite that storage and
            # manufacture an exact-zero correctness difference.
            with torch.inference_mode():
                base_ref = base_logits.float().clone()
            # Candidate graph is made from the same weights and allocator.  The
            # graph capture freezes the patched dispatch, after which the old
            # baseline graph remains valid even when methods are restored.
            saved = _patch_with_restore(model)
            try:
                # The baseline and candidate share one model object.  Reset
                # Dynamo after installing the candidate so its graph cannot
                # reuse the baseline graph from the prior compile.
                torch._dynamo.reset()
                cand_compiled = torch.compile(model, fullgraph=True, mode="default")
                gcand, cand_logits = _capture(cand_compiled, ids, mask, pos,
                                              pos.reshape(-1))
                with torch.inference_mode():
                    ref = base_ref
                    got = cand_logits.float().clone()
                d = (got - ref).abs()
                ref_nll = float(F.cross_entropy(
                    ref[:, :-1].reshape(-1, ref.shape[-1]),
                    ids[:, 1:].reshape(-1)).cpu())
                got_nll = float(F.cross_entropy(
                    got[:, :-1].reshape(-1, got.shape[-1]),
                    ids[:, 1:].reshape(-1)).cpu())
                preliminary_corr = {
                    "read_after_graph_replay": True,
                    "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                    "baseline_min": float(ref.min()), "baseline_max": float(ref.max()),
                    "candidate_min": float(got.min()), "candidate_max": float(got.max()),
                    "top1": float((got.argmax(-1) == ref.argmax(-1)).float().mean()),
                    "nll_baseline": ref_nll, "nll_candidate": got_nll,
                    "nll_abs": abs(got_nll - ref_nll),
                    "logits_allclose": bool(torch.allclose(got, ref, atol=ATOL, rtol=RTOL)),
                    "finite": bool(torch.isfinite(got).all() and torch.isfinite(ref).all()),
                }
                if not (preliminary_corr["logits_allclose"] and preliminary_corr["finite"]
                        and preliminary_corr["top1"] >= .999 and preliminary_corr["nll_abs"] <= 1e-3):
                    rec["rows"].append({"S": S, "status": "tested_invalid",
                                        "correctness": preliminary_corr,
                                        "reason": "Replayed compiled model outputs fail consistency; timing not run"})
                    print("CORRECTNESS_FAILED", json.dumps(preliminary_corr), flush=True)
                    raise RuntimeError("Replayed model outputs failed pre-timing correctness gate")
                base_t, cand_t = _paired(gb, gcand, args.warmup,
                                         args.repeats, args.rounds, seed=0)
                speedup = base_t["mean_ms"] / cand_t["mean_ms"]
                retest = None
                if speedup > 1.01:
                    rb, rc = _paired(gb, gcand, args.warmup,
                                     args.repeats, args.rounds, seed=1000003)
                    retest = {"required": True, "speedup_initial": speedup,
                              "baseline_forward": rb, "candidate_forward": rc,
                              "speedup_retest": rb["mean_ms"] / rc["mean_ms"]}
                else:
                    retest = {"required": False, "speedup_initial": speedup}
                bp = trace_dir / f"baseline_S{S}.chrome_trace.json.gz"
                cp = trace_dir / f"candidate_S{S}.chrome_trace.json.gz"
                _trace(lambda: gb.replay(), bp, f"{args.model_name}.baseline.S{S}")
                _trace(lambda: gcand.replay(), cp, f"{args.model_name}.candidate.S{S}")
                corr = {
                    **preliminary_corr,
                    "max_abs": float(d.max()), "mean_abs": float(d.mean()),
                    "top1": float((got.argmax(-1) == ref.argmax(-1)).float().mean()),
                    "nll_baseline": ref_nll, "nll_candidate": got_nll,
                    "nll_abs": abs(got_nll - ref_nll),
                    "logits_allclose": bool(torch.allclose(got, ref, atol=ATOL, rtol=RTOL)),
                    "finite": bool(torch.isfinite(got).all()),
                    "forward_pass": bool(torch.allclose(got, ref, atol=ATOL, rtol=RTOL)),
                }
                observations = monitor.finish()
                after_exclusive = gpu_exclusivity_snapshot()
                exclusivity_ok = (
                    before_exclusive.get("exclusive", False)
                    and after_exclusive.get("exclusive", False)
                    and all(x.get("exclusive", False) for x in observations)
                )
                row = {
                    "S": S, "baseline_forward": base_t,
                    "candidate_forward": cand_t, "correctness": corr,
                    "traces": [str(bp), str(cp)], "retest": retest,
                    "gpu_exclusivity": {"before": before_exclusive,
                                        "observations": observations,
                                        "after": after_exclusive,
                                        "passed": exclusivity_ok},
                }
                row["status"] = (
                    "tested_pass" if corr["logits_allclose"] and corr["finite"]
                    and corr["top1"] >= .999 and corr["nll_abs"] <= 1e-3
                    and exclusivity_ok
                    and all(Path(p).is_file() and Path(p).stat().st_size > 0
                            for p in row["traces"])
                    else "tested_invalid"
                )
                rec["rows"].append(row)
                out.write_text(json.dumps(rec, indent=2, default=str) + "\n")
                print("DONE", args.model_name, S, row["status"], flush=True)
                del gcand, cand_compiled
            finally:
                if monitor.thread.is_alive():
                    monitor.finish()
                _restore(saved)
                del gb, base_logits
                gc.collect(); torch.cuda.empty_cache()
        rec["status"] = (
            "tested_pass" if len(rec["rows"]) == len(args.seqlens)
            and all(r["status"] == "tested_pass" for r in rec["rows"])
            else "tested_invalid"
        )
        rec["publication_eligible"] = rec["status"] == "tested_pass"
    except Exception as exc:
        rec.update({"status": "error", "error": repr(exc),
                    "traceback": traceback.format_exc()})
        print(rec["traceback"], flush=True)
    finally:
        out.write_text(json.dumps(rec, indent=2, default=str) + "\n")
        del model
        gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"status": rec["status"], "out": str(out),
                      "error": rec.get("error")}), flush=True)


if __name__ == "__main__":
    main()
