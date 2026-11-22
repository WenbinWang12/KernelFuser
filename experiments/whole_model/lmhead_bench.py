#!/usr/bin/env python3
"""Kernel-level study of the two language-model-head fusions in lmhead_kernels.py.

--mode train   LM-head GEMM + mean cross-entropy + backward to the hidden states and the
               head weights (what one training step does at the head).
    eager      F.linear -> logits.float() -> F.cross_entropy, autograd backward
               (the computation of HuggingFace causal-LM losses)
    compiled   torch.compile(default) of that forward; backward through AOTAutograd
    candidate  F.linear (vendor GEMM) -> fused loss+gradient kernel; backward continues
               into the vendor GEMMs.  No Triton GEMM is involved.
    bf16_ce    F.cross_entropy on the BF16 logits (reduced-precision variant; timed for
               information, not part of the verdict)

--mode loss    LM-head GEMM + mean cross-entropy, forward only (scoring, evaluation).
    eager, compiled, bf16_ce as above (no backward)
    same_anchor  Triton GEMM that stores the logits, then F.cross_entropy
    candidate    the same Triton GEMM with the log-sum-exp epilogue (no logits written)

Shapes are (K, V) of released models; the head weights are read from the checkpoint when
it is present under --models-dir, otherwise drawn at random.  Hidden states are random.
Every 512th target is the ignore index, as at the last position of a 512-token sequence.

Correctness is gated before timing against an FP32 oracle; the rule is fixed here.
  train: loss error <= max(2 x eager's error, 5e-4 relative); relative L2 error of each
         gradient <= min(0.05, max(1.5 x eager's error, 1e-3)).
  loss:  loss error <= max(2 x eager's error, 5e-4 relative).
Timing: randomized interleaved pairs of CUDA-graph replays (initial + retest), as in the
attention census.  If an arm cannot be captured in a CUDA graph, all pairs of that row
are timed by CUDA events around direct calls instead, and the row records it.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, random, statistics, sys, time, traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lmhead_kernels as K

WIN = 1.01
SHAPES = {  # model: (hidden K, vocabulary V)
    "Gemma-3-270M": (640, 262144), "Qwen2.5-0.5B-Instruct": (896, 151936), "SmolLM2-360M-Instruct": (960, 49152),
    "Qwen3-0.6B": (1024, 151936), "Qwen3-1.7B": (2048, 151936), "OLMo-1B": (2048, 50304),
    "Qwen3-4B": (2560, 151936), "Gemma-3-4B": (2560, 262208), "Qwen3-8B": (4096, 151936),
    "Llama-3.1-8B": (4096, 128256),
    "SmolLM2-135M-Instruct": (576, 49152), "Gemma-3-1B": (1152, 262144), "Qwen2.5-1.5B-Instruct": (1536, 151936),
    "SmolLM2-1.7B-Instruct": (2048, 49152), "Qwen3-14B": (5120, 151936),
    # added 2026-11-22: heads of the Qwen3.5 / Qwen3.8 checkpoints (hybrid attention, and a 256-expert MoE)
    "Qwen3.5-9B": (4096, 248320), "Qwen3.8-27B": (5120, 248320), "Qwen3.5-35B-A3B": (2048, 248320),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def head_weight(models_dir, name, Kdim, V, device, dtype, gen):
    """Head weights of the checkpoint if available, else random. Returns (W, source, (K, V))."""
    d = Path(models_dir) / name if models_dir else None
    if d and (d / "config.json").exists():
        try:
            from safetensors import safe_open
            best = None
            for f in sorted(d.glob("*.safetensors")):
                with safe_open(str(f), framework="pt", device="cpu") as sf:
                    for key in sf.keys():
                        if key.endswith("lm_head.weight") or key.endswith("embed_tokens.weight"):
                            if best is None or key.endswith("lm_head.weight"):
                                best = (f, key)
            if best is not None:
                with safe_open(str(best[0]), framework="pt", device="cpu") as sf:
                    W = sf.get_tensor(best[1])
                if W.ndim == 2:
                    return W.to(device=device, dtype=dtype).contiguous(), f"checkpoint:{best[1]}", (W.shape[1], W.shape[0])
        except Exception as exc:
            print(f"[warn] {name}: could not read head weights ({exc!r}); using random weights", flush=True)
    W = (torch.randn((V, Kdim), generator=gen, dtype=torch.float32) * Kdim ** -0.5).to(device=device, dtype=dtype)
    return W.contiguous(), "random", (Kdim, V)


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def paired_events(a, b, rounds, samples, seed):
    rng = random.Random(seed)
    out = {"a": [], "b": [], "arm_order": [], "mode": "paired_cuda_events_direct_calls", "paired": True,
           "samples_per_round": samples, "seed": seed}
    for _ in range(rounds):
        for _ in range(3):
            a(); b()
        torch.cuda.synchronize()
        aa, bb, orders = [], [], []
        for _ in range(samples):
            first_a = bool(rng.getrandbits(1)); orders.append("a_first" if first_a else "b_first")
            for fn, vals in (((a, aa), (b, bb)) if first_a else ((b, bb), (a, aa))):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); fn(); e.record(); e.synchronize(); vals.append(float(s.elapsed_time(e)))
        out["a"].append(aa); out["b"].append(bb); out["arm_order"].append(orders)
    out["a_p50_ms"] = [statistics.median(x) for x in out["a"]]
    out["b_p50_ms"] = [statistics.median(x) for x in out["b"]]
    out["ratio_median"] = statistics.median([x / y for x, y in zip(out["a_p50_ms"], out["b_p50_ms"])])
    return out


def timed_pair(pair_fn, base, cand, seed, args):
    initial = pair_fn(base, cand, args.rounds_initial, args.samples, seed)
    retest = pair_fn(base, cand, args.rounds_retest, args.samples, seed + 100000)
    b, c = statistics.median(retest["a_p50_ms"]), statistics.median(retest["b_p50_ms"])
    return {"initial": initial, "retest": retest, "baseline_p50_ms": b, "candidate_p50_ms": c,
            "initial_speedup": statistics.median(initial["a_p50_ms"]) / statistics.median(initial["b_p50_ms"]),
            "speedup": b / c}


def build(mode, h, W, y, inv_n, use_compile=True):
    """Returns (arms, verdict_arm_names). Arms return (loss, gh, gW) in train mode and loss in loss mode."""
    fwd = lambda h_, W_, y_: F.cross_entropy(F.linear(h_, W_).float(), y_, ignore_index=K.IGNORE)
    cf = torch.compile(fwd, fullgraph=True, dynamic=False, mode="default") if use_compile else fwd
    if mode == "train":
        def with_grad(loss_fn):
            def run():
                loss = loss_fn()
                gh, gW = torch.autograd.grad(loss, (h, W))
                return loss, gh, gW
            return run
        arms = {
            "eager": with_grad(lambda: fwd(h, W, y)),
            "compiled": with_grad(lambda: cf(h, W, y)),
            "bf16_ce": with_grad(lambda: F.cross_entropy(F.linear(h, W), y, ignore_index=K.IGNORE)),
            "candidate": with_grad(lambda: K.fused_cross_entropy(F.linear(h, W), y, inv_n)),
        }
        return arms, ("eager", "compiled")
    arms = {
        "eager": lambda: fwd(h, W, y),
        "compiled": lambda: cf(h, W, y),
        "bf16_ce": lambda: F.cross_entropy(F.linear(h, W), y, ignore_index=K.IGNORE),
        "same_anchor": lambda: F.cross_entropy(K.lmhead_logits(h, W, y).float(), y, ignore_index=K.IGNORE),
        "candidate": lambda: K.lmhead_loss(h, W, y, inv_n),
    }
    return arms, ("eager", "compiled", "same_anchor")


def oracle(mode, h, W, y):
    h32 = h.detach().float().requires_grad_(mode == "train")
    W32 = W.detach().float().requires_grad_(mode == "train")
    with torch.enable_grad():
        loss = F.cross_entropy(F.linear(h32, W32), y, ignore_index=K.IGNORE)
        if mode == "train":
            gh, gW = torch.autograd.grad(loss, (h32, W32))
            return float(loss), gh, gW
    return float(loss), None, None


def check(mode, arms, ref):
    ref_loss, ref_gh, ref_gW = ref
    out = {}
    for name, fn in arms.items():
        try:
            r = fn()
            if mode == "train":
                loss, gh, gW = r
                out[name] = {"loss": float(loss), "loss_abs_err": abs(float(loss) - ref_loss),
                             "gh_rel_l2": rel_l2(gh, ref_gh), "gW_rel_l2": rel_l2(gW, ref_gW),
                             "finite": bool(torch.isfinite(gh).all() and torch.isfinite(gW).all())}
            else:
                out[name] = {"loss": float(r), "loss_abs_err": abs(float(r) - ref_loss),
                             "finite": bool(torch.isfinite(r))}
            del r
        except Exception as exc:
            out[name] = {"error": repr(exc), "traceback": traceback.format_exc()[-2500:]}
    g = {"reference_loss": ref_loss}
    e, c = out.get("eager", {}), out.get("candidate", {})
    if "error" in e or "error" in c or not c.get("finite"):
        g["pass"] = False; g["reason"] = "eager or candidate arm failed"
        return out, g
    g["loss_limit"] = max(2 * e["loss_abs_err"], 5e-4 * abs(ref_loss))
    ok = c["loss_abs_err"] <= g["loss_limit"]
    if mode == "train":
        for k in ("gh_rel_l2", "gW_rel_l2"):
            g[k + "_limit"] = min(0.05, max(1.5 * e[k], 1e-3))
            ok = ok and c[k] <= g[k + "_limit"]
    g["pass"] = bool(ok)
    return out, g


def one_row(name, mode, M, args, impl, device, dtype):
    Kdim, V = SHAPES[name]
    seed = 7001 + 13 * M + 7 * Kdim + V % 9973
    row = {"model": name, "mode": mode, "M": M, "seed": seed, "status": "running", "errors": {}}
    monitor = None
    if impl is not None:
        row["gpu_exclusivity"] = {"before": impl.gpu_snapshot()}
        monitor = impl.ExclusiveMonitor(); monitor.start()
    try:
        if device == "cuda":
            torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        W, src, (Kdim, V) = head_weight(args.models_dir, name, Kdim, V, device, dtype, gen)
        row.update({"K": Kdim, "V": V, "weights": src})
        h = torch.randn((M, Kdim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype).contiguous()
        y = torch.randint(0, V, (M,), generator=gen, dtype=torch.long)
        y[511::512] = K.IGNORE
        y = y.to(device)
        inv_n = 1.0 / max(1, int((y != K.IGNORE).sum()))
        if mode == "train":
            h.requires_grad_(True); W.requires_grad_(True)
        ctx = torch.enable_grad() if mode == "train" else torch.no_grad()
        with ctx:
            if device == "cuda" and K.BACKEND == "triton":
                with torch.no_grad():
                    if mode == "train":
                        z = F.linear(h, W)
                        row["config"] = {"ce": list(K.tune_ce(z, y))}
                        del z
                    else:
                        row["config"] = {"gemm_fused": list(K.tune_gemm(h, W, y, False)),
                                         "gemm_store": list(K.tune_gemm(h, W, y, True))}
            arms, verdict = build(mode, h, W, y, inv_n, use_compile=(device == "cuda"))
            ref = oracle(mode, h, W, y)
            row["correctness"], row["gate"] = check(mode, arms, ref)
            del ref
            for n in [n for n, v in row["correctness"].items() if "error" in v]:
                row["errors"][n] = row["correctness"][n]; arms.pop(n, None)
            if not row["gate"]["pass"]:
                row["status"] = "tested_invalid"; row["gate_reasons"] = ["candidate failed the correctness gate"]
                return row
            if device != "cuda":
                row["status"] = "selftest_ok"
                return row
            torch.cuda.empty_cache()
            for _ in range(3):
                for fn in arms.values():
                    fn()
            torch.cuda.synchronize()
            names = [n for n in arms if n != "candidate"]
            for pair_fn, label in ((impl.paired, "cuda_graph"), (paired_events, "cuda_events")):
                row["timing"] = {}; row["timing_mode"] = label
                try:
                    for j, n in enumerate(names):
                        row["timing"][n] = timed_pair(pair_fn, arms[n], arms["candidate"], seed + 1000 * (j + 1), args)
                    aa = pair_fn(arms["eager"], arms["eager"], args.rounds_retest, args.samples, seed + 9000)
                    row["self_comparison_eager"] = statistics.median(aa["a_p50_ms"]) / statistics.median(aa["b_p50_ms"])
                    break
                except Exception as exc:
                    row["errors"]["timing_" + label] = {"error": repr(exc), "traceback": traceback.format_exc()[-2500:]}
                    row["timing"] = {}
                    torch.cuda.synchronize()
            t = {n: row["timing"][n] for n in verdict if n in row["timing"]}
            if len(t) == len(verdict):
                row["fastest_baseline"] = min(t, key=lambda n: t[n]["baseline_p50_ms"])
                row["speedup_vs_fastest"] = min(x["speedup"] for x in t.values())
                row["initial_speedup_vs_fastest"] = min(x["initial_speedup"] for x in t.values())
                row["kept"] = bool(row["speedup_vs_fastest"] > WIN and row["initial_speedup_vs_fastest"] > WIN)
                row["status"] = "tested_pass"
            else:
                row["status"] = "tested_invalid"; row["gate_reasons"] = ["a baseline arm is missing"]
    except Exception as exc:
        row["status"] = "tested_invalid"; row["gate_reasons"] = [repr(exc)]
        row["errors"]["row"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3500:]}
    finally:
        if monitor is not None:
            row["gpu_exclusivity"]["observations"] = monitor.finish()
            row["gpu_exclusivity"]["after"] = impl.gpu_snapshot()
            ex = row["gpu_exclusivity"]
            if row["status"] == "tested_pass" and not (ex["before"].get("exclusive") and ex["after"].get("exclusive")
                                                        and ex["observations"] and all(x.get("exclusive") for x in ex["observations"])):
                row["status"] = "tested_invalid"; row["gate_reasons"] = ["selected physical GPU was not exclusive"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train", "loss"), required=True)
    ap.add_argument("--models", default=",".join(SHAPES))
    ap.add_argument("--tokens", default="1024,4096", help="comma list of M (tokens per step)")
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--rounds-initial", type=int, default=2)
    ap.add_argument("--rounds-retest", type=int, default=3)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--selftest", action="store_true", help="CPU, PyTorch stand-ins for the kernels, tiny shapes")
    args = ap.parse_args()
    if args.selftest:
        K.BACKEND = "torch"
        SHAPES["tiny"] = (32, 1000)
        for mode in ("train", "loss"):
            row = one_row("tiny", mode, 1024, args, None, "cpu", torch.float32)
            print(json.dumps({"selftest": mode, "status": row["status"], "gate": row.get("gate"),
                              "correctness": row.get("correctness"), "errors": row["errors"]}, default=str))
        return
    if args.output is None:
        ap.error("--output is required")
    import e1_rope_refined_impl as impl
    torch._dynamo.config.cache_size_limit = 512
    rec = {"schema": "osdi-lmhead-v1", "mode": args.mode,
           "scope": "isolated LM-head GEMM + cross-entropy" + (" + backward" if args.mode == "train" else ", forward only"),
           "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                           "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                           "triton": impl.triton.__version__, "cuda": torch.version.cuda,
                           "compute_capability": list(torch.cuda.get_device_capability())},
           "source_hashes": {p.name: sha(p) for p in (HERE / "lmhead_bench.py", HERE / "lmhead_kernels.py")},
           "protocol": {"dtype": "bfloat16", "rounds_initial": args.rounds_initial, "rounds_retest": args.rounds_retest,
                        "samples": args.samples, "win_threshold": WIN, "ignore_every": 512},
           "rows": [], "status": "running", "publication_eligible": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in [m for m in args.models.split(",") if m]:
        for M in [int(x) for x in args.tokens.split(",")]:
            print(f"{name} mode={args.mode} M={M}", flush=True)
            row = one_row(name, args.mode, M, args, impl, "cuda", torch.bfloat16)
            rec["rows"].append(row)
            rec["tuning"] = K.TUNE_LOG
            args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
            t = row.get("timing", {})
            print(json.dumps({"model": name, "M": M, "status": row["status"], "gate": row.get("gate", {}).get("pass"),
                              "timing_mode": row.get("timing_mode"),
                              "speedup": {k: round(v["speedup"], 4) for k, v in t.items()},
                              "vs_fastest": row.get("speedup_vs_fastest"), "kept": row.get("kept"),
                              "errors": sorted(row["errors"]), "reasons": row.get("gate_reasons")}), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
