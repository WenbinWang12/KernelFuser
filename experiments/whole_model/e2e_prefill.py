#!/usr/bin/env python3
"""Model-level prefill latency of Qwen3 with the fused Q/K-norm + RoPE attention kernel.

What is measured
  One causal prefill of a released Qwen3 checkpoint (embedding, every decoder
  layer, final norm, last-token logits) for B sequences of S tokens in BF16.
  Every arm is captured in a CUDA graph and timed by replay, in randomized
  interleaved pairs, with the timing code of the kernel-level census
  (e1_rope_refined_impl.paired).  The earlier model-level diagnostic compared a
  compiled, graph-captured baseline with an eager, per-call candidate; here all
  arms are graph-captured, so Python and launch overhead are excluded for all.

Arms
  eager      functional re-implementation of the HuggingFace Qwen3 forward:
             cuBLAS GEMMs, PyTorch normalization/pointwise kernels, SDPA.
  compiled   torch.compile(mode="default", fullgraph=True) of that forward for
             the whole model.
  unfused    compiled segments between attention blocks; attention is the
             Triton anchor after a separate Q/K preparation kernel
             (optimized_attention(mode="unfused")).
  candidate  the same compiled segments; attention is the fused kernel
             (optimized_attention(mode="query")): K is prepared once, and Q
             normalization and RoPE run inside the attention kernel.
  The kernels are the checked-in ones of e1_rope_refined_impl.py, unmodified.

Correctness (before any timing)
  All-position logits of every arm are compared with an FP32 eager oracle of
  the same checkpoint and inputs.  The gate is fixed here, before timing:
  an arm passes if its logits are finite, its relative L2 error to the oracle
  is at most REL_L2_ABS and at most REL_L2_FACTOR times the larger error of the
  two PyTorch arms (eager, compiled), and its top-1 agreement with the oracle
  is at least the smaller agreement of the two PyTorch arms minus TOP1_SLACK.
  That is, a Triton arm may not be further from FP32 than the BF16 PyTorch
  paths are, beyond the stated margins.

--cpu-check compares the functional forward with transformers' Qwen3 model on
the CPU in FP32 (no GPU, no Triton launch).
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, statistics, sys, time, traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REL_L2_ABS = 0.05
REL_L2_FACTOR = 1.5
REL_L2_FLOOR = 1.0e-3
TOP1_SLACK = 0.02
WIN = 1.01            # same repeat threshold as the census


# ----------------------------------------------------------------------------- model
class Layer:
    __slots__ = ("ln1", "wq", "wk", "wv", "gq", "gk", "wo", "ln2", "wg", "wu", "wd")


class Qwen3:
    def __init__(self, model_dir: Path, device: str, dtype: torch.dtype):
        from safetensors.torch import load_file
        cfg = json.loads((model_dir / "config.json").read_text())
        if cfg.get("model_type") != "qwen3":
            raise ValueError(f"expected a qwen3 checkpoint, got {cfg.get('model_type')}")
        self.cfg = cfg
        self.hidden = int(cfg["hidden_size"])
        self.Hq = int(cfg["num_attention_heads"])
        self.Hk = int(cfg["num_key_value_heads"])
        self.D = int(cfg.get("head_dim") or self.hidden // self.Hq)
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.theta = float(cfg.get("rope_theta", 10000.0))
        self.vocab = int(cfg["vocab_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        sd = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            sd.update(load_file(str(f), device=device))
        get = lambda k: sd[k].to(dtype).contiguous()
        self.embed = get("model.embed_tokens.weight")
        self.lm_head = get("lm_head.weight") if "lm_head.weight" in sd else self.embed
        self.final_norm = get("model.norm.weight")
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            L = Layer()
            L.ln1 = get(p + "input_layernorm.weight")
            L.wq = get(p + "self_attn.q_proj.weight")
            L.wk = get(p + "self_attn.k_proj.weight")
            L.wv = get(p + "self_attn.v_proj.weight")
            L.gq = get(p + "self_attn.q_norm.weight")
            L.gk = get(p + "self_attn.k_norm.weight")
            L.wo = get(p + "self_attn.o_proj.weight")
            L.ln2 = get(p + "post_attention_layernorm.weight")
            L.wg = get(p + "mlp.gate_proj.weight")
            L.wu = get(p + "mlp.up_proj.weight")
            L.wd = get(p + "mlp.down_proj.weight")
            self.layers.append(L)
        biased = [k for k in sd if k.endswith(".bias")]
        if biased:
            raise ValueError(f"checkpoint has bias terms that this forward ignores: {biased[:3]}")
        del sd

    def tables(self, S: int, device, dtype):
        """Rotary tables as in Qwen3RotaryEmbedding: FP32 angles, cast to the activation dtype."""
        inv = 1.0 / (self.theta ** (torch.arange(0, self.D, 2, device=device, dtype=torch.float32) / self.D))
        ang = torch.arange(S, device=device, dtype=torch.float32)[:, None] * inv[None, :]
        emb = torch.cat((ang, ang), dim=-1)
        return emb.cos().to(dtype).contiguous(), emb.sin().to(dtype).contiguous()


# ----------------------------------------------------------------------------- functional forward
def rms(x, w, eps):
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope(x, cos, sin):          # cos/sin [S, D] broadcast over [B, H, S, D]
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, n):
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def attn_vendor(x, L, cos, sin, Hq, Hk, D, eps):
    B, S, _ = x.shape
    q = rms(F.linear(x, L.wq).view(B, S, Hq, D), L.gq, eps).transpose(1, 2)
    k = rms(F.linear(x, L.wk).view(B, S, Hk, D), L.gk, eps).transpose(1, 2)
    v = F.linear(x, L.wv).view(B, S, Hk, D).transpose(1, 2)
    q, k = rope(q, cos, sin), rope(k, cos, sin)
    n = Hq // Hk
    o = F.scaled_dot_product_attention(q.contiguous(), repeat_kv(k, n).contiguous(),
                                       repeat_kv(v, n).contiguous(), is_causal=True, scale=D ** -0.5)
    return F.linear(o.transpose(1, 2).reshape(B, S, Hq * D), L.wo)


def mlp(x, wg, wu, wd):
    return F.linear(F.silu(F.linear(x, wg)) * F.linear(x, wu), wd)


def forward_vendor(m: Qwen3, ids, cos, sin):
    h = F.embedding(ids, m.embed)
    for L in m.layers:
        h = h + attn_vendor(rms(h, L.ln1, m.eps), L, cos, sin, m.Hq, m.Hk, m.D, m.eps)
        h = h + mlp(rms(h, L.ln2, m.eps), L.wg, L.wu, L.wd)
    hf = rms(h, m.final_norm, m.eps)
    return hf, F.linear(hf[:, -1], m.lm_head)


def forward_oracle(m: Qwen3, ids):
    """FP32 eager oracle; weights are widened one at a time."""
    S = ids.shape[1]
    cos, sin = m.tables(S, ids.device, torch.float32)
    f = lambda t: t.float()
    h = F.embedding(ids, m.embed).float()
    for L in m.layers:
        x = rms(h, f(L.ln1), m.eps)
        B = x.shape[0]
        q = rms(F.linear(x, f(L.wq)).view(B, S, m.Hq, m.D), f(L.gq), m.eps).transpose(1, 2)
        k = rms(F.linear(x, f(L.wk)).view(B, S, m.Hk, m.D), f(L.gk), m.eps).transpose(1, 2)
        v = F.linear(x, f(L.wv)).view(B, S, m.Hk, m.D).transpose(1, 2)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        n = m.Hq // m.Hk
        o = F.scaled_dot_product_attention(q, repeat_kv(k, n), repeat_kv(v, n), is_causal=True, scale=m.D ** -0.5)
        h = h + F.linear(o.transpose(1, 2).reshape(B, S, m.Hq * m.D), f(L.wo))
        h = h + mlp(rms(h, f(L.ln2), m.eps), f(L.wg), f(L.wu), f(L.wd))
    return rms(h, f(m.final_norm), m.eps)


def make_segments(m: Qwen3):
    """Compiled code between two attention blocks; one artifact per segment kind, reused by all layers."""
    Hq, Hk, D, eps = m.Hq, m.Hk, m.D, m.eps

    def qkv(x, wq, wk, wv):
        B, S, _ = x.shape
        q = F.linear(x, wq).view(B, S, Hq, D).transpose(1, 2).contiguous()
        k = F.linear(x, wk).view(B, S, Hk, D).transpose(1, 2).contiguous()
        v = F.linear(x, wv).view(B, S, Hk, D).transpose(1, 2).contiguous()
        return q, k, v

    def post(h, o, wo, ln2, wg, wu, wd):
        B, _, S, _ = o.shape
        h = h + F.linear(o.transpose(1, 2).reshape(B, S, Hq * D), wo)
        return h + mlp(rms(h, ln2, eps), wg, wu, wd)

    def first(ids, embed, ln1, wq, wk, wv):
        h = F.embedding(ids, embed)
        q, k, v = qkv(rms(h, ln1, eps), wq, wk, wv)
        return h, q, k, v

    def mid(h, o, wo, ln2, wg, wu, wd, ln1n, wqn, wkn, wvn):
        h = post(h, o, wo, ln2, wg, wu, wd)
        q, k, v = qkv(rms(h, ln1n, eps), wqn, wkn, wvn)
        return h, q, k, v

    def last(h, o, wo, ln2, wg, wu, wd, fnorm, lm):
        h = post(h, o, wo, ln2, wg, wu, wd)
        hf = rms(h, fnorm, eps)
        return hf, F.linear(hf[:, -1], lm)

    comp = lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, mode="default")
    return comp(first), comp(mid), comp(last)


_HOPPER = None


def attention(impl, q, k, v, gq, gk, cos, sin, mode, bn):
    """The checked-in attention kernels; impl.optimized_attention() only accepts Hopper GPUs."""
    global _HOPPER
    if _HOPPER is None:
        _HOPPER = torch.cuda.get_device_capability(q.device)[0] == 9
    if _HOPPER:
        return impl.optimized_attention(q, k, v, gq, gk, cos, sin, causal=True, mode=mode, bn=bn,
                                        gather=False, norm_style="hf")
    out = torch.empty_like(q)
    kn = torch.empty_like(k)
    qn = torch.empty_like(q) if mode == "unfused" else None
    with torch.cuda.device(q.device):
        impl.launch_attention((q, k, v, gq, gk, cos, sin), out, causal=True, mode=mode, bm=64, bn=bn,
                              gather=False, rope=True, hf_norm=True, qn=qn, kn=kn)
    return out


def forward_segments(m: Qwen3, segs, impl, ids, cos, sin, mode, bn):
    first, mid, last = segs
    Ls = m.layers
    h, q, k, v = first(ids, m.embed, Ls[0].ln1, Ls[0].wq, Ls[0].wk, Ls[0].wv)
    for i, L in enumerate(Ls):
        o = attention(impl, q, k, v, L.gq, L.gk, cos, sin, mode, bn)
        if i + 1 < len(Ls):
            N = Ls[i + 1]
            h, q, k, v = mid(h, o, L.wo, L.ln2, L.wg, L.wu, L.wd, N.ln1, N.wq, N.wk, N.wv)
        else:
            return last(h, o, L.wo, L.ln2, L.wg, L.wu, L.wd, m.final_norm, m.lm_head)


# ----------------------------------------------------------------------------- checks and timing
def logits_metrics(m: Qwen3, hf_arm, oracle_logits, oracle_top1, chunk=256):
    """All-position logits of an arm (its own dtype) against the FP32 oracle logits."""
    x = hf_arm.reshape(-1, hf_arm.shape[-1])
    num = den = 0.0
    agree = 0
    max_abs = 0.0
    finite = True
    for s in range(0, x.shape[0], chunk):
        a = F.linear(x[s:s + chunk], m.lm_head).float()
        b = oracle_logits[s:s + chunk]
        if not bool(torch.isfinite(a).all()):
            finite = False
            break
        d = a - b
        num += float(d.pow(2).sum()); den += float(b.pow(2).sum())
        max_abs = max(max_abs, float(d.abs().max()))
        agree += int((a.argmax(-1) == oracle_top1[s:s + chunk]).sum())
    if not finite:
        return {"finite": False, "relative_l2": float("inf"), "max_abs": float("inf"), "top1_agreement": 0.0}
    return {"finite": True, "relative_l2": (num / max(den, 1e-30)) ** 0.5, "max_abs": max_abs,
            "top1_agreement": agree / x.shape[0]}


def gate(metrics: dict) -> dict:
    ref = [metrics[k] for k in ("eager", "compiled") if k in metrics and metrics[k].get("finite")]
    out = {"thresholds": {"rel_l2_abs": REL_L2_ABS, "rel_l2_factor": REL_L2_FACTOR,
                           "rel_l2_floor": REL_L2_FLOOR, "top1_slack": TOP1_SLACK}}
    if not ref:
        out["pass"] = False; out["reason"] = "no finite PyTorch arm"
        return out
    l2_lim = min(REL_L2_ABS, max(REL_L2_FACTOR * max(r["relative_l2"] for r in ref), REL_L2_FLOOR))
    top_lim = min(r["top1_agreement"] for r in ref) - TOP1_SLACK
    out["limits"] = {"relative_l2": l2_lim, "top1_agreement": top_lim}
    out["arms"] = {}
    for name in ("unfused", "candidate"):
        if name not in metrics:
            out["arms"][name] = False
            continue
        r = metrics[name]
        out["arms"][name] = bool(r["finite"] and r["relative_l2"] <= l2_lim and r["top1_agreement"] >= top_lim)
    out["pass"] = bool(out["arms"].get("candidate"))
    return out


def timed_pair(impl, base, cand, seed, rounds_i, rounds_r, samples):
    initial = impl.paired(base, cand, rounds=rounds_i, samples=samples, seed=seed)
    retest = impl.paired(base, cand, rounds=rounds_r, samples=samples, seed=seed + 100000)
    b, c = statistics.median(retest["a_p50_ms"]), statistics.median(retest["b_p50_ms"])
    return {"initial": initial, "retest": retest, "baseline_p50_ms": b, "candidate_p50_ms": c,
            "initial_speedup": statistics.median(initial["a_p50_ms"]) / statistics.median(initial["b_p50_ms"]),
            "speedup": b / c}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one_row(m: Qwen3, impl, B: int, S: int, args, model_name: str) -> dict:
    seed = 4201 + 101 * B + 7 * S
    row = {"model": model_name, "B": B, "S": S, "causal": True, "seed": seed, "status": "running",
           "errors": {}, "gpu_exclusivity": {"before": impl.gpu_snapshot()}}
    monitor = impl.ExclusiveMonitor(); monitor.start()
    try:
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.randint(0, m.vocab, (B, S), generator=g, dtype=torch.long).cuda()
        cos, sin = m.tables(S, ids.device, torch.bfloat16)

        t0 = time.time()
        arms = {"eager": lambda: forward_vendor(m, ids, cos, sin)}
        compiled = torch.compile(lambda i, c, s: forward_vendor(m, i, c, s), fullgraph=True, dynamic=False, mode="default")
        arms["compiled"] = lambda: compiled(ids, cos, sin)
        segs = make_segments(m)
        arms["unfused"] = lambda: forward_segments(m, segs, impl, ids, cos, sin, "unfused", args.bn)
        arms["candidate"] = lambda: forward_segments(m, segs, impl, ids, cos, sin, "query", args.bn)

        hf32 = forward_oracle(m, ids)
        lm32 = m.lm_head.float()
        oracle_logits = F.linear(hf32.reshape(-1, hf32.shape[-1]), lm32)
        oracle_top1 = oracle_logits.argmax(-1)
        del lm32, hf32
        row["correctness"] = {}
        row["build_seconds"] = {}
        for name in list(arms):
            t1 = time.time()
            try:
                hf, last = arms[name]()
                torch.cuda.synchronize()
                row["correctness"][name] = logits_metrics(m, hf, oracle_logits, oracle_top1)
                del hf, last
            except Exception as exc:
                row["errors"][name] = {"stage": "build_or_first_call", "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
                del arms[name]
            row["build_seconds"][name] = round(time.time() - t1, 2)
        del oracle_logits, oracle_top1
        torch.cuda.empty_cache()
        row["gate"] = gate(row["correctness"])
        if "candidate" not in arms or not row["gate"]["pass"]:
            row["status"] = "tested_invalid"
            row["gate_reasons"] = ["candidate missing or failed the correctness gate"]
            return row

        for _ in range(3):                    # finish lazy specialization before any capture
            for fn in arms.values():
                fn()
        torch.cuda.synchronize()
        row["timing"] = {}
        for j, name in enumerate([n for n in ("eager", "compiled", "unfused") if n in arms]):
            try:
                row["timing"][name] = timed_pair(impl, arms[name], arms["candidate"], seed + 1000 * (j + 1),
                                                 args.rounds_initial, args.rounds_retest, args.samples)
            except Exception as exc:
                row["errors"][name + "_timing"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
        if "compiled" in arms:                 # self-comparison of the strongest PyTorch arm
            try:
                aa = impl.paired(arms["compiled"], arms["compiled"], rounds=args.rounds_retest, samples=args.samples, seed=seed + 9000)
                row["self_comparison_compiled"] = {"ratio": statistics.median(aa["a_p50_ms"]) / statistics.median(aa["b_p50_ms"]), "raw": aa}
            except Exception as exc:
                row["errors"]["self_comparison"] = {"error": repr(exc)}
        row["seconds"] = round(time.time() - t0, 1)
        t = row["timing"]
        if t:
            row["fastest_baseline"] = min(t, key=lambda n: t[n]["baseline_p50_ms"])
            row["speedup_vs_fastest"] = min(x["speedup"] for x in t.values())
            row["initial_speedup_vs_fastest"] = min(x["initial_speedup"] for x in t.values())
            row["kept"] = bool(row["speedup_vs_fastest"] > WIN and row["initial_speedup_vs_fastest"] > WIN)
    except Exception as exc:
        row["status"] = "tested_invalid"
        row["gate_reasons"] = [repr(exc)]
        row["errors"]["row"] = {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]}
        return row
    finally:
        row["gpu_exclusivity"]["observations"] = monitor.finish()
        row["gpu_exclusivity"]["after"] = impl.gpu_snapshot()
    ex = row["gpu_exclusivity"]
    exclusive = bool(ex["before"].get("exclusive") and ex["after"].get("exclusive") and ex["observations"]
                     and all(x.get("exclusive") for x in ex["observations"]))
    complete = all(n in row.get("timing", {}) for n in ("eager", "compiled", "unfused"))
    if not exclusive:
        row["status"] = "tested_invalid"; row["gate_reasons"] = ["selected physical GPU was not exclusive"]
    elif not complete:
        row["status"] = "tested_invalid"; row["gate_reasons"] = ["a baseline arm is missing: " + ",".join(sorted(row["errors"]))]
    else:
        row["status"] = "tested_pass"
    return row


def cpu_check(model_dir: Path):
    torch.manual_seed(0)
    m = Qwen3(model_dir, "cpu", torch.float32)
    ids = torch.randint(0, m.vocab, (2, 24))
    cos, sin = m.tables(24, "cpu", torch.float32)
    hf, last = forward_vendor(m, ids, cos, sin)
    mine = F.linear(hf, m.lm_head)
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32, local_files_only=True).eval()
    with torch.no_grad():
        theirs = ref(input_ids=ids, use_cache=False).logits
    d = (mine - theirs)
    print(json.dumps({"cpu_check": str(model_dir.name), "layers": m.n_layers, "Hq": m.Hq, "Hk": m.Hk, "D": m.D,
                      "rope_theta": m.theta, "max_abs": float(d.abs().max()),
                      "relative_l2": float(d.norm() / theirs.norm()),
                      "top1_agreement": float((mine.argmax(-1) == theirs.argmax(-1)).float().mean()),
                      "last_token_consistent": bool(torch.allclose(last, mine[:, -1], atol=1e-4))}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--shapes", default="1x128,1x512,8x256,32x128", help="comma list of BxS")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--bn", type=int, default=128, choices=(64, 128))
    ap.add_argument("--rounds-initial", type=int, default=2)
    ap.add_argument("--rounds-retest", type=int, default=3)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--cpu-check", action="store_true")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.cpu_check:
        cpu_check(args.model_dir)
        return
    if args.output is None:
        ap.error("--output is required")
    import e1_rope_refined_impl as impl
    torch._dynamo.config.cache_size_limit = 512
    shapes = [tuple(int(x) for x in s.lower().split("x")) for s in args.shapes.split(",") if s]
    rec = {
        "schema": "osdi-e2e-prefill-v1",
        "scope": "whole-model causal prefill, last-token logits; all arms CUDA-graph replays",
        "arms": {"eager": "functional HF-equivalent forward, eager PyTorch + SDPA",
                 "compiled": "torch.compile(default, fullgraph) of the whole forward",
                 "unfused": "compiled segments + Triton anchor after a separate Q/K preparation kernel",
                 "candidate": "compiled segments + fused Q-norm/RoPE attention kernel (K prepared once)"},
        "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                        "triton": impl.triton.__version__, "cuda": torch.version.cuda,
                        "compute_capability": list(torch.cuda.get_device_capability())},
        "source_hashes": {p.name: sha(p) for p in (HERE / "e2e_prefill.py", HERE / "e1_rope_refined_impl.py")},
        "protocol": {"shapes": shapes, "bn": args.bn, "dtype": "bfloat16", "inputs": "seeded random token ids",
                     "timing": "randomized paired CUDA-graph replay, fresh captures per round, initial + retest",
                     "rounds_initial": args.rounds_initial, "rounds_retest": args.rounds_retest,
                     "samples": args.samples, "win_threshold": WIN,
                     "correctness": "all-position logits vs FP32 eager oracle; see module docstring"},
        "model": str(args.model_dir.name), "rows": [], "status": "running", "publication_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rec, indent=1) + "\n")
    t0 = time.time()
    m = Qwen3(args.model_dir, "cuda", torch.bfloat16)
    rec["model_config"] = {"layers": m.n_layers, "hidden": m.hidden, "Hq": m.Hq, "Hk": m.Hk, "D": m.D,
                           "rope_theta": m.theta, "load_seconds": round(time.time() - t0, 1)}
    for (B, S) in shapes:
        print(f"{args.model_dir.name} B={B} S={S}", flush=True)
        row = one_row(m, impl, B, S, args, args.model_dir.name)
        rec["rows"].append(row)
        args.output.write_text(json.dumps(rec, indent=1) + "\n")
        t = row.get("timing", {})
        print(json.dumps({"B": B, "S": S, "status": row["status"], "gate": row.get("gate", {}).get("pass"),
                          "speedup": {k: round(v["speedup"], 4) for k, v in t.items()},
                          "vs_fastest": row.get("speedup_vs_fastest"), "kept": row.get("kept"),
                          "errors": sorted(row["errors"]), "reasons": row.get("gate_reasons")}), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
