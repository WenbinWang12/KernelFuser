#!/usr/bin/env python3
"""MLP blocks of a released decoder at decode shapes (written 2026-11-21).

Site.  In token-by-token decoding every layer's MLP block sees M rows, one per sequence:
    n   = RMSNorm(x) * g                       x  [M, K]
    h   = act(n @ Wg^T) * (n @ Wu^T)           Wg, Wu  [I, K]
    out = x + h @ Wd^T                         Wd  [K, I]
With M small the three GEMMs are bounded by reading the weights, and the norm, the
activation product and the residual add each cost a kernel for a few kilobytes of data.

Arms.  The measured unit is a chain of --layers consecutive MLP blocks of the checkpoint
(the output of one block is the input of the next), so that the weights of a block are not
in the cache when it runs, as in a model.
  candidate  the faster of two sets of fusions, chosen per shape before the measurement by the
             same CUDA-graph replay timing that chooses the tiles (largest set first):
             norm_act_res  two kernels per block.
               A: gate and up GEMMs in one loop over K with the norm as prologue (gamma is
                  applied to the input tile, 1/rms to the two accumulators; the row statistic
                  comes from the tiles the loop reads anyway) and act(gate)*up as epilogue.
               B: down GEMM with the residual add as epilogue.
             act_res       the norm stays a separate (torch.compile'd) kernel; A keeps the
                           activation epilogue and B the residual epilogue.
             Version 4 = version 3 with six more tile configurations (see CONFIGS).
             Version 2 of this script had only norm_act_res; decode_mlp_diag.py showed that the
             norm prologue costs more inside the GEMM loop than the kernel it removes.
  unfused    the same two GEMM kernels with prologue and epilogue switched off (A stores
             gate and up, B stores the product); norm, activation product and add are
             separate kernels, each one torch.compile'd.
  compiled   torch.compile(default) of the PyTorch chain.
  eager      the PyTorch chain (HuggingFace computation).

Correctness is gated before timing against an FP32 oracle of the chain; the rule is fixed
here, before any run: with e(arm) the relative L2 error of the chain output,
    e(candidate) <= min(0.05, max(1.5 x e(eager), 1e-3)),
and the same rule for the change of the residual stream (output minus input).
Timing: randomized interleaved pairs of CUDA-graph replays, initial + retest; a row is kept
only if the candidate is more than 1% faster than each of eager, compiled and unfused in both.
Tile configurations are chosen per shape, for the candidate and for the unfused arm separately,
from an explicit list by the CUDA-graph replay time of the chain before the measurement; the
list and the timings are recorded.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, re, statistics, sys, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

WIN = 1.01
REL_ABS, FACTOR, FLOOR = 0.05, 1.5, 1e-3
# (BLOCK_N, BLOCK_K, warps, stages)
CONFIGS = [(16, 256, 4, 2), (16, 512, 4, 2), (32, 128, 4, 3), (32, 256, 4, 2), (32, 256, 4, 3), (32, 512, 4, 2),
           (64, 64, 4, 3), (64, 128, 4, 3), (64, 128, 4, 4), (64, 256, 4, 2), (64, 256, 8, 2),
           (128, 64, 8, 3), (128, 128, 8, 3),
           # added in version 4: the schedules that TorchInductor max-autotune picks for these GEMMs
           # on PyTorch 2.11 (two warps, five pipeline stages), and neighbours of them
           (32, 128, 2, 5), (16, 128, 2, 5), (32, 64, 2, 5), (64, 128, 4, 5), (32, 128, 4, 5), (32, 128, 2, 3)]
DEFAULT = (64, 128, 4, 3)
TUNE_LOG: dict = {}
_BEST: dict = {}


# ----------------------------------------------------------------------------- kernels
@triton.jit
def _gate_up(x_ptr, g_ptr, wg_ptr, wu_ptr, o1_ptr, o2_ptr, M, I, K, eps,
             NORM: tl.constexpr, ACT: tl.constexpr, GELU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_K: tl.constexpr):
    """gate = x @ Wg^T, up = x @ Wu^T.  NORM: x is the raw input and its RMSNorm runs as prologue.
    ACT: o1 = act(gate) * up.  Otherwise o1 = gate, o2 = up."""
    om = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = om < M
    mn = on < I
    accg = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    accu = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        ok = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = ok < K
        x = tl.load(x_ptr + om[:, None] * K + ok[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        if NORM:
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
            g = tl.load(g_ptr + ok, mask=mk, other=0.0)
            x = (xf * g[None, :]).to(tl.bfloat16)
        wg = tl.load(wg_ptr + on[:, None] * K + ok[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        wu = tl.load(wu_ptr + on[:, None] * K + ok[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        accg = tl.dot(x, tl.trans(wg), accg)
        accu = tl.dot(x, tl.trans(wu), accu)
    mo = mm[:, None] & mn[None, :]
    if NORM:
        r = tl.rsqrt(ss / K + eps)
        gate = accg * r[:, None]
        up = accu * r[:, None]
    else:
        gate = accg
        up = accu
    if ACT:
        if GELU:      # tanh approximation: 0.5 g (1 + tanh(u)) = g * sigmoid(2u), u = sqrt(2/pi) (g + 0.044715 g^3)
            a = gate * tl.sigmoid(1.5957691216057308 * (gate + 0.044715 * gate * gate * gate))
        else:
            a = gate * tl.sigmoid(gate)
        tl.store(o1_ptr + om[:, None] * I + on[None, :], (a * up).to(tl.bfloat16), mask=mo)
    else:
        tl.store(o1_ptr + om[:, None] * I + on[None, :], gate.to(tl.bfloat16), mask=mo)
        tl.store(o2_ptr + om[:, None] * I + on[None, :], up.to(tl.bfloat16), mask=mo)


@triton.jit
def _down(h_ptr, w_ptr, x_ptr, o_ptr, M, N, I, RES: tl.constexpr, BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """o = h @ Wd^T (+ x if RES)."""
    om = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    on = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mm = om < M
    mn = on < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(I, BLOCK_K)):
        ok = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = ok < I
        h = tl.load(h_ptr + om[:, None] * I + ok[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        w = tl.load(w_ptr + on[:, None] * I + ok[None, :], mask=mn[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(h, tl.trans(w), acc)
    mo = mm[:, None] & mn[None, :]
    if RES:
        acc += tl.load(x_ptr + om[:, None] * N + on[None, :], mask=mo, other=0.0).to(tl.float32)
    tl.store(o_ptr + om[:, None] * N + on[None, :], acc.to(tl.bfloat16), mask=mo)


def block_m(M):
    return 16 if M <= 16 else (32 if M <= 32 else 64)


def gate_up(x, g32, Wg, Wu, cfg, fused, gelu, eps, norm=None):
    """fused: activation epilogue on.  norm: RMSNorm prologue on (default: same as fused)."""
    norm = fused if norm is None else norm
    M, K = x.shape
    I = Wg.shape[0]
    bm = block_m(M)
    bn, bk, warps, stages = cfg
    o1 = torch.empty((M, I), device=x.device, dtype=x.dtype)
    o2 = o1 if fused else torch.empty_like(o1)
    _gate_up[(triton.cdiv(I, bn), triton.cdiv(M, bm))](x, g32, Wg, Wu, o1, o2, M, I, K, eps, NORM=norm, ACT=fused, GELU=gelu,
                                                       BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=warps,
                                                       num_stages=stages)
    return o1, o2


def down(h, Wd, x, cfg, res):
    M, I = h.shape
    N = Wd.shape[0]
    bm = block_m(M)
    bn, bk, warps, stages = cfg
    o = torch.empty((M, N), device=h.device, dtype=h.dtype)
    _down[(triton.cdiv(N, bn), triton.cdiv(M, bm))](h, Wd, x, o, M, N, I, RES=res, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                                    num_warps=warps, num_stages=stages)
    return o


def graph_time(fn, impl, reps=30):
    """Median replay time (ms) of a CUDA graph of fn."""
    g, out = impl.capture(fn, warmups=3)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    vals = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        vals.append(float(a.elapsed_time(b)))
    del g, out
    return statistics.median(vals)


def tune_chain(make, key, impl):
    """Configurations (gate/up kernel, down kernel) of one arm for one shape.

    make(cfg_gate_up, cfg_down) returns the chain callable.  Each kernel's configuration is
    chosen in turn by the replay time of the whole chain in a CUDA graph, the other kernel
    held fixed.  (Version 1 of this script timed single direct calls; at these shapes a
    direct call costs about 50 us of launch overhead whatever the configuration, so that
    choice was noise.  results/decode_mlp_smoke.json is the receipt of that version.)
    """
    if key in _BEST:
        return _BEST[key]
    best, log = [DEFAULT, DEFAULT], {}
    for pos, label in ((0, "gate_up"), (1, "down")):
        res = {}
        for cfg in CONFIGS:
            trial = list(best); trial[pos] = cfg
            try:
                res[cfg] = graph_time(make(*trial), impl)
            except Exception as exc:                   # a configuration that does not compile or launch is skipped
                res[cfg] = repr(exc)[:160]
                torch.cuda.synchronize()
        ok = {c: t for c, t in res.items() if isinstance(t, float)}
        if not ok:
            raise RuntimeError(f"no configuration ran for {key} {label}: {res}")
        best[pos] = min(ok, key=ok.get)
        log[label] = {str(c): t for c, t in res.items()}
    _BEST[key] = tuple(best)
    TUNE_LOG[str(key)] = log
    return _BEST[key]


# ----------------------------------------------------------------------------- model data
def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def model_info(d: Path):
    cfg = json.loads((d / "config.json").read_text())
    t = dict(cfg.get("text_config") or {})
    get = lambda k, dv=None: t.get(k, cfg.get(k, dv))
    mtype = str(get("model_type", cfg.get("model_type", "")))
    gemma = "gemma" in mtype
    act = get("hidden_activation") or get("hidden_act") or ("gelu_pytorch_tanh" if gemma else "silu")
    if act not in ("silu", "gelu_pytorch_tanh"):
        raise ValueError(f"unsupported activation {act}")
    if gemma:
        # Gemma-3 normalizes the MLP output before the residual add, so the add is not an
        # epilogue of the down GEMM there.  This script covers the Llama-style block only.
        raise ValueError("Gemma-3 block (norm after the MLP) is outside the scope of this script")
    return {"model_type": mtype, "family": "gemma" if gemma else "llama", "act": act,
            "eps": float(get("rms_norm_eps", 1e-6)), "num_layers": get("num_hidden_layers")}


def load_layers(d: Path, first, count, device):
    """MLP weights and the norm weight in front of the MLP for layers first..first+count-1 (BF16)."""
    from safetensors import safe_open
    index = {}
    for f in sorted(d.glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for k in sf.keys():
                index[k] = f
    pat = re.compile(r"(?:^|\.)layers\.(\d+)\.(mlp\.gate_proj|mlp\.up_proj|mlp\.down_proj|post_attention_layernorm|pre_feedforward_layernorm)\.weight$")
    found = {}
    for k in index:
        if "vision" in k:
            continue
        m = pat.search(k)
        if m:
            found[(int(m.group(1)), m.group(2).split(".")[-1])] = k
    n_layers = 1 + max(i for i, _ in found)
    first = min(first, max(0, n_layers - count))
    out, keys = [], []
    for i in range(first, min(n_layers, first + count)):
        norm = found.get((i, "post_attention_layernorm")) or found.get((i, "pre_feedforward_layernorm"))
        names = [norm, found[(i, "gate_proj")], found[(i, "up_proj")], found[(i, "down_proj")]]
        ts = []
        for k in names:
            with safe_open(str(index[k]), framework="pt", device="cpu") as sf:
                ts.append(sf.get_tensor(k).to(device=device, dtype=torch.bfloat16).contiguous())
        out.append(tuple(ts)); keys.append(names)
    return out, keys, n_layers


# ----------------------------------------------------------------------------- arms
def make_torch_block(family, act, eps):
    actf = F.silu if act == "silu" else (lambda z: F.gelu(z, approximate="tanh"))
    if family == "gemma":
        def norm(x, w):
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            return (xf * (1.0 + w.float())).to(x.dtype)
    else:
        def norm(x, w):
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            return w * xf.to(x.dtype)

    def chain(x, layers):
        for w, Wg, Wu, Wd in layers:
            n = norm(x, w)
            x = x + F.linear(actf(F.linear(n, Wg)) * F.linear(n, Wu), Wd)
        return x
    return norm, actf, chain


def oracle(x, layers, family, act, eps):
    actf = F.silu if act == "silu" else (lambda z: F.gelu(z, approximate="tanh"))
    y = x.float()
    for w, Wg, Wu, Wd in layers:
        g = 1.0 + w.float() if family == "gemma" else w.float()
        n = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps) * g
        y = y + F.linear(actf(F.linear(n, Wg.float())) * F.linear(n, Wu.float()), Wd.float())
    return y


def rel_l2(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def one_row(name, d, M, args, impl):
    info = model_info(d)
    seed = 9100 + 17 * M + sum(map(ord, name))
    row = {"model": name, "M": M, "seed": seed, "status": "running", "errors": {}, **info,
           "gpu_exclusivity": {"before": impl.gpu_snapshot()}}
    monitor = impl.ExclusiveMonitor(); monitor.start()
    try:
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        layers, keys, n_layers = load_layers(d, args.first_layer, args.layers, "cuda")
        K, I = layers[0][1].shape[1], layers[0][1].shape[0]
        fam, act, eps = info["family"], info["act"], info["eps"]
        gelu = act != "silu"
        row.update({"K": K, "I": I, "layers": len(layers), "model_layers": n_layers, "first_key": keys[0][1]})
        gen = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn((M, K), generator=gen, dtype=torch.float32).to(device="cuda", dtype=torch.bfloat16).contiguous()
        g32 = [((1.0 + w.float()) if fam == "gemma" else w.float()).contiguous() for w, *_ in layers]
        norm, actf, chain = make_torch_block(fam, act, eps)
        with torch.no_grad():
            def make_candidate(cA, cB, norm_fused=True):
                def run():
                    y = x
                    for (w, Wg, Wu, Wd), g in zip(layers, g32):
                        xin = y if norm_fused else norm_c(y, w)
                        h, _ = gate_up(xin, g, Wg, Wu, cA, True, gelu, eps, norm=norm_fused)
                        y = down(h, Wd, y, cB, True)
                    return y
                return run

            norm_c = torch.compile(norm, fullgraph=True, dynamic=False)
            act_c = torch.compile(lambda a, b: actf(a) * b, fullgraph=True, dynamic=False)
            add_c = torch.compile(lambda a, b: a + b, fullgraph=True, dynamic=False)

            def make_unfused(cA, cB):
                def run():
                    y = x
                    for (w, Wg, Wu, Wd), g in zip(layers, g32):
                        n = norm_c(y, w)
                        a, b = gate_up(n, g, Wg, Wu, cA, False, gelu, eps)
                        y = add_c(y, down(act_c(a, b), Wd, y, cB, False))
                    return y
                return run

            warm = make_unfused(DEFAULT, DEFAULT)
            for _ in range(5):                         # finish the lazy compilation of the three small kernels
                warm()
            torch.cuda.synchronize()
            shape = (M, K, I, len(layers))
            variants = {}
            for vname, nf in (("norm_act_res", True), ("act_res", False)):
                va, vb = tune_chain(lambda a, b, nf=nf: make_candidate(a, b, nf), ("candidate", vname) + shape, impl)
                variants[vname] = {"gate_up": list(va), "down": list(vb), "norm_fused": nf,
                                   "chain_ms": graph_time(make_candidate(va, vb, nf), impl, reps=60)}
            best = min(variants, key=lambda v: variants[v]["chain_ms"])
            cA1, cB1 = tuple(variants[best]["gate_up"]), tuple(variants[best]["down"])
            cA0, cB0 = tune_chain(make_unfused, ("unfused",) + shape, impl)
            row["candidate_variants"], row["candidate_variant"] = variants, best
            row["config"] = {"variant": best, "gate_up_fused": list(cA1), "gate_up_plain": list(cA0), "down_fused": list(cB1), "down_plain": list(cB0)}
            candidate, unfused = make_candidate(cA1, cB1, variants[best]["norm_fused"]), make_unfused(cA0, cB0)

            chain_c = torch.compile(chain, fullgraph=True, dynamic=False, mode="default")
            arms = {"eager": lambda: chain(x, layers), "compiled": lambda: chain_c(x, layers),
                    "unfused": unfused, "candidate": candidate}
            ref = oracle(x, layers, fam, act, eps)
            dref = ref - x.float()
            corr = {}
            for n_, fn in arms.items():
                y = fn()
                corr[n_] = {"rel_l2": rel_l2(y, ref), "delta_rel_l2": rel_l2(y.float() - x.float(), dref),
                            "finite": bool(torch.isfinite(y).all())}
            del ref, dref
            row["correctness"] = corr
            lim = {k: min(REL_ABS, max(FACTOR * corr["eager"][k], FLOOR)) for k in ("rel_l2", "delta_rel_l2")}
            row["gate"] = {"limits": lim, "pass": bool(corr["candidate"]["finite"] and
                                                         all(corr["candidate"][k] <= lim[k] for k in lim))}
            if not row["gate"]["pass"]:
                row["status"] = "tested_invalid"; row["gate_reasons"] = ["candidate failed the correctness gate"]
                return row
            for _ in range(5):
                for fn in arms.values():
                    fn()
            torch.cuda.synchronize()
            row["timing"] = {}
            for j, n_ in enumerate(("eager", "compiled", "unfused")):
                ini = impl.paired(arms[n_], arms["candidate"], args.rounds_initial, args.samples, seed + 1000 * (j + 1))
                ret = impl.paired(arms[n_], arms["candidate"], args.rounds_retest, args.samples, seed + 1000 * (j + 1) + 100000)
                b, c = statistics.median(ret["a_p50_ms"]), statistics.median(ret["b_p50_ms"])
                row["timing"][n_] = {"initial": ini, "retest": ret, "baseline_p50_ms": b, "candidate_p50_ms": c,
                                     "initial_speedup": statistics.median(ini["a_p50_ms"]) / statistics.median(ini["b_p50_ms"]),
                                     "speedup": b / c}
            aa = impl.paired(arms["compiled"], arms["compiled"], args.rounds_retest, args.samples, seed + 9000)
            row["self_comparison_compiled"] = statistics.median(aa["a_p50_ms"]) / statistics.median(aa["b_p50_ms"])
            t = row["timing"]
            row["fastest_baseline"] = min(t, key=lambda n: t[n]["baseline_p50_ms"])
            row["speedup_vs_fastest"] = min(v["speedup"] for v in t.values())
            row["initial_speedup_vs_fastest"] = min(v["initial_speedup"] for v in t.values())
            row["kept"] = bool(row["speedup_vs_fastest"] > WIN and row["initial_speedup_vs_fastest"] > WIN)
            row["status"] = "tested_pass"
    except Exception as exc:
        row["status"] = "tested_invalid"; row["gate_reasons"] = [repr(exc)]
        row["errors"]["row"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3500:]}
    finally:
        row["gpu_exclusivity"]["observations"] = monitor.finish()
        row["gpu_exclusivity"]["after"] = impl.gpu_snapshot()
        ex = row["gpu_exclusivity"]
        if row["status"] == "tested_pass" and not (ex["before"].get("exclusive") and ex["after"].get("exclusive")
                                                    and ex["observations"] and all(o.get("exclusive") for o in ex["observations"])):
            row["status"] = "tested_invalid"; row["gate_reasons"] = ["selected physical GPU was not exclusive"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--models", required=True, help="comma list of directory names under --models-dir")
    ap.add_argument("--rows", default="1,2,4,8,16,32,64", help="comma list of M (sequences decoding together)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--first-layer", type=int, default=4)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds-initial", type=int, default=2)
    ap.add_argument("--rounds-retest", type=int, default=3)
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()
    import e1_rope_refined_impl as impl
    torch._dynamo.config.cache_size_limit = 512
    rec = {"schema": "osdi-decode-mlp-v4", "tile_search": "cuda_graph_replay_of_chain",
           "candidate_variants": {"norm_act_res": "norm prologue, activation epilogue, residual epilogue (2 kernels per block)",
                                  "act_res": "separate norm kernel, activation epilogue, residual epilogue (3 kernels per block)"},
           "scope": "chain of consecutive MLP blocks (RMSNorm, gate/up GEMMs, activation product, down GEMM, residual add) at decode shapes",
           "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                           "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                           "triton": triton.__version__, "cuda": torch.version.cuda,
                           "compute_capability": list(torch.cuda.get_device_capability())},
           "source_hashes": {"decode_mlp.py": sha(HERE / "decode_mlp.py")},
           "protocol": {"dtype": "bfloat16", "layers": args.layers, "first_layer": args.first_layer,
                        "rounds_initial": args.rounds_initial, "rounds_retest": args.rounds_retest,
                        "samples": args.samples, "win_threshold": WIN, "configs": [list(c) for c in CONFIGS],
                        "gate": {"abs": REL_ABS, "factor": FACTOR, "floor": FLOOR}},
           "rows": [], "status": "running", "publication_eligible": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in [m for m in args.models.split(",") if m]:
        for M in [int(v) for v in args.rows.split(",")]:
            print(f"{name} M={M}", flush=True)
            row = one_row(name, args.models_dir / name, M, args, impl)
            rec["rows"].append(row); rec["tuning"] = TUNE_LOG
            args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
            t = row.get("timing", {})
            print(json.dumps({"model": name, "M": M, "K": row.get("K"), "I": row.get("I"), "status": row["status"],
                              "gate": row.get("gate", {}).get("pass"),
                              "err": {k: round(v["rel_l2"], 5) for k, v in row.get("correctness", {}).items()},
                              "cand_ms": round(t["compiled"]["candidate_p50_ms"], 4) if "compiled" in t else None,
                              "speedup": {k: round(v["speedup"], 4) for k, v in t.items()},
                              "vs_fastest": row.get("speedup_vs_fastest"), "kept": row.get("kept"),
                              "config": row.get("config"), "reasons": row.get("gate_reasons")}), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
