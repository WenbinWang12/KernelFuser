#!/usr/bin/env python3
"""Whole-model decode step for Llama-style decoders (written 2026-11-21).

Generalization of e2e_decode.py (Qwen3 only) to the model types llama, qwen2 and qwen3:
per-head Q/K RMSNorm is optional (Qwen3 has it), and the Q/K/V projections may carry a bias
(Qwen2.5).  Measured unit, arms, correctness rule, tile search and timing are those of
e2e_decode.py and are imported from it; read its docstring first.  What differs:

  * the checkpoint loader (Decoder) and the functional PyTorch code are written for the three
    model types; --cpu-check compares them with transformers;
  * the stacked Q/K/V kernel (_qkv2) has two more compile-time switches, HEADNORM and BIAS.
    The bias is added inside the GEMM kernel in the unfused arm as well, as cuBLAS-backed
    F.linear does;
  * the head dimension may be 64 (two half-head accumulators of 32 columns);
  * the candidate is the fastest of four sets of fusions (VARIANTS): each of the two RMSNorm
    prologues (in the Q/K/V kernel and in the gate/up kernel) is either fused or left as a separate
    torch.compile'd kernel; all epilogues stay fused.  The set is chosen per shape before the
    measurement, by the CUDA-graph replay time of the tuning layers with the tiles found for the
    largest set (largest set first).  decode_mlp_diag.py is the reason: inside the GEMM loop the
    norm prologue costs more than the small kernel it removes.  e2e_decode2_v1_snapshot.py is the
    version before this change (largest set only); results/e2e_decode2_smoke*.json came from it;
  * anchors (schema v3, after the PyTorch 2.11 runs of e2e_decode.py, where max-autotune beat the
    candidate at batch 1): the tile lists contain the two-warp, five-stage schedules that
    TorchInductor picks there; the attention over the cache may be PyTorch SDPA instead of the
    Triton kernel ("sdpa" is one choice of the search, used by the candidate and the unfused arm
    alike); and with --autotune the final norm + LM head of both Triton arms is the faster of the
    default and the max-autotune compilation.  None of these is a fusion; they keep the two Triton
    arms on anchors as good as the ones the PyTorch arms get.
"""
from __future__ import annotations

import argparse, gc, json, os, platform, statistics, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import e2e_prefill as P
import e2e_decode as D1
import decode_mlp as dm

WIN = D1.WIN
# (BLOCK_K, warps, stages) of the stacked Q/K/V kernel: the list of e2e_decode.py plus two-warp, five-stage schedules
QKV_CONFIGS = D1.QKV_CONFIGS + [(128, 2, 5), (128, 4, 5), (64, 2, 5), (128, 2, 3)]


# ----------------------------------------------------------------------------- checkpoint
class Decoder:
    def __init__(self, model_dir: Path, device: str, dtype: torch.dtype):
        from safetensors.torch import load_file
        cfg = json.loads((model_dir / "config.json").read_text())
        self.model_type = cfg.get("model_type")
        if self.model_type not in ("llama", "qwen2", "qwen3"):
            raise ValueError(f"unsupported model type {self.model_type}")
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError(f"unsupported activation {cfg.get('hidden_act')}")
        self.rope_scaling = cfg.get("rope_scaling") or None
        if self.rope_scaling and self.rope_scaling.get("rope_type", self.rope_scaling.get("type")) != "llama3":
            raise ValueError(f"unsupported rope_scaling {self.rope_scaling}")
        self.hidden = int(cfg["hidden_size"])
        self.Hq = int(cfg["num_attention_heads"])
        self.Hk = int(cfg.get("num_key_value_heads") or self.Hq)
        self.D = int(cfg.get("head_dim") or self.hidden // self.Hq)
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.theta = float(cfg.get("rope_theta", 10000.0))
        self.vocab = int(cfg["vocab_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        sd = {}
        for f in sorted(model_dir.glob("*.safetensors")):
            sd.update(load_file(str(f), device=device))
        get = lambda k: sd[k].to(dtype).contiguous()
        opt = lambda k: get(k) if k in sd else None
        self.embed = get("model.embed_tokens.weight")
        self.lm_head = get("lm_head.weight") if "lm_head.weight" in sd else self.embed
        self.final_norm = get("model.norm.weight")
        self.headnorm = "model.layers.0.self_attn.q_norm.weight" in sd
        self.bias = "model.layers.0.self_attn.q_proj.bias" in sd
        self.layers = []
        used = set()
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            a = p + "self_attn."
            L = SimpleNamespace(
                ln1=get(p + "input_layernorm.weight"), wq=get(a + "q_proj.weight"), wk=get(a + "k_proj.weight"),
                wv=get(a + "v_proj.weight"), bq=opt(a + "q_proj.bias"), bk=opt(a + "k_proj.bias"), bv=opt(a + "v_proj.bias"),
                gq=opt(a + "q_norm.weight"), gk=opt(a + "k_norm.weight"), wo=get(a + "o_proj.weight"),
                ln2=get(p + "post_attention_layernorm.weight"), wg=get(p + "mlp.gate_proj.weight"),
                wu=get(p + "mlp.up_proj.weight"), wd=get(p + "mlp.down_proj.weight"))
            used.update(a + n + ".bias" for n in ("q_proj", "k_proj", "v_proj"))
            self.layers.append(L)
        other = [k for k in sd if k.endswith(".bias") and k not in used]
        if other:
            raise ValueError(f"checkpoint has bias terms that this code ignores: {other[:3]}")
        del sd

    def tables(self, S, device, dtype):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.D, 2, device=device, dtype=torch.float32) / self.D))
        if self.rope_scaling:                       # Llama-3.1 ("llama3"): low frequencies are divided by a factor
            import math
            rs = self.rope_scaling
            factor, lo, hi = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
            old = rs["original_max_position_embeddings"]
            wavelen = 2 * math.pi / inv
            scaled = torch.where(wavelen > old / lo, inv / factor, inv)
            smooth = (old / wavelen - lo) / (hi - lo)
            mid = (1 - smooth) * scaled / factor + smooth * scaled
            inv = torch.where(~(wavelen < old / hi) & ~(wavelen > old / lo), mid, scaled)
        ang = torch.arange(S, device=device, dtype=torch.float32)[:, None] * inv[None, :]
        emb = torch.cat((ang, ang), dim=-1)
        return emb.cos().to(dtype).contiguous(), emb.sin().to(dtype).contiguous()


# ----------------------------------------------------------------------------- functional code (PyTorch arms, oracle)
def forward_full(m, ids, wide, tmax=None):
    """Eager causal forward on [B, S] tokens; returns the final-normed states and, if tmax is given, the caches.

    wide=True is the FP32 oracle: every weight is widened where it is used.
    """
    f = (lambda t: None if t is None else t.float()) if wide else (lambda t: t)
    B, S = ids.shape
    dt = torch.float32 if wide else m.embed.dtype
    cos, sin = m.tables(S, ids.device, dt)
    n = m.Hq // m.Hk
    h = f(F.embedding(ids, m.embed))
    cK, cV = [], []
    for L in m.layers:
        x = P.rms(h, f(L.ln1), m.eps)
        q = F.linear(x, f(L.wq), f(L.bq)).view(B, S, m.Hq, m.D)
        k = F.linear(x, f(L.wk), f(L.bk)).view(B, S, m.Hk, m.D)
        v = F.linear(x, f(L.wv), f(L.bv)).view(B, S, m.Hk, m.D).transpose(1, 2)
        if m.headnorm:
            q, k = P.rms(q, f(L.gq), m.eps), P.rms(k, f(L.gk), m.eps)
        q, k = P.rope(q.transpose(1, 2), cos, sin), P.rope(k.transpose(1, 2), cos, sin)
        if tmax is not None:
            ck = torch.zeros((B, m.Hk, tmax, m.D), device=ids.device, dtype=dt); ck[:, :, :S] = k
            cv = torch.zeros((B, m.Hk, tmax, m.D), device=ids.device, dtype=dt); cv[:, :, :S] = v
            cK.append(ck); cV.append(cv)
        o = F.scaled_dot_product_attention(q.contiguous(), P.repeat_kv(k, n).contiguous(), P.repeat_kv(v, n).contiguous(),
                                           is_causal=True, scale=m.D ** -0.5)
        h = h + F.linear(o.transpose(1, 2).reshape(B, S, m.Hq * m.D), f(L.wo))
        h = h + P.mlp(P.rms(h, f(L.ln2), m.eps), f(L.wg), f(L.wu), f(L.wd))
        del q, k, v, o
    return P.rms(h, f(m.final_norm), m.eps), cK, cV


def step_vendor(m, ids, cK, cV, cosT, sinT, T):
    Hq, Hk, D, eps = m.Hq, m.Hk, m.D, m.eps
    h = F.embedding(ids, m.embed)
    B = h.shape[0]
    for i, L in enumerate(m.layers):
        x = P.rms(h, L.ln1, eps)
        q = F.linear(x, L.wq, L.bq).view(B, Hq, D)
        k = F.linear(x, L.wk, L.bk).view(B, Hk, D)
        if m.headnorm:
            q, k = P.rms(q, L.gq, eps), P.rms(k, L.gk, eps)
        q, k = P.rope(q, cosT, sinT), P.rope(k, cosT, sinT)
        cK[i][:, :, T] = k
        cV[i][:, :, T] = F.linear(x, L.wv, L.bv).view(B, Hk, D)
        o = F.scaled_dot_product_attention(q.view(B, Hk, Hq // Hk, D), cK[i][:, :, :T + 1], cV[i][:, :, :T + 1],
                                           scale=D ** -0.5)
        h = h + F.linear(o.reshape(B, Hq * D), L.wo)
        h = h + P.mlp(P.rms(h, L.ln2, eps), L.wg, L.wu, L.wd)
    return F.linear(P.rms(h, m.final_norm, eps), m.lm_head)


# ----------------------------------------------------------------------------- stacked Q/K/V kernel
@triton.jit
def _qkv2(x_ptr, g_ptr, w_ptr, b_ptr, gq_ptr, gk_ptr, cos_ptr, sin_ptr, q_ptr, ck_ptr, cv_ptr, o_ptr,
          M, K, eps, T, TMAX,
          HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, PRO: tl.constexpr, EPI: tl.constexpr,
          HEADNORM: tl.constexpr, BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """e2e_decode._qkv with an optional bias, an optional per-head norm, and separate switches for the
    prologue (PRO: x is the raw residual stream, RMSNorm inside) and the epilogue (EPI: head norm, RoPE
    and the stores to the query buffer and the cache; otherwise the raw projection goes to o_ptr)."""
    head = tl.program_id(0)
    om = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = om < M
    oh = tl.arange(0, HALF)
    na = head * D + oh
    nb = na + HALF
    acca = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    accb = tl.zeros((BLOCK_M, HALF), dtype=tl.float32)
    ss = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        ok = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
        mk = ok < K
        x = tl.load(x_ptr + om[:, None] * K + ok[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
        if PRO:
            xf = x.to(tl.float32)
            ss += tl.sum(xf * xf, axis=1)
            g = tl.load(g_ptr + ok, mask=mk, other=0.0)
            x = (xf * g[None, :]).to(tl.bfloat16)
        wmask = (oh[:, None] < HALF) & mk[None, :]
        wa = tl.load(w_ptr + na[:, None] * K + ok[None, :], mask=wmask, other=0.0)
        wb = tl.load(w_ptr + nb[:, None] * K + ok[None, :], mask=wmask, other=0.0)
        acca = tl.dot(x, tl.trans(wa), acca)
        accb = tl.dot(x, tl.trans(wb), accb)
    full = mm[:, None] & (oh[None, :] < HALF)
    if PRO:
        r = tl.rsqrt(ss / K + eps)
        a = acca * r[:, None]
        b = accb * r[:, None]
    else:
        a = acca
        b = accb
    if BIAS:
        a += tl.load(b_ptr + na)[None, :]
        b += tl.load(b_ptr + nb)[None, :]
    if EPI:
        is_q = head < HQ
        is_v = head >= HQ + HK
        is_k = (head >= HQ) & (head < HQ + HK)
        if HEADNORM:
            hr = tl.rsqrt((tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / D + eps)
            ga = tl.where(is_q, tl.load(gq_ptr + oh), tl.load(gk_ptr + oh))
            gb = tl.where(is_q, tl.load(gq_ptr + HALF + oh), tl.load(gk_ptr + HALF + oh))
            an = a * hr[:, None] * ga[None, :]
            bn = b * hr[:, None] * gb[None, :]
        else:
            an = a
            bn = b
        ca = tl.load(cos_ptr + oh)
        sa = tl.load(sin_ptr + oh)
        cb = tl.load(cos_ptr + HALF + oh)
        sb = tl.load(sin_ptr + HALF + oh)
        ra = an * ca[None, :] - bn * sa[None, :]
        rb = bn * cb[None, :] + an * sb[None, :]
        oa = tl.where(is_v, a, ra).to(tl.bfloat16)
        ob = tl.where(is_v, b, rb).to(tl.bfloat16)
        qoff = om[:, None] * (HQ * D) + head * D + oh[None, :]
        tl.store(q_ptr + qoff, oa, mask=full & is_q)
        tl.store(q_ptr + qoff + HALF, ob, mask=full & is_q)
        kv = tl.where(is_k, head - HQ, head - HQ - HK)
        coff = om[:, None].to(tl.int64) * (HK * TMAX * D) + (kv * TMAX + T) * D + oh[None, :]
        tl.store(ck_ptr + coff, oa, mask=full & is_k)
        tl.store(ck_ptr + coff + HALF, ob, mask=full & is_k)
        tl.store(cv_ptr + coff, oa, mask=full & is_v)
        tl.store(cv_ptr + coff + HALF, ob, mask=full & is_v)
    else:
        ooff = om[:, None] * ((HQ + 2 * HK) * D) + head * D + oh[None, :]
        tl.store(o_ptr + ooff, a.to(tl.bfloat16), mask=full)
        tl.store(o_ptr + ooff + HALF, b.to(tl.bfloat16), mask=full)


class Kernels:
    def __init__(self, m):
        self.dm, self.m = dm, m
        self.layers = []                            # layouts used by the Triton arms (same values as the checkpoint)
        for L in m.layers:
            g1 = L.ln1.float().contiguous()
            self.layers.append(SimpleNamespace(
                ln1=L.ln1, ln2=L.ln2, wo=L.wo, wg=L.wg, wu=L.wu, wd=L.wd,
                gq=L.gq if m.headnorm else L.ln1, gk=L.gk if m.headnorm else L.ln1,
                wqkv=torch.cat((L.wq, L.wk, L.wv), dim=0).contiguous(),
                bqkv=torch.cat((L.bq, L.bk, L.bv), dim=0).float().contiguous() if m.bias else g1,
                g1=g1, g2=L.ln2.float().contiguous(),
                gq32=L.gq.float().contiguous() if m.headnorm else g1,
                gk32=L.gk.float().contiguous() if m.headnorm else g1))

    def qkv(self, x, L, st, i, cfg, pro, epi):
        m, M = self.m, x.shape[0]
        bm = dm.block_m(M)
        bk, warps, stages = cfg
        if epi:
            out = torch.empty((M, m.Hq * m.D), device=x.device, dtype=x.dtype)
            q, ck, cv, o = out, st.cK[i], st.cV[i], out
        else:
            out = torch.empty((M, (m.Hq + 2 * m.Hk) * m.D), device=x.device, dtype=x.dtype)
            q, ck, cv, o = out, out, out, out
        _qkv2[(m.Hq + 2 * m.Hk, triton.cdiv(M, bm))](
            x, L.g1, L.wqkv, L.bqkv, L.gq32, L.gk32, st.cos32, st.sin32, q, ck, cv, o, M, x.shape[1], m.eps, st.T, st.T + 1,
            HQ=m.Hq, HK=m.Hk, D=m.D, HALF=m.D // 2, PRO=pro, EPI=epi, HEADNORM=m.headnorm, BIAS=m.bias,
            BLOCK_M=bm, BLOCK_K=bk, num_warps=warps, num_stages=stages)
        return out

    def attn(self, q, st, i, cfg):
        m = self.m
        bn, warps, stages = cfg
        o = torch.empty_like(q)
        D1._attn[(m.Hk, q.shape[0])](q, st.cK[i], st.cV[i], o, st.T + 1, st.T + 1, m.D ** -0.5,
                                     HQ=m.Hq, HK=m.Hk, D=m.D, NG=m.Hq // m.Hk, NQ=D1.BQ, BN=bn,
                                     num_warps=warps, num_stages=stages)
        return o


def make_small(m, T):
    Hq, Hk, D, eps, headnorm = m.Hq, m.Hk, m.D, m.eps, m.headnorm
    comp = lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, mode="default")

    def post_qkv(buf, gq, gk, cosT, sinT, ck, cv):
        B = buf.shape[0]
        q = buf[:, :Hq * D].view(B, Hq, D)
        k = buf[:, Hq * D:(Hq + Hk) * D].view(B, Hk, D)
        if headnorm:
            q, k = P.rms(q, gq, eps), P.rms(k, gk, eps)
        q, k = P.rope(q, cosT, sinT), P.rope(k, cosT, sinT)
        ck[:, :, T] = k
        cv[:, :, T] = buf[:, (Hq + Hk) * D:].view(B, Hk, D)
        return q.reshape(B, Hq * D)

    def attn_sdpa(q, ck, cv):
        B = q.shape[0]
        o = F.scaled_dot_product_attention(q.view(B, Hk, Hq // Hk, D), ck, cv, scale=D ** -0.5)
        return o.reshape(B, Hq * D)

    head = lambda h, w, lm: F.linear(P.rms(h, w, eps), lm)
    return {"norm": comp(lambda x, w: P.rms(x, w, eps)), "post_qkv": comp(post_qkv),
            "act": comp(lambda a, b: F.silu(a) * b), "add": comp(lambda a, b: a + b),
            "attn_sdpa": comp(attn_sdpa), "head": comp(head),
            "head_autotune": torch.compile(head, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs")}


VARIANTS = {   # candidate fusion sets: (norm prologue in the Q/K/V kernel, norm prologue in the gate/up kernel)
    "all": (True, True), "qkv_norm_separate": (False, True), "mlp_norm_separate": (True, False),
    "both_norms_separate": (False, False)}


def layers_candidate(kn, st, small, cfg, h, layers, var=(True, True)):
    eps = kn.m.eps
    for i, L in layers:
        q = kn.qkv(h if var[0] else small["norm"](h, L.ln1), L, st, i, cfg["qkv"], var[0], True)
        o = small["attn_sdpa"](q, st.cK[i], st.cV[i]) if cfg["attn"] == "sdpa" else kn.attn(q, st, i, cfg["attn"])
        h = dm.down(o, L.wo, h, cfg["oproj"], True)
        a, _ = dm.gate_up(h if var[1] else small["norm"](h, L.ln2), L.g2, L.wg, L.wu, cfg["gate_up"], True, False, eps,
                          norm=var[1])
        h = dm.down(a, L.wd, h, cfg["down"], True)
    return h


def layers_unfused(kn, st, small, cfg, h, layers):
    eps = kn.m.eps
    for i, L in layers:
        buf = kn.qkv(small["norm"](h, L.ln1), L, st, i, cfg["qkv"], False, False)
        q = small["post_qkv"](buf, L.gq, L.gk, st.cos16, st.sin16, st.cK[i], st.cV[i])
        o = small["attn_sdpa"](q, st.cK[i], st.cV[i]) if cfg["attn"] == "sdpa" else kn.attn(q, st, i, cfg["attn"])
        h = small["add"](h, dm.down(o, L.wo, h, cfg["oproj"], False))
        a, b = dm.gate_up(small["norm"](h, L.ln2), L.g2, L.wg, L.wu, cfg["gate_up"], False, False, eps)
        h = small["add"](h, dm.down(small["act"](a, b), L.wd, h, cfg["down"], False))
    return h


_TEXT = {}


def text_ids(model_dir, data_path):
    """Token ids of a text file under the tokenizer of the checkpoint, with the facts that identify the data."""
    key = (str(model_dir), str(data_path))
    if key not in _TEXT:
        import hashlib
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        raw = Path(data_path).read_bytes()
        ids = tok(raw.decode("utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        _TEXT[key] = (ids, {"file": Path(data_path).name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "tokens": int(ids.numel()), "tokenizer": Path(model_dir).name})
    return _TEXT[key]


def one_row(m, kn, impl, B, T, args, name):
    ablation_mode = getattr(args, "ablation_mode", "full-controller")
    fixed_variant = getattr(args, "fixed_variant", "all")
    if fixed_variant not in VARIANTS:
        raise ValueError(f"unknown fixed variant {fixed_variant}")
    seed = 6301 + 131 * B + 3 * T
    row = {"model": name, "B": B, "T": T, "seed": seed, "status": "running", "errors": {},
           "gpu_exclusivity": {"before": impl.gpu_snapshot()}}
    monitor = impl.ExclusiveMonitor(); monitor.start()
    try:
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        t0 = time.time()
        g = torch.Generator(device="cpu").manual_seed(seed)
        data = getattr(args, "data", None)
        if data:
            # B windows of T + 1 consecutive tokens of the text, at seeded positions: T cached tokens and the new one
            stream, meta = text_ids(args.model_dir, data)
            starts = torch.randint(0, int(stream.numel()) - (T + 1), (B,), generator=g)
            ids = torch.stack([stream[int(a):int(a) + T + 1] for a in starts]).cuda()
            row["inputs"] = dict(meta, kind="text", window_seed=seed)
        else:
            ids = torch.randint(0, m.vocab, (B, T + 1), generator=g, dtype=torch.long).cuda()
            row["inputs"] = {"kind": "seeded random token ids"}
        new = ids[:, T].contiguous()
        st = D1.State()
        st.T = T
        _, st.cK, st.cV = forward_full(m, ids[:, :T], False, T + 1)
        cos, sin = m.tables(T + 1, "cuda", torch.float32)
        st.cos32, st.sin32 = cos[T].contiguous(), sin[T].contiguous()
        st.cos16, st.sin16 = st.cos32.to(torch.bfloat16), st.sin32.to(torch.bfloat16)
        hf32 = forward_full(m, ids, True)[0][:, -1]
        oracle = F.linear(hf32, m.lm_head.float())
        del hf32
        torch.cuda.empty_cache()

        small = make_small(m, T)
        all_layers = list(enumerate(kn.layers))
        sub = all_layers[:args.tune_layers]
        h0 = F.embedding(new, m.embed)
        defaults = {"qkv": (128, 4, 3), "attn": (64, 4, 2), "oproj": dm.DEFAULT, "gate_up": dm.DEFAULT, "down": dm.DEFAULT}
        # "sdpa" = PyTorch scaled-dot-product attention instead of the Triton attention kernel; it is one of
        # the choices of the search, for the candidate and for the unfused arm alike
        space = {"qkv": QKV_CONFIGS, "attn": D1.ATT_CONFIGS + ["sdpa"], "oproj": dm.CONFIGS, "gate_up": dm.CONFIGS, "down": dm.CONFIGS}
        for _ in range(5):
            layers_unfused(kn, st, small, defaults, h0, sub)
            small["attn_sdpa"](torch.zeros((B, m.Hq * m.D), device="cuda", dtype=torch.bfloat16), st.cK[0], st.cV[0])
        torch.cuda.synchronize()
        # Full-controller performs the closed-loop coordinate search and back-off.
        # The two controls use the same model, shapes and harness budget, but do not
        # search or persist controller state.  model-only is a one-shot fused plan,
        # while harness-only sends the unfused path through the immutable gates.
        if ablation_mode == "full-controller":
            cfg_c, log_c = D1.tune(lambda c: (lambda: layers_candidate(kn, st, small, c, h0, sub)), impl, dm, defaults, space)
            vt = {v: dm.graph_time(lambda v=v: layers_candidate(kn, st, small, cfg_c, h0, sub, VARIANTS[v]), impl, reps=60)
                  for v in VARIANTS}
            best = min(vt, key=vt.get)
            var = VARIANTS[best]
        elif ablation_mode == "model-only":
            cfg_c, log_c = dict(defaults), {"policy": "one-shot-default", "trials": 0}
            vt, best, var = {}, fixed_variant, VARIANTS[fixed_variant]
        elif ablation_mode == "harness-only":
            cfg_c, log_c = dict(defaults), {"policy": "fixed-template-unfused", "trials": 0}
            vt, best, var = {}, "harness-only-unfused", (False, False)
        else:
            raise ValueError(f"unknown ablation mode {ablation_mode}")
        row["variants_ms"], row["variant"] = vt, best
        if ablation_mode == "full-controller":
            cfg_u, log_u = D1.tune(lambda c: (lambda: layers_unfused(kn, st, small, c, h0, sub)), impl, dm, defaults, space)
        else:
            cfg_u, log_u = dict(defaults), {"policy": "fixed-template", "trials": 0}
        row["config"] = {"variant": best, "candidate": {k: (v if isinstance(v, str) else list(v)) for k, v in cfg_c.items()},
                         "unfused": {k: (v if isinstance(v, str) else list(v)) for k, v in cfg_u.items()}}
        row["tuning"] = {"candidate": log_c, "unfused": log_u}
        candidate_trials = sum(len(v) for v in log_c.values() if isinstance(v, dict))
        unfused_trials = sum(len(v) for v in log_u.values() if isinstance(v, dict))
        row["controller"] = {
            "ablation_mode": ablation_mode,
            "policy": {"full-controller": "coordinate-search-with-backoff-and-ledger",
                        "model-only": "one-shot-default-fused-plan",
                        "harness-only": "fixed-unfused-template"}[ablation_mode],
            "proposal_count": 1 if ablation_mode != "full-controller" else candidate_trials,
            "tuning_trials": candidate_trials + unfused_trials,
            "iterations": 1 if ablation_mode != "full-controller" else 2,
            "repair_count": 0,
            "backoff_count": int(ablation_mode == "full-controller" and best != "all"),
            "ledger": ablation_mode == "full-controller",
            "language_model_tokens": None,
            "matched_budget": {"tune_layers": args.tune_layers, "rounds_initial": args.rounds_initial,
                               "rounds_retest": args.rounds_retest, "samples": args.samples}}
        row["tune_seconds"] = round(time.time() - t0, 1)

        # final norm + LM head of the two Triton arms: the faster of the default and the max-autotune compilation
        hx = F.embedding(new, m.embed)
        heads = {"head": dm.graph_time(lambda: small["head"](hx, m.final_norm, m.lm_head), impl)}
        if args.autotune:
            try:
                for _ in range(3):
                    small["head_autotune"](hx, m.final_norm, m.lm_head)
                heads["head_autotune"] = dm.graph_time(lambda: small["head_autotune"](hx, m.final_norm, m.lm_head), impl)
            except Exception as exc:
                row["errors"]["head_autotune"] = {"error": repr(exc)[:500]}
        hbest = min(heads, key=heads.get)
        row["head_ms"], row["head_choice"] = heads, hbest
        head = lambda h: small[hbest](h, m.final_norm, m.lm_head)
        arms = {"eager": lambda: step_vendor(m, new, st.cK, st.cV, st.cos16, st.sin16, T)}
        comp = torch.compile(lambda i, c, s: step_vendor(m, i, st.cK, st.cV, c, s, T), fullgraph=True, dynamic=False, mode="default")
        arms["compiled"] = lambda: comp(new, st.cos16, st.sin16)
        if args.autotune:
            auto = torch.compile(lambda i, c, s: step_vendor(m, i, st.cK, st.cV, c, s, T), fullgraph=True, dynamic=False,
                                 mode="max-autotune-no-cudagraphs")
            arms["autotune"] = lambda: auto(new, st.cos16, st.sin16)
        arms["unfused"] = lambda: head(layers_unfused(kn, st, small, cfg_u, F.embedding(new, m.embed), all_layers))
        if ablation_mode == "harness-only":
            arms["candidate"] = arms["unfused"]
        else:
            arms["candidate"] = lambda: head(layers_candidate(kn, st, small, cfg_c, F.embedding(new, m.embed), all_layers, var))

        row["correctness"], row["build_seconds"] = {}, {}
        for n_ in list(arms):
            t1 = time.time()
            try:
                z = arms[n_]()
                torch.cuda.synchronize()
                row["correctness"][n_] = D1.logit_metrics(z, oracle)
                del z
            except Exception as exc:
                row["errors"][n_] = {"stage": "build_or_first_call", "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
                del arms[n_]
            row["build_seconds"][n_] = round(time.time() - t1, 2)
        del oracle
        torch.cuda.empty_cache()
        row["gate"] = D1.gate(row["correctness"])
        if "candidate" not in arms or not row["gate"]["pass"]:
            row["status"] = "tested_invalid"; row["gate_reasons"] = ["candidate missing or failed the correctness gate"]
            return row
        for _ in range(3):
            for fn in arms.values():
                fn()
        torch.cuda.synchronize()
        row["timing"] = {}
        for j, n_ in enumerate([n for n in arms if n != "candidate"]):
            try:
                row["timing"][n_] = P.timed_pair(impl, arms[n_], arms["candidate"], seed + 1000 * (j + 1),
                                                 args.rounds_initial, args.rounds_retest, args.samples)
            except Exception as exc:
                row["errors"][n_ + "_timing"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
        try:
            aa = impl.paired(arms["compiled"], arms["compiled"], rounds=args.rounds_retest, samples=args.samples, seed=seed + 9000)
            row["self_comparison_compiled"] = statistics.median(aa["a_p50_ms"]) / statistics.median(aa["b_p50_ms"])
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
        row["status"] = "tested_invalid"; row["gate_reasons"] = [repr(exc)]
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
    m = Decoder(model_dir, "cpu", torch.float32)
    B, T = 2, int(os.environ.get("CPU_CHECK_T", "24"))
    ids = torch.randint(0, m.vocab, (B, T + 1))
    _, cK, cV = forward_full(m, ids[:, :T], False, T + 1)
    cos, sin = m.tables(T + 1, "cpu", torch.float32)
    mine = step_vendor(m, ids[:, T].contiguous(), cK, cV, cos[T], sin[T], T)
    full = F.linear(forward_full(m, ids, True)[0][:, -1], m.lm_head)
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32, local_files_only=True).eval()
    with torch.no_grad():
        theirs = ref(input_ids=ids, use_cache=False).logits[:, -1]
    d = mine - theirs
    print(json.dumps({"cpu_check": model_dir.name, "model_type": m.model_type, "headnorm": m.headnorm, "bias": m.bias,
                      "Hq": m.Hq, "Hk": m.Hk, "D": m.D, "layers": m.n_layers,
                      "step_vs_hf_max_abs": float(d.abs().max()), "step_vs_hf_relative_l2": float(d.norm() / theirs.norm()),
                      "full_vs_hf_relative_l2": float((full - theirs).norm() / theirs.norm()),
                      "top1_agreement": float((mine.argmax(-1) == theirs.argmax(-1)).float().mean())}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--shapes", default="1x256,1x1024,4x256,16x256,16x1024", help="comma list of BxT (sequences x cached tokens)")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--tune-layers", type=int, default=4)
    ap.add_argument("--autotune", action="store_true")
    ap.add_argument("--rounds-initial", type=int, default=2)
    ap.add_argument("--rounds-retest", type=int, default=3)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--data", type=Path, help="text file; the prompts are windows of its token stream (default: seeded random ids)")
    ap.add_argument("--ablation-mode", choices=("full-controller", "model-only", "harness-only"),
                     default="full-controller", help="matched controller ablation mode")
    ap.add_argument("--fixed-variant", choices=tuple(VARIANTS), default="all",
                    help="fixed candidate variant for model-only episodes")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    if args.cpu_check:
        cpu_check(args.model_dir)
        return
    if args.output is None:
        ap.error("--output is required")
    import e1_rope_refined_impl as impl
    torch.set_num_threads(4)       # the A800 container has a CPU quota of 24 cores for four concurrent stages (2026-11-22)
    torch._dynamo.config.cache_size_limit = 512
    shapes = [tuple(int(x) for x in s.lower().split("x")) for s in args.shapes.split(",") if s]
    m = Decoder(args.model_dir, "cuda", torch.bfloat16)
    kn = Kernels(m)
    rec = {"schema": "osdi-e2e-decode-v3", "script": "e2e_decode2.py", "candidate_variants": {k: list(v) for k, v in VARIANTS.items()},
           "ablation": {"mode": args.ablation_mode, "fixed_variant": args.fixed_variant,
                        "matched": True, "language_model": None},
           "scope": "whole-model decode step (one new token per sequence over a key/value cache); all arms CUDA-graph replays",
           "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                           "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                           "triton": triton.__version__, "cuda": torch.version.cuda,
                           "compute_capability": list(torch.cuda.get_device_capability())},
           "source_hashes": {p.name: D1.sha(p) for p in (HERE / "e2e_decode2.py", HERE / "e2e_decode.py", HERE / "decode_mlp.py",
                                                        HERE / "e2e_prefill.py")},
           "protocol": {"shapes": shapes, "dtype": "bfloat16",
                        "inputs": ("windows of " + args.data.name) if args.data else "seeded random token ids", "tune_layers": args.tune_layers,
                        "rounds_initial": args.rounds_initial, "rounds_retest": args.rounds_retest, "samples": args.samples,
                        "win_threshold": WIN, "autotune_arm": bool(args.autotune), "cpu_threads": 4,
                        "qkv_configs": [list(c) for c in QKV_CONFIGS], "attn_configs": [list(c) for c in D1.ATT_CONFIGS] + ["sdpa"],
                        "gemm_configs": [list(c) for c in dm.CONFIGS]},
           "model": args.model_dir.name,
           "model_config": {"model_type": m.model_type, "layers": m.n_layers, "hidden": m.hidden, "Hq": m.Hq, "Hk": m.Hk,
                            "D": m.D, "headnorm": m.headnorm, "bias": m.bias},
           "rows": [], "status": "running", "publication_eligible": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for (B, T) in shapes:
        print(f"{args.model_dir.name} B={B} T={T}", flush=True)
        row = one_row(m, kn, impl, B, T, args, args.model_dir.name)
        rec["rows"].append(row)
        args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
        t = row.get("timing", {})
        print(json.dumps({"B": B, "T": T, "status": row["status"], "gate": row.get("gate", {}).get("pass"),
                          "err": {k: round(v["relative_l2"], 5) for k, v in row.get("correctness", {}).items()},
                          "cand_ms": round(t["compiled"]["candidate_p50_ms"], 4) if "compiled" in t else None,
                          "speedup": {k: round(v["speedup"], 4) for k, v in t.items()},
                          "vs_fastest": row.get("speedup_vs_fastest"), "kept": row.get("kept"),
                          "variant": row.get("variant"), "config": row.get("config", {}).get("candidate"),
                          "errors": {k: v.get("error", "")[:300] for k, v in row["errors"].items()},
                          "reasons": row.get("gate_reasons")}), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
