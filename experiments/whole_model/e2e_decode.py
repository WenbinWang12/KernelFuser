#!/usr/bin/env python3
"""Whole-model decode step of Qwen3 with fused kernels (written 2026-11-21).

What is measured
  One decode step of a released Qwen3 checkpoint in BF16: B sequences, each with a key/value
  cache of T tokens, receive one new token; the step runs the embedding, every decoder layer
  (the new key and value are written into the cache at position T), the final norm and the
  LM head, and returns the next-token logits [B, V].  Every arm is captured in a CUDA graph
  and timed by replay in randomized interleaved pairs (e1_rope_refined_impl.paired).

Arms
  eager      functional HuggingFace-equivalent step: cuBLAS GEMMs, PyTorch normalization and
             pointwise kernels, SDPA over the cache.  The cache is a preallocated tensor that
             the step writes in place (no concatenation), and the n query heads that share a
             key head are passed to SDPA as n queries (no repeat of keys and values).
  compiled   torch.compile(mode="default", fullgraph=True) of that step for the whole model.
  autotune   (--autotune) torch.compile(mode="max-autotune-no-cudagraphs") of the same step;
             TorchInductor may then choose its own Triton GEMM templates with fused epilogues.
  unfused    the candidate's kernels with every prologue and epilogue switched off, and the
             removed operators as separate torch.compile'd kernels.
  candidate  five kernels per layer:
             1. Q/K/V projection as one GEMM over the stacked weights, with the input RMSNorm
                as prologue and, as epilogue, the per-head RMSNorm and RoPE of Q and K and the
                write of K and V into the cache;
             2. attention of the new token over the cache;
             3. output projection with the residual add as epilogue;
             4. gate and up GEMMs in one loop with the RMSNorm as prologue and SiLU(gate)*up
                as epilogue;                                   (kernels 3-5 are decode_mlp.py)
             5. down GEMM with the residual add as epilogue.
             Embedding, final norm and LM head are the PyTorch operators in all arms
             (compiled in the unfused and candidate arms).

Correctness (before any timing; rule fixed here before the first run)
  Oracle: FP32 eager forward of the same checkpoint on the T+1 tokens, logits of the last
  position.  The BF16 arms share one BF16 cache, produced by an eager BF16 prefill.
  With e(arm) the relative L2 error of the logits and kl(arm) the mean KL divergence of
  softmax(oracle) from softmax(arm), the candidate passes if its logits are finite,
      e(candidate)  <= min(0.05, max(1.5 x max(e(eager), e(compiled)), 1e-3)), and
      kl(candidate) <= max(1.5 x max(kl(eager), kl(compiled)), 1e-4).
  Top-1 agreement with the oracle is recorded.

Tile configurations of the Triton kernels are chosen per shape, separately for the candidate
and for the unfused arm, from explicit lists by the CUDA-graph replay time of the first
--tune-layers layers; lists and timings are recorded.

--cpu-check compares prefill + step of the functional code with transformers on the CPU (FP32).
"""
from __future__ import annotations

import argparse, gc, hashlib, json, os, platform, statistics, sys, time, traceback
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import e2e_prefill as P

REL_L2_ABS, REL_L2_FACTOR, REL_L2_FLOOR = 0.05, 1.5, 1.0e-3
KL_FACTOR, KL_FLOOR = 1.5, 1.0e-4
WIN = 1.01
BQ = 16                                             # query rows per attention program (n query heads, padded)

QKV_CONFIGS = [(64, 4, 3), (128, 4, 3), (128, 4, 4), (256, 4, 2), (256, 4, 3), (512, 4, 2)]   # (BLOCK_K, warps, stages)
ATT_CONFIGS = [(32, 4, 2), (64, 4, 2), (64, 4, 3), (128, 4, 2), (128, 8, 2)]                    # (keys per step, warps, stages)


# ----------------------------------------------------------------------------- functional step (PyTorch arms)
def prefill_cache(m, ids, tmax):
    """Eager causal prefill in the dtype of m; returns per-layer K and V caches [B, Hk, tmax, D]."""
    B, S = ids.shape
    dt = m.embed.dtype
    cos, sin = m.tables(S, ids.device, dt)
    n = m.Hq // m.Hk
    h = F.embedding(ids, m.embed)
    cK, cV = [], []
    for L in m.layers:
        x = P.rms(h, L.ln1, m.eps)
        q = P.rms(F.linear(x, L.wq).view(B, S, m.Hq, m.D), L.gq, m.eps).transpose(1, 2)
        k = P.rms(F.linear(x, L.wk).view(B, S, m.Hk, m.D), L.gk, m.eps).transpose(1, 2)
        v = F.linear(x, L.wv).view(B, S, m.Hk, m.D).transpose(1, 2)
        q, k = P.rope(q, cos, sin), P.rope(k, cos, sin)
        ck = torch.zeros((B, m.Hk, tmax, m.D), device=ids.device, dtype=dt); ck[:, :, :S] = k
        cv = torch.zeros((B, m.Hk, tmax, m.D), device=ids.device, dtype=dt); cv[:, :, :S] = v
        cK.append(ck); cV.append(cv)
        o = F.scaled_dot_product_attention(q.contiguous(), P.repeat_kv(k, n).contiguous(), P.repeat_kv(v, n).contiguous(),
                                           is_causal=True, scale=m.D ** -0.5)
        h = h + F.linear(o.transpose(1, 2).reshape(B, S, m.Hq * m.D), L.wo)
        h = h + P.mlp(P.rms(h, L.ln2, m.eps), L.wg, L.wu, L.wd)
        del q, k, v, o
    return cK, cV


def layer_vendor(h, L, ck, cv, cosT, sinT, T, Hq, Hk, D, eps):
    B = h.shape[0]
    x = P.rms(h, L.ln1, eps)
    q = P.rope(P.rms(F.linear(x, L.wq).view(B, Hq, D), L.gq, eps), cosT, sinT)
    k = P.rope(P.rms(F.linear(x, L.wk).view(B, Hk, D), L.gk, eps), cosT, sinT)
    ck[:, :, T] = k
    cv[:, :, T] = F.linear(x, L.wv).view(B, Hk, D)
    o = F.scaled_dot_product_attention(q.view(B, Hk, Hq // Hk, D), ck[:, :, :T + 1], cv[:, :, :T + 1], scale=D ** -0.5)
    h = h + F.linear(o.reshape(B, Hq * D), L.wo)
    return h + P.mlp(P.rms(h, L.ln2, eps), L.wg, L.wu, L.wd)


def step_vendor(m, ids, cK, cV, cosT, sinT, T):
    h = F.embedding(ids, m.embed)
    for i, L in enumerate(m.layers):
        h = layer_vendor(h, L, cK[i], cV[i], cosT, sinT, T, m.Hq, m.Hk, m.D, m.eps)
    return F.linear(P.rms(h, m.final_norm, m.eps), m.lm_head)


# ----------------------------------------------------------------------------- Triton kernels
@triton.jit
def _qkv(x_ptr, g_ptr, w_ptr, gq_ptr, gk_ptr, cos_ptr, sin_ptr, q_ptr, ck_ptr, cv_ptr, o_ptr,
         M, K, eps, T, TMAX,
         HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, FUSED: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """One program = one head of the stacked [Q | K | V] projection (two half-head accumulators).

    FUSED: x is the raw residual stream; RMSNorm prologue, head norm + RoPE epilogue for Q and K,
    results stored to the query buffer or into the cache at position T.
    Otherwise: x is already normalized and the raw projection is stored to o_ptr.
    """
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
        if FUSED:
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
    if FUSED:
        r = tl.rsqrt(ss / K + eps)
        a = acca * r[:, None]
        b = accb * r[:, None]
        is_q = head < HQ
        is_v = head >= HQ + HK
        is_k = (head >= HQ) & (head < HQ + HK)
        hr = tl.rsqrt((tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)) / D + eps)
        ga = tl.where(is_q, tl.load(gq_ptr + oh), tl.load(gk_ptr + oh))
        gb = tl.where(is_q, tl.load(gq_ptr + HALF + oh), tl.load(gk_ptr + HALF + oh))
        an = a * hr[:, None] * ga[None, :]
        bn = b * hr[:, None] * gb[None, :]
        ca = tl.load(cos_ptr + oh)
        sa = tl.load(sin_ptr + oh)
        cb = tl.load(cos_ptr + HALF + oh)
        sb = tl.load(sin_ptr + HALF + oh)
        ra = an * ca[None, :] - bn * sa[None, :]          # x*cos + rotate_half(x)*sin, first half
        rb = bn * cb[None, :] + an * sb[None, :]          # second half
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
        tl.store(o_ptr + ooff, acca.to(tl.bfloat16), mask=full)
        tl.store(o_ptr + ooff + HALF, accb.to(tl.bfloat16), mask=full)

@triton.jit
def _attn(q_ptr, ck_ptr, cv_ptr, o_ptr, T1, TMAX, scale,
          HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, NG: tl.constexpr, NQ: tl.constexpr,
          BN: tl.constexpr):
    """One program = one (sequence, key head): its NG query heads attend over T1 cached keys."""
    hk = tl.program_id(0)
    b = tl.program_id(1)
    rq = tl.arange(0, NQ)
    dj = tl.arange(0, D)
    qmask = (rq[:, None] < NG) & (dj[None, :] < D)
    qoff = b * (HQ * D) + (hk * NG + rq[:, None]) * D + dj[None, :]
    q = tl.load(q_ptr + qoff, mask=qmask, other=0.0)
    base = b.to(tl.int64) * (HK * TMAX * D) + hk * (TMAX * D)
    m_i = tl.full((NQ,), float("-inf"), tl.float32)
    l_i = tl.zeros((NQ,), tl.float32)
    acc = tl.zeros((NQ, D), tl.float32)
    for t0 in range(0, T1, BN):
        tj = t0 + tl.arange(0, BN)
        tmask = tj < T1
        kmask = tmask[:, None] & (dj[None, :] < D)
        k = tl.load(ck_ptr + base + tj[:, None] * D + dj[None, :], mask=kmask, other=0.0)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(tmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(cv_ptr + base + tj[:, None] * D + dj[None, :], mask=kmask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + qoff, o.to(tl.bfloat16), mask=qmask)



class Kernels:
    def __init__(self, m):
        import decode_mlp as dm
        self.dm = dm
        self.triton, self._qkv, self._attn = triton, _qkv, _attn
        self.m = m
        from types import SimpleNamespace
        self.layers = []                            # layouts used by the Triton arms (same values as the checkpoint)
        for L in m.layers:
            self.layers.append(SimpleNamespace(
                ln1=L.ln1, ln2=L.ln2, gq=L.gq, gk=L.gk, wo=L.wo, wg=L.wg, wu=L.wu, wd=L.wd,
                wqkv=torch.cat((L.wq, L.wk, L.wv), dim=0).contiguous(),
                g1=L.ln1.float().contiguous(), g2=L.ln2.float().contiguous(),
                gq32=L.gq.float().contiguous(), gk32=L.gk.float().contiguous()))

    def qkv(self, x, L, st, i, cfg, fused):
        m, M = self.m, x.shape[0]
        bm = self.dm.block_m(M)
        bk, warps, stages = cfg
        if fused:
            out = torch.empty((M, m.Hq * m.D), device=x.device, dtype=x.dtype)
            q, ck, cv, o = out, st.cK[i], st.cV[i], out
        else:
            out = torch.empty((M, (m.Hq + 2 * m.Hk) * m.D), device=x.device, dtype=x.dtype)
            q, ck, cv, o = out, out, out, out
        self._qkv[(m.Hq + 2 * m.Hk, self.triton.cdiv(M, bm))](
            x, L.g1, L.wqkv, L.gq32, L.gk32, st.cos32, st.sin32, q, ck, cv, o, M, x.shape[1], m.eps, st.T, st.T + 1,
            HQ=m.Hq, HK=m.Hk, D=m.D, HALF=m.D // 2, FUSED=fused, BLOCK_M=bm, BLOCK_K=bk,
            num_warps=warps, num_stages=stages)
        return out

    def attn(self, q, st, i, cfg):
        m = self.m
        bn, warps, stages = cfg
        o = torch.empty_like(q)
        self._attn[(m.Hk, q.shape[0])](q, st.cK[i], st.cV[i], o, st.T + 1, st.T + 1, m.D ** -0.5,
                                         HQ=m.Hq, HK=m.Hk, D=m.D, NG=m.Hq // m.Hk, NQ=BQ, BN=bn,
                                         num_warps=warps, num_stages=stages)
        return o


class State:
    pass


def make_small(m, T):
    """The separate kernels of the unfused arm (torch.compile'd, one artifact each, reused by all layers)."""
    Hq, Hk, D, eps = m.Hq, m.Hk, m.D, m.eps
    comp = lambda fn: torch.compile(fn, fullgraph=True, dynamic=False, mode="default")

    def post_qkv(buf, gq, gk, cosT, sinT, ck, cv):
        B = buf.shape[0]
        q = P.rope(P.rms(buf[:, :Hq * D].view(B, Hq, D), gq, eps), cosT, sinT)
        k = P.rope(P.rms(buf[:, Hq * D:(Hq + Hk) * D].view(B, Hk, D), gk, eps), cosT, sinT)
        ck[:, :, T] = k
        cv[:, :, T] = buf[:, (Hq + Hk) * D:].view(B, Hk, D)
        return q.reshape(B, Hq * D)

    return {"norm": comp(lambda x, w: P.rms(x, w, eps)), "post_qkv": comp(post_qkv),
            "act": comp(lambda a, b: F.silu(a) * b), "add": comp(lambda a, b: a + b),
            "head": comp(lambda h, w, lm: F.linear(P.rms(h, w, eps), lm))}


def layers_candidate(kn, st, cfg, h, layers):
    dm, eps = kn.dm, kn.m.eps
    for i, L in layers:
        q = kn.qkv(h, L, st, i, cfg["qkv"], True)
        o = kn.attn(q, st, i, cfg["attn"])
        h = dm.down(o, L.wo, h, cfg["oproj"], True)
        a, _ = dm.gate_up(h, L.g2, L.wg, L.wu, cfg["gate_up"], True, False, eps)
        h = dm.down(a, L.wd, h, cfg["down"], True)
    return h


def layers_unfused(kn, st, small, cfg, h, layers):
    dm, eps = kn.dm, kn.m.eps
    for i, L in layers:
        buf = kn.qkv(small["norm"](h, L.ln1), L, st, i, cfg["qkv"], False)
        q = small["post_qkv"](buf, L.gq, L.gk, st.cos16, st.sin16, st.cK[i], st.cV[i])
        o = kn.attn(q, st, i, cfg["attn"])
        h = small["add"](h, dm.down(o, L.wo, h, cfg["oproj"], False))
        a, b = dm.gate_up(small["norm"](h, L.ln2), L.g2, L.wg, L.wu, cfg["gate_up"], False, False, eps)
        h = small["add"](h, dm.down(small["act"](a, b), L.wd, h, cfg["down"], False))
    return h


def tune(make, impl, dm, defaults, space):
    """Coordinate search: one kernel at a time, by the replay time of the tuning chain."""
    best, log = dict(defaults), {}
    for name in ("qkv", "attn", "oproj", "gate_up", "down"):
        res = {}
        for c in space[name]:
            trial = dict(best); trial[name] = c
            try:
                res[c] = dm.graph_time(make(trial), impl)
            except Exception as exc:                # a configuration that does not compile or launch is skipped
                res[c] = repr(exc)[:160]
                torch.cuda.synchronize()
        ok = {c: t for c, t in res.items() if isinstance(t, float)}
        if not ok:
            raise RuntimeError(f"no configuration ran for {name}: {res}")
        best[name] = min(ok, key=ok.get)
        log[name] = {str(c): t for c, t in res.items()}
    return best, log


# ----------------------------------------------------------------------------- checks
def logit_metrics(z, ref):
    z, ref = z.float(), ref.float()
    if not bool(torch.isfinite(z).all()):
        return {"finite": False, "relative_l2": float("inf"), "kl": float("inf"), "top1_agreement": 0.0}
    lp, lq = F.log_softmax(ref, -1), F.log_softmax(z, -1)
    return {"finite": True, "relative_l2": float((z - ref).norm() / ref.norm().clamp_min(1e-30)),
            "kl": float((lp.exp() * (lp - lq)).sum(-1).mean()),
            "top1_agreement": float((z.argmax(-1) == ref.argmax(-1)).float().mean()),
            "max_abs": float((z - ref).abs().max())}


def gate(metrics):
    ref = [metrics[k] for k in ("eager", "compiled") if k in metrics and metrics[k].get("finite")]
    out = {"thresholds": {"rel_l2_abs": REL_L2_ABS, "rel_l2_factor": REL_L2_FACTOR, "rel_l2_floor": REL_L2_FLOOR,
                           "kl_factor": KL_FACTOR, "kl_floor": KL_FLOOR}}
    c = metrics.get("candidate")
    if not ref or c is None:
        out["pass"] = False; out["reason"] = "no finite PyTorch arm or no candidate"
        return out
    l2 = min(REL_L2_ABS, max(REL_L2_FACTOR * max(r["relative_l2"] for r in ref), REL_L2_FLOOR))
    kl = max(KL_FACTOR * max(r["kl"] for r in ref), KL_FLOOR)
    out["limits"] = {"relative_l2": l2, "kl": kl}
    out["pass"] = bool(c["finite"] and c["relative_l2"] <= l2 and c["kl"] <= kl)
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def one_row(m, kn, impl, B, T, args, name):
    dm = kn.dm
    seed = 6301 + 131 * B + 3 * T
    row = {"model": name, "B": B, "T": T, "seed": seed, "status": "running", "errors": {},
           "gpu_exclusivity": {"before": impl.gpu_snapshot()}}
    monitor = impl.ExclusiveMonitor(); monitor.start()
    try:
        torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
        t0 = time.time()
        g = torch.Generator(device="cpu").manual_seed(seed)
        ids = torch.randint(0, m.vocab, (B, T + 1), generator=g, dtype=torch.long).cuda()
        new = ids[:, T].contiguous()
        st = State()
        st.T = T
        st.cK, st.cV = prefill_cache(m, ids[:, :T], T + 1)
        cos, sin = m.tables(T + 1, "cuda", torch.float32)
        st.cos32, st.sin32 = cos[T].contiguous(), sin[T].contiguous()
        st.cos16, st.sin16 = st.cos32.to(torch.bfloat16), st.sin32.to(torch.bfloat16)
        hf32 = P.forward_oracle(m, ids)[:, -1]
        oracle = F.linear(hf32, m.lm_head.float())
        del hf32
        torch.cuda.empty_cache()

        small = make_small(m, T)
        all_layers = list(enumerate(kn.layers))
        sub = all_layers[:args.tune_layers]
        h0 = F.embedding(new, m.embed)
        defaults = {"qkv": (128, 4, 3), "attn": (64, 4, 2), "oproj": dm.DEFAULT, "gate_up": dm.DEFAULT, "down": dm.DEFAULT}
        space = {"qkv": QKV_CONFIGS, "attn": ATT_CONFIGS, "oproj": dm.CONFIGS, "gate_up": dm.CONFIGS, "down": dm.CONFIGS}
        for _ in range(5):                          # finish the lazy compilation of the small kernels
            layers_unfused(kn, st, small, defaults, h0, sub)
        torch.cuda.synchronize()
        cfg_c, log_c = tune(lambda c: (lambda: layers_candidate(kn, st, c, h0, sub)), impl, dm, defaults, space)
        cfg_u, log_u = tune(lambda c: (lambda: layers_unfused(kn, st, small, c, h0, sub)), impl, dm, defaults, space)
        row["config"] = {"candidate": {k: list(v) for k, v in cfg_c.items()}, "unfused": {k: list(v) for k, v in cfg_u.items()}}
        row["tuning"] = {"candidate": log_c, "unfused": log_u}
        row["tune_seconds"] = round(time.time() - t0, 1)

        head = lambda h: small["head"](h, m.final_norm, m.lm_head)
        arms = {"eager": lambda: step_vendor(m, new, st.cK, st.cV, st.cos16, st.sin16, T)}
        comp = torch.compile(lambda i, c, s: step_vendor(m, i, st.cK, st.cV, c, s, T), fullgraph=True, dynamic=False, mode="default")
        arms["compiled"] = lambda: comp(new, st.cos16, st.sin16)
        if args.autotune:
            auto = torch.compile(lambda i, c, s: step_vendor(m, i, st.cK, st.cV, c, s, T), fullgraph=True, dynamic=False,
                                 mode="max-autotune-no-cudagraphs")
            arms["autotune"] = lambda: auto(new, st.cos16, st.sin16)
        arms["unfused"] = lambda: head(layers_unfused(kn, st, small, cfg_u, F.embedding(new, m.embed), all_layers))
        arms["candidate"] = lambda: head(layers_candidate(kn, st, cfg_c, F.embedding(new, m.embed), all_layers))

        row["correctness"], row["build_seconds"] = {}, {}
        for n_ in list(arms):
            t1 = time.time()
            try:
                z = arms[n_]()
                torch.cuda.synchronize()
                row["correctness"][n_] = logit_metrics(z, oracle)
                del z
            except Exception as exc:
                row["errors"][n_] = {"stage": "build_or_first_call", "error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
                del arms[n_]
            row["build_seconds"][n_] = round(time.time() - t1, 2)
        del oracle
        torch.cuda.empty_cache()
        row["gate"] = gate(row["correctness"])
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
    m = P.Qwen3(model_dir, "cpu", torch.float32)
    B, T = 2, 24
    ids = torch.randint(0, m.vocab, (B, T + 1))
    cK, cV = prefill_cache(m, ids[:, :T], T + 1)
    cos, sin = m.tables(T + 1, "cpu", torch.float32)
    mine = step_vendor(m, ids[:, T].contiguous(), cK, cV, cos[T], sin[T], T)
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch.float32, local_files_only=True).eval()
    with torch.no_grad():
        theirs = ref(input_ids=ids, use_cache=False).logits[:, -1]
    d = mine - theirs
    print(json.dumps({"cpu_check": model_dir.name, "max_abs": float(d.abs().max()),
                      "relative_l2": float(d.norm() / theirs.norm()),
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
    m = P.Qwen3(args.model_dir, "cuda", torch.bfloat16)
    kn = Kernels(m)
    rec = {"schema": "osdi-e2e-decode-v1",
           "scope": "whole-model decode step (one new token per sequence over a key/value cache); all arms CUDA-graph replays",
           "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                           "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                           "triton": impl.triton.__version__, "cuda": torch.version.cuda,
                           "compute_capability": list(torch.cuda.get_device_capability())},
           "source_hashes": {p.name: sha(p) for p in (HERE / "e2e_decode.py", HERE / "decode_mlp.py", HERE / "e2e_prefill.py")},
           "protocol": {"shapes": shapes, "dtype": "bfloat16", "inputs": "seeded random token ids", "tune_layers": args.tune_layers,
                        "rounds_initial": args.rounds_initial, "rounds_retest": args.rounds_retest, "samples": args.samples,
                        "win_threshold": WIN, "autotune_arm": bool(args.autotune),
                        "qkv_configs": [list(c) for c in QKV_CONFIGS], "attn_configs": [list(c) for c in ATT_CONFIGS],
                        "gemm_configs": [list(c) for c in kn.dm.CONFIGS]},
           "model": args.model_dir.name,
           "model_config": {"layers": m.n_layers, "hidden": m.hidden, "Hq": m.Hq, "Hk": m.Hk, "D": m.D},
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
                          "config": row.get("config", {}).get("candidate"),
                          "errors": {k: v.get("error", "")[:300] for k, v in row["errors"].items()},
                          "reasons": row.get("gate_reasons")}), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
