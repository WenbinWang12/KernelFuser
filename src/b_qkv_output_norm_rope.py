#!/usr/bin/env python3
"""Minimal auditable B lane: Qwen3-8B QKV-output norm+RoPE + vendor SDPA.

This is an operator smoke, not a model result.  The custom candidate writes the
Q/K projection output after RMSNorm and split-half RoPE in Triton and writes V
unchanged; attention is always PyTorch's vendor SDPA.  A standalone custom
QKV projection is timed against ``torch.mm`` and the row is rejected unless it
is within 1% of that vendor GEMM.  No checkpoint is downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import e1_rope_publication_v1 as rope

MODEL = {
    "name": "Qwen3-8B",
    "hidden": 4096,
    "q_heads": 32,
    "kv_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
}
DTYPE = torch.bfloat16
BLOCK_K = 64
VOCAB = 256
ATOL = 0.05
RTOL = 0.05


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@triton.jit
def _qkv_kernel(
    X, W, GQ, GK, COS, SIN, Y,
    M: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    NHEADS: tl.constexpr, BASE_HEAD: tl.constexpr,
    DO_NORM: tl.constexpr, Q_IS: tl.constexpr,
    EPS: tl.constexpr, BLOCK_D: tl.constexpr, BK: tl.constexpr,
):
    """One token/head program: GEMM tile followed by optional Q/K epilogue.

    ``DO_NORM`` and ``Q_IS`` are compile-time flags, so Q, K, and V launches
    have no runtime branch.  The temporary normalized Q/K values are written
    before the partner-half load; the compiler barrier keeps that read-after-
    write explicit within the fused program.
    """
    pid = tl.program_id(0)
    row = pid // NHEADS
    head = pid - row * NHEADS
    d = tl.arange(0, BLOCK_D)
    acc = tl.zeros((1, BLOCK_D), dtype=tl.float32)
    base = (BASE_HEAD + head) * D
    xbase = X + row * K
    for kk0 in range(0, K, BK):
        kk = kk0 + tl.arange(0, BK)
        xm = kk < K
        xv = tl.load(xbase + kk, mask=xm, other=0).to(tl.bfloat16)
        wm = (d[None, :] < D) & xm[:, None]
        w = tl.load(W + (base + d[None, :]) * K + kk[:, None], mask=wm, other=0).to(tl.bfloat16)
        acc += tl.dot(xv[None, :], w, input_precision="ieee")
    # QKV projection writes BF16 before Q/K RMSNorm, matching vendor projection boundary.
    # Triton 3.1 rejects indexing a size-one constexpr dimension.  Reducing
    # that dimension is equivalent and keeps this kernel portable.
    y = tl.sum(acc, axis=0).to(tl.bfloat16).to(tl.float32)
    out_ptr = Y + row * ((HQ + 2 * HK) * D) + base + d
    mask = d < D
    if DO_NORM:
        if Q_IS:
            gamma = tl.load(GQ + d, mask=mask, other=1).to(tl.bfloat16)
        else:
            gamma = tl.load(GK + d, mask=mask, other=1).to(tl.bfloat16)
        inv = tl.rsqrt(tl.sum(y * y, axis=0) / D + EPS)
        # Match Qwen3/HF's BF16 boundaries: normalize -> BF16 -> gamma -> BF16.
        yn = (y * inv).to(tl.bfloat16)
        yn = (yn * gamma).to(tl.bfloat16)
        tl.store(out_ptr, yn, mask=mask)
        tl.debug_barrier()
        jp = tl.where(d < D // 2, d + D // 2, d - D // 2)
        partner = tl.load(Y + row * ((HQ + 2 * HK) * D) + base + jp,
                          mask=(jp >= 0) & (jp < D), other=0).to(tl.bfloat16)
        pos = row % S
        c = tl.load(COS + pos * D + d, mask=mask, other=1).to(tl.bfloat16)
        s = tl.load(SIN + pos * D + d, mask=mask, other=0).to(tl.bfloat16)
        p0 = (yn * c).to(tl.bfloat16)
        p1 = (partner * s).to(tl.bfloat16)
        rot = tl.where(d < D // 2,
                       (p0 - p1).to(tl.bfloat16),
                       (p0 + p1).to(tl.bfloat16))
        tl.store(out_ptr, rot, mask=mask)
    else:
        tl.store(out_ptr, y.to(tl.bfloat16), mask=mask)


def _launch(x: torch.Tensor, w: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor,
            cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor,
            *, b: int, s: int, hq: int, hk: int, d: int,
            norm: bool, q_is: bool, base_head: int, nheads: int) -> None:
    m, k = x.shape
    _qkv_kernel[(m * nheads,)](
        x, w, gq, gk, cos, sin, out,
        m, k, s, hq, hk, d, nheads, base_head, norm, q_is,
        float(MODEL["rms_norm_eps"]), d, BLOCK_K,
        num_warps=4, num_stages=2,
    )


def _views(y: torch.Tensor, b: int, s: int, hq: int, hk: int, d: int):
    qn = y[:, :hq * d].view(b, s, hq, d).transpose(1, 2)
    kn = y[:, hq * d:(hq + hk) * d].view(b, s, hk, d).transpose(1, 2)
    v = y[:, (hq + hk) * d:].view(b, s, hk, d).transpose(1, 2)
    return qn, kn, v


def _make_tables(s: int, d: int, theta: float, device: torch.device):
    pos = torch.arange(s, device=device, dtype=torch.float32)
    inv = theta ** (-torch.arange(0, d, 2, device=device, dtype=torch.float32) / d)
    ang = pos[:, None] * inv[None, :]
    c = torch.cat((torch.cos(ang), torch.cos(ang)), dim=1).to(DTYPE).contiguous()
    si = torch.cat((torch.sin(ang), torch.sin(ang)), dim=1).to(DTYPE).contiguous()
    return c, si


def _make_case(b: int, s: int, seed: int):
    h = MODEL["hidden"]; hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    n = (hq + 2 * hk) * d
    gen = torch.Generator(device="cuda").manual_seed(int(seed))
    x = torch.randn((b * s, h), device="cuda", dtype=DTYPE, generator=gen)
    w = (torch.randn((n, h), device="cuda", dtype=DTYPE, generator=gen) * (h ** -0.5)).contiguous()
    gq = (torch.ones((d,), device="cuda", dtype=DTYPE) +
          torch.randn((d,), device="cuda", dtype=DTYPE, generator=gen) * 0.05).contiguous()
    gk = (torch.ones((d,), device="cuda", dtype=DTYPE) +
          torch.randn((d,), device="cuda", dtype=DTYPE, generator=gen) * 0.05).contiguous()
    cos, sin = _make_tables(s, d, MODEL["rope_theta"], x.device)
    return x, w, gq, gk, cos, sin


def _candidate_projection(case, out):
    x, w, gq, gk, cos, sin = case
    b_s, h = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=True, q_is=True, base_head=0, nheads=hq)
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=True, q_is=False, base_head=hq, nheads=hk)
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=False, q_is=False, base_head=hq + hk, nheads=hk)
    return _views(out, b, s, hq, hk, d)


def _custom_raw_projection(case, out):
    x, w, gq, gk, cos, sin = case
    b_s, _ = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=False, q_is=True, base_head=0, nheads=hq)
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=False, q_is=False, base_head=hq, nheads=hk)
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=False, q_is=False, base_head=hq + hk, nheads=hk)
    return out


def _vendor_projection(case, out):
    x, w, *_ = case
    return torch.mm(x, w.t(), out=out)


def _vendor_forward(case, out, causal: bool):
    x, w, gq, gk, cos, sin = case
    b_s, _ = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    _vendor_projection(case, out)
    q, k, v = _views(out, b, s, hq, hk, d)
    q = rope.rope_torch(rope.norm_torch(q, gq, MODEL["rms_norm_eps"]), cos, sin)
    k = rope.rope_torch(rope.norm_torch(k, gk, MODEL["rms_norm_eps"]), cos, sin)
    return F.scaled_dot_product_attention(q, rope.expand(k, hq), rope.expand(v, hq),
                                          is_causal=causal, scale=d ** -0.5)


def _vendor_compile_forward(x, w, gq, gk, cos, sin, causal: bool):
    """Allocation based vendor baseline used only by TorchInductor.

    The timed eager vendor arm uses a fixed output buffer.  This equivalent
    functional form lets TorchInductor see the complete GEMM -> norm+RoPE ->
    SDPA graph without an ``out=`` mutation that would force a graph break.
    """
    b_s, _ = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    raw = x @ w.t()
    q, k, v = _views(raw, b, s, hq, hk, d)
    q = rope.rope_torch(rope.norm_torch(q, gq, MODEL["rms_norm_eps"]), cos, sin)
    k = rope.rope_torch(rope.norm_torch(k, gk, MODEL["rms_norm_eps"]), cos, sin)
    return F.scaled_dot_product_attention(q, rope.expand(k, hq), rope.expand(v, hq),
                                          is_causal=causal, scale=d ** -0.5)


def _candidate_forward(case, out, causal: bool):
    x, w, gq, gk, cos, sin = case
    b_s, _ = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    q, k, v = _candidate_projection(case, out)
    return F.scaled_dot_product_attention(q, rope.expand(k, hq), rope.expand(v, hq),
                                          is_causal=causal, scale=d ** -0.5)


def _oracle(case, causal: bool):
    x, w, gq, gk, cos, sin = case
    b_s, _ = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    raw = x.float() @ w.float().t()
    q, k, v = _views(raw, b, s, hq, hk, d)
    q = rope.rope_torch(q * torch.rsqrt(q.square().mean(-1, keepdim=True) + MODEL["rms_norm_eps"]) * gq.float(), cos, sin, fp32=True)
    k = rope.rope_torch(k * torch.rsqrt(k.square().mean(-1, keepdim=True) + MODEL["rms_norm_eps"]) * gk.float(), cos, sin, fp32=True)
    o = F.scaled_dot_product_attention(q, rope.expand(k, hq), rope.expand(v, hq),
                                       is_causal=causal, scale=d ** -0.5)
    return raw, q, k, v, o


def _metric(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    a = got.float(); b = ref.float(); finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "max_abs": float("inf"), "relative_l2": float("inf"), "pass": False}
    diff = a - b
    rel = float(diff.norm() / b.norm().clamp_min(1e-12))
    close = bool(torch.allclose(a, b, atol=ATOL, rtol=RTOL))
    return {"finite": True, "max_abs": float(diff.abs().max()), "relative_l2": rel,
            "pass": bool(close and rel <= 0.05)}


def _logits(o: torch.Tensor) -> torch.Tensor:
    # Shape-only diagnostic logits.  The increasing bias prevents accidental
    # ties while keeping the metric sensitive to context changes.
    ctx = o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1).float()
    bias = torch.arange(VOCAB, device=o.device, dtype=torch.float32) * 0.25
    return ctx[..., :VOCAB] + bias


def _nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)))


def _compiler_evidence(fn: Callable, args: tuple) -> tuple[Callable | None, dict[str, Any]]:
    try:
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, mode="max-autotune")
        compiled(*args)
        torch.cuda.synchronize()
        evidence: dict[str, Any] = {"status": "compiled", "graph_count": None, "graph_break_count": None}
        try:
            exp = torch._dynamo.explain(fn)(*args)
            evidence["graph_count"] = len(getattr(exp, "graphs", []))
            evidence["graph_break_count"] = len(getattr(exp, "break_reasons", []))
            evidence["ops_per_graph"] = [len(g.graph.nodes) for g in getattr(exp, "graphs", [])]
            evidence["graph_targets"] = [[str(n.target) for n in g.graph.nodes]
                                           for g in getattr(exp, "graphs", [])]
        except Exception as exc:
            evidence["explain_error"] = type(exc).__name__
        return compiled, evidence
    except Exception as exc:
        return None, {"status": "compile_failed", "error_type": type(exc).__name__}


def _anchor(case, out_custom, out_vendor, rounds: int, samples: int):
    _custom_raw_projection(case, out_custom)
    _vendor_projection(case, out_vendor)
    torch.cuda.synchronize()
    custom = lambda: _custom_raw_projection(case, out_custom)
    vendor = lambda: _vendor_projection(case, out_vendor)
    timing = rope.paired(vendor, custom, rounds, samples, seed=4401)
    vendor_ms = statistics.median(timing["a_p50_ms"])
    custom_ms = statistics.median(timing["b_p50_ms"])
    return timing, {"vendor_p50_ms": vendor_ms, "custom_p50_ms": custom_ms,
                    "custom_over_vendor": custom_ms / max(vendor_ms, 1e-12),
                    "pass": bool(custom_ms <= 1.01 * vendor_ms)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", "--output", dest="output", type=Path, required=True)
    ap.add_argument("--models", default="qwen3-8b", help="accepted for campaign CLI compatibility; only qwen3-8b is implemented")
    ap.add_argument("--config-json", type=Path, default=None, help="optional local shape manifest; never downloads weights")
    ap.add_argument("--batch", default="1")
    ap.add_argument("--seq", default="128")
    ap.add_argument("--causal-modes", default="false,true")
    ap.add_argument("--seeds", default="1701,2803,3907")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--trace-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="one B=1,S=128,non-causal row; low timing count")
    args = ap.parse_args()
    if args.smoke:
        args.batch, args.seq, args.causal_modes = "1", "128", "false"
        args.seeds, args.rounds, args.samples = "1701", 1, 3
    if str(args.models).lower().replace("_", "-") not in ("qwen3-8b", "qwen3-8b,", "qwen3-8b"):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema": "kernelfuser-abc-b-qkv-output-v1",
                                           "status": "blocked_dependency", "publication_eligible": False,
                                           "reasons": ["only_qwen3_8b_minimal_runner_is_implemented"]}, indent=2) + "\n", encoding="utf-8")
        return
    if args.config_json is not None and not args.config_json.is_file():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"schema": "kernelfuser-abc-b-qkv-output-v1",
                                           "status": "blocked_dependency", "publication_eligible": False,
                                           "reasons": ["config_json_missing"]}, indent=2) + "\n", encoding="utf-8")
        return
    if not torch.cuda.is_available():
        rec = {"schema": "kernelfuser-abc-b-qkv-output-v1", "status": "blocked_dependency",
               "publication_eligible": False, "reasons": ["cuda_unavailable"]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        return
    batches = [int(x) for x in str(args.batch).split(",") if x.strip()]
    seqs = [int(x) for x in str(args.seq).split(",") if x.strip()]
    causals = [x.strip().lower() == "true" for x in str(args.causal_modes).split(",") if x.strip()]
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    trace_root = args.trace_dir or (args.output.parent / "b-qkv-traces")
    here = Path(__file__).resolve()
    rec: dict[str, Any] = {
        "schema": "kernelfuser-abc-b-qkv-output-v1", "track": "B",
        "status": "running", "publication_eligible": False,
        "source_commit": os.getenv("KERNELFUSER_SOURCE_COMMIT", "working-tree"),
        "claim_boundary": "Qwen3-8B shape-only QKV operator smoke; no checkpoint/full-model/training claim",
        "source_sha256": _sha256(here),
        "config_manifest_sha256": _sha256(args.config_json) if args.config_json is not None else None,
        "config_json": ({"path": str(args.config_json), "sha256": _sha256(args.config_json)}
                        if args.config_json is not None and args.config_json.is_file() else None),
        "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "triton": triton.__version__, "visible_gpu": os.getenv("CUDA_VISIBLE_DEVICES")},
        "geometry": MODEL,
        "protocol": {"dtype": "bfloat16", "B": batches, "S": seqs, "causal_modes": causals,
                     "warmup": 20, "rounds": args.rounds, "samples": args.samples,
                     "timing": "paired randomized CUDA-graph replay; vendor SDPA for both arms",
                     "anchor_gate": "standalone custom QKV projection <= 1% over vendor torch.mm",
                     "correctness": "Q,K,V,attention,synthetic logits/top-1/NLL vs eager FP32 oracle",
                     "rope": "split-half; theta=1e6; Qwen3 RMSNorm eps=1e-6"},
        "rows": [],
    }
    for b in batches:
        for s in seqs:
            for causal in causals:
                row: dict[str, Any] = {"B": b, "S": s, "causal": causal, "errors": [], "traces": []}
                row["gpu_exclusivity"] = {"before": rope.gpu_snapshot()}
                monitor = rope.ExclusiveMonitor().start()
                try:
                    case = _make_case(b, s, seeds[0])
                    x, w, gq, gk, cos, sin = case
                    n = (MODEL["q_heads"] + 2 * MODEL["kv_heads"]) * MODEL["head_dim"]
                    cand_buf = torch.empty((b * s, n), device="cuda", dtype=DTYPE)
                    vend_buf = torch.empty_like(cand_buf)
                    raw_c = torch.empty_like(cand_buf)
                    raw_v = torch.empty_like(cand_buf)
                    row["correctness"] = {"seeds": [], "pass": False}
                    for seed in seeds:
                        case = _make_case(b, s, seed)
                        x, w, gq, gk, cos, sin = case
                        cand_buf = torch.empty((b * s, n), device="cuda", dtype=DTYPE)
                        vend_buf = torch.empty_like(cand_buf)
                        raw, oq, ok, ov, oo = _oracle(case, causal)
                        cq, ck, cv = _candidate_projection(case, cand_buf)
                        vo = _vendor_forward(case, vend_buf, causal)
                        vq, vk, vv = _views(vend_buf, b, s, MODEL["q_heads"], MODEL["kv_heads"], MODEL["head_dim"])
                        vq = rope.rope_torch(rope.norm_torch(vq, gq, MODEL["rms_norm_eps"]), cos, sin)
                        vk = rope.rope_torch(rope.norm_torch(vk, gk, MODEL["rms_norm_eps"]), cos, sin)
                        co = F.scaled_dot_product_attention(cq, rope.expand(ck, MODEL["q_heads"]),
                                                            rope.expand(cv, MODEL["q_heads"]),
                                                            is_causal=causal, scale=MODEL["head_dim"] ** -0.5)
                        labels = (torch.arange(b * s, device="cuda", dtype=torch.long).view(b, s) % VOCAB)
                        rlog = _logits(oo); clog = _logits(co); vlog = _logits(vo)
                        rnll = _nll(rlog, labels); cnll = _nll(clog, labels); vnll = _nll(vlog, labels)
                        cseed = {"seed": seed, "candidate": {"Q": _metric(cq, oq), "K": _metric(ck, ok),
                            "V": _metric(cv, ov), "attention": _metric(co, oo), "logits": _metric(clog, rlog),
                            "top1_match": bool(torch.equal(clog.argmax(-1), rlog.argmax(-1))),
                            "nll": {"value": cnll, "reference": rnll, "abs_diff": abs(cnll - rnll), "pass": abs(cnll - rnll) <= 0.05}},
                            "vendor": {"Q": _metric(vq, oq), "K": _metric(vk, ok), "V": _metric(vv, ov),
                                       "attention": _metric(vo, oo), "logits": _metric(vlog, rlog),
                                       "top1_match": bool(torch.equal(vlog.argmax(-1), rlog.argmax(-1))),
                                       "nll": {"value": vnll, "reference": rnll, "abs_diff": abs(vnll - rnll), "pass": abs(vnll - rnll) <= 0.05}}}
                        cseed["pass"] = all(cseed["candidate"][k]["pass"] for k in ("Q", "K", "V", "attention", "logits")) and all(cseed["vendor"][k]["pass"] for k in ("Q", "K", "V", "attention", "logits")) and cseed["candidate"]["top1_match"] and cseed["candidate"]["nll"]["pass"] and cseed["vendor"]["top1_match"] and cseed["vendor"]["nll"]["pass"]
                        row["correctness"]["seeds"].append(cseed)
                    row["correctness"]["pass"] = all(x["pass"] for x in row["correctness"]["seeds"])
                    # Standalone anchor is measured on the first case only.
                    case = _make_case(b, s, seeds[0]); raw_c = torch.empty((b * s, n), device="cuda", dtype=DTYPE); raw_v = torch.empty_like(raw_c)
                    row["anchor_timing"], row["anchor_gate"] = _anchor(case, raw_c, raw_v, args.rounds, max(3, args.samples))
                    raw_ref = case[0].float() @ case[1].float().t()
                    row["anchor_correctness"] = {"custom_raw_vs_fp32": _metric(raw_c, raw_ref), "vendor_raw_vs_fp32": _metric(raw_v, raw_ref)}
                    if args.smoke:
                        # The smoke checks candidate correctness and the
                        # standalone vendor anchor.  Full rows still run the
                        # compiler-fusion audit; skipping Inductor here avoids
                        # a large worker allocation on the diagnostic stack.
                        compile_fn = None
                        row["compiler_evidence"] = {"status": "smoke_skipped"}
                        row["compiler_already_fused"] = None
                        row["compile_correctness"] = {"pass": False}
                    else:
                        compile_fn, row["compiler_evidence"] = _compiler_evidence(lambda xx, ww, ggq, ggk, cc, ss: _vendor_compile_forward(xx, ww, ggq, ggk, cc, ss, causal), case)
                        # FX graph evidence is recorded above, but it cannot prove
                        # whether Inductor lowered RMSNorm/RoPE into one backend
                        # kernel.  Keep this boundary diagnostic explicit rather
                        # than claiming an unfused compiler path from FX alone.
                        row["compiler_already_fused"] = None
                        if compile_fn is not None:
                            row["compile_correctness"] = _metric(compile_fn(*case), _oracle(case, causal)[4])
                        else:
                            row["compile_correctness"] = {"pass": False}
                    # Timed closures reuse fixed buffers and vendor SDPA.
                    case = _make_case(b, s, seeds[0]); cand_buf = torch.empty((b * s, n), device="cuda", dtype=DTYPE); vend_buf = torch.empty_like(cand_buf)
                    cand = lambda: _candidate_forward(case, cand_buf, causal)
                    vend = lambda: _vendor_forward(case, vend_buf, causal)
                    row["timing"] = {"vendor_sdpa_vs_candidate": rope.paired(vend, cand, args.rounds, args.samples, seed=5501)}
                    if compile_fn is not None:
                        comp = lambda: compile_fn(*case)
                        row["timing"]["torch_compile_vs_candidate"] = rope.paired(comp, cand, args.rounds, args.samples, seed=6601)
                    if args.trace_dir is not None:
                        for label, fn in (("vendor", vend), ("candidate", cand)):
                            path = trace_root / f"B{b}_S{s}_C{int(causal)}_{label}.chrome_trace.json.gz"
                            row["traces"].append(rope.capture_trace(fn, path, f"abc_b.{label}"))
                        if compile_fn is not None:
                            path = trace_root / f"B{b}_S{s}_C{int(causal)}_torch_compile.chrome_trace.json.gz"
                            row["traces"].append(rope.capture_trace(lambda: compile_fn(*case), path, "abc_b.torch_compile"))
                except Exception as exc:
                    row["errors"].append({"stage": "execution", "error_type": type(exc).__name__, "error": str(exc)[:500]})
                row["gpu_exclusivity"]["observations"] = monitor.finish()
                row["gpu_exclusivity"]["after"] = rope.gpu_snapshot()
                reasons = []
                if row["errors"]: reasons.append("execution_error")
                if not row.get("correctness", {}).get("pass"): reasons.append("fp32_correctness_failed")
                if not row.get("anchor_gate", {}).get("pass"): reasons.append("custom_qkv_gemm_over_1pct_vendor")
                if not row.get("anchor_correctness", {}).get("custom_raw_vs_fp32", {}).get("pass"): reasons.append("custom_qkv_gemm_correctness_failed")
                if row.get("compiler_evidence", {}).get("status") != "compiled": reasons.append("torch_compile_unavailable")
                if not row.get("compile_correctness", {}).get("pass"): reasons.append("torch_compile_correctness_failed")
                if row.get("compiler_already_fused") is None: reasons.append("compiler_fusion_status_unknown_boundary_diagnostic")
                ex = row["gpu_exclusivity"]
                if not ex["before"].get("exclusive") or not ex["after"].get("exclusive") or not ex["observations"] or not all(x.get("exclusive") for x in ex["observations"]): reasons.append("gpu_exclusivity_failed")
                row["publication_gate"] = {"pass": not reasons, "reasons": reasons}
                row["status"] = "tested_pass" if not reasons else "tested_invalid"
                rec["rows"].append(row)
                rec["status"] = "running"
                args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
                print(json.dumps({"B": b, "S": s, "causal": causal, "status": row["status"], "gate": row["publication_gate"]}), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r.get("status") == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "publication_eligible": rec["publication_eligible"]}), flush=True)


if __name__ == "__main__":
    main()
