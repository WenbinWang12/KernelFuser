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
# The original B smoke used one program per token/head.  That shape leaves a
# 4096-wide reduction in every program and is especially poor for Qwen's
# 128-wide heads.  The production candidate below uses a row tile and a small
# output tile.  HEAD_TILE=1 keeps the epilogue race-free on Triton while the
# row tile amortizes the K reduction across tokens.
BLOCK_K = 128
BLOCK_M = 64
HEAD_TILE = 1
VOCAB = 256
ATOL = 0.05
RTOL = 0.05
NLL_ATOL = 0.05
METRIC_ATOL = ATOL
METRIC_RTOL = RTOL
METRIC_NLL_ATOL = NLL_ATOL


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@triton.jit
def _qkv_tiled_kernel(
    X, W, GQ, GK, COS, SIN, Y,
    M: tl.constexpr, K: tl.constexpr, S: tl.constexpr,
    HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
    TOTAL_HEADS: tl.constexpr, APPLY_NORM: tl.constexpr,
    EPS: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    H_TILE: tl.constexpr, BD: tl.constexpr,
):
    """Tiled QKV projection with an inline Q/K RMSNorm + split-half RoPE.

    A program owns ``BM`` tokens and ``H_TILE`` output heads.  The GEMM is a
    normal row/output tile (``BM x BK`` by ``BK x D``); Q/K are normalized and
    rotated before they leave the program, while V is written directly.  The
    normalized values are first written to ``Y`` and read back for the partner
    half so the epilogue has the same BF16 boundary as the reference path.
    ``APPLY_NORM=False`` is used by the raw projection anchor.
    """
    pid = tl.program_id(0)
    tiles_m = (M + BM - 1) // BM
    tile_m = pid % tiles_m
    tile_h = pid // tiles_m
    rows = tile_m * BM + tl.arange(0, BM)
    rm = rows < M
    kk_vec = tl.arange(0, BK)
    d = tl.arange(0, BD)
    out_cols = tl.arange(0, H_TILE * BD)
    n_total = TOTAL_HEADS * D
    head0 = tile_h * H_TILE
    # One BM x (H_TILE*D) GEMM tile covers several adjacent output heads.  This
    # is the key difference from the original row/head launcher: the K loop is
    # amortized over a wide output tile and maps to tensor-core dot operations.
    acc = tl.zeros((BM, H_TILE * BD), dtype=tl.float32)
    for kk0 in range(0, K, BK):
        kk = kk0 + kk_vec
        x = tl.load(X + rows[:, None] * K + kk[None, :],
                    mask=rm[:, None] & (kk[None, :] < K), other=0).to(tl.bfloat16)
        w = tl.load(W + (head0 * D + out_cols[None, :]) * K + kk[:, None],
                    mask=((head0 * D + out_cols[None, :]) < n_total) &
                         (kk[:, None] < K),
                    other=0).to(tl.bfloat16)
        acc += tl.dot(x, w, input_precision="ieee")
    # Round the full projection tile to BF16 once.  The fused epilogue reads
    # these values back by head; this avoids unsupported register-tensor
    # slicing in Triton while keeping GEMM and Q/K post-processing in one
    # kernel launch.
    raw_tile = acc.to(tl.bfloat16)
    raw_ptr = Y + rows[:, None] * n_total + head0 * D + out_cols[None, :]
    raw_mask = rm[:, None] & ((head0 * D + out_cols[None, :]) < n_total)
    tl.store(raw_ptr, raw_tile, mask=raw_mask)
    if not APPLY_NORM:
        return
    tl.debug_barrier()
    # The epilogue is constexpr-unrolled over the heads in the output tile.
    for hi in tl.static_range(H_TILE):
        head = head0 + hi
        hm = head < TOTAL_HEADS
        base = head * D
        out_ptr = Y + rows[:, None] * n_total + base + d[None, :]
        out_mask = rm[:, None] & hm & (d[None, :] < D)
        y = tl.load(out_ptr, mask=out_mask, other=0).to(tl.bfloat16).to(tl.float32)
        # Keep the projection boundary identical to torch.mm(out=BF16): the
        # accumulator is rounded to BF16 before RMSNorm/RoPE (and before V is
        # written).  This also makes the standalone raw anchor comparable.
        is_q = head < HQ
        gq = tl.load(GQ + d, mask=d < D, other=1).to(tl.bfloat16)
        gk = tl.load(GK + d, mask=d < D, other=1).to(tl.bfloat16)
        gamma = tl.where(is_q, gq, gk)
        inv = tl.rsqrt(tl.sum(y * y, axis=1) / D + EPS)
        yn = (y * inv[:, None]).to(tl.bfloat16)
        yn = (yn * gamma[None, :]).to(tl.bfloat16)
        # V does not receive RMSNorm/RoPE and retains the rounded projection.
        tl.store(out_ptr, y.to(tl.bfloat16), mask=out_mask & (head >= HQ + HK))
        tl.store(out_ptr, yn, mask=out_mask & (head < HQ + HK))
        tl.debug_barrier()
        jp = tl.where(d < D // 2, d + D // 2, d - D // 2)
        partner = tl.load(Y + rows[:, None] * n_total + base + jp[None, :],
                          mask=out_mask, other=0).to(tl.bfloat16)
        pos = rows % S
        c = tl.load(COS + pos[:, None] * D + d[None, :],
                    mask=rm[:, None] & (d[None, :] < D), other=1).to(tl.bfloat16)
        s = tl.load(SIN + pos[:, None] * D + d[None, :],
                    mask=rm[:, None] & (d[None, :] < D), other=0).to(tl.bfloat16)
        p0 = (yn * c).to(tl.bfloat16)
        p1 = (partner * s).to(tl.bfloat16)
        rot = tl.where(d[None, :] < D // 2,
                       (p0 - p1).to(tl.bfloat16),
                       (p0 + p1).to(tl.bfloat16))
        tl.store(out_ptr, rot, mask=out_mask & (head < HQ + HK))


def _launch(x: torch.Tensor, w: torch.Tensor, gq: torch.Tensor, gk: torch.Tensor,
            cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor,
            *, b: int, s: int, hq: int, hk: int, d: int,
            norm: bool, q_is: bool, base_head: int, nheads: int) -> None:
    m, k = x.shape
    total = hq + 2 * hk
    # Preserve the historical helper signature.  All candidate/raw calls now
    # use one launch covering Q, K, and V; the legacy arguments are retained
    # so external smoke scripts do not break while the runner migrates.
    _qkv_tiled_kernel[((m + BLOCK_M - 1) // BLOCK_M) * ((total + HEAD_TILE - 1) // HEAD_TILE),](
        x, w, gq, gk, cos, sin, out,
        m, k, s, hq, hk, d, total, norm,
        float(MODEL["rms_norm_eps"]), BLOCK_M, BLOCK_K, HEAD_TILE, d,
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
            norm=True, q_is=True, base_head=0, nheads=hq + 2 * hk)
    return _views(out, b, s, hq, hk, d)


def _custom_raw_projection(case, out):
    x, w, gq, gk, cos, sin = case
    b_s, _ = x.shape; s = cos.shape[0]; b = b_s // s
    hq = MODEL["q_heads"]; hk = MODEL["kv_heads"]; d = MODEL["head_dim"]
    _launch(x, w, gq, gk, cos, sin, out, b=b, s=s, hq=hq, hk=hk, d=d,
            norm=False, q_is=True, base_head=0, nheads=hq + 2 * hk)
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
    close = bool(torch.allclose(a, b, atol=METRIC_ATOL, rtol=METRIC_RTOL))
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


def _anchor(case, out_custom, out_vendor, rounds: int, samples: int,
            anchor_max_ratio: float = 1.01):
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
                    "pass": bool(custom_ms <= anchor_max_ratio * vendor_ms),
                    "max_ratio": anchor_max_ratio}


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
    ap.add_argument("--timing-only", action="store_true",
                    help="retain timing after relaxed gates; never publication eligible")
    ap.add_argument("--correctness-atol", type=float, default=ATOL)
    ap.add_argument("--correctness-rtol", type=float, default=RTOL)
    ap.add_argument("--nll-atol", type=float, default=NLL_ATOL)
    ap.add_argument("--anchor-max-ratio", type=float, default=1.01)
    ap.add_argument("--trace-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="one B=1,S=128,non-causal row; low timing count")
    args = ap.parse_args()
    global METRIC_ATOL, METRIC_RTOL, METRIC_NLL_ATOL
    METRIC_ATOL, METRIC_RTOL, METRIC_NLL_ATOL = (args.correctness_atol,
                                                  args.correctness_rtol,
                                                  args.nll_atol)
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
                     "timing_only": bool(args.timing_only),
                     "correctness_atol": args.correctness_atol,
                     "correctness_rtol": args.correctness_rtol,
                     "nll_atol": args.nll_atol,
                     "anchor_max_ratio": args.anchor_max_ratio,
                     "timing": "paired randomized CUDA-graph replay; vendor SDPA for both arms",
                     "anchor_gate": f"standalone custom QKV projection <= {args.anchor_max_ratio:.6g}x over vendor torch.mm",
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
                            "nll": {"value": cnll, "reference": rnll, "abs_diff": abs(cnll - rnll), "pass": abs(cnll - rnll) <= METRIC_NLL_ATOL}},
                            "vendor": {"Q": _metric(vq, oq), "K": _metric(vk, ok), "V": _metric(vv, ov),
                                       "attention": _metric(vo, oo), "logits": _metric(vlog, rlog),
                                       "top1_match": bool(torch.equal(vlog.argmax(-1), rlog.argmax(-1))),
                                       "nll": {"value": vnll, "reference": rnll, "abs_diff": abs(vnll - rnll), "pass": abs(vnll - rnll) <= METRIC_NLL_ATOL}}}
                        cseed["pass"] = all(cseed["candidate"][k]["pass"] for k in ("Q", "K", "V", "attention", "logits")) and all(cseed["vendor"][k]["pass"] for k in ("Q", "K", "V", "attention", "logits")) and cseed["candidate"]["top1_match"] and cseed["candidate"]["nll"]["pass"] and cseed["vendor"]["top1_match"] and cseed["vendor"]["nll"]["pass"]
                        row["correctness"]["seeds"].append(cseed)
                    row["correctness"]["pass"] = all(x["pass"] for x in row["correctness"]["seeds"])
                    # Standalone anchor is measured on the first case only.
                    case = _make_case(b, s, seeds[0]); raw_c = torch.empty((b * s, n), device="cuda", dtype=DTYPE); raw_v = torch.empty_like(raw_c)
                    row["anchor_timing"], row["anchor_gate"] = _anchor(
                        case, raw_c, raw_v, args.rounds, max(3, args.samples),
                        args.anchor_max_ratio)
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
                    elif args.timing_only:
                        # The timing-only lane measures the eager vendor SDPA
                        # and candidate arms without TorchInductor.  This
                        # avoids compiler CUDA-graph capture state leaking into
                        # the paired timing harness; strict mode still runs
                        # the complete compiler audit below.
                        compile_fn = None
                        row["compiler_evidence"] = {"status": "timing_only_skipped"}
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
                row["status"] = ("timing_only_diagnostic" if args.timing_only
                                  else ("tested_pass" if not reasons else "tested_invalid"))
                rec["rows"].append(row)
                rec["status"] = "running"
                args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
                print(json.dumps({"B": b, "S": s, "causal": causal, "status": row["status"], "gate": row["publication_gate"]}), flush=True)
    rec["status"] = ("timing_only_diagnostic" if args.timing_only else
                      ("tested_pass" if rec["rows"] and all(r.get("status") == "tested_pass" for r in rec["rows"])
                       else "tested_invalid"))
    rec["publication_eligible"] = (not args.timing_only) and rec["status"] == "tested_pass"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(rec, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "publication_eligible": rec["publication_eligible"]}), flush=True)


if __name__ == "__main__":
    main()
