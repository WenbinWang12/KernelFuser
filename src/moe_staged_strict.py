#!/usr/bin/env python3
"""Strict isolated MoE norm->gate_up staged benchmark.

This is intentionally an expert-local experiment.  It uses the real published
geometry (hidden, expert count, top-k and intermediate width) for nine MoE
architectures, but does not claim routing, dispatch, shared experts, or a
full-checkpoint end-to-end result.  The pre-MoE norm is RMSNorm for every
architecture in this census.  The candidate computes row RMS statistics in a
Triton kernel, then applies the statistics and the gate_up GEMM in a staged
path.  The custom statistics operator has an explicit analytical backward;
all four gradients (input, projection weight, RMS scale, bias) are checked.

The receipt is publication-ready for this isolated scope only when every row
passes three independent seeds, the 20/100 warmup/sample timing contract, and
both Chrome traces.  A negative speed ratio is retained as a valid result.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# The nine E>=64 architectures used by the paper's MoE census.  These values
# are copied from the corresponding public HF configs / the campaign census.
# n_shared is recorded as metadata; the isolated routed-expert site excludes
# shared experts by design.
ARCH = [
    dict(model="Qwen3-30B-A3B", hidden=2048, top_k=8, experts=128,
         expert_inter=768, n_shared=0),
    dict(model="Qwen3-235B-A22B", hidden=4096, top_k=8, experts=128,
         expert_inter=1536, n_shared=0),
    dict(model="DeepSeek-V3", hidden=7168, top_k=8, experts=256,
         expert_inter=2048, n_shared=1),
    dict(model="DeepSeek-V2", hidden=5120, top_k=6, experts=160,
         expert_inter=1536, n_shared=2),
    dict(model="OLMoE-1B-7B", hidden=2048, top_k=8, experts=64,
         expert_inter=1024, n_shared=0),
    dict(model="GPT-OSS-120B", hidden=2880, top_k=4, experts=128,
         expert_inter=2880, n_shared=0),
    dict(model="Qwen2-57B-A14B", hidden=3584, top_k=8, experts=64,
         expert_inter=2560, n_shared=1),
    dict(model="Qwen3-Next-80B-A3B", hidden=2048, top_k=10, experts=512,
         expert_inter=512, n_shared=1),
    dict(model="Llama-4-Maverick", hidden=5120, top_k=1, experts=128,
         expert_inter=8192, n_shared=1),
]
TOKENS = (2048, 8192, 32768, 131072)
SEEDS = (20261009, 20261010, 20261011)
EPS = 1.0e-5
ATOL = 2.0e-2
RTOL = 2.0e-2


def balanced_m(tokens: int, top_k: int, experts: int) -> int:
    return max(16, int(round(tokens * top_k / experts)))


@triton.jit
def _rms_stats_kernel(x_ptr, r_ptr, M, K, sxm, sxk, sr, EPS: tl.constexpr,
                      BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x = tl.load(x_ptr + row * sxm + cols * sxk, mask=mask, other=0.0,
                eviction_policy="evict_last").to(tl.float32)
    ss = tl.sum(x * x, axis=0)
    rstd = 1.0 / tl.sqrt(ss / K + EPS)
    tl.store(r_ptr + row * sr, rstd)


def rms_stats(x: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    M, K = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    bk = triton.next_power_of_2(K)
    _rms_stats_kernel[(M,)](x, out, M, K, x.stride(0), x.stride(1),
                             out.stride(0), EPS=eps, BLOCK_K=bk,
                             num_warps=8 if K >= 2048 else 4)
    return out


class RMSStats(torch.autograd.Function):
    """Triton forward with an explicit, complete RMS derivative."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float):
        r = rms_stats(x, eps)
        ctx.save_for_backward(x, r)
        ctx.K = x.shape[-1]
        ctx.eps = float(eps)
        return r

    @staticmethod
    def backward(ctx, grad_r):
        x, r = ctx.saved_tensors
        # r = (mean(x^2)+eps)^(-1/2); dr/dx = -x*r^3/K.
        gx = grad_r.to(torch.float32) * (-x.to(torch.float32) * r.pow(3) / ctx.K)
        return gx.to(x.dtype), None


def candidate(x, w, gamma, bias):
    r = RMSStats.apply(x, EPS)
    z = x * r.to(x.dtype) * gamma
    return torch.matmul(z, w.t()) + bias


def baseline(x, w, gamma, bias):
    z = F.rms_norm(x, (x.shape[-1],), weight=gamma, eps=EPS)
    return torch.matmul(z, w.t()) + bias


def close(a, b):
    aa, bb = a.detach().float(), b.detach().float()
    den = bb.abs().amax().clamp_min(1.0e-6)
    err = float((aa - bb).abs().amax().cpu() / den.cpu())
    return {"max_abs": float((aa - bb).abs().amax().cpu()),
            "relative": err, "allclose": bool(torch.allclose(aa, bb,
                                                                atol=ATOL,
                                                                rtol=RTOL))}


def one_correctness(M, K, N, seed):
    torch.manual_seed(seed)
    x0 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w0 = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * (K ** -0.5)
    g0 = (torch.randn((K,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0)
    b0 = torch.randn((N,), device="cuda", dtype=torch.bfloat16) * 0.1

    def run(fn):
        x = x0.detach().clone().requires_grad_(True)
        w = w0.detach().clone().requires_grad_(True)
        g = g0.detach().clone().requires_grad_(True)
        b = b0.detach().clone().requires_grad_(True)
        y = fn(x, w, g, b)
        loss = (y.float() * y.float()).mean()
        grads = torch.autograd.grad(loss, (x, w, g, b), create_graph=False)
        return y.detach(), grads, float(loss.detach().cpu())

    ref, rg, rl = run(baseline)
    got, gg, gl = run(candidate)
    names = ("dinput", "dweight", "dgamma", "dbias")
    checks = {"forward": close(got, ref)}
    checks.update({n: close(g, r) for n, g, r in zip(names, gg, rg)})
    return {"seed": seed, "loss_baseline": rl, "loss_candidate": gl,
            "checks": checks,
            "pass": all(v["allclose"] for v in checks.values()) and
                    all(torch.isfinite(g).all().item() for g in gg)}


def _capture(fn, x, w, g, b, backward):
    # Static tensors and a dedicated graph make every timed replay identical.
    sx = x.detach().clone().requires_grad_(backward)
    sw = w.detach().clone().requires_grad_(backward)
    sg = g.detach().clone().requires_grad_(backward)
    sb = b.detach().clone().requires_grad_(backward)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        for _ in range(3):
            if backward:
                y = fn(sx, sw, sg, sb)
                (y.float() * y.float()).mean().backward()
                for p in (sx, sw, sg, sb):
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                fn(sx, sw, sg, sb)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            if backward:
                y = fn(sx, sw, sg, sb)
                (y.float() * y.float()).mean().backward()
                for p in (sx, sw, sg, sb):
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                fn(sx, sw, sg, sb)
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def graph_times(g0, g1, warmup=20, samples=100):
    # Warmup and then interleave A/B, alternating first arm to suppress drift.
    for _ in range(warmup):
        g0.replay(); g1.replay()
    torch.cuda.synchronize()
    vals = [[], []]
    for i in range(samples):
        order = (0, 1) if i % 2 == 0 else (1, 0)
        for j in order:
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record(); (g0 if j == 0 else g1).replay(); en.record(); en.synchronize()
            vals[j].append(float(st.elapsed_time(en)))
    return {"baseline_ms": statistics.median(vals[0]),
            "candidate_ms": statistics.median(vals[1]),
            "baseline_samples_ms": vals[0], "candidate_samples_ms": vals[1],
            "ratio_baseline_over_candidate": statistics.median(vals[0]) /
                                             max(statistics.median(vals[1]), 1.0e-9),
            "warmup": warmup, "samples": samples, "interleaved": True,
            "cuda_graph": True}


def direct_backward_times(fn0, fn1, x, w, g, b, warmup=20, samples=100):
    """Interleaved full backward timing with fresh native autograd graphs.

    CUDA graph replay is used for forward timing above.  PyTorch 2.11's graph
    allocator cannot safely retain two independent backward graphs when one
    contains a user Triton autograd Function (it aliases the saved RMS buffer
    after the forward graph is destroyed).  Fresh graphs preserve the actual
    backward work and make the failure mode explicit in the receipt rather
    than silently timing a corrupted replay.
    """
    def step(fn):
        xx = x.detach().clone().requires_grad_(True)
        ww = w.detach().clone().requires_grad_(True)
        gg = g.detach().clone().requires_grad_(True)
        bb = b.detach().clone().requires_grad_(True)
        y = fn(xx, ww, gg, bb)
        loss = (y.float() * y.float()).mean()
        torch.autograd.grad(loss, (xx, ww, gg, bb), create_graph=False)
    for _ in range(warmup):
        step(fn0); step(fn1)
    torch.cuda.synchronize(); vals = [[], []]
    for i in range(samples):
        order = (0, 1) if i % 2 == 0 else (1, 0)
        for j in order:
            st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
            st.record(); step(fn0 if j == 0 else fn1); en.record(); en.synchronize()
            vals[j].append(float(st.elapsed_time(en)))
    return {"baseline_ms": statistics.median(vals[0]),
            "candidate_ms": statistics.median(vals[1]),
            "baseline_samples_ms": vals[0], "candidate_samples_ms": vals[1],
            "ratio_baseline_over_candidate": statistics.median(vals[0]) /
                                             max(statistics.median(vals[1]), 1.0e-9),
            "warmup": warmup, "samples": samples, "interleaved": True,
            "cuda_graph": False,
            "graph_note": "fresh native autograd graphs; see allocator note"}


def trace(fn, x, w, g, b, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    # The profiler export is the Chrome trace artifact; gzip keeps NFS usage low.
    plain = out.with_suffix("")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                               record_shapes=True, profile_memory=False) as p:
        for _ in range(3):
            xx = x.detach().clone().requires_grad_(True)
            ww = w.detach().clone().requires_grad_(True)
            gg = g.detach().clone().requires_grad_(True)
            bb = b.detach().clone().requires_grad_(True)
            y = fn(xx, ww, gg, bb)
            (y.float() * y.float()).mean().backward()
        torch.cuda.synchronize()
    p.export_chrome_trace(str(plain))
    with open(plain, "rb") as src, gzip.open(out, "wb") as dst:
        dst.write(src.read())
    plain.unlink(missing_ok=True)


def run_row(spec, tokens, outdir):
    M = balanced_m(tokens, spec["top_k"], spec["experts"])
    K, N = spec["hidden"], 2 * spec["expert_inter"]
    row = {"model": spec["model"], "tokens": tokens, "per_expert_M": M,
           "K": K, "N_gate_up": N, "top_k": spec["top_k"],
           "n_experts": spec["experts"], "expert_inter": spec["expert_inter"],
           "n_shared_experts": spec["n_shared"], "norm_type": "RMSNorm",
           "routing": "not measured; balanced per-expert geometry only",
           "status": "started"}
    row["correctness"] = [one_correctness(M, K, N, s) for s in SEEDS]
    row["correctness_pass"] = all(x["pass"] for x in row["correctness"])
    # Use the last seed for timing and traces.
    torch.manual_seed(SEEDS[-1])
    x = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * (K ** -0.5)
    g = torch.randn((K,), device="cuda", dtype=torch.bfloat16) * 0.1 + 1.0
    b = torch.randn((N,), device="cuda", dtype=torch.bfloat16) * 0.1
    torch.cuda.synchronize()
    try:
        gf0 = _capture(baseline, x, w, g, b, False)
        gf1 = _capture(candidate, x, w, g, b, False)
        row["forward_timing"] = graph_times(gf0, gf1)
        # Do not leave forward graph pools live while capturing the backward
        # graphs.  Hopper's graph allocator can alias the temporary rstd/output
        # buffers across pools; retaining both pairs caused an asynchronous
        # illegal-address on replay for some BF16 geometries.
        del gf0, gf1
        torch.cuda.synchronize()
        row["forward_backward_timing"] = direct_backward_times(
            baseline, candidate, x, w, g, b)
    except Exception as e:
        row["timing_error"] = repr(e)
        row["status"] = "timing_error"
        return row
    trace(fn=baseline, x=x, w=w, g=g, b=b,
          out=outdir / f"{spec['model']}_{tokens}_baseline.chrome_trace.json.gz")
    trace(fn=candidate, x=x, w=w, g=g, b=b,
          out=outdir / f"{spec['model']}_{tokens}_candidate.chrome_trace.json.gz")
    row["status"] = "tested_pass" if row["correctness_pass"] else "tested_invalid"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trace-dir", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated model names")
    ap.add_argument("--tokens", default=",")
    a = ap.parse_args()
    models = ARCH
    if a.only:
        want = {x.strip().lower() for x in a.only.split(",")}
        models = [x for x in models if x["model"].lower() in want]
    toks = TOKENS if not a.tokens or a.tokens == "," else tuple(int(x) for x in a.tokens.split(","))
    src_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rec = {"suite_id": "fusionagent-moe-staged-rmsnorm-publication-v1",
           "status": "started", "device": torch.cuda.get_device_name(0),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
           "torch": torch.__version__, "triton": triton.__version__,
           "source_sha256": src_hash, "eps": EPS, "atol": ATOL, "rtol": RTOL,
           "seeds": list(SEEDS), "warmup": 20, "samples": 100,
           "architectures": [x["model"] for x in models], "tokens": list(toks),
           "scope": {"level": "kernel", "operation": "RMSNorm -> routed expert gate_up GEMM",
                     "weights": "synthetic frozen expert-local tensors with public geometry",
                     "routing": "excluded; balanced per-expert M only",
                     "shared_experts": "excluded", "full_model": False,
                     "backward": "explicit RMSStats analytical backward plus native matmul autograd",
                     "claim_boundary": "isolated expert-local staged kernel; no full-model speedup claim"},
           "rows": []}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.trace_dir.mkdir(parents=True, exist_ok=True)
    for s in models:
        for t in toks:
            print(f"RUN {s['model']} tokens={t} M={balanced_m(t,s['top_k'],s['experts'])}", flush=True)
            row = run_row(s, t, a.trace_dir); rec["rows"].append(row)
            rec["completed_rows"] = len(rec["rows"])
            rec["status"] = "partial"
            a.out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = bool(rec["status"] == "tested_pass")
    rec["development_only"] = False
    a.out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("status", "publication_eligible", "completed_rows", "source_sha256")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
