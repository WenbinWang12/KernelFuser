#!/usr/bin/env python3
"""Qwen3-8B prefill check for the RoPE-aware attention fusion.

This is the end-to-end companion to ``e1_rope_publication_v1.py``.  The
projection and output-projection GEMMs are kept in the HuggingFace model; the
only replacement is Q/K RMSNorm + split-half RoPE + attention.  The fused
attention entry point is imported from the isolated runner so the two
experiments cannot silently use different kernels.

The script deliberately validates the full logits, top-1 choices, and NLL
before it records any timings.  A failed correctness gate therefore has no
timing result to accidentally promote.  A row is also marked ineligible when
the candidate cannot be captured in a static CUDA graph or when the compiled
vendor baseline cannot be built.

Example (on the H200 Qwen environment)::

  python e2_qwen3_rope_prefill.py --model /path/Qwen3-8B --out e2_qwen3.json \
      --sizes 128 256 --batches 1 8 32 --rounds 5 --repeats 20

The model path is intentionally a command-line argument; no checkpoint or
credential is stored in this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM
except Exception as exc:  # keep --help useful on a CPU-only workstation
    AutoModelForCausalLM = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = repr(exc)


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import e1_rope_publication_v1 as rope_impl  # noqa: E402


DTYPE = torch.bfloat16
ATOL = 2.5e-2
RTOL = 2.5e-2
TOP1_MIN = 0.999
NLL_ATOL = 2.5e-2


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot() -> dict[str, Any]:
    """Record selected-GPU compute apps without failing a diagnostic run."""
    out: dict[str, Any] = {
        "host": platform.node(),
        "pid": os.getpid(),
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        out["apps"] = [x.strip() for x in p.stdout.splitlines() if x.strip()]
        out["exclusive"] = len(out["apps"]) == 0
    except Exception as exc:
        out["apps_error"] = repr(exc)
        out["exclusive"] = False
    return out


def _metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    d = (candidate.float() - reference.float()).detach()
    den = reference.float().norm().clamp_min(1e-12)
    return {
        "max_abs": float(d.abs().max().item()),
        "relative_l2": float((d.norm() / den).item()),
        "pass": bool(torch.allclose(candidate.float(), reference.float(), atol=ATOL, rtol=RTOL)),
    }


def _nll(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # The final token has no next-token target.
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        ids[:, 1:].reshape(-1), reduction="mean",
    )


def _quality(candidate: torch.Tensor, reference: torch.Tensor, ids: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        ref_top = reference.argmax(dim=-1)
        cand_top = candidate.argmax(dim=-1)
        top_agreement = float((ref_top == cand_top).float().mean().item())
        ref_nll = float(_nll(reference, ids).item())
        cand_nll = float(_nll(candidate, ids).item())
    metric = _metric(candidate, reference)
    metric["top1_agreement"] = top_agreement
    metric["nll_reference"] = ref_nll
    metric["nll_candidate"] = cand_nll
    metric["nll_abs_diff"] = abs(cand_nll - ref_nll)
    metric["pass"] = bool(
        metric["pass"] and top_agreement >= TOP1_MIN and
        abs(cand_nll - ref_nll) <= NLL_ATOL
    )
    return metric


def _attention_patch(model: torch.nn.Module, *, causal: bool) -> list[tuple[Any, Any]]:
    """Patch every Qwen3Attention and return (module, old_forward) pairs."""
    import types

    saved: list[tuple[Any, Any]] = []
    count = 0
    for mod in model.modules():
        if type(mod).__name__ != "Qwen3Attention":
            continue
        old = mod.forward

        def candidate(self, hidden_states, position_embeddings=None,
                      attention_mask=None, past_key_value=None,
                      cache_position=None, **kwargs):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            if position_embeddings is None:
                raise RuntimeError("Qwen3 did not provide position_embeddings")
            cos, sin = position_embeddings
            eps = float(getattr(self.q_norm, "variance_epsilon", 1e-6))
            # This call is the same implementation used by the isolated
            # RoPE experiment.  It includes GQA expansion inside attention.
            out = rope_impl.fused(
                q, k, v, self.q_norm.weight, self.k_norm.weight,
                cos, sin, bool(getattr(self, "is_causal", causal)),
                eps=eps, scale=float(self.scaling), norm_style="qwen3",
            )
            out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            out = self.o_proj(out)
            # Qwen3 4.51 returns exactly two values when no cache is used.
            return out, None

        mod.forward = types.MethodType(candidate, mod)
        saved.append((mod, old))
        count += 1
    if count == 0:
        raise RuntimeError("no Qwen3Attention modules found in checkpoint")
    return saved


def _restore(saved: list[tuple[Any, Any]]) -> None:
    for mod, old in saved:
        mod.forward = old


def _forward(model: torch.nn.Module, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
        return out.logits


def _compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    # Fullgraph is intentional: a graph break would make the vendor baseline
    # incomparable with a native eager model and is recorded as an error.
    return torch.compile(model, fullgraph=True, dynamic=False, mode="max-autotune")


def _capture(fn: Callable[[], torch.Tensor], warmup: int) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(max(3, warmup)):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static = fn()
        graph.replay()
        torch.cuda.synchronize()
    return graph, static


def _timed_pair(base: torch.cuda.CUDAGraph, cand: torch.cuda.CUDAGraph,
                repeats: int, rounds: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    bvals: list[float] = []
    cvals: list[float] = []
    order: list[str] = []

    def measure(g: torch.cuda.CUDAGraph) -> float:
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); z.record(); z.synchronize()
        return float(a.elapsed_time(z))

    for _ in range(rounds):
        for _ in range(repeats):
            if rng.random() < 0.5:
                order.append("baseline,candidate")
                bvals.append(measure(base)); cvals.append(measure(cand))
            else:
                order.append("candidate,baseline")
                cvals.append(measure(cand)); bvals.append(measure(base))
    return {
        "baseline_ms": bvals, "candidate_ms": cvals, "arm_order": order,
        "seed": seed, "baseline_median_ms": statistics.median(bvals),
        "candidate_median_ms": statistics.median(cvals),
        "speedup": statistics.median(bvals) / statistics.median(cvals),
        "repeats": repeats, "rounds": rounds, "cuda_graph": True,
    }


def _load(model_path: str) -> torch.nn.Module:
    if AutoModelForCausalLM is None:
        raise RuntimeError(f"transformers import failed: {_TRANSFORMERS_IMPORT_ERROR}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE, device_map={"": 0},
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def _row(model: torch.nn.Module, b: int, s: int, args: argparse.Namespace) -> dict[str, Any]:
    seed = 9107 + 101 * b + s
    torch.manual_seed(seed)
    vocab = int(model.config.vocab_size)
    ids = torch.randint(0, vocab, (b, s), device="cuda", dtype=torch.long)
    mask = torch.ones((b, s), device="cuda", dtype=torch.long)
    row: dict[str, Any] = {"B": b, "S": s, "causal": True, "seed": seed,
                           "gpu_before": _snapshot(), "status": "running"}

    # Baseline is vendor SDPA from the unmodified Qwen3 implementation.  The
    # compiled graph is built before patching; capture preserves that graph.
    baseline = lambda: _forward(model, ids, mask)
    try:
        with torch.inference_mode():
            ref = baseline()
        compiled = _compiled_model(model)
        compiled_fn = lambda: _forward(compiled, ids, mask)
        with torch.inference_mode():
            comp_ref = compiled_fn()
        row["compiled_baseline_check"] = _quality(comp_ref, ref, ids)
        if not row["compiled_baseline_check"]["pass"]:
            row["status"] = "fail_compiled_baseline_correctness"
            row["correctness"] = {"compiled_baseline": row["compiled_baseline_check"]}
            return row
    except Exception as exc:
        row["status"] = "fail_compiled_baseline_build"
        row["error"] = repr(exc)
        return row

    # Capture the compiled vendor baseline while the model is still
    # unmodified.  ``torch.compile(model)`` wraps the same module object, so
    # capturing it after the attention methods are patched would silently turn
    # this arm into a candidate graph.
    try:
        gb, _ = _capture(compiled_fn, args.warmup)
    except Exception as exc:
        row["status"] = "fail_compiled_baseline_capture"
        row["error"] = repr(exc)
        return row

    saved = _attention_patch(model, causal=True)
    try:
        candidate = lambda: _forward(model, ids, mask)
        with torch.inference_mode():
            got = candidate()
        row["correctness"] = {"candidate_vs_vendor": _quality(got, ref, ids)}
        if not row["correctness"]["candidate_vs_vendor"]["pass"]:
            row["status"] = "fail_candidate_correctness"
            return row
        # The candidate itself must be graph-capturable.  This also prevents
        # recording a timing row when Triton falls back to a Python path.
        gc, _ = _capture(candidate, args.warmup)
        row["timing"] = _timed_pair(gb, gc, args.repeats, args.rounds, seed)
        row["status"] = "pass"
    except Exception as exc:
        row["status"] = "fail_candidate_capture"
        row["error"] = repr(exc)
    finally:
        _restore(saved)
    row["gpu_after"] = _snapshot()
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[128, 256])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="only B=1,S=128")
    args = ap.parse_args()
    if args.smoke:
        args.sizes, args.batches = [128], [1]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen3 prefill")
    torch.cuda.set_device(0)
    rec: dict[str, Any] = {
        "schema": "fusionagent-e2-qwen3-rope-prefill-v1",
        "model_name": "Qwen3-8B",
        "model_path": args.model,
        "scope": "Qwen3-8B causal prefill; native projections/output projection; Q/K RMSNorm+RoPE+attention fused",
        "source_hash": _sha(Path(__file__)),
        "isolated_kernel_source": str((HERE / "e1_rope_publication_v1.py").name),
        "protocol": {
            "S": args.sizes, "B": args.batches, "causal": True,
            "warmup": args.warmup, "repeats": args.repeats, "rounds": args.rounds,
            "baseline": "unmodified Qwen3 attention with vendor SDPA",
            "compiled_baseline": "torch.compile(fullgraph=True,dynamic=False,mode=max-autotune)",
            "candidate": "same imported RoPE-aware Triton fused function",
            "timing": "randomized paired CUDA-graph replay; no timing before correctness",
            "correctness": "full logits max_abs/relative_l2, top1, and next-token NLL",
            "dispatch": "candidate is selected by exact (B,S) shape; failed rows do not dispatch",
        },
        "environment": {"host": platform.node(), "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "model_path_exists": Path(args.model).exists()},
        "rows": [], "status": "running", "publication_eligible": False,
    }
    model = _load(args.model)
    try:
        for s in args.sizes:
            for b in args.batches:
                print(f"Qwen3-8B B={b} S={s}", flush=True)
                row = _row(model, b, s, args)
                rec["rows"].append(row)
                print(json.dumps({k: row.get(k) for k in ("B", "S", "status", "timing", "error")}, default=str), flush=True)
    finally:
        del model
        torch.cuda.empty_cache()
    rec["status"] = "tested_pass" if rec["rows"] and all(x.get("status") == "pass" for x in rec["rows"]) else "tested_with_errors"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.out, "status": rec["status"], "rows": len(rec["rows"])}, indent=2))


if __name__ == "__main__":
    main()
