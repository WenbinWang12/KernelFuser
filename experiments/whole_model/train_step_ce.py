#!/usr/bin/env python3
"""Whole-model training step of a released causal LM with the fused loss kernel.

reference  model(input_ids, labels=input_ids).loss        (HuggingFace: logits.float() + cross-entropy)
candidate  the same model body and the same vendor GEMM for the head, then
           lmhead_kernels.fused_cross_entropy: one kernel returns the loss and the
           gradient with respect to the logits; the rest of the backward pass is unchanged.

Per shape (B sequences of S tokens, BF16 weights, AdamW):
  1. one step from identical weights: the loss and the gradient of every trainable parameter
     are computed by the reference, by the candidate, and by an FP32 oracle (the same model
     and loss in single precision).  Gate, version 2:
       |loss(candidate) - loss(reference)| <= 1e-3 relative, and for every parameter p
       err(candidate, p) <= max(1.5 x err(reference, p), 5e-3),
     where err(x, p) is the relative L2 distance of x's gradient of p from the oracle's.
     That is, the candidate may not be further from single precision than the unfused BF16
     model is, beyond the stated margin.
     History: version 1 required a relative L2 difference of at most 2e-2 between candidate
     and reference for every parameter.  On 2026-11-21 its first run (Qwen3-0.6B, 2x256, A800)
     failed with 0.0213 on model.layers.0.self_attn.k_norm.weight (128 elements) while the
     losses agreed to 7e-8; the reference repeats bit-identically, so version 1 measured
     sensitivity of deep BF16 backpropagation to any rounding change and had no noise scale.
     Version 2 was written before any training-step timing existed.  The version-1 quantity
     is still recorded in every receipt (candidate_vs_reference).
  2. --steps optimizer steps on the same batches from the same initial weights;
     the per-step losses of both runs are recorded.
     A row that fails the gate is reported as invalid; its timing is still recorded and
     marked timing_is_diagnostic_only.
  3. paired step timing: the two runs alternate in random order, CUDA events around
     forward + backward + optimizer step.  --compile adds a second pair in which the
     reference is torch.compile(model) and the candidate compiles the model body.

Inputs are seeded random token ids unless --data names a text file.

Script version 3 (2026-11-22; train_step_ce_v2_snapshot.py is the version before).  The gate is
unchanged (version 2).  What was added, all before any run of this version:
  * --data FILE: every batch is B windows of S tokens of that file, tokenized with the tokenizer
    of the checkpoint.  Training windows are drawn with the row seed from the first 95% of the
    token stream; the last 5% is held out.  The SHA-256 of the file and the token count are recorded.
  * held-out loss (reference loss definition, no gradient) of the initial weights and of both
    models after the optimizer steps of the first measurement, as mean loss and perplexity.
  * --retest: the timing is repeated on fresh copies of the checkpoint with other windows; a row
    is kept only if the candidate wins by more than 1% in both measurements.
  * --aa: one more pair in which both arms run the reference loss (noise scale of the protocol).
  * --optimizer adamw|sgd and --grad-device cuda|cpu (for checkpoints whose FP32 gradients do
    not fit on the GPU next to the model; the compared numbers are the same).
  * every measurement starts from fresh copies of the checkpoint (version 2 reused the model of
    the gate for the first pair).
  * --lean: one copy of the checkpoint and one optimizer in memory at a time, for checkpoints of
    which two training copies do not fit on one GPU.  Timing: on every batch both arms take an
    optimizer step on the same model, in random order (the weights move twice per batch; the
    step time does not depend on their values).  The comparable losses come from a separate
    phase: the reference and the candidate are each run alone from the checkpoint on the same
    batches (per-step losses, held-out loss after the last step); with --aa the reference is
    run a second time, which gives the run-to-run difference of the trajectory itself.
  * --oracle-device cpu: the FP32 oracle of the gate runs on the CPU (same model, same loss,
    same precision), for checkpoints whose FP32 gradients do not fit on the GPU.
  * two-arm record (added 2026-11-22 after the smoke runs manual/smoke3/*, in which gate version 2
    failed on text input for Qwen3-0.6B at 2x256: 1 of 310 parameters, a 128-element norm weight,
    candidate 0.072 against a limit of 0.064, while the summed distances to the oracle were 0.0504
    and 0.0495; no tier run and no compiled-reference gradient had been seen).  With --compile the
    gradient of the compiled reference (torch.compile of the whole model, the second PyTorch arm
    of the timing) is computed as well, and "two_arm" repeats the per-parameter rule with the
    worse of the two PyTorch arms as the yardstick,
        err(candidate, p) <= max(1.5 x max(err(eager, p), err(compiled, p)), 5e-3),
    which is the form of the decode gate (e2e_decode.py).  In --lean mode the compiled reference
    is also run alone for all steps, so the trajectory and held-out differences of the candidate
    can be read against those between the two PyTorch arms ("multi_step").  These are records for
    the authors.  The verdict of record (status, kept) is still gate version 2; kept_two_arm is
    stored next to it.
  * memory (2026-11-22, after the first tier stages were killed by the container's memory limit of
    196 GiB: the gate kept four FP32 copies of all gradients).  The gate now streams: the FP32
    oracle runs first and only its gradients are kept; each BF16 arm is reduced to per-parameter
    norms right after its backward pass, and one BF16 copy of the reference gradient is kept for
    the candidate-against-reference numbers.  The quantities and the rule are those of version 2.
    Devices are chosen from the parameter count (recorded as gate_memory); --mem-lock DIR lets
    only one stage at a time hold a large CPU-side oracle.
  * CPU threads and a second statistic (2026-11-22, after the first tier row; the container has a
    CPU quota of 24 cores for four concurrent stages, and step times drifted within a pair when
    another stage used many threads).  The process now uses --cpu-threads threads (default 4) and
    a CPU-side oracle --oracle-threads (default 16).  Every pair also records the median and the
    quartiles of the per-step ratio reference/candidate (the two arms of a step run back to back,
    so this ratio is insensitive to drift).  The verdict still uses the ratio of the two medians,
    as before; the paired ratio is a record.
  * --optimizer muon (2026-11-22, requested by the authors): Muon for the hidden weight matrices
    (momentum 0.95 with Nesterov, five quintic Newton-Schulz iterations with coefficients
    3.4445, -4.7750, 2.0315, update scaled by sqrt(max(1, rows/columns)), learning rate
    --muon-lr), and AdamW with --lr for the embedding, the LM head and all one-dimensional
    parameters, as in the reference implementation of Muon.  The learning rate for fine-tuning
    a released checkpoint is our choice and is not tuned; the step time does not depend on it.
"""
from __future__ import annotations

import argparse, copy, gc, hashlib, json, math, os, platform, random, shutil, statistics, sys, time, traceback
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import lmhead_kernels as K

WIN = 1.01
GATE_VERSION = 2
LOSS_REL = 1e-3
GRAD_FACTOR = 1.5
GRAD_FLOOR = 5e-3
GRAD_REL_L2_V1 = 2e-2


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(model_dir, device, dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=dtype, local_files_only=True)
    m.to(device); m.train()
    if hasattr(m.config, "use_cache"):
        m.config.use_cache = False
    if getattr(m.lm_head, "bias", None) is not None:
        raise ValueError("this script assumes a bias-free LM head")
    return m


def ref_loss(model, ids):
    return model(input_ids=ids, labels=ids, use_cache=False).loss


def cand_loss(model, ids):
    hs = model.model(input_ids=ids, use_cache=False).last_hidden_state          # [B, S, K], final norm applied
    labels = F.pad(ids, (0, 1), value=K.IGNORE)[:, 1:].reshape(-1)              # next-token targets, last position ignored
    logits = F.linear(hs.reshape(-1, hs.shape[-1]), model.lm_head.weight)       # vendor GEMM, BF16 logits
    return K.fused_cross_entropy(logits, labels, 1.0 / (ids.shape[0] * (ids.shape[1] - 1)))


MEM_LOCK_GIB = 20.0


def placement(P, args):
    """Where the FP32 oracle runs and where its gradients (go) and the BF16 reference gradients (gr) are kept."""
    G = 2 ** 30
    total = torch.cuda.get_device_properties(0).total_memory
    bf, fp = 2 * P, 4 * P
    oracle = "cuda" if 2 * fp <= 0.55 * total else "cpu"          # FP32 weights + gradients, room for activations
    if args.oracle_device == "cpu":
        oracle = "cpu"
    if (2 * bf + fp + bf) * 1.08 + 6 * G < total:                 # BF16 model + gradients, go and gr on the GPU
        go, gr = "cuda", "cuda"
    elif (2 * bf + bf) * 1.08 + 10 * G < total:
        go, gr = "cpu", "cuda"
    else:
        go, gr = "cpu", "cpu"
    if args.grad_device == "cpu":
        go, gr = "cpu", "cpu"
    cpu = max(2 * fp if oracle == "cpu" else 0, (fp if go == "cpu" else 0) + (bf if gr == "cpu" else 0))
    return {"oracle": oracle, "oracle_gradients": go, "reference_gradients": gr, "cpu_gib_estimate": round(cpu / G, 1)}


class MemLock:
    """Directory lock: one holder at a time; a lock whose holder died is taken over."""
    def __init__(self, path):
        self.path = Path(path)

    def acquire(self):
        t0 = time.time()
        while True:
            try:
                self.path.mkdir()
                (self.path / "pid").write_text(str(os.getpid()))
                return round(time.time() - t0, 1)
            except FileExistsError:
                stale = False
                try:
                    os.kill(int((self.path / "pid").read_text()), 0)
                except ProcessLookupError:
                    stale = True
                except (ValueError, OSError):
                    try:
                        stale = time.time() - self.path.stat().st_mtime > 120
                    except OSError:
                        stale = False
                if stale:
                    shutil.rmtree(self.path, ignore_errors=True)
                else:
                    time.sleep(5)

    def release(self):
        shutil.rmtree(self.path, ignore_errors=True)


def arm(model, loss_fn, ids, go, gr, keep=None):
    """One backward pass, reduced at once to per-parameter numbers.
    go: FP32 oracle gradients or None; gr: BF16 reference gradients or None; keep: device on which this
    arm's gradients are returned (BF16), or None."""
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, ids)
    loss.backward()
    st, kept = {}, ({} if keep else None)
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        e = {"n2": float(g.pow(2).sum())}
        if go is not None and n in go:
            o = go[n].to(g.device)
            e["err_orc"] = float((g - o).norm() / o.norm().clamp_min(1e-30))
            e["d2_orc"] = float((g - o).pow(2).sum()); e["o2"] = float(o.pow(2).sum()); e["elements"] = int(o.numel())
            del o
        if gr is not None and n in gr:
            r = gr[n].to(g.device).float()
            e["rel_ref"] = float((g - r).norm() / r.norm().clamp_min(1e-30))
            e["d2_ref"] = float((g - r).pow(2).sum())
            del r
        st[n] = e
        if keep:
            kept[n] = p.grad.detach().clone() if keep == "cuda" else p.grad.detach().to(keep)
        del g
    model.zero_grad(set_to_none=True)
    return float(loss), st, kept


_TOKENS = {}
HOLDOUT = 0.05


def text_tokens(model_dir, data_path):
    """Token ids of the whole file (cached per checkpoint) and the facts that identify the data."""
    key = (str(model_dir), str(data_path))
    if key not in _TOKENS:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        raw = Path(data_path).read_bytes()
        ids = tok(raw.decode("utf-8"), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        _TOKENS[key] = (ids, {"file": Path(data_path).name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                              "tokens": int(ids.numel()), "tokenizer": Path(model_dir).name})
    return _TOKENS[key]


def text_batches(ids, B, S, total, seed):
    """total batches of B windows of S tokens, drawn with the seed from the first 95% of the stream."""
    split = int(ids.numel() * (1.0 - HOLDOUT))
    if split - S <= 0:
        raise ValueError("text too short for this sequence length")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(0, split - S, (total * B,), generator=gen)
    return [torch.stack([ids[int(a):int(a) + S] for a in starts[i * B:(i + 1) * B]]).cuda() for i in range(total)]


def heldout_batches(ids, B, S, count):
    """count batches of consecutive, non-overlapping windows from the last 5% of the stream."""
    pos = int(ids.numel() * (1.0 - HOLDOUT))
    out = []
    while len(out) < count and pos + B * S <= ids.numel():
        out.append(ids[pos:pos + B * S].reshape(B, S).cuda()); pos += B * S
    return out


def heldout_loss(model, batches):
    if not batches:
        return None
    with torch.no_grad():
        v = [float(ref_loss(model, ids)) for ids in batches]
    m = sum(v) / len(v)
    return {"batches": len(v), "mean_loss": m, "perplexity": math.exp(m)}


def cand_loss_with(body, head_weight, ids):
    """cand_loss with an explicit (for example compiled) model body."""
    hs = body(input_ids=ids, use_cache=False).last_hidden_state
    labels = F.pad(ids, (0, 1), value=K.IGNORE)[:, 1:].reshape(-1)
    logits = F.linear(hs.reshape(-1, hs.shape[-1]), head_weight)
    return K.fused_cross_entropy(logits, labels, 1.0 / (ids.shape[0] * (ids.shape[1] - 1)))


def _orthogonalize(g, steps=5):
    """Quintic Newton-Schulz iteration of Muon: an approximation of U V^T for g = U S V^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.to(torch.bfloat16)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.t()
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        A = x @ x.t()
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    return x.t() if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                u = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                u = _orthogonalize(u, group["ns_steps"])
                p.add_(u.to(p.dtype), alpha=-group["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


class Optimizers:
    """Several optimizers stepped together (Muon for the hidden matrices, AdamW for the rest)."""
    def __init__(self, *opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def make_opt(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)
    if args.optimizer == "muon":
        hidden, rest, seen = [], [], set()
        for n, p in model.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            (hidden if p.ndim == 2 and "embed" not in n and "lm_head" not in n else rest).append(p)
        return Optimizers(Muon(hidden, lr=args.muon_lr), torch.optim.AdamW(rest, lr=args.lr, weight_decay=0.0))
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)


def verdict(t):
    """Fastest reference among the pairs against the fastest candidate (the rule of version 2)."""
    pairs = {k: v for k, v in t.items() if k != "aa"}
    fastest = min(pairs, key=lambda n: pairs[n]["reference_p50_ms"])
    return fastest, pairs[fastest]["reference_p50_ms"] / min(x["candidate_p50_ms"] for x in pairs.values())


def measure(model_dir, batches, args, seed, row, tag, heldout=None):
    """Paired step timing from fresh copies of the checkpoint: eager pair, optional A/A pair, optional compiled pair."""
    out = {}
    a = load(model_dir, "cuda"); b = copy.deepcopy(a)
    out["eager"] = paired_steps(step_fn(a, ref_loss, make_opt(a, args)), step_fn(b, cand_loss, make_opt(b, args)),
                                batches, args.warmup, seed + 1)
    if heldout is not None:
        row["heldout"]["reference_after"] = heldout_loss(a, heldout)
        row["heldout"]["candidate_after"] = heldout_loss(b, heldout)
        row["heldout"]["optimizer_steps"] = len(batches)
    del a, b; gc.collect(); torch.cuda.empty_cache()
    if args.aa and tag == "initial":
        a = load(model_dir, "cuda"); b = copy.deepcopy(a)
        out["aa"] = paired_steps(step_fn(a, ref_loss, make_opt(a, args)), step_fn(b, ref_loss, make_opt(b, args)),
                                 batches, args.warmup, seed + 3)
        del a, b; gc.collect(); torch.cuda.empty_cache()
    if args.compile:
        try:
            torch._dynamo.reset()
            a = load(model_dir, "cuda"); b = copy.deepcopy(a)
            a_c = torch.compile(a, dynamic=False)
            b.model = torch.compile(b.model, dynamic=False)
            ra = step_fn(a, lambda m, ids: a_c(input_ids=ids, labels=ids, use_cache=False).loss, make_opt(a, args))
            rb = step_fn(b, cand_loss, make_opt(b, args))
            t0 = time.time()
            for ids in batches[:3]:              # compile outside the timed steps
                ra(ids); rb(ids)
            torch.cuda.synchronize()
            row.setdefault("compile_seconds", {})[tag] = round(time.time() - t0, 1)
            out["compiled"] = paired_steps(ra, rb, batches, args.warmup, seed + 2)
            del a, b, a_c, ra, rb
        except Exception as exc:
            row["errors"]["compiled_" + tag] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
        gc.collect(); torch.cuda.empty_cache()
    return out


def measure_lean(model_dir, batches, args, seed, row, tag, heldout=None):
    """Paired step timing with one copy of the checkpoint and one optimizer (see --lean in the docstring).
    The losses recorded here are not comparable between the arms; trajectories() provides those."""
    out = {}
    m = load(model_dir, "cuda"); opt = make_opt(m, args)
    out["eager"] = paired_steps(step_fn(m, ref_loss, opt), step_fn(m, cand_loss, opt), batches, args.warmup, seed + 1)
    if args.aa and tag == "initial":
        out["aa"] = paired_steps(step_fn(m, ref_loss, opt), step_fn(m, ref_loss, opt), batches, args.warmup, seed + 3)
    if args.compile:
        try:
            torch._dynamo.reset()
            m_c = torch.compile(m, dynamic=False)
            body_c = torch.compile(m.model, dynamic=False)
            ra = step_fn(m, lambda mm, ids: m_c(input_ids=ids, labels=ids, use_cache=False).loss, opt)
            rb = step_fn(m, lambda mm, ids: cand_loss_with(body_c, mm.lm_head.weight, ids), opt)
            t0 = time.time()
            for ids in batches[:3]:              # compile outside the timed steps
                ra(ids); rb(ids)
            torch.cuda.synchronize()
            row.setdefault("compile_seconds", {})[tag] = round(time.time() - t0, 1)
            out["compiled"] = paired_steps(ra, rb, batches, args.warmup, seed + 2)
            del m_c, body_c, ra, rb
        except Exception as exc:
            row["errors"]["compiled_" + tag] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
    for v in out.values():
        v["shared_weights"] = True
    del m, opt; gc.collect(); torch.cuda.empty_cache()
    return out


def trajectories(model_dir, batches, args, heldout, row):
    """Reference and candidate, each alone from the checkpoint on the same batches (one copy in memory)."""
    out = {"steps": len(batches)}
    runs = [("reference", ref_loss), ("candidate", cand_loss)] + ([("reference_rerun", ref_loss)] if args.aa else [])
    if args.compile:
        runs.append(("reference_compiled", None))
    for name, fn in runs:
        m = load(model_dir, "cuda")
        if name == "reference_compiled":
            try:
                torch._dynamo.reset()
                m_c = torch.compile(m, dynamic=False)
                fn = lambda mm, ids, m_c=m_c: m_c(input_ids=ids, labels=ids, use_cache=False).loss
                run = step_fn(m, fn, make_opt(m, args))
                out[name + "_losses"] = [float(run(ids)) for ids in batches]
            except Exception as exc:
                row["errors"]["trajectory_compiled"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
                del m; gc.collect(); torch.cuda.empty_cache(); continue
        else:
            run = step_fn(m, fn, make_opt(m, args))
            out[name + "_losses"] = [float(run(ids)) for ids in batches]
        if heldout is not None and name != "reference_rerun":
            row["heldout"][name + "_after"] = heldout_loss(m, heldout)
            row["heldout"]["optimizer_steps"] = len(batches)
        del m, run; gc.collect(); torch.cuda.empty_cache()
    diff = lambda a, b: max(abs(x - y) for x, y in zip(out[a], out[b]))
    out["loss_max_abs_diff"] = diff("reference_losses", "candidate_losses")
    if args.aa:
        out["reference_rerun_loss_max_abs_diff"] = diff("reference_losses", "reference_rerun_losses")
    if "reference_compiled_losses" in out:
        out["reference_compiled_loss_max_abs_diff"] = diff("reference_losses", "reference_compiled_losses")
        h = row.get("heldout") or {}
        if h.get("reference_after") and h.get("candidate_after") and h.get("reference_compiled_after"):
            r0 = h["reference_after"]["mean_loss"]
            dc = abs(h["candidate_after"]["mean_loss"] - r0); d2 = abs(h["reference_compiled_after"]["mean_loss"] - r0)
            # record: the candidate against the difference between the two PyTorch arms (same factor and
            # the loss tolerance of the one-step gate as the floor)
            out["multi_step"] = {"heldout_abs_diff_candidate": dc, "heldout_abs_diff_compiled_reference": d2,
                                 "heldout_limit": max(GRAD_FACTOR * d2, LOSS_REL * abs(r0)),
                                 "heldout_within": bool(dc <= max(GRAD_FACTOR * d2, LOSS_REL * abs(r0))),
                                 "trajectory_limit": max(GRAD_FACTOR * out["reference_compiled_loss_max_abs_diff"],
                                                         LOSS_REL * abs(out["reference_losses"][0])),
                                 "trajectory_within": bool(out["loss_max_abs_diff"] <= max(
                                     GRAD_FACTOR * out["reference_compiled_loss_max_abs_diff"],
                                     LOSS_REL * abs(out["reference_losses"][0])))}
    return out


def compare(s_a, s_ref):
    """Per-parameter relative L2 difference of an arm from the reference (from the numbers of arm())."""
    worst, worst_name, missing, num = 0.0, None, 0, 0.0
    for n in s_ref:
        a = s_a.get(n)
        if a is None or "rel_ref" not in a:
            missing += 1; continue
        if a["rel_ref"] > worst:
            worst, worst_name = a["rel_ref"], n
        num += a["d2_ref"]
    den = sum(e["n2"] for e in s_ref.values())
    return {"parameters": len(s_ref), "missing": missing, "max_rel_l2": worst, "max_rel_l2_parameter": worst_name,
            "all_parameters_rel_l2": (num / max(den, 1e-30)) ** 0.5}


def oracle_gate(names, s_cand, s_ref, s_refc=None):
    """Gate version 2: per-parameter distance to the FP32 oracle, candidate against reference.
    names: parameters that have an oracle gradient.  With s_refc (compiled reference) the two-arm record
    is added (see the docstring)."""
    worst = {"ratio": 0.0}; worst2 = {"ratio": 0.0}
    fails = 0; fails2 = 0; refc_fails = 0
    num = {"cand_orc": 0.0, "ref_orc": 0.0, "cand_ref": 0.0, "refc_orc": 0.0, "refc_ref": 0.0}; den = {"orc": 0.0, "ref": 0.0}
    for n in names:
        c, r = s_cand.get(n), s_ref.get(n)
        if c is None or r is None or "err_orc" not in c or "err_orc" not in r:
            fails += 1; fails2 += 1; continue
        ec, er = c["err_orc"], r["err_orc"]
        limit = max(GRAD_FACTOR * er, GRAD_FLOOR)
        if ec > limit:
            fails += 1
        if ec / limit > worst["ratio"]:
            worst = {"ratio": ec / limit, "parameter": n, "candidate_err": ec, "reference_err": er, "limit": limit,
                     "elements": r["elements"]}
        num["cand_orc"] += c["d2_orc"]; num["ref_orc"] += r["d2_orc"]
        num["cand_ref"] += c.get("d2_ref", 0.0); den["orc"] += r["o2"]; den["ref"] += r["n2"]
        if s_refc is not None:
            x = s_refc.get(n)
            if x is None or "err_orc" not in x:
                fails2 += 1; continue
            e2 = x["err_orc"]
            limit2 = max(GRAD_FACTOR * max(er, e2), GRAD_FLOOR)
            if ec > limit2:
                fails2 += 1
            if e2 > limit:                      # the compiled reference under the rule of version 2
                refc_fails += 1
            if ec / limit2 > worst2["ratio"]:
                worst2 = {"ratio": ec / limit2, "parameter": n, "candidate_err": ec, "reference_err": er,
                          "compiled_reference_err": e2, "limit": limit2, "elements": r["elements"]}
            num["refc_orc"] += x["d2_orc"]; num["refc_ref"] += x.get("d2_ref", 0.0)
    out = {"parameters": len(names), "failing": fails, "worst": worst,
           "all_parameters_rel_l2": {"candidate_vs_oracle": (num["cand_orc"] / max(den["orc"], 1e-30)) ** 0.5,
                                     "reference_vs_oracle": (num["ref_orc"] / max(den["orc"], 1e-30)) ** 0.5,
                                     "candidate_vs_reference": (num["cand_ref"] / max(den["ref"], 1e-30)) ** 0.5}}
    if s_refc is not None:
        out["two_arm"] = {"failing": fails2, "worst": worst2,
                          "compiled_reference_failing_under_version_2": refc_fails,
                          "all_parameters_rel_l2": {"compiled_reference_vs_oracle": (num["refc_orc"] / max(den["orc"], 1e-30)) ** 0.5,
                                                    "compiled_reference_vs_reference": (num["refc_ref"] / max(den["ref"], 1e-30)) ** 0.5}}
    return out


def step_fn(model, loss_fn, opt):
    def run(ids):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, ids)
        loss.backward()
        opt.step()
        return loss.detach()
    return run


def paired_steps(run_a, run_b, batches, warmup, seed):
    rng = random.Random(seed)
    ta, tb, la, lb = [], [], [], []
    for i, ids in enumerate(batches):
        order = ((run_a, ta, la), (run_b, tb, lb)) if rng.getrandbits(1) else ((run_b, tb, lb), (run_a, ta, la))
        for fn, ts, ls in order:
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record(); loss = fn(ids); e.record(); e.synchronize()
            if i >= warmup:
                ts.append(float(s.elapsed_time(e)))
            ls.append(float(loss))
    pr = sorted(x / y for x, y in zip(ta, tb))
    paired = {"median": statistics.median(pr), "q25": pr[len(pr) // 4], "q75": pr[(3 * len(pr)) // 4], "n": len(pr)}
    return {"reference_ms": ta, "candidate_ms": tb, "paired_ratio": paired, "reference_p50_ms": statistics.median(ta),
            "candidate_p50_ms": statistics.median(tb), "speedup": statistics.median(ta) / statistics.median(tb),
            "reference_losses": la, "candidate_losses": lb,
            "loss_max_abs_diff": max(abs(x - y) for x, y in zip(la, lb)), "warmup": warmup, "steps": len(ta)}


def one_row(model_dir, B, S, args, impl):
    seed = 5101 + 31 * B + S
    row = {"model": model_dir.name, "B": B, "S": S, "tokens": B * S, "seed": seed, "status": "running", "errors": {},
           "gpu_exclusivity": {"before": impl.gpu_snapshot()}}
    monitor = impl.ExclusiveMonitor(); monitor.start()
    try:
        gc.collect(); torch.cuda.empty_cache(); torch._dynamo.reset()
        base = load(model_dir, "cuda")
        vocab = int(base.config.vocab_size)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        total = args.warmup + args.steps
        heldout = None
        if args.data:
            ids_all, meta = text_tokens(model_dir, args.data)
            batches = text_batches(ids_all, B, S, total, seed)
            batches2 = text_batches(ids_all, B, S, total, seed + 1000) if args.retest else None
            heldout = heldout_batches(ids_all, B, S, args.heldout_batches)
            row["data"] = dict(meta, kind="text", holdout_fraction=HOLDOUT, window_seed=seed,
                               retest_window_seed=(seed + 1000) if args.retest else None)
        else:
            batches = [torch.randint(0, vocab, (B, S), generator=gen, dtype=torch.long).cuda() for _ in range(total)]
            batches2 = ([torch.randint(0, vocab, (B, S), generator=gen, dtype=torch.long).cuda() for _ in range(total)]
                        if args.retest else None)
            row["data"] = {"kind": "seeded random token ids"}
        row["model_config"] = {"hidden": int(base.config.hidden_size), "vocab": vocab,
                               "layers": int(base.config.num_hidden_layers),
                               "tied_head": bool(base.lm_head.weight is base.get_input_embeddings().weight),
                               "parameters": sum(p.numel() for p in base.parameters())}
        # choose the kernel configuration for this logits shape (outside any timing)
        with torch.no_grad():
            z = torch.empty((B * S, vocab), device="cuda", dtype=torch.bfloat16).normal_()
            y = torch.randint(0, vocab, (B * S,), device="cuda")
            row["ce_config"] = list(K.tune_ce(z, y) or ())
            del z, y
        torch.cuda.empty_cache()

        # 1. one step from identical weights (streaming, see "memory" in the docstring)
        place = placement(row["model_config"]["parameters"], args)
        row["gate_memory"] = place
        lock = MemLock(args.mem_lock) if (args.mem_lock and place["cpu_gib_estimate"] > MEM_LOCK_GIB) else None
        base = None; gc.collect(); torch.cuda.empty_cache()
        if lock is not None:
            row["gate_memory"]["lock_wait_s"] = lock.acquire()
        try:
            if place["oracle"] == "cpu":
                torch.set_num_threads(args.oracle_threads)
            orc = load(model_dir, place["oracle"], torch.float32)
            orc.zero_grad(set_to_none=True)
            l_orc = ref_loss(orc, batches[0].to(place["oracle"])); l_orc.backward(); l_orc = float(l_orc)
            go = {}
            for n, p in orc.named_parameters():       # keep only the gradients, one parameter at a time
                if p.grad is not None:
                    go[n] = p.grad.detach().to(place["oracle_gradients"]); p.grad = None
            del orc; gc.collect(); torch.cuda.empty_cache()
            torch.set_num_threads(args.cpu_threads)
            names = list(go)
            base = load(model_dir, "cuda")
            l_ref, s_ref, gr = arm(base, ref_loss, batches[0], go, None, keep=place["reference_gradients"])
            l_ref2, s_ref2, _ = arm(base, ref_loss, batches[0], None, gr)
            l_cand, s_cand, _ = arm(base, cand_loss, batches[0], go, gr)
            l_refc, s_refc = None, None
            if args.compile:                      # second PyTorch arm, for the two-arm record
                try:
                    torch._dynamo.reset()
                    base_c = torch.compile(base, dynamic=False)
                    l_refc, s_refc, _ = arm(base, lambda mm, ids: base_c(input_ids=ids, labels=ids, use_cache=False).loss,
                                            batches[0], go, gr)
                    del base_c
                except Exception as exc:
                    row["errors"]["gate_compiled"] = {"error": repr(exc), "traceback": traceback.format_exc()[-3000:]}
                torch._dynamo.reset()
            if heldout is not None:
                row["heldout"] = {"before": heldout_loss(base, heldout)}
        finally:
            go = None; gr = None; base = None; gc.collect(); torch.cuda.empty_cache()
            if lock is not None:
                lock.release()
        row["one_step"] = {"gate_version": GATE_VERSION, "oracle_loss": l_orc, "oracle": oracle_gate(names, s_cand, s_ref, s_refc),
                           "compiled_reference_loss": l_refc,
                           "reference_loss": l_ref, "candidate_loss": l_cand,
                           "loss_rel_diff": abs(l_cand - l_ref) / max(abs(l_ref), 1e-30),
                           "reference_rerun_loss_rel_diff": abs(l_ref2 - l_ref) / max(abs(l_ref), 1e-30),
                           "candidate_vs_reference": compare(s_cand, s_ref),
                           "reference_rerun_vs_reference": compare(s_ref2, s_ref),
                           "thresholds": {"loss_rel": LOSS_REL, "grad_factor": GRAD_FACTOR, "grad_floor": GRAD_FLOOR,
                                          "v1_grad_rel_l2": GRAD_REL_L2_V1}}
        del s_ref, s_ref2, s_cand, s_refc
        c = row["one_step"]["candidate_vs_reference"]
        row["gate"] = {"version": GATE_VERSION,
                       "pass": bool(row["one_step"]["loss_rel_diff"] <= LOSS_REL and c["missing"] == 0
                                    and row["one_step"]["oracle"]["failing"] == 0),
                       "v1_would_pass": bool(c["max_rel_l2"] <= GRAD_REL_L2_V1)}
        ta = row["one_step"]["oracle"].get("two_arm")
        if ta is not None:
            row["gate"]["two_arm_pass"] = bool(row["one_step"]["loss_rel_diff"] <= LOSS_REL and c["missing"] == 0
                                               and ta["failing"] == 0)
        if not row["gate"]["pass"]:
            # The row stays invalid.  Its timing is still taken and labelled as a diagnostic, so that the
            # data exists if the row is examined later; it does not count as a result.
            row["gate_reasons"] = ["candidate failed the one-step gate"]
            row["timing_is_diagnostic_only"] = True

        # 2 + 3. two runs from the same initial weights on the same batches (fresh copies of the checkpoint)
        meas = measure_lean if args.lean else measure
        row["timing"] = meas(model_dir, batches, args, seed, row, "initial", heldout)
        t = row["timing"]
        row["speedup_vs_fastest"] = min(x["speedup"] for k, x in t.items() if k != "aa")
        row["fastest_reference"], row["speedup_vs_fastest_reference"] = verdict(t)
        wins = row["speedup_vs_fastest_reference"] > WIN
        if args.retest:
            row["retest"] = meas(model_dir, batches2, args, seed + 1000, row, "retest")
            row["retest_fastest_reference"], row["retest_speedup_vs_fastest_reference"] = verdict(row["retest"])
            wins = wins and row["retest_speedup_vs_fastest_reference"] > WIN
        if args.lean:
            row["trajectory"] = trajectories(model_dir, batches, args, heldout, row)
        row["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        row["kept"] = bool(row["gate"]["pass"] and wins)
        if "two_arm_pass" in row["gate"]:
            row["kept_two_arm"] = bool(row["gate"]["two_arm_pass"] and wins)
        row["status"] = "tested_pass" if row["gate"]["pass"] else "tested_invalid"
    except Exception as exc:
        row["status"] = "tested_invalid"; row["gate_reasons"] = [repr(exc)]
        row["errors"]["row"] = {"error": repr(exc), "traceback": traceback.format_exc()[-4000:]}
    finally:
        row["gpu_exclusivity"]["observations"] = monitor.finish()
        row["gpu_exclusivity"]["after"] = impl.gpu_snapshot()
        ex = row["gpu_exclusivity"]
        if row["status"] == "tested_pass" and not (ex["before"].get("exclusive") and ex["after"].get("exclusive")
                                                    and ex["observations"] and all(x.get("exclusive") for x in ex["observations"])):
            row["status"] = "tested_invalid"; row["gate_reasons"] = ["selected physical GPU was not exclusive"]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--shapes", default="8x512", help="comma list of BxS")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--data", type=Path, help="text file; batches are windows of its token stream")
    ap.add_argument("--heldout-batches", type=int, default=8)
    ap.add_argument("--retest", action="store_true")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--optimizer", choices=("adamw", "sgd", "muon"), default="adamw")
    ap.add_argument("--muon-lr", type=float, default=2e-4)
    ap.add_argument("--grad-device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--oracle-device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--mem-lock", type=Path, help="lock directory shared by stages whose gate holds a large CPU-side oracle")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--oracle-threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.cpu_threads)
    try:
        import torch._inductor.config as _ic
        _ic.compile_threads = args.cpu_threads
    except Exception:
        pass
    import e1_rope_refined_impl as impl
    rec = {"schema": "osdi-train-step-ce-v1",
           "scope": "whole-model training step; candidate replaces the loss and its backward by one fused kernel",
           "environment": {"host": platform.node(), "gpu": torch.cuda.get_device_name(),
                           "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "torch": torch.__version__,
                           "triton": impl.triton.__version__, "cuda": torch.version.cuda,
                           "compute_capability": list(torch.cuda.get_device_capability())},
           "source_hashes": {p.name: sha(p) for p in (HERE / "train_step_ce.py", HERE / "lmhead_kernels.py")},
           "script_version": 3,
           "protocol": {"dtype": "bfloat16", "optimizer": {"adamw": "AdamW", "sgd": "SGD", "muon": "Muon"}[args.optimizer], "lr": args.lr,
                        "muon": ({"lr": args.muon_lr, "momentum": 0.95, "nesterov": True, "ns_steps": 5, "other_parameters": "AdamW"}
                                 if args.optimizer == "muon" else None),
                        "steps": args.steps, "warmup": args.warmup, "win_threshold": WIN, "compile": bool(args.compile),
                        "retest": bool(args.retest), "aa": bool(args.aa), "grad_device": args.grad_device,
                        "oracle_device": args.oracle_device, "lean": bool(args.lean),
                        "cpu_threads": args.cpu_threads, "oracle_threads": args.oracle_threads,
                        "data": str(args.data.name) if args.data else "seeded random token ids",
                        "gate_version": GATE_VERSION},
           "model": args.model_dir.name, "rows": [], "status": "running", "publication_eligible": False}
    try:
        import transformers
        rec["environment"]["transformers"] = transformers.__version__
    except Exception:
        pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for s in [x for x in args.shapes.split(",") if x]:
        B, S = (int(v) for v in s.lower().split("x"))
        print(f"{args.model_dir.name} B={B} S={S}", flush=True)
        row = one_row(args.model_dir, B, S, args, impl)
        rec["rows"].append(row); rec["tuning"] = K.TUNE_LOG
        args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
        print(json.dumps({"B": B, "S": S, "status": row["status"], "gate": row.get("gate"),
                          "one_step": {k: row.get("one_step", {}).get(k) for k in ("loss_rel_diff", "candidate_vs_reference", "oracle")},
                          "speedup": {k: round(v["speedup"], 4) for k, v in row.get("timing", {}).items()},
                          "retest": {k: round(v["speedup"], 4) for k, v in row.get("retest", {}).items()},
                          "heldout": row.get("heldout"),
                          "trajectory_loss_diff": {k: v for k, v in row.get("trajectory", {}).items() if k.endswith("diff")},
                          "multi_step": row.get("trajectory", {}).get("multi_step"), "kept_two_arm": row.get("kept_two_arm"),
                          "kept": row.get("kept"), "errors": sorted(row["errors"]), "reasons": row.get("gate_reasons")},
                         default=str), flush=True)
    rec["status"] = "tested_pass" if rec["rows"] and all(r["status"] == "tested_pass" for r in rec["rows"]) else "tested_invalid"
    rec["publication_eligible"] = rec["status"] == "tested_pass"
    args.output.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({"output": str(args.output), "status": rec["status"], "rows": len(rec["rows"])}), flush=True)


if __name__ == "__main__":
    main()
