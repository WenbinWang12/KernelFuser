# OSU GPU handoff: remaining publication runs

This file is the run contract for the remaining OSU work. It deliberately
does not include company or production data. A result is publishable only when
the receipt contains the source commit and hashes, exact software versions,
physical GPU identity, correctness checks before timing, paired timing and an
independent retest, an exclusivity record, and non-empty traces. A failed gate
must be recorded as `tested_invalid`; it must not contribute a speedup claim.

## Hardware and software rules

- Use OSU H100 and OSU H200 only.
- Do not launch any new job on H200 GPU IDs 0, 1, 2, or 3. Use an available
  H200 GPU ID in 4--7 and record the physical ID in every receipt. Do not
  displace another user's process.
- For the H100/H200 comparison, use the same kernel revision, baseline code,
  benchmark protocol, and a mutually pinned released PyTorch/Triton/CUDA
  stack. Record the exact versions; development build labels are not enough.
- Raw receipts and traces may be sent privately to the authors, but the public
  repository must contain only sanitized CSV/JSON and repository-relative
  source hashes. Remove hostnames, UUIDs, PIDs, private paths, credentials,
  weights, and company production logs.

## P0-1: RoPE-aware isolated attention census

Run the same improved kernel on both devices. The required matrix is

`9 geometries x S={128,256,384,512} x B={1,8,32} x {causal, non-causal}`

for 216 rows per GPU. Each row must compare the candidate with all three
controls: the same anchor with separate RMSNorm and RoPE, `torch.compile`, and
vendor SDPA. Alternate candidate and each control in paired timing, then run
an independent retest for rows near or above the 1% threshold. Check the FP32
oracle, tails and causal mask before timing, and keep the trace and GPU
exclusivity evidence for every promoted row.

The nine geometries must be named in the receipt. Do not label a Qwen3-8B-only
24-row run as a nine-architecture census. If an architecture uses a different
epsilon, scale, sliding window, or rotary convention, encode that parameter in
the candidate and in the receipt rather than silently using Qwen defaults.

## P0-2: Qwen3-8B prefill

After the isolated kernel passes, run Qwen3-8B prefill at
`S={128,256}` and `B={1,8,32}` with shape dispatch enabled. First compare the
compiled and eager unfused implementations. Require agreement of full logits,
top-1 predictions, NLL, and parameter/state fingerprints before timing. Then
measure compiled unfused versus compiled fused with the same validated kernel.
If the compiled/eager baseline gate fails, emit a diagnostic receipt and no
timing row; do not relax the tolerance to manufacture a pass. Gemma
sliding-window prefill is out of scope.

## P1 follow-ups

1. **Optimization ablation.** On one fixed software stack and identical shapes,
   measure old kernel -> K prepared once -> schedule change -> D=256 inverse-RMS
   factor on scores. The last step is a no-RoPE transformation; do not present
   it as a RoPE optimization.
2. **B200 ADS.** Re-run against the exact public ADS commit, or record a
   cryptographic source fingerprint and explain every difference. Keep the
   existing BF16 strict-correctness failures diagnostic and separate from the
   timing claim.
3. **Correctness.** Re-run the 13-checkpoint study on H100 before adding it to
   the main text. The audited H200 set is the current primary result.
4. **Optional appendix.** H200 improved no-RoPE census and E8 H100 may be run
   after P0; they are not blockers for the main conclusion.

## Do not run

Do not spend GPU time on E3c, E6, E9, Qwen3-30B-A3B E3b, LM-head, Gemma
sliding-window prefill, or residual E4. The existing residual and SwiGLU
results remain appendix diagnostics under the current performance gates.

## Return bundle

For each run, return:

1. a raw JSON receipt privately;
2. a sanitized CSV with one row per configuration and the three baseline
   columns;
3. a source/version manifest containing the commit, file hashes, GPU ID,
   PyTorch/Triton/CUDA versions and command line; and
4. a trace manifest proving that promoted rows have non-empty traces and
   exclusivity checks.

The public PR should include only items 2--4. A reviewer must be able to
recompute every reported ratio from the sanitized CSV without receiving a
private path or a production artifact.


