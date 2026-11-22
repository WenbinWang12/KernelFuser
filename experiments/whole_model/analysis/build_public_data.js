// Builds the public, sanitized summaries of the campaign from the slim receipts:
//   <out>/data.js      window.KF_DATA = {...} for the project page
//   <out>/csv/*.csv    one table per experiment for results/ in the repository
// No host names, user names, paths or process information are copied; only model names, shapes, GPU model,
// PyTorch version and measured numbers.
//   node analysis/build_public_data.js <results_a800> <results_h200> <results_h100> <out dir>
const fs = require('fs'), path = require('path');
const [A800, H200, H100, OUT] = process.argv.slice(2);
fs.mkdirSync(path.join(OUT, 'csv'), { recursive: true });
const gm = a => a.length ? Math.exp(a.reduce((x, y) => x + Math.log(y), 0) / a.length) : null;
const r3 = x => (typeof x === 'number' && Number.isFinite(x)) ? Math.round(x * 1000) / 1000 : null;
const r4 = x => (typeof x === 'number' && Number.isFinite(x)) ? Math.round(x * 10000) / 10000 : null;
const load = (d, fn) => { try { return JSON.parse(fs.readFileSync(path.join(d, fn), 'utf8')); } catch (e) { return null; } };
const ls = (d, re) => fs.existsSync(d) ? fs.readdirSync(d).filter(x => re.test(x) && !/yielded|smoke/.test(x)).sort() : [];
const GPU = { 'NVIDIA H200 NVL': 'H200', 'NVIDIA H100 NVL': 'H100', 'NVIDIA A800 80GB PCIe': 'A800' };
const torchOf = d => String((d.environment || {}).torch || '').replace(/\+.*/, '');
const short = m => String(m).replace('-Instruct', '');
const csv = (name, header, rows) => { fs.writeFileSync(path.join(OUT, 'csv', name), [header.join(','), ...rows.map(r => r.map(v => v === null || v === undefined ? '' : String(v)).join(','))].join('\n') + '\n'); console.log(`${name}: ${rows.length} rows`); };
const ORDER = ['SmolLM2-135M-Instruct', 'Gemma-3-270M', 'SmolLM2-360M-Instruct', 'Qwen2.5-0.5B-Instruct', 'Qwen3-0.6B', 'Gemma-3-1B', 'Qwen2.5-1.5B-Instruct',
  'SmolLM2-1.7B-Instruct', 'Qwen3-1.7B', 'Qwen3-4B', 'Gemma-3-4B', 'Llama-3.1-8B', 'Qwen3-8B', 'Qwen3.5-9B', 'Qwen3-14B', 'Qwen3-32B', 'Qwen3-30B-A3B', 'Gemma-3-12B', 'Gemma-3-27B',
  'Qwen3.8-27B', 'Qwen3.5-35B-A3B'];
const ord = m => { const i = ORDER.indexOf(m); return i < 0 ? 999 : i; };
const DATA = { generated: new Date().toISOString().slice(0, 10) };

// ------------------------------------------------------------------ whole-model decode step (A800)
function decode(prefix, sfx, label) {
  const rows = [];
  for (const fn of ls(A800, new RegExp(`^${prefix}_[A-Za-z0-9.-]+${sfx}\\.json$`))) {
    if (sfx === '' && /_(t211|t215|broad|rerun|new)\.json$/.test(fn)) continue;
    const d = load(A800, fn); if (!d || d.status === 'running') continue;       // interrupted or unfinished receipts are not aggregated
    for (const r of d.rows || []) rows.push({ ...r, _cfg: d.model_config, _torch: torchOf(d) });
  }
  const best = r => Math.min(r.timing.compiled.speedup, r.timing.autotune ? r.timing.autotune.speedup : Infinity);
  const models = [...new Set(rows.map(r => r.model))].sort((a, b) => ord(a) - ord(b));
  const per = models.map(m => {
    const all = rows.filter(r => r.model === m), ok = all.filter(r => r.status === 'tested_pass');
    const rg = f => ok.length ? [r3(Math.min(...ok.map(f))), r3(Math.max(...ok.map(f)))] : null;
    return { model: short(m), layers: all[0]._cfg.layers, hidden: all[0]._cfg.hidden, rows: all.length, valid: ok.length, kept: ok.filter(r => r.kept).length,
      vsCompiled: rg(best), vsAnchor: rg(r => r.timing.unfused.speedup), vsEager: rg(r => r.timing.eager.speedup) };
  });
  const ok = rows.filter(r => r.status === 'tested_pass'), kept = ok.filter(r => r.kept);
  const out = { label, torch: rows[0] ? rows[0]._torch : null, rows: rows.length, valid: ok.length, kept: kept.length, models: per,
    keptVsCompiled: kept.length ? [r3(Math.min(...kept.map(best))), r3(Math.max(...kept.map(best)))] : null,
    keptVsAnchor: kept.length ? [r3(Math.min(...kept.map(r => r.timing.unfused.speedup))), r3(Math.max(...kept.map(r => r.timing.unfused.speedup)))] : null };
  return { out, table: rows.map(r => { const ok2 = r.status === 'tested_pass', t = r.timing || {};
    return [label, 'A800', r._torch, short(r.model), r.B, r.T, r.status, ok2 ? r4(t.compiled.candidate_p50_ms) : '', ok2 ? r4(t.eager.speedup) : '', ok2 ? r4(t.compiled.speedup) : '',
      ok2 && t.autotune ? r4(t.autotune.speedup) : '', ok2 ? r4(t.unfused.speedup) : '', ok2 ? r4(r.speedup_vs_fastest) : '', ok2 ? r.kept : '']; }) };
}
{
  const a = decode('dec4', '', 'seeded token ids'), b = decode('dec5', '', 'WikiText-2 prompts'), c = decode('dec5L', '', 'WikiText-2 prompts, long cache');
  DATA.decode = { random: a.out, text: b.out, longCache: c.out };
  csv('decode_step_a800.csv', ['inputs', 'gpu', 'pytorch', 'model', 'batch', 'cached_tokens', 'status', 'fused_ms', 'vs_eager', 'vs_compile_default', 'vs_compile_max_autotune', 'vs_same_kernels_unfused', 'vs_fastest_baseline', 'kept'],
    [...a.table, ...b.table, ...c.table]);
}

// ------------------------------------------------------------------ census with the optimized attention kernel
{
  const table = [], heat = {}, perModel = [], longseq = {};
  for (const [dir, files] of [[H200, ls(H200, /^census_(rope|norope)(_long)?_.*\.json$/)], [H100, ls(H100, /^census_(rope|norope)(_long)?_.*\.json$/)], [A800, ls(A800, /^census_(rope|norope)(_long)?_.*\.json$/)]]) {
    for (const fn of files) {
      if (/_bm\d+/.test(fn)) continue;
      const d = load(dir, fn); if (!d) continue;
      const mode = /^census_rope_long/.test(fn) ? 'ropeLong' : /^census_rope/.test(fn) ? 'rope' : 'norm', gpu = GPU[(d.environment || {}).gpu] || '?', tv = torchOf(d);
      const model = fn.replace(/^census_(rope|norope)(_long)?_/, '').replace(/(_t211|_t215)?\.json$/, '');
      let k = 0, n = 0;
      for (const r of d.rows || []) {
        if (r.status !== 'tested_pass') { table.push([mode, gpu, tv, model, r.B, r.S, r.causal, r.status]); continue; }
        const t = r.timing, u = t.unfused.speedup, c = t.compiled.speedup, v = t.vendor.speedup, best = Math.min(u, c, v);
        const init = Math.min(t.unfused.initial.ratio_median, t.compiled.initial.ratio_median, t.vendor.initial.ratio_median);
        const kept = best > 1.01 && init > 1.01; n++; if (kept) k++;
        table.push([mode, gpu, tv, model, r.B, r.S, r.causal, r.status, r4(u), r4(c), r4(v), r4(best), kept]);
        const key = `${mode}|${gpu}|${tv}`; (heat[key] = heat[key] || []).push({ model, B: r.B, S: r.S, u, c, v, best, kept });
      }
      perModel.push({ mode, gpu, torch: tv, model, kept: k, valid: n });
    }
  }
  csv('attention_census_optimized_kernel.csv', ['fusion', 'gpu', 'pytorch', 'model', 'batch', 'seq_len', 'causal', 'status', 'vs_same_anchor', 'vs_torch_compile', 'vs_eager_sdpa', 'vs_best_unfused', 'kept'], table);
  DATA.census = {};
  for (const [key, a] of Object.entries(heat)) {
    const [mode, gpu, tv] = key.split('|');
    const cells = []; const Bs = [...new Set(a.map(x => x.B))].sort((x, y) => x - y), Ss = [...new Set(a.map(x => x.S))].sort((x, y) => x - y);
    for (const B of Bs) for (const S of Ss) { const c = a.filter(x => x.B === B && x.S === S); if (!c.length) continue;
      cells.push({ B, S, n: c.length, kept: c.filter(x => x.kept).length, best: r3(gm(c.map(x => x.best))), sameAnchor: r3(gm(c.map(x => x.u))), compiled: r3(gm(c.map(x => x.c))), sdpa: r3(gm(c.map(x => x.v))),
        anchorVsCompiled: r3(gm(c.map(x => x.c / x.u))) }); }
    DATA.census[key] = { fusion: mode === 'rope' ? 'Q/K norm + RoPE' : mode === 'ropeLong' ? 'Q/K norm + RoPE at 1024 to 4096 tokens' : 'Q/K norm', gpu, torch: tv, models: [...new Set(a.map(x => x.model))].sort((x, y) => ord(x) - ord(y)), configs: a.length, kept: a.filter(x => x.kept).length,
      keptRange: a.some(x => x.kept) ? [r3(Math.min(...a.filter(x => x.kept).map(x => x.best))), r3(Math.max(...a.filter(x => x.kept).map(x => x.best)))] : null, cells };
  }
  DATA.censusModels = perModel.sort((a, b) => a.gpu.localeCompare(b.gpu) || a.torch.localeCompare(b.torch, undefined, { numeric: true }) || ord(a.model) - ord(b.model));
}

// ------------------------------------------------------------------ training step on WikiText-2 (fused loss)
{
  // Verdict rule (2026-11-22): one-step gradients within 1.5x of the worse PyTorch arm's distance to the FP32 oracle
  // (two-arm rule), loss trajectory and held-out loss within 1.5x of the spread between equivalent PyTorch runs,
  // and a paired per-step speedup above 1.01 against both PyTorch arms in the initial measurement and the retest.
  const rows = [];
  for (const fn of ls(A800, /^train3_.*\.json$/)) { const d = load(A800, fn); if (!d) continue;
    for (const r of d.rows || []) rows.push({ ...r, _opt: (d.protocol || {}).optimizer, _torch: torchOf(d) }); }
  rows.sort((a, b) => ord(a.model) - ord(b.model) || String(a._opt).localeCompare(String(b._opt)) || a.B * a.S - b.B * b.S || a.S - b.S);
  const pr = (r, tag, k) => { const t = (r[tag] || {})[k]; return t && t.paired_ratio ? t.paired_ratio.median : null; };
  const out = [], table = [];
  for (const r of rows) {
    const o = (r.one_step || {}).oracle || {}, h = r.heldout || {}, tr = r.trajectory || {};
    const timed = !!(r.timing && r.timing.eager);
    const comp = [pr(r, 'timing', 'compiled'), pr(r, 'retest', 'compiled')], eag = [pr(r, 'timing', 'eager'), pr(r, 'retest', 'eager')];
    const LR = 1e-3, F = 1.5;
    const refLoss = (tr.reference_losses || [])[0];
    const trajYard = Math.max(F * Math.max(tr.reference_rerun_loss_max_abs_diff || 0, tr.reference_compiled_loss_max_abs_diff || 0), LR * Math.abs(refLoss || 0));
    const trajOk = typeof tr.loss_max_abs_diff === 'number' ? tr.loss_max_abs_diff <= trajYard : null;
    const ms = tr.multi_step || null, heldOk = ms ? ms.heldout_within : null;
    const oneStep = r.gate ? (r.gate.two_arm_pass === undefined ? r.gate.pass : r.gate.two_arm_pass) : null;
    const correct = oneStep === true && trajOk !== false && heldOk !== false;
    const all = [...comp, ...eag].filter(x => x !== null);
    const win = timed && all.length === 4 && Math.min(...all) > 1.01;
    const e = { model: short(r.model), params: r.model_config ? r3(r.model_config.parameters / 1e9) : null, B: r.B, S: r.S, optimizer: r._opt, torch: r._torch, timed,
      vsCompiled: comp.map(r4), vsEager: eag.map(r4), aa: r4(pr(r, 'timing', 'aa')),
      gateV2: r.gate ? r.gate.pass : null, failingV2: o.failing, parameters: o.parameters, twoArm: r.gate ? r.gate.two_arm_pass : null,
      compiledRefFailsV2: o.two_arm ? o.two_arm.compiled_reference_failing_under_version_2 : null,
      trajDiff: tr.loss_max_abs_diff, trajRerun: tr.reference_rerun_loss_max_abs_diff, trajCompiled: tr.reference_compiled_loss_max_abs_diff, trajWithin: trajOk, heldoutWithin: heldOk,
      pplBefore: h.before ? r3(h.before.perplexity) : null, pplRef: h.reference_after ? r3(h.reference_after.perplexity) : null, pplCand: h.candidate_after ? r3(h.candidate_after.perplexity) : null,
      pplCompiledRef: h.reference_compiled_after ? r3(h.reference_compiled_after.perplexity) : null,
      tokensPerSec: timed && r.timing.compiled ? Math.round(r.tokens / r.timing.compiled.candidate_p50_ms * 1000) : null,
      correct, win, kept: correct && win, error: r.errors && r.errors.row ? String(r.errors.row.error).slice(0, 120) : null };
    out.push(e);
    table.push(['A800', e.torch, e.model, e.params, e.B, e.S, e.optimizer, e.vsCompiled[0], e.vsCompiled[1], e.vsEager[0], e.vsEager[1], e.aa, e.gateV2, e.twoArm, e.trajDiff, e.trajRerun, e.trajCompiled, e.trajWithin, e.heldoutWithin,
      e.pplBefore, e.pplRef, e.pplCand, e.pplCompiledRef, e.tokensPerSec, e.correct, e.win, e.kept]);
  }
  DATA.training = { rows: out, timed: out.filter(e => e.timed).length, correct: out.filter(e => e.correct).length, kept: out.filter(e => e.kept).length, models: new Set(out.map(e => e.model)).size };
  { const nk = out.filter(e => e.timed && !e.kept), v = nk.flatMap(e => e.vsCompiled.filter(x => x !== null));
    DATA.training.notKept = nk.length; DATA.training.notKeptRange = v.length ? [r3(Math.min(...v)), r3(Math.max(...v))] : null;
    DATA.training.notKeptLabel = `${nk.length} of ${DATA.training.timed} rows`; }
  csv('training_step_wikitext2_a800.csv', ['gpu', 'pytorch', 'model', 'params_B', 'batch', 'seq_len', 'optimizer', 'vs_compiled_initial', 'vs_compiled_retest', 'vs_eager_initial', 'vs_eager_retest', 'self_comparison',
    'one_step_gate_v2', 'one_step_two_arm', 'trajectory_max_abs_loss_diff', 'reference_rerun_diff', 'compiled_reference_diff', 'trajectory_within', 'heldout_within',
    'heldout_ppl_before', 'heldout_ppl_reference', 'heldout_ppl_fused', 'heldout_ppl_compiled_reference', 'fused_tokens_per_s', 'correct', 'wins', 'kept'], table);
}

// ------------------------------------------------------------------ fused loss at the LM head
{
  const out = [], table = [];
  for (const [dir, gpu] of [[H200, 'H200'], [A800, 'A800']]) for (const fn of ls(dir, /^lmhead_train(_more|_arch)?(_t211|_t215)?\.json$/)) {
    const d = load(dir, fn); if (!d) continue; const ok = (d.rows || []).filter(r => r.status === 'tested_pass'); if (!ok.length) continue;
    const c = ok.map(r => r.timing.compiled ? r.timing.compiled.speedup : NaN).filter(Number.isFinite), e = ok.map(r => r.timing.eager.speedup);
    out.push({ gpu, torch: torchOf(d), set: fn.replace(/\.json$/, ''), rows: d.rows.length, valid: ok.length, kept: ok.filter(r => r.kept).length, models: new Set(ok.map(r => r.model)).size,
      vsCompiled: [r3(Math.min(...c)), r3(Math.max(...c))], vsEager: [r3(Math.min(...e)), r3(Math.max(...e))] });
    for (const r of ok) table.push([gpu, torchOf(d), short(r.model), r.K, r.V, r.M, r4(r.timing.eager.speedup), r.timing.compiled ? r4(r.timing.compiled.speedup) : '', r4(r.speedup_vs_fastest), r.kept]);
  }
  DATA.lossHead = out;
  csv('fused_loss_lm_head.csv', ['gpu', 'pytorch', 'model', 'hidden', 'vocab', 'tokens', 'vs_eager', 'vs_compiled', 'vs_fastest', 'kept'], table);
}

// ------------------------------------------------------------------ whole-model prefill (like for like)
{
  const table = [];
  for (const [dir, gpu] of [[H200, 'H200'], [A800, 'A800']]) for (const fn of ls(dir, /^e2e_Qwen3-[0-9.]+B(_rerun)?(_t211|_t215)?\.json$/)) {
    const d = load(dir, fn); if (!d) continue;
    for (const r of d.rows || []) if (r.status === 'tested_pass') table.push([gpu, torchOf(d), short(r.model), r.B, r.S, r4(r.timing.eager.speedup), r4(r.timing.compiled.speedup), r4(r.timing.unfused.speedup), r4(r.speedup_vs_fastest), r.kept]);
  }
  csv('prefill_whole_model.csv', ['gpu', 'pytorch', 'model', 'batch', 'seq_len', 'vs_eager', 'vs_compiled', 'vs_same_anchor', 'vs_fastest', 'kept'], table);
  const by = g => table.filter(t => t[0] === g);
  DATA.prefill = ['H200', 'A800'].map(g => ({ gpu: g, rows: by(g).length, kept: by(g).filter(t => t[9]).length,
    range: by(g).length ? [r3(Math.min(...by(g).map(t => t[8]))), r3(Math.max(...by(g).map(t => t[8])))] : null }));
}
fs.writeFileSync(path.join(OUT, 'data.js'), '// generated by analysis/build_public_data.js from the receipts; do not edit\nwindow.KF_DATA = ' + JSON.stringify(DATA) + ';\n');
console.log('data.js written:', Object.keys(DATA).join(', '));
console.log('training rows:', DATA.training.rows.map(e => `${e.model} ${e.B}x${e.S} ${e.optimizer} comp ${e.vsCompiled.join('/')} eager ${e.vsEager.join('/')} correct ${e.correct} win ${e.win}`).join('\n               '));
