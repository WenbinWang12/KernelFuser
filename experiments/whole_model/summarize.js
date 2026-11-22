// node summarize.js <results dir>     Prints one block per receipt found in the directory.
const fs = require('fs'), path = require('path');
const dir = process.argv[2] || 'results';
const gm = a => a.length ? Math.exp(a.reduce((x, y) => x + Math.log(y), 0) / a.length) : NaN;
const f = (x, n = 2) => (x === undefined || x === null || Number.isNaN(x)) ? '  -  ' : x.toFixed(n);
const files = fs.readdirSync(dir).filter(x => x.endsWith('.json') && !/yielded/.test(x)).sort();
for (const fn of files) {
  let d; try { d = JSON.parse(fs.readFileSync(path.join(dir, fn), 'utf8')); } catch (e) { console.log(`\n## ${fn}: unreadable`); continue; }
  const rows = d.rows || []; const env = d.environment || {};
  const st = {}; rows.forEach(r => st[r.status] = (st[r.status] || 0) + 1);
  console.log(`\n## ${fn}  [${d.schema || '?'}]  ${env.gpu || ''} torch ${env.torch || '?'} triton ${env.triton || '?'}  rows ${rows.length} ${JSON.stringify(st)}`);
  const bad = rows.filter(r => r.status !== 'tested_pass');
  for (const r of bad.slice(0, 6)) console.log(`   INVALID ${r.model || ''} ${r.B !== undefined ? 'B=' + r.B : ''} ${r.S !== undefined ? 'S=' + r.S : ''} ${r.M !== undefined ? 'M=' + r.M : ''}: ${JSON.stringify(r.gate_reasons || Object.keys(r.errors || {})).slice(0, 220)}`);
  const ok = rows.filter(r => r.status === 'tested_pass');
  if (!ok.length) continue;
  if (/census/.test(d.schema || '')) {
    for (const r of ok) { const t = r.timing; r._s = { unfused: t.unfused.speedup, compiled: t.compiled.speedup, vendor: t.vendor.speedup }; r._best = Math.min(...Object.values(r._s)); r._init = Math.min(t.unfused.initial.ratio_median, t.compiled.initial.ratio_median, t.vendor.initial.ratio_median); }
    const kept = ok.filter(r => r._best > 1.01 && r._init > 1.01);
    console.log(`   vs torch.compile: ${ok.filter(r => r._s.compiled > 1.01).length}/${ok.length} faster, gm ${f(gm(ok.map(r => r._s.compiled)))} (${f(Math.min(...ok.map(r => r._s.compiled)))}-${f(Math.max(...ok.map(r => r._s.compiled)))}) | vs same anchor: ${ok.filter(r => r._s.unfused > 1.01).length}/${ok.length}, gm ${f(gm(ok.map(r => r._s.unfused)), 3)} | vs eager SDPA gm ${f(gm(ok.map(r => r._s.vendor)))} | kept (all three, initial and retest): ${kept.length}/${ok.length}${kept.length ? ', gm ' + f(gm(kept.map(r => r._best)), 3) + ' max ' + f(Math.max(...kept.map(r => r._best)), 3) : ''}`);
    const byS = {}; ok.forEach(r => (byS[`B${r.B} S${r.S}`] = byS[`B${r.B} S${r.S}`] || []).push(r));
    console.log('   ' + Object.entries(byS).map(([k, a]) => `${k}: c ${f(gm(a.map(r => r._s.compiled)))} u ${f(gm(a.map(r => r._s.unfused)))} kept ${a.filter(r => r._best > 1.01 && r._init > 1.01).length}/${a.length}`).join(' | '));
  } else if (d.schema === 'osdi-e2e-prefill-v1') {
    for (const r of ok) { const t = r.timing; console.log(`   ${r.model} B=${r.B} S=${r.S}: cand ${f(t.compiled.candidate_p50_ms, 3)} ms | vs eager ${f(t.eager.speedup, 3)} vs compiled ${f(t.compiled.speedup, 3)} vs same anchor ${f(t.unfused.speedup, 3)} | fastest ${r.fastest_baseline} -> ${f(r.speedup_vs_fastest, 3)} kept ${r.kept} | relL2 cand ${f(r.correctness.candidate.relative_l2, 4)} compiled ${f((r.correctness.compiled || {}).relative_l2, 4)} top1 ${f(r.correctness.candidate.top1_agreement, 3)} | A/A ${f((r.self_comparison_compiled || {}).ratio, 4)}`); }
  } else if (d.schema === 'osdi-lmhead-v1') {
    for (const r of ok) { const t = r.timing; console.log(`   ${String(r.model).padEnd(22)} K=${String(r.K).padStart(4)} V=${String(r.V).padStart(6)} M=${String(r.M).padStart(4)} ${r.weights === 'random' ? 'rand' : 'ckpt'} [${r.timing_mode}] cand ${f(Object.values(t)[0].candidate_p50_ms, 2)} ms | ` + Object.entries(t).map(([k, v]) => `${k} ${f(v.speedup, 3)}`).join(' ') + ` | fastest ${r.fastest_baseline} -> ${f(r.speedup_vs_fastest, 3)} kept ${r.kept}`); }
    console.log(`   kept ${ok.filter(r => r.kept).length}/${ok.length}; gm vs fastest ${f(gm(ok.map(r => r.speedup_vs_fastest)), 3)}`);
  } else if (d.schema === 'osdi-train-step-ce-v1') {
    for (const r of ok) { console.log(`   ${r.model} B=${r.B} S=${r.S}: one-step loss rel diff ${r.one_step.loss_rel_diff.toExponential(1)}, max grad relL2 ${r.one_step.candidate_vs_reference.max_rel_l2.toExponential(1)} (rerun ${r.one_step.reference_rerun_vs_reference.max_rel_l2.toExponential(1)}) | ` + Object.entries(r.timing).map(([k, v]) => `${k}: ref ${f(v.reference_p50_ms, 1)} ms cand ${f(v.candidate_p50_ms, 1)} ms x${f(v.speedup, 3)} maxdLoss ${v.loss_max_abs_diff.toExponential(1)}`).join(' | ') + ` | vs fastest reference ${f(r.speedup_vs_fastest_reference, 3)} kept ${r.kept}${Object.keys(r.errors || {}).length ? ' errors ' + Object.keys(r.errors) : ''}`); }
  } else if (/^osdi-decode-mlp/.test(d.schema || '')) {
    const byModel = {}; ok.forEach(r => (byModel[r.model] = byModel[r.model] || []).push(r));
    for (const [mname, a] of Object.entries(byModel)) {
      console.log(`   ${mname.padEnd(22)} K=${a[0].K} I=${a[0].I} L=${a[0].layers}`);
      for (const r of a) { const t = r.timing; console.log(`      M=${String(r.M).padStart(2)} cand ${f(t.compiled.candidate_p50_ms * 1000 / r.layers, 1).padStart(6)} us/blk | eager ${f(t.eager.speedup, 3)} compiled ${f(t.compiled.speedup, 3)} same anchor ${f(t.unfused.speedup, 3)} | fastest ${r.fastest_baseline} -> ${f(r.speedup_vs_fastest, 3)} kept ${r.kept} | err cand ${f(r.correctness.candidate.rel_l2, 4)} eager ${f(r.correctness.eager.rel_l2, 4)}`); }
    }
    const kept = ok.filter(r => r.kept);
    console.log(`   kept ${kept.length}/${ok.length}${kept.length ? `; kept rows vs fastest ${f(Math.min(...kept.map(r => r.speedup_vs_fastest)), 3)}-${f(Math.max(...kept.map(r => r.speedup_vs_fastest)), 3)}, vs compiled ${f(Math.min(...kept.map(r => r.timing.compiled.speedup)), 3)}-${f(Math.max(...kept.map(r => r.timing.compiled.speedup)), 3)}` : ''}`);
  } else if (d.schema === 'osdi-e2e-decode-v1') {
    for (const r of ok) { const t = r.timing; console.log(`   ${r.model} B=${r.B} T=${r.T}: cand ${f(t.compiled.candidate_p50_ms, 3)} ms | ` + Object.entries(t).map(([k, v]) => `${k} ${f(v.speedup, 3)}`).join(' ') + ` | fastest ${r.fastest_baseline} -> ${f(r.speedup_vs_fastest, 3)} kept ${r.kept} | relL2 cand ${f(r.correctness.candidate.relative_l2, 4)} eager ${f(r.correctness.eager.relative_l2, 4)} KL cand ${r.correctness.candidate.kl.toExponential(1)} eager ${r.correctness.eager.kl.toExponential(1)} top1 ${f(r.correctness.candidate.top1_agreement, 2)} | A/A ${f(r.self_comparison_compiled, 4)}${Object.keys(r.errors || {}).length ? ' errors ' + Object.keys(r.errors) : ''}`); }
    console.log(`   kept ${ok.filter(r => r.kept).length}/${ok.length}`);
  } else { console.log('   status', d.status); }
}
