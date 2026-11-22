// One line per row of the WikiText-2 training-step receipts (train_step_ce.py, script version 3).
// node analysis/train3_numbers.js <results dir> [prefix, default train3_]
const fs = require('fs'), path = require('path');
const dir = process.argv[2], prefix = process.argv[3] || 'train3_';
const f = (x, n = 4) => (typeof x === 'number' && Number.isFinite(x)) ? x.toFixed(n) : '-';
for (const fn of fs.readdirSync(dir).filter(x => x.startsWith(prefix) && x.endsWith('.json')).sort()) {
  let d; try { d = JSON.parse(fs.readFileSync(path.join(dir, fn), 'utf8')); } catch (e) { console.log(fn, 'unreadable'); continue; }
  const pr = d.protocol || {};
  console.log(`\n## ${fn}  ${d.status}  torch ${(d.environment || {}).torch} ${pr.optimizer} steps ${pr.steps} lean ${pr.lean} data ${pr.data}`);
  for (const r of d.rows || []) {
    const o = r.one_step || {}, g = o.oracle || {}, w = g.worst || {}, ta = g.two_arm || {}, t = r.timing || {}, rt = r.retest || {}, h = r.heldout || {}, tr = r.trajectory || {}, ms = tr.multi_step || {};
    const sp = x => Object.entries(x).map(([k, v]) => `${k} ${f(v.speedup)}`).join(' ');
    console.log(`  ${r.B}x${r.S} ${(r.model_config ? (r.model_config.parameters / 1e9).toFixed(2) + 'B' : '').padEnd(6)} ${r.status} | gate v2 ${r.gate ? r.gate.pass : '-'} (failing ${g.failing}/${g.parameters}, worst ${f(w.ratio, 3)} ${w.parameter || ''} n=${w.elements}) two-arm ${r.gate ? r.gate.two_arm_pass : '-'} (failing ${ta.failing}, compiled ref fails v2: ${ta.compiled_reference_failing_under_version_2})`);
    console.log(`      timing: ${sp(t)} | retest: ${sp(rt)} | vs fastest ref ${f(r.speedup_vs_fastest_reference)} / retest ${f(r.retest_speedup_vs_fastest_reference)} | kept ${r.kept} kept_two_arm ${r.kept_two_arm}`);
    console.log(`      step ms: ref eager ${f(t.eager && t.eager.reference_p50_ms, 1)} compiled ${f(t.compiled && t.compiled.reference_p50_ms, 1)} cand ${f(t.compiled && t.compiled.candidate_p50_ms, 1)} | tokens/s cand ${t.compiled ? Math.round(r.tokens / t.compiled.candidate_p50_ms * 1000) : '-'}`);
    console.log(`      heldout ppl: before ${f(h.before && h.before.perplexity, 3)} ref ${f(h.reference_after && h.reference_after.perplexity, 3)} cand ${f(h.candidate_after && h.candidate_after.perplexity, 3)} compiled ref ${f(h.reference_compiled_after && h.reference_compiled_after.perplexity, 3)} | traj max |dloss| cand ${f(tr.loss_max_abs_diff, 5)} rerun ${f(tr.reference_rerun_loss_max_abs_diff, 5)} compiled ${f(tr.reference_compiled_loss_max_abs_diff, 5)} | multi-step within: heldout ${ms.heldout_within} traj ${ms.trajectory_within}`);
    if (r.errors && Object.keys(r.errors).length) console.log(`      ERRORS: ${Object.entries(r.errors).map(([k, v]) => k + ': ' + String(v.error).slice(0, 160)).join(' | ')}`);
    if (r.gate_memory) console.log(`      gate memory: ${JSON.stringify(r.gate_memory)} peak GPU ${r.peak_memory_gib} GiB`);
  }
}
