// Numbers of the attention census with the optimized (K-once, query-fused) kernel, from the receipts of
// recensus_driver.py.   node analysis/recensus_numbers.js <results dir> [more dirs...]
// A row is kept if it beats the same anchor, torch.compile and eager SDPA by more than 1% in the initial
// measurement and in the retest (the rule printed by summarize.js).
const fs = require('fs'), path = require('path');
const gm = a => a.length ? Math.exp(a.reduce((x, y) => x + Math.log(y), 0) / a.length) : NaN;
const f = (x, n = 2) => Number.isFinite(x) ? x.toFixed(n) : '-';
for (const dir of process.argv.slice(2)) {
  const acc = {};
  for (const fn of fs.readdirSync(dir).filter(x => /^census_(rope|norope)_.*\.json$/.test(x) && !/yielded|_bm\d+/.test(x)).sort()) {
    let d; try { d = JSON.parse(fs.readFileSync(path.join(dir, fn), 'utf8')); } catch (e) { continue; }
    const mode = /^census_rope_/.test(fn) ? 'rope' : 'norope';
    const stack = (fn.match(/_(t211|t215)\.json$/) || [, 'formal'])[1];
    const key = `${mode} | ${stack} | ${(d.environment || {}).gpu} | torch ${(d.environment || {}).torch}`;
    const a = acc[key] = acc[key] || { models: [], rows: 0, invalid: 0, same: 0, compiled: 0, kept: [], keptB1: 0, keptB8: 0, keptB32: 0, bestAll: [] };
    const rows = d.rows || [];
    const model = fn.replace(/^census_(rope|norope)_/, '').replace(/(_t211|_t215)?\.json$/, '');
    let k = 0;
    for (const r of rows) {
      a.rows++;
      if (r.status !== 'tested_pass') { a.invalid++; continue; }
      const t = r.timing;
      const s = { unfused: t.unfused.speedup, compiled: t.compiled.speedup, vendor: t.vendor.speedup };
      const best = Math.min(s.unfused, s.compiled, s.vendor);
      const init = Math.min(t.unfused.initial.ratio_median, t.compiled.initial.ratio_median, t.vendor.initial.ratio_median);
      a.bestAll.push(best);
      if (s.unfused > 1.01) a.same++;
      if (s.compiled > 1.01) a.compiled++;
      if (best > 1.01 && init > 1.01) { a.kept.push(best); k++; if (r.B === 1) a.keptB1++; else if (r.B === 8) a.keptB8++; else if (r.B === 32) a.keptB32++; }
    }
    a.models.push(`${model} ${k}/${rows.length}`);
  }
  console.log(`\n# ${dir}`);
  for (const [key, a] of Object.entries(acc)) {
    console.log(`${key}\n   models: ${a.models.join(', ')}\n   rows ${a.rows}, invalid ${a.invalid}; faster than same anchor ${a.same}; faster than torch.compile ${a.compiled}; kept ${a.kept.length}` +
      ` (B=1: ${a.keptB1}, B=8: ${a.keptB8}, B=32: ${a.keptB32}); kept speedup vs best unfused ${f(Math.min(...a.kept))}-${f(Math.max(...a.kept))}, gm ${f(gm(a.kept), 3)}; all rows vs best unfused ${f(Math.min(...a.bestAll))}-${f(Math.max(...a.bestAll))}`);
  }
}
