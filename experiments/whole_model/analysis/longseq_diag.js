// Long-sequence diagnosis from the census with the optimized kernel (normalization only, batch 1, S = 128..2048).
// For every sequence length: geometric mean of the fused kernel's speedup over (u) the same attention kernel
// after a separate normalization kernel, (c) torch.compile, (v) eager SDPA, and the implied ratios
// anchor/compiled = c/u and anchor/SDPA = v/u, i.e. how the Triton anchor path itself compares with the
// PyTorch paths (values below 1 mean that the anchor path is slower, so no fusion overhead could win).
//   node analysis/longseq_diag.js <results dir> [file regex]
const fs = require('fs'), path = require('path');
const dir = process.argv[2], re = new RegExp(process.argv[3] || '^census_norope_.*\\.json$');
const gm = a => a.length ? Math.exp(a.reduce((x, y) => x + Math.log(y), 0) / a.length) : NaN;
const f = x => Number.isFinite(x) ? x.toFixed(2) : ' -  ';
const groups = {};
for (const fn of fs.readdirSync(dir).filter(x => re.test(x) && !/yielded/.test(x)).sort()) {
  let d; try { d = JSON.parse(fs.readFileSync(path.join(dir, fn), 'utf8')); } catch (e) { continue; }
  const key = `${(d.environment || {}).gpu} torch ${(d.environment || {}).torch}` + (/_bm\d+/.test(fn) ? ' tiles ' + fn.match(/_(bm\d+_bn\d+)/)[1] : '');
  const g = groups[key] = groups[key] || { models: new Set(), rows: [] };
  for (const r of d.rows || []) if (r.status === 'tested_pass') { g.rows.push(r); g.models.add(r.model || fn.replace(/^census_(no)?rope_/, '').replace(/(_t21[15])?(_bm\d+_bn\d+)?\.json$/, '')); }
}
for (const [key, g] of Object.entries(groups)) {
  console.log(`\n${key}  (${g.models.size} models: ${[...g.models].join(', ')})`);
  console.log('   B    S | fused vs same anchor | vs torch.compile | vs eager SDPA | anchor path vs compile | anchor path vs SDPA | kept');
  const keys = [...new Set(g.rows.map(r => `${r.B}|${r.S}`))].sort((a, b) => { const [b1, s1] = a.split('|').map(Number), [b2, s2] = b.split('|').map(Number); return b1 - b2 || s1 - s2; });
  for (const k of keys) {
    const [B, S] = k.split('|').map(Number);
    const a = g.rows.filter(r => r.B === B && r.S === S);
    const u = gm(a.map(r => r.timing.unfused.speedup)), c = gm(a.map(r => r.timing.compiled.speedup)), v = gm(a.map(r => r.timing.vendor.speedup));
    const kept = a.filter(r => { const t = r.timing; return Math.min(t.unfused.speedup, t.compiled.speedup, t.vendor.speedup) > 1.01 && Math.min(t.unfused.initial.ratio_median, t.compiled.initial.ratio_median, t.vendor.initial.ratio_median) > 1.01; }).length;
    console.log(`  ${String(B).padStart(2)} ${String(S).padStart(4)} |        ${f(u)}          |      ${f(c)}        |     ${f(v)}      |          ${f(c / u)}          |        ${f(v / u)}         | ${kept}/${a.length}`);
  }
}
