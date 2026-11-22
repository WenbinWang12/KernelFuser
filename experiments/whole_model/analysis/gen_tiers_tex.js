// Writes the LaTeX table bodies and number macros of the campaign matrix from data.js, the output of
// build_public_data.js (one implementation of the verdict rules; see EXPERIMENT_DESIGN.md).
//   node analysis/gen_tiers_tex.js <path to data.js> <gen dir>
// Writes train3_rows.tex, dec5_main.tex, recensus_patterns.tex, longseq.tex and numbers_tiers.tex.
const fs = require('fs'), path = require('path');
const [dataJs, out] = process.argv.slice(2);
global.window = {};
eval(fs.readFileSync(dataJs, 'utf8'));
const D = window.KF_DATA;
const f2 = x => x.toFixed(2), f3 = x => x === null ? '--' : x.toFixed(3);
const rng = r => !r ? '--' : (f2(r[0]) === f2(r[1]) ? f2(r[0]) : `${f2(r[0])}--${f2(r[1])}`);
const span = a => a.length ? [Math.min(...a), Math.max(...a)] : null;
const write = (name, lines) => { fs.writeFileSync(path.join(out, name), lines.join('\n') + (lines.length ? '\n' : '')); console.log(`${name}: ${lines.length} lines`); };
const macros = [];
const mac = (k, v) => macros.push(`\\newcommand{\\${k}}{${v}}`);
const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'];
const list = a => a.length <= 1 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];

// ---------------------------------------------------------------- training step on WikiText-2
{
  const rows = D.training.rows, timed = rows.filter(r => r.timed), kept = timed.filter(r => r.kept), notKept = timed.filter(r => !r.kept);
  const body = rows.map(r => r.timed
    ? `${r.model} & ${r.params.toFixed(1)} & ${r.B}$\\times$${r.S} & ${r.optimizer} & ${r.vsCompiled.map(f3).join('\\,/\\,')} & ${r.vsEager.map(f3).join('\\,/\\,')} & ${f3(r.aa)} & ${r.pplBefore.toFixed(1)} & ${r.pplRef.toFixed(2)} & ${r.pplCand.toFixed(2)} & ${r.correct ? 'yes' : 'no'} & ${r.kept ? '\\textbf{yes}' : 'no'} \\\\`
    : `${r.model} & ${r.params === null ? '--' : r.params.toFixed(1)} & ${r.B}$\\times$${r.S} & ${r.optimizer} & \\multicolumn{8}{l}{no result} \\\\`);
  write('train3_rows.tex', body);
  mac('trnRows', timed.length); mac('trnModels', new Set(timed.map(r => r.model)).size);
  mac('trnCorrect', timed.filter(r => r.correct).length); mac('trnKept', kept.length);
  mac('trnKeptModels', list([...new Set(kept.map(r => r.model))]) || 'none');
  mac('trnKeptVsCompiled', rng(span(kept.flatMap(r => r.vsCompiled.filter(v => v !== null)))));
  mac('trnKeptVsEager', rng(span(kept.flatMap(r => r.vsEager.filter(v => v !== null)))));
  mac('trnOtherVsCompiled', rng(span(notKept.flatMap(r => r.vsCompiled.filter(v => v !== null)))));
  mac('trnOptimizers', list([...new Set(timed.map(r => r.optimizer))]));
  mac('trnVtwoFail', timed.filter(r => r.gateV2 === false).length);                       // rows that fail the earlier per-parameter rule
  mac('trnCompiledRefFails', timed.filter(r => (r.compiledRefFailsV2 || 0) > 0).length);   // rows in which torch.compile fails that rule
}

// ---------------------------------------------------------------- decode step on WikiText-2 prompts
{
  const d = D.decode.text, ms = d.models;
  write('dec5_main.tex', ms.map(m => m.vsCompiled
    ? `${m.model} & ${m.layers} & ${m.hidden} & ${m.kept}/${m.rows} & ${rng(m.vsCompiled)} & ${rng(m.vsAnchor)} & ${rng(m.vsEager)} \\\\`
    : `${m.model} & ${m.layers} & ${m.hidden} & 0/${m.rows} & -- & -- & -- \\\\`));
  mac('dtxRows', d.rows); mac('dtxValid', d.valid); mac('dtxKept', d.kept); mac('dtxModels', ms.length);
  mac('dtxVsPytorch', rng(d.keptVsCompiled)); mac('dtxVsAnchor', rng(d.keptVsAnchor));
  const l = D.decode.longCache; mac('dlcRows', l.rows); mac('dlcValid', l.valid); mac('dlcKept', l.kept); mac('dlcVsPytorch', rng(l.keptVsCompiled)); mac('dlcVsAnchor', rng(l.keptVsAnchor));
}

// ---------------------------------------------------------------- census with the optimized kernel: rows of the pattern table
{
  const rows = [];
  for (const key of ['rope|H200|2.5.1', 'rope|A800|2.5.1']) {
    const c = D.census[key]; if (!c) continue;
    const kb = B => c.cells.filter(x => x.B === B).reduce((s, x) => s + x.kept, 0), nb = B => c.cells.filter(x => x.B === B).reduce((s, x) => s + x.n, 0);
    const tag = c.gpu === 'H200' ? 'Hcen' : 'Acen';
    mac(tag + 'Configs', c.configs); mac(tag + 'Kept', c.kept); mac(tag + 'Models', words[c.models.length] || c.models.length); mac(tag + 'Range', rng(c.keptRange));
    mac(tag + 'KeptBone', kb(1)); mac(tag + 'RowsBone', nb(1)); mac(tag + 'KeptBatched', kb(8) + kb(32)); mac(tag + 'RowsBatched', nb(8) + nb(32));
    rows.push(`${rows.length ? '' : '\\multirow{2}{=}{Q/K norm, RoPE, optimized kernel}'} & ${c.gpu} & ${c.configs} & -- & -- & ${c.kept} \\\\`);
  }
  write('recensus_patterns.tex', rows);
}

// ---------------------------------------------------------------- long sequences: fusion against anchor quality
{
  const lines = [];
  const keys = ['norm|H200|2.5.1', 'norm|H200|2.11.0', 'norm|H200|2.15.0', 'norm|A800|2.5.1'].filter(k => D.census[k]);
  let Ss = null;
  for (const k of keys) {
    const c = D.census[k], pts = c.cells.filter(x => x.B === 1).sort((a, b) => a.S - b.S);
    Ss = Ss || pts.map(p => p.S);
    const name = `\\multirow{2}{*}{${c.gpu}, ${c.torch.replace(/\.0$/, '')}${c.models.length > 1 ? `, ${c.models.length} models` : ''}}`;
    lines.push(`${name} & fused / same anchor & ${Ss.map(S => { const p = pts.find(x => x.S === S); return p ? f2(p.sameAnchor) : '--'; }).join(' & ')} \\\\`);
    lines.push(` & anchor path / compiled & ${Ss.map(S => { const p = pts.find(x => x.S === S); return p ? f2(p.anchorVsCompiled) : '--'; }).join(' & ')} \\\\`);
    if (k !== keys[keys.length - 1]) lines.push('\\addlinespace[2pt]');
    const last = pts[pts.length - 1], tag = { 'norm|H200|2.5.1': 'LsHa', 'norm|H200|2.11.0': 'LsHb', 'norm|H200|2.15.0': 'LsHc', 'norm|A800|2.5.1': 'LsA' }[k];
    mac(tag + 'Fusion', f2(last.sameAnchor)); mac(tag + 'Anchor', f2(last.anchorVsCompiled)); mac(tag + 'S', last.S);
  }
  write('longseq.tex', lines);
}
write('numbers_tiers.tex', ['% generated by analysis/gen_tiers_tex.js; do not edit', ...macros]);
