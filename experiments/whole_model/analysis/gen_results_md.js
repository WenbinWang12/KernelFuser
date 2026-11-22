// Writes the summary README of a results folder from data.js (the output of build_public_data.js).
//   node analysis/gen_results_md.js <path to data.js> <output README.md>
const fs = require('fs');
global.window = {};
eval(fs.readFileSync(process.argv[2], 'utf8'));
const D = window.KF_DATA;
const rg = r => !r ? '-' : `${r[0].toFixed(2)}-${r[1].toFixed(2)}x`;
const L = [];
L.push('# Open-model campaign, November 2026', '',
  `Generated ${D.generated} by \`experiments/whole_model/analysis/gen_results_md.js\` from the measurement receipts.`,
  'The CSV files in this folder hold every measured row; receipts themselves are not published because they carry host identifiers.',
  'A configuration is *kept* only if it passes the correctness gate and beats every unfused baseline by more than 1% in an initial measurement and in a retest.', '');

L.push('## Whole-model decode step (`decode_step_a800.csv`)', '',
  'One new token per sequence over a key/value cache, five fused kernels per layer, against eager PyTorch, `torch.compile` (default and max-autotune) and the same kernels without fusion. A800.', '',
  '| Inputs | Models | Rows | Valid | Kept | Kept rows vs torch.compile | Fusion alone (vs same kernels) |', '|---|---:|---:|---:|---:|---|---|');
for (const k of ['random', 'text', 'longCache']) { const d = D.decode[k]; if (d && d.rows) L.push(`| ${d.label} | ${d.models.length} | ${d.rows} | ${d.valid} | ${d.kept} | ${rg(d.keptVsCompiled)} | ${rg(d.keptVsAnchor)} |`); }
L.push('', '| Model (seeded token ids) | Layers | Width | Kept | vs torch.compile | Fusion alone | vs eager |', '|---|---:|---:|---|---|---|---|');
for (const m of D.decode.random.models) L.push(`| ${m.model} | ${m.layers} | ${m.hidden} | ${m.kept}/${m.rows} | ${rg(m.vsCompiled)} | ${rg(m.vsAnchor)} | ${rg(m.vsEager)} |`);

L.push('', '## Attention census with the optimized fused kernel (`attention_census_optimized_kernel.csv`)', '',
  'Fused kernel against the best of: the same attention kernel after separate kernels, `torch.compile`, eager SDPA.', '',
  '| Fusion | GPU | PyTorch | Models | Configurations | Kept | Kept rows vs best unfused |', '|---|---|---|---:|---:|---:|---|');
for (const c of Object.values(D.census)) L.push(`| ${c.fusion} | ${c.gpu} | ${c.torch} | ${c.models.length} | ${c.configs} | ${c.kept} | ${rg(c.keptRange)} |`);

L.push('', '## Training step on WikiText-2 with the fused loss (`training_step_wikitext2_a800.csv`)', '',
  '100 timed steps per arm after 10 warm-up steps. Speedups are medians of the per-step ratio reference/fused, initial / retest.',
  'Correct: one-step gradients within 1.5x of the worse PyTorch arm\'s distance to an FP32 oracle, loss trajectory and held-out loss within 1.5x of the spread between equivalent PyTorch runs.', '',
  '| Model | Params | Shape | Optimizer | vs torch.compile | vs eager | Held-out perplexity (before, reference, fused) | Correct | Kept |', '|---|---:|---|---|---|---|---|---|---|');
const f3 = v => v === null ? '-' : v.toFixed(3);
for (const r of D.training.rows) if (r.timed) L.push(`| ${r.model} | ${r.params}B | ${r.B}x${r.S} | ${r.optimizer} | ${r.vsCompiled.map(f3).join(' / ')} | ${r.vsEager.map(f3).join(' / ')} | ${r.pplBefore}, ${r.pplRef}, ${r.pplCand} | ${r.correct ? 'yes' : 'no'} | ${r.kept ? 'yes' : 'no'} |`);
const bad = D.training.rows.filter(r => !r.timed);
if (bad.length) L.push('', 'Rows without a result: ' + bad.map(r => `${r.model} ${r.B}x${r.S} ${r.optimizer}` + (r.error ? ` (${r.error})` : '')).join('; ') + '.');

L.push('', '## Fused loss and gradient at the LM head (`fused_loss_lm_head.csv`)', '', '| GPU | PyTorch | Set | Heads x token counts | Kept | vs compiled head | vs eager head |', '|---|---|---|---:|---:|---|---|');
for (const h of D.lossHead) L.push(`| ${h.gpu} | ${h.torch} | ${h.set} | ${h.rows} | ${h.kept} | ${rg(h.vsCompiled)} | ${rg(h.vsEager)} |`);

L.push('', '## Whole-model prefill with the fused attention kernel (`prefill_whole_model.csv`)', '', '| GPU | Rows | Kept | Fused vs fastest unfused model |', '|---|---:|---:|---|');
for (const p of D.prefill) L.push(`| ${p.gpu} | ${p.rows} | ${p.kept} | ${rg(p.range)} |`);
L.push('', 'Attention is a small share of prefill, so the kernel-level gain does not carry over; the gate keeps the unfused model.', '');
fs.writeFileSync(process.argv[3], L.join('\n'));
console.log('written', process.argv[3], L.length, 'lines');
