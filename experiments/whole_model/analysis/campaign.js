// The A800 campaign as one matrix (EXPERIMENT_DESIGN.md describes it).  Single source for the stage list.
//   node analysis/campaign.js plan <out.tsv> [older plan to append]   writes the queue plan, in priority order
//   node analysis/campaign.js status <results dir>                    prints what is done per group
const fs = require('fs'), path = require('path');
const M = process.env.KF_MODELS || 'models';                       // directory with one folder per checkpoint
const DATA = process.env.KF_TEXT || 'data/wikitext2_train.txt';      // WikiText-2 train split (raw text)
const LOCK = process.env.KF_LOCK || 'mem.lock';                      // lock directory shared by the queue workers
const groups = [];
const group = (id, title) => { const g = { id, title, stages: [] }; groups.push(g); return g; };
const stage = (g, name, py, deps, cmd, note) => g.stages.push({ name, py, deps, cmd, note: note || '' });

// ---------------------------------------------------------------- T: training step with the fused loss, WikiText-2
// 100 timed + 10 warm-up steps per arm, paired with eager and compiled PyTorch, retest, self-comparison,
// held-out perplexity, one copy of the checkpoint (--lean).  Optimizer is a factor of the design.
const train = (g, name, model, shapes, opt, py = 'formal', note = '') => stage(g, name, py, '-',
  `train_step_ce.py --model-dir ${M}/${model} --shapes ${shapes} --steps 100 --warmup 10 --data ${DATA} --retest --aa --compile --lean --mem-lock ${LOCK}` +
  (opt === 'adamw' ? '' : ` --optimizer ${opt}`) + ` --output results/${name}.json`, note);
const TA = group('T-AdamW', 'Training step, fused loss, WikiText-2, AdamW (small and medium tiers; 8B at batch 1)');
train(TA, 'train3_Qwen3-0.6B', 'Qwen3-0.6B', '8x512,4x1024,16x256', 'adamw');
train(TA, 'train3_Qwen3-4B', 'Qwen3-4B', '4x512', 'adamw');
train(TA, 'train3_Qwen3-1.7B', 'Qwen3-1.7B', '8x512', 'adamw');
train(TA, 'train3_SmolLM2-135M', 'SmolLM2-135M-Instruct', '8x512', 'adamw');
train(TA, 'train3_SmolLM2-360M', 'SmolLM2-360M-Instruct', '8x512', 'adamw');
train(TA, 'train3_Qwen2.5-0.5B', 'Qwen2.5-0.5B-Instruct', '8x512', 'adamw');
train(TA, 'train3_Gemma-3-270M', 'Gemma-3-270M', '8x512', 'adamw');
train(TA, 'train3_Gemma-3-1B', 'Gemma-3-1B', '8x512', 'adamw');
train(TA, 'train3_Qwen2.5-1.5B', 'Qwen2.5-1.5B-Instruct', '8x512', 'adamw');
train(TA, 'train3_SmolLM2-1.7B', 'SmolLM2-1.7B-Instruct', '8x512', 'adamw');
train(TA, 'train3_Gemma-3-4B', 'Gemma-3-4B', '4x512', 'adamw');
train(TA, 'train3_Qwen3-8B_adamw', 'Qwen3-8B', '1x512', 'adamw', 'formal', 'model, gradients and AdamW state take about 66 GB; may not fit');
const TM = group('T-Muon', 'Training step, fused loss, WikiText-2, Muon (hidden matrices) + AdamW (embedding, head, gains)');
train(TM, 'train3_Qwen3-0.6B_muon', 'Qwen3-0.6B', '8x512', 'muon');
train(TM, 'train3_SmolLM2-360M_muon', 'SmolLM2-360M-Instruct', '8x512', 'muon');
train(TM, 'train3_Qwen3-1.7B_muon', 'Qwen3-1.7B', '8x512', 'muon');
train(TM, 'train3_Qwen3-4B_muon', 'Qwen3-4B', '4x512', 'muon');
train(TM, 'train3_Qwen3-8B_muon', 'Qwen3-8B', '2x512', 'muon');
const TS = group('T-SGD', 'Training step, fused loss, WikiText-2, SGD (large tier, where optimizer state does not fit; two bridges)');
train(TS, 'train3_Qwen3-14B_sgd', 'Qwen3-14B', '1x512', 'sgd');
train(TS, 'train3_Qwen3-8B_sgd', 'Qwen3-8B', '2x512', 'sgd');
train(TS, 'train3_Llama-3.1-8B_sgd', 'Llama-3.1-8B', '2x512', 'sgd');
train(TS, 'train3_Qwen3.5-9B_sgd', 'Qwen3.5-9B', '2x512', 'sgd', 't215', 'hybrid-attention architecture; needs the newer transformers of the 2.15 stack');
train(TS, 'train3_Qwen3-4B_sgd', 'Qwen3-4B', '4x512', 'sgd', 'formal', 'bridge: same checkpoint and shape as the AdamW row');
train(TS, 'train3_Qwen3-0.6B_sgd', 'Qwen3-0.6B', '8x512', 'sgd', 'formal', 'bridge');

// ---------------------------------------------------------------- D: whole-model decode step, WikiText-2 prompts
const decode = (g, name, model, shapes) => stage(g, name, 'formal', '-',
  `e2e_decode2.py --model-dir ${M}/${model} --autotune --data ${DATA}` + (shapes ? ` --shapes ${shapes}` : '') + ` --output results/${name}.json`);
const DD = group('D', 'Decode step, five fused kernels per layer, WikiText-2 prompts; 1/4/16 sequences, cache of 256 or 1024 tokens');
for (const [n, m] of [['Qwen3-0.6B', 'Qwen3-0.6B'], ['Qwen3-4B', 'Qwen3-4B'], ['Qwen3-14B', 'Qwen3-14B'], ['Qwen3-8B', 'Qwen3-8B'], ['Qwen3-1.7B', 'Qwen3-1.7B'],
  ['SmolLM2-135M', 'SmolLM2-135M-Instruct'], ['SmolLM2-360M', 'SmolLM2-360M-Instruct'], ['Qwen2.5-0.5B', 'Qwen2.5-0.5B-Instruct'], ['Qwen2.5-1.5B', 'Qwen2.5-1.5B-Instruct'],
  ['SmolLM2-1.7B', 'SmolLM2-1.7B-Instruct'], ['Llama-3.1-8B', 'Llama-3.1-8B']]) decode(DD, `dec5_${n}`, m);
const DL = group('D-long', 'Decode step over a long cache (2048 and 4096 tokens), one checkpoint per tier');
for (const m of ['Qwen3-0.6B', 'Qwen3-4B', 'Qwen3-8B']) decode(DL, `dec5L_${m}`, m, '1x2048,1x4096,4x2048,4x4096');

// ---------------------------------------------------------------- A: attention census with the optimized kernel
const AL = group('A-long', 'Q/K norm + RoPE in attention at 1024 to 4096 tokens (the census of nine models up to 512 tokens is complete)');
for (const m of ['Qwen3-8B', 'Qwen3-0.6B', 'Gemma-3-27B']) stage(AL, `census_rope_long_${m}`, 'formal', 'census_rope_Qwen3-8B',
  `recensus_driver.py --mode rope --models ${m} --sizes 1024,2048,4096 --batches 1,8 --causal-modes false,true --bn 128 --output results/census_rope_long_${m}.json --trace-dir traces/census_rope_long_${m}`);

// ---------------------------------------------------------------- H: fused loss at the LM head
const HH = group('H', 'Fused loss and gradient at the LM heads of the hybrid-attention and MoE checkpoints (15 other heads are complete)');
stage(HH, 'lmhead_train_arch', 'formal', 'lmhead_train_smoke',
  `lmhead_bench.py --mode train --models Qwen3.5-9B,Qwen3.8-27B,Qwen3.5-35B-A3B --tokens 1024,4096 --models-dir ${M} --output results/lmhead_train_arch.json`);

// priority: the three Qwen3 tiers under each optimizer first, then decode, the long-sequence diagnosis, the other families
const ORDER = ['train3_Qwen3-0.6B', 'train3_Qwen3-4B', 'train3_Qwen3-14B_sgd', 'train3_Qwen3-8B_sgd', 'dec5_Qwen3-0.6B', 'dec5_Qwen3-4B', 'dec5_Qwen3-14B',
  'train3_Qwen3-0.6B_muon', 'train3_Qwen3-4B_muon', 'train3_Qwen3-8B_muon', 'train3_Qwen3-1.7B', 'dec5_Qwen3-8B', 'dec5_Qwen3-1.7B', 'train3_Qwen3-1.7B_muon',
  'census_rope_long_Qwen3-8B', 'dec5L_Qwen3-0.6B', 'dec5L_Qwen3-4B', 'dec5L_Qwen3-8B', 'census_rope_long_Qwen3-0.6B', 'census_rope_long_Gemma-3-27B', 'lmhead_train_arch',
  'train3_SmolLM2-360M_muon', 'train3_Qwen3-8B_adamw'];
const all = groups.flatMap(g => g.stages.map(s => ({ ...s, group: g.id })));
const ordered = [...ORDER.map(n => all.find(s => s.name === n)), ...all.filter(s => !ORDER.includes(s.name))];
if (ordered.some(s => !s)) { console.error('ORDER names a stage that does not exist'); process.exit(1); }

const [cmd, arg, older] = process.argv.slice(2);
if (cmd === 'plan') {
  const lines = ['# A800 campaign matrix (analysis/campaign.js, EXPERIMENT_DESIGN.md). Stage order = priority.'];
  for (const s of ordered) lines.push([s.name, s.py, s.deps, s.cmd].join('\t'));
  let text = lines.join('\n') + '\n';
  if (older) text += '# ---- group S: repeats on PyTorch 2.11 / 2.15 and older stages (from the plan before 2026-11-22)\n' + fs.readFileSync(older, 'utf8');
  fs.writeFileSync(arg, text);
  console.log(`${ordered.length} matrix stages written to ${arg}` + (older ? ` (+ older plan appended)` : ''));
} else if (cmd === 'status') {
  const have = n => { try { const d = JSON.parse(fs.readFileSync(path.join(arg, n + '.json'), 'utf8')); return d.status === 'running' ? 'running' : 'done'; } catch (e) { return 'pending'; } };
  let tot = 0, done = 0;
  for (const g of groups) {
    const st = g.stages.map(s => [s.name, have(s.name)]);
    const d = st.filter(x => x[1] === 'done').length; tot += st.length; done += d;
    console.log(`\n[${g.id}] ${g.title}\n   ${d}/${st.length} done` + (st.some(x => x[1] === 'running') ? `, running: ${st.filter(x => x[1] === 'running').map(x => x[0]).join(', ')}` : '') +
      (st.some(x => x[1] === 'pending') ? `\n   pending: ${st.filter(x => x[1] === 'pending').map(x => x[0]).join(', ')}` : ''));
  }
  console.log(`\nmatrix: ${done}/${tot} stages have a finished receipt`);
} else console.log('usage: node analysis/campaign.js plan <out.tsv> [older plan] | status <results dir>');
