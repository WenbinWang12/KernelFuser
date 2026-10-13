// node experiments/rescore_legacy_e1.js H100.json H200.json output-directory
// Post-hoc descriptive analysis only: additional arms were not candidate-paired.
const fs = require('fs');
const path = require('path');
const [h100, h200, out] = process.argv.slice(2);
if (!h100 || !h200 || !out) throw Error('H100.json H200.json output-directory required');
const median = xs => {
  if (!xs.length || xs.some(x => !Number.isFinite(x))) throw Error('Missing or nonfinite latency');
  const a = [...xs].sort((x,y) => x-y), n=a.length;
  return n%2 ? a[(n-1)/2] : (a[n/2-1]+a[n/2])/2;
};
const gm = a => Math.exp(a.reduce((s,x) => s+Math.log(x),0)/a.length);
const rows=[];
for (const [gpu,file] of [['H100',h100],['H200',h200]]) {
  const d=JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/,''));
  for (const [model,m] of Object.entries(d.models)) for (const r of m.rows) {
    const lat = key => median(r.timing[key].rounds.map(x => x.stats.p50_ms));
    const t={anchor:lat('t_F_unfused'),compiled:lat('torchinductor_compiled'),vendor:lat('vendor_sdpa')};
    const fused=lat('t_F_fused');
    const fastest=Object.keys(t).reduce((a,b) => t[a]<=t[b]?a:b);
    rows.push({gpu,model,B:1,S:r.S,causal:r.causal,head_dim:m.geometry.D,
      anchor_ms:t.anchor,compiled_ms:t.compiled,vendor_ms:t.vendor,fused_ms:fused,
      fastest_arm:fastest,speedup:t[fastest]/fused});
  }
}
const summarize = a => ({n:a.length,wins:a.filter(r=>r.speedup>1).length,
  gm:gm(a.map(r=>r.speedup)),min:Math.min(...a.map(r=>r.speedup)),
  max:Math.max(...a.map(r=>r.speedup))});
const summary={scope:'Legacy B=1 norm-only proxy; no RoPE; post-hoc medians, not paired significance',
  estimator:'For each arm, median of round p50s; min of three unfused arm medians divided by candidate median',devices:{}};
for (const gpu of ['H100','H200']) {
  const a=rows.filter(r=>r.gpu===gpu);
  summary.devices[gpu]={all:summarize(a),short:summarize(a.filter(r=>r.S<=256)),
    by_sequence:Object.fromEntries([...new Set(a.map(r=>r.S))].map(s=>[s,summarize(a.filter(r=>r.S===s))]))};
}
fs.mkdirSync(out,{recursive:true});
fs.writeFileSync(path.join(out,'legacy_e1_strongest_summary.json'),JSON.stringify(summary,null,2)+'\n');
const keys=Object.keys(rows[0]);
fs.writeFileSync(path.join(out,'legacy_e1_strongest.csv'),[keys.join(','),...rows.map(r=>keys.map(k=>r[k]).join(','))].join('\n')+'\n');
console.log(JSON.stringify(summary,null,2));
