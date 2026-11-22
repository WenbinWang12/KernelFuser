/* Charts of the KernelFuser project page. Plain SVG, no dependency. All numbers come from assets/data.js. */
(function () {
  'use strict';
  const D = window.KF_DATA || {};
  const NS = 'http://www.w3.org/2000/svg';
  const $ = (s, r) => (r || document).querySelector(s);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const svgEl = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
  const html = (tag, attrs, parent, text) => { const e = document.createElement(tag); for (const k in (attrs || {})) e.setAttribute(k, attrs[k]); if (text !== undefined) e.textContent = text; if (parent) parent.appendChild(e); return e; };
  const x2 = v => v.toFixed(2) + '×';
  const x3 = v => v.toFixed(3) + '×';
  const range = r => !r ? '–' : (r[0].toFixed(2) === r[1].toFixed(2) ? x2(r[0]) : r[0].toFixed(2) + '–' + x2(r[1]));
  const redraws = [];

  // ---------- shared pieces
  function plot(host, height, draw) {
    host.innerHTML = '';
    const box = html('div', { class: 'plot' }, host);
    const tip = html('div', { class: 'tip', role: 'status' }, box);
    const render = () => {
      const old = box.querySelector('svg'); if (old) old.remove();
      const W = Math.max(320, box.clientWidth), H = typeof height === 'function' ? height(W) : height;
      const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: 'img' });
      box.insertBefore(svg, tip);
      draw(svg, W, H, {
        show(evt, title, lines) {
          tip.innerHTML = ''; html('b', {}, tip, title);
          lines.forEach(l => { const d = html('div', {}, tip); html('span', {}, d, l[0] + ' '); d.appendChild(document.createTextNode(l[1])); });
          const r = box.getBoundingClientRect(); let x = evt.clientX - r.left + 14, y = evt.clientY - r.top + 14;
          tip.style.opacity = 1; if (x + tip.offsetWidth > r.width) x = Math.max(0, evt.clientX - r.left - tip.offsetWidth - 14);
          tip.style.left = x + 'px'; tip.style.top = y + 'px';
        },
        hide() { tip.style.opacity = 0; }
      });
    };
    render(); redraws.push(render);
    return box;
  }
  function seg(host, options, onPick) {
    const g = html('div', { class: 'seg', role: 'group' }, host);
    options.forEach((o, i) => { const b = html('button', { type: 'button', 'aria-pressed': i === 0 ? 'true' : 'false' }, g, o.label);
      b.addEventListener('click', () => { g.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', 'false')); b.setAttribute('aria-pressed', 'true'); onPick(o.key); }); });
    return g;
  }
  function table(host, head, rows, label) {
    const d = html('details', { class: 'tbl' }, host); html('summary', {}, d, label || 'View as table');
    const sc = html('div', { class: 'scroll' }, d), t = html('table', {}, sc), tr = html('tr', {}, html('thead', {}, t));
    head.forEach(h => html('th', {}, tr, h)); const tb = html('tbody', {}, t);
    rows.forEach(r => { const row = html('tr', {}, tb); r.forEach(c => html('td', {}, row, c === null || c === undefined ? '–' : String(c))); });
  }
  function axisX(svg, x, ticks, y0, y1, fmt) {
    while (ticks.length > 3 && Math.abs(x(ticks[1]) - x(ticks[0])) < 58) ticks = ticks.filter((t, i) => i % 2 === 0);
    if (!ticks.includes(1)) ticks = ticks.concat([1]).sort((a, b) => a - b);
    ticks.forEach(t => { svgEl('line', { class: t === 1 ? 'one' : 'gridl', x1: x(t), x2: x(t), y1: y0, y2: y1 }, svg);
      const tx = svgEl('text', { class: 'tick', x: x(t), y: y1 + 16, 'text-anchor': 'middle' }, svg); tx.textContent = fmt(t); });
  }
  const ticksFor = (lo, hi, step) => { const a = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) a.push(Math.round(v * 1000) / 1000); return a; };

  // ---------- 1. whole-model decode step
  function decodeChart(host, set) {
    const ms = (set && set.models || []).filter(m => m.vsCompiled);
    if (!ms.length) { host.innerHTML = '<p class="note">No rows yet for this input set.</p>'; return; }
    const hi = Math.max(1.5, Math.ceil(Math.max(...ms.map(m => m.vsCompiled[1])) * 10) / 10);
    const lo = Math.min(1, Math.floor(Math.min(...ms.map(m => Math.min(m.vsCompiled[0], m.vsAnchor[0]))) * 10) / 10);
    plot(host, () => 34 + ms.length * 40 + 30, (svg, W, H, tip) => {
      const L = 118, R = 74, T = 12, x = v => L + (v - lo) / (hi - lo) * (W - L - R);
      axisX(svg, x, ticksFor(lo, hi, 0.1), T, T + ms.length * 40, v => v.toFixed(1) + '×');
      ms.forEach((m, i) => {
        const cy = T + i * 40 + 20;
        const nm = svgEl('text', { class: 'name', x: L - 12, y: cy + 4, 'text-anchor': 'end' }, svg); nm.textContent = m.model;
        [['vsCompiled', '--s1', -7], ['vsAnchor', '--s2', 7]].forEach(([k, col, dy]) => {
          const r = m[k], c = css(col), y = cy + dy;
          svgEl('line', { x1: x(1), x2: x(r[0]), y1: y, y2: y, stroke: c, 'stroke-width': 2, 'stroke-opacity': .28 }, svg);
          svgEl('line', { x1: x(r[0]), x2: x(r[1]), y1: y, y2: y, stroke: c, 'stroke-width': 6, 'stroke-linecap': 'round' }, svg);
          [r[0], r[1]].forEach(v => svgEl('circle', { cx: x(v), cy: y, r: 5, fill: c, stroke: css('--surface'), 'stroke-width': 2 }, svg));
        });
        const k = svgEl('text', { class: 'val', x: W - R + 12, y: cy + 4 }, svg); k.textContent = `${m.kept}/${m.rows} kept`;
        const hit = svgEl('rect', { class: 'hit', x: 0, y: cy - 20, width: W, height: 40 }, svg);
        hit.addEventListener('mousemove', e => tip.show(e, m.model, [['layers · width', `${m.layers} · ${m.hidden}`], ['vs torch.compile', range(m.vsCompiled)],
          ['fusion alone', range(m.vsAnchor)], ['vs eager PyTorch', range(m.vsEager)], ['kept', `${m.kept} of ${m.rows} (valid ${m.valid})`]]));
        hit.addEventListener('mouseleave', tip.hide);
      });
    });
    table(host, ['Model', 'Layers', 'Width', 'Kept', 'vs torch.compile', 'Fusion alone', 'vs eager'],
      ms.map(m => [m.model, m.layers, m.hidden, `${m.kept}/${m.rows}`, range(m.vsCompiled), range(m.vsAnchor), range(m.vsEager)]));
  }

  // ---------- 2. attention census heat map
  const hexRgb = h => { const n = parseInt(h.replace('#', ''), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
  function divColor(v) {
    const t = Math.max(-1, Math.min(1, Math.log(v) / Math.log(1.2)));
    const a = hexRgb(css('--div-mid')), b = hexRgb(css(t >= 0 ? '--div-hi' : '--div-lo')), k = Math.abs(t);
    const c = a.map((x, i) => Math.round(x + (b[i] - x) * k));
    const lum = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;
    return { fill: `rgb(${c.join(',')})`, ink: lum > 0.6 ? '#0b0b0b' : '#ffffff' };
  }
  function heatChart(host, c) {
    if (!c) { host.innerHTML = '<p class="note">No data.</p>'; return; }
    const Bs = [...new Set(c.cells.map(x => x.B))].sort((a, b) => a - b), Ss = [...new Set(c.cells.map(x => x.S))].sort((a, b) => a - b);
    plot(host, () => 30 + Bs.length * 62 + 26, (svg, W, H, tip) => {
      const L = 84, T = 26, cw = (W - L - 4) / Ss.length, ch = 62;
      Ss.forEach((S, j) => { const t = svgEl('text', { class: 'tick', x: L + j * cw + cw / 2, y: 16, 'text-anchor': 'middle' }, svg); t.textContent = 'S = ' + S; });
      Bs.forEach((B, i) => {
        const t = svgEl('text', { class: 'name', x: L - 12, y: T + i * ch + ch / 2 + 4, 'text-anchor': 'end' }, svg); t.textContent = 'batch ' + B;
        Ss.forEach((S, j) => {
          const cell = c.cells.find(x => x.B === B && x.S === S); if (!cell) return;
          const col = divColor(cell.best), X = L + j * cw + 1, Y = T + i * ch + 1;
          svgEl('rect', { x: X, y: Y, width: cw - 2, height: ch - 2, rx: 6, fill: col.fill }, svg);
          const v = svgEl('text', { x: X + cw / 2 - 1, y: Y + ch / 2 - 2, 'text-anchor': 'middle', style: `fill:${col.ink};font-size:15px;font-weight:700` }, svg); v.textContent = x2(cell.best);
          if (cw > 74) { const k = svgEl('text', { x: X + cw / 2 - 1, y: Y + ch / 2 + 15, 'text-anchor': 'middle', style: `fill:${col.ink};font-size:11px;opacity:.85` }, svg); k.textContent = `${cell.kept}/${cell.n} kept`; }
          const hit = svgEl('rect', { class: 'hit', x: X, y: Y, width: cw - 2, height: ch - 2 }, svg);
          hit.addEventListener('mousemove', e => tip.show(e, `batch ${B}, ${S} tokens`, [['vs best unfused', x3(cell.best)], ['vs same anchor', x3(cell.sameAnchor)], ['vs torch.compile', x3(cell.compiled)],
            ['vs eager SDPA', x3(cell.sdpa)], ['kept', `${cell.kept} of ${cell.n} configurations`]]));
          hit.addEventListener('mouseleave', tip.hide);
        });
      });
    });
    table(host, ['Batch', 'Tokens', 'Configs', 'Kept', 'vs best unfused', 'vs same anchor', 'vs torch.compile', 'vs eager SDPA'],
      c.cells.map(x => [x.B, x.S, x.n, x.kept, x3(x.best), x3(x.sameAnchor), x3(x.compiled), x3(x.sdpa)]));
  }

  // ---------- 3. long sequences: fusion gain against anchor quality
  function longChart(host, c) {
    const pts = c ? c.cells.filter(x => x.B === 1).sort((a, b) => a.S - b.S) : [];
    if (pts.length < 2) { host.innerHTML = '<p class="note">No data.</p>'; return; }
    const series = [['sameAnchor', '--s1', 'fused ÷ same anchor'], ['anchorVsCompiled', '--s2', 'anchor path ÷ torch.compile']];
    const vals = pts.flatMap(p => series.map(s => p[s[0]]));
    const lo = Math.min(0.6, Math.floor(Math.min(...vals) * 10) / 10), hi = Math.max(1.2, Math.ceil(Math.max(...vals) * 10) / 10);
    plot(host, 300, (svg, W, H, tip) => {
      const L = 46, R = 96, T = 14, B = 34, x = i => L + i / (pts.length - 1) * (W - L - R), y = v => T + (hi - v) / (hi - lo) * (H - T - B);
      ticksFor(lo, hi, 0.2).forEach(t => { svgEl('line', { class: 'gridl', x1: L, x2: W - R, y1: y(t), y2: y(t) }, svg);
        const tx = svgEl('text', { class: 'tick', x: L - 8, y: y(t) + 4, 'text-anchor': 'end' }, svg); tx.textContent = t.toFixed(1) + '×'; });
      svgEl('line', { class: 'one', x1: L, x2: W - R, y1: y(1), y2: y(1) }, svg);
      pts.forEach((p, i) => { const tx = svgEl('text', { class: 'tick', x: x(i), y: H - B + 18, 'text-anchor': 'middle' }, svg); tx.textContent = p.S; });
      const cap = svgEl('text', { class: 'tick', x: (L + W - R) / 2, y: H - 2, 'text-anchor': 'middle' }, svg); cap.textContent = 'sequence length (tokens), batch 1';
      const ends = series.map(s => y(pts[pts.length - 1][s[0]]));
      if (Math.abs(ends[0] - ends[1]) < 16) { const m = (ends[0] + ends[1]) / 2, up = ends[0] < ends[1] ? 0 : 1; ends[up] = m - 9; ends[1 - up] = m + 9; }
      series.forEach((s, k) => {
        const c2 = css(s[1]);
        svgEl('path', { d: pts.map((p, i) => (i ? 'L' : 'M') + x(i) + ' ' + y(p[s[0]])).join(' '), fill: 'none', stroke: c2, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
        pts.forEach((p, i) => svgEl('circle', { cx: x(i), cy: y(p[s[0]]), r: 4.5, fill: c2, stroke: css('--surface'), 'stroke-width': 2 }, svg));
        const lab = svgEl('text', { class: 'val', x: W - R + 12, y: ends[k] + 4 }, svg); lab.textContent = `${k ? 'anchor' : 'fusion'} ${x2(pts[pts.length - 1][s[0]])}`;
      });
      const guide = svgEl('line', { class: 'base', y1: T, y2: H - B, x1: 0, x2: 0, visibility: 'hidden' }, svg);
      const hit = svgEl('rect', { class: 'hit', x: L - 10, y: T, width: W - L - R + 20, height: H - T - B }, svg);
      hit.addEventListener('mousemove', e => { const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
        const i = Math.max(0, Math.min(pts.length - 1, Math.round((px - L) / (W - L - R) * (pts.length - 1)))), p = pts[i];
        guide.setAttribute('x1', x(i)); guide.setAttribute('x2', x(i)); guide.setAttribute('visibility', 'visible');
        tip.show(e, `${p.S} tokens`, [['fused ÷ same anchor', x3(p.sameAnchor)], ['anchor path ÷ torch.compile', x3(p.anchorVsCompiled)], ['fused ÷ torch.compile', x3(p.compiled)], ['kept', `${p.kept} of ${p.n}`]]); });
      hit.addEventListener('mouseleave', () => { tip.hide(); guide.setAttribute('visibility', 'hidden'); });
    });
    table(host, ['Tokens', 'Fused ÷ same anchor', 'Anchor path ÷ torch.compile', 'Fused ÷ torch.compile', 'Kept'], pts.map(p => [p.S, x3(p.sameAnchor), x3(p.anchorVsCompiled), x3(p.compiled), `${p.kept}/${p.n}`]));
  }

  // ---------- 4. training step on WikiText-2
  function trainChart(host, t) {
    const rows = (t && t.rows || []).filter(r => r.timed && r.vsCompiled[0] !== null);
    if (!rows.length) { host.innerHTML = '<p class="note">No rows yet.</p>'; return; }
    const mean = a => (a[0] + (a[1] === null ? a[0] : a[1])) / 2;
    const hi = Math.max(1.1, Math.ceil(Math.max(...rows.map(r => Math.max(...r.vsEager.filter(v => v !== null)))) * 50) / 50), lo = 0.98;
    plot(host, () => 22 + rows.length * 36 + 30, (svg, W, H, tip) => {
      const L = 190, R = 86, T = 10, x = v => L + (Math.max(lo, v) - lo) / (hi - lo) * (W - L - R);
      axisX(svg, x, ticksFor(lo, hi, 0.02), T, T + rows.length * 36, v => v.toFixed(2) + '×');
      rows.forEach((r, i) => {
        const cy = T + i * 36 + 18;
        const nm = svgEl('text', { class: 'name', x: L - 12, y: cy + 4, 'text-anchor': 'end' }, svg); nm.textContent = `${r.model}  ${r.B}×${r.S}`;
        [['vsCompiled', '--s1', -6], ['vsEager', '--s2', 6]].forEach(([k, col, dy]) => {
          const c = css(col), y = cy + dy, v = mean(r[k]);
          svgEl('line', { x1: x(1), x2: x(v), y1: y, y2: y, stroke: c, 'stroke-width': 2, 'stroke-opacity': .45 }, svg);
          svgEl('circle', { cx: x(v), cy: y, r: 5, fill: c, stroke: css('--surface'), 'stroke-width': 2 }, svg);
        });
        const k = svgEl('text', { class: 'val', x: W - R + 12, y: cy + 4 }, svg); k.textContent = r.kept ? 'kept' : (r.correct ? 'no gain' : 'rejected');
        const hit = svgEl('rect', { class: 'hit', x: 0, y: cy - 18, width: W, height: 36 }, svg);
        hit.addEventListener('mousemove', e => tip.show(e, `${r.model} · ${r.params}B · ${r.optimizer}`, [['vs torch.compile', r.vsCompiled.map(v => v === null ? '–' : x3(v)).join(' / ')],
          ['vs eager', r.vsEager.map(v => v === null ? '–' : x3(v)).join(' / ')], ['same code twice', r.aa === null ? '–' : x3(r.aa)],
          ['held-out perplexity', `${r.pplBefore} → ${r.pplRef} (reference), ${r.pplCand} (fused)`], ['correctness', r.correct ? 'passes' : 'fails'], ['fused throughput', r.tokensPerSec ? r.tokensPerSec.toLocaleString('en-US') + ' tokens/s' : '–']]));
        hit.addEventListener('mouseleave', tip.hide);
      });
    });
    table(host, ['Model', 'Params', 'Shape', 'Optimizer', 'vs torch.compile (initial / retest)', 'vs eager (initial / retest)', 'Perplexity before', 'reference', 'fused', 'Correct', 'Kept'],
      rows.map(r => [r.model, r.params + 'B', `${r.B}×${r.S}`, r.optimizer, r.vsCompiled.map(v => v === null ? '–' : v.toFixed(3)).join(' / '), r.vsEager.map(v => v === null ? '–' : v.toFixed(3)).join(' / '),
        r.pplBefore, r.pplRef, r.pplCand, r.correct ? 'yes' : 'no', r.kept ? 'yes' : 'no']));
  }

  // ---------- wiring
  function tiles() {
    const set = (id, v) => { const e = document.getElementById(id); if (e && v !== undefined && v !== null) e.textContent = v; };
    const c = D.census || {}, h = c['rope|H200|2.5.1'], dr = D.decode && D.decode.random, lh = (D.lossHead || []).filter(x => x.gpu === 'H200' && x.set === 'lmhead_train')[0];
    if (h) { set('k-census', `${h.kept}/${h.configs}`); set('k-census-r', range(h.keptRange)); set('k-census-m', h.models.length); }
    if (dr) { set('k-decode', range(dr.keptVsCompiled)); set('k-decode-n', `${dr.kept} of ${dr.valid}`); set('k-decode-m', dr.models.length); set('k-decode-a', range(dr.keptVsAnchor)); }
    if (lh) { set('k-loss', `${lh.kept}/${lh.rows}`); set('k-loss-r', range(lh.vsCompiled)); }
    const tr = D.training; if (tr) { set('k-train', `${tr.kept}/${tr.timed}`); set('k-train-m', tr.models); }
    set('gen-date', D.generated);
    document.querySelectorAll('[data-fill]').forEach(e => { try { const v = e.getAttribute('data-fill').split('.').reduce((o, k) => o[k], D); if (v !== undefined && v !== null) e.textContent = Array.isArray(v) ? range(v) : v; } catch (err) { /* keep the fallback text */ } });
  }
  function init() {
    tiles();
    const dec = $('#chart-decode'); if (dec) {
      const sets = [['random', 'Seeded token ids'], ['text', 'WikiText-2 prompts'], ['longCache', 'Long cache']].filter(s => D.decode && D.decode[s[0]] && D.decode[s[0]].models.some(m => m.vsCompiled));
      const body = html('div'); if (sets.length > 1) seg($('#ctl-decode'), sets.map(s => ({ key: s[0], label: s[1] })), k => decodeChart(body, D.decode[k]));
      dec.appendChild(body); if (sets.length) decodeChart(body, D.decode[sets[0][0]]);
    }
    const lab = k => { const c = D.census[k]; return c.gpu + " · PyTorch " + c.torch.replace(/[.]0$/, "") + " · " + (c.models.length > 1 ? c.models.length + " models" : c.models[0]); };
    const heat = $('#chart-census'); if (heat && D.census) {
      const keys = ['rope|H200|2.5.1', 'rope|A800|2.5.1', 'rope|H100|2.5.1', 'rope|H200|2.11.0', 'rope|H200|2.15.0'].filter(k => D.census[k]);
      const body = html('div'), note = $('#note-census');
      const draw = k => { heatChart(body, D.census[k]); const c = D.census[k]; if (note) note.textContent = `${c.kept} of ${c.configs} configurations kept` + (c.keptRange ? `, ${range(c.keptRange)} over the best unfused implementation` : '') + `. Models: ${c.models.join(', ')}.`; };
      seg($('#ctl-census'), keys.map(k => ({ key: k, label: lab(k) })), draw); heat.appendChild(body); if (keys.length) draw(keys[0]);
    }
    const lng = $('#chart-long'); if (lng && D.census) {
      const keys = ['norm|H200|2.5.1', 'norm|H200|2.11.0', 'norm|H200|2.15.0', 'norm|A800|2.5.1'].filter(k => D.census[k]);
      const body = html('div'); seg($('#ctl-long'), keys.map(k => ({ key: k, label: lab(k) })), k => longChart(body, D.census[k])); lng.appendChild(body); if (keys.length) longChart(body, D.census[keys[0]]);
    }
    const trn = $('#chart-train'); if (trn) trainChart(trn, D.training);
    let timer; window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(() => redraws.forEach(f => f()), 120); });
    const btn = $('#theme'); if (btn) btn.addEventListener('click', () => {
      const dark = document.documentElement.getAttribute('data-theme') === 'dark' || (!document.documentElement.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
      document.documentElement.setAttribute('data-theme', dark ? 'light' : 'dark'); try { localStorage.setItem('kf-theme', dark ? 'light' : 'dark'); } catch (e) { /* storage unavailable */ }
      redraws.forEach(f => f());
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
