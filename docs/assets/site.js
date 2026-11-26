// Renders every number on the page from window.KF_DATA (assets/data.js, generated from the paper sources)
// and every link from window.KF_CONFIG (assets/config.js). No number is typed into index.html.
(function () {
  "use strict";
  const D = window.KF_DATA, CFG = window.KF_CONFIG || {};
  const $ = (s) => document.querySelector(s), $$ = (s) => Array.from(document.querySelectorAll(s));
  const C = { gemm: "#4C78A8", mem: "#F2A33A", fused: "#3FA35B", hbm: "#D64545", ink: "#2B2F36", soft: "#7C8490", bar: "#DDE0E4", fsoft: "#EEF7F0", line: "#D3D9E1" };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const get = (p) => p.split(".").reduce((o, k) => (o == null ? o : o[k]), D);

  // ---- config: links and authors (anonymous by default)
  const repo = $("#btn-code"), paper = $("#btn-paper");
  if (CFG.repoUrl) { repo.href = CFG.repoUrl; repo.classList.remove("off"); repo.querySelector("small").textContent = "Repository"; $$("[data-repo]").forEach((a) => { a.href = CFG.repoUrl; a.hidden = false; }); }
  if (CFG.paperUrl) { paper.href = CFG.paperUrl; paper.classList.remove("off"); paper.querySelector("small").textContent = "PDF"; }
  if (CFG.authorsHtml) { $("#byline").innerHTML = CFG.authorsHtml; $("#affil").innerHTML = CFG.affiliationsHtml || ""; $("#eyebrow").hidden = true; }

  if (!D) { document.body.insertAdjacentHTML("afterbegin", '<p style="padding:12px 24px;background:#fdecec">assets/data.js is missing. Run <code>node site/tools/build_data.mjs</code>.</p>'); return; }
  $$("[data-k]").forEach((e) => { const v = get(e.dataset.k); e.textContent = v == null ? "?" : v; });

  // ---- diagrams (same vocabulary as the paper figures: blue anchor, orange operator, green fused kernel, red HBM round trip)
  const defs = `<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker>
    <pattern id="hatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="9" height="9" fill="${C.fsoft}"/><rect width="4" height="9" fill="${C.fused}"/></pattern></defs>`;
  const opC = (x, y, t) => `<circle cx="${x}" cy="${y}" r="26" fill="${C.mem}"/><text x="${x}" y="${y + 5}" text-anchor="middle" font-size="14" font-weight="700" fill="#fff">${t}</text>`;
  const anc = (x, y, t) => `<rect x="${x - 56}" y="${y - 26}" width="112" height="52" rx="5" fill="${C.gemm}"/><text x="${x}" y="${y + 6}" text-anchor="middle" font-size="17" font-weight="700" fill="#fff">${t}</text>`;
  const hbm = `<rect x="14" y="196" width="392" height="46" rx="6" fill="${C.bar}"/><text x="28" y="226" font-size="16" font-weight="700" fill="${C.ink}">HBM</text>`;
  const trip = (xa, xb) => { const m = (xa + xb) / 2; return `<path d="M${xa},98 C${xa},160 ${m - 7},150 ${m - 7},194 M${m + 7},194 C${m + 7},150 ${xb},160 ${xb},104" fill="none" stroke="${C.hbm}" stroke-width="2.5" marker-end="url(#ah)"/>`; };
  const io = `<path d="M40,196 V70 H${88 - 30}" fill="none" stroke="${C.soft}" stroke-width="2" marker-end="url(#ah)"/><path d="M${332 + 28},70 H384 V194" fill="none" stroke="${C.soft}" stroke-width="2" marker-end="url(#ah)"/>`;
  $("#dg-separate").innerHTML = `<svg viewBox="0 0 420 256" role="img" aria-label="Three separate kernels. The normalization, the GEMM and the residual addition each write their result to HBM, and the next kernel reads it back.">${defs}${hbm}${io}${trip(88, 210)}${trip(210, 332)}${opC(88, 70, "norm")}${anc(210, 70, "GEMM")}${opC(332, 70, "add")}</svg>`;
  const cut = (m) => `<path d="M${m - 12},116 V140 H${m + 12} V116" fill="none" stroke="${C.hbm}" stroke-width="2" stroke-dasharray="6 4"/><text x="${m}" y="147" text-anchor="middle" font-size="18" font-weight="700" fill="${C.hbm}" stroke="#fff" stroke-width="4" paint-order="stroke">×</text>`;
  $("#dg-fused").innerHTML = `<svg viewBox="0 0 420 256" role="img" aria-label="One fused kernel. The normalization runs as a prologue and the residual addition as an epilogue of the GEMM, so the two intermediate results never pass through HBM.">${defs}${hbm}${io}
    <rect x="52" y="28" width="316" height="84" rx="14" fill="${C.fsoft}" stroke="${C.fused}" stroke-width="3"/>
    <line x1="116" y1="70" x2="151" y2="70" stroke="${C.fused}" stroke-width="3" marker-end="url(#ah)"/><line x1="268" y1="70" x2="303" y2="70" stroke="${C.fused}" stroke-width="3" marker-end="url(#ah)"/>
    ${cut(134)}${cut(286)}${opC(88, 70, "norm")}${anc(210, 70, "GEMM")}${opC(332, 70, "add")}</svg>`;

  // schematic of f and c at two input shapes (lengths are illustrative, as in the overview video)
  const costSvg = (tG, ops, c, win) => { const X = 64, s = 3.1, w1 = tG * s, w2 = ops * s, w3 = tG * c * s;
    return `<svg viewBox="0 0 420 150" role="img" aria-label="${win ? "The fused kernel is shorter than the unfused path: fuse." : "The fused kernel is longer than the unfused path: keep the original path."}">${defs}
    <text x="${X - 8}" y="39" text-anchor="end" font-size="13" fill="${C.soft}">unfused</text><text x="${X - 8}" y="95" text-anchor="end" font-size="13" fill="${C.soft}">fused</text>
    <rect x="${X}" y="22" width="${w1}" height="26" rx="3" fill="${C.gemm}"/><rect x="${X + w1 + 2}" y="22" width="${w2 - 2}" height="26" rx="3" fill="${C.mem}"/>
    <text x="${X + w1 + w2 / 2}" y="16" text-anchor="middle" font-size="16" font-style="italic" font-family="Georgia,serif" fill="${C.ink}">f</text>
    <rect x="${X}" y="78" width="${w1}" height="26" rx="3" fill="${C.gemm}"/><rect x="${X + w1 + 2}" y="79" width="${w3 - 2}" height="24" rx="3" fill="url(#hatch)" stroke="${C.fused}" stroke-width="2"/>
    <text x="${X + w1 + w3 / 2}" y="124" text-anchor="middle" font-size="16" font-style="italic" font-family="Georgia,serif" fill="${C.ink}">c</text>
    <line x1="${X + w1 + w2}" y1="14" x2="${X + w1 + w2}" y2="112" stroke="${C.soft}" stroke-width="1.5" stroke-dasharray="4 4"/>
    <text x="${X}" y="143" font-size="13.5" fill="${C.ink}"><tspan font-weight="700" fill="${win ? C.fused : C.hbm}">${win ? "✓" : "×"}</tspan> ${win ? "fused is faster: fuse at this shape" : "fused is slower: keep the original path"}</text></svg>`; };
  $("#dg-cost1").innerHTML = costSvg(70, 30, 0.10, true);
  $("#dg-cost2").innerHTML = costSvg(92, 8, 0.20, false);

  // ---- production table
  const H = D.prod.headline; let fam = "";
  $("#prod-body").innerHTML = D.prod.kernels.map((k) => `${k.family !== fam ? `<tr class="fam"><td colspan="5">${esc(fam = k.family)}</td></tr>` : ""}
    <tr><td><b>${esc(k.name)}</b><div class="muted small">${esc(k.technique)}</div></td><td class="nw">${esc(k.backend)}</td>
    <td><span class="pill ${k.deployed ? "yes" : ""}">${k.deployed ? "in production" : "kernel level only"}</span></td>
    <td class="r">${esc(k.fwd)}${k.removedShare ? "" : "×"}</td><td class="r">${k.bwd === "–" ? "–" : esc(k.bwd) + (k.removedShare ? "" : "×")}</td></tr>`).join("");
  $("#prod-gain").textContent = ">" + D.prodGainPct + "%";

  // ---- decode chart: deployed geometric mean per model (HTML rows, so text stays readable at any width)
  const dec = D.decode.filter((r) => r.gpu === "A800"), top = Math.ceil(Math.max(...dec.map((r) => +r.geomean)) * 10) / 10;
  const dots = (k, n) => Array.from({ length: n }, (_, j) => `<i class="${j < k ? "" : "o"}"></i>`).join("");
  $("#chart-decode").innerHTML = `<div class="rows">` + dec.map((r) => `<div class="brow" title="${esc(r.model)}: ${r.kept} of ${r.total} shapes kept; deployed geometric mean ${r.geomean}× (a shape that is not kept counts as 1.000×)">
      <span class="lab">${esc(r.model)}</span><span class="track"><span class="fill" style="width:${((+r.geomean - 1) / (top - 1) * 100).toFixed(1)}%"></span></span>
      <span class="val">${r.geomean}×</span><span class="dots">${dots(r.kept, r.total)}</span></div>`).join("") +
    `<div class="brow axis"><span class="lab"></span><span class="track"><span>1.0×</span><span>${top.toFixed(1)}×</span></span><span class="val"></span><span class="dots">kept shapes</span></div></div>`;

  // ---- attention census: one dot per configuration, grouped by batch size
  const sets = [["H200", "2.5.1"], ["A800", "2.5.1"]].map(([g, t]) => ({ g, t, rows: D.census.filter((r) => r.gpu === g && r.torch === t) })).filter((s) => s.rows.length);
  const drawCensus = (i) => { const s = sets[i], kept = s.rows.reduce((a, r) => a + r.kept, 0), all = s.rows.reduce((a, r) => a + r.valid, 0);
    $("#chart-census").innerHTML = `<div class="crow head"><span class="lab"></span><span class="grp">B = 1</span><span class="grp">B = 8</span><span class="grp">B = 32</span><span class="val">kept</span></div>` +
      s.rows.map((r) => `<div class="crow" title="${esc(r.model)} on ${s.g}: ${r.kept} of ${r.valid} configurations kept${r.range !== "–" ? ", " + r.range + "× over the best unfused implementation" : ""}">
        <span class="lab">${esc(r.model)}</span>${[r.b1, r.b8, r.b32].map(([k, n]) => `<span class="grp dots">${dots(k, n)}</span>`).join("")}<span class="val">${r.kept}/${r.valid}</span></div>`).join("");
    const mac = s.g === "H200" ? ["HcenKept", "HcenConfigs", "HcenRange", "HcenModels"] : ["AcenKept", "AcenConfigs", "AcenRange", "AcenModels"];
    $("#census-big").innerHTML = `<b>${D.m[mac[0]]} of ${D.m[mac[1]]} kept</b><span>${D.m[mac[2]]}× faster than the best unfused implementation · ${D.m[mac[3]]} models on ${s.g}, PyTorch ${s.t}</span>`;
    if (kept !== +D.m[mac[0]] || all !== +D.m[mac[1]]) $("#census-big").insertAdjacentHTML("beforeend", `<span>(per-model rows add up to ${kept} of ${all})</span>`);
    $$("#seg-census button").forEach((b, j) => b.classList.toggle("on", j === i)); };
  $("#seg-census").innerHTML = sets.map((s, i) => `<button type="button" data-i="${i}">${s.g}</button>`).join("");
  $("#seg-census").addEventListener("click", (e) => { if (e.target.dataset.i) drawCensus(+e.target.dataset.i); });
  if (sets.length) drawCensus(0);

  // ---- training cells of the open-weight table
  const tr = D.train.filter((r) => r.gpu === "A800"), models = [...new Set(tr.map((r) => r.model))];
  const cell = (m, o) => { const r = tr.find((x) => x.model === m && x.optimizer === o); return r ? `<td class="nw">${r.shape}</td><td><span class="pill ${r.kept ? "yes" : ""}">${r.kept ? "kept" : "original path"}</span></td><td class="r">${r.speedup}×</td>` : `<td colspan="3" class="muted">not measured</td>`; };
  $("#train-body").innerHTML = models.map((m) => `<tr><td><b>${esc(m)}</b></td>${cell(m, "AdamW")}${cell(m, "Muon")}</tr>`).join("");

  // ---- where the gates say no
  $("#neg-body").innerHTML = D.s.rejected.sites.map((s) => `<tr><td class="nw muted">${esc(s.type)}</td><td><b>${esc(s.site)}</b></td>
    <td class="nw">${s.cells.map(([g, k, n]) => `${g} <b>${k}/${n}</b>`).join(" &nbsp; ")}</td><td>${esc(s.why)}</td></tr>`).join("");
  $("#ctl-body").innerHTML = D.s.controller.rows.map((r) => `<tr><td><b>${esc(r.mode)}</b><div class="muted small">${esc(r.what)}</div></td><td class="r">${r.proposals}</td><td class="r">${r.trials}</td>
    <td class="r">${r.candidate}×</td><td><span class="pill ${r.gate === "keep" ? "yes" : "no"}">${r.gate}</span></td><td class="r">${r.deployed}×</td></tr>`).join("");

  // ---- table of contents highlight
  const links = $$(".toc a"), secs = links.map((a) => $(a.getAttribute("href")));
  const spy = () => { let k = 0; secs.forEach((s, i) => { if (s && s.getBoundingClientRect().top < 140) k = i; }); links.forEach((a, i) => a.classList.toggle("on", i === k)); };
  window.addEventListener("scroll", spy, { passive: true }); spy();
})();
