"use strict";

const CFG = window.VIEWER_CFG || {};
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtNum = (v) => (Number.isFinite(v) ? (Number.isInteger(v) ? String(v) : String(+v.toPrecision(6))) : "n/a");

const state = {
  atlas: null,
  features: [],
  clauses: [],
  engine: null,
  pred: null, // { cols, rows, featCols, actualIdx }
  comp: null, // computed over rows
  selectedClass: 0,
  selectedRow: -1,
  activeTab: "repr",
};

function parseCSV(text) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n").filter((l) => l.length);
  const parseLine = (line) => {
    const out = [];
    let cur = "", q = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (q) {
        if (ch === '"') {
          if (line[i + 1] === '"') {
            cur += '"';
            i++;
          } else q = false;
        } else cur += ch;
      } else {
        if (ch === '"') q = true;
        else if (ch === ",") {
          out.push(cur);
          cur = "";
        } else cur += ch;
      }
    }
    out.push(cur);
    return out;
  };
  const cols = parseLine(lines[0]);
  const rows = lines.slice(1).map(parseLine);
  return { cols, rows };
}

function readFile(file, cb) {
  const r = new FileReader();
  r.onload = () => cb(null, r.result);
  r.onerror = () => cb(new Error("Failed to read file"));
  r.readAsText(file);
}

function setStatus(msg, muted = true) {
  const box = $("#status");
  box.classList.toggle("muted", muted);
  box.textContent = msg;
}

function buildEngine(atlas) {
  const classes = (atlas.model?.classes || []).map(String);
  const clsIdx = {};
  classes.forEach((c, i) => (clsIdx[c] = i));
  const clauses = (atlas.clauses || []).map((c) => ({
    id: c.id,
    cls: String(c.class),
    clsIdx: clsIdx[String(c.class)],
    polarity: c.polarity === "negative" ? "negative" : "positive",
    sign: c.polarity === "negative" ? -1 : 1,
    clamp: Number(c.clamp || 1),
    lits: (c.literals || []).map((l) => ({
      feature: l.feature,
      geq: l.operator === "≥",
      thr: Number(l.threshold),
    })),
  }));
  return { classes, clsIdx, clauses };
}

function evalSample(fv, full = true) {
  const eng = state.engine;
  const scores = new Array(eng.classes.length).fill(0);
  const clauseOut = full ? {} : null;
  let nActive = 0;
  for (const c of eng.clauses) {
    let viol = 0;
    for (const l of c.lits) {
      const x = fv[l.feature];
      const sat = l.geq ? x >= l.thr : x < l.thr;
      if (!sat) viol++;
    }
    const out = Math.max(c.clamp - viol, 0);
    if (full) clauseOut[c.id] = { out, viol };
    if (out > 0) {
      nActive++;
      scores[c.clsIdx] += c.sign * out;
    }
  }
  let best = 0;
  for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
  return { predIdx: best, predLabel: eng.classes[best], scores, clauseOut, nActive };
}

function thermoConds(clause) {
  const byFeat = {};
  for (const l of clause.lits || []) {
    const g = (byFeat[l.feature] ||= { on: [], off: [] });
    if (l.geq) g.on.push(l.thr);
    else g.off.push(l.thr);
  }
  const out = [];
  Object.keys(byFeat).sort().forEach((f) => {
    const g = byFeat[f];
    const onT = g.on.length ? Math.max(...g.on) : null;
    const offT = g.off.length ? Math.min(...g.off) : null;
    if (onT !== null) out.push({ feature: f, thr: onT, state: "on" });
    if (offT !== null) out.push({ feature: f, thr: offT, state: "off" });
  });
  return out;
}

function median(vals) {
  if (!vals.length) return null;
  const a = vals.slice().sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

function rangeLabel(feat, onVals, offVals) {
  const lo = median(onVals);
  const hi = median(offVals);
  if (Number.isFinite(lo) && Number.isFinite(hi)) {
    if (lo < hi) return { kind: "band", text: `${feat} in [${fmtNum(lo)}, ${fmtNum(hi)})` };
    return { kind: "band", text: `${feat} around ${fmtNum((lo + hi) / 2)} (mixed)` };
  }
  if (Number.isFinite(lo)) return { kind: "hi", text: `${feat} ≥ ${fmtNum(lo)}` };
  if (Number.isFinite(hi)) return { kind: "lo", text: `${feat} < ${fmtNum(hi)}` };
  return { kind: "neutral", text: feat };
}

function clauseWeight(clauseIndex, clause) {
  if (state.comp?.clauseVotes && state.comp.clauseFire[clauseIndex] > 0) {
    return state.comp.clauseVotes[clauseIndex] / state.comp.clauseFire[clauseIndex];
  }
  return Math.max(1, clause.clamp || 1);
}

function buildClassTree(clsIdx) {
  const eng = state.engine;
  const support = {};
  const oppose = {};
  const bucket = (map, feat) => {
    if (!map[feat]) map[feat] = { feat, weight: 0, n: 0, onVals: [], offVals: [], co: {} };
    return map[feat];
  };

  eng.clauses.forEach((c, ci) => {
    if (c.clsIdx !== clsIdx) return;
    const conds = thermoConds(c);
    if (!conds.length) return;
    const lead = conds.find((x) => x.state === "on") || conds[0];
    const map = c.polarity === "positive" ? support : oppose;
    const w = clauseWeight(ci, c);
    const b = bucket(map, lead.feature);
    b.weight += w;
    b.n++;
    for (const cd of conds) {
      if (cd.feature === lead.feature) {
        if (cd.state === "on") b.onVals.push(cd.thr);
        else b.offVals.push(cd.thr);
        continue;
      }
      const co = (b.co[cd.feature] ||= { feat: cd.feature, onVals: [], offVals: [], weight: 0 });
      if (cd.state === "on") co.onVals.push(cd.thr);
      else co.offVals.push(cd.thr);
      co.weight += w;
    }
  });

  const finalize = (map) =>
    Object.values(map)
      .map((x) => {
        const lead = rangeLabel(x.feat, x.onVals, x.offVals);
        const co = Object.values(x.co)
          .map((c) => ({ ...rangeLabel(c.feat, c.onVals, c.offVals), weight: c.weight }))
          .sort((a, b) => b.weight - a.weight)
          .slice(0, CFG.topCoFeatures || 4);
        return { label: lead.text, kind: lead.kind, n: x.n, weight: x.weight, co };
      })
      .sort((a, b) => b.weight - a.weight);

  return {
    cls: eng.classes[clsIdx],
    support: finalize(support),
    oppose: finalize(oppose),
    nPos: eng.clauses.filter((c) => c.clsIdx === clsIdx && c.polarity === "positive").length,
    nNeg: eng.clauses.filter((c) => c.clsIdx === clsIdx && c.polarity === "negative").length,
  };
}

function renderTreeSvg(tree) {
  const support = tree.support.slice(0, CFG.topConcepts || 5);
  const oppose = tree.oppose.slice(0, CFG.topConcepts || 5);
  const colW = 280;
  const laneGap = 90;
  const W = colW * 2 + laneGap + 120;
  const cx = W / 2;
  const startY = 128;
  const nodeH = 32;
  const step = 54;
  const leftX = 60;
  const rightX = W - 60 - colW;
  const H = Math.max(260, startY + Math.max(support.length, oppose.length, 1) * step + 30);
  const maxW = Math.max(1, ...support.map((x) => x.weight), ...oppose.map((x) => x.weight));
  const edge = (x1, y1, x2, y2, col) =>
    `<path d="M${x1} ${y1} C${x1} ${Math.round((y1 + y2) / 2)},${x2} ${Math.round((y1 + y2) / 2)},${x2} ${y2}" fill="none" stroke="${col}" stroke-width="1.5" opacity="0.8"/>`;
  const escTxt = (s, n = 34) => {
    const t = String(s);
    return t.length > n ? t.slice(0, n - 1) + "…" : t;
  };
  let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interpretability tree for ${esc(tree.cls)}">`;
  // root and split
  s += `<rect x="${cx - 140}" y="8" width="280" height="40" rx="8" fill="#1e2a45" stroke="#7aa2f7"/><text x="${cx}" y="26" text-anchor="middle" fill="#e6e8ee" font-size="14" font-weight="700">${escTxt(tree.cls, 28)}</text><text x="${cx}" y="40" text-anchor="middle" fill="#9aa3b2" font-size="10">learned representation</text>`;
  s += edge(cx, 48, cx, 68, "#5a7a9a");
  s += `<rect x="${cx - 110}" y="68" width="220" height="28" rx="7" fill="#1f2431" stroke="#5a7a9a"/><text x="${cx}" y="86" text-anchor="middle" fill="#c7cfdd" font-size="11">score = support − opposition</text>`;
  s += edge(cx, 96, leftX + colW / 2, startY - 16, "#9ece6a");
  s += edge(cx, 96, rightX + colW / 2, startY - 16, "#f7768e");
  s += `<text x="${leftX + colW / 2}" y="${startY - 22}" text-anchor="middle" fill="#9ece6a" font-size="11">support branch</text>`;
  s += `<text x="${rightX + colW / 2}" y="${startY - 22}" text-anchor="middle" fill="#f7768e" font-size="11">opposition branch</text>`;

  const drawLane = (arr, x, tone) => {
    if (!arr.length) {
      s += `<rect x="${x}" y="${startY}" width="${colW}" height="${nodeH}" rx="7" fill="#181b24" stroke="#2b3142"/><text x="${x + 10}" y="${startY + 20}" fill="#9aa3b2" font-size="11">no strong concepts</text>`;
      return;
    }
    arr.forEach((c, i) => {
      const y = startY + i * step;
      const col = tone === "pos" ? "#9ece6a" : "#f7768e";
      const fill = tone === "pos" ? "#16241a" : "#241418";
      const sourceX = tone === "pos" ? leftX + colW / 2 : rightX + colW / 2;
      s += edge(sourceX, startY - 16, x + colW / 2, y, col);
      s += `<rect x="${x}" y="${y}" width="${colW}" height="${nodeH}" rx="7" fill="${fill}" stroke="${col}"/>`;
      s += `<text x="${x + 8}" y="${y + 14}" fill="#e6e8ee" font-size="11" font-weight="600">${escTxt(c.label, 40)}</text>`;
      s += `<text x="${x + 8}" y="${y + 26}" fill="#9aa3b2" font-size="10">${c.n} patterns · strength ${fmtNum(Math.round(c.weight))}</text>`;
      const bw = Math.max(6, ((colW - 16) * c.weight) / maxW);
      s += `<rect x="${x + 8}" y="${y + nodeH - 4}" width="${colW - 16}" height="3" rx="2" fill="#2b3142"/><rect x="${x + 8}" y="${y + nodeH - 4}" width="${bw}" height="3" rx="2" fill="${col}"/>`;
    });
  };
  drawLane(support, leftX, "pos");
  drawLane(oppose, rightX, "neg");
  s += `<text x="12" y="${H - 8}" fill="#9aa3b2" font-size="10">Top node: class · left: evidence for class · right: evidence against class</text>`;
  s += `</svg>`;
  return s;
}

function renderTabs() {
  $$(".tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.tab === state.activeTab);
  });
  ["repr", "behavior", "why"].forEach((t) => $("#tab-" + t).classList.toggle("hidden", t !== state.activeTab));
}

function renderRepresentation() {
  const clsIdx = state.selectedClass;
  const tree = buildClassTree(clsIdx);
  const top = (arr) => arr.slice(0, CFG.topConcepts || 5);
  const concept = (x) =>
    `<div class="concept">
      <div class="title">${esc(x.label)}</div>
      <div class="muted">${x.n} merged patterns · strength ~${Math.round(x.weight)}</div>
      <div class="chips">${x.co.length ? x.co.map((c) => `<span class="chip">${esc(c.text)}</span>`).join("") : '<span class="chip">no strong co-feature</span>'}</div>
    </div>`;
  $("#reprTree").innerHTML = `
    <div class="repr-root">
      <div class="muted">Learned representation</div>
      <div class="name">${esc(tree.cls)}</div>
      <div class="muted">Model combines supporting evidence and subtracts opposing evidence.</div>
    </div>
    <div class="tree-wrap">${renderTreeSvg(tree)}</div>
    <div class="repr-grid">
      <div class="lane pos">
        <h3>What the model looks for</h3>
        ${top(tree.support).map(concept).join("") || '<div class="muted">No strong supporting concepts.</div>'}
      </div>
      <div class="lane neg">
        <h3>What pushes away from this class</h3>
        ${top(tree.oppose).map(concept).join("") || '<div class="muted">No strong opposing concepts.</div>'}
      </div>
    </div>
  `;
}

function renderBehavior() {
  const total = state.engine.clauses.length;
  let avgActive = "—", activeRules = "—", neverRules = "—";
  let featureRows = [];
  if (state.comp) {
    avgActive = state.comp.nSamples ? (state.comp.sumActive / state.comp.nSamples).toFixed(1) : "0";
    const fire = state.comp.clauseFire;
    const active = fire.filter((x) => x > 0).length;
    activeRules = `${active} / ${total}`;
    neverRules = String(fire.filter((x) => x === 0).length);
    const max = Math.max(1, ...Object.values(state.comp.featUse));
    featureRows = Object.entries(state.comp.featUse)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([name, c]) => ({ name, pct: (100 * c) / max, val: `${Math.round((100 * c) / state.comp.nSamples)}%` }));
  } else {
    const use = {};
    state.engine.clauses.forEach((c) => {
      const seen = new Set(c.lits.map((l) => l.feature));
      seen.forEach((f) => (use[f] = (use[f] || 0) + 1));
    });
    const max = Math.max(1, ...Object.values(use));
    featureRows = Object.entries(use)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15)
      .map(([name, c]) => ({ name, pct: (100 * c) / max, val: `${c} rules` }));
  }

  const bars = featureRows
    .map(
      (r) => `<div class="bar"><div>${esc(r.name)}</div><div class="track"><div class="fill" style="width:${r.pct}%"></div></div><div class="muted">${esc(r.val)}</div></div>`
    )
    .join("");

  $("#behaviorContent").innerHTML = `
    <div class="kpis">
      <div class="kpi"><div class="k">Total patterns</div><div class="v">${total}</div></div>
      <div class="kpi"><div class="k">Active patterns</div><div class="v">${activeRules}</div></div>
      <div class="kpi"><div class="k">Average active per row</div><div class="v">${avgActive}</div></div>
      <div class="kpi"><div class="k">Never active</div><div class="v">${neverRules}</div></div>
    </div>
    <div class="panel2">
      <h3 style="margin:0 0 6px">Most used features</h3>
      <div class="muted" style="margin-bottom:6px">${state.comp ? "Based on loaded sample behavior." : "Based on learned pattern structure (load CSV for behavior view)."}</div>
      <div class="bars">${bars || '<div class="muted">No feature stats available.</div>'}</div>
    </div>
  `;
}

function buildPredData(csv) {
  const featSet = new Set(state.features.map((f) => f.name));
  const featCols = csv.cols.map((c, i) => ({ name: c, col: i })).filter((x) => featSet.has(x.name));
  const actualIdx = csv.cols.findIndex((c) => /^actual$/i.test(c));
  state.pred = { ...csv, featCols, actualIdx };
}

function rowFV(row) {
  const out = {};
  state.pred.featCols.forEach((fc) => {
    const s = row[fc.col];
    out[fc.name] = s === "" || s === undefined ? NaN : +s;
  });
  return out;
}

function recomputePred() {
  if (!state.pred) return;
  const n = state.pred.rows.length;
  const pred = new Array(n);
  const correct = new Array(n).fill(null);
  const clauseFire = new Array(state.engine.clauses.length).fill(0);
  const clauseVotes = new Array(state.engine.clauses.length).fill(0);
  const featUse = {};
  let sumActive = 0;

  for (let i = 0; i < n; i++) {
    const fv = rowFV(state.pred.rows[i]);
    const res = evalSample(fv, true);
    pred[i] = res.predIdx;
    sumActive += res.nActive;
    const used = new Set();
    state.engine.clauses.forEach((c, ci) => {
      const o = res.clauseOut[c.id];
      if (o.out > 0) {
        clauseFire[ci]++;
        clauseVotes[ci] += o.out;
        c.lits.forEach((l) => used.add(l.feature));
      }
    });
    used.forEach((f) => (featUse[f] = (featUse[f] || 0) + 1));

    if (state.pred.actualIdx >= 0) {
      const truth = String(state.pred.rows[i][state.pred.actualIdx]);
      correct[i] = truth === state.engine.classes[res.predIdx];
    }
  }
  state.comp = { pred, correct, clauseFire, clauseVotes, featUse, nSamples: n, sumActive };
}

function renderSampleTable() {
  if (!state.pred || !state.comp) return;
  const n = Math.min(CFG.topRows || 200, state.pred.rows.length);
  const actual = state.pred.actualIdx >= 0;
  $("#sampleTable thead").innerHTML = `<tr><th>#</th>${actual ? "<th>actual</th>" : ""}<th>predicted</th></tr>`;
  $("#sampleTable tbody").innerHTML = state.pred.rows
    .slice(0, n)
    .map((r, i) => {
      const cls = i === state.selectedRow ? "sel" : "";
      const pred = state.engine.classes[state.comp.pred[i]];
      const act = actual ? `<td>${esc(String(r[state.pred.actualIdx]))}</td>` : "";
      return `<tr class="${cls}" data-i="${i}"><td>${i}</td>${act}<td><b>${esc(pred)}</b></td></tr>`;
    })
    .join("");

  $$("#sampleTable tbody tr").forEach((tr) => {
    tr.onclick = () => {
      state.selectedRow = +tr.dataset.i;
      renderSampleTable();
      renderSampleExplain(state.selectedRow);
    };
  });
}

function themeSummaryForRow(fv, res, clsIdx) {
  const byFeat = {};
  state.engine.clauses.forEach((c) => {
    if (c.clsIdx !== clsIdx || c.polarity !== "positive") return;
    const out = res.clauseOut[c.id]?.out || 0;
    if (!out) return;
    const conds = thermoConds(c);
    if (!conds.length) return;
    const lead = conds.find((x) => x.state === "on") || conds[0];
    const key = lead.feature;
    if (!byFeat[key]) byFeat[key] = { feat: key, onVals: [], offVals: [], pts: 0, n: 0 };
    byFeat[key].pts += out;
    byFeat[key].n++;
    conds.forEach((cd) => {
      if (cd.feature !== key) return;
      if (cd.state === "on") byFeat[key].onVals.push(cd.thr);
      else byFeat[key].offVals.push(cd.thr);
    });
  });
  return Object.values(byFeat)
    .map((x) => ({ ...x, label: rangeLabel(x.feat, x.onVals, x.offVals).text }))
    .sort((a, b) => b.pts - a.pts)
    .slice(0, 5);
}

function renderSampleExplain(i) {
  const row = state.pred.rows[i];
  const fv = rowFV(row);
  const res = evalSample(fv, true);
  const winIdx = res.predIdx;
  const winner = res.predLabel;
  const ranked = res.scores.map((v, k) => ({ k, v })).sort((a, b) => b.v - a.v);
  const runner = ranked.find((x) => x.k !== winIdx) || ranked[0];
  const themes = themeSummaryForRow(fv, res, winIdx);
  const keyVals = themes.map((t) => `<span class="chip">${esc(t.feat)} = ${esc(fmtNum(fv[t.feat]))}</span>`).join("");
  const scoreRows = ranked
    .slice(0, CFG.topScoreboard || 8)
    .map((r) => `<div class="bar"><div>${esc(state.engine.classes[r.k])}</div><div class="track"><div class="fill" style="width:${(100 * Math.abs(r.v)) / Math.max(1, Math.max(...ranked.map((x) => Math.abs(x.v))))}%"></div></div><div>${fmtNum(r.v)}</div></div>`)
    .join("");

  const whyLost = runner
    ? `<div class="section"><h3 style="margin:0 0 6px">Why ${esc(state.engine.classes[runner.k])} lost</h3>
         <div class="muted">${esc(winner)} scored ${fmtNum(res.scores[winIdx])}; ${esc(state.engine.classes[runner.k])} scored ${fmtNum(runner.v)}.</div>
       </div>`
    : "";

  $("#sampleExplain").innerHTML = `
    <div class="outcome">
      <div class="muted">Predicted class</div>
      <div class="big">${esc(winner)}</div>
      <div class="muted">Margin over next class: ${fmtNum(res.scores[winIdx] - (runner?.v ?? res.scores[winIdx]))}</div>
    </div>
    <div class="section">
      <h3 style="margin:0 0 6px">Main reason</h3>
      ${themes.length ? themes.map((t) => `<div><b>${esc(t.label)}</b> · +${fmtNum(t.pts)} points (${t.n} patterns)</div>`).join("") : '<div class="muted">No strong supporting concepts on this row.</div>'}
    </div>
    <div class="section">
      <h3 style="margin:0 0 6px">Key input values used</h3>
      <div class="chips">${keyVals || '<span class="muted">No key values</span>'}</div>
    </div>
    <div class="section">
      <h3 style="margin:0 0 6px">Scoreboard</h3>
      ${scoreRows}
    </div>
    ${whyLost}
  `;
}

function renderWhy() {
  if (!state.pred || !state.comp) {
    $("#whyEmpty").classList.remove("hidden");
    $("#whyContent").classList.add("hidden");
    return;
  }
  $("#whyEmpty").classList.add("hidden");
  $("#whyContent").classList.remove("hidden");
  if (state.selectedRow < 0) state.selectedRow = 0;
  renderSampleTable();
  renderSampleExplain(state.selectedRow);
}

function renderAll() {
  if (!state.engine) return;
  $("#app").classList.remove("hidden");
  const clsSel = $("#reprClass");
  clsSel.innerHTML = state.engine.classes.map((c, i) => `<option value="${i}">${esc(c)}</option>`).join("");
  clsSel.value = String(state.selectedClass);

  const gSel = $("#globalFocusClass");
  gSel.innerHTML = state.engine.classes.map((c, i) => `<option value="${i}">${esc(c)}</option>`).join("");
  gSel.value = String(state.selectedClass);

  renderRepresentation();
  renderBehavior();
  renderWhy();
}

function wireUI() {
  $("#atlasFile").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f) return;
    readFile(f, (err, text) => {
      if (err) return setStatus("Failed to load atlas JSON.", false);
      try {
        state.atlas = JSON.parse(text);
        state.features = state.atlas.features || [];
        state.engine = buildEngine(state.atlas);
        state.clauses = state.engine.clauses;
        state.selectedClass = 0;
        setStatus(`Loaded model: ${state.engine.classes.length} classes, ${state.engine.clauses.length} patterns.`);
        renderAll();
      } catch (ex) {
        setStatus("Invalid atlas JSON format.", false);
      }
    });
  });

  $("#csvFile").addEventListener("change", (e) => {
    const f = e.target.files[0];
    if (!f || !state.engine) return;
    readFile(f, (err, text) => {
      if (err) return setStatus("Failed to load CSV.", false);
      try {
        const csv = parseCSV(text);
        buildPredData(csv);
        recomputePred();
        renderAll();
        setStatus(`Loaded ${state.pred.rows.length} samples.`);
      } catch (ex) {
        setStatus("Invalid CSV.", false);
      }
    });
  });

  $("#reprClass").addEventListener("change", (e) => {
    state.selectedClass = +e.target.value;
    $("#globalFocusClass").value = String(state.selectedClass);
    renderRepresentation();
  });
  $("#globalFocusClass").addEventListener("change", (e) => {
    state.selectedClass = +e.target.value;
    $("#reprClass").value = String(state.selectedClass);
    renderRepresentation();
  });

  $$(".tabs button").forEach((b) => {
    b.addEventListener("click", () => {
      state.activeTab = b.dataset.tab;
      renderTabs();
    });
  });

  // Auto-load via query params (GitHub Pages friendly)
  const p = new URLSearchParams(location.search);
  const src = p.get("src");
  const csv = p.get("csv");
  if (src) {
    fetch(src)
      .then((r) => r.json())
      .then((atlas) => {
        state.atlas = atlas;
        state.features = atlas.features || [];
        state.engine = buildEngine(atlas);
        state.clauses = state.engine.clauses;
        setStatus(`Loaded model: ${state.engine.classes.length} classes, ${state.engine.clauses.length} patterns.`);
        renderAll();
        if (csv) {
          fetch(csv)
            .then((r) => r.text())
            .then((txt) => {
              buildPredData(parseCSV(txt));
              recomputePred();
              renderAll();
              setStatus(`Loaded model + ${state.pred.rows.length} samples.`);
            })
            .catch(() => setStatus("Model loaded. CSV could not be fetched.", false));
        }
      })
      .catch(() => setStatus("Could not fetch atlas from URL.", false));
  }
}

wireUI();
renderTabs();
