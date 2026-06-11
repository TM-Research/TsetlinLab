#!/usr/bin/env python3
"""
tree_diagram.py  —  Full-fidelity decision-tree visualiser for a trained
Tsetlin Machine (Fuzzy-Pattern TM) atlas.

Unlike the "Learned Representation" summary in viewer.html (which folds all
clauses for a class into a handful of merged concepts), this produces the
COMPLETE tree: every class, every clause, every condition, with nothing
pruned. The model is reconstructed from the atlas exactly as trained, and
real per-clause firing counts + per-class sample distributions are read from
the predictions CSV so each node carries a true sample count.

Output is a single self-contained static HTML file (no server, no build, no
JS deps) that works straight off the filesystem.

    python3 tree_diagram.py data/wustl_atlas.json data/wustl_predictions.csv \
            -o docs/wustl_full_tree.html

The math it visualises (Fuzzy-Pattern TM inference):
    bit            = value >= threshold
    literal "≥ t"  satisfied iff value >= t     (else a "miss")
    clause output  = max(clamp − misses, 0)     (a SOFT and: up to clamp−1 misses allowed)
    class score    = Σ positive-clause outputs − Σ negative-clause outputs
    prediction     = argmax over classes
"""

import argparse
import csv
import html
import json
import os
import sys
from collections import Counter, defaultdict


# --------------------------------------------------------------------------- #
#  Load + compute real firing statistics from the predictions CSV             #
# --------------------------------------------------------------------------- #
def load_firing_stats(csv_path, class_names):
    """Return (n_samples, clause_fire[id]->count, actual_dist, predicted_dist,
    correct_count) computed from the predictions CSV. The CSV stores `actual`
    and `predicted` as class *indices* and `ActivatedClauses` as a
    ';'-separated list of clause ids that fired on that row."""
    clause_fire = Counter()
    actual_dist = Counter()
    predicted_dist = Counter()
    n = 0
    correct = 0
    if not csv_path or not os.path.exists(csv_path):
        return 0, clause_fire, actual_dist, predicted_dist, 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n += 1
            a = row.get("actual", "")
            p = row.get("predicted", "")
            try:
                ai = int(float(a)); actual_dist[ai] += 1
            except (ValueError, TypeError):
                ai = None
            try:
                pi = int(float(p)); predicted_dist[pi] += 1
            except (ValueError, TypeError):
                pi = None
            if ai is not None and pi is not None and ai == pi:
                correct += 1
            ac = row.get("ActivatedClauses", "") or ""
            for tok in ac.split(";"):
                tok = tok.strip()
                if tok:
                    try:
                        clause_fire[int(tok)] += 1
                    except ValueError:
                        pass
    return n, clause_fire, actual_dist, predicted_dist, correct


# --------------------------------------------------------------------------- #
#  Plain-language helpers                                                      #
# --------------------------------------------------------------------------- #
def fmt_num(x):
    try:
        xf = float(x)
    except (ValueError, TypeError):
        return str(x)
    if xf == int(xf):
        return str(int(xf))
    return f"{xf:.4g}"


def band_plain(band):
    """One short, jargon-free line for a feature band. Prefer the atlas's own
    `plain`, but re-derive a crisp version so the tree reads consistently."""
    feat = band["feature"]
    lower = band.get("lower")
    upper = band.get("upper")
    impossible = band.get("impossible")
    if impossible:
        return (f"{feat} can't be both ≥ {fmt_num(lower)} and < {fmt_num(upper)} "
                f"— impossible, so this check can never be met")
    if lower is not None and upper is not None:
        return f"{feat} is between {fmt_num(lower)} and {fmt_num(upper)}"
    if lower is not None:
        return f"{feat} is at least {fmt_num(lower)} (high)"
    if upper is not None:
        return f"{feat} is below {fmt_num(upper)} (low)"
    # fall back to whatever the atlas provided
    return band.get("plain") or band.get("text") or feat


# --------------------------------------------------------------------------- #
#  Build the tree data structure (no pruning whatsoever)                       #
# --------------------------------------------------------------------------- #
def build_tree(atlas, stats):
    n_samples, clause_fire, actual_dist, predicted_dist, correct = stats
    classes = [str(c) for c in atlas["model"]["classes"]]
    cls_index = {c: i for i, c in enumerate(classes)}
    meta = atlas.get("metadata", {})

    # group clauses by (class, polarity), preserving id order
    by_cp = defaultdict(list)
    for c in atlas["clauses"]:
        by_cp[(str(c["class"]), c["polarity"])].append(c)

    class_nodes = []
    for ci, cls in enumerate(classes):
        pol_nodes = []
        for pol in ("positive", "negative"):
            clause_objs = by_cp.get((cls, pol), [])
            clause_nodes = []
            for c in clause_objs:
                clamp = int(c["clamp"])
                n_lit = int(c.get("nLiterals", len(c.get("literals", []))))
                needed = max(1, n_lit - (clamp - 1))  # min checks to fire
                bands = c.get("featureBands", [])
                cond_nodes = []
                n_impossible = 0
                for b in bands:
                    imp = bool(b.get("impossible"))
                    if imp:
                        n_impossible += 1
                    cond_nodes.append({
                        "text": band_plain(b),
                        "impossible": imp,
                        "n_bits": b.get("nBits", 1),
                    })
                fired = clause_fire.get(c["id"], 0)
                clause_nodes.append({
                    "id": c["id"],
                    "polarity": pol,
                    "clamp": clamp,
                    "n_lit": n_lit,
                    "n_feat": int(c.get("nFeatures", len(bands))),
                    "needed": needed,
                    "n_impossible": n_impossible,
                    "fired": fired,
                    "fired_pct": (100.0 * fired / n_samples) if n_samples else None,
                    "conds": cond_nodes,
                })
            pol_nodes.append({
                "polarity": pol,
                "n_clauses": len(clause_nodes),
                "clauses": clause_nodes,
            })
        class_nodes.append({
            "name": cls,
            "index": ci,
            "n_actual": actual_dist.get(ci, 0),
            "n_predicted": predicted_dist.get(ci, 0),
            "polarities": pol_nodes,
        })

    return {
        "classes": classes,
        "meta": meta,
        "n_samples": n_samples,
        "correct": correct,
        "accuracy": (100.0 * correct / n_samples) if n_samples else None,
        "class_nodes": class_nodes,
        "model_type": atlas["model"].get("type"),
        "task": atlas["model"].get("task"),
    }


# --------------------------------------------------------------------------- #
#  Render to a self-contained HTML page                                        #
# --------------------------------------------------------------------------- #
def esc(s):
    return html.escape(str(s))


def render_html(tree, title):
    payload = json.dumps(tree, ensure_ascii=False)
    return HTML_TEMPLATE.replace("/*__TITLE__*/", esc(title)).replace(
        "\"__DATA__\"", payload
    )


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>/*__TITLE__*/</title>
<style>
  :root{
    --bg:#0b0d12; --panel:#11141c; --panel2:#161a24; --line:#262c3a;
    --ink:#e7eaf0; --mut:#98a2b3; --acc:#7aa2f7;
    --pos:#9ece6a; --posbg:#15241a; --posln:#2f5f33;
    --neg:#f7768e; --negbg:#241419; --negln:#6e2b37;
    --warn:#e0af68; --warnbg:#2a2415;
  }
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  a{color:var(--acc)}
  .wrap{max-width:1180px;margin:0 auto;padding:24px 20px 80px}
  header h1{font-size:22px;margin:0 0 4px}
  header .sub{color:var(--mut);margin:0 0 16px;font-size:13px}

  /* model summary strip */
  .summary{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 16px}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;
    padding:9px 13px;min-width:96px}
  .stat .k{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
  .stat .v{font-size:18px;font-weight:700;margin-top:2px}

  /* legend */
  .legend{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:14px 16px;margin:0 0 16px;font-size:13px}
  .legend h2{font-size:14px;margin:0 0 8px}
  .legend .row{display:flex;flex-wrap:wrap;gap:14px 20px}
  .legend .item{display:flex;align-items:center;gap:7px;color:var(--mut)}
  .swatch{width:14px;height:14px;border-radius:4px;flex:none;border:1px solid var(--line)}
  .legend p{margin:8px 0 0;color:var(--mut)}
  .formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    background:var(--panel2);border:1px solid var(--line);border-radius:8px;
    padding:8px 11px;margin:10px 0 0;color:var(--ink);font-size:12.5px;overflow:auto}
  .formula .gp{color:var(--pos)} .formula .gn{color:var(--neg)} .formula .hl{color:var(--acc)}

  /* toolbar */
  .toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 14px}
  .toolbar input[type=search]{background:var(--panel2);border:1px solid var(--line);
    color:var(--ink);border-radius:8px;padding:7px 11px;min-width:230px;font-size:13px}
  .toolbar button{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:7px 12px;font-size:12.5px;cursor:pointer}
  .toolbar button:hover{border-color:var(--acc)}
  .toolbar label{display:flex;align-items:center;gap:6px;color:var(--mut);font-size:12.5px;cursor:pointer}
  .toolbar .spacer{flex:1}
  .toolbar .count{color:var(--mut);font-size:12px}

  /* tree */
  ul.tree, ul.tree ul{list-style:none;margin:0;padding:0}
  ul.tree ul{margin-left:13px;padding-left:14px;border-left:1.5px solid var(--line)}
  ul.tree li{margin:3px 0;position:relative}
  .node{display:flex;align-items:flex-start;gap:8px;border:1px solid var(--line);
    border-radius:9px;background:var(--panel);padding:8px 11px}
  .node.tog{cursor:pointer}
  .node.tog:hover{border-color:var(--acc)}
  .chev{flex:none;width:13px;color:var(--mut);font-size:11px;
    transition:transform .12s ease;margin-top:2px}
  li.open > .node > .chev{transform:rotate(90deg)}
  li > ul{display:none}
  li.open > ul{display:block}
  .body{flex:1;min-width:0}
  .ttl{font-weight:600}
  .meta{color:var(--mut);font-size:12px;margin-top:2px}
  .tag{display:inline-block;font-size:11px;border-radius:6px;padding:1px 7px;
    margin-left:6px;border:1px solid var(--line);color:var(--mut);vertical-align:middle}

  /* level accents */
  .n-root > .node{border-color:#34406a;background:#141a2c}
  .n-root > .node .ttl{font-size:16px}
  .n-class > .node{border-color:#3a4674;background:#141826}
  .n-class > .node .ttl{font-size:15px}
  .n-pos > .node{border-color:var(--posln);background:var(--posbg)}
  .n-neg > .node{border-color:var(--negln);background:var(--negbg)}
  .n-pos > .node .ttl{color:var(--pos)} .n-neg > .node .ttl{color:var(--neg)}
  .n-clause.pos > .node{border-left:3px solid var(--pos)}
  .n-clause.neg > .node{border-left:3px solid var(--neg)}
  .n-cond > .node{background:var(--panel2);padding:6px 10px;border-style:dashed}
  .n-cond.impossible > .node{border-color:var(--warn);background:var(--warnbg)}
  .n-cond .ttl{font-weight:500;font-size:13px}

  .pill{display:inline-block;font-size:11px;padding:1px 7px;border-radius:999px;
    margin-left:6px;vertical-align:middle}
  .pill.fire{background:#152033;color:#9cc4ff;border:1px solid #2c456e}
  .pill.imp{background:var(--warnbg);color:var(--warn);border:1px solid #6b551f}
  .firebar{height:4px;border-radius:3px;background:#1c2230;margin-top:6px;overflow:hidden;max-width:260px}
  .firebar > span{display:block;height:100%;background:var(--acc)}

  .dist{display:inline-flex;gap:2px;margin-left:8px;vertical-align:middle}
  .dist i{width:5px;border-radius:2px;display:inline-block;align-self:flex-end}

  .hidden{display:none!important}
  mark{background:#3a3416;color:#ffe9a8;border-radius:3px;padding:0 1px}
  footer{margin-top:30px;color:var(--mut);font-size:12px;border-top:1px solid var(--line);padding-top:14px}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>/*__TITLE__*/</h1>
    <p class="sub">The <b>complete</b> decision logic of the trained Tsetlin Machine — every class, every rule,
       every condition. Nothing is pruned or summarised. Click any node to expand it.</p>
  </header>

  <div class="summary" id="summary"></div>

  <div class="legend">
    <h2>How to read this tree</h2>
    <div class="row">
      <div class="item"><span class="swatch" style="background:#141826;border-color:#3a4674"></span> a <b>class</b> the model can predict</div>
      <div class="item"><span class="swatch" style="background:var(--posbg);border-color:var(--posln)"></span> <b>votes FOR</b> the class (add to its score)</div>
      <div class="item"><span class="swatch" style="background:var(--negbg);border-color:var(--negln)"></span> <b>votes AGAINST</b> the class (subtract from its score)</div>
      <div class="item"><span class="swatch" style="background:var(--panel2);border-style:dashed"></span> a single <b>condition</b> (a check on one reading)</div>
      <div class="item"><span class="swatch" style="background:var(--warnbg);border-color:var(--warn)"></span> an <b>impossible</b> condition (can never be true)</div>
    </div>
    <p>A <b>rule</b> (clause) is a checklist of conditions joined by AND. It is a <b>soft</b> AND: the rule still
       counts if a few checks miss. Each rule says how many of its checks must hold to fire, and the
       <span class="pill fire" style="margin:0">fired on N</span> pill shows how many real test samples actually triggered it.</p>
    <div class="formula">
      <span class="hl">score(class)</span> = <span class="gp">Σ (votes FOR that fired)</span> − <span class="gn">Σ (votes AGAINST that fired)</span>
      &nbsp;&nbsp;→&nbsp;&nbsp; the class with the <span class="hl">highest score wins</span>
    </div>
  </div>

  <div class="toolbar">
    <input type="search" id="q" placeholder="filter by feature, class, or rule #…" autocomplete="off">
    <button id="expandClasses">expand classes</button>
    <button id="expandRules">expand all rules</button>
    <button id="expandAll">expand everything</button>
    <button id="collapseAll">collapse all</button>
    <label><input type="checkbox" id="hideImp"> hide impossible conditions</label>
    <span class="spacer" style="flex:1"></span>
    <span class="count" id="count"></span>
  </div>

  <ul class="tree" id="tree"></ul>

  <footer id="footer"></footer>
</div>

<script>
const DATA = "__DATA__";

const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const fmt = n => n==null ? '' : (n>=1000 ? n.toLocaleString() : String(n));

// ---- summary strip ---------------------------------------------------------
function renderSummary(){
  const m = DATA.meta || {};
  const stats = [
    ['Classes', DATA.classes.length],
    ['Rules (clauses)', m.numClauses ?? countClauses()],
    ['Rules / class', m.numClausesPerClass ?? '—'],
    ['Conditions (literals)', m.numLiterals ?? '—'],
    ['Binariser', m.binariser ?? '—'],
    ['Soft-AND tolerance', m.LF != null ? ('miss up to '+(m.LF-1)) : '—'],
  ];
  if(DATA.n_samples){
    stats.push(['Test samples', DATA.n_samples]);
    if(DATA.accuracy!=null) stats.push(['Accuracy', DATA.accuracy.toFixed(1)+'%']);
  }
  document.getElementById('summary').innerHTML = stats.map(([k,v])=>
    `<div class="stat"><div class="k">${esc(k)}</div><div class="v">${esc(fmt(v))}</div></div>`).join('');
}
function countClauses(){
  let n=0; DATA.class_nodes.forEach(c=>c.polarities.forEach(p=>n+=p.clauses.length)); return n;
}

// ---- distribution bars for a class ----------------------------------------
function distBars(node){
  const tot = DATA.n_samples || 0;
  if(!tot) return '';
  const a = node.n_actual, p = node.n_predicted;
  const h = v => Math.max(2, Math.round(14 * v / Math.max(1, tot)));
  return ` <span class="dist" title="actual: ${a} samples · predicted: ${p} samples">`+
         `<i style="height:${h(a)}px;background:var(--acc)"></i>`+
         `<i style="height:${h(p)}px;background:var(--mut)"></i></span>`+
         `<span class="meta" style="margin-left:6px">${fmt(a)} samples truly in this class · model assigns ${fmt(p)}</span>`;
}

// ---- build the DOM tree ----------------------------------------------------
function li(cls, headHtml, childrenHtml, toggle){
  const hasKids = !!childrenHtml;
  const chev = (toggle||hasKids) ? '<span class="chev">▶</span>' : '<span class="chev" style="visibility:hidden">▶</span>';
  return `<li class="${cls}">`+
    `<div class="node${(toggle||hasKids)?' tog':''}">${chev}<div class="body">${headHtml}</div></div>`+
    (hasKids?`<ul>${childrenHtml}</ul>`:'')+
  `</li>`;
}

function condHtml(c){
  const cls = 'n-cond' + (c.impossible?' impossible':'');
  const bits = c.n_bits ? `<span class="tag">${c.n_bits} bit${c.n_bits===1?'':'s'}</span>` : '';
  const imp = c.impossible ? ' <span class="pill imp">impossible</span>' : '';
  const head = `<div class="ttl" data-feat="${esc(c.text)}">${esc(c.text)}${imp}</div>`+
               `<div class="meta">${bits}</div>`;
  return li(cls, head, '', false);
}

function clauseHtml(cl){
  const pol = cl.polarity==='positive' ? 'pos':'neg';
  const sign = cl.polarity==='positive' ? '+':'−';
  const conds = cl.conds.map(condHtml).join('');
  const firePill = (cl.fired_pct!=null)
    ? `<span class="pill fire">fired on ${fmt(cl.fired)} sample${cl.fired===1?'':'s'} (${cl.fired_pct.toFixed(1)}%)</span>` : '';
  const impPill = cl.n_impossible ? `<span class="pill imp">${cl.n_impossible} impossible</span>` : '';
  const bar = (cl.fired_pct!=null)
    ? `<div class="firebar"><span style="width:${Math.min(100,cl.fired_pct).toFixed(1)}%"></span></div>` : '';
  const head =
    `<div class="ttl">Rule #${cl.id} <span class="meta">(${sign} vote)</span>${firePill}${impPill}</div>`+
    `<div class="meta">checklist of <b>${cl.n_feat}</b> feature checks (${cl.n_lit} bit-tests) joined by AND · `+
      `fires when at least <b>${cl.needed}</b> of ${cl.n_lit} hold (soft AND: up to ${cl.clamp-1} may miss) · `+
      `vote strength up to ${cl.clamp}</div>${bar}`;
  return li('n-clause '+pol, head, conds, true);
}

function polHtml(p){
  const pol = p.polarity==='positive' ? 'pos':'neg';
  const word = p.polarity==='positive' ? 'Votes FOR' : 'Votes AGAINST';
  const arrow = p.polarity==='positive' ? '▲' : '▼';
  const verb = p.polarity==='positive' ? 'adds to' : 'subtracts from';
  const clauses = p.clauses.map(clauseHtml).join('');
  const head = `<div class="ttl">${arrow} ${word} this class <span class="tag">${p.n_clauses} rules</span></div>`+
               `<div class="meta">each rule that fires ${verb} the class score</div>`;
  return li('n-'+pol, head, clauses, true);
}

function classHtml(c){
  const pols = c.polarities.map(polHtml).join('');
  const nPos = (c.polarities.find(p=>p.polarity==='positive')||{}).n_clauses||0;
  const nNeg = (c.polarities.find(p=>p.polarity==='negative')||{}).n_clauses||0;
  const head = `<div class="ttl">Class: ${esc(c.name)}${distBars(c)}</div>`+
               `<div class="meta">${nPos} rules vote FOR · ${nNeg} rules vote AGAINST</div>`;
  return li('n-class', head, pols, true);
}

function rootHtml(){
  const classes = DATA.class_nodes.map(classHtml).join('');
  const head = `<div class="ttl">How this Tsetlin Machine decides</div>`+
    `<div class="meta">For a record, every rule below is checked. Rules that fire cast votes; `+
    `each class sums its votes; the highest-scoring class is the prediction.</div>`+
    `<div class="formula" style="margin-top:8px">`+
      `<span class="hl">prediction</span> = argmax over classes of `+
      `[ <span class="gp">Σ votes FOR</span> − <span class="gn">Σ votes AGAINST</span> ]</div>`;
  return li('n-root', head, classes, true);
}

document.getElementById('tree').innerHTML = rootHtml();

// open the root + class level by default so the structure is visible at a glance
document.querySelectorAll('#tree > li').forEach(openLi);
function openLi(li){ li.classList.add('open'); }
function closeLi(li){ li.classList.remove('open'); }

// ---- toggle on click -------------------------------------------------------
document.getElementById('tree').addEventListener('click', e=>{
  const node = e.target.closest('.node.tog');
  if(!node) return;
  const li = node.parentElement;
  li.classList.toggle('open');
});

// ---- toolbar ---------------------------------------------------------------
const tree = document.getElementById('tree');
function setOpen(selector, open){
  tree.querySelectorAll(selector).forEach(li=> open?li.classList.add('open'):li.classList.remove('open'));
}
document.getElementById('expandClasses').onclick = ()=>{ setOpen('li',false);
  tree.querySelectorAll('.n-root, .n-class').forEach(li=>li.classList.add('open')); };
document.getElementById('expandRules').onclick = ()=>{
  tree.querySelectorAll('.n-root, .n-class, .n-pos, .n-neg, .n-clause').forEach(li=>li.classList.add('open')); };
document.getElementById('expandAll').onclick = ()=> setOpen('li', true);
document.getElementById('collapseAll').onclick = ()=>{ setOpen('li',false);
  tree.querySelectorAll('#tree > li').forEach(li=>li.classList.add('open')); };

// hide impossible conditions
document.getElementById('hideImp').onchange = e=>{
  tree.querySelectorAll('.n-cond.impossible').forEach(li=> li.classList.toggle('hidden', e.target.checked));
};

// ---- live filter -----------------------------------------------------------
const q = document.getElementById('q');
const countEl = document.getElementById('count');
function clearMarks(){ tree.querySelectorAll('mark').forEach(m=>{ m.replaceWith(m.textContent); }); }
function runFilter(){
  const term = q.value.trim().toLowerCase();
  clearMarks();
  const allLi = tree.querySelectorAll('li');
  if(!term){
    allLi.forEach(li=>li.classList.remove('hidden'));
    if(document.getElementById('hideImp').checked)
      tree.querySelectorAll('.n-cond.impossible').forEach(li=>li.classList.add('hidden'));
    countEl.textContent='';
    return;
  }
  let matches=0;
  // a li matches if its own head text contains the term
  allLi.forEach(li=>{
    const head = li.querySelector(':scope > .node');
    const txt = head ? head.textContent.toLowerCase() : '';
    li.dataset.self = txt.includes(term) ? '1':'0';
    if(txt.includes(term)) matches++;
  });
  // show a li if it or any descendant matches; open the path to matches
  function visit(li){
    let kidVisible=false;
    li.querySelectorAll(':scope > ul > li').forEach(c=>{ if(visit(c)) kidVisible=true; });
    const self = li.dataset.self==='1';
    const visible = self || kidVisible;
    li.classList.toggle('hidden', !visible);
    if(visible){ li.classList.add('open'); }
    if(self){
      const head = li.querySelector(':scope > .node .ttl');
      if(head) highlight(head, term);
    }
    return visible;
  }
  tree.querySelectorAll('#tree > li').forEach(visit);
  countEl.textContent = matches+' node'+(matches===1?'':'s')+' match';
}
function highlight(el, term){
  const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const nodes=[]; while(walk.nextNode()) nodes.push(walk.currentNode);
  nodes.forEach(n=>{
    const i = n.textContent.toLowerCase().indexOf(term);
    if(i<0) return;
    const span = document.createElement('span');
    span.append(document.createTextNode(n.textContent.slice(0,i)));
    const mk = document.createElement('mark'); mk.textContent = n.textContent.slice(i,i+term.length);
    span.append(mk); span.append(document.createTextNode(n.textContent.slice(i+term.length)));
    n.replaceWith(span);
  });
}
let t=null; q.addEventListener('input', ()=>{ clearTimeout(t); t=setTimeout(runFilter,140); });

document.getElementById('footer').innerHTML =
  'Generated from the trained atlas by <code>code/exporter/tree_diagram.py</code>. '+
  'Sample counts and rule firing rates are measured on the real test set in the predictions CSV. '+
  'This is the full model as trained — no branches removed.';
renderSummary();
</script>
</body>
</html>
"""


INDEX_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full decision trees — trained Tsetlin Machines</title>
<style>
  :root{--bg:#0b0d12;--panel:#11141c;--line:#262c3a;--ink:#e7eaf0;--mut:#98a2b3;--acc:#7aa2f7;--pos:#9ece6a}
  *{box-sizing:border-box}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:860px;margin:0 auto;padding:40px 22px 80px}
  h1{font-size:25px;margin:0 0 6px}
  .sub{color:var(--mut);margin:0 0 28px;font-size:14px}
  a.card{display:block;text-decoration:none;color:inherit;background:var(--panel);
    border:1px solid var(--line);border-radius:14px;padding:18px 20px;margin:0 0 14px;transition:border-color .12s}
  a.card:hover{border-color:var(--acc)}
  .card h2{margin:0 0 4px;font-size:18px;color:var(--ink)}
  .card .desc{color:var(--mut);font-size:13.5px;margin:0 0 10px}
  .chips{display:flex;flex-wrap:wrap;gap:8px}
  .chip{font-size:12px;background:#161a24;border:1px solid var(--line);border-radius:999px;padding:3px 10px;color:var(--mut)}
  .chip b{color:var(--ink);font-weight:600}
  .note{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:0 0 26px;color:var(--mut);font-size:13.5px}
  .note b{color:var(--ink)}
  code{background:#161a24;border:1px solid var(--line);border-radius:5px;padding:1px 5px;font-size:12.5px}
</style></head><body><div class="wrap">
<h1>Full decision trees of the trained Tsetlin Machines</h1>
<p class="sub">Each link opens the <b>complete</b> learned logic of one model — every class, every rule (clause),
and every condition, with nothing pruned or summarised.</p>
<div class="note">These trees show the model <b>exactly as trained</b>. A Tsetlin Machine is not one decision tree —
it is many rules voting at once. Each rule is a soft AND-checklist of conditions; rules that fire cast votes
for or against a class; the highest-scoring class wins. Rule firing rates and class sample counts are measured
on the real test set. Open any model and use <b>expand / collapse</b> and the <b>filter</b> box to navigate.</div>
__CARDS__
<p class="sub" style="margin-top:24px">Generated by <code>code/exporter/tree_diagram.py</code>.</p>
</div></body></html>
"""


def write_index(out_dir, entries):
    cards = []
    for e in entries:
        chips = (
            f'<span class="chip"><b>{e["classes"]}</b> classes</span>'
            f'<span class="chip"><b>{e["rules"]}</b> rules</span>'
            f'<span class="chip"><b>{e["conds"]:,}</b> conditions</span>'
            f'<span class="chip">{esc(e["binariser"])}</span>'
        )
        if e.get("accuracy") is not None:
            chips += f'<span class="chip"><b>{e["accuracy"]:.1f}%</b> test accuracy</span>'
        cards.append(
            f'<a class="card" href="{esc(e["file"])}">'
            f'<h2>{esc(e["title"])}</h2>'
            f'<p class="desc">{esc(e["desc"])}</p>'
            f'<div class="chips">{chips}</div></a>'
        )
    htmlout = INDEX_TEMPLATE.replace("__CARDS__", "\n".join(cards))
    path = os.path.join(out_dir, "index.html")
    with open(path, "w") as f:
        f.write(htmlout)
    print(f"wrote {path}  ({len(entries)} models)")


DATASET_DESC = {
    "wustl": "WUSTL-EHMS-2020 — IoT health-monitoring intrusion detection.",
    "nslkdd": "NSL-KDD — classic network intrusion benchmark.",
    "ton_iot": "TON_IoT — telemetry / network attacks across IoT devices.",
    "medsec": "MedSec-25 — medical-device security telemetry.",
}


def build_one(atlas_path, csv_path=None, out=None, title=None):
    """Generate one full-tree HTML file; return an index-entry dict."""
    with open(atlas_path) as f:
        atlas = json.load(f)
    if csv_path is None:
        guess = atlas_path.replace("_atlas.json", "_predictions.csv")
        csv_path = guess if os.path.exists(guess) else None

    stats = load_firing_stats(csv_path, [str(c) for c in atlas["model"]["classes"]])
    tree = build_tree(atlas, stats)

    name = os.path.basename(atlas_path).replace("_atlas.json", "")
    title = title or f"Full decision tree — {name} Tsetlin Machine"
    out = out or atlas_path.replace("_atlas.json", "_full_tree.html")

    with open(out, "w") as f:
        f.write(render_html(tree, title))

    nclauses = sum(len(p["clauses"]) for c in tree["class_nodes"] for p in c["polarities"])
    nconds = sum(len(cl["conds"]) for c in tree["class_nodes"]
                 for p in c["polarities"] for cl in p["clauses"])
    print(f"wrote {out}")
    print(f"  classes : {len(tree['classes'])}   rules : {nclauses}   conditions : {nconds}")
    if tree["accuracy"] is not None:
        print(f"  test samples: {tree['n_samples']}  accuracy: {tree['accuracy']:.2f}%")
    return {
        "name": name,
        "file": os.path.basename(out),
        "title": title,
        "desc": DATASET_DESC.get(name, f"{name} Tsetlin Machine."),
        "classes": len(tree["classes"]),
        "rules": nclauses,
        "conds": nconds,
        "binariser": tree["meta"].get("binariser", "—"),
        "accuracy": tree["accuracy"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("atlas", nargs="*", help="one or more <name>_atlas.json paths")
    ap.add_argument("csv", nargs="?", default=None,
                    help="predictions CSV (only when a single atlas is given; else auto-located)")
    ap.add_argument("-o", "--out", default=None, help="output HTML path (single-atlas mode)")
    ap.add_argument("-t", "--title", default=None, help="page title (single-atlas mode)")
    ap.add_argument("--index-dir", default=None,
                    help="also write an index.html linking all generated trees, into this dir")
    args = ap.parse_args(argv)

    if not args.atlas:
        ap.error("at least one atlas path is required")

    # In multi-atlas mode, place each tree in --index-dir (if given) so the
    # index and the pages it links live together.
    def out_for(atlas_path):
        if not args.index_dir:
            return None
        base = os.path.basename(atlas_path).replace("_atlas.json", "_full_tree.html")
        return os.path.join(args.index_dir, base)

    # Single-atlas mode keeps the original positional (atlas, csv) calling form.
    if len(args.atlas) == 1:
        entries = [build_one(args.atlas[0], args.csv,
                             args.out or out_for(args.atlas[0]), args.title)]
    else:
        entries = [build_one(a, out=out_for(a)) for a in args.atlas]

    if args.index_dir:
        write_index(args.index_dir, entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
