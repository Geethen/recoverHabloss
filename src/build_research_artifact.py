"""Render the two-tower research ledger as a self-contained interactive page.

Reads ``data/analysis_results/twotower_lab_metrics.csv`` (written by
``rescore_ledger.py`` from the cached out-of-fold probabilities), the
append-only ``twotower_lab_ledger.csv`` for run order, and the backlog table in
``TWOTOWER_RESEARCH.md``. Writes one HTML file with the data inlined as JSON --
no network, no CDN, so it publishes as an Artifact under a strict CSP.

The page is a lab readout with three switchable objectives on one curve, because
the search had been optimising *one* of them and only one:

``change_f1``           the historical headline -- binary change vs no-change.
``macro_f1``            unweighted mean over the four merged2 transitions, so a
                        gain cannot come from the 71%-prevalent stable class.
``art_stable_recall``   stable Artificial found -- the failure the deployed maps
                        show and the ledger could not see.

Re-run after every experiment; the Artifact keeps its URL.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent.parent
LEDGER = project_data_dir("analysis_results") / "twotower_lab_ledger.csv"
METRICS = project_data_dir("analysis_results") / "twotower_lab_metrics.csv"
BACKLOG = SRC / "TWOTOWER_RESEARCH.md"
OUT = ROOT / "scratchpad" / "twotower_research.html"

NOISE = 0.005  # seed-noise band: a delta inside this is not a result
DEPLOYED = "mc_dropout_scalars"
DEPLOY_RUN = "best_20260725_114640"

READS = {
    "full": {"label": "Deploy set", "sub": "6,414 plots · Tessera on 36%"},
    "subset": {"label": "Covered set", "sub": "2,309 plots · both modalities real"},
}

# The three objectives the page can plot. ``goal`` says which direction is good,
# so the running-best line and the win/neg verdicts follow the metric rather than
# assuming higher is always better.
OBJECTIVES = {
    "change_f1": {
        "label": "Change F1",
        "blurb": "Binary change vs no-change on the merged2 labels — the metric "
                 "every idea in this ledger was selected on.",
        "goal": "max",
    },
    "macro_f1": {
        "label": "Macro F1",
        "blurb": "Unweighted mean F1 over the four merged2 transitions. Stable "
                 "Vegetation is 71% of plots, so this is the read that will not "
                 "reward a model for getting the majority class right.",
        "goal": "max",
    },
    "art_stable_recall": {
        "label": "Stable-Artificial recall",
        "blurb": "Share of the 979 stable built-up plots the model actually calls "
                 "stable built-up. Invisible to change-F1 — both this class and "
                 "stable Vegetation are 'no change' — and the largest error mass "
                 "on the map.",
        "goal": "max",
    },
}


def load_rows() -> list[dict]:
    """One row per idea x read, carrying every metric plus the run order."""
    if not METRICS.exists():
        return []
    frame = pd.read_csv(METRICS)
    if LEDGER.exists():
        order = pd.read_csv(LEDGER).drop_duplicates(subset=["idea", "read"], keep="last")
        order = order[["idea", "read", "timestamp"]]
        frame = frame.merge(order, on=["idea", "read"], how="left")
        frame = frame.sort_values("timestamp", na_position="first")
    return frame.reset_index(drop=True).to_dict("records")


def parse_backlog() -> list[dict]:
    """The backlog table rows: id, idea, status -- the queue shown on the page."""
    if not BACKLOG.exists():
        return []
    items, section = [], ""
    for line in BACKLOG.read_text().splitlines():
        head = re.match(r"^### [A-Z]\. (.+)$", line.strip())
        if head:
            section = head.group(1)
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"[A-Z]\d+", cells[0]):
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", cells[1])
            bold = re.match(r"^([^.]+)\.\s*(.*)$", title)
            items.append({
                "id": cells[0], "section": section,
                "title": bold.group(1) if bold else title,
                "detail": bold.group(2) if bold else "",
                "status": cells[2] or "TODO", "result": cells[3],
            })
    return items


def confusion() -> dict | None:
    """The deployed model's merged2 confusion matrix, from its cached OOF probs.

    Recomputed here rather than stored, so the panel can never drift out of step
    with the probabilities the rest of the page is scoring.
    """
    try:
        import twotower_lab as lab
    except Exception:
        return None
    cached = [lab.load_oof(DEPLOYED, "full", s) for s in range(5)]
    stack = [c[0] for c in cached if c is not None]
    if not stack:
        return None
    ctx = lab.load_context()
    view = ctx.view("full")
    classes = [str(c) for c in view.merged_classes]
    pred = lab.labels_from_probs(np.mean(stack, axis=0), view.merged_classes)
    truth = view.truth_merged
    matrix = [[int(((truth == t) & (pred == p)).sum()) for p in classes] for t in classes]
    return {"classes": classes, "matrix": matrix,
            "support": [int((truth == c).sum()) for c in classes],
            "seeds": len(stack)}


def build_payload() -> dict:
    rows = load_rows()
    refs, points = {}, []
    # The reference is pinned to the AlphaEarth-only model, not to "the best
    # thing in the baseline group" -- otherwise a two-tower baseline becomes its
    # own yardstick and every delta on the page silently changes meaning.
    for read in READS:
        base = next((r for r in rows
                     if r["read"] == read and r["idea"] == "baseline_aef"), None)
        if base is not None:
            refs[read] = {"idea": base["idea"],
                          **{m: base.get(f"{m}_mean") for m in OBJECTIVES}}

    def num(value):
        return None if value is None or (isinstance(value, float) and np.isnan(value)) \
            else round(float(value), 4)

    for i, r in enumerate(rows):
        metrics = {m: num(r.get(f"{m}_mean")) for m in OBJECTIVES}
        sds = {m: num(r.get(f"{m}_std")) for m in OBJECTIVES}
        ref = refs.get(r["read"], {})
        deltas, status = {}, {}
        for m in OBJECTIVES:
            base, val = ref.get(m), metrics[m]
            d = None if base is None or val is None else round(val - base, 4)
            deltas[m] = d
            status[m] = ("reference" if r["group"] == "baseline"
                         else "flat" if d is None or abs(d) <= NOISE
                         else "win" if d > 0 else "neg")
        points.append({
            "n": i + 1, "idea": r["idea"], "read": r["read"], "group": r["group"],
            "m": metrics, "sd": sds, "delta": deltas, "status": status,
            "seeds": int(r["n_seeds"]), "desc": r.get("desc") or "",
            "recall": num(r.get("change_recall_mean")),
            "precision": num(r.get("change_precision_mean")),
            "artAsVeg": num(r.get("art_stable_as_veg_mean")),
            "vegAsArt": num(r.get("veg_stable_as_art_mean")),
            "tessGap": num(r.get("tess_recall_gap_mean")),
            "ens": {m: num(r.get(f"ens_{m}")) for m in OBJECTIVES},
        })

    deployed = next((p for p in points
                     if p["idea"] == DEPLOYED and p["read"] == "full"), None)
    return {
        "points": points, "refs": refs, "noise": NOISE, "reads": READS,
        "objectives": OBJECTIVES, "backlog": parse_backlog(),
        "deployed": deployed, "deployRun": DEPLOY_RUN, "confusion": confusion(),
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


HTML = """<title>Two-tower accuracy · research log</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f4f6f7; --panel: #ffffff; --panel-2: #fafbfb;
    --ink: #0e1418; --ink-2: #55636e; --ink-3: #8b97a1;
    --rule: #dfe5e9; --rule-2: #eef2f4;
    --full: #2a78d6; --subset: #eb6834;
    --win: #007a52; --neg: #c8433f; --flat: #8b7300;
    --heat: 210 60% 45%;
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
    --sans: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
    --serif: "Iowan Old Style", Charter, Georgia, "Times New Roman", serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --bg: #0f1317; --panel: #161c21; --panel-2: #1b2229;
      --ink: #eef2f5; --ink-2: #9dabb6; --ink-3: #6d7c88;
      --rule: #27313a; --rule-2: #1e262d;
      --full: #3987e5; --subset: #d95926;
      --win: #35b083; --neg: #e66767; --flat: #d0a92b;
      --heat: 210 55% 62%;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --bg: #0f1317; --panel: #161c21; --panel-2: #1b2229;
    --ink: #eef2f5; --ink-2: #9dabb6; --ink-3: #6d7c88;
    --rule: #27313a; --rule-2: #1e262d;
    --full: #3987e5; --subset: #d95926;
    --win: #35b083; --neg: #e66767; --flat: #d0a92b;
    --heat: 210 55% 62%;
  }

  body { margin: 0; background: var(--bg); color: var(--ink);
         font-family: var(--sans); line-height: 1.55;
         -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 32px 20px 72px;
          display: flex; flex-direction: column; gap: 28px; }

  .eyebrow { font-family: var(--mono); font-size: 11px; letter-spacing: .14em;
             text-transform: uppercase; color: var(--ink-3); }
  h1 { font-family: var(--mono); font-size: clamp(21px, 3.4vw, 30px);
       font-weight: 600; letter-spacing: -.01em; margin: 6px 0 10px;
       text-wrap: balance; }
  .lede { font-family: var(--serif); font-size: 17px; color: var(--ink-2);
          max-width: 64ch; margin: 0; }
  h2 { font-family: var(--mono); font-size: 12px; letter-spacing: .12em;
       text-transform: uppercase; color: var(--ink-2); font-weight: 600;
       margin: 0 0 14px; }

  .tiles { display: grid; gap: 12px;
           grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }
  .tile { background: var(--panel); border: 1px solid var(--rule);
          border-radius: 3px; padding: 16px 18px; display: flex;
          flex-direction: column; gap: 3px; }
  .tile .k { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em;
             text-transform: uppercase; color: var(--ink-3); }
  .tile .v { font-family: var(--mono); font-size: 30px; font-weight: 600;
             font-variant-numeric: tabular-nums; letter-spacing: -.02em; }
  .tile .s { font-size: 12.5px; color: var(--ink-2); }
  .tile.lead-full { border-top: 2px solid var(--full); }
  .tile.lead-subset { border-top: 2px solid var(--subset); }
  .tile.alarm { border-top: 2px solid var(--neg); }

  .card { background: var(--panel); border: 1px solid var(--rule);
          border-radius: 3px; padding: 20px; }
  .chart-head { display: flex; flex-wrap: wrap; gap: 14px;
                align-items: baseline; justify-content: space-between;
                margin-bottom: 10px; }
  .legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 12.5px;
            color: var(--ink-2); font-family: var(--mono); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 11px; height: 11px; border-radius: 50%; }

  .seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 3px;
         overflow: hidden; background: var(--panel-2); }
  .seg button { font-family: var(--mono); font-size: 11.5px; letter-spacing: .04em;
                padding: 7px 13px; border: 0; background: transparent;
                color: var(--ink-2); cursor: pointer; border-right: 1px solid var(--rule); }
  .seg button:last-child { border-right: 0; }
  .seg button[aria-pressed="true"] { background: var(--ink); color: var(--panel); }
  .seg button:focus-visible { outline: 2px solid var(--full); outline-offset: -2px; }
  .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
              margin-bottom: 14px; }
  .blurb { font-family: var(--serif); font-size: 14px; color: var(--ink-2);
           max-width: 70ch; margin: 0 0 14px; }

  figure { margin: 0; }
  figcaption { font-size: 12px; color: var(--ink-3); margin-top: 10px;
               font-family: var(--serif); font-style: italic; }
  .plot { position: relative; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .grid line { stroke: var(--rule-2); stroke-width: 1; }
  .axis text { font-family: var(--mono); font-size: 10.5px; fill: var(--ink-3);
               font-variant-numeric: tabular-nums; }
  .reflabel { font-family: var(--mono); font-size: 10px;
              font-variant-numeric: tabular-nums; }
  .dot { cursor: pointer; }
  .dot:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }

  #tip { position: absolute; pointer-events: none; opacity: 0;
         transition: opacity .12s; background: var(--panel);
         border: 1px solid var(--rule); border-radius: 3px; padding: 10px 12px;
         box-shadow: 0 6px 22px rgba(6,14,20,.16); max-width: 300px; z-index: 5; }
  #tip .t { font-family: var(--mono); font-size: 12.5px; font-weight: 600; }
  #tip .m { font-family: var(--mono); font-size: 11.5px; color: var(--ink-2);
            font-variant-numeric: tabular-nums; margin-top: 3px; }
  #tip .d { font-size: 12px; color: var(--ink-2); margin-top: 6px;
            font-family: var(--serif); }

  .split { display: grid; gap: 20px; grid-template-columns: 1fr; }
  @media (min-width: 780px) { .split { grid-template-columns: 1.05fr .95fr; } }

  .cm { border-collapse: separate; border-spacing: 2px; font-family: var(--mono);
        font-size: 12px; font-variant-numeric: tabular-nums; }
  .cm th { font-size: 10px; letter-spacing: .06em; text-transform: uppercase;
           color: var(--ink-3); font-weight: 600; padding: 3px 6px;
           white-space: nowrap; text-align: right; border: 0; }
  .cm th.col { text-align: center; }
  .cm td { text-align: center; padding: 9px 6px; border-radius: 2px;
           min-width: 58px; color: var(--ink); }
  .cm td.diag { font-weight: 600; }
  .cm td.worst { outline: 2px solid var(--neg); outline-offset: -2px; }

  .facts { list-style: none; margin: 0; padding: 0; display: flex;
           flex-direction: column; gap: 12px; }
  .facts li { border-left: 2px solid var(--rule); padding-left: 12px; }
  .facts li.bad { border-left-color: var(--neg); }
  .facts li.ok { border-left-color: var(--win); }
  .facts .fk { font-family: var(--mono); font-size: 10.5px; letter-spacing: .09em;
               text-transform: uppercase; color: var(--ink-3); }
  .facts .fv { font-family: var(--mono); font-size: 15px; font-weight: 600;
               font-variant-numeric: tabular-nums; }
  .facts .fd { font-size: 12.5px; color: var(--ink-2); font-family: var(--serif); }

  table.board { width: 100%; border-collapse: collapse; font-size: 13px; }
  .scroll { overflow-x: auto; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--rule-2);
           white-space: nowrap; }
  table.board th { font-family: var(--mono); font-size: 10.5px; letter-spacing: .08em;
       text-transform: uppercase; color: var(--ink-3); font-weight: 600; }
  td.num { font-family: var(--mono); font-variant-numeric: tabular-nums;
           text-align: right; }
  td.idea { font-family: var(--mono); font-size: 12.5px; }
  tr.is-win td { background: color-mix(in srgb, var(--win) 8%, transparent); }
  tr.is-deployed td { background: color-mix(in srgb, var(--full) 9%, transparent); }

  .chip { font-family: var(--mono); font-size: 10px; letter-spacing: .08em;
          text-transform: uppercase; padding: 2px 7px; border-radius: 2px;
          border: 1px solid currentColor; }
  .c-win { color: var(--win); } .c-neg { color: var(--neg); }
  .c-flat { color: var(--flat); } .c-reference { color: var(--ink-3); }
  .c-todo { color: var(--ink-3); } .c-running { color: var(--full); }
  .c-drop { color: var(--ink-3); }

  .queue { display: grid; gap: 10px;
           grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
  .item { background: var(--panel-2); border: 1px solid var(--rule);
          border-left: 2px solid var(--rule); border-radius: 2px; padding: 12px 14px; }
  .item.s-win { border-left-color: var(--win); }
  .item.s-neg { border-left-color: var(--neg); }
  .item.s-flat { border-left-color: var(--flat); }
  .item.s-running { border-left-color: var(--full); }
  .item .h { display: flex; gap: 8px; align-items: center;
             justify-content: space-between; }
  .item .id { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
  .item .n { font-family: var(--mono); font-size: 13px; font-weight: 600;
             margin: 4px 0 3px; }
  .item .d { font-size: 12.5px; color: var(--ink-2); font-family: var(--serif); }
  .sect { font-family: var(--mono); font-size: 11px; letter-spacing: .1em;
          text-transform: uppercase; color: var(--ink-3); margin: 18px 0 8px; }

  .note { font-family: var(--serif); font-size: 14.5px; color: var(--ink-2);
          max-width: 68ch; }
  .note b { color: var(--ink); font-weight: 600; }
  footer { font-family: var(--mono); font-size: 11px; color: var(--ink-3);
           border-top: 1px solid var(--rule); padding-top: 14px; line-height: 1.9; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow">RECOVER / HABLOSS · vegetation→artificial transitions</div>
    <h1>Two-tower accuracy: what moves it</h1>
    <p class="lede">AlphaEarth gives context — dense, global, present on every
    plot. Tessera gives detail — 10&nbsp;m Sentinel-1+2, sharper but noisier and
    present for only a third of plots. Forty-odd fusions of the two have been
    scored here on binary change-F1. Under that metric a built-up block read as
    stable nature is a <em>correct</em> answer, which is why the largest error on
    the map never appeared in the ledger.</p>
  </header>

  <section class="tiles" id="tiles"></section>

  <section class="card">
    <h2>Deployed model · <span id="deployName"></span></h2>
    <div class="split">
      <figure>
        <div class="scroll"><table class="cm" id="cm"></table></div>
        <figcaption id="cmCap"></figcaption>
      </figure>
      <ul class="facts" id="facts"></ul>
    </div>
  </section>

  <section class="card">
    <div class="controls">
      <div class="seg" id="metricSeg" role="group" aria-label="Objective"></div>
      <div class="legend" id="legend"></div>
    </div>
    <p class="blurb" id="blurb"></p>
    <div class="chart-head"><h2 style="margin:0" id="chartTitle"></h2></div>
    <figure>
      <div class="plot">
        <svg id="chart" viewBox="0 0 960 420" role="img" aria-labelledby="chartTitle"></svg>
        <div id="tip" role="status"></div>
      </div>
      <figcaption>Each point is one idea scored over 1–5 torch seeds under the same
      5-fold spatially blocked CV; the step line is the best result so far. The
      horizontal rule is the AlphaEarth-only incumbent for that read. Vertical axis
      is zoomed — the whole story lives inside a 0.1 band.</figcaption>
    </figure>
  </section>

  <section class="card">
    <h2>Every result · sorted by <span id="boardMetric"></span></h2>
    <div class="scroll"><table class="board" id="board"></table></div>
  </section>

  <section class="card">
    <h2>Queue</h2>
    <p class="note" style="margin-top:-4px">Sections A–E asked <b>how should the two
    modalities be fused</b>, and the answer came back the same way every time: not
    smarter, just regularise the noisy one harder and average over the uncertainty
    in trusting it. Section F asks a different question — <b>what is the model
    actually getting wrong</b> — and it is not change detection.</p>
    <div id="queue"></div>
  </section>

  <footer id="foot"></footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const COL = { full: "var(--full)", subset: "var(--subset)" };
const f3 = v => (v == null || Number.isNaN(v)) ? "—" : v.toFixed(3);
const f4 = v => (v == null || Number.isNaN(v)) ? "—" : v.toFixed(4);
const pct = v => (v == null || Number.isNaN(v)) ? "—" : (100 * v).toFixed(1) + "%";
const sgn = v => (v == null) ? "—" : (v >= 0 ? "+" : "") + v.toFixed(4);
const el = (tag, cls, txt) => { const n = document.createElement(tag);
  if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };

let METRIC = "change_f1";
const has = p => p.m[METRIC] != null;

/* ---- tiles ------------------------------------------------------------- */
function drawTiles() {
  const tiles = document.getElementById("tiles");
  tiles.innerHTML = "";
  for (const [rd, meta] of Object.entries(D.reads)) {
    const pts = D.points.filter(p => p.read === rd && has(p));
    if (!pts.length) continue;
    const L = pts.reduce((a, b) => (b.m[METRIC] > a.m[METRIC] ? b : a));
    const ref = D.refs[rd];
    const t = el("div", "tile lead-" + rd);
    t.append(el("div", "k", meta.label + " · best " + D.objectives[METRIC].label),
             el("div", "v", f4(L.m[METRIC])));
    const d = ref && ref[METRIC] != null ? L.m[METRIC] - ref[METRIC] : null;
    t.append(el("div", "s", L.idea + (d != null ? ` · ${sgn(d)} vs ${ref.idea}` : "")));
    tiles.append(t);
  }
  const wins = D.points.filter(p => p.status[METRIC] === "win").length;
  const t3 = el("div", "tile");
  t3.append(el("div", "k", "Ideas scored"), el("div", "v", String(D.points.length)));
  t3.append(el("div", "s", wins + " past the ±" + D.noise + " seed-noise band"));
  tiles.append(t3);

  if (D.deployed) {
    const t4 = el("div", "tile alarm");
    t4.append(el("div", "k", "Stable built-up → stable nature"),
              el("div", "v", pct(D.deployed.artAsVeg)));
    t4.append(el("div", "s", "of the 979 stable-Artificial plots · unmoved by every idea tested"));
    tiles.append(t4);
  }
}

/* ---- deployed-model panel ---------------------------------------------- */
function drawDeployed() {
  const dep = D.deployed;
  document.getElementById("deployName").textContent =
    (dep ? dep.idea : "—") + " · " + D.deployRun;

  const cm = document.getElementById("cm");
  const cap = document.getElementById("cmCap");
  cm.innerHTML = "";
  if (D.confusion) {
    const { classes, matrix } = D.confusion;
    const short = c => c.replace(/Artificial/g, "Art").replace(/Vegetation/g, "Veg")
                        .replace(" -> ", " → ");
    const head = el("tr");
    head.append(el("th", "", "truth ↓ / predicted →"));
    classes.forEach(c => head.append(el("th", "col", short(c))));
    cm.append(head);
    let worst = { v: -1 };
    matrix.forEach((row, i) => row.forEach((v, j) => {
      if (i !== j && v > worst.v) worst = { v, i, j };
    }));
    matrix.forEach((row, i) => {
      const tr = el("tr");
      tr.append(el("th", "", short(classes[i])));
      const total = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((v, j) => {
        const td = el("td", (i === j ? "diag" : "") +
          (i === worst.i && j === worst.j ? " worst" : ""), String(v));
        const a = Math.pow(v / total, 0.6);
        td.style.background = `hsl(var(--heat) / ${(0.06 + 0.72 * a).toFixed(3)})`;
        if (a > 0.55) td.style.color = "var(--panel)";
        td.title = `${classes[i]} predicted as ${classes[j]}: ${v} plots (${pct(v / total)} of the class)`;
        tr.append(td);
      });
      cm.append(tr);
    });
    cap.textContent = `Out-of-fold, ${D.confusion.seeds} seeds averaged, ` +
      `6,414 plots. Cell shade is the share of that truth row. The outlined cell ` +
      `is the largest confusion: ${worst.v} stable-Artificial plots returned as ` +
      `stable Vegetation.`;
  }

  const facts = document.getElementById("facts");
  facts.innerHTML = "";
  if (!dep) return;
  const add = (cls, k, v, d) => {
    const li = el("li", cls);
    li.append(el("div", "fk", k), el("div", "fv", v), el("div", "fd", d));
    facts.append(li);
  };
  add("ok", "Change F1", f4(dep.m.change_f1),
      `+${(dep.delta.change_f1 ?? 0).toFixed(4)} over AlphaEarth alone — three ` +
      `independent gains in one network: asymmetric Tessera dropout, per-modality ` +
      `change scalars, and a gate left stochastic at test time.`);
  add("ok", "Change recall where Tessera fires", f3(dep.tessGap),
      `Recall gap between Tessera-covered plots and the rest. Positive, so at plot ` +
      `level the detail modality finds more change than it omits — the earlier ` +
      `symmetric two-tower was the one that omitted there (0.694 vs 0.754).`);
  add("bad", "Stable Artificial found", f3(dep.m.art_stable_recall),
      `One stable built-up plot in three is missed, and ${pct(dep.artAsVeg)} of the ` +
      `class comes back as stable Vegetation. Both labels are "no change", so ` +
      `change-F1 scores this failure as a success.`);
  add("bad", "Macro F1 vs change F1", f4(dep.m.macro_f1),
      `The four-class read sits above the binary one only because stable ` +
      `Vegetation (71% of plots) is easy. Judge new ideas on both.`);
}

/* ---- chart ------------------------------------------------------------- */
const NS = "http://www.w3.org/2000/svg";
const mk = (tag, attrs = {}) => { const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };
const tipEl = document.getElementById("tip");

function drawChart() {
  const svg = document.getElementById("chart");
  svg.innerHTML = "";
  document.getElementById("chartTitle").textContent =
    "Running best · " + D.objectives[METRIC].label;
  document.getElementById("blurb").textContent = D.objectives[METRIC].blurb;

  const byRead = {};
  for (const rd of Object.keys(D.reads)) {
    let best = -Infinity;
    byRead[rd] = D.points.filter(p => p.read === rd && has(p))
      .map(p => ({ ...p, best: (best = Math.max(best, p.m[METRIC])) }));
  }

  const W = 960, H = 420, M = { t: 18, r: 126, b: 40, l: 54 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const vals = D.points.filter(has).map(p => p.m[METRIC])
    .concat(Object.values(D.refs).map(r => r[METRIC]).filter(v => v != null));
  if (!vals.length) return;
  const lo = Math.min(...vals) - 0.012, hi = Math.max(...vals) + 0.012;
  const nMax = Math.max(6, ...D.points.map(p => p.n));
  const x = n => M.l + (nMax > 1 ? (n - 1) / (nMax - 1) : 0.5) * iw;
  const y = v => M.t + (1 - (v - lo) / (hi - lo)) * ih;

  const step = (hi - lo) > 0.12 ? 0.02 : 0.01;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(+v.toFixed(3));
  const g = mk("g", { class: "grid" });
  ticks.forEach(v => g.append(mk("line", { x1: M.l, x2: M.l + iw, y1: y(v), y2: y(v) })));
  svg.append(g);

  const ax = mk("g", { class: "axis" });
  ticks.forEach(v => {
    const tx = mk("text", { x: M.l - 10, y: y(v) + 3.5, "text-anchor": "end" });
    tx.textContent = v.toFixed(2); ax.append(tx);
  });
  const xl = mk("text", { x: M.l + iw / 2, y: H - 6, "text-anchor": "middle" });
  xl.textContent = "experiments, in the order they were run"; ax.append(xl);
  // Anchored at the plot's left edge, not right of the tick column: the longest
  // objective name is wider than the margin the ticks leave.
  const yl = mk("text", { x: M.l, y: M.t - 6, "text-anchor": "start" });
  yl.textContent = D.objectives[METRIC].label; ax.append(yl);
  svg.append(ax);

  // Right-margin labels are collected, de-collided, then drawn: the two reads'
  // incumbents can land within a pixel of each other on a zoomed axis.
  const labels = [];
  for (const [rd, ref] of Object.entries(D.refs)) {
    if (ref[METRIC] == null) continue;
    svg.append(mk("line", { x1: M.l, x2: M.l + iw, y1: y(ref[METRIC]), y2: y(ref[METRIC]),
      stroke: COL[rd], "stroke-width": 1, opacity: .45, "stroke-dasharray": "4 3" }));
    labels.push({ y: y(ref[METRIC]), text: "AlphaEarth " + f4(ref[METRIC]),
                  fill: COL[rd], weight: "400" });
  }

  for (const [rd, pts] of Object.entries(byRead)) {
    if (!pts.length) continue;
    let d = "";
    pts.forEach((p, i) => {
      d += i === 0 ? `M${x(p.n)},${y(p.best)}`
                   : `L${x(p.n)},${y(pts[i-1].best)}L${x(p.n)},${y(p.best)}`;
    });
    d += `L${M.l + iw},${y(pts[pts.length - 1].best)}`;
    svg.append(mk("path", { d, fill: "none", stroke: COL[rd], "stroke-width": 2,
      "stroke-linejoin": "round", opacity: .9 }));

    const end = pts[pts.length - 1];
    labels.push({ y: y(end.best) - 8, text: D.reads[rd].label + " " + f4(end.best),
                  fill: COL[rd], weight: "600" });

    pts.forEach(p => {
      const isTop = p.m[METRIC] === end.best;
      const isDep = p.idea === (D.deployed || {}).idea && p.read === "full";
      const c = mk("circle", { cx: x(p.n), cy: y(p.m[METRIC]), r: isTop ? 6 : 4.5,
        fill: p.status[METRIC] === "reference" ? "var(--panel)" : COL[rd],
        stroke: isDep ? "var(--ink)" : COL[rd], "stroke-width": isDep ? 2.5 : 2,
        class: "dot", tabindex: "0", role: "img",
        "aria-label": `${p.idea}, ${D.reads[rd].label}, ` +
          `${D.objectives[METRIC].label} ${f4(p.m[METRIC])}` });
      const show = ev => tip(ev, p, svg, x, y);
      c.addEventListener("mouseenter", show);
      c.addEventListener("focus", show);
      c.addEventListener("mousemove", show);
      c.addEventListener("mouseleave", hide);
      c.addEventListener("blur", hide);
      svg.append(c);
    });
  }

  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++)
    labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 13);
  labels.forEach(L => {
    const t = mk("text", { x: M.l + iw + 8, y: L.y + 3.5, class: "reflabel",
      fill: L.fill, "font-weight": L.weight });
    t.textContent = L.text;
    svg.append(t);
  });
}

function tip(ev, p, svg, x, y) {
  const r = svg.getBoundingClientRect();
  tipEl.innerHTML = "";
  tipEl.append(el("div", "t", p.idea));
  tipEl.append(el("div", "m",
    `${D.reads[p.read].label} · ${D.objectives[METRIC].label} ` +
    `${f4(p.m[METRIC])} ±${f3(p.sd[METRIC])} · ${p.seeds} seed${p.seeds > 1 ? "s" : ""}`));
  tipEl.append(el("div", "m",
    `change ${f3(p.m.change_f1)} · macro ${f3(p.m.macro_f1)} · ` +
    `art-stable ${f3(p.m.art_stable_recall)}`));
  tipEl.append(el("div", "m",
    `art→veg ${pct(p.artAsVeg)} · veg→art ${pct(p.vegAsArt)}` +
    (p.tessGap != null ? ` · tess recall gap ${sgn(p.tessGap)}` : "")));
  if (p.delta[METRIC] != null && p.status[METRIC] !== "reference")
    tipEl.append(el("div", "m",
      `${sgn(p.delta[METRIC])} vs AlphaEarth · ${p.status[METRIC]}`));
  if (p.desc) tipEl.append(el("div", "d", p.desc));
  const px = (ev.clientX ?? r.left + x(p.n)) - r.left;
  const py = (ev.clientY ?? r.top + y(p.m[METRIC])) - r.top;
  tipEl.style.opacity = 1;
  tipEl.style.left = Math.min(Math.max(px + 14, 0), r.width - tipEl.offsetWidth - 4) + "px";
  tipEl.style.top = Math.max(py - tipEl.offsetHeight - 12, 0) + "px";
}
function hide() { tipEl.style.opacity = 0; }

/* ---- leaderboard ------------------------------------------------------- */
function drawBoard() {
  document.getElementById("boardMetric").textContent = D.objectives[METRIC].label;
  const board = document.getElementById("board");
  board.innerHTML = `<thead><tr><th>Idea</th><th>Read</th>
    <th class="num">${D.objectives[METRIC].label}</th><th class="num">±sd</th>
    <th class="num">Δ vs AlphaEarth</th><th class="num">Change F1</th>
    <th class="num">Macro F1</th><th class="num">Art-stable recall</th>
    <th class="num">Art→Veg</th><th>Verdict</th></tr></thead>`;
  const tb = el("tbody");
  const dep = D.deployed || {};
  [...D.points].filter(has).sort((a, b) => b.m[METRIC] - a.m[METRIC]).forEach(p => {
    const isDep = p.idea === dep.idea && p.read === dep.read;
    const tr = el("tr", isDep ? "is-deployed"
                     : p.status[METRIC] === "win" ? "is-win" : "");
    [[p.idea + (isDep ? "  ← deployed" : ""), "idea"],
     [D.reads[p.read].label, ""], [f4(p.m[METRIC]), "num"],
     [f3(p.sd[METRIC]), "num"],
     [p.status[METRIC] === "reference" ? "—" : sgn(p.delta[METRIC]), "num"],
     [f4(p.m.change_f1), "num"], [f4(p.m.macro_f1), "num"],
     [f3(p.m.art_stable_recall), "num"], [pct(p.artAsVeg), "num"],
    ].forEach(([v, c]) => tr.append(el("td", c, v)));
    const td = el("td");
    td.append(el("span", "chip c-" + p.status[METRIC], p.status[METRIC]));
    tr.append(td); tb.append(tr);
  });
  board.append(tb);
}

/* ---- static chrome ----------------------------------------------------- */
const lg = document.getElementById("legend");
for (const [rd, meta] of Object.entries(D.reads)) {
  const s = el("span");
  const sw = el("span", "swatch"); sw.style.background = COL[rd];
  s.append(sw, document.createTextNode(meta.label + " · " + meta.sub));
  lg.append(s);
}

const seg = document.getElementById("metricSeg");
for (const [key, meta] of Object.entries(D.objectives)) {
  const b = el("button", "", meta.label);
  b.setAttribute("aria-pressed", String(key === METRIC));
  b.addEventListener("click", () => {
    METRIC = key;
    [...seg.children].forEach(c => c.setAttribute("aria-pressed",
      String(c === b)));
    render();
  });
  seg.append(b);
}

const q = document.getElementById("queue");
[...new Set(D.backlog.map(b => b.section))].forEach(s => {
  q.append(el("div", "sect", s));
  const grid = el("div", "queue");
  D.backlog.filter(b => b.section === s).forEach(b => {
    const st = b.status.toLowerCase();
    const it = el("div", "item s-" + st);
    const h = el("div", "h");
    h.append(el("span", "id", b.id), el("span", "chip c-" + st, b.status));
    it.append(h, el("div", "n", b.title));
    if (b.detail) it.append(el("div", "d", b.detail));
    if (b.result) it.append(el("div", "d", b.result));
    grid.append(it);
  });
  q.append(grid);
});

document.getElementById("foot").textContent =
  `built ${D.built} · ledger: data/analysis_results/twotower_lab_ledger.csv · ` +
  `metrics: twotower_lab_metrics.csv · harness: src/twotower_lab.py · ` +
  `backlog: src/TWOTOWER_RESEARCH.md`;

function render() { drawTiles(); drawChart(); drawBoard(); }
drawDeployed();
render();
</script>
"""


def main() -> None:
    payload = build_payload()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print(f"wrote {OUT} ({len(payload['points'])} results, "
          f"{len(payload['backlog'])} backlog items)")


if __name__ == "__main__":
    main()
