"""Draw the label-audit batches: re-read labels already on file, blind.

No new locations. The ledger says more labels buy +0.026 change-F1 per doubling
and that the Cropland/Nature label noise is the ceiling, so this round checks
the labels we train on instead of adding to them.

Two sets, two pairs of readers
------------------------------
**Expert set** (`aud_sup*`, read by both super experts, e3 Zander and e4
Geethen). The hard cases:

* every **disagreement** already on file --
  - round one (cov001 + rar001): e1 and e2 gave different transitions, or one
    called the point and the other called it uninterpretable;
  - the original frame: a PLOTID labelled twice (the RECOVER re-verification)
    whose two readings differ at the coarse (Nature/Cropland/Artificial) level;
* the rest of the budget is the most **uncertain** points of both pools under
  the best four-class Nature model (`infer_nature4.py`'s class-wise ensemble),
  read at nine classes by its chain-rule composition
  (`src/audit_posteriors_nature4.py`): entropy of the seed-mean nine-class
  posterior and the seed disagreement of the ensemble (mutual information),
  combined as the mean of their percentile ranks. Out-of-fold on the original
  frame (the members' cached OOF), and fitted on the original frame only for
  round one, so no point scores itself. (The first draw of 2026-10-13 used the
  deployed coarse3 recipe, `src/audit_posteriors.py`; superseded.)

**Reader set** (`aud_reg*`, read by both e1 Dina and e2 Øyvind). A stratified
random draw from the original frame, after the expert set is removed: a floor
per coarse transition and the rest proportional. Each row carries its inclusion
probability, so this set -- together with the expert set's uncertain stratum,
which is a census of the top of the ranking -- estimates the frame's error rate.
Round-one points are not in it: e1 and e2 made those calls themselves.

Every point is read by both readers of its set. A single re-reading against the
file is one person against one person, and cannot be settled.

What the app does with it
-------------------------
The label on file travels as `assigned` and is NOT shown until the reader has
saved their own call (see `app/js/app.js`, "audit"). The first save is kept as
`blind_transition`; that is the unanchored measurement, and nothing after the
reveal can change it.

Usage::

    G=/home/geethen.singh/.pixi/envs/geo/bin/python
    $G src/build_audit_batches.py --readings data/label_audit/sheet_export_20261013.csv

Writes ``app/batches/aud_sup00N.json``, ``aud_reg00N.json``, merges them into
``index.json``, and writes the draw itself (stratum, scores, inclusion
probability, every reading on file) to ``data/label_audit/audit_sets_<date>.csv``.
Refuses to overwrite an existing audit batch: a re-draw part-way through a
round moves the points under the readers.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import build_label_batches as blb
from project_paths import project_data_dir

REPO = Path(__file__).resolve().parents[1]
AUDIT_DIR = project_data_dir("label_audit")
ORIG = project_data_dir("embeddings") / "embeddings_habloss_recover.parquet"
ROUND1 = project_data_dir("labels_round1") / "round1_points_dina_20261005.csv"
ROUND1_BATCHES = ("cov001", "rar001")
ROUND1_READERS = ("e1", "e2")
POSTERIORS = AUDIT_DIR / "posteriors_nature4_20261013_summary.parquet"

#: How an uninterpretable reading is written in `assigned`. The app compares a
#: call against these strings, and maps its own uninterpretable save to it.
CANNOT = "cannot interpret"

LEGEND = {"Nature", "Cropland", "Artificial"}

SUPER_EXPERTS = ["e3", "e4"]
READER_EXPERTS = ["e1", "e2"]

INSTRUCTIONS = {
    "sup": (
        "Label audit, EXPERT set. Every sample here already has a label on file, "
        "and these are the hard ones: points where two earlier readers disagreed, "
        "and points the model is least sure of. Label each one exactly as a "
        "normal sample. After you save, the label on file is shown (two, where "
        "two readers disagreed). If your call matches neither, say why -- or "
        "revise your call. Your first call is kept as the blind reading either way."
    ),
    "reg": (
        "Label audit. Every sample here already has a label on file, drawn at "
        "random from the original training data. Label each one exactly as a "
        "normal sample. After you save, the label on file is shown. If your call "
        "differs, say why -- or revise your call. Your first call is kept as "
        "the blind reading either way."
    ),
}


def coarse(label: str) -> str:
    """`Nature - forest` and `Nature - other` are HABLOSS's split of `Nature`."""
    return re.sub(r"Nature - \w+", "Nature", str(label))


def transition(a: str, b: str) -> str:
    return f"{coarse(a)} -> {coarse(b)}"


# ---------------------------------------------------------------------------
# disagreements
# ---------------------------------------------------------------------------
def round1_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """One row per round-one point: e1's and e2's call, `CANNOT` for no call."""
    import label_rounds
    r = label_rounds.dedupe(readings)
    r = r[r.batch_id.isin(ROUND1_BATCHES) & r.expert_id.isin(ROUND1_READERS)]
    r = r.assign(call=r.transition.fillna("").replace("", CANNOT))
    wide = r.pivot_table(index="point_id", columns="expert_id", values="call",
                         aggfunc="first")
    return wide.reindex(columns=list(ROUND1_READERS))


def round1_disagreements(wide: pd.DataFrame) -> pd.Index:
    """Both read it and the calls differ. One call against an uninterpretable
    is a disagreement; two uninterpretables is agreement that it cannot be read."""
    both = wide.dropna()
    return both.index[both[ROUND1_READERS[0]] != both[ROUND1_READERS[1]]]


def orig_conflicts(orig: pd.DataFrame) -> pd.DataFrame:
    """PLOTIDs labelled twice whose readings differ at the coarse level."""
    dup = orig[orig.PLOTID.duplicated(keep=False)]
    n = dup.groupby("PLOTID").t.nunique()
    return dup[dup.PLOTID.isin(n.index[n > 1])]


# ---------------------------------------------------------------------------
# the draw
# ---------------------------------------------------------------------------
def uncertainty_score(frame: pd.DataFrame, terms: list[str]) -> pd.Series:
    """Mean percentile rank of the named terms, ranked over the pooled frame.

    Ranks rather than raw values because the terms live on different scales --
    mutual information here is ~1% of entropy, and a sum would be entropy alone.
    `1-p_assigned` is accepted as a term: the model's doubt about the label on
    file specifically, which is the closest single column to "this label is wrong".
    """
    cols = []
    for term in terms:
        v = 1 - frame["p_assigned"] if term == "1-p_assigned" else frame[term]
        cols.append(v.rank(pct=True))
    return pd.concat(cols, axis=1).mean(axis=1)


def stratified(frame: pd.DataFrame, n: int, by: str, floor: int,
               rng: np.random.Generator) -> pd.DataFrame:
    """`floor` per stratum (or all of it), the rest proportional by largest
    remainder. Adds `incl_prob` = drawn / available within the stratum."""
    sizes = frame.groupby(by).size()
    take = np.minimum(sizes, floor)
    left = n - int(take.sum())
    if left < 0:
        raise SystemExit(f"floor {floor} x {len(sizes)} strata exceeds n={n}")
    room = sizes - take
    share = room / room.sum() * left
    extra = np.floor(share).astype(int)
    order = (share - extra).sort_values(ascending=False).index
    for s in order[: left - int(extra.sum())]:
        extra[s] += 1
    take = take + np.minimum(extra, room)
    parts = []
    for s, k in take.items():
        g = frame[frame[by] == s]
        pick = g.iloc[rng.permutation(len(g))[: int(k)]].copy()
        pick["incl_prob"] = k / len(g)
        parts.append(pick)
    return pd.concat(parts)


def build(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)

    orig = pd.read_parquet(ORIG, columns=["PLOTID", "lc_2018", "lc_2024",
                                          "source", "lon", "lat"])
    # Every PLOTID table is joined on the string form: a numeric id in one and
    # a string in another would silently fail every exclusion below.
    orig["PLOTID"] = orig.PLOTID.astype(str)
    orig["t"] = [transition(a, b) for a, b in zip(orig.lc_2018, orig.lc_2024)]
    orig["fine"] = orig.lc_2018 + " -> " + orig.lc_2024
    post = pd.read_parquet(args.posteriors)
    post["PLOTID"] = post.PLOTID.astype(str)

    r1 = pd.read_csv(ROUND1, dtype={"PLOTID": str})
    r1_pts = pd.concat([pd.DataFrame(json.loads(
        (blb.BATCH_DIR / f"{b}.json").read_text())["points"])[["id", "lon", "lat"]]
        for b in ROUND1_BATCHES]).drop_duplicates("id").set_index("id")
    wide = round1_readings(pd.read_csv(args.readings))
    r1_dis = round1_disagreements(wide)
    conflicts = orig_conflicts(orig)

    rows = []
    # -- disagreements: all of them, to the super experts ------------------
    for pid in r1_dis:
        calls = [wide.at[pid, e] for e in ROUND1_READERS]
        rows.append(dict(pool="b1", plotid=pid, stratum="disagreement",
                         lon=r1_pts.at[pid, "lon"], lat=r1_pts.at[pid, "lat"],
                         assigned="|".join(calls),
                         assigned_note="Labelling round one: the two readers disagreed.",
                         incl_prob=1.0))
    for pid, g in conflicts.groupby("PLOTID", sort=True):
        rows.append(dict(pool="orig", plotid=pid, stratum="disagreement",
                         lon=g.lon.iloc[0], lat=g.lat.iloc[0],
                         assigned="|".join(g.t),
                         assigned_note="Original data, labelled twice: "
                         + "; ".join(f"{s}: {f}" for s, f in zip(g.source, g.fine)),
                         incl_prob=1.0))
    dis = pd.DataFrame(rows)
    n_fill = args.super_n - len(dis)
    if n_fill < 0:
        raise SystemExit(f"{len(dis)} disagreements exceed --super-n {args.super_n}")

    # -- the uncertain fill, ranked jointly over both pools ----------------
    taken = set(zip(dis.pool, dis.plotid))
    cand = post[[(p, i) not in taken for p, i in zip(post.pool, post.PLOTID)]].copy()
    cand["score"] = uncertainty_score(cand, args.terms)
    top = cand.sort_values("score", ascending=False).head(n_fill)
    first = orig.drop_duplicates("PLOTID").set_index("PLOTID")
    r1_lab = r1.set_index("PLOTID")
    unc = []
    for _, row in top.iterrows():
        pid = row.PLOTID
        if row.pool == "orig":
            o = first.loc[pid]
            lon, lat, assigned = o.lon, o.lat, o.t
            note = f"Original data ({o.source}): {o.fine}"
        else:
            o = r1_lab.loc[pid]
            lon, lat = o.lon, o.lat
            assigned = transition(o.lc_2018, o.lc_2024)
            note = f"Labelling round one ({o.rule.replace('_', ' ')})."
        unc.append(dict(pool=row.pool, plotid=pid, stratum="uncertain",
                        lon=lon, lat=lat, assigned=assigned, assigned_note=note,
                        entropy=row.entropy, mutual_info=row.mutual_info,
                        p_assigned=row.p_assigned, score=row.score,
                        incl_prob=1.0))
    sup = pd.concat([dis, pd.DataFrame(unc)], ignore_index=True)
    sup["set"] = "sup"

    # -- the reader set: stratified random, original frame only ------------
    used = set(sup.loc[sup.pool == "orig", "plotid"])
    pool = post[(post.pool == "orig") & ~post.PLOTID.isin(used)].copy()
    pool = pool.rename(columns={"truth": "stratum_t"})
    drawn = stratified(pool, args.reader_n, "stratum_t", args.floor, rng)
    reg = []
    for _, row in drawn.iterrows():
        o = first.loc[row.PLOTID]
        reg.append(dict(pool="orig", plotid=row.PLOTID, stratum="random",
                        lon=o.lon, lat=o.lat, assigned=o.t,
                        assigned_note=f"Original data ({o.source}): {o.fine}",
                        entropy=row.entropy, mutual_info=row.mutual_info,
                        p_assigned=row.p_assigned, incl_prob=row.incl_prob,
                        stratum_t=row.stratum_t))
    reg = pd.DataFrame(reg)
    reg["set"] = "reg"

    for frame in (sup, reg):
        # Prefixed: the two pools' id schemes are disjoint today, and nothing
        # guarantees it. The point id is part of the sheet key.
        frame["id"] = frame.pool + "_" + frame.plotid.astype(str)
        assert frame["id"].is_unique
        # Mixed, not cut by stratum: a batch that is all disagreements tells
        # the reader so before they look.
        frame.loc[:, "_order"] = rng.permutation(len(frame))
    # The app matches a call against `assigned` as a string, so anything that is
    # not one of the nine transitions (a new legend spelling in a source table)
    # would read as a mismatch on every point carrying it.
    for frame in (sup, reg):
        for a in frame.assigned.str.split("|").explode():
            parts = a.split(" -> ")
            assert a == CANNOT or (len(parts) == 2 and set(parts) <= LEGEND), \
                f"non-canonical label on file: {a!r}"
    # One location, one set: a point in both would be read by four people
    # under two batch ids and counted twice by the report.
    assert not set(sup["id"]) & set(reg["id"]), "expert and reader sets overlap"
    sup = sup.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    reg = reg.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return sup, reg


def batch_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Only what the app may carry. The stratum, the scores and the source stay
    in the sidecar: each one tells the reader something before the call."""
    out = frame[["id", "lon", "lat", "assigned", "assigned_note"]].copy()
    out["channel"] = "audit"
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readings", type=Path, required=True,
                    help="the sheet export holding round one's e1/e2 readings "
                         "(label_rounds.fetch_sheet); a file, so the draw is "
                         "reproducible")
    ap.add_argument("--posteriors", type=Path, default=POSTERIORS)
    ap.add_argument("--super-n", type=int, default=300)
    ap.add_argument("--reader-n", type=int, default=200)
    ap.add_argument("--floor", type=int, default=15,
                    help="reader set: minimum points per coarse transition")
    ap.add_argument("--terms", nargs="+", default=["entropy", "mutual_info"],
                    choices=["entropy", "mutual_info", "1-p_assigned"])
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--campaign", default="recover-habloss")
    ap.add_argument("--outdir", type=Path, default=blb.BATCH_DIR)
    ap.add_argument("--sidecar", type=Path,
                    help="where the draw is written (default "
                         "data/label_audit/audit_sets_<date>.csv); never overwritten")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing aud_* batches (moves points under "
                         "anyone already reading them)")
    args = ap.parse_args()

    sup, reg = build(args)
    print(f"expert set: {len(sup)}  ("
          + ", ".join(f"{k} {v}" for k, v in
                      sup.groupby(["stratum", "pool"]).size().items()) + ")")
    print(f"reader set: {len(reg)}")
    print(reg.stratum_t.value_counts().to_string())
    if args.dry_run:
        return

    existing = sorted(args.outdir.glob("aud_*.json"))
    if existing and not args.force:
        raise SystemExit("audit batches already exist ("
                         + ", ".join(p.name for p in existing)
                         + "); a re-draw moves points under the readers. "
                           "--force if that is really wanted.")
    out = args.sidecar or AUDIT_DIR / f"audit_sets_{date.today():%Y%m%d}.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; not overwriting (CIFS)")
    manifest = []
    for name, frame, experts in (("sup", sup, SUPER_EXPERTS),
                                 ("reg", reg, READER_EXPERTS)):
        entries = blb.write_batches(
            batch_frame(frame), campaign=args.campaign, channel="audit",
            size=args.batch_size, prefix=f"aud_{name}", outdir=args.outdir,
            instructions=INSTRUCTIONS[name], feedback="immediate",
            stage="audit", experts=experts, double_frac=1.0, audit=True)
        for i, e in enumerate(entries):
            frame.loc[i * args.batch_size:(i + 1) * args.batch_size - 1,
                      "batch_id"] = e["batch_id"]
        manifest += entries
    blb.write_manifest(manifest, args.outdir, merge=True)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([sup, reg], ignore_index=True).to_csv(out, index=False)
    print(f"  {out}")


if __name__ == "__main__":
    main()
