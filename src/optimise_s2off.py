"""Price every decision in the deployed `s2off_deploy` recipe, in time.

The model that is mapped (`infer_s2.fit_models["s2off_deploy"]`) is
`mc_s2_drop0.7` trained with both towers and **served with the detail gate off**.
Its accuracy has been argued about for eleven iterations; its *cost* never has.
This asks the other question: for each decision the recipe inherited, is there a
cheaper setting that does not move the numbers?

The decisions, and which clock each one is on
---------------------------------------------
Serving cost and training cost are different budgets and most decisions only
touch one of them, because Sentinel-2 is gate-off privileged information:

    decision                     training   serving   rung prefix
    204 Sentinel-2 columns          yes        no        S_
    192 AlphaEarth columns          yes       YES        A_
    tower_dim 256                   yes       YES        C_
    epochs 30                       yes        no        E_
    modality_dropout 0.5            yes        no        M_
    5-seed ensemble                  x5        x5        (see --seeds phase)

Only the AlphaEarth block and the tower width are on the serving clock, because
the detail tower is never run (`probs_aef_only`). Everything Sentinel-2 is paid
once, at training -- but "once" includes the patch extraction and feature build
upstream of it, which is the most expensive stage in the whole project, so a
column family that earns nothing is worth deleting even though the map never
sees it.

Method
------
Every rung is the deployed recipe with **one** decision changed, scored
out-of-fold on the same spatially blocked folds under the **gate-off read** --
the deployed read, and the only one that governs the shipped map. 15 seeds, the
bar `experiment_s2off_training.py` set after two verdicts on this model reversed
between 3 and 5 and between 5 and 15 seeds. Fit seconds are recorded per rung so
the accuracy delta can be read against what it costs.

The reference is not `A_aef_flat`; it is `R0_deployed`. A rung that ties it and
runs cheaper is a win even if both sit on top of an AlphaEarth-only baseline
that is itself inside noise -- that comparison is S16's, already answered, and
is reproduced here as `S_none` so the two readings appear in one table.

What it found (15 seeds; seed spread on change-F1 is ~0.004)
------------------------------------------------------------
Cuttable, at no measurable cost:

    S2 columns 204 -> 15   every subset ties within +/-0.0005. Built fraction
                           alone is the one to keep: it is the only subset that
                           *improves* both built-up metrics (art_stable_recall
                           0.6420 -> 0.6539, art->veg 0.1921 -> 0.1814) while
                           tying change-F1 and macro-F1. Shipped as
                           `infer_s2.fit_models["s2off_slim"]`.

Not cuttable, and the ladder is the argument:

    AlphaEarth diff block  -0.0479 change-F1. It is algebraically a linear
                           function of the 2018 and 2024 blocks and a first
                           linear layer could synthesise it -- and it still
                           costs ten seed-spreads to remove, in the two-tower
                           (-0.0479) and flat (-0.0405) alike.
    epochs 30 -> 20/15/10  -0.0281 / -0.0444 / -0.0477, and art->veg blows out
                           to 0.36 at 10. The 30-epoch ceiling is a noise
                           argument, not a budget one; below it the model is
                           simply undertrained.
    tower_dim 256 -> 128   -0.0021 change-F1 but -0.0118 coarse3, and it buys
                           **6%** of the tower's runtime: the hard-coded
                           1024/512 hidden widths in `_TwoTowerTrunk.tower` are
                           the FLOPs, not `tower_dim`. Wrong knob. (Halving
                           those widths is 2.1x and is untested.)

Not a cost decision at all, but free accuracy: `modality_dropout` 0.5 -> 0.9 is
+0.0025 change-F1 on the full block (S16). It does not compose with the slim
block -- at 15 columns md=0.9 gives back the built-up gain that is the reason to
keep those columns -- so `s2off_slim` stays at 0.5.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model_zoo import HierarchicalSoftmaxNN
from project_paths import project_data_dir
from twotower_lab import (AEF_MASK, BASE, S2_MASK, S2_SUBSETS, load_context,
                          s2_families, s2_subset_columns, score_probs)

DEPLOYED = dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
                mask_column=S2_MASK, aef_mask_column=AEF_MASK)


# ---------------------------------------------------------------------------
def rungs(ctx) -> list[dict]:
    aef = list(ctx.aef_cols)
    s2 = list(ctx.s2_stat_cols)
    fams = s2_families(s2)
    # The ladder's own coarse groupings, alongside the named subsets that
    # `S2_SUBSETS` defines for reporting. Both are read from the same family
    # split, so a rung here and a mapped subset cannot mean different columns.
    means = fams["m3"] + fams["m9"] + fams["m25"]
    stds = fams["s3"] + fams["s9"] + fams["s25"]
    aef_nodiff = [c for c in aef if not c.endswith("_diff")]

    def tt(cols_aef, cols_s2, **over):
        kw = dict(DEPLOYED, aef_columns=cols_aef, tess_columns=cols_s2)
        kw.update(over)
        return dict(cols=cols_aef + cols_s2, kwargs=kw, gate="off")

    def flat(cols):
        return dict(cols=cols, kwargs=dict(BASE), gate="none")

    out = [
        dict(key="R0_deployed", note="the shipped recipe", **tt(aef, s2)),

        # -- Sentinel-2 block: training-only cost, upstream pipeline attached ---
        dict(key="S_none", note="no Sentinel-2 at all (flat AlphaEarth MLP)",
             **flat(aef)),
        dict(key="S_bf", note="built fraction only (15 cols)",
             **tt(aef, fams["bf"])),
        dict(key="S_c", note="centre reflectance only (21)", **tt(aef, fams["c"])),
        dict(key="S_c_bf", note="centre + built fraction (36)",
             **tt(aef, fams["c"] + fams["bf"])),
        dict(key="S_notexture", note="drop std/contrast/gradient (99)",
             **tt(aef, fams["c"] + means + fams["bf"])),
        dict(key="S_texture", note="texture families only (105)",
             **tt(aef, stds + fams["lc"] + fams["g"])),
        dict(key="S_2024", note="single date, no 2018 and no diff (68)",
             **tt(aef, [c for c in s2 if c.endswith("_2024")])),
        dict(key="S_nodiff", note="both dates, drop the diff block (136)",
             **tt(aef, [c for c in s2 if not c.endswith("_diff")])),
        dict(key="S_no25", note="drop the 25 px scale (162)",
             **tt(aef, [c for c in s2
                        if not c.startswith(("S2m25_", "S2s25_", "S2bf25"))])),

        # -- AlphaEarth block: this one IS on the serving clock ----------------
        dict(key="A_nodiff", note="AlphaEarth without the diff block (128)",
             **tt(aef_nodiff, s2)),
        dict(key="A_nodiff_flat", note="S_none without the diff block (128)",
             **flat(aef_nodiff)),

        # -- capacity: serving clock ------------------------------------------
        dict(key="C_dim128", note="tower_dim 128", **tt(aef, s2, tower_dim=128)),
        dict(key="C_dim64", note="tower_dim 64", **tt(aef, s2, tower_dim=64)),

        # -- schedule: training clock -----------------------------------------
        dict(key="E_ep20", note="20 epochs", **tt(aef, s2, epochs=20)),
        dict(key="E_ep15", note="15 epochs", **tt(aef, s2, epochs=15)),
        dict(key="E_ep10", note="10 epochs", **tt(aef, s2, epochs=10)),

        # -- the free accuracy knob S16 found, carried here for one table ------
        dict(key="M_md0.9", note="modality_dropout 0.9 (matches serving)",
             **tt(aef, s2, modality_dropout=0.9)),
    ]

    # The named subsets that `compare_s2_subsets.py` maps. Plot metrics cannot
    # choose between them -- that is S18's whole point, they all tie -- but a
    # subset picked on map detail still has to be shown not to *regress* on
    # plots before it is reported, and that is what these rungs are for.
    out += [dict(key=f"N_{name}",
                 note=f"named subset ({len(s2_subset_columns(s2, name))})",
                 **tt(aef, s2_subset_columns(s2, name)))
            for name in S2_SUBSETS if name != "full"]
    return out


# ---------------------------------------------------------------------------
def _merge_csv(new: pd.DataFrame, path: Path, keys: list[str]) -> None:
    """Write ``new`` into ``path``, replacing only the rows it re-computes.

    ``--only`` re-runs a handful of rungs, and a plain ``to_csv`` would silently
    delete every rung it did not run -- which is how a fifteen-rung ladder became
    a five-rung one after a follow-up question. Rows are keyed so a re-run
    updates its own rungs and leaves the rest of the table standing.
    """
    if path.exists():
        old = pd.read_csv(path)
        if set(keys) <= set(old.columns):
            old = old.merge(new[keys].drop_duplicates(), on=keys, how="left",
                            indicator=True)
            old = old[old["_merge"] == "left_only"].drop(columns="_merge")
            new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def gate_off_cv(view, rung: dict, seeds: list[int]) -> list[dict]:
    """Out-of-fold metrics under the deployed gate-off read, plus fit seconds."""
    classes = view.merged_classes
    zero = rung["gate"] == "off"
    rows = []
    for seed in seeds:
        probs = np.zeros((len(view.target), len(classes)))
        fine = np.empty(len(view.target), dtype=object)
        fit_s = 0.0
        for tr, te in view.folds:
            model = HierarchicalSoftmaxNN(rung["cols"], seed=seed, **rung["kwargs"])
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
            fit_s += time.time() - t0
            te_frame = view.frame.iloc[te]
            if zero:
                te_frame = te_frame.copy()
                te_frame[S2_MASK] = 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p_fine, p_merged = model._probs(te_frame)
            block = np.zeros((len(te), len(classes)))
            block[:, [classes.index(c) for c in model.merged_classes_]] = p_merged
            probs[te] = block
            fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
        row = score_probs(view, probs, fine)
        row.update(seed=seed, key=rung["key"], fit_s=fit_s / len(view.folds))
        rows.append(row)
    return rows


def seed_sizing(view, rung: dict, n_seeds: int) -> list[dict]:
    """Exact ensemble metrics at every size, from one pass of cached OOF posteriors.

    The ensemble is the serving cost now -- one member is one pass over every
    pixel -- so its size is a cost decision and deserves a number rather than the
    inherited 5. Members differ only in torch seed, so averaging cached per-seed
    OOF posteriors *is* the ensemble's out-of-fold posterior; nothing needs
    refitting per size. Sizes are scored over disjoint groups of members
    (0-2, 3-5, ... at k=3) so a size is never scored on a subset of the members
    that produced the size above it.
    """
    classes = view.merged_classes
    cache, fine_classes = [], None
    for seed in range(n_seeds):
        probs = np.zeros((len(view.target), len(classes)))
        fine = None
        for tr, te in view.folds:
            model = HierarchicalSoftmaxNN(rung["cols"], seed=seed, **rung["kwargs"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
                te_frame = view.frame.iloc[te].copy()
                te_frame[S2_MASK] = 0.0
                p_fine, p_merged = model._probs(te_frame)
            if fine is None:
                fine = np.zeros((len(view.target), p_fine.shape[1]))
                fine_classes = list(model.fine_classes_)
            # `fine_classes_` is `sorted(set(y))` and therefore fold-LOCAL. The
            # merged block below is placed by name; this one is placed
            # positionally, so a fold that trains on no rows of a rare
            # transition would shift every column to its right and average
            # posteriors for different classes together. Every fold carries all
            # nine transitions at MIN_COUNT=20, so this is a guard rather than a
            # live fault -- but it is silent if it ever stops holding.
            if list(model.fine_classes_) != fine_classes:
                raise SystemExit("folds disagree on fine classes; the positional "
                                 "fine block would permute them")
            probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
            fine[te] = p_fine
        cache.append((probs, fine))

    rows = []
    for k in sorted({k for k in (1, 2, 3, 5, n_seeds) if k <= n_seeds}):
        for start in range(0, n_seeds - k + 1, k):
            members = range(start, start + k)
            pm = np.mean([cache[i][0] for i in members], 0)
            pf = np.mean([cache[i][1] for i in members], 0)
            row = score_probs(view, pm,
                              np.array(fine_classes, dtype=object)[pf.argmax(1)])
            row.update(key=rung["key"], k=k, members=str(tuple(members)))
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["ladder", "seeds"], default="ladder")
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--only", default="", help="comma-separated rung keys")
    ap.add_argument("--out", type=Path, default=project_data_dir("analysis_results"))
    args = ap.parse_args()

    ctx = load_context()
    view = ctx.view("full")
    ladder = rungs(ctx)
    if args.only:
        want = set(args.only.split(","))
        ladder = [r for r in ladder if r["key"] in want]
    seeds = list(range(args.n_seeds))
    print(f"{len(view.target):,} plots | aef={len(ctx.aef_cols)} "
          f"s2={len(ctx.s2_stat_cols)} | {len(ladder)} rungs | {args.n_seeds} seeds "
          f"| GATE-OFF read", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    if args.phase == "seeds":
        rows = []
        for rung in ladder:
            got = seed_sizing(view, rung, args.n_seeds)
            rows.extend(got)
            d = pd.DataFrame(got)
            print(f"  {rung['key']}", flush=True)
            print(d.groupby("k").agg(
                n=("change_f1", "size"), change_f1=("change_f1", "mean"),
                spread=("change_f1", "std"), macro_f1=("macro_f1", "mean"),
                coarse3=("fine_change_f1", "mean"),
                artStab=("art_stable_recall", "mean")).round(4).to_string(),
                flush=True)
            _merge_csv(pd.DataFrame(rows), args.out / "s2off_seed_sizing.csv",
                       ["key", "k", "members"])
        print(f"\n-> {args.out / 's2off_seed_sizing.csv'}")
        return

    rows = []
    for rung in ladder:
        t0 = time.time()
        got = gate_off_cv(view, rung, seeds)
        for r in got:
            r.update(note=rung["note"], n_cols=len(rung["cols"]),
                     n_aef=len(rung["kwargs"].get("aef_columns", rung["cols"])),
                     n_s2=len(rung["kwargs"].get("tess_columns", [])))
        rows.extend(got)
        d = pd.DataFrame(got)
        print(f"  {rung['key']:16s} change_f1={d.change_f1.mean():.4f}"
              f"±{d.change_f1.std():.4f}  macro={d.macro_f1.mean():.4f}  "
              f"coarse3={d.fine_change_f1.mean():.4f}  "
              f"asVeg={d.art_stable_as_veg.mean():.4f}  "
              f"fit={d.fit_s.mean():.2f}s  ({time.time() - t0:.0f}s)", flush=True)
        _merge_csv(pd.DataFrame(rows), args.out / "s2off_cost_ladder.csv",
                   ["key", "seed"])
    print(f"\n-> {args.out / 's2off_cost_ladder.csv'}")


if __name__ == "__main__":
    main()
