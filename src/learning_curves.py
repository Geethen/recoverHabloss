"""Per-class learning curves for the deployed model, at both reads.

The question this answers is the one `CLAUDE.md` and `AUTORESEARCH.md` both end
on: **is the bottleneck data or modelling, and for which class?** Eleven
iterations of architecture search moved change-F1 by less than a seed spread, and
the standing explanation is label noise on the Cropland/Nature boundary. A
learning curve tests that explanation directly and per class, because the two
regimes look different:

    OOF still climbing at 100%          -> more labelled plots would help
    OOF flat, train score far above it  -> variance; more plots or more
                                           regularisation would help
    OOF flat, train score ALSO near it  -> the model has fitted everything the
                                           features support; the ceiling is the
                                           label/feature pair, not the network

The third is the shape that says "stop modelling" -- and it is a per-class
statement, not a global one, which is why a single change-F1 curve cannot
substitute for nine.

Method
------
Exactly the deployed recipe -- `s2off_centre_m3s3_bf`, the 78-column detail
tower, scored under the **gate-off read** (`S2_MASK = 0`), which is the read the
shipped map uses (`infer_s2.probs_aef_only_matrix`). Only the training-set size
varies.

Sizes are **nested** (the 10% draw is a subset of the 20% draw) and **stratified
on the fine transition**, so a curve's shape is a data-quantity effect and not a
resampling artefact. Every class keeps at least `MIN_PER_CLASS` training rows so
the fine head always emits all nine classes; the honest x-axis for a per-class
curve is therefore that class's own training count, which is recorded per row.

The test side never shrinks: at every size the model is scored out-of-fold over
**all** plots on the same spatially blocked folds, so the y-axis is comparable
across the whole curve. The train-side score is taken on the subsample the model
was fitted on, which is what makes the generalisation gap readable.

Class alignment is by NAME, not position
----------------------------------------
At the small end of the curve a fold can train on very few rows of a rare
transition, and `HierarchicalSoftmaxNN.fine_classes_` is `sorted(set(y))` -- a
fold-local list. Assigning its probability block positionally into a global
array would silently permute classes exactly where the curve is most delicate.
Every block here is placed through a name lookup, and a fold that somehow loses a
class leaves that column at zero rather than shifting its neighbours.

Run
---
    python src/learning_curves.py --seeds 5          # ~6 min on one GPU
    python src/learning_curves.py --plot-only        # re-draw from the CSV
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model_zoo import HierarchicalSoftmaxNN, is_change_label
from project_paths import project_data_dir
from twotower_lab import (AEF_MASK, S2_MASK, load_context, s2_subset_columns)
from twotower_metrics import prf

#: The deployed recipe (CLAUDE.md). Kept as one dict so this file and
#: `optimise_s2off.DEPLOYED` cannot drift into scoring different models.
DEPLOYED = dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
                mask_column=S2_MASK, aef_mask_column=AEF_MASK)
SUBSET = "centre_m3s3_bf"

#: Fractions of each training fold. Dense at the low end, where the curve bends.
SIZES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.55, 0.70, 0.85, 1.0)
#: Floor so the fine head always carries all nine classes (the rarest,
#: Artificial -> Cropland, has 46 plots and would otherwise vanish below ~10%).
MIN_PER_CLASS = 2


# ---------------------------------------------------------------------------
def nested_pools(target_tr: np.ndarray, tr_idx: np.ndarray, rng) -> dict:
    """Per-class shuffled training positions, drawn once so sizes nest."""
    return {c: tr_idx[rng.permutation(np.flatnonzero(target_tr == c))]
            for c in np.unique(target_tr)}


def take(pools: dict, frac: float) -> np.ndarray:
    """The first ``frac`` of every class pool -- stratified and nested."""
    out = []
    for pool in pools.values():
        k = min(len(pool), max(MIN_PER_CLASS, int(round(frac * len(pool)))))
        out.append(pool[:k])
    return np.sort(np.concatenate(out))


def place(block: np.ndarray, local: list, classes: list) -> np.ndarray:
    """Fold-local probability block -> global class order, matched by name."""
    out = np.zeros((len(block), len(classes)))
    out[:, [classes.index(c) for c in local]] = block
    return out


# ---------------------------------------------------------------------------
def curve_rows(view, cols, kwargs, seeds: list) -> list[dict]:
    """One row per (seed, size, level, class, split)."""
    fine_classes = sorted(set(view.truth_fine))
    merged_classes = list(view.merged_classes)
    truth_fine = np.asarray(view.truth_fine)
    truth_merged = np.asarray(view.truth_merged)
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        pools = [nested_pools(view.target.iloc[tr].to_numpy(), tr, rng)
                 for tr, _ in view.folds]
        for frac in SIZES:
            t0 = time.time()
            oof_f = np.zeros((len(truth_fine), len(fine_classes)))
            oof_m = np.zeros((len(truth_fine), len(merged_classes)))
            tr_hits_f, tr_hits_m, n_train = [], [], []
            for fold, (tr, te) in enumerate(view.folds):
                sub = take(pools[fold], frac)
                n_train.append(len(sub))
                model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(view.frame.iloc[sub],
                              view.target.iloc[sub].to_numpy())
                    # Both reads are taken with the detail gate forced off --
                    # the deployed serving configuration.
                    te_frame = view.frame.iloc[te].copy()
                    te_frame[S2_MASK] = 0.0
                    pf, pm = model._probs(te_frame)
                    tr_frame = view.frame.iloc[sub].copy()
                    tr_frame[S2_MASK] = 0.0
                    pf_tr, pm_tr = model._probs(tr_frame)
                local_f = list(model.fine_classes_)
                local_m = list(model.merged_classes_)
                oof_f[te] = place(pf, local_f, fine_classes)
                oof_m[te] = place(pm, local_m, merged_classes)
                # The train side is scored per fold and pooled as predictions,
                # because the subsamples of different folds overlap.
                tr_hits_f.append((view.target.iloc[sub].to_numpy(),
                                  np.array(local_f, dtype=object)[pf_tr.argmax(1)]))
                tr_hits_m.append((truth_merged[sub],
                                  np.array(local_m, dtype=object)[pm_tr.argmax(1)]))

            pred_f = np.array(fine_classes, dtype=object)[oof_f.argmax(1)]
            pred_m = np.array(merged_classes, dtype=object)[oof_m.argmax(1)]
            tr_truth_f = np.concatenate([a for a, _ in tr_hits_f])
            tr_pred_f = np.concatenate([b for _, b in tr_hits_f])
            tr_truth_m = np.concatenate([a for a, _ in tr_hits_m])
            tr_pred_m = np.concatenate([b for _, b in tr_hits_m])

            common = dict(seed=seed, frac=frac,
                          n_train=int(np.mean(n_train)))
            oof_change_f1 = np.nan
            for level, classes, (tt, tp), (ot, op) in (
                ("coarse3", fine_classes, (tr_truth_f, tr_pred_f),
                 (truth_fine, pred_f)),
                ("merged2", merged_classes, (tr_truth_m, tr_pred_m),
                 (truth_merged, pred_m)),
            ):
                for split, t, p in (("train", tt, tp), ("oof", ot, op)):
                    f1s = []
                    for c in classes:
                        prec, rec, f1 = prf(t == c, p == c)
                        f1s.append(f1)
                        rows.append(dict(**common, level=level, split=split,
                                         cls=str(c), precision=prec, recall=rec,
                                         f1=f1, support=int((t == c).sum()),
                                         n_train_cls=int((tt == c).sum())
                                         / len(view.folds)))
                    # The two aggregate reads, carried as pseudo-classes so the
                    # whole curve lives in one tidy frame.
                    rows.append(dict(**common, level=level, split=split,
                                     cls="MACRO", precision=np.nan, recall=np.nan,
                                     f1=float(np.mean(f1s)), support=len(t),
                                     n_train_cls=np.nan))
                    cprec, crec, cf1 = prf(
                        np.array([is_change_label(x) for x in t]),
                        np.array([is_change_label(x) for x in p]))
                    rows.append(dict(**common, level=level, split=split,
                                     cls="CHANGE", precision=cprec, recall=crec,
                                     f1=cf1, support=len(t), n_train_cls=np.nan))
                    if level == "merged2" and split == "oof":
                        oof_change_f1 = cf1
            print(f"  seed {seed} frac {frac:<5} n_train~{int(np.mean(n_train)):5d} "
                  f"oof change-F1 {oof_change_f1:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=project_data_dir("analysis_results"))
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    csv = args.out / "learning_curves.csv"

    if not args.plot_only:
        ctx = load_context()
        view = ctx.view("full")
        s2 = s2_subset_columns(ctx.s2_stat_cols, SUBSET)
        kwargs = dict(DEPLOYED, aef_columns=ctx.aef_cols, tess_columns=s2)
        cols = ctx.aef_cols + s2
        print(f"{len(view.target):,} plots | aef={len(ctx.aef_cols)} "
              f"s2={len(s2)} ({SUBSET}) | {len(SIZES)} sizes x {args.seeds} seeds "
              f"| GATE-OFF read", flush=True)
        rows = curve_rows(view, cols, kwargs, list(range(args.seeds)))
        pd.DataFrame(rows).to_csv(csv, index=False)
        print(f"\n-> {csv}")

    from plot_learning_curves import draw
    draw(csv, args.out)


if __name__ == "__main__":
    main()
