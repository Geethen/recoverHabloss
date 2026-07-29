"""Ablation: best hier NN -> the symmetric two-tower fusion options.

Walks the ladder from the deployed AlphaEarth-only network to the two-tower
AlphaEarth+Tessera fusions, all scored on one footing -- the same model
(``HierarchicalSoftmaxNN``, the wide/focal/30-epoch recipe that wins the hier-NN
sweep), the same spatially blocked CV, the same merged2 (Vegetation/Artificial)
change-F1 deploy metric with the coarse3 fine-F1 alongside, over ``--n-seeds``
torch seeds (sub-1pt gaps here are within seed noise; see memory).

The architecture under test is the *symmetric* two-tower (``model_zoo._TwoTowerTrunk``,
``aef_mask_column`` + ``fusion``): both the AlphaEarth and the Tessera tower are
independently mask-gated, so the network produces a prediction from whichever
modalities are present -- **both, AlphaEarth only, or Tessera only** -- and never
from neither. Modality dropout is applied to *both* towers (with the guard that a
row's only present modality is never dropped), and ``gated_mean`` fusion keeps the
fused representation at a constant scale however many towers fire, so a
single-modality row is not down-scaled relative to a both-present one. This
generalises the original Plan B, where AlphaEarth was a hard-wired always-on base
(``fusion='additive'``, no AlphaEarth mask) -- that config is kept in the ladder
as the reference two-tower.

Two reads
---------
**Full set (deploy metric).** All ~6.4k plots, blocked CV. AlphaEarth is present
everywhere and Tessera is sparse (~36% both-years), so this is the realistic
deployment score -- what you would actually ship. Ladder::

    aef                    best hier NN, AlphaEarth only (wide/focal)     [baseline]
    tt_additive_md0        two-tower, AlphaEarth always-on, Tessera gated, NO dropout
    tt_additive_md0.5      two-tower, AlphaEarth always-on, Tessera gated, md0.5
    tt_symmetric_md0.3     symmetric two-tower (both towers gated), gated_mean, md0.3
    tt_symmetric_md0.5     symmetric two-tower (both towers gated), gated_mean, md0.5

The full set cannot exercise a missing-AlphaEarth row (AlphaEarth covers every
plot), so the symmetric model's extra capability is invisible here -- it should
*match* the additive two-tower on the deploy metric, not beat it. The point of
the full-set read is to confirm the symmetric generalisation costs nothing where
AlphaEarth is always available.

**Covered-subset robustness (the reason the architecture changed).** Restrict to
the both-years-covered plots (~2.3k, every feature real), blocked CV, train the
symmetric two-tower once, then score its out-of-fold predictions under three eval
maskings of the *same* held-out plots::

    both       AlphaEarth + Tessera present   (the fused prediction)
    aef_only   Tessera masked off             (AlphaEarth tower alone)
    tess_only  AlphaEarth masked off          (Tessera tower alone)

This is the direct test of "supports sparse Tessera, and no data from either
side": one model degrades gracefully to whichever single modality survives. The
additive/always-on two-tower cannot represent ``tess_only`` at all (its
AlphaEarth tower has no gate), so only the symmetric model appears here.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hier_tessera import BASE, TESSERA, attach_tessera, score_seeds
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from model_zoo import (
    DEFAULT_INPUT,
    HierarchicalSoftmaxNN,
    is_change_label,
    make_splitter,
    scores,
    to_merged_label,
)
from project_paths import project_data_dir

AEF_MASK = "aef_present"


def two_tower_kwargs(aef_cols, tess_cols, *, modality_dropout, fusion,
                     aef_maskable):
    """A two-tower ``HierarchicalSoftmaxNN`` config on the best-hier-NN recipe.

    The wide/focal/30-epoch recipe is inherited from ``BASE`` (loss + epochs); the
    trunk is the two-tower one whose towers are themselves wide MLPs, so every
    ablation rung shares the winning hier-NN settings and differs only in the
    fusion structure being tested.
    """
    return dict(
        arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"],
        aef_columns=aef_cols, tess_columns=tess_cols,
        mask_column="tess_present", modality_dropout=modality_dropout,
        tower_dim=256, fusion=fusion,
        aef_mask_column=AEF_MASK if aef_maskable else None,
    )


# -- full-set deploy read ----------------------------------------------------
def run_full(frame, aef_cols, tess_cols, folds, seeds, out_dir, tag):
    """The deploy-metric ladder over all plots (AlphaEarth dense, Tessera sparse)."""
    target = target_for_legend(frame, LEGENDS["coarse3"], MIN_COUNT)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])

    cols = aef_cols + tess_cols
    configs = {
        "aef": (aef_cols, None),  # best hier NN (AlphaEarth only, BASE)
        "tt_additive_md0": (cols, two_tower_kwargs(
            aef_cols, tess_cols, modality_dropout=0.0, fusion="additive",
            aef_maskable=False)),
        "tt_additive_md0.5": (cols, two_tower_kwargs(
            aef_cols, tess_cols, modality_dropout=0.5, fusion="additive",
            aef_maskable=False)),
        "tt_symmetric_md0.3": (cols, two_tower_kwargs(
            aef_cols, tess_cols, modality_dropout=0.3, fusion="gated_mean",
            aef_maskable=True)),
        "tt_symmetric_md0.5": (cols, two_tower_kwargs(
            aef_cols, tess_cols, modality_dropout=0.5, fusion="gated_mean",
            aef_maskable=True)),
    }

    print(f"\n[full] {len(frame):,} plots | blocked CV | ladder from best hier NN "
          f"to the two-tower options", flush=True)
    rows = []
    for name, (c, kw) in configs.items():
        s = score_seeds(frame, c, target, truth_merged, truth_fine, folds, seeds,
                        model_kwargs=kw)
        rows.append({"config": name, "n_features": len(c), **s})
        mf1 = s["merged_change_f1"]
        print(f"  {name:20s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
    _report(rows, "aef", out_dir / f"hier_twotower_ablation_full_{tag}.csv",
            "FULL / deploy: best hier NN vs the two-tower fusions")


# -- covered-subset robustness read ------------------------------------------
def _predict_masked(model, frame, aef_on: bool, tess_on: bool):
    """Merged2 OOF labels with a modality forced off via its eval mask.

    Flipping ``aef_present`` / ``tess_present`` to 0 on the eval frame gates that
    tower out inside the trunk (the gate is the mask at eval), so this reads the
    same trained model under a single-modality view without refitting.
    """
    view = frame.copy()
    view[AEF_MASK] = 1.0 if aef_on else 0.0
    view["tess_present"] = view["tess_present"] * (1.0 if tess_on else 0.0)
    return model.predict_merged(view)


def run_robustness(frame, valid, aef_cols, tess_cols, n_splits, seeds, out_dir,
                   tag):
    """Train the symmetric two-tower on covered plots; score 3 eval maskings.

    Uses the both-years-covered subset (every Tessera feature real), so a
    ``tess_only`` view is a genuine Tessera-only prediction, not an imputed one.
    """
    sub = frame.loc[frame["_effboth"]].reset_index(drop=True)
    target = target_for_legend(sub, LEGENDS["coarse3"], MIN_COUNT)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    folds = list(make_splitter("blocked", n_splits).split(
        sub[aef_cols], target, sub["block_id"]))

    kw = two_tower_kwargs(aef_cols, tess_cols, modality_dropout=0.5,
                          fusion="gated_mean", aef_maskable=True)
    views = ("both", "aef_only", "tess_only")
    cols = aef_cols + tess_cols

    n_ch = int(sum(is_change_label(t) for t in truth_merged))
    print(f"\n[robustness] {len(sub):,} both-covered plots | {n_ch} change | "
          f"symmetric two-tower md0.5, one model scored under 3 eval maskings",
          flush=True)

    per_view = {v: [] for v in views}  # per-view list of per-seed change-F1
    for seed in seeds:
        oof = {v: np.empty(len(target), dtype=object) for v in views}
        for tr, te in folds:
            model = HierarchicalSoftmaxNN(cols, seed=seed, **kw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(sub.iloc[tr], target.iloc[tr].to_numpy())
                te_frame = sub.iloc[te]
                oof["both"][te] = _predict_masked(model, te_frame, True, True)
                oof["aef_only"][te] = _predict_masked(model, te_frame, True, False)
                oof["tess_only"][te] = _predict_masked(model, te_frame, False, True)
        for v in views:
            per_view[v].append(scores(truth_merged, oof[v], is_change_label)["change_f1"])

    rows = []
    for v in views:
        arr = per_view[v]
        rows.append({"eval_view": v,
                     "merged_change_f1_mean": round(float(np.mean(arr)), 4),
                     "merged_change_f1_std": round(float(np.std(arr)), 4)})
        print(f"  {v:10s} merged_f1={np.mean(arr):.4f}±{np.std(arr):.3f}", flush=True)
    out_path = out_dir / f"hier_twotower_ablation_robustness_{tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}", flush=True)


# -- shared reporting --------------------------------------------------------
def _report(rows, baseline, out_path, title):
    board = pd.DataFrame(rows)
    board["mean"] = board["merged_change_f1"].map(lambda t: t[0])
    board = board.sort_values("mean", ascending=False).reset_index(drop=True)
    base = board.loc[board.config == baseline, "mean"]
    base = float(base.iloc[0]) if len(base) else float("nan")
    board["d_vs_base"] = (board["mean"] - base).round(4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = board.copy()
    for c in ("merged_change_f1", "merged_change_recall", "merged_change_precision",
              "merged_bal_acc", "fine_change_f1"):
        flat[c + "_mean"] = flat[c].map(lambda t: t[0])
        flat[c + "_std"] = flat[c].map(lambda t: t[1])
        flat = flat.drop(columns=[c])
    flat.drop(columns=["mean"]).to_csv(out_path, index=False)
    print(f"\n=== {title} (baseline={baseline}) ===")
    for _, r in board.iterrows():
        mf1, ff1 = r["merged_change_f1"], r["fine_change_f1"]
        print(f"  {r.config:20s} n_feat={r.n_features:4d}  "
              f"merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}  d={r.d_vs_base:+.4f}  "
              f"fine_f1={ff1[0]:.4f}±{ff1[1]:.3f}")
    print(f"wrote {out_path}")


MIN_COUNT = 20


def main() -> None:
    global MIN_COUNT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tessera", type=Path,
                        default=project_data_dir("embeddings", TESSERA))
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--mode", choices=["full", "robustness", "both"],
                        default="both")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="2yr")
    args = parser.parse_args()
    MIN_COUNT = args.min_class_count

    frame, aef_cols = build_frame(args.input)
    frame, tgroups, valid = attach_tessera(frame, args.tessera)
    # AlphaEarth covers every plot in this frame; .copy() de-fragments the frame
    # that attach_tessera built up column-by-column.
    frame = frame.assign(**{AEF_MASK: 1.0}).copy()
    tess_cols = tgroups["tess_2yr"]
    seeds = list(range(args.n_seeds))

    print(f"{len(frame):,} plots | aef={len(aef_cols)} tess_2yr={len(tess_cols)} "
          f"| both-covered={int(frame['_effboth'].sum())} | seeds={seeds} "
          f"| best-hier-NN recipe={BASE}", flush=True)

    if args.mode in ("full", "both"):
        folds = list(make_splitter("blocked", args.n_splits).split(
            frame[aef_cols],
            target_for_legend(frame, LEGENDS["coarse3"], MIN_COUNT),
            frame["block_id"]))
        run_full(frame, aef_cols, tess_cols, folds, seeds, args.output_dir, args.tag)

    if args.mode in ("robustness", "both"):
        run_robustness(frame, valid, aef_cols, tess_cols, args.n_splits, seeds,
                       args.output_dir, args.tag)


if __name__ == "__main__":
    main()
