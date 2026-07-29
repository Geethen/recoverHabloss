"""AlphaEarth vs Tessera (and their fusion) for the hierarchical net.

Two questions, one script, both scored on the deployed model
(``HierarchicalSoftmaxNN``, wide/focal) and its real metric -- change-F1 at the
merged2 (Vegetation/Artificial) deploy level, with the coarse3 fine-F1 alongside
-- under the same spatially blocked CV as ``experiment_hier_change_features.py``.

Data
----
AlphaEarth (``A??_{2018,2024,diff}``) is dense: every plot has it. Tessera
(``TE???_{2018,2024}``, joined by PLOTID from
``embeddings_tessera_habloss_recover.parquet``) is dense for 2024 (~99%) but
sparse for 2018 / the change (~36% of plots have both years). A row's Tessera is
treated as *present* only when covered AND not the all-zero nodata/edge vector.
The 2024->2018 per-band difference (``TE???_diff``) is computed locally and is
present only where both years are.

Mode ``subset`` -- the honest signal check (the user's step 1)
--------------------------------------------------------------
Restrict to the both-years-covered plots (~2.3k), where every feature is *real*
(no imputation), and compare on the same points and folds:

    aef_2yr      A??_2018 + A??_2024 + A??_diff        (the hier default)
    aef_2024     A??_2024                              (single-date AlphaEarth)
    tess_2yr     TE???_2018 + TE???_2024 + TE???_diff  (full Tessera)
    tess_2024    TE???_2024                            (single-date Tessera)
    aef+tess_2yr concat of both change feature sets
    aef+tess_2024 concat of both single-date sets

If ``tess_*`` beats / complements ``aef_*`` here, Tessera carries signal the
AlphaEarth features do not. Tessera's expected clean win is the 2024-dense
single-date map, not the sparse change -- watch ``*_2024`` vs ``*_2yr``.

Mode ``fullA`` -- concat + availability mask (the user's Plan A)
---------------------------------------------------------------
All ~6.4k plots. Append the Tessera columns to the AlphaEarth default, impute a
missing Tessera value with the *train-fold* mean of its covered rows (so after
the net's internal standardisation the imputed rows sit at 0 = the column mean,
exactly as specified), and add a binary ``tess_present`` feature. BatchNorm
handles the mostly-mean columns; the trunk learns to use Tessera when the mask
is on. Compare:

    aef              A??_{2018,2024,diff}                    (baseline, full set)
    aef+tessA_2024   + TE???_2024 (dense) + tess_present     (the cheap win)
    aef+tessA_2yr    + all Tessera (2018/24/diff) + mask     (the full concat)

Mode ``twotower`` -- two-tower late fusion + modality dropout (Plan B)
---------------------------------------------------------------------
All ~6.4k plots, ``HierarchicalSoftmaxNN(arch="two_tower")``: an always-on
AlphaEarth tower (guarantees a prediction for every plot) plus a Tessera tower
gated by the ``tess_present`` mask, fused as ``rep_aef + gate * rep_tess`` so the
Tessera term is zeroed where absent. ``--modality-dropout`` randomly zeroes the
gate on present rows during training so the trunk never depends on a feature
missing for 64% of plots -- swept here (0 = the ablation). The Tessera columns
are passed RAW (NaN where absent); the model's ``_prepare`` does the mask-aware
standardise + zero-impute. This is the right structure for Plan A's failure mode.

Result (3 seeds): modality dropout is *essential* -- without it the two-tower
regresses (~-0.7pt), with it (0.5-0.7) it reaches parity with AlphaEarth-only
(~+0.1pt, within seed noise). So Plan B rescues Plan A's -2.1pt regression back
to break-even, but does NOT beat AlphaEarth on the full-set deploy metric: the
real +1.8pt gain lives on the covered subset and is averaged away by the 64% of
plots (and most change points) that have no Tessera. The bottleneck is 2018
coverage, not architecture.

Honest caveat (kept in the report): with 36% joint coverage and spatially
blocked CV the availability mask correlates with geography and can quietly
become a location proxy -- so the ``subset`` result, where availability is held
constant, is the trustworthy signal check; ``fullA``/``twotower`` are the
deployment-realistic reads. Every configuration is run over ``--n-seeds`` torch
seeds and reported as mean +/- std, because sub-1pt differences here are within
seed noise (see memory: seed-average before trusting sub-1pt wins).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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

BASE = dict(arch="wide", loss="focal", epochs=30)
TESSERA = "embeddings_tessera_habloss_recover.parquet"


# -- Tessera attachment ------------------------------------------------------
def attach_tessera(frame: pd.DataFrame, tess_path: Path):
    """Join Tessera onto the AlphaEarth frame and build its column groups.

    Returns ``(frame, groups, valid)`` where ``frame`` gains the Tessera columns
    (NaN where a year is not present) plus a float ``tess_present`` flag,
    ``groups`` maps a feature-set name to its Tessera columns, and ``valid`` is a
    boolean frame (one column per Tessera feature) marking the rows whose value
    is real -- used to drive the covered-subset filter and the fold imputation.
    """
    tess = pd.read_parquet(tess_path).drop_duplicates("PLOTID").reset_index(drop=True)
    te18 = sorted(c for c in tess.columns if c.startswith("TE") and c.endswith("_2018"))
    te24 = sorted(c for c in tess.columns if c.startswith("TE") and c.endswith("_2024"))

    # "Present" = flagged covered AND not the all-zero nodata/edge vector.
    z18 = tess[te18].abs().sum(axis=1) == 0
    z24 = tess[te24].abs().sum(axis=1) == 0
    eff18 = (tess["tess_covered_2018"] & ~z18).to_numpy()
    eff24 = (tess["tess_covered_2024"] & ~z24).to_numpy()
    tess = tess.assign(_eff18=eff18, _eff24=eff24, _effboth=eff18 & eff24)

    cols = ["PLOTID", *te18, *te24, "_eff18", "_eff24", "_effboth"]
    frame = frame.merge(tess[cols], on="PLOTID", how="left")
    for f in ("_eff18", "_eff24", "_effboth"):
        frame[f] = frame[f].fillna(False)

    tediff = [c.replace("_2018", "_diff") for c in te18]
    diff = frame[te24].to_numpy("float32") - frame[te18].to_numpy("float32")
    extra = pd.DataFrame(diff, columns=tediff, index=frame.index)
    extra["tess_present"] = frame["_effboth"].astype("float32")
    frame = pd.concat([frame, extra], axis=1)

    # Blank a year's columns on the rows where it is not present, so a missing
    # value is unambiguously NaN (the diff is present only where both years are).
    frame.loc[~frame["_eff18"], te18] = np.nan
    frame.loc[~frame["_eff24"], te24] = np.nan
    frame.loc[~frame["_effboth"], tediff] = np.nan

    groups = {
        "tess_2024": te24,
        "tess_2yr": [*te18, *te24, *tediff],
    }
    valid = pd.DataFrame(
        {c: frame[c].notna().to_numpy() for c in groups["tess_2yr"]}
    )
    return frame, groups, valid


def impute_fold(frame: pd.DataFrame, cols: list[str], valid: pd.DataFrame,
                train_idx: np.ndarray) -> pd.DataFrame:
    """Fill missing Tessera with the train-fold mean of each column's real rows.

    Imputing at the (train-only) mean means the net's per-column standardisation
    maps every imputed row to ~0 -- the ``0 = the mean`` convention in the plan,
    with no test-fold leakage.
    """
    out = frame[cols].astype("float32").copy()
    tr = np.zeros(len(frame), dtype=bool)
    tr[train_idx] = True
    for c in cols:
        v = valid[c].to_numpy()
        vals = np.asarray(out[c].to_numpy(), dtype="float32").copy()
        mu = vals[tr & v].mean() if (tr & v).any() else 0.0
        if not np.isfinite(mu):
            mu = 0.0
        vals[~v] = mu
        out[c] = vals
    return out


# -- CV ----------------------------------------------------------------------
def run_cv(frame, cols, target, folds, seed, tess_cols=None, valid=None,
           model_kwargs=None):
    """One blocked-CV pass; returns (oof_merged, oof_fine) object arrays.

    When ``tess_cols``/``valid`` are given, those columns are mean-imputed per
    fold (Plan A); otherwise the frame is used as-is (covered subset, or the
    two-tower model which does its own mask-aware imputation in ``_prepare``).
    ``model_kwargs`` overrides ``BASE`` (e.g. the two-tower config).
    """
    oof_merged = np.empty(len(target), dtype=object)
    oof_fine = np.empty(len(target), dtype=object)
    for tr, te in folds:
        f = frame
        if tess_cols is not None:
            imp = impute_fold(frame, tess_cols, valid, tr)
            f = frame.copy()
            f[tess_cols] = imp
        model = HierarchicalSoftmaxNN(cols, seed=seed, **(model_kwargs or BASE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(f.iloc[tr], target.iloc[tr].to_numpy())
            oof_merged[te] = model.predict_merged(f.iloc[te])
            oof_fine[te] = model.predict(f.iloc[te])
    return oof_merged, oof_fine


def score_seeds(frame, cols, target, truth_merged, truth_fine, folds, seeds,
                tess_cols=None, valid=None, model_kwargs=None):
    """Run every seed, return per-metric mean/std dicts over seeds."""
    mf1, ff1, mrec, mprec, mbal = [], [], [], [], []
    for s in seeds:
        oof_m, oof_f = run_cv(frame, cols, target, folds, s, tess_cols, valid,
                              model_kwargs)
        m = scores(truth_merged, oof_m, is_change_label)
        f = scores(truth_fine, oof_f, is_change_label)
        mf1.append(m["change_f1"]); ff1.append(f["change_f1"])
        mrec.append(m["change_recall"]); mprec.append(m["change_precision"])
        mbal.append(m["balanced_accuracy"])
    agg = lambda a: (round(float(np.mean(a)), 4), round(float(np.std(a)), 4))
    return {
        "merged_change_f1": agg(mf1), "merged_change_recall": agg(mrec),
        "merged_change_precision": agg(mprec), "merged_bal_acc": agg(mbal),
        "fine_change_f1": agg(ff1),
    }


def transfer_seeds(train_frame, eval_frame, cols, target, truth_merged_eval,
                   truth_fine_eval, seeds, model_kwargs=None):
    """Train on the whole covered subset, evaluate on the disjoint uncovered set.

    No CV split: one model per seed is fit on all of ``train_frame`` (the covered
    subset) and scored on ``eval_frame`` (the uncovered plots), so this is an
    out-of-domain transfer read, NOT a spatially blocked in-domain score. For the
    Tessera feature sets the uncovered plots carry no Tessera -- those columns are
    filled with the covered-train column mean before predicting, so after the
    net's standardisation they sit at 0 (neutral) and the Tessera tower/columns
    contribute nothing there. So a Tessera set's transfer score is expected to
    fall back to roughly its AlphaEarth counterpart: the point is to check whether
    the covered subset's numbers survive the domain shift, not to use Tessera
    where it is absent.
    """
    mf1, ff1, mrec, mprec, mbal = [], [], [], [], []
    for s in seeds:
        model = HierarchicalSoftmaxNN(cols, seed=s, **(model_kwargs or BASE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_frame, target.to_numpy())
            oof_m = model.predict_merged(eval_frame)
            oof_f = model.predict(eval_frame)
        m = scores(truth_merged_eval, oof_m, is_change_label)
        f = scores(truth_fine_eval, oof_f, is_change_label)
        mf1.append(m["change_f1"]); ff1.append(f["change_f1"])
        mrec.append(m["change_recall"]); mprec.append(m["change_precision"])
        mbal.append(m["balanced_accuracy"])
    agg = lambda a: (round(float(np.mean(a)), 4), round(float(np.std(a)), 4))
    return {
        "merged_change_f1": agg(mf1), "merged_change_recall": agg(mrec),
        "merged_change_precision": agg(mprec), "merged_bal_acc": agg(mbal),
        "fine_change_f1": agg(ff1),
    }


def report(rows, baseline_name, sort_key, out_path, title):
    board = pd.DataFrame(rows)
    board["mean"] = board[sort_key].map(lambda t: t[0])
    board["std"] = board[sort_key].map(lambda t: t[1])
    board = board.sort_values("mean", ascending=False).reset_index(drop=True)
    base = board.loc[board.feature_set == baseline_name, "mean"]
    base = float(base.iloc[0]) if len(base) else float("nan")
    board["d_vs_base"] = (board["mean"] - base).round(4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = board.copy()
    for c in ("merged_change_f1", "merged_change_recall", "merged_change_precision",
              "merged_bal_acc", "fine_change_f1"):
        flat[c + "_mean"] = flat[c].map(lambda t: t[0])
        flat[c + "_std"] = flat[c].map(lambda t: t[1])
        flat = flat.drop(columns=[c])
    flat.drop(columns=["mean", "std"]).to_csv(out_path, index=False)
    print(f"\n=== {title} (baseline={baseline_name}, sorted by {sort_key}) ===")
    for _, r in board.iterrows():
        mf1 = r["merged_change_f1"]; ff1 = r["fine_change_f1"]
        print(f"  {r.feature_set:15s} n_feat={r.n_features:4d}  "
              f"merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}  "
              f"d={r.d_vs_base:+.4f}  fine_f1={ff1[0]:.4f}±{ff1[1]:.3f}")
    print(f"wrote {out_path}")
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tessera", type=Path,
                        default=project_data_dir("embeddings", TESSERA))
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument(
        "--mode",
        choices=["subset", "fullA", "twotower", "both", "all"],
        default="both",
        help="'both' = subset+fullA; 'all' = +twotower (Plan B).",
    )
    parser.add_argument("--modality-dropout", type=float, nargs="+",
                        default=[0.0, 0.5],
                        help="Plan B: modality-dropout rates to sweep.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="2yr")
    args = parser.parse_args()

    frame, aef_cols = build_frame(args.input)
    frame, tgroups, valid = attach_tessera(frame, args.tessera)
    seeds = list(range(args.n_seeds))

    aef_per_year = [c for c in aef_cols if not c.endswith("_diff")]
    aef_2024 = [c for c in aef_cols if c.endswith("_2024")]
    aef_sets = {"aef_2yr": aef_cols, "aef_2024": aef_2024}

    print(f"{len(frame):,} plots | aef={len(aef_cols)} tess_2yr={len(tgroups['tess_2yr'])} "
          f"| both-covered={int(frame['_effboth'].sum())} "
          f"2024-covered={int(frame['_eff24'].sum())} | seeds={seeds} base={BASE}",
          flush=True)

    if args.mode in ("subset", "both", "all"):
        sub = frame.loc[frame["_effboth"]].reset_index(drop=True)
        sub_valid = valid.loc[frame["_effboth"].to_numpy()].reset_index(drop=True)
        target = target_for_legend(sub, LEGENDS["coarse3"], args.min_class_count)
        truth_fine = target.to_numpy()
        truth_merged = np.array([to_merged_label(t) for t in truth_fine])
        splitter = make_splitter("blocked", args.n_splits)
        folds = list(splitter.split(sub[aef_cols], target, sub["block_id"]))
        sets = {
            **aef_sets,
            "tess_2yr": tgroups["tess_2yr"],
            "tess_2024": tgroups["tess_2024"],
            "aef+tess_2yr": aef_cols + tgroups["tess_2yr"],
            "aef+tess_2024": aef_2024 + tgroups["tess_2024"],
        }
        n_ch = int(truth_merged.__ne__(None).sum()
                   and sum(is_change_label(t) for t in truth_merged))
        print(f"\n[subset] {len(sub):,} both-covered plots | {n_ch} change | "
              f"{sub['block_id'].nunique()} blocks", flush=True)
        rows = []
        for name, cols in sets.items():
            s = score_seeds(sub, cols, target, truth_merged, truth_fine, folds, seeds)
            rows.append({"feature_set": name, "n_features": len(cols), **s})
            mf1 = s["merged_change_f1"]
            print(f"  {name:15s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
        report(rows, "aef_2yr", "merged_change_f1",
               args.output_dir / f"hier_tessera_subset_{args.tag}.csv",
               "SUBSET: AlphaEarth vs Tessera on both-covered plots")

        # Transfer: train on the whole covered subset, evaluate on the disjoint
        # uncovered plots (Tessera absent there -> its columns filled with the
        # covered-train mean, i.e. neutral). Checks whether the covered subset is
        # representative / whether its scores survive the domain shift.
        unc = frame.loc[~frame["_effboth"]].reset_index(drop=True)
        unc_imp = unc.copy()
        for c in tgroups["tess_2yr"]:
            unc_imp[c] = unc_imp[c].fillna(sub[c].mean()).astype("float32")
        t_fine = target_for_legend(unc_imp, LEGENDS["coarse3"], 1).to_numpy()
        t_merged = target_for_legend(unc_imp, LEGENDS["merged2"], 1).to_numpy()
        n_ch_u = int(sum(is_change_label(t) for t in t_merged))
        print(f"\n[subset->transfer] eval on {len(unc):,} uncovered plots | "
              f"{n_ch_u} change ({n_ch_u / len(unc):.3f}) | "
              f"{unc['block_id'].nunique()} blocks", flush=True)
        trows = []
        for name, cols in sets.items():
            s = transfer_seeds(sub, unc_imp, cols, target, t_merged, t_fine, seeds)
            trows.append({"feature_set": name, "n_features": len(cols), **s})
            mf1 = s["merged_change_f1"]
            print(f"  {name:15s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
        # Plan B two-tower: trained on the (all-present) covered subset, so
        # modality dropout is the ONLY source of gate=0 in training. On the
        # uncovered plots the mask is 0 -> the trunk falls back to the AlphaEarth
        # tower. md0 never saw a gated-off Tessera and should degrade like the
        # flat concat; md0.5 did and should transfer gracefully (~= aef_2yr).
        tt_cols = aef_cols + tgroups["tess_2yr"]
        for p in (0.0, 0.5):
            kw = dict(arch="two_tower", loss="focal", epochs=30,
                      aef_columns=aef_cols, tess_columns=tgroups["tess_2yr"],
                      mask_column="tess_present", modality_dropout=p, tower_dim=256)
            s = transfer_seeds(sub, unc_imp, tt_cols, target, t_merged, t_fine,
                               seeds, model_kwargs=kw)
            name = f"two_tower_md{p:g}"
            trows.append({"feature_set": name, "n_features": len(tt_cols), **s})
            mf1 = s["merged_change_f1"]
            print(f"  {name:15s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
        report(trows, "aef_2yr", "merged_change_f1",
               args.output_dir / f"hier_tessera_subset_transfer_{args.tag}.csv",
               "SUBSET->TRANSFER: covered-trained models on uncovered plots")

    if args.mode in ("fullA", "both", "all"):
        target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
        truth_fine = target.to_numpy()
        truth_merged = np.array([to_merged_label(t) for t in truth_fine])
        splitter = make_splitter("blocked", args.n_splits)
        folds = list(splitter.split(frame[aef_cols], target, frame["block_id"]))
        tess_2024_mask = tgroups["tess_2024"] + ["tess_present"]
        tess_2yr_mask = tgroups["tess_2yr"] + ["tess_present"]
        valid_full = valid.assign(tess_present=True)  # the mask is always real
        sets = {
            "aef": (aef_cols, None),
            "aef+tessA_2024": (aef_cols + tess_2024_mask, tess_2024_mask),
            "aef+tessA_2yr": (aef_cols + tess_2yr_mask, tess_2yr_mask),
        }
        print(f"\n[fullA] {len(frame):,} plots (Plan A: concat + mask + fold-mean impute)",
              flush=True)
        rows = []
        for name, (cols, tcols) in sets.items():
            s = score_seeds(frame, cols, target, truth_merged, truth_fine, folds,
                            seeds, tess_cols=tcols, valid=valid_full)
            rows.append({"feature_set": name, "n_features": len(cols), **s})
            mf1 = s["merged_change_f1"]
            print(f"  {name:15s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
        report(rows, "aef", "merged_change_f1",
               args.output_dir / f"hier_tessera_fullA_{args.tag}.csv",
               "FULL / Plan A: does sparse Tessera help the deployed model")

    if args.mode in ("twotower", "all"):
        # Plan B: two-tower late fusion + modality dropout, full set. The Tessera
        # columns are passed RAW (NaN where absent); the model's _prepare does the
        # mask-aware standardise + zero-impute and the trunk gates the Tessera term
        # off where the mask is 0. Compared against the plain AlphaEarth net on the
        # same folds/seeds; modality_dropout is swept (0 = no dropout ablation).
        target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
        truth_fine = target.to_numpy()
        truth_merged = np.array([to_merged_label(t) for t in truth_fine])
        splitter = make_splitter("blocked", args.n_splits)
        folds = list(splitter.split(frame[aef_cols], target, frame["block_id"]))
        tess_cols = tgroups["tess_2yr"]
        cols = aef_cols + tess_cols
        print(f"\n[twotower] {len(frame):,} plots (Plan B: AlphaEarth tower always on "
              f"+ mask-gated Tessera tower + modality dropout)", flush=True)
        rows = []
        # AlphaEarth-only baseline on the same folds/seeds (arch=wide, the deploy net).
        s = score_seeds(frame, aef_cols, target, truth_merged, truth_fine, folds, seeds)
        rows.append({"feature_set": "aef", "n_features": len(aef_cols), **s})
        mf1 = s["merged_change_f1"]
        print(f"  {'aef':18s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
        for p in args.modality_dropout:
            kw = dict(arch="two_tower", loss="focal", epochs=30,
                      aef_columns=aef_cols, tess_columns=tess_cols,
                      mask_column="tess_present", modality_dropout=p, tower_dim=256)
            s = score_seeds(frame, cols, target, truth_merged, truth_fine, folds,
                            seeds, model_kwargs=kw)
            name = f"two_tower_md{p:g}"
            rows.append({"feature_set": name, "n_features": len(cols), **s})
            mf1 = s["merged_change_f1"]
            print(f"  {name:18s} merged_f1={mf1[0]:.4f}±{mf1[1]:.3f}", flush=True)
        report(rows, "aef", "merged_change_f1",
               args.output_dir / f"hier_tessera_twotower_{args.tag}.csv",
               "FULL / Plan B: two-tower late fusion + modality dropout")


if __name__ == "__main__":
    main()
