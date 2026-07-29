"""Confident-learning noise cleaning for the hierarchical net -- the one lever
aimed at the actual ceiling (Cropland/Nature interpreter noise), not at capacity.

Every model/optimisation/layer knob saturates within the label-noise floor
(experiment_hier_improve.py, hier_variants archtypes). The remaining hypothesis
is that a subset of *training* plots carry a wrong label -- especially a wrong
change flag on the deployed Veg/Artificial boundary -- and that finding and
removing (or relabelling) them sharpens the decision boundary.

Protocol (nested, honest): for each outer spatially-blocked fold, inner blocked
CV over the *training* plots gives cross-fitted OOF probabilities; suspected
label errors are detected from those, the training set is cleaned, a final model
is fit on the cleaned training set, and the **untouched** outer test fold is
scored on its **original** labels. Cleaning never touches the test fold, so a
gain is a real generalisation gain, not the artefact of dropping hard test cases.

Detectors:
* **self-confidence** -- prune the training plots whose OOF probability of their
  own label is in the bottom q quantile.
* **confident-joint** (Northcutt et al. 2021) -- prune plots the model confidently
  assigns to a *different* class (off-diagonal prob above that class's mean
  self-confidence). Self-calibrating; no quantile to pick.
* **relabel** -- confident-joint detection, but overwrite the label with the
  model's prediction instead of dropping the plot (coarse3 only, where the
  relabel target is well defined).

Cleaning is applied at the merged2 level (deploy-relevant) and, as a contrast, at
the coarse3 level (targets the Cropland/Nature noise, which merged2 already hides
-- so it should move fine-F1 but not merged-F1). Writes a table to
analysis_results/.
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


def aligned_oof(frame, columns, y, groups, idx, global_fine, global_merged, n_inner):
    """Inner blocked-CV OOF fine/merged probs for rows ``idx``, globally aligned."""
    sub, suby, subg = frame.iloc[idx], y.iloc[idx], groups.iloc[idx]
    pf = np.zeros((len(idx), len(global_fine)))
    pm = np.zeros((len(idx), len(global_merged)))
    fmap = {c: i for i, c in enumerate(global_fine)}
    mmap = {c: i for i, c in enumerate(global_merged)}
    for itr, ite in make_splitter("blocked", n_inner).split(sub[columns], suby, subg):
        model = HierarchicalSoftmaxNN(columns, **BASE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(sub.iloc[itr], suby.iloc[itr].to_numpy())
            f, m = model._probs(sub.iloc[ite])
        for j, c in enumerate(model.fine_classes_):
            pf[ite, fmap[c]] = f[:, j]
        for j, c in enumerate(model.merged_classes_):
            pm[ite, mmap[c]] = m[:, j]
    return pf, pm


def confident_joint(probs, y_idx):
    """Boolean mask of plots the model confidently assigns off their given label."""
    n, k = probs.shape
    thresh = np.ones(k)
    for j in range(k):
        mask = y_idx == j
        if mask.any():
            thresh[j] = probs[mask, j].mean()
    pred = probs.argmax(1)
    conf = probs[np.arange(n), pred]
    return (pred != y_idx) & (conf >= thresh[pred])


def self_conf_low(probs, y_idx, q):
    """Bottom-q by the OOF probability assigned to the plot's own label."""
    sc = probs[np.arange(len(y_idx)), y_idx]
    return sc <= np.quantile(sc, q)


METHODS = [
    ("baseline", None),
    ("prune_selfconf_q10", ("prune", "merged", "selfconf", 0.10)),
    ("prune_selfconf_q20", ("prune", "merged", "selfconf", 0.20)),
    ("prune_confjoint_merged", ("prune", "merged", "confjoint", None)),
    ("prune_confjoint_coarse3", ("prune", "fine", "confjoint", None)),
    ("relabel_confjoint_coarse3", ("relabel", "fine", "confjoint", None)),
]


def detect(spec, pf, pm, fine_idx, merged_idx):
    _, level, rule, q = spec
    probs, y_idx = (pf, fine_idx) if level == "fine" else (pm, merged_idx)
    if rule == "confjoint":
        return confident_joint(probs, y_idx)
    return self_conf_low(probs, y_idx, q)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-inner", type=int, default=4)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="2yr")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])

    global_fine = sorted(set(truth_fine))
    global_merged = sorted(set(truth_merged))
    fmap = {c: i for i, c in enumerate(global_fine)}
    mmap = {c: i for i, c in enumerate(global_merged)}
    fine_idx_all = np.array([fmap[c] for c in truth_fine])
    merged_idx_all = np.array([mmap[c] for c in truth_merged])
    fine_arr = np.array(global_fine, dtype=object)
    print(f"{len(frame):,} plots | {len(columns)} features | "
          f"{len(global_fine)} coarse3 / {len(global_merged)} merged2 | base={BASE}",
          flush=True)

    oof = {name: np.empty(len(target), dtype=object) for name, _ in METHODS}
    oof_fine = {name: np.empty(len(target), dtype=object) for name, _ in METHODS}
    n_touched = {name: 0 for name, _ in METHODS}

    outer = make_splitter("blocked", args.n_splits)
    for fold, (tr, te) in enumerate(outer.split(frame[columns], target, groups), 1):
        # Cross-fitted OOF probabilities for the training plots only.
        pf, pm = aligned_oof(frame, columns, target, groups, tr,
                             global_fine, global_merged, args.n_inner)
        fine_idx, merged_idx = fine_idx_all[tr], merged_idx_all[tr]
        for name, spec in METHODS:
            y_tr = target.iloc[tr]
            keep = np.ones(len(tr), dtype=bool)
            if spec is not None:
                action = spec[0]
                suspect = detect(spec, pf, pm, fine_idx, merged_idx)
                n_touched[name] += int(suspect.sum())
                if action == "prune":
                    keep = ~suspect
                else:  # relabel (fine level): overwrite with the model's argmax
                    y_tr = y_tr.copy()
                    y_tr.iloc[suspect] = fine_arr[pf[suspect].argmax(1)]
            fit_frame = frame.iloc[tr[keep]]
            fit_y = y_tr.iloc[keep].to_numpy() if hasattr(y_tr, "iloc") else y_tr[keep]
            model = HierarchicalSoftmaxNN(columns, **BASE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(fit_frame, fit_y)
                oof[name][te] = model.predict_merged(frame.iloc[te])
                oof_fine[name][te] = model.predict(frame.iloc[te])
        print(f"  fold {fold}/{args.n_splits} done", flush=True)

    rows = []
    for name, _ in METHODS:
        m = scores(truth_merged, oof[name], is_change_label)
        f = scores(truth_fine, oof_fine[name], is_change_label)
        rows.append({
            "method": name,
            "merged_change_f1": round(m["change_f1"], 4),
            "merged_change_recall": round(m["change_recall"], 4),
            "merged_change_precision": round(m["change_precision"], 4),
            "merged_bal_acc": round(m["balanced_accuracy"], 4),
            "fine_change_f1": round(f["change_f1"], 4),
            "fine_bal_acc": round(f["balanced_accuracy"], 4),
            "avg_plots_touched": round(n_touched[name] / args.n_splits, 1),
        })
    board = pd.DataFrame(rows)
    base_m = board.loc[board.method == "baseline", "merged_change_f1"].iloc[0]
    base_f = board.loc[board.method == "baseline", "fine_change_f1"].iloc[0]
    board["d_merged"] = (board["merged_change_f1"] - base_m).round(4)
    board["d_fine"] = (board["fine_change_f1"] - base_f).round(4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"hier_clean_{args.tag}.csv"
    board.to_csv(out, index=False)
    print("\n" + board[["method", "merged_change_f1", "d_merged", "merged_bal_acc",
                        "fine_change_f1", "d_fine", "avg_plots_touched"]]
          .to_string(index=False))
    print(f"\nbaseline merged={base_m:.4f} fine={base_f:.4f} | ~{len(frame)*(args.n_splits-1)//args.n_splits} train plots/fold")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
