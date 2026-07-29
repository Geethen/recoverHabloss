"""Does a Squeeze-and-Excitation gate on the input help on the full feature set?

The hierarchical net is normally fed ~193 channels (2-year embeddings + 64 diff
bands); with cos + LOO added the full set is 198, and many are redundant (the
diff bands duplicate the endpoint signal; the LOO scalars correlate ~0.8 with
cosine). An SE gate (arch="wide_se", model_zoo._SEInput) re-weights the channels
per plot before the dense trunk, so this asks whether learned feature selection
buys anything at that width.

To see past the net's CUDA run-to-run wobble (~+/-0.01 on merged change-F1),
every config is averaged over several seeds. Compares, on merged2 change-F1:

    wide     / diff    deployed baseline (2-year + diff)
    wide     / full    same trunk, full feature set (diff + cos + LOO)
    wide_se  / full    SE gate on the full set          <- the test
    wide_se  / diff    SE gate on the deployed set

Writes a ranked table to analysis_results/.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hier_loo_medoid import (
    ANNUAL_INPUT,
    LOO_COLS,
    load_annual,
    attach_features,
    year_cols,
)
from experiment_merged_legend import LEGENDS, target_for_legend
from model_zoo import (
    HierarchicalSoftmaxNN,
    is_change_label,
    make_splitter,
    scores,
    to_merged_label,
)
from project_paths import project_data_dir

BASE = dict(loss="focal", epochs=30)


def feature_sets() -> dict[str, list[str]]:
    per_year = year_cols(2018) + year_cols(2024)
    diff = per_year + [f"A{i:02d}_diff" for i in range(64)]
    return {
        "diff": diff,
        "full": diff + ["cos_dist"] + LOO_COLS,
    }


CONFIGS = [
    ("wide", "diff"),
    ("wide", "full"),
    ("wide_se", "full"),
    ("wide_se", "diff"),
]


def run_config(frame, target, groups, folds, arch, cols, seeds):
    """Mean/std over seeds of merged & fine change-F1 for one (arch, feature) pair."""
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    m_f1, f_f1, m_bal = [], [], []
    for seed in seeds:
        oof_m = np.empty(len(target), dtype=object)
        oof_f = np.empty(len(target), dtype=object)
        for tr, te in folds:
            model = HierarchicalSoftmaxNN(cols, arch=arch, seed=seed, **BASE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(frame.iloc[tr], target.iloc[tr].to_numpy())
                oof_m[te] = model.predict_merged(frame.iloc[te])
                oof_f[te] = model.predict(frame.iloc[te])
        m = scores(truth_merged, oof_m, is_change_label)
        f = scores(truth_fine, oof_f, is_change_label)
        m_f1.append(m["change_f1"]); f_f1.append(f["change_f1"]); m_bal.append(m["balanced_accuracy"])
    return {
        "merged_change_f1": float(np.mean(m_f1)),
        "merged_change_f1_std": float(np.std(m_f1)),
        "fine_change_f1": float(np.mean(f_f1)),
        "merged_bal_acc": float(np.mean(m_bal)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ANNUAL_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[20250717, 1, 2])
    parser.add_argument("--tag", default="annual")
    args = parser.parse_args()

    frame = attach_features(load_annual(args.input))
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    sets = feature_sets()
    base_cols = year_cols(2018) + year_cols(2024)
    folds = list(make_splitter("blocked", args.n_splits).split(
        frame[base_cols], target, groups))
    print(f"{len(frame):,} plots | {groups.nunique()} blocks | "
          f"seeds={args.seeds} | base={BASE}", flush=True)

    rows = []
    for arch, fs in CONFIGS:
        res = run_config(frame, target, groups, folds, arch, sets[fs], args.seeds)
        row = {"arch": arch, "feature_set": fs, "n_features": len(sets[fs]), **res}
        rows.append(row)
        print(f"  {arch:8s} / {fs:5s}  merged_f1={res['merged_change_f1']:.4f} "
              f"+/-{res['merged_change_f1_std']:.4f}  fine_f1={res['fine_change_f1']:.4f}  "
              f"bal={res['merged_bal_acc']:.4f}", flush=True)

    board = pd.DataFrame(rows).sort_values("merged_change_f1", ascending=False)
    ref = board[(board.arch == "wide") & (board.feature_set == "full")]["merged_change_f1"].iloc[0]
    board["d_vs_wide_full"] = (board["merged_change_f1"] - ref).round(4)
    for c in ["merged_change_f1", "merged_change_f1_std", "fine_change_f1", "merged_bal_acc"]:
        board[c] = board[c].round(4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"hier_se_{args.tag}.csv"
    board.to_csv(out, index=False)
    print("\n" + board[["arch", "feature_set", "n_features", "merged_change_f1",
                        "merged_change_f1_std", "d_vs_wide_full", "fine_change_f1"]]
          .to_string(index=False))
    print(f"\nwide/full mean merged_change_f1={ref:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
