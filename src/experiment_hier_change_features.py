"""Change features vs the raw difference bands, for the hierarchical net.

Same question as experiment_change_features.py, but scored on the deployed
model (HierarchicalSoftmaxNN, wide/focal) and its real metric: change-F1 at the
merged2 (Vegetation/Artificial) deploy level, with coarse3 fine-F1 alongside.

Because the AlphaEarth vectors are unit-norm, the 64 ``A??_diff`` bands already
carry the cosine-distance signal (``||diff||^2 = 2 - 2 cos``); a scalar cosine
distance is a compression that keeps change *magnitude* but drops the per-band
*direction*. This measures whether that trade helps or hurts the net.

Feature sets (all built locally from bands already in the parquet):

    per_year    A??_2018 + A??_2024                (no change features)
    diff        + the 64 raw difference bands      (current hier default)
    cosine      + a single cosine-distance scalar
    scalars     + cos-dist, L1, Chebyshev, norm change
    diff+cos    + diff bands and the scalar together
    dot         + 64 per-band products t2018_i * t2024_i
    nd          + 64 per-band normalised differences (t2 - t1)/(|t1| + |t2|)
    diff+dot    + diff bands and the per-band products together
    diff+nd     + diff bands and the per-band normalised differences together

The dot and nd families are the per-band analogues of the diff bands: the
product keeps per-dimension agreement (its sum is cosine similarity), and the
normalised difference is a scale-free, bounded contrast per dimension.

Nested spatial CV is not needed here (no cleaning/tuning on the fold), so a
single blocked outer CV is scored, mirroring experiment_hier_clean's outer loop.
Writes a ranked table to analysis_results/.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_change_features import (
    change_features,
    dot_columns,
    dot_features,
    nd_columns,
    nd_features,
)
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
SCALARS = ["cos_dist", "l1", "chebyshev", "norm_change"]


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    per_year = [c for c in columns if not c.endswith("_diff")]
    diff = list(columns)  # per-year + the 64 A??_diff bands (hier default)
    dot = dot_columns()   # 64 per-band products t2018_i * t2024_i
    nd = nd_columns()     # 64 per-band normalised differences
    return {
        "per_year": per_year,
        "diff": diff,
        "cosine": per_year + ["cos_dist"],
        "scalars": per_year + SCALARS,
        "diff+cos": diff + ["cos_dist"],
        "dot": per_year + dot,
        "nd": per_year + nd,
        "diff+dot": diff + dot,
        "diff+nd": diff + nd,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="2yr")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    # Derive the change descriptors and attach them as real columns so the net's
    # _prepare standardises them alongside the embeddings: scalar summaries plus
    # the two per-band families under test (product and normalised difference).
    frame = pd.concat(
        [frame, change_features(frame), dot_features(frame), nd_features(frame)],
        axis=1,
    )
    groups = frame["block_id"]

    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])

    sets = feature_sets(columns)
    print(f"{len(frame):,} plots | {len(set(truth_fine))} coarse3 / "
          f"{len(set(truth_merged))} merged2 | {groups.nunique()} blocks | base={BASE}",
          flush=True)

    splitter = make_splitter("blocked", args.n_splits)
    folds = list(splitter.split(frame[columns], target, groups))

    rows = []
    for name, cols in sets.items():
        oof_merged = np.empty(len(target), dtype=object)
        oof_fine = np.empty(len(target), dtype=object)
        for tr, te in folds:
            model = HierarchicalSoftmaxNN(cols, **BASE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(frame.iloc[tr], target.iloc[tr].to_numpy())
                oof_merged[te] = model.predict_merged(frame.iloc[te])
                oof_fine[te] = model.predict(frame.iloc[te])
        m = scores(truth_merged, oof_merged, is_change_label)
        f = scores(truth_fine, oof_fine, is_change_label)
        rows.append({
            "feature_set": name,
            "n_features": len(cols),
            "merged_change_f1": round(m["change_f1"], 4),
            "merged_change_recall": round(m["change_recall"], 4),
            "merged_change_precision": round(m["change_precision"], 4),
            "merged_bal_acc": round(m["balanced_accuracy"], 4),
            "fine_change_f1": round(f["change_f1"], 4),
            "fine_bal_acc": round(f["balanced_accuracy"], 4),
        })
        print(f"  {name:10s} merged_f1={m['change_f1']:.4f} "
              f"fine_f1={f['change_f1']:.4f} merged_bal={m['balanced_accuracy']:.4f}",
              flush=True)

    board = pd.DataFrame(rows).sort_values("merged_change_f1", ascending=False)
    base = board.loc[board.feature_set == "diff", "merged_change_f1"].iloc[0]
    board["d_vs_diff"] = (board["merged_change_f1"] - base).round(4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"hier_change_features_{args.tag}.csv"
    board.to_csv(out, index=False)
    print("\n" + board[["feature_set", "n_features", "merged_change_f1", "d_vs_diff",
                        "merged_bal_acc", "fine_change_f1"]].to_string(index=False))
    print(f"\nbaseline (diff) merged_change_f1={base:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
