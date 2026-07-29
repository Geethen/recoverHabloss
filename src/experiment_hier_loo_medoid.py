"""Does a leave-one-out medoid anomaly score help the hierarchical net?

The LOO medoid score (loo_medoid.py, after Wherobots' AlphaEarth change-detection
write-up) measures how unusual each annual embedding is relative to the robust
centre of the plot's other years. It needs the full annual trajectory, so this
runs on the annual parquet (2018..2024, 7 years) rather than the two endpoints.

To keep the comparison honest against the deployed model, the *base* embedding
features are always the deployed 2-year set (A??_2018 + A??_2024 + A??_diff);
the trajectory is used only to derive the LOO scalars that are added on top. We
also carry the cosine-distance scalar (the previous experiment's small winner) so
we can see whether LOO adds anything cosine does not.

Feature sets (scored on merged2 change-F1, the deploy metric):

    diff          A??_2018 + A??_2024 + A??_diff         (deployed baseline)
    diff+cos      + cosine-distance scalar
    diff+loo      + 5 LOO medoid scalars
    diff+cos+loo  + both
    per_year+loo  A??_2018 + A??_2024 + LOO              (LOO without diff bands)

Writes a ranked table to analysis_results/.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_change_features import change_features
from experiment_merged_legend import LEGENDS, target_for_legend
from loo_medoid import loo_medoid_scores, loo_summary
from model_zoo import (
    HierarchicalSoftmaxNN,
    is_change_label,
    make_splitter,
    scores,
    to_merged_label,
)
from project_paths import project_data_dir

ANNUAL_INPUT = project_data_dir("embeddings", "embeddings_habloss_recover_annual.parquet")
BASE = dict(arch="wide", loss="focal", epochs=30)
YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
LOO_COLS = ["loo_start", "loo_end", "loo_max", "loo_mean", "loo_end_minus_start"]


def year_cols(year: int) -> list[str]:
    return [f"A{i:02d}_{year}" for i in range(64)]


def load_annual(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    dup = int(frame["PLOTID"].duplicated().sum())
    if dup:
        print(f"Dropping {dup} duplicate PLOTID rows")
        frame = frame.drop_duplicates("PLOTID").reset_index(drop=True)
    band_cols = [c for c in year_cols(2018) + year_cols(2024) + [f"A{i:02d}_diff" for i in range(64)]]
    traj_cols = [c for y in YEARS for c in year_cols(y)]
    needed = sorted(set(band_cols + traj_cols))
    complete = frame[needed].notna().all(axis=1)
    if not complete.all():
        print(f"Dropping {int((~complete).sum())} plots with missing embeddings")
        frame = frame.loc[complete].reset_index(drop=True)
    frame[needed] = frame[needed].astype("float64")
    return frame


def attach_features(frame: pd.DataFrame) -> pd.DataFrame:
    # Cosine + basic change scalars, from the two endpoints.
    change = change_features(frame)
    # LOO medoid, from the full annual trajectory.
    traj = np.stack([frame[year_cols(y)].to_numpy(float) for y in YEARS], axis=1)
    loo = loo_summary(loo_medoid_scores(traj), YEARS)
    loo = pd.DataFrame(loo, index=frame.index)
    return pd.concat([frame, change, loo], axis=1)


def feature_sets() -> dict[str, list[str]]:
    per_year = year_cols(2018) + year_cols(2024)
    diff = per_year + [f"A{i:02d}_diff" for i in range(64)]
    return {
        "diff": diff,
        "diff+cos": diff + ["cos_dist"],
        "diff+loo": diff + LOO_COLS,
        "diff+cos+loo": diff + ["cos_dist"] + LOO_COLS,
        "per_year+loo": per_year + LOO_COLS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ANNUAL_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="annual")
    args = parser.parse_args()

    frame = load_annual(args.input)
    frame = attach_features(frame)
    groups = frame["block_id"]

    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])

    sets = feature_sets()
    print(f"{len(frame):,} plots | {len(set(truth_fine))} coarse3 / "
          f"{len(set(truth_merged))} merged2 | {groups.nunique()} blocks | base={BASE}",
          flush=True)
    # Quick look at how the LOO signal separates change from no-change plots.
    change_mask = np.array([is_change_label(t) for t in truth_merged])
    print(f"loo_max  change={frame['loo_max'][change_mask].mean():.3f}  "
          f"stable={frame['loo_max'][~change_mask].mean():.3f} | "
          f"loo_end change={frame['loo_end'][change_mask].mean():.3f} "
          f"stable={frame['loo_end'][~change_mask].mean():.3f}", flush=True)

    base_cols = year_cols(2018) + year_cols(2024)
    splitter = make_splitter("blocked", args.n_splits)
    folds = list(splitter.split(frame[base_cols], target, groups))

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
        print(f"  {name:13s} merged_f1={m['change_f1']:.4f} "
              f"fine_f1={f['change_f1']:.4f} merged_bal={m['balanced_accuracy']:.4f}",
              flush=True)

    board = pd.DataFrame(rows).sort_values("merged_change_f1", ascending=False)
    base = board.loc[board.feature_set == "diff", "merged_change_f1"].iloc[0]
    board["d_vs_diff"] = (board["merged_change_f1"] - base).round(4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"hier_loo_medoid_{args.tag}.csv"
    board.to_csv(out, index=False)
    print("\n" + board[["feature_set", "n_features", "merged_change_f1", "d_vs_diff",
                        "merged_bal_acc", "fine_change_f1"]].to_string(index=False))
    print(f"\nbaseline (diff) merged_change_f1={base:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
