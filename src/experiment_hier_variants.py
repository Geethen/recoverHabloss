"""Sweep hierarchical-softmax network variants (architecture x loss).

Builds on the finding that the merged2 (Vegetation/Artificial) legend is the
clean, deployable target and the Cropland/Nature split is interpreter noise
(analyse_label_noise.py). ``HierarchicalSoftmaxNN`` puts merged2 in the *middle*
of a nested gate -> merged2 -> coarse3 hierarchy and supervises all three
levels off one fine head. This script asks two questions on one footing, under
the same spatially blocked CV as the leaderboard:

* **Architecture** -- does the trunk matter? Flat MLP vs a wider MLP vs a
  residual MLP vs a GRU over the annual trajectory (the last only when the input
  carries the per-year sequence).
* **Loss** -- does the objective matter? Plain cross-entropy vs inverse-frequency
  weighting vs focal vs class-balanced focal, applied identically at all three
  levels.

Every variant is scored twice: at the **merged2** level (the deploy read, from
``predict_merged``) and at the **coarse3** level (the informative read, from
``predict``). A ``fine_only`` ablation (all loss weight on the fine head, none on
the gate/merged levels) is included so the contribution of the intermediate
supervision itself -- not just the architecture -- is visible.

Leaderboard is sorted by merged2 change-F1 and written to analysis_results/.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from model_zoo import (
    DEFAULT_INPUT,
    HierarchicalSoftmaxNN,
    feature_columns,
    is_change_label,
    make_splitter,
    scores,
    to_merged_label,
)
from project_paths import project_data_dir

LOSSES = ["ce", "weighted_ce", "focal", "cb_focal"]
FLAT_ARCHS = ["mlp", "wide", "deep_res"]


def variant_configs(archs, losses, epochs):
    """(name, kwargs) for every arch x loss, plus a fine-only ablation."""
    configs = []
    for arch in archs:
        for loss in losses:
            configs.append((f"{arch}|{loss}", dict(arch=arch, loss=loss,
                                                    epochs=epochs)))
    # Ablation: strip the intermediate/gate supervision from the strongest-default
    # arch, so the lift from putting merged2 in the middle is measurable.
    configs.append(("mlp|ce|fine_only",
                    dict(arch="mlp", loss="ce", epochs=epochs,
                         level_weights=(0.0, 0.0, 1.0))))
    return configs


def evaluate_variant(name, kwargs, frame, columns, target, groups, n_splits):
    """Blocked-CV out-of-fold fine and merged predictions for one variant."""
    features = frame[columns]
    oof_fine = np.empty(len(target), dtype=object)
    oof_merged = np.empty(len(target), dtype=object)
    started = time.time()
    splitter = make_splitter("blocked", n_splits)
    for tr, te in splitter.split(features, target, groups):
        model = HierarchicalSoftmaxNN(columns, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame.iloc[tr], target.iloc[tr].to_numpy())
            oof_fine[te] = model.predict(frame.iloc[te])
            oof_merged[te] = model.predict_merged(frame.iloc[te])
    return oof_fine, oof_merged, round(time.time() - started, 1)


def score_variant(name, oof_fine, oof_merged, truth_fine, truth_merged, seconds):
    fine = scores(truth_fine, oof_fine, is_change_label)
    merged = scores(truth_merged, oof_merged, is_change_label)
    return {
        "variant": name,
        "merged_change_f1": round(merged["change_f1"], 4),
        "merged_change_recall": round(merged["change_recall"], 4),
        "merged_change_precision": round(merged["change_precision"], 4),
        "merged_bal_acc": round(merged["balanced_accuracy"], 4),
        "merged_accuracy": round(merged["accuracy"], 4),
        "fine_change_f1": round(fine["change_f1"], 4),
        "fine_bal_acc": round(fine["balanced_accuracy"], 4),
        "fine_accuracy": round(fine["accuracy"], 4),
        "seconds": seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--archs", nargs="+", default=None,
                        help="Trunks to sweep (default: flat archs; add 'gru' on "
                             "annual input)")
    parser.add_argument("--losses", nargs="+", default=LOSSES)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])

    # Default archs: flat set, plus gru only if the input carries >=2 year blocks.
    archs = args.archs
    if archs is None:
        archs = list(FLAT_ARCHS)
        year_blocks = {c.split("_")[1] for c in columns
                       if c.split("_")[1:] and c.split("_")[1].isdigit()}
        if len(year_blocks) > 2:
            archs.append("gru")

    print(f"{len(frame):,} plots | {len(columns)} features | "
          f"{target.nunique()} coarse3 / {len(set(truth_merged))} merged2 classes | "
          f"archs={archs} | losses={args.losses}")

    configs = variant_configs(archs, args.losses, args.epochs)
    rows = []
    for name, kwargs in configs:
        try:
            oof_fine, oof_merged, secs = evaluate_variant(
                name, kwargs, frame, columns, target, groups, args.n_splits
            )
            row = score_variant(name, oof_fine, oof_merged, truth_fine,
                                 truth_merged, secs)
            rows.append(row)
            print(f"  {name:24s} merged_chgF1={row['merged_change_f1']:.3f} "
                  f"merged_balAcc={row['merged_bal_acc']:.3f} | "
                  f"fine_chgF1={row['fine_change_f1']:.3f} ({secs}s)", flush=True)
        except Exception as error:
            print(f"  {name:24s} FAILED — {type(error).__name__}: {error}", flush=True)

    board = pd.DataFrame(rows).sort_values("merged_change_f1", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    board.to_csv(args.output_dir / f"hier_variants{tag}.csv", index=False)
    (args.output_dir / f"hier_variants_meta{tag}.json").write_text(
        json.dumps({"input": str(args.input), "archs": archs,
                    "losses": args.losses, "epochs": args.epochs,
                    "n_splits": args.n_splits}, indent=2),
        encoding="utf-8",
    )
    print("\n" + board.to_string(index=False))


if __name__ == "__main__":
    main()
