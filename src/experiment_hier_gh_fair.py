"""Fair-budget control for the G-H sampler test (see experiment_hier_gh.py).

The first G-H run is confounded: class-balanced sampling makes ceil(max_count/gh_m)
~= 300 mini-batches per epoch, so at 30 epochs the G-H nets take ~9,500 optimiser
steps against the full-batch baselines' 30. Any deficit could be over-training on
the (noisy) rare change classes rather than the balanced *composition* per se.

This isolates the composition by matching the optimiser-step budget. Two shuffle
controls take the same per-epoch minibatch count as G-H but draw batches at the
natural (imbalanced) class frequency; if G-H still loses to those, the balanced
composition -- not the step budget -- is what hurts. A short-epoch G-H is also
included so G-H is seen at a budget comparable to the full-batch baselines.

Scored on the merged2 deploy read under the same spatially blocked CV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hier_variants import evaluate_variant, score_variant
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from model_zoo import DEFAULT_INPUT, is_change_label, to_merged_label
from project_paths import project_data_dir

ARCH = "wide"


def configs():
    """Matched-budget comparison: G-H vs equal-step shuffle minibatches."""
    return [
        # Full-batch references (30 steps total).
        ("shuffle|focal|fullbatch|e30", dict(arch=ARCH, loss="focal", epochs=30)),
        # Equal-step shuffle controls: same minibatch count as G-H, natural
        # (imbalanced) composition. bs=64 over 6,400 rows ~= 100 steps/epoch.
        ("shuffle|focal|bs64|e30", dict(arch=ARCH, loss="focal", epochs=30,
                                        batch_size=64)),
        ("shuffle|weighted_ce|bs64|e30", dict(arch=ARCH, loss="weighted_ce",
                                              epochs=30, batch_size=64)),
        # G-H at a comparable step budget (few epochs) and at a large per-class
        # quota (fewer, less-oversampled batches).
        ("gh|focal|e5", dict(arch=ARCH, loss="focal", epochs=5, sampler="gh")),
        ("gh|focal|m32|e30", dict(arch=ARCH, loss="focal", epochs=30,
                                  sampler="gh", gh_m=32)),
        ("gh|focal|m64|e30", dict(arch=ARCH, loss="focal", epochs=30,
                                  sampler="gh", gh_m=64)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    print(f"{len(frame):,} plots | {len(columns)} features | "
          f"{target.nunique()} coarse3 classes")

    rows = []
    for name, kwargs in configs():
        of, om, secs = evaluate_variant(name, kwargs, frame, columns, target,
                                        groups, args.n_splits)
        row = score_variant(name, of, om, truth_fine, truth_merged, secs)
        rows.append(row)
        print(f"  {name:28s} merged_chgF1={row['merged_change_f1']:.3f} "
              f"(R={row['merged_change_recall']:.3f} P={row['merged_change_precision']:.3f}) "
              f"balAcc={row['merged_bal_acc']:.3f} ({secs}s)", flush=True)

    board = pd.DataFrame(rows).sort_values("merged_change_f1", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    board.to_csv(args.output_dir / f"hier_gh_fair{tag}.csv", index=False)
    (args.output_dir / f"hier_gh_fair_meta{tag}.json").write_text(
        json.dumps({"input": str(args.input), "arch": ARCH,
                    "n_splits": args.n_splits}, indent=2), encoding="utf-8")
    print("\n" + board.to_string(index=False))


if __name__ == "__main__":
    main()
