"""Test the Global-Hierarchical (G-H) sampler on the hierarchical-softmax net.

Ports the sample-balancing idea from Zhang et al. (2022, ISPRS J. P&RS 184:63,
"Siam-GL"). Their diagnosis: patch-based change detection is biased because the
stable/no-change majority swamps the rare *change* classes in every gradient
step, so the network never sees enough of the transitions that matter. Their fix
(Algorithm 1, Global Hierarchical sampling) draws a fixed mini-batch *per class*
each step, so every change class is present in every step, and adds a per-class
loss weight w_j to equalise the residual imbalance.

Our analogue is exact: the coarse3 transition target is dominated by the stable
Veg->Veg / Art->Art classes, and the change transitions (esp. the Cropland/Nature
boundary, see analyse_label_noise.py) are the starved minority. We already
balance from the *loss* side (weighted_ce / focal / cb_focal); G-H balances from
the *sampling* side. The question this script answers, on the same spatially
blocked CV as experiment_hier_variants.py and scored on the merged2 deploy read:

  does class-balanced G-H sampling beat -- or stack with -- loss reweighting?

Configs (all arch=wide, epochs=30, the hier_novel base):
  * shuffle|ce / focal / weighted_ce  -- the loss-side baselines
  * gh|ce                             -- pure G-H sampling, no loss weight
  * gh|focal                          -- sampling + focal
  * gh|weighted_ce                    -- sampling + w_j loss (full Siam-GL recipe)
  * gh|ce sweep over gh_m             -- per-class quota sensitivity

Writes a leaderboard sorted by merged2 change-F1 to analysis_results/.
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
EPOCHS = 30


def configs(gh_m_sweep):
    """(name, kwargs) baselines then G-H variants then the gh_m sweep."""
    base = dict(arch=ARCH, epochs=EPOCHS)
    out = [
        ("shuffle|ce", dict(**base, loss="ce")),
        ("shuffle|focal", dict(**base, loss="focal")),
        ("shuffle|weighted_ce", dict(**base, loss="weighted_ce")),
        ("gh|ce", dict(**base, loss="ce", sampler="gh")),
        ("gh|focal", dict(**base, loss="focal", sampler="gh")),
        ("gh|weighted_ce", dict(**base, loss="weighted_ce", sampler="gh")),
    ]
    for m in gh_m_sweep:
        out.append((f"gh|ce|m{m}", dict(**base, loss="ce", sampler="gh", gh_m=m)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--gh-m-sweep", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    counts = target.value_counts()
    print(f"{len(frame):,} plots | {len(columns)} features | "
          f"{target.nunique()} coarse3 / {len(set(truth_merged))} merged2 classes")
    print(f"class support: min={counts.min()} median={int(counts.median())} "
          f"max={counts.max()}")

    rows = []
    for name, kwargs in configs(args.gh_m_sweep):
        of, om, secs = evaluate_variant(name, kwargs, frame, columns, target,
                                        groups, args.n_splits)
        row = score_variant(name, of, om, truth_fine, truth_merged, secs)
        rows.append(row)
        print(f"  {name:20s} merged_chgF1={row['merged_change_f1']:.3f} "
              f"(R={row['merged_change_recall']:.3f} P={row['merged_change_precision']:.3f}) "
              f"balAcc={row['merged_bal_acc']:.3f} | "
              f"fine_chgF1={row['fine_change_f1']:.3f} ({secs}s)", flush=True)

    board = pd.DataFrame(rows).sort_values("merged_change_f1", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    board.to_csv(args.output_dir / f"hier_gh{tag}.csv", index=False)
    (args.output_dir / f"hier_gh_meta{tag}.json").write_text(
        json.dumps({"input": str(args.input), "arch": ARCH, "epochs": EPOCHS,
                    "n_splits": args.n_splits, "gh_m_sweep": args.gh_m_sweep},
                   indent=2), encoding="utf-8")
    print("\n" + board.to_string(index=False))


if __name__ == "__main__":
    main()
