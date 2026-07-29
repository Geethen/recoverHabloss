"""Empirically test the accuracy-improvement ideas for HierarchicalSoftmaxNN.

The arch/loss/head/noise/SSL knobs all saturate within the label-noise floor
(experiment_hier_variants.py, experiment_hier_novel.py). This script measures the
levers that were *not* yet tried, on the merged2 deploy metric under the same
spatially blocked CV, all against the wide|focal|30ep control:

* **Optimisation** -- the trunk trains full-batch with OneCycle total_steps=epochs,
  i.e. ~30 gradient steps total. ``batch_size`` switches to shuffled minibatches
  (many more steps + SGD noise); ``early_stop`` holds out a within-fold split and
  restores the best-epoch weights by merged2 change-F1.
* **Seed ensembling** -- average the merged softmax over K seeds. The variant
  deltas so far are ~0.03 F1, i.e. within single-seed noise; this measures it.
* **Deploy-weighted loss** -- ``level_weights`` toward merged2 (down-weight the
  noisy fine head), since merged2 is what ships.
* **Mixup** -- noise-robust input/label interpolation.
* **Annual trajectory** -- run the same set on the 7-year annual parquet, adding
  the GRU trunk, to see whether the path (not just the endpoints) carries change.

Writes a leaderboard (delta vs baseline) to analysis_results/.
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hier_variants import evaluate_variant, score_variant
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


def evaluate_ensemble(kwargs, seeds, frame, columns, target, groups, n_splits):
    """Blocked-CV OOF predictions averaging the softmax over ``seeds``."""
    features = frame[columns]
    oof_fine = np.empty(len(target), dtype=object)
    oof_merged = np.empty(len(target), dtype=object)
    started = time.time()
    for tr, te in make_splitter("blocked", n_splits).split(features, target, groups):
        pf = pm = None
        fine_classes = merged_classes = None
        for seed in seeds:
            model = HierarchicalSoftmaxNN(columns, **kwargs, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(frame.iloc[tr], target.iloc[tr].to_numpy())
                f, m = model._probs(frame.iloc[te])
            pf = f if pf is None else pf + f
            pm = m if pm is None else pm + m
            fine_classes, merged_classes = model.fine_classes_, model.merged_classes_
        oof_fine[te] = np.array(fine_classes, dtype=object)[pf.argmax(1)]
        oof_merged[te] = np.array(merged_classes, dtype=object)[pm.argmax(1)]
    return oof_fine, oof_merged, round(time.time() - started, 1)


def configs_for(has_sequence: bool):
    """(name, kind, kwargs) rows. 'single' one fit/fold; 'ensemble' K seeds."""
    rows = [
        ("baseline", "single", dict(**BASE)),
        # -- optimisation: minibatch + more effective steps --
        ("minibatch256_30", "single", dict(**BASE, batch_size=256)),
        ("minibatch256_60", "single", dict(arch="wide", loss="focal", epochs=60,
                                            batch_size=256)),
        ("minibatch128_60", "single", dict(arch="wide", loss="focal", epochs=60,
                                            batch_size=128)),
        ("earlystop_fullbatch", "single", dict(arch="wide", loss="focal",
                                                epochs=200, early_stop=True,
                                                patience=20)),
        ("earlystop_minibatch", "single", dict(arch="wide", loss="focal",
                                                epochs=120, batch_size=256,
                                                early_stop=True, patience=10)),
        # -- deploy-weighted loss (wg, wm, wf): down-weight the noisy fine head --
        ("lw_fine0.3", "single", dict(**BASE, level_weights=(1.0, 1.0, 0.3))),
        ("lw_merged2_fine0.5", "single", dict(**BASE, level_weights=(1.0, 2.0, 0.5))),
        # -- mixup (needs minibatches to mix distinct pairs each step) --
        ("mixup0.2_mb256", "single", dict(arch="wide", loss="focal", epochs=60,
                                          batch_size=256, mixup_alpha=0.2)),
        # -- seed ensembles --
        ("ensemble5", "ensemble", dict(**BASE)),
        ("ensemble5_mb256_60", "ensemble", dict(arch="wide", loss="focal",
                                                epochs=60, batch_size=256)),
    ]
    if has_sequence:
        rows += [
            ("gru_30", "single", dict(arch="gru", loss="focal", epochs=30)),
            ("gru_60", "single", dict(arch="gru", loss="focal", epochs=60)),
            ("gru_mb256_60", "single", dict(arch="gru", loss="focal", epochs=60,
                                            batch_size=256)),
            ("gru_earlystop", "single", dict(arch="gru", loss="focal", epochs=120,
                                             early_stop=True, patience=12)),
            ("gru_ensemble5", "ensemble", dict(arch="gru", loss="focal", epochs=30)),
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--names", nargs="+", default=None,
                        help="Keep only configs whose name contains one of these "
                             "substrings (default: all applicable)")
    parser.add_argument("--tag", default="2yr")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])

    year_blocks = {c.split("_")[1] for c in columns
                   if c.split("_")[1:] and c.split("_")[1].isdigit()}
    has_sequence = len(year_blocks) > 2
    print(f"{len(frame):,} plots | {len(columns)} features | "
          f"{len(year_blocks)} year blocks (sequence={has_sequence}) | base={BASE}",
          flush=True)

    selected = configs_for(has_sequence)
    if args.names:
        selected = [c for c in selected
                    if any(sub in c[0] for sub in args.names) or c[0] == "baseline"]
    rows = []
    for name, kind, kwargs in selected:
        try:
            if kind == "ensemble":
                of, om, secs = evaluate_ensemble(kwargs, args.seeds, frame, columns,
                                                 target, groups, args.n_splits)
            else:
                of, om, secs = evaluate_variant(name, kwargs, frame, columns,
                                                 target, groups, args.n_splits)
            row = score_variant(name, of, om, truth_fine, truth_merged, secs)
            rows.append(row)
            print(f"  {name:22s} merged_chgF1={row['merged_change_f1']:.4f} "
                  f"balAcc={row['merged_bal_acc']:.4f} recall={row['merged_change_recall']:.3f} "
                  f"prec={row['merged_change_precision']:.3f} fineF1={row['fine_change_f1']:.3f} "
                  f"({secs}s)", flush=True)
        except Exception as error:
            print(f"  {name:22s} FAILED - {type(error).__name__}: {error}", flush=True)

    board = pd.DataFrame(rows)
    base_f1 = board.loc[board["variant"] == "baseline", "merged_change_f1"]
    base_f1 = float(base_f1.iloc[0]) if len(base_f1) else float("nan")
    board["delta_vs_baseline"] = (board["merged_change_f1"] - base_f1).round(4)
    board = board.sort_values("merged_change_f1", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"hier_improve_{args.tag}.csv"
    board.to_csv(out, index=False)
    print(f"\nbaseline merged_change_f1 = {base_f1:.4f}\n")
    print(board[["variant", "merged_change_f1", "delta_vs_baseline", "merged_bal_acc",
                 "merged_change_recall", "merged_change_precision", "fine_change_f1",
                 "seconds"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
