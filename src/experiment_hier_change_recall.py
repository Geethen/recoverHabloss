"""Two cheap levers for change recall on the wide|focal hier-softmax net.

The interleaved-noise sweep (experiment_hier_moe_noise.py) bought no reliable
change-class gain. This script quantifies the two levers that actually move
change detection, against the *same* ``arch=wide, loss=focal``
``HierarchicalSoftmaxNN`` baseline, under the same spatially blocked CV and
merged2 (Vegetation/Artificial) deploy read:

1. **Gate-threshold tuning.** The deploy label is ``argmax`` of the merged
   probability, i.e. an implicit 0.5 change gate. But change is a ~15% minority
   and the map exists to stratify a change area estimate, so a lower gate trades
   precision for recall. We collect out-of-fold merged probabilities *once* and
   re-label at every candidate threshold with no refit -- change is called when
   ``P(change) = P(Veg->Art) + P(Art->Veg) >= t``, naming the transition by the
   arg-max within the chosen side. Reports change P/R/F1 and the map's relative
   efficiency vs simple random sampling (map_efficiency.py).

2. **Seed ensembling.** Average the merged probabilities of N independently
   seeded nets before the arg-max. Same per-fold cost x N, zero inference change
   beyond storing N nets; the question is how much the variance reduction alone
   lifts the rare change classes.

The two compose (ensemble, then threshold). Everything is reported at a fixed
seed set so the comparison is like-for-like with the noise sweep. Tables ->
analysis_results/.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from map_efficiency import efficiency, srs_term, stratified_term
from model_zoo import (
    DEFAULT_INPUT,
    HierarchicalSoftmaxNN,
    is_change_label,
    make_splitter,
    scores,
    to_merged_label,
)
from project_paths import project_data_dir

BASE = dict(arch="wide", loss="focal")


def collect_merged_probs(frame, columns, target, groups, seeds, epochs, n_splits):
    """Per-seed out-of-fold merged-probability matrix in one global class order."""
    merged_classes = sorted({to_merged_label(c) for c in target.to_numpy()})
    col = {c: i for i, c in enumerate(merged_classes)}
    n = len(target)
    probs = {s: np.zeros((n, len(merged_classes))) for s in seeds}
    splitter = make_splitter("blocked", n_splits)
    for seed in seeds:
        for tr, te in splitter.split(frame[columns], target, groups):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = HierarchicalSoftmaxNN(columns, epochs=epochs, seed=seed, **BASE)
                m.fit(frame.iloc[tr], target.iloc[tr].to_numpy())
                _, pm = m._probs(frame.iloc[te])
            # Re-index the fold's merged columns into the global order.
            for j, c in enumerate(m.merged_classes_):
                probs[seed][te, col[c]] = pm[:, j]
        print(f"  collected seed {seed}", flush=True)
    return merged_classes, probs


def label_at_threshold(p_merged, merged_classes, threshold):
    """Deploy label when change is called at ``P(change) >= threshold``.

    Below threshold the plot is named by the arg-max stable class, above it by
    the arg-max change class -- the merged analogue of HierarchicalTorchNN.combine.
    """
    is_chg = np.array([is_change_label(c) for c in merged_classes])
    classes = np.array(merged_classes, dtype=object)
    chg_cols = np.where(is_chg)[0]
    stab_cols = np.where(~is_chg)[0]
    p_change = p_merged[:, chg_cols].sum(1)
    chg_pick = classes[chg_cols][p_merged[:, chg_cols].argmax(1)]
    stab_pick = classes[stab_cols][p_merged[:, stab_cols].argmax(1)]
    return np.where(p_change >= threshold, chg_pick, stab_pick)


def score_row(name, pred, truth, y_change, srs, threshold, seconds=None):
    s = scores(truth, pred, is_change_label)
    eta = efficiency(stratified_term(pred, y_change), srs)
    return {
        "config": name,
        "threshold": round(float(threshold), 3),
        "change_frac": round(float(np.mean([is_change_label(p) for p in pred])), 4),
        "change_recall": round(s["change_recall"], 4),
        "change_precision": round(s["change_precision"], 4),
        "change_f1": round(s["change_f1"], 4),
        "balanced_accuracy": round(s["balanced_accuracy"], 4),
        "efficiency_vs_srs": round(float(eta), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-thresholds", type=int, default=19)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth = np.array([to_merged_label(t) for t in target.to_numpy()])
    y_change = np.array([is_change_label(t) for t in truth], dtype=float)
    srs = srs_term(y_change)
    base_rate = float(y_change.mean())
    print(f"{len(frame):,} plots | merged2 | base={BASE} | seeds={args.seeds} | "
          f"change rate {base_rate:.1%} | {args.n_splits}-fold blocked CV")

    merged_classes, probs = collect_merged_probs(
        frame, columns, target, groups, args.seeds, args.epochs, args.n_splits)
    thresholds = np.linspace(0.05, 0.95, args.n_thresholds)

    # --- Lever B building blocks: single-seed probs vs the N-seed mean ---
    single = probs[args.seeds[0]]
    ensemble = np.mean([probs[s] for s in args.seeds], axis=0)

    # 1. Baseline: single seed, implicit 0.5 gate (arg-max == threshold 0.5).
    rows = [score_row("baseline (1 seed, t=0.5)",
                      label_at_threshold(single, merged_classes, 0.5),
                      truth, y_change, srs, 0.5)]

    # 2. Gate-threshold curve on the single-seed probs; report best-F1 and the
    #    highest-recall point that still holds change precision >= baseline.
    curve = pd.DataFrame([
        score_row("threshold sweep", label_at_threshold(single, merged_classes, t),
                  truth, y_change, srs, t)
        for t in thresholds
    ])
    best_f1 = curve.loc[curve["change_f1"].idxmax()]
    best_eta = curve.loc[curve["efficiency_vs_srs"].idxmax()]
    rows.append({**best_f1.to_dict(), "config": "threshold: best change-F1 (1 seed)"})
    rows.append({**best_eta.to_dict(), "config": "threshold: best efficiency (1 seed)"})

    # 3. Seed ensemble at the default gate, and ensemble + its own best-F1 threshold.
    rows.append(score_row(f"ensemble ({len(args.seeds)} seeds, t=0.5)",
                          label_at_threshold(ensemble, merged_classes, 0.5),
                          truth, y_change, srs, 0.5))
    ens_curve = pd.DataFrame([
        score_row("ens sweep", label_at_threshold(ensemble, merged_classes, t),
                  truth, y_change, srs, t)
        for t in thresholds
    ])
    ens_best = ens_curve.loc[ens_curve["change_f1"].idxmax()]
    rows.append({**ens_best.to_dict(),
                 "config": f"ensemble + threshold: best change-F1"})

    summary = pd.DataFrame(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    summary.to_csv(args.output_dir / f"hier_change_recall{tag}.csv", index=False)
    curve.to_csv(args.output_dir / f"hier_change_recall_curve{tag}.csv", index=False)
    (args.output_dir / f"hier_change_recall_meta{tag}.json").write_text(
        json.dumps({"input": str(args.input), "base_config": BASE,
                    "seeds": args.seeds, "epochs": args.epochs,
                    "n_splits": args.n_splits, "base_change_rate": round(base_rate, 4)},
                   indent=2), encoding="utf-8")

    print("\nThreshold curve (single seed):\n" + curve.drop(columns="config").to_string(index=False))
    print("\nSummary vs baseline:\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
