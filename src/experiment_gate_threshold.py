"""Tune the change-gate operating point of the gated networks.

The gated models (``hier_nn``, ``temporal_nn``) decide change vs no-change at a
fixed 0.5 gate probability, then let the matching head name the label. But 0.5
is rarely the right operating point: the change class is a ~15-20% minority, and
what the map is *for* -- stratifying the sample for a design-based area estimate
-- rewards a split that isolates change, not one that maximises raw accuracy.

This sweep collects the gate probability and both heads' labels out-of-fold
under the same spatially blocked CV, then re-labels every plot at each candidate
threshold without refitting (``predict_parts`` + ``combine``). For each threshold
it reports the change precision / recall / F1 *and* the relative efficiency eta
of the resulting map as a change stratifier versus simple random sampling
(Gallego 2007, as in ``map_efficiency.py``). The two optima need not coincide:
F1 balances precision and recall symmetrically, while eta rewards carving off a
small, almost-pure change stratum even at the cost of recall.

Writes the full threshold curve and the two selected operating points to
``analysis_results/``.
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
    build_registry,
    is_change_label,
    make_splitter,
    scores,
)
from project_paths import project_data_dir


def collect_oof(model_name, frame, columns, target, groups, n_splits):
    """Out-of-fold gate probability and both head labels for every plot."""
    registry = build_registry(columns, balanced=True)
    if model_name not in registry:
        raise SystemExit(f"{model_name} not available. Have: {sorted(registry)}")
    factory, needs_frame = registry[model_name]
    if not hasattr(factory(), "predict_parts"):
        raise SystemExit(
            f"{model_name} has no gate to tune; use a gated model "
            "(hier_nn, temporal_nn)"
        )

    features = frame[columns]
    p_change = np.empty(len(target), dtype="float64")
    change_label = np.empty(len(target), dtype=object)
    stable_label = np.empty(len(target), dtype=object)

    splitter = make_splitter("blocked", n_splits)
    for tr, te in splitter.split(features, target, groups):
        model = factory()
        fit_data = frame.iloc[tr] if needs_frame else features.iloc[tr].to_numpy()
        test_data = frame.iloc[te] if needs_frame else features.iloc[te].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(fit_data, target.iloc[tr].to_numpy())
            p, chg, stab = model.predict_parts(test_data)
        p_change[te], change_label[te], stable_label[te] = p, chg, stab
    return p_change, change_label, stable_label


def sweep(p_change, change_label, stable_label, truth, thresholds):
    from model_zoo import HierarchicalTorchNN

    y_change = np.array([is_change_label(t) for t in truth], dtype=float)
    srs = srs_term(y_change)
    rows = []
    for t in thresholds:
        pred = HierarchicalTorchNN.combine(p_change, change_label, stable_label, t)
        s = scores(truth, pred, is_change_label)
        eta = efficiency(stratified_term(pred, y_change), srs)
        rows.append({
            "threshold": round(float(t), 3),
            "change_frac": round(float(np.mean([is_change_label(p) for p in pred])), 4),
            "change_f1": round(s["change_f1"], 4),
            "change_recall": round(s["change_recall"], 4),
            "change_precision": round(s["change_precision"], 4),
            "balanced_accuracy": round(s["balanced_accuracy"], 4),
            "accuracy": round(s["accuracy"], 4),
            "efficiency_vs_srs": round(float(eta), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--model", default="hier_nn",
                        help="Gated model to tune (hier_nn, temporal_nn)")
    parser.add_argument("--legend", choices=list(LEGENDS), default="merged2",
                        help="Target legend (default: the merged Veg/Artificial)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--n-thresholds", type=int, default=19)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    target = target_for_legend(frame, LEGENDS[args.legend], args.min_class_count)
    groups = frame["block_id"]
    truth = target.to_numpy()
    base_change = float(np.mean([is_change_label(t) for t in truth]))
    print(f"{len(frame):,} plots | {len(columns)} features | {args.legend} | "
          f"{args.model} | base change rate {base_change:.1%}")

    p_change, change_label, stable_label = collect_oof(
        args.model, frame, columns, target, groups, args.n_splits
    )
    thresholds = np.linspace(0.05, 0.95, args.n_thresholds)
    curve = sweep(p_change, change_label, stable_label, truth, thresholds)

    best_f1 = curve.loc[curve["change_f1"].idxmax()]
    best_eta = curve.loc[curve["efficiency_vs_srs"].idxmax()]
    default = curve.iloc[(curve["threshold"] - 0.5).abs().idxmin()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else f"_{args.model}_{args.legend}"
    curve.to_csv(args.output_dir / f"gate_threshold{tag}.csv", index=False)
    (args.output_dir / f"gate_threshold{tag}.json").write_text(
        json.dumps({
            "input": str(args.input), "model": args.model, "legend": args.legend,
            "base_change_rate": round(base_change, 4),
            "default_0.5": default.to_dict(),
            "best_change_f1": best_f1.to_dict(),
            "best_efficiency": best_eta.to_dict(),
        }, indent=2),
        encoding="utf-8",
    )

    print("\n" + curve.to_string(index=False))
    print(f"\nDefault (t~0.5):  change_f1={default['change_f1']:.3f}  "
          f"eta={default['efficiency_vs_srs']:.2f}")
    print(f"Best change-F1:   t={best_f1['threshold']:.2f}  "
          f"f1={best_f1['change_f1']:.3f}  (recall={best_f1['change_recall']:.2f} "
          f"prec={best_f1['change_precision']:.2f})  eta={best_f1['efficiency_vs_srs']:.2f}")
    print(f"Best efficiency:  t={best_eta['threshold']:.2f}  "
          f"eta={best_eta['efficiency_vs_srs']:.2f}  "
          f"(f1={best_eta['change_f1']:.3f})")


if __name__ == "__main__":
    main()
