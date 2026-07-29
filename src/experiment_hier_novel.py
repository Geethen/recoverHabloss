"""Three noise-aware / data-efficient variants of the hierarchical-softmax net.

All three attack the established bottleneck -- Cropland/Nature interpreter noise
(analyse_label_noise.py) -- rather than model capacity, and all are measured
against the current best config (arch=wide, loss=focal, ~30 epochs) under the
same spatially blocked CV, on the merged2 deploy metric.

1. **Forward loss correction** (Patrini et al. 2017). The fine head predicts the
   *clean* posterior and is pushed through a Cropland<->Nature noise matrix ``T``
   to explain the noisy labels. ``noise_rate`` (the symmetric endpoint flip
   probability) is swept; the RECOVER reverifications put it near 0.3 on the
   contested subset, an upper bound because that subset is change-enriched.
2. **Bilinear endpoint head**. The transition logit factorises as
   ``from(A) + to(B) + bias``, so rare transitions borrow strength from common
   ones sharing an endpoint.
3. **Semi-supervised learning** (FixMatch-style, pseudo-label on the robust
   merged2 level). Measured by label masking: a fraction of training labels is
   hidden and the plots reused as unlabelled, so the question is whether the
   unlabelled signal recovers what the hidden labels would have given.

Writes two tables (variants, ssl label-masking) to analysis_results/. The
headline finding, recorded for honesty, is that none of the three beats the
plain wide|focal baseline by more than noise: the merged-legend restructuring
already removed the noise these methods target.
"""
from __future__ import annotations

import argparse
import json
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


def variant_table(frame, columns, target, groups, truth_fine, truth_merged, n_splits):
    configs = [
        ("baseline_flat", dict(**BASE)),
        ("bilinear", dict(**BASE, head="bilinear")),
        ("fwd_corr_0.1", dict(**BASE, noise_rate=0.1)),
        ("fwd_corr_0.2", dict(**BASE, noise_rate=0.2)),
        ("fwd_corr_0.3", dict(**BASE, noise_rate=0.3)),
        ("bilinear_fwd_0.2", dict(**BASE, head="bilinear", noise_rate=0.2)),
    ]
    rows = []
    for name, kw in configs:
        of, om, secs = evaluate_variant(name, kw, frame, columns, target,
                                        groups, n_splits)
        row = score_variant(name, of, om, truth_fine, truth_merged, secs)
        rows.append(row)
        print(f"  {name:18s} merged_chgF1={row['merged_change_f1']:.3f} "
              f"balAcc={row['merged_bal_acc']:.3f} fineF1={row['fine_change_f1']:.3f}",
              flush=True)
    return pd.DataFrame(rows)


def ssl_oof(frame, columns, target, groups, mode, frac, n_splits, seed=0):
    rng = np.random.default_rng(seed)
    pred = np.empty(len(target), dtype=object)
    for tr, te in make_splitter("blocked", n_splits).split(frame[columns], target, groups):
        if mode == "full":
            lab, unlab = tr, tr[:0]
        else:
            perm = rng.permutation(tr)
            cut = int(frac * len(tr))
            lab, unlab = perm[:cut], perm[cut:]
        model = HierarchicalSoftmaxNN(
            columns, **BASE, ssl=("fixmatch" if mode == "ssl" else None),
            ssl_threshold=0.9,
        )
        uf = frame.iloc[unlab] if mode == "ssl" and len(unlab) else None
        model.fit(frame.iloc[lab], target.iloc[lab].to_numpy(), unlabelled_frame=uf)
        pred[te] = model.predict_merged(frame.iloc[te])
    return pred


def ssl_table(frame, columns, target, groups, truth_merged, fractions, n_splits):
    rows = []
    full = ssl_oof(frame, columns, target, groups, "full", 1.0, n_splits)
    s = scores(truth_merged, full, is_change_label)
    rows.append({"regime": "full_supervised", "label_fraction": 1.0,
                 "merged_change_f1": round(s["change_f1"], 4),
                 "merged_bal_acc": round(s["balanced_accuracy"], 4)})
    print(f"  full-supervised            merged_chgF1={s['change_f1']:.3f}")
    for frac in fractions:
        for mode in ("sup", "ssl"):
            pred = ssl_oof(frame, columns, target, groups, mode, frac, n_splits)
            sc = scores(truth_merged, pred, is_change_label)
            rows.append({"regime": mode, "label_fraction": frac,
                         "merged_change_f1": round(sc["change_f1"], 4),
                         "merged_bal_acc": round(sc["balanced_accuracy"], 4)})
            print(f"  {int(frac*100)}% labels {mode:3s}          "
                  f"merged_chgF1={sc['change_f1']:.3f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--ssl-fractions", nargs="+", type=float,
                        default=[0.15, 0.30])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    print(f"{len(frame):,} plots | {len(columns)} features | base={BASE}")

    print("\n[1+3] forward correction + bilinear head:")
    variants = variant_table(frame, columns, target, groups, truth_fine,
                             truth_merged, args.n_splits)
    print("\n[SSL] label-masking:")
    ssl = ssl_table(frame, columns, target, groups, truth_merged,
                    args.ssl_fractions, args.n_splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    variants.to_csv(args.output_dir / f"hier_novel_variants{tag}.csv", index=False)
    ssl.to_csv(args.output_dir / f"hier_novel_ssl{tag}.csv", index=False)
    (args.output_dir / f"hier_novel_meta{tag}.json").write_text(
        json.dumps({"input": str(args.input), "base_config": BASE,
                    "n_splits": args.n_splits}, indent=2), encoding="utf-8")
    print("\nVariants:\n" + variants.to_string(index=False))
    print("\nSSL:\n" + ssl.to_string(index=False))


if __name__ == "__main__":
    main()
