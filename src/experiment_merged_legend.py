"""Test whether merging the noisy Cropland/Nature boundary lifts change-F1.

``analyse_label_noise.py`` shows that when RECOVER re-reads the same plot, the
disagreements are overwhelmingly Cropland vs Nature -- the fallow-cropland /
grassland boundary -- and the per-class report shows the model failing on
exactly those transitions (``Cropland -> Nature`` F1 ~0.07). If that boundary is
interpreter noise rather than signal, no feature or architecture recovers it,
but *removing it from the target* should: merge Cropland and Nature into a single
``Vegetation`` class and the change problem becomes the one the project is really
about -- vegetation lost to (or gained from) artificial surface, i.e. sealing.

This runs the same spatially blocked CV as the leaderboard under two legends:

* **coarse3** -- Nature / Cropland / Artificial (the current 9-transition target).
* **merged2** -- Vegetation (Nature + Cropland) / Artificial (a 4-transition
  target; the two changed classes are Veg<->Artificial).

For each legend it reports overall change-F1 plus per-transition F1, so the
question is answered directly: does collapsing the contested boundary move the
change metric, and by how much, on the same plots and folds.

This is an experiment, not the pipeline -- it does not touch ``model_zoo``'s
tested target construction; it rebuilds the target locally so the two legends
are the only thing that differs.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from model_zoo import (
    DEFAULT_INPUT,
    RARE_LABEL,
    build_registry,
    feature_columns,
    is_change_label,
    make_splitter,
    scores,
)
from project_paths import project_data_dir

# Two legends over the coarse classes. 'coarse3' is the identity; 'merged2'
# folds Cropland into the Vegetation super-class with Nature.
LEGENDS = {
    "coarse3": {"nature": "Nature", "cropland": "Cropland", "artificial": "Artificial",
                "nature - forest": "Nature", "nature - other": "Nature",
                "nature (not cropland / artificial)": "Nature"},
    "merged2": {"nature": "Vegetation", "cropland": "Vegetation",
                "artificial": "Artificial", "nature - forest": "Vegetation",
                "nature - other": "Vegetation",
                "nature (not cropland / artificial)": "Vegetation"},
}


def build_frame(path: Path):
    frame = pd.read_parquet(path)
    duplicated = int(frame["PLOTID"].duplicated().sum())
    if duplicated:
        frame = frame.drop_duplicates("PLOTID").reset_index(drop=True)
    columns = feature_columns(frame)
    complete = frame[columns].notna().all(axis=1)
    frame = frame.loc[complete].reset_index(drop=True)
    frame[columns] = frame[columns].astype("float64")
    return frame, columns


def target_for_legend(frame: pd.DataFrame, legend: dict, min_count: int) -> pd.Series:
    def code(series):
        cleaned = series.astype(str).str.strip().str.lower()
        unknown = sorted(set(cleaned) - set(legend))
        if unknown:
            raise ValueError(f"Values outside legend: {unknown}")
        return cleaned.map(legend)

    transition = code(frame["lc_2018"]) + " -> " + code(frame["lc_2024"])
    counts = transition.value_counts()
    keep = set(counts[counts >= min_count].index)
    return transition.where(transition.isin(keep), RARE_LABEL)


def run_legend(name, legend, frame, columns, groups, models, n_splits, min_count):
    target = target_for_legend(frame, legend, min_count)
    features = frame[columns]
    registry = build_registry(columns, balanced=True, hier_bases=["lda"])
    n_change = int(target.map(is_change_label).sum())
    print(f"\n=== {name}: {target.nunique()} classes, {n_change:,} change plots ===")

    rows, per_class = [], {}
    splitter = make_splitter("blocked", n_splits)
    for model in models:
        factory, needs_frame = registry[model]
        oof = np.empty(len(target), dtype=object)
        for tr, te in splitter.split(features, target, groups):
            est = factory()
            fit_X = frame.iloc[tr] if needs_frame else features.iloc[tr].to_numpy()
            te_X = frame.iloc[te] if needs_frame else features.iloc[te].to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(fit_X, target.iloc[tr].to_numpy())
                oof[te] = est.predict(te_X)
        s = scores(target.to_numpy(), oof, is_change_label)
        s.update(legend=name, model=model)
        rows.append(s)
        # per-transition F1, change classes only (where the boundary noise lives)
        labels = sorted(set(target) - {RARE_LABEL})
        f1s = f1_score(target, oof, labels=labels, average=None, zero_division=0)
        per_class[(name, model)] = {
            lab: round(float(v), 3)
            for lab, v in zip(labels, f1s) if is_change_label(lab)
        }
        print(f"  {model:10s} change_f1={s['change_f1']:.3f} "
              f"recall={s['change_recall']:.3f} prec={s['change_precision']:.3f} "
              f"bal_acc={s['balanced_accuracy']:.3f}")
    return rows, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=project_data_dir("analysis_results"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=["lda", "catboost", "hier_nn"])
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    print(f"{len(frame):,} plots | {len(columns)} features | {groups.nunique()} blocks")

    all_rows, all_per_class = [], {}
    for name, legend in LEGENDS.items():
        rows, per_class = run_legend(
            name, legend, frame, columns, groups, args.models,
            args.n_splits, args.min_class_count,
        )
        all_rows.extend(rows)
        all_per_class.update(per_class)

    board = pd.DataFrame(all_rows)[
        ["legend", "model", "change_f1", "change_recall", "change_precision",
         "balanced_accuracy", "accuracy", "f1_macro"]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    board.to_csv(args.output_dir / f"merged_legend{tag}.csv", index=False)
    (args.output_dir / f"merged_legend_perclass{tag}.json").write_text(
        json.dumps({f"{k[0]}|{k[1]}": v for k, v in all_per_class.items()}, indent=2),
        encoding="utf-8",
    )

    print("\n" + board.to_string(index=False))
    # Headline: the change-F1 delta from merging, per model.
    print("\nChange-F1 lift from merging Cropland into Nature:")
    for model in args.models:
        c = next((r for r in all_rows if r["legend"] == "coarse3" and r["model"] == model), None)
        m = next((r for r in all_rows if r["legend"] == "merged2" and r["model"] == model), None)
        if c and m:
            print(f"  {model:10s} {c['change_f1']:.3f} -> {m['change_f1']:.3f} "
                  f"({m['change_f1'] - c['change_f1']:+.3f})")


if __name__ == "__main__":
    main()
