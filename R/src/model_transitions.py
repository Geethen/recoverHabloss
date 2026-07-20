"""Map RECOVER change transitions directly from AlphaEarth embeddings.

The target is the interpreted transition ``lc_2018 -> lc_2024``. Because plots
are spatially clustered, a random split leaks neighbouring plots between train
and test and reports an optimistic score. Folds are therefore grouped on the
spatial block assigned during extraction, so every plot in a block is held out
together.

Rare transitions are collapsed into an ``other`` class: a transition seen only
a handful of times cannot be estimated, and leaving it in produces folds where
the class is absent from training.

Outputs a per-fold score table, a pooled out-of-fold confusion matrix, and
per-class precision/recall to ``data/analysis_results/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path("data/embeddings/embeddings_labelled.parquet")
DEFAULT_OUTPUT = Path("data/analysis_results")
DEFAULT_GROUP_COLUMN = "block_id"
RARE_LABEL = "other"


def feature_columns(frame: pd.DataFrame, use_diff: bool = True) -> list[str]:
    """Embedding feature columns, optionally excluding the difference bands."""
    columns = [c for c in frame.columns if c.startswith("A") and "_" in c]
    if not use_diff:
        columns = [c for c in columns if not c.endswith("_diff")]
    if not columns:
        raise ValueError("No embedding feature columns found (expected 'A00_2018' style names)")
    return sorted(columns)


def build_target(
    frame: pd.DataFrame, min_count: int
) -> tuple[pd.Series, pd.DataFrame]:
    """Transition label per plot, with rare transitions pooled into 'other'."""
    for column in ("lc_2018", "lc_2024"):
        if column not in frame:
            raise ValueError(
                f"Column '{column}' missing. The extraction must carry the "
                "interpreted transition labels."
            )
    raw = (
        frame["lc_2018"].astype(str).str.strip()
        + " -> "
        + frame["lc_2024"].astype(str).str.strip()
    )
    counts = raw.value_counts()
    keep = set(counts[counts >= min_count].index)
    target = raw.where(raw.isin(keep), RARE_LABEL)
    summary = (
        counts.rename_axis("transition")
        .reset_index(name="n")
        .assign(retained=lambda d: d["transition"].isin(keep))
    )
    return target, summary


def make_model(kind: str, seed: int):
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )
    if kind == "logistic":
        # Embeddings are dense and already well scaled; a linear probe is the
        # honest baseline for how much of the transition is linearly encoded.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
        )
    raise ValueError(f"Unknown model kind: {kind}")


def spatial_cv(
    features: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    kind: str,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Grouped stratified CV; returns fold scores and out-of-fold predictions."""
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(
            f"Only {n_groups} spatial groups available for {n_splits} folds. "
            "Reduce --n-splits or use a smaller --block-size at extraction."
        )
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    oof = np.empty(len(target), dtype=object)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(features, target, groups), start=1
    ):
        model = make_model(kind, seed)
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        predicted = model.predict(features.iloc[test_idx])
        oof[test_idx] = predicted
        truth = target.iloc[test_idx]
        rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_blocks": groups.iloc[test_idx].nunique(),
                "accuracy": accuracy_score(truth, predicted),
                "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                "f1_macro": f1_score(truth, predicted, average="macro", zero_division=0),
            }
        )
        print(
            f"Fold {fold}: acc={rows[-1]['accuracy']:.3f} "
            f"balanced={rows[-1]['balanced_accuracy']:.3f} "
            f"f1_macro={rows[-1]['f1_macro']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows), oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--model", choices=["gbm", "logistic"], default="gbm")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20250717)
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=20,
        help="Transitions rarer than this are pooled into 'other'",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Exclude the 2024-2018 difference bands from the features",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    if args.group_column not in frame:
        raise ValueError(
            f"Group column '{args.group_column}' not in {args.input}. "
            f"Available: {sorted(frame.columns)[:20]}"
        )

    columns = feature_columns(frame, use_diff=not args.no_diff)
    target, class_summary = build_target(frame, args.min_class_count)

    # Plots whose embedding lookup returned nothing cannot be modelled.
    complete = frame[columns].notna().all(axis=1)
    if not complete.all():
        print(f"Dropping {int((~complete).sum()):,} plots with missing embeddings")
    frame = frame.loc[complete].reset_index(drop=True)
    target = target.loc[complete].reset_index(drop=True)

    features = frame[columns]
    groups = frame[args.group_column]
    print(
        f"{len(frame):,} plots | {len(columns)} features | "
        f"{target.nunique()} classes | {groups.nunique()} spatial blocks"
    )

    scores, oof = spatial_cv(
        features, target, groups, args.model, args.n_splits, args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "transition_cv_folds.csv", index=False)
    class_summary.to_csv(
        args.output_dir / "transition_class_counts.csv", index=False
    )

    labels = sorted(target.unique())
    matrix = pd.DataFrame(
        confusion_matrix(target, oof, labels=labels),
        index=pd.Index(labels, name="observed"),
        columns=pd.Index(labels, name="predicted"),
    )
    matrix.to_csv(args.output_dir / "transition_confusion.csv")

    report = pd.DataFrame(
        classification_report(
            target, oof, labels=labels, output_dict=True, zero_division=0
        )
    ).transpose()
    report.to_csv(args.output_dir / "transition_class_report.csv")

    summary = {
        "input": str(args.input),
        "model": args.model,
        "n_plots": len(frame),
        "n_features": len(columns),
        "use_difference_bands": not args.no_diff,
        "group_column": args.group_column,
        "n_blocks": int(groups.nunique()),
        "n_splits": args.n_splits,
        "min_class_count": args.min_class_count,
        "classes": labels,
        "oof_accuracy": float(accuracy_score(target, oof)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(target, oof)),
        "oof_f1_macro": float(
            f1_score(target, oof, average="macro", zero_division=0)
        ),
        "fold_accuracy_mean": float(scores["accuracy"].mean()),
        "fold_accuracy_sd": float(scores["accuracy"].std(ddof=1)),
    }
    (args.output_dir / "transition_model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        f"Out-of-fold accuracy {summary['oof_accuracy']:.3f} | "
        f"balanced {summary['oof_balanced_accuracy']:.3f} | "
        f"macro F1 {summary['oof_f1_macro']:.3f}"
    )


if __name__ == "__main__":
    main()
