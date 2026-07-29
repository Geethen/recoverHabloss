"""Do embedding-based change features beat the raw difference bands?

The AlphaEarth vectors are unit-norm, so the 64 endpoint difference bands
(``A??_diff = t2024 - t2018``) already encode the cosine-distance signal:
``||diff||^2 = 2 - 2 * cos(t1, t2)``. A scalar cosine distance is therefore a
*compression* of the diff bands that keeps change magnitude but throws away
change direction -- and direction is what separates transition types
(forest->urban vs forest->crop). This script tests that intuition by scoring
several feature sets through the identical grouped spatial CV used by
``model_transitions``:

    per_year    A??_2018 + A??_2024                (no change features)
    diff        + the 64 raw difference bands      (current default)
    cosine      + a single cosine-distance scalar
    scalars     + cos-dist, L1, Chebyshev, norm change
    diff+cos    + diff bands and the scalar together

Outputs a ranked table to ``data/analysis_results/change_feature_experiment.csv``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from model_transitions import (
    DEFAULT_GROUP_COLUMN,
    DEFAULT_INPUT,
    build_target,
    spatial_cv,
)
from project_paths import project_data_dir

EMBEDDING_DIM = 64
DEFAULT_OUTPUT = project_data_dir("analysis_results")


def year_columns(frame: pd.DataFrame, year: int) -> list[str]:
    return [f"A{i:02d}_{year}" for i in range(EMBEDDING_DIM)]


def change_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compact change descriptors derived from the two per-year embeddings.

    Everything here is computed locally from bands already in the parquet, so
    no re-extraction is needed. Because the vectors are unit-norm, cosine
    distance and Euclidean distance are monotone with each other; only cosine
    distance is kept, alongside two norms that carry genuinely different
    information about *how* the change is distributed across dimensions.
    """
    t1 = frame[year_columns(frame, 2018)].to_numpy(dtype=float)
    t2 = frame[year_columns(frame, 2024)].to_numpy(dtype=float)
    diff = t2 - t1
    n1 = np.linalg.norm(t1, axis=1)
    n2 = np.linalg.norm(t2, axis=1)
    cos_sim = (t1 * t2).sum(axis=1) / np.clip(n1 * n2, 1e-12, None)
    return pd.DataFrame(
        {
            # 1 - cos: the canonical AlphaEarth change-magnitude metric.
            "cos_dist": 1.0 - cos_sim,
            # L1 and Chebyshev see change spread differently than L2/cosine:
            # a change concentrated in one dimension vs smeared across many.
            "l1": np.abs(diff).sum(axis=1),
            "chebyshev": np.abs(diff).max(axis=1),
            # Did the vector move toward/away from the origin at all (norms are
            # ~1 but not exactly, so this is a small honest signal).
            "norm_change": n2 - n1,
        },
        index=frame.index,
    )


def diff_columns(frame: pd.DataFrame) -> list[str]:
    return [f"A{i:02d}_diff" for i in range(EMBEDDING_DIM)]


def dot_columns() -> list[str]:
    return [f"A{i:02d}_dot" for i in range(EMBEDDING_DIM)]


def nd_columns() -> list[str]:
    return [f"A{i:02d}_nd" for i in range(EMBEDDING_DIM)]


def dot_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-band product ``t2018_i * t2024_i`` (64 bands).

    The elementwise product keeps the *direction* of agreement per dimension
    that the scalar cosine similarity (its sum) throws away: a dimension that
    stays strongly positive across both years and one that flips sign both move
    the sum, but only the per-band form tells the net which dimension did it.
    """
    t1 = frame[year_columns(frame, 2018)].to_numpy(dtype=float)
    t2 = frame[year_columns(frame, 2024)].to_numpy(dtype=float)
    return pd.DataFrame(t1 * t2, columns=dot_columns(), index=frame.index)


def nd_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-band normalised difference ``(t2024_i - t2018_i) / (|t2018_i| + |t2024_i|)``.

    NDVI-style contrast, bounded in [-1, 1]. The AlphaEarth dimensions are
    signed and cross zero, so the textbook ``(t2 - t1) / (t2 + t1)`` is unstable
    (the denominator vanishes when the two years are opposite); normalising by
    the sum of magnitudes keeps the same scale-free contrast without blowing up.
    A dimension unchanged in both years contributes 0.
    """
    t1 = frame[year_columns(frame, 2018)].to_numpy(dtype=float)
    t2 = frame[year_columns(frame, 2024)].to_numpy(dtype=float)
    denom = np.clip(np.abs(t1) + np.abs(t2), 1e-6, None)
    return pd.DataFrame((t2 - t1) / denom, columns=nd_columns(), index=frame.index)


def build_matrix(frame: pd.DataFrame, kind: str, change: pd.DataFrame) -> pd.DataFrame:
    per_year = year_columns(frame, 2018) + year_columns(frame, 2024)
    base = frame[per_year]
    if kind == "per_year":
        return base
    if kind == "diff":
        return pd.concat([base, frame[diff_columns(frame)]], axis=1)
    if kind == "cosine":
        return pd.concat([base, change[["cos_dist"]]], axis=1)
    if kind == "scalars":
        return pd.concat([base, change], axis=1)
    if kind == "diff+cos":
        return pd.concat([base, frame[diff_columns(frame)], change[["cos_dist"]]], axis=1)
    raise ValueError(f"Unknown feature set: {kind}")


FEATURE_SETS = ["per_year", "diff", "cosine", "scalars", "diff+cos"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--model", choices=["gbm", "logistic"], default="gbm")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20250717)
    parser.add_argument("--min-class-count", type=int, default=20)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)

    labelled = frame[["lc_2018", "lc_2024"]].notna().all(axis=1)
    if not labelled.all():
        print(f"Dropping {int((~labelled).sum()):,} plots with no interpreted transition")
        frame = frame.loc[labelled].reset_index(drop=True)

    # Keep only plots whose full embedding pair is present, so every feature set
    # is scored on exactly the same plots.
    needed = year_columns(frame, 2018) + year_columns(frame, 2024) + diff_columns(frame)
    complete = frame[needed].notna().all(axis=1)
    if not complete.all():
        print(f"Dropping {int((~complete).sum()):,} plots with missing embeddings")
    frame = frame.loc[complete].reset_index(drop=True)

    target, _ = build_target(frame, args.min_class_count)
    groups = frame[args.group_column]
    change = change_features(frame)
    print(
        f"{len(frame):,} plots | {target.nunique()} classes | "
        f"{groups.nunique()} spatial blocks | model={args.model}"
    )

    rows = []
    for kind in FEATURE_SETS:
        features = build_matrix(frame, kind, change)
        print(f"\n=== {kind}  ({features.shape[1]} features) ===", flush=True)
        _, oof = spatial_cv(
            features, target, groups, args.model, args.n_splits, args.seed
        )
        rows.append(
            {
                "feature_set": kind,
                "n_features": features.shape[1],
                "oof_accuracy": accuracy_score(target, oof),
                "oof_balanced_accuracy": balanced_accuracy_score(target, oof),
                "oof_f1_macro": f1_score(target, oof, average="macro", zero_division=0),
            }
        )

    table = pd.DataFrame(rows).sort_values("oof_f1_macro", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "change_feature_experiment.csv"
    table.to_csv(out, index=False)
    print("\n" + table.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
