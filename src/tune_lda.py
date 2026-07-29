"""Tune the LDA transition model -- direct and post-classification -- for F1.

The leaderboard in ``model_zoo.py`` runs LDA with scikit-learn defaults: the SVD
solver, empirical priors, no shrinkage. That is a reasonable baseline but leaves
LDA's one real regularisation lever -- covariance shrinkage -- switched off, and
with 192 correlated embedding channels and a heavy class imbalance that lever is
exactly what F1 responds to.

This script grid-searches the levers that matter under the *same* spatially
blocked CV and the *same* legend/dedup as the leaderboard, so the tuned numbers
drop straight into the existing comparison:

* ``solver`` -- ``svd`` (no shrinkage possible), ``lsqr`` and ``eigen`` (both
  admit shrinkage). Prediction is identical across solvers only when shrinkage
  is off; the point of the sweep is the shrinkage the SVD path cannot take.
* ``shrinkage`` -- ``None``, Ledoit-Wolf ``"auto"``, and a manual grid. Shrinks
  the per-class covariance toward a scaled identity, which is what stabilises
  the discriminant directions when p is large relative to the per-class n.
* ``priors`` -- empirical vs uniform. Uniform priors stop the stable-diagonal
  majority from swamping the rare change classes, which is where macro-F1 and
  change-F1 are won or lost.

Two F1s are reported because the project cares about both: ``f1_macro`` over the
full 7-way transition label, and ``change_f1`` over the collapsed changed/stable
indicator that the area estimators actually consume. The winner is chosen on the
metric passed to ``--optimise`` (default ``change_f1``).

Usage::

    python tune_lda.py                        # both framings, pick on change_f1
    python tune_lda.py --optimise f1_macro
    python tune_lda.py --framing direct       # only the direct 7-way model
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_zoo import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    PostClassification,
    RARE_LABEL,
    TunedLDA,
    evaluate,
    feature_columns,
    load,
)


def is_change(label: str) -> bool:
    return label == RARE_LABEL or label.split(" -> ")[0] != label.split(" -> ")[1]


def lda_configs() -> list[dict]:
    """Every LDA setting the sweep tries, as kwargs to ``LinearDiscriminantAnalysis``.

    ``svd`` cannot take a shrinkage argument at all, so it appears once with the
    default covariance; ``lsqr``/``eigen`` carry the shrinkage grid. Priors are
    crossed over every solver.
    """
    configs: list[dict] = []
    priors = [None, "uniform"]
    shrinkage_grid = [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9]

    for prior in priors:
        # SVD: no shrinkage knob, but the fastest and the current default path.
        configs.append({"solver": "svd", "shrinkage": None, "priors": prior})
        for solver in ("lsqr", "eigen"):
            for shrink in shrinkage_grid:
                configs.append(
                    {"solver": solver, "shrinkage": shrink, "priors": prior}
                )
    return configs


def make_lda(config: dict):
    """A scaled LDA estimator for the given config.

    Uses ``TunedLDA`` (shared with ``model_zoo.py``), which resolves
    ``prior_mode="uniform"`` against the classes at fit time -- LDA itself wants
    an explicit prior array, and the class count is not known until fit.
    """
    prior = config["priors"]
    prior_mode = "uniform" if prior == "uniform" else None

    def factory():
        est = TunedLDA(
            solver=config["solver"],
            shrinkage=config["shrinkage"],
            prior_mode=prior_mode,
        )
        return make_pipeline(StandardScaler(), est)

    return factory


def label(config: dict) -> str:
    prior = config["priors"] or "empirical"
    shrink = config["shrinkage"]
    shrink = "none" if shrink is None else shrink
    return f"{config['solver']}|shrink={shrink}|priors={prior}"


def sweep(
    framing: str,
    features: pd.DataFrame,
    frame: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    columns: list[str],
    n_splits: int,
    cv: str,
    postclass_shared: bool,
) -> pd.DataFrame:
    rows = []
    for config in lda_configs():
        base_factory = make_lda(config)
        if framing == "direct":
            factory, needs_frame = base_factory, False
        else:
            factory = lambda bf=base_factory: PostClassification(
                columns, bf, shared=postclass_shared
            )
            needs_frame = True
        started = time.time()
        try:
            row, _ = evaluate(
                label(config), factory, features, frame, target, groups,
                n_splits, is_change, needs_frame, cv,
            )
        except Exception as error:  # a broken config must not sink the sweep
            print(f"  {label(config):40s} FAILED {type(error).__name__}: {error}")
            continue
        row["framing"] = framing
        row["config"] = label(config)
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        print(
            f"  {label(config):40s} f1_macro={row['f1_macro']:.4f} "
            f"change_f1={row['change_f1']:.4f} "
            f"bal_acc={row['balanced_accuracy']:.4f} ({row['seconds']}s)",
            flush=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv", choices=["blocked", "random"], default="blocked")
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument(
        "--framing",
        choices=["direct", "postclass", "both"],
        default="both",
    )
    parser.add_argument("--postclass-per-date", action="store_true")
    parser.add_argument(
        "--optimise",
        choices=["change_f1", "f1_macro"],
        default="change_f1",
        help="Metric the winner is chosen on",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, target, groups = load(args.input, args.min_class_count)
    columns = feature_columns(frame)
    features = frame[columns]

    print(
        f"{len(frame):,} plots | {len(columns)} features | "
        f"{target.nunique()} classes | {groups.nunique()} blocks | "
        f"{int(sum(is_change(t) for t in target)):,} change plots | "
        f"optimise={args.optimise}"
    )

    framings = ["direct", "postclass"] if args.framing == "both" else [args.framing]
    tables = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for framing in framings:
            print(f"\n=== {framing} LDA ({len(lda_configs())} configs) ===")
            tables.append(
                sweep(
                    framing, features, frame, target, groups, columns,
                    args.n_splits, args.cv, not args.postclass_per_date,
                )
            )
    board = pd.concat(tables, ignore_index=True)
    board = board.sort_values(args.optimise, ascending=False).reset_index(drop=True)

    ordered = ["framing", "config", "f1_macro", "change_f1", "balanced_accuracy",
               "accuracy", "change_recall", "change_precision", "seconds"]
    board = board[[c for c in ordered if c in board.columns]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    board.to_csv(args.output_dir / f"tune_lda{tag}.csv", index=False)

    winners = {}
    for framing in framings:
        sub = board[board["framing"] == framing]
        best = sub.iloc[0]
        winners[framing] = {
            "config": best["config"],
            args.optimise: float(best[args.optimise]),
            "f1_macro": float(best["f1_macro"]),
            "change_f1": float(best["change_f1"]),
        }
    (args.output_dir / f"tune_lda_winners{tag}.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "cv": args.cv,
                "n_splits": args.n_splits,
                "optimise": args.optimise,
                "winners": winners,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n=== leaderboard (sorted by {args.optimise}) ===")
    print(board.to_string(index=False))
    print("\n=== winners ===")
    for framing, win in winners.items():
        print(f"{framing:10s} {win['config']:40s} "
              f"f1_macro={win['f1_macro']:.4f} change_f1={win['change_f1']:.4f}")


if __name__ == "__main__":
    main()
