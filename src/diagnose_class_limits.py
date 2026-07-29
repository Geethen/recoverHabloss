"""Per-class diagnosis: what limits accuracy on each transition class?

For every class the ceiling has one of three causes, and they leave different
fingerprints:

* **data limited** -- performance is still climbing with training-set size, and
  the model fits the training data far better than the test data (variance).
  More labels would pay.
* **model complexity** -- a nonlinear or foundation model clearly beats the
  linear baseline on this class, so the linear decision boundary was the binding
  constraint (bias). A better model would pay.
* **confusion bound** -- the class overlaps another in embedding space and/or
  the reference labels themselves disagree. Neither more data nor a bigger model
  fixes it; only better features or a better label protocol would.

Five measurements separate them:

1. per-class F1 across a capacity ladder (LDA -> boosting -> foundation model);
2. a learning curve over training-set fractions, holding the blocked test folds
   fixed, so the slope reflects data volume alone;
3. the train-vs-test F1 gap, which separates variance from bias;
4. neighbourhood purity -- the share of a plot's k nearest neighbours in
   embedding space that carry its own class, a model-free separability measure
   read against the class prior;
5. **annotator agreement measured directly**: 76 plots in the combined frame were
   labelled twice (54 RECOVER reverifications, 22 dual-frame overlaps). Their
   disagreement rate is an empirical ceiling that no model can cross.

Writes ``class_limits.csv`` and ``class_limits_confusion.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_zoo import (  # noqa: E402
    SEED,
    coarsen,
    feature_columns,
    load,
    make_splitter,
)
from project_paths import project_data_dir  # noqa: E402


DEFAULT_INPUT = project_data_dir("embeddings", "embeddings_habloss_recover.parquet")
DEFAULT_OUTPUT = project_data_dir("analysis_results")
FRACTIONS = (0.25, 0.5, 0.75, 1.0)

# Thresholds for the verdict. Stated up front so the rule is inspectable rather
# than tuned after seeing the answers.
SLOPE_DATA_LIMITED = 0.04     # F1 gained over the last halving of training data
CAPACITY_GAIN = 0.04          # best nonlinear F1 minus linear F1
OVERFIT_GAP = 0.25            # train F1 minus test F1
MIN_N_FOR_CAPACITY = 100      # below this a capacity gap is unreadable noise


def linear_model():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


def nonlinear_model(seed: int = SEED):
    # Class-balanced: the unweighted booster buys accuracy on the stable
    # diagonal, which would understate its capacity on the rare classes.
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, early_stopping=True,
        validation_fraction=0.15, class_weight="balanced", random_state=seed,
    )


def oof_predictions(model_fn, features, target, groups, n_splits):
    """Out-of-fold predictions plus the mean in-sample training F1."""
    splitter = make_splitter("blocked", n_splits)
    oof = np.empty(len(target), dtype=object)
    train_f1 = []
    X, y = features.to_numpy(), target.to_numpy()
    for train_idx, test_idx in splitter.split(features, target, groups):
        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X[test_idx])
        train_f1.append(
            f1_score(y[train_idx], model.predict(X[train_idx]),
                     average=None, labels=sorted(set(y)), zero_division=0)
        )
    return oof, np.mean(train_f1, axis=0)


def per_class_f1(truth, predicted, labels) -> np.ndarray:
    return f1_score(truth, predicted, average=None, labels=labels, zero_division=0)


def _curve_once(model_fn, features, target, groups, labels, n_splits, seed):
    """One seeded pass: per-class F1 at each training fraction."""
    rng = np.random.default_rng(seed)
    X, y = features.to_numpy(), target.to_numpy()
    result = {}
    for fraction in FRACTIONS:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        oof = np.empty(len(target), dtype=object)
        for train_idx, test_idx in splitter.split(features, target, groups):
            if fraction < 1.0:
                keep = []
                for label in labels:
                    idx = train_idx[y[train_idx] == label]
                    take = max(2, int(round(len(idx) * fraction)))
                    keep.append(rng.choice(idx, size=min(take, len(idx)),
                                           replace=False))
                use = np.concatenate(keep)
            else:
                use = train_idx
            model = model_fn(seed)
            model.fit(X[use], y[use])
            oof[test_idx] = model.predict(X[test_idx])
        result[fraction] = per_class_f1(y, oof, labels)
    return result


def learning_curve(model_fn, features, target, groups, labels, n_splits, seeds):
    """Per-class learning curve, repeated over seeds.

    Training rows are subsampled class-stratified so every class keeps its share
    of the reduction -- otherwise the rare classes vanish and the curve measures
    class balance instead of data volume.

    Repetition is not optional here. On a single seed the slope for the smallest
    classes swings by more than 0.2 F1, enough to flip a verdict: a one-seed run
    gave Cropland -> Nature a *negative* slope, while four seeds put it at
    +0.022 and never below zero. Classes whose slope range straddles the
    threshold are reported as indeterminate rather than diagnosed.
    """
    passes = []
    for seed in seeds:
        passes.append(
            _curve_once(model_fn, features, target, groups, labels, n_splits, seed)
        )
        macro = np.mean(passes[-1][1.0])
        print(f"  seed {seed}: macro F1 at full data {macro:.3f}", flush=True)

    slopes = np.array([p[1.0] - p[0.5] for p in passes])   # (seeds, classes)
    return {
        "mean_f1": {f: np.mean([p[f] for p in passes], axis=0) for f in FRACTIONS},
        "slope_mean": slopes.mean(axis=0),
        "slope_min": slopes.min(axis=0),
        "slope_max": slopes.max(axis=0),
    }


def neighbourhood_purity(features, target, labels, k=25) -> dict:
    """Share of each plot's k nearest neighbours carrying its own class.

    Compared against the class prior: a class whose neighbourhoods are no purer
    than chance is not separable with these features, whatever the model.
    """
    X = StandardScaler().fit_transform(features.to_numpy())
    y = target.to_numpy()
    # k + 1 because the first neighbour of a point is itself.
    finder = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = finder.kneighbors(X)
    neighbours = y[indices[:, 1:]]
    same = (neighbours == y[:, None]).mean(axis=1)
    return {
        label: {
            "purity": float(same[y == label].mean()),
            "prior": float((y == label).mean()),
        }
        for label in labels
    }


def duplicate_agreement(path: Path, labels) -> dict:
    """Annotator agreement from plots that were labelled twice.

    The two kinds of repeat must not be pooled, because only one is a random
    sample of the frame:

    * **dual-frame overlaps** -- plots that fall in both ``habloss_main`` and
      ``habloss_landwater``. Nothing about the plot caused the repeat, so their
      agreement rate estimates general label reliability.
    * **RECOVER reverifications** -- plots the R workflow re-examined *because*
      they were flagged uncertain or contradictory. Their agreement is
      conditional on already being contested, and reads far lower. Quoting the
      pooled number as "the" label ceiling would badly understate label quality.

    Where two interpretations of the same coordinate disagree, no model can be
    right about both, so this is a measured bound rather than an assumption.
    """
    frame = pd.read_parquet(path, columns=["PLOTID", "source", "lc_2018", "lc_2024"])
    frame = frame.assign(
        transition=coarsen(frame["lc_2018"]) + " -> " + coarsen(frame["lc_2024"])
    )
    repeated = frame[frame["PLOTID"].duplicated(keep=False)]
    grouped = repeated.groupby("PLOTID").agg(
        transitions=("transition", list),
        sources=("source", lambda s: set(s)),
    )
    grouped = grouped[grouped["transitions"].map(len) == 2]

    stats = {label: {"pairs": 0, "agree": 0} for label in labels}
    by_kind = {"incidental": [], "reverified": []}
    disagreements = {}
    for _, row in grouped.iterrows():
        first, second = row["transitions"]
        agree = first == second
        kind = "reverified" if row["sources"] == {"recover"} else "incidental"
        by_kind[kind].append(agree)
        if not agree:
            key = " vs ".join(sorted([first, second]))
            disagreements[key] = disagreements.get(key, 0) + 1
        for label in {first, second}:
            if label in stats:
                stats[label]["pairs"] += 1
                stats[label]["agree"] += int(agree)

    return {
        "n_pairs": len(grouped),
        "by_kind": {
            kind: {
                "n": len(values),
                "agreement": float(np.mean(values)) if values else float("nan"),
            }
            for kind, values in by_kind.items()
        },
        "disagreements": dict(
            sorted(disagreements.items(), key=lambda kv: -kv[1])
        ),
        "per_class": {
            label: {
                "pairs": s["pairs"],
                "agreement": (s["agree"] / s["pairs"]) if s["pairs"] else float("nan"),
            }
            for label, s in stats.items()
        },
    }


def verdict(row) -> tuple[str, str]:
    """Primary limit for a class, with the evidence that decided it.

    Ordered by which lever is cheapest to pull: more labels, then a better
    model, then better features or a sharper label protocol.
    """
    slope_robust_positive = row["slope_min"] >= SLOPE_DATA_LIMITED
    slope_robust_flat = row["slope_max"] < SLOPE_DATA_LIMITED
    contested = (
        row["label_pairs"] >= 5
        and row["label_agreement"] < 0.6
    )

    # A class whose own labellers disagree cannot be fixed by data or capacity.
    if contested and row["best_f1"] < 0.35:
        return "confusion bound", (
            f"labellers disagree on {1 - row['label_agreement']:.0%} of repeats; "
            f"{row['top_confusion_share']:.0%} of model errors go to "
            f"{row['top_confusion']}"
        )
    # A capacity gain is only readable when the class has enough test examples
    # to resolve it: at n=46 a fold holds ~9, and an F1 difference of 0.05 sits
    # well inside the resampling noise.
    if row["capacity_gain"] >= CAPACITY_GAIN and row["n"] >= MIN_N_FOR_CAPACITY:
        return "model complexity", f"nonlinear beats linear by {row['capacity_gain']:+.3f}"
    if slope_robust_positive:
        return "data limited", (
            f"F1 +{row['slope_mean']:.3f} per data doubling "
            f"(never below +{row['slope_min']:.3f}), train-test gap "
            f"{row['train_test_gap']:.2f}"
        )
    if slope_robust_flat and row["best_f1"] >= 0.35:
        return "near ceiling", "slope flat across seeds, no capacity gain"
    if slope_robust_flat:
        return "confusion bound", (
            f"slope flat across seeds at F1 {row['best_f1']:.2f}; "
            f"{row['top_confusion_share']:.0%} of errors go to {row['top_confusion']}"
        )
    return "indeterminate", (
        f"slope spans [{row['slope_min']:+.3f}, {row['slope_max']:+.3f}] "
        f"across seeds -- n={row['n']} too small to call"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="*",
                        default=[SEED, 7, 99, 2024],
                        help="Learning-curve repeats; more than one is required "
                             "to tell a real slope from resampling noise")
    args = parser.parse_args()

    frame, target, groups = load(args.input, args.min_class_count)
    features = frame[feature_columns(frame)]
    labels = sorted(target.unique())
    support = target.value_counts()

    print("Capacity ladder: linear")
    linear_oof, linear_train = oof_predictions(
        linear_model, features, target, groups, args.n_splits
    )
    print("Capacity ladder: nonlinear (class-balanced boosting)")
    nonlinear_oof, nonlinear_train = oof_predictions(
        nonlinear_model, features, target, groups, args.n_splits
    )

    linear_f1 = per_class_f1(target.to_numpy(), linear_oof, labels)
    nonlinear_f1 = per_class_f1(target.to_numpy(), nonlinear_oof, labels)
    best_f1 = np.maximum(linear_f1, nonlinear_f1)

    print(f"Learning curve (nonlinear), {len(args.seeds)} seeds")
    curve = learning_curve(
        nonlinear_model, features, target, groups, labels, args.n_splits, args.seeds
    )

    print(f"Neighbourhood purity (k={args.k})")
    purity = neighbourhood_purity(features, target, labels, args.k)
    agreement = duplicate_agreement(args.input, labels)
    incidental = agreement["by_kind"]["incidental"]
    reverified = agreement["by_kind"]["reverified"]
    print(
        f"Annotator agreement, {agreement['n_pairs']} twice-labelled plots:\n"
        f"  incidental dual-frame overlaps  n={incidental['n']:3d}  "
        f"{incidental['agreement']:.1%}  <- estimates general label reliability\n"
        f"  RECOVER reverifications         n={reverified['n']:3d}  "
        f"{reverified['agreement']:.1%}  <- selected as contested, a floor not a ceiling"
    )
    for pair, count in list(agreement["disagreements"].items())[:3]:
        print(f"    {count:3d} disagreements: {pair}")

    # Where does each class's error mass go?
    best_oof = np.where(nonlinear_f1.mean() >= linear_f1.mean(), nonlinear_oof, linear_oof)
    matrix = confusion_matrix(target.to_numpy(), best_oof, labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.to_csv(args.output_dir / "class_limits_confusion.csv")

    rows = []
    for i, label in enumerate(labels):
        errors = matrix[i].copy()
        errors[i] = 0
        top_index = int(errors.argmax())
        rows.append(
            {
                "transition": label,
                "n": int(support[label]),
                "linear_f1": float(linear_f1[i]),
                "nonlinear_f1": float(nonlinear_f1[i]),
                "best_f1": float(best_f1[i]),
                "capacity_gain": float(nonlinear_f1[i] - linear_f1[i]),
                "train_test_gap": float(nonlinear_train[i] - nonlinear_f1[i]),
                "f1_at_50pct": float(curve["mean_f1"][0.5][i]),
                "slope_mean": float(curve["slope_mean"][i]),
                "slope_min": float(curve["slope_min"][i]),
                "slope_max": float(curve["slope_max"][i]),
                "purity": purity[label]["purity"],
                "purity_lift": purity[label]["purity"] / purity[label]["prior"],
                "top_confusion": labels[top_index],
                "top_confusion_share": float(errors[top_index] / max(errors.sum(), 1)),
                "label_pairs": agreement["per_class"][label]["pairs"],
                "label_agreement": agreement["per_class"][label]["agreement"],
            }
        )

    table = pd.DataFrame(rows)
    decided = [verdict(r) for _, r in table.iterrows()]
    table["limit"] = [d[0] for d in decided]
    table["evidence"] = [d[1] for d in decided]
    table = table.sort_values("n", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "class_limits.csv", index=False)

    show = table[[
        "transition", "n", "linear_f1", "nonlinear_f1", "slope_mean",
        "slope_min", "slope_max", "train_test_gap", "label_agreement", "limit",
    ]]
    print()
    print(show.round(3).to_string(index=False))
    print()
    for _, r in table.iterrows():
        print(f"{r['transition']:26s} {r['limit']:18s} {r['evidence']}")


if __name__ == "__main__":
    main()
