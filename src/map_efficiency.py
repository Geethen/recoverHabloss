"""Relative efficiency of each model-as-map for design-based area estimation.

Accuracy answers "is this map right?". Efficiency answers the question the
estimator actually asks: **if this map were used to stratify the sample, how
much variance would it remove from the area estimate of class k?** A map can be
mediocre by F1 and still be an excellent stratifier, and vice versa -- what
matters is whether it splits the population into strata that are internally
homogeneous with respect to the class being estimated.

Under Neyman allocation the variance of the stratified estimator of the area
proportion of class ``k`` is ``(sum_i W_i S_i)^2 / n``, so following Gallego
(2007) the efficiency of map A relative to map B is the ratio of those squared
terms, B over A::

    eta^k_{A,B} = (sum_i W_i^B S_i^B)^2 / (sum_i W_i^A S_i^A)^2

with ``W_i`` the areal weight of map stratum ``i`` and ``S_i = sqrt(p_ik
(1 - p_ik))`` the within-stratum standard deviation of the indicator for
reference class ``k``. ``eta > 1`` means A is the better stratifier: at fixed
``n`` the variance falls by ``eta``, or equivalently the sample size needed to
hit a target variance falls by the same factor. The per-stratum cost ratio
``v_i / u_i`` in the published form is taken as 1 here -- every stratum costs
the same to sample, and both maps are allocated by the same rule -- so the
comparison is purely about stratum homogeneity.

The default baseline B is **no map at all**: one stratum, ``S = sqrt(p_k
(1 - p_k))``, i.e. simple random sampling. That makes the headline number read
directly as "this map cuts the required sample size by a factor of eta".

**A map beating SRS is not a finding.** ``sqrt(p (1 - p))`` is concave, so by
Jensen ``sum_i W_i S_i <= S`` for *any* partition whatsoever: under Neyman
allocation eta >= 1 is a theorem, not evidence. A map of random noise scores
1.0, and nothing can score below it. The informative content is entirely in the
*ranking* -- which map removes more variance than which -- and in how far above
1 a map gets. It also assumes Neyman allocation, which needs the ``S_i`` known
in advance; proportional allocation realises less of the gain.

Three further caveats, all of them real:

* The "map" is the out-of-fold prediction at the sample plots, not a wall-to-wall
  product. ``W_i`` is therefore the plot share of each predicted class, not a
  pixel count, and it inherits every bias in the sample design.
* The sample pools three designs whose weights the README warns against pooling.
  These weights are unweighted plot counts, so eta is a *relative* diagnostic
  between models on a common footing, not a design-based quantity to publish.
* ``p_ik`` is estimated from the plots in stratum ``i``. Thin strata give noisy
  ``S_i``, which flatters maps that produce small confident strata; the
  per-stratum counts are written out so this can be checked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_zoo import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    RARE_LABEL,
    build_registry,
    evaluate,
    feature_columns,
    load,
)

CHANGE_KEY = "any change"


def is_change(label: str) -> bool:
    return label == RARE_LABEL or label.split(" -> ")[0] != label.split(" -> ")[1]


def indicator(labels: np.ndarray, klass: str) -> np.ndarray:
    """Binary reference indicator for class k, or for change of any kind."""
    if klass == CHANGE_KEY:
        return np.array([is_change(v) for v in labels], dtype=float)
    return (labels == klass).astype(float)


def stratified_term(strata: np.ndarray, y: np.ndarray) -> float:
    """``sum_i W_i S_i`` -- the Neyman standard-error term, times sqrt(n)."""
    total = len(strata)
    term = 0.0
    for value in np.unique(strata):
        member = strata == value
        weight = member.sum() / total
        p = y[member].mean()
        term += weight * np.sqrt(max(p * (1.0 - p), 0.0))
    return float(term)


def srs_term(y: np.ndarray) -> float:
    """The same quantity for a single stratum: sqrt(p (1 - p))."""
    p = y.mean()
    return float(np.sqrt(max(p * (1.0 - p), 0.0)))


def efficiency(term_a: float, term_b: float) -> float:
    """eta_{A,B}: how many times smaller A's variance is than B's."""
    if term_a <= 0:
        return float("inf")
    return (term_b / term_a) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--cv", choices=["blocked", "random"], default="blocked")
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--postclass-base", nargs="*", default=[])
    parser.add_argument("--postclass-per-date", action="store_true")
    parser.add_argument("--hier-base", nargs="*", default=[])
    parser.add_argument("--models", nargs="+", required=True,
                        help="Models to score as maps")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, target, groups = load(args.input, args.min_class_count)
    columns = feature_columns(frame)
    features = frame[columns]
    truth = target.to_numpy()

    registry = build_registry(
        columns,
        balanced=args.balanced,
        postclass_bases=args.postclass_base,
        postclass_shared=not args.postclass_per_date,
        hier_bases=args.hier_base,
    )
    missing = [m for m in args.models if m not in registry]
    if missing:
        raise ValueError(f"Not available: {missing}. Have: {sorted(registry)}")

    classes = [CHANGE_KEY] + sorted(target.unique())
    print(f"{len(frame):,} plots | {len(args.models)} maps | "
          f"{len(classes) - 1} classes | {args.cv} CV")

    predictions: dict[str, np.ndarray] = {}
    for name in args.models:
        factory, needs_frame = registry[name]
        _, oof = evaluate(name, factory, features, frame, target, groups,
                          args.n_splits, is_change, needs_frame, args.cv)
        predictions[name] = np.asarray(oof, dtype=object)
        print(f"  {name}: out-of-fold map complete", flush=True)

    rows = []
    for klass in classes:
        y = indicator(truth, klass)
        base = srs_term(y)
        for name, mapped in predictions.items():
            term = stratified_term(mapped, y)
            eta = efficiency(term, base)
            rows.append({
                "class": klass,
                "map": name,
                "n_reference": int(y.sum()),
                "n_strata": int(len(np.unique(mapped))),
                "se_term_map": round(term, 5),
                "se_term_srs": round(base, 5),
                "efficiency_vs_srs": round(eta, 3),
                "sample_size_ratio": round(1 / eta, 3) if eta else float("inf"),
            })
    table = pd.DataFrame(rows)

    # Pairwise eta on the change indicator -- the estimand the project turns on.
    y_change = indicator(truth, CHANGE_KEY)
    terms = {n: stratified_term(m, y_change) for n, m in predictions.items()}
    pairwise = pd.DataFrame(
        [[round(efficiency(terms[a], terms[b]), 3) for b in args.models]
         for a in args.models],
        index=pd.Index(args.models, name="map_A"),
        columns=pd.Index(args.models, name="map_B"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    table.to_csv(args.output_dir / f"map_efficiency{tag}.csv", index=False)
    pairwise.to_csv(args.output_dir / f"map_efficiency_pairwise{tag}.csv")
    (args.output_dir / f"map_efficiency_meta{tag}.json").write_text(
        json.dumps({
            "input": str(args.input),
            "n_plots": len(frame),
            "cv": args.cv,
            "n_splits": args.n_splits,
            "balanced_weights": args.balanced,
            "maps": args.models,
            "baseline": "simple random sampling (single stratum)",
            "cost_ratio_v_over_u": 1,
            "allocation": "Neyman",
        }, indent=2),
        encoding="utf-8",
    )

    wide = table.pivot(index="class", columns="map", values="efficiency_vs_srs")
    print("\nRelative efficiency vs simple random sampling "
          "(>1 = fewer samples needed for the same variance)\n")
    print(wide.to_string())
    print("\nPairwise efficiency on 'any change' — eta[A][B] > 1 means row A "
          "beats column B\n")
    print(pairwise.to_string())


if __name__ == "__main__":
    main()
