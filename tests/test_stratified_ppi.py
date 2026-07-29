import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from estimators import (
    optimal_lam_stratified,
    stratified_ppi_bootstrap,
    stratified_ppi_prop,
    stratified_prop,
)


def test_constant_proxy_within_strata_adds_no_information():
    y = np.array([0, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    strata = np.array(["a"] * 4 + ["b"] * 4)
    yhat = np.array([0] * 4 + [1] * 4, dtype=float)
    # Nh is mapped area in this application, so the labelled-sample FPC is
    # effectively zero. Use realistic magnitudes when comparing SE formulas.
    Nh = {"a": 1_000_000.0, "b": 3_000_000.0}
    means = {"a": 0.0, "b": 1.0}
    variances = {"a": 0.0, "b": 0.0}
    counts = {"a": 1000, "b": 1000}

    classical = stratified_prop(y, strata, Nh)
    fixed = stratified_ppi_prop(
        y, yhat, strata, Nh, means, variances, counts, lam=1.0
    )
    lam = optimal_lam_stratified(y, yhat, strata, Nh, variances, counts)

    assert np.isclose(fixed[0], classical[0])
    assert np.isclose(fixed[1], classical[1], rtol=1e-5)
    assert lam == 0.0


def test_finite_predicted_sample_variance_is_included():
    y = np.array([0, 1, 0, 1], dtype=float)
    yhat = np.array([0.1, 0.9, 0.2, 0.8])
    strata = np.array(["a"] * 4)
    Nh = {"a": 100.0}
    means = {"a": 0.5}
    variances = {"a": 0.25}

    small = stratified_ppi_prop(
        y, yhat, strata, Nh, means, variances, {"a": 100}, lam=1.0
    )
    large = stratified_ppi_prop(
        y, yhat, strata, Nh, means, variances, {"a": 10_000}, lam=1.0
    )
    assert large[1] < small[1]


def test_stratified_bootstrap_preserves_strata_and_retunes_lambda():
    y = np.array([0, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    strata = np.array(["a"] * 4 + ["b"] * 4)
    yhat = np.array([0] * 4 + [1] * 4, dtype=float)
    unlabelled = np.array([0] * 20 + [1] * 20, dtype=float)
    unlabelled_strata = np.array(["a"] * 20 + ["b"] * 20)
    Nh = {"a": 1.0, "b": 3.0}

    draws, lambdas = stratified_ppi_bootstrap(
        y,
        yhat,
        strata,
        Nh,
        unlabelled,
        unlabelled_strata,
        n_boot=500,
        seed=42,
    )

    assert len(draws) == len(lambdas) == 500
    assert np.all(lambdas == 0.0)
    assert abs(draws.mean() - stratified_prop(y, strata, Nh)[0]) < 0.02
