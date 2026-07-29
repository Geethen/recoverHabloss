import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from estimators import (
    composition_ratio_ci,
    contrast_ci,
    optimal_lam_multinomial_diag,
    stratified_multinomial,
    stratified_ppi_multinomial,
    stratified_prop,
)


def _one_hot(labels, classes):
    labels = np.asarray(labels)
    return np.stack([(labels == c).astype(float) for c in classes], axis=1)


def test_vector_marginals_match_scalar_stratified_prop():
    rng = np.random.default_rng(0)
    classes = ["nature", "cropland", "artificial"]
    strat = np.array(["a"] * 30 + ["b"] * 30)
    labels = rng.choice(classes, size=60, p=[0.5, 0.3, 0.2])
    Y = _one_hot(labels, classes)
    Nh = {"a": 1_000_000.0, "b": 3_000_000.0}

    p, Sigma = stratified_multinomial(Y, strat, Nh)

    for k, c in enumerate(classes):
        scalar_p, scalar_se, _ = stratified_prop(Y[:, k], strat, Nh)
        assert np.isclose(p[k], scalar_p)
        assert np.isclose(np.sqrt(Sigma[k, k]), scalar_se, rtol=1e-6)


def test_composition_covariance_is_symmetric_psd_and_shares_sum_to_one():
    rng = np.random.default_rng(1)
    classes = ["nature", "cropland", "artificial"]
    strat = np.array(["a"] * 40 + ["b"] * 40)
    labels = rng.choice(classes, size=80, p=[0.4, 0.35, 0.25])
    Y = _one_hot(labels, classes)
    Nh = {"a": 2_000_000.0, "b": 1_000_000.0}

    p, Sigma = stratified_multinomial(Y, strat, Nh)

    assert np.isclose(p.sum(), 1.0)
    assert np.allclose(Sigma, Sigma.T)
    eigvals = np.linalg.eigvalsh(Sigma)
    assert eigvals.min() > -1e-12


def test_off_diagonal_covariance_shapes_contrast_widths():
    # Shares of a partitioned whole are negatively correlated. That widens a
    # DIFFERENCE (c = e_i - e_j, because -2*Cov > 0) and narrows a SUM/share
    # (c = e_i + e_j) relative to treating the classes as independent. Both
    # follow from the same off-diagonal term; ignoring it is simply wrong.
    rng = np.random.default_rng(2)
    classes = ["nature", "cropland", "artificial"]
    strat = np.array(["a"] * 50 + ["b"] * 50)
    labels = rng.choice(classes, size=100, p=[0.45, 0.4, 0.15])
    Y = _one_hot(labels, classes)
    Nh = {"a": 1_000_000.0, "b": 1_000_000.0}

    p, Sigma = stratified_multinomial(Y, strat, Nh)
    diag = np.diag(np.diag(Sigma))
    assert Sigma[0, 1] < 0  # nature and cropland negatively correlated

    _, se_diff_full, _ = contrast_ci(p, Sigma, np.array([1.0, -1.0, 0.0]))
    _, se_diff_indep, _ = contrast_ci(p, diag, np.array([1.0, -1.0, 0.0]))
    assert se_diff_full > se_diff_indep  # difference: covariance widens it

    _, se_sum_full, _ = contrast_ci(p, Sigma, np.array([1.0, 1.0, 0.0]))
    _, se_sum_indep, _ = contrast_ci(p, diag, np.array([1.0, 1.0, 0.0]))
    assert se_sum_full < se_sum_indep  # combined share: covariance tightens it


def test_composition_ratio_matches_two_class_share():
    rng = np.random.default_rng(3)
    classes = ["cropland", "nature"]  # exits from Artificial -> {crop, nature}
    strat = np.array(["a"] * 40 + ["b"] * 40)
    labels = rng.choice(classes, size=80, p=[0.25, 0.75])
    Y = _one_hot(labels, classes)
    Nh = {"a": 1_000_000.0, "b": 2_000_000.0}

    p, Sigma = stratified_multinomial(Y, strat, Nh)
    ratio, se, ci = composition_ratio_ci(p, Sigma, num_idx=[0], den_idx=[0, 1])

    # Denominator sums to 1 here, so the share equals the marginal proportion.
    scalar_p, scalar_se, _ = stratified_prop(Y[:, 0], strat, Nh)
    assert np.isclose(ratio, scalar_p)
    assert np.isclose(se, scalar_se, rtol=1e-6)
    assert ci[0] < ratio < ci[1]


def test_ppi_multinomial_with_constant_proxy_reduces_to_design_based():
    classes = ["nature", "cropland", "artificial"]
    strat = np.array(["a"] * 6 + ["b"] * 6)
    labels = np.array(
        ["nature", "nature", "cropland", "artificial", "nature", "cropland"] * 2
    )
    Y = _one_hot(labels, classes)
    # Proxy constant within each stratum -> carries no within-stratum information.
    Yhat = np.zeros_like(Y)
    Yhat[:6] = [1.0, 0.0, 0.0]
    Yhat[6:] = [0.0, 1.0, 0.0]
    Nh = {"a": 1_000_000.0, "b": 3_000_000.0}
    pop_mean = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0])}
    pop_cov = {"a": np.zeros((3, 3)), "b": np.zeros((3, 3))}
    pop_n = {"a": 1000, "b": 1000}

    p_design, Sigma_design = stratified_multinomial(Y, strat, Nh)
    lam = optimal_lam_multinomial_diag(Y, Yhat, strat, Nh, pop_cov, pop_n)
    p_ppi, Sigma_ppi = stratified_ppi_multinomial(
        Y, Yhat, strat, Nh, pop_mean, pop_cov, pop_n, lam=1.0
    )

    assert np.allclose(lam, 0.0)
    assert np.allclose(p_ppi, p_design)
    # lam is clipped near (not exactly) zero, so the PPI residual covariance
    # matches the design-based one up to that negligible tuning offset.
    assert np.allclose(np.diag(Sigma_ppi), np.diag(Sigma_design), rtol=1e-3)
