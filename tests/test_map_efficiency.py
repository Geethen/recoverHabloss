import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from map_efficiency import (  # noqa: E402
    CHANGE_KEY,
    efficiency,
    indicator,
    srs_term,
    stratified_term,
)


def test_pure_strata_have_zero_variance_and_infinite_efficiency():
    # Every stratum is entirely one reference class, so no sample is needed to
    # estimate its area. Infinity is the honest answer, not a divide-by-zero.
    y = np.array([1.0, 1.0, 0.0, 0.0])
    strata = np.array(["a", "a", "b", "b"])
    assert stratified_term(strata, y) == 0.0
    assert efficiency(stratified_term(strata, y), srs_term(y)) == float("inf")


def test_a_map_uncorrelated_with_the_class_is_exactly_as_good_as_no_map():
    # Both strata carry the population proportion, so stratifying buys nothing.
    y = np.array([1.0, 0.0, 1.0, 0.0])
    strata = np.array(["a", "a", "b", "b"])
    assert stratified_term(strata, y) == pytest.approx(srs_term(y))
    assert efficiency(stratified_term(strata, y), srs_term(y)) == pytest.approx(1.0)


def test_efficiency_is_the_ratio_of_variances_not_of_standard_errors():
    # eta must square the SE ratio: halving the SE quarters the sample size.
    assert efficiency(0.25, 0.5) == pytest.approx(4.0)


def test_efficiency_below_one_means_the_map_hurts():
    # Stratification cannot hurt under Neyman allocation, but a *comparison*
    # between two maps must be able to report the worse one.
    assert efficiency(0.5, 0.25) == pytest.approx(0.25)


def test_change_indicator_counts_every_off_diagonal_transition():
    labels = np.array(
        ["Nature -> Nature", "Nature -> Artificial", "Cropland -> Cropland", "other"]
    )
    # 'other' pools rare transitions, all of which are change.
    assert list(indicator(labels, CHANGE_KEY)) == [0.0, 1.0, 0.0, 1.0]


def test_single_class_indicator_selects_only_that_transition():
    labels = np.array(["Nature -> Nature", "Nature -> Artificial", "other"])
    assert list(indicator(labels, "Nature -> Artificial")) == [0.0, 1.0, 0.0]


def test_srs_term_is_the_binomial_standard_deviation():
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert srs_term(y) == pytest.approx(0.5)


def test_stratified_term_weights_strata_by_their_share():
    # Stratum 'a' is 3/4 of the plots at p=1/3, stratum 'b' is 1/4 at p=0.
    y = np.array([1.0, 0.0, 0.0, 0.0])
    strata = np.array(["a", "a", "a", "b"])
    expected = 0.75 * np.sqrt((1 / 3) * (2 / 3))
    assert stratified_term(strata, y) == pytest.approx(expected)
