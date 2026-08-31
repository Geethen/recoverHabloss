"""Protocol invariants of the active-learning replay (`src/al_lab.py`).

The verdicts in `docs/research/ACTIVE_LEARNING.md` are only worth anything if the
simulation cannot cheat, and there are exactly three ways it could: leak the test
region into the pool, let a strategy read a label it has not acquired, or place a
fold-local probability block positionally so classes permute in the early rounds
where a rare class drops out. Each has a test here.

These run on a synthetic frame rather than the 6,414-plot table, so they need no
GPU, no parquet and no network -- the point is the bookkeeping, not the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from al_lab import (  # noqa: E402
    CHANGE_CLASSES, GAP_SLOPE_DEG, GAP_WORLDCOVER, MIN_PER_CLASS, STRATEGIES,
    AlContext, Round, _pick, _place, nearest_distance, spatial_folds,
)
from acquisition import novelty_to_reference  # noqa: E402


# ---------------------------------------------------------------------------
# fold geometry
# ---------------------------------------------------------------------------
def test_spatial_folds_group_neighbours_not_rows():
    """Two tight continental blobs must not be split across folds at k=2.

    A random row split would put half of each blob in each fold, and an
    acquisition function that simply picks plots near the test set would then
    win on spatial autocorrelation rather than on ranking.
    """
    rng = np.random.default_rng(0)
    lon = np.concatenate([rng.normal(-60, 2, 200), rng.normal(120, 2, 200)])
    lat = np.concatenate([rng.normal(-10, 2, 200), rng.normal(35, 2, 200)])
    folds = spatial_folds(lon, lat, 2, seed=0)
    assert len(set(folds[:200])) == 1
    assert len(set(folds[200:])) == 1
    assert folds[0] != folds[200]


def test_fold_ids_are_canonical_west_to_east():
    """Fold 0 is the westernmost cluster at every seed, so a per-fold read is
    comparable across seeds instead of averaging South America into Asia."""
    rng = np.random.default_rng(1)
    lon = np.concatenate([rng.normal(-120, 1, 100), rng.normal(0, 1, 100),
                          rng.normal(120, 1, 100)])
    lat = rng.normal(20, 1, 300)
    for seed in range(4):
        folds = spatial_folds(lon, lat, 3, seed=seed)
        assert folds[0] == 0 and folds[100] == 1 and folds[200] == 2


# ---------------------------------------------------------------------------
# class alignment
# ---------------------------------------------------------------------------
def test_place_matches_classes_by_name_not_position():
    """A fold that never saw class 'b' must leave b's column at zero rather
    than shift c's probabilities into it."""
    classes = ["a", "b", "c"]
    block = np.array([[0.3, 0.7]])          # this fold only emitted a and c
    out = _place(block, ["a", "c"], classes)
    assert out.tolist() == [[0.3, 0.0, 0.7]]


def test_place_is_not_fooled_by_a_permuted_local_order():
    classes = ["a", "b", "c"]
    out = _place(np.array([[0.1, 0.9]]), ["c", "a"], classes)
    assert out.tolist() == [[0.9, 0.0, 0.1]]


# ---------------------------------------------------------------------------
# the novelty reduction
# ---------------------------------------------------------------------------
def test_nearest_distance_is_the_per_row_form_of_novelty_to_reference():
    """`acq.novelty_to_reference` reduces a patch of pixels at p90 and returns a
    scalar. A plot is one point, so the reduction is the identity -- and the two
    must agree on a single row or the replay is scoring a different quantity
    from the one the design document describes."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(50, 8))
    x = rng.normal(size=(1, 8))
    assert nearest_distance(x, ref / np.linalg.norm(ref, axis=1, keepdims=True))[0] \
        == pytest.approx(novelty_to_reference(x, ref), abs=1e-9)


def test_nearest_distance_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(4)
    ref = rng.normal(size=(30, 6))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    x = rng.normal(size=(97, 6))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.testing.assert_allclose(nearest_distance(x, ref, chunk=7),
                               nearest_distance(x, ref, chunk=1000))


# ---------------------------------------------------------------------------
# a synthetic context, enough for the bookkeeping tests
# ---------------------------------------------------------------------------
def _toy_context(n: int = 400, seed: int = 0) -> AlContext:
    rng = np.random.default_rng(seed)
    classes = ["Nature -> Nature", "Cropland -> Cropland",
               "Artificial -> Artificial", "Nature -> Artificial",
               "Artificial -> Cropland"]
    target = np.asarray(rng.choice(classes, n, p=[0.45, 0.25, 0.15, 0.1, 0.05]),
                        dtype=object)
    x = rng.normal(size=(n, 16))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    frame = pd.DataFrame({"lon": rng.uniform(-180, 180, n),
                          "lat": rng.uniform(-55, 60, n)})
    gap = rng.random(n) < 0.2
    return AlContext(frame=frame, target=target,
                     truth_merged=np.asarray(["Stable" if a.split(" -> ")[0]
                                              == a.split(" -> ")[1] else "Change"
                                              for a in target], dtype=object),
                     cols=[], kwargs={}, fine_classes=sorted(set(target)),
                     merged_classes=["Change", "Stable"],
                     spaces={"state": x, "delta": x[:, ::-1].copy()}, gap=gap)


def _round(ctx: AlContext, n_lab: int = 100) -> Round:
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ctx.target))
    labelled = np.sort(perm[:n_lab])
    pool = np.sort(perm[n_lab:])
    return Round(ctx, labelled, pool, seed=0, round_idx=0, n_rounds=4)


# ---------------------------------------------------------------------------
# what a strategy is allowed to pick
# ---------------------------------------------------------------------------
MODEL_FREE = ["random", "random_b", "random_c", "novelty", "kcenter", "rarity",
              "delta_rarity", "fa", "proto_sim", "delta_mag"]


@pytest.mark.parametrize("name", MODEL_FREE)
def test_model_free_strategies_pick_inside_the_pool(name):
    """Selections are positions *within* the pool. An index out of range would
    silently acquire a test-fold row, which is the leak this harness exists to
    avoid."""
    ctx = _toy_context()
    rd = _round(ctx)
    take = _pick(rd, name, 25)
    assert len(take) == 25
    assert len(set(take.tolist())) == 25, "a strategy acquired the same row twice"
    assert take.min() >= 0 and take.max() < len(rd.pool)


@pytest.mark.parametrize("name", MODEL_FREE)
def test_model_free_scores_are_finite_and_pool_shaped(name):
    ctx = _toy_context()
    rd = _round(ctx)
    if name == "kcenter":
        pytest.skip("kcenter returns an order, not a score")
    scores = STRATEGIES[name][0](rd)
    assert np.asarray(scores).shape == (len(rd.pool),)
    assert np.isfinite(scores).all()


def test_kcenter_never_returns_an_already_labelled_row():
    """It is seeded with the labelled set stacked *after* the pool; a return
    value at or above len(pool) would mean it re-acquired a held label."""
    ctx = _toy_context()
    rd = _round(ctx, n_lab=150)
    take = _pick(rd, "kcenter", 40)
    assert take.max() < len(rd.pool)


def test_the_two_null_streams_disagree_with_random():
    """`random` at a fixed (seed, fold) is deterministic and so cannot be run
    against itself -- that is the whole reason random_b/random_c exist. If they
    ever coincided, the measured floor would be exactly zero and every arm would
    read as a win."""
    ctx = _toy_context()
    rd = _round(ctx)
    picks = {n: set(_pick(rd, n, 40).tolist())
             for n in ("random", "random_b", "random_c")}
    assert picks["random"] != picks["random_b"]
    assert picks["random"] != picks["random_c"]
    assert picks["random_b"] != picks["random_c"]


def test_proto_sim_declines_to_guess_before_it_has_change_labels():
    """With almost no change plots acquired there is no prototype to point at,
    and inventing one from three rows would be noise dressed as a direction."""
    ctx = _toy_context()
    stable = np.flatnonzero(ctx.target == "Nature -> Nature")[:20]
    rd = Round(ctx, stable, np.setdiff1d(np.arange(len(ctx.target)), stable),
               seed=0, round_idx=0, n_rounds=4)
    assert np.allclose(STRATEGIES["proto_sim"][0](rd), 0.0)


# ---------------------------------------------------------------------------
# the oracle arms are the only ones allowed to read a hidden label
# ---------------------------------------------------------------------------
def test_oracle_change_is_the_ceiling_it_claims_to_be():
    ctx = _toy_context()
    rd = _round(ctx)
    take = _pick(rd, "oracle_change", 30)
    picked = ctx.target[rd.pool[take]]
    assert all(t.split(" -> ")[0] != t.split(" -> ")[1] for t in picked)


def test_every_non_oracle_strategy_is_registered_outside_the_oracle_group():
    """A method that quietly grew a label read would have to be moved into the
    oracle group to keep this passing, which makes the change visible in review."""
    oracles = {n for n, (_, g, _) in STRATEGIES.items() if g == "oracle"}
    assert oracles == {"oracle_change", "oracle_balance"}


# ---------------------------------------------------------------------------
# the coverage gap
# ---------------------------------------------------------------------------
def test_gap_definition_matches_the_documented_strata():
    """The withheld stratum is 'steep OR bare/snow/wetland/mangrove/moss'.
    Widening it silently would change what AL3 measures without changing its
    name, so the constants are pinned here as well as in the module."""
    assert GAP_SLOPE_DEG == 8.0
    assert set(GAP_WORLDCOVER) == {60, 70, 90, 95, 100}


def test_change_class_list_holds_every_transition_and_no_stable_one():
    ctx = _toy_context()
    for c in CHANGE_CLASSES:
        a, b = c.split(" -> ")
        assert a != b
    assert MIN_PER_CLASS >= 2, "the fine head needs two rows per class to emit it"
    assert set(ctx.merged_classes) == {"Change", "Stable"}
