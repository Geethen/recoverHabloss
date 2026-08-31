"""Invariants of the state-pretraining lab (docs/research/STATE_PRETRAIN_RESEARCH.md).

Three of these lock down the thing that makes the whole package readable -- that
a year-augmented pool cannot score itself on its own near-duplicates -- and one
locks down the *opposite*: that the deliberately leaky ``random`` split really
does leak, so the ladder in ``llto.py`` keeps measuring something.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from statepre import data as sp_data  # noqa: E402
from statepre import llto as sp_llto  # noqa: E402
from statepre import models as sp_models  # noqa: E402


def _synthetic(n_loc=60, years=(2018, 2019, 2024), seed=0):
    """A long frame with the columns the package agrees on, and no real data."""
    rng = np.random.default_rng(seed)
    states = rng.choice(sp_data.STATES, n_loc)
    changed = rng.random(n_loc) < 0.2
    rows = []
    for i in range(n_loc):
        for year in years:
            row = {
                "loc_id": f"p{i:03d}",
                "lon": float(rng.uniform(-170, 170)),
                "lat": float(rng.uniform(-50, 70)),
                "block_id": f"b{i % 7}",
                "year": year,
                "state": states[i],
                "kind": "observed" if year in sp_data.ENDPOINTS else "assumed",
                "source": "recover",
                "changed": bool(changed[i]),
            }
            row.update({c: v for c, v in zip(sp_data.FEATURES,
                                             rng.normal(size=64))})
            rows.append(row)
    frame = pd.DataFrame(rows)
    # One lon/lat per location, or k-means would split a location across folds.
    first = frame.drop_duplicates("loc_id").set_index("loc_id")
    frame["lon"] = frame["loc_id"].map(first["lon"])
    frame["lat"] = frame["loc_id"].map(first["lat"])
    return frame


@pytest.mark.parametrize("split", ["llto", "loc", "block"])
def test_location_folds_are_constant_within_a_location(split):
    """The time-out half of LLTO: a location gets one fold, not one per year."""
    frame = _synthetic()
    mapping = sp_llto.location_folds(frame, n_folds=5, seed=0, split=split)
    folds = frame["loc_id"].map(mapping)
    assert frame.assign(f=folds).groupby("loc_id")["f"].nunique().max() == 1


def test_llto_run_holds_every_year_of_a_test_location_out():
    """The invariant, exercised through `run` rather than asserted in isolation.

    `assert_no_leak` fires inside every fold, so a passing run *is* the proof;
    this pins that the assertion is actually reached, by checking a fold count.
    """
    frame = _synthetic()
    test_frame = frame.loc[frame["kind"] == "observed"].reset_index(drop=True)
    result = sp_llto.run(frame, sp_models.factory("linear"),
                         test_frame=test_frame, n_folds=4, seed=0)
    assert result["n_folds_run"] == 4
    assert result["leaked_locations"] == 0


def test_the_random_split_really_does_leak():
    """The control the ladder needs. If this ever passes at 0, the ladder is dead."""
    frame = _synthetic()
    test_frame = frame.loc[frame["kind"] == "observed"].reset_index(drop=True)
    result = sp_llto.run(frame, sp_models.factory("linear"),
                         test_frame=test_frame, n_folds=4, seed=0, split="random")
    assert result["leaked_locations"] > 0


def test_a_pseudo_row_can_never_be_scored():
    frame = _synthetic()
    frame.loc[frame["year"] == 2019, ["kind", "state"]] = ["pseudo", None]
    test_frame = frame.copy()          # everything offered as test, on purpose
    with pytest.raises(AssertionError, match="non-observed row"):
        sp_llto.assert_no_leak(np.array([]), test_frame["loc_id"].to_numpy(),
                               test_frame["kind"].to_numpy())


@pytest.mark.parametrize("scheme,unit", [("per_loc", "loc_id"),
                                         ("per_source", "source"),
                                         ("per_cell", "block_id")])
def test_weights_give_every_unit_the_same_vote(scheme, unit):
    """The claim each scheme makes: equal *total* weight per its own unit.

    Asserted as equality across units rather than as "sums to 1". The schemes
    are normalised to mean weight 1 (section V2) so that changing the balance
    cannot also change the effective learning rate as a side effect, which puts
    a location's total at `rows_per_location`, not at 1. The invariant that
    matters -- no unit outvotes another -- is scale-free, and pinning the scale
    instead would have this test fail for a change that preserves the property
    it exists to protect. It did.
    """
    frame = _synthetic(n_loc=10, years=(2018, 2019, 2024))
    idx = np.arange(len(frame))
    w = sp_llto._row_weights(frame, idx, scheme)
    totals = pd.Series(w).groupby(frame[unit].to_numpy()).sum()
    assert np.allclose(totals.to_numpy(), totals.to_numpy()[0])
    assert np.isclose(w.mean(), 1.0)


def test_per_loc_is_a_no_op_when_every_location_has_one_row():
    """The control section U1b leans on: nothing to fix, nothing changed.

    A pool with one row per location already votes uniformly, so `per_loc` must
    return flat weights there. Without this, a `per_loc` gain on an augmented
    pool could not be attributed to undoing the vote distortion.
    """
    frame = _synthetic(n_loc=10, years=(2018,))
    w = sp_llto._row_weights(frame, np.arange(len(frame)), "per_loc")
    assert np.allclose(w, 1.0)


# -- dataset arms, on the real frame ---------------------------------------

@pytest.fixture(scope="module")
def plots():
    try:
        return sp_data.load_plots()
    except FileNotFoundError:                            # pragma: no cover
        pytest.skip("the annual embedding frame is not on this machine")


def test_stable_years_augments_only_the_stable_plots(plots):
    frame = sp_data.build("stable_years", plots)
    extra = frame.loc[frame["kind"] == "assumed"]
    assert not extra["changed"].any()
    assert set(extra["year"]) == set(sp_data.INTERMEDIATE)
    # Every stable plot contributes all five intermediate years, and the assumed
    # state is its endpoint state.
    n_stable = int((~plots["changed"]).sum())
    assert len(extra) == n_stable * len(sp_data.INTERMEDIATE)
    endpoint = dict(zip(plots["loc_id"], plots["state_2018"]))
    assert (extra["state"] == extra["loc_id"].map(endpoint)).all()


def test_the_controls_are_row_matched_to_the_hypothesis(plots):
    """`_dup` and `_jit` must differ from `stable_years` only in the vectors."""
    hypothesis = sp_data.build("stable_years", plots)
    for control in ("stable_years_dup", "stable_years_jit"):
        arm = sp_data.build(control, plots)
        assert len(arm) == len(hypothesis)
        assert (arm.groupby(["year", "state"]).size()
                == hypothesis.groupby(["year", "state"]).size()).all()


def test_the_dup_control_carries_no_year_information(plots):
    """Every synthetic row of a location is that location's 2018 vector, exactly."""
    frame = sp_data.build("stable_years_dup", plots)
    synthetic = frame.loc[frame["kind"] == "synthetic"]
    base = (frame.loc[(frame["kind"] == "observed") & (frame["year"] == 2018)]
            .set_index("loc_id")[sp_data.FEATURES])
    aligned = base.loc[synthetic["loc_id"]].to_numpy()
    assert np.allclose(synthetic[sp_data.FEATURES].to_numpy(), aligned)


def test_the_jitter_control_moves_by_the_observed_inter_year_spread(plots):
    frame = sp_data.build("stable_years_jit", plots, seed=0)
    synthetic = frame.loc[frame["kind"] == "synthetic"]
    base = (frame.loc[(frame["kind"] == "observed") & (frame["year"] == 2018)]
            .set_index("loc_id")[sp_data.FEATURES])
    delta = (synthetic[sp_data.FEATURES].to_numpy()
             - base.loc[synthetic["loc_id"]].to_numpy())
    target = sp_data._year_spread(plots)
    # Per-band, within 10% of the spread it is calibrated to.
    assert np.allclose(delta.std(axis=0), target, rtol=0.1)


# -- the triplet term (section Z) -------------------------------------------

def _triplet(z, y, margin):
    torch = pytest.importorskip("torch")
    return float(sp_models._batch_hard_triplet(
        torch.tensor(z, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long), margin))


def test_triplet_is_zero_when_the_classes_are_already_apart():
    """Separation on the unit sphere is 2.0, so a small margin must be inactive.

    The hinge is the whole point of a triplet term -- a version that charged for
    already-separated pairs would pull every embedding together indefinitely.
    """
    z = [[1., 0.], [1., 0.01], [-1., 0.], [-1., 0.01]]
    y = [0, 0, 1, 1]
    assert _triplet(z, y, 0.0) == 0.0
    assert _triplet(z, y, 0.2) == 0.0
    # Interleaving the same points is the maximally wrong arrangement.
    assert _triplet([[1., 0.], [-1., 0.], [1., .01], [-1., .01]], y, 0.2) > 1.0


def test_triplet_is_non_decreasing_in_the_margin():
    z = [[1., 0.], [0.7, 0.7], [0., 1.], [-0.2, 1.]]
    y = [0, 0, 1, 1]
    values = [_triplet(z, y, m) for m in (0.0, 0.2, 0.5, 1.0)]
    assert values == sorted(values)
    assert values[-1] > values[0]


def test_a_batch_with_no_negative_costs_nothing():
    """A single-state batch has no triplet at all -- it must be 0, not NaN.

    Load-bearing for the hcropland arms: that pool is cropland-only, so a batch
    drawn mostly from it can legitimately contain one state, and an anchor whose
    state is alone has no positive either. Both are dropped rather than allowed
    to take a 'hardest positive' from the wrong class, which would invert the
    objective rather than skip it.
    """
    assert _triplet([[1., 0.], [0., 1.]], [0, 0], 0.2) == 0.0
    assert _triplet([[1., 0.], [1., .01], [-1., 0.]], [0, 0, 1], 0.2) == 0.0


def test_the_triplet_arm_changes_the_fit_and_the_plain_arm_is_untouched(plots):
    """`triplet_weight=0` must leave `mlp` byte-identical, and 0.3 must not."""
    pytest.importorskip("torch")
    frame = sp_data.build("endpoints", plots).head(2000)
    X = frame[sp_data.FEATURES].to_numpy("float64")
    y = frame["state"].to_numpy(dtype=str)
    fit = lambda name: sp_models.factory(name, epochs=2)(0).fit(
        X, y, year=frame["year"].to_numpy(), loc=frame["loc_id"].to_numpy())
    base, again, trip = fit("mlp"), fit("mlp"), fit("mlp_trip")
    assert (base.predict(X) == again.predict(X)).all()      # seeded, so exact
    assert base.train_loss_ != trip.train_loss_


# -- the hcropland30 pool (section W) ---------------------------------------

@pytest.fixture(scope="module")
def hcrop():
    try:
        return sp_data.load_hcropland("all")
    except FileNotFoundError:                                # pragma: no cover
        pytest.skip("the hcropland30 pool has not been extracted on this machine")


def test_hcropland_is_cropland_at_2020_and_nothing_else(hcrop):
    """The pool's whole content, asserted: one state, one year.

    Both halves matter. A non-cropland row would mean the ``type == 0`` points
    got a state they cannot have, and a 2018 row would mean a 2020 map label was
    read against the wrong year's embedding.
    """
    assert set(hcrop["state"]) == {"cropland"}
    assert set(hcrop["year"]) == {2020}
    assert set(hcrop["kind"]) == {"observed"}


def test_the_unanimity_filter_is_a_subset_of_the_same_rows(hcrop):
    strict = sp_data.load_hcropland("strict")
    assert 0 < len(strict) < len(hcrop)
    assert set(strict["loc_id"]) <= set(hcrop["loc_id"])
    # Same location, same vector: `strict` is a cut on the `all` extraction, not
    # a second extraction, so a difference between the arms is the filter alone.
    both = strict.merge(hcrop, on="loc_id", suffixes=("_s", "_a"))
    assert np.allclose(both[[f"{c}_s" for c in sp_data.FEATURES]].to_numpy(),
                       both[[f"{c}_a" for c in sp_data.FEATURES]].to_numpy())


def test_the_cropdup_control_matches_the_hypothesis_row_for_row(plots, hcrop):
    """The control must differ from the arm only in where the cropland came from."""
    hypothesis = sp_data.build("glance_hcrop_endpoints", plots, seed=0)
    control = sp_data.build("glance_cropdup_endpoints", plots, seed=0)
    assert len(control) == len(hypothesis)
    assert (control.groupby("state").size()
            == hypothesis.groupby("state").size()).all()
    # And the added rows are GLanCE's own, not hcropland's.
    added = control.loc[control["kind"] == "synthetic"]
    assert set(added["source"]) == {"glance"}
    assert added["loc_id"].nunique() == len(added)


def test_the_hcrop_arms_carry_the_full_test_pool(plots, hcrop):
    """Every arm must still contain every RECOVER endpoint row, unaltered.

    `llto.run` scores the test frame, not the arm, but a training pool that had
    quietly dropped or duplicated plots would change what each fold trains on
    while the comparison still looked paired.
    """
    endpoints = sp_data.build("endpoints", plots)
    for name in ("hcrop_endpoints", "glance_hcrop_endpoints",
                 "glance_hcropall_endpoints", "glance_cropdup_endpoints"):
        arm = sp_data.build(name, plots, seed=0)
        recover = arm.loc[arm["source"] == "recover"]
        assert len(recover) == len(endpoints)
        assert set(recover["loc_id"]) == set(endpoints["loc_id"])


def test_the_mlp_standardises_on_the_training_rows_only(plots):
    torch = pytest.importorskip("torch")
    del torch
    frame = sp_data.build("endpoints", plots).head(3000)
    X = frame[sp_data.FEATURES].to_numpy("float64")
    y = frame["state"].to_numpy(dtype=str)
    model = sp_models.factory("mlp", epochs=1)(0)
    model.fit(X[:2000], y[:2000], year=frame["year"].to_numpy()[:2000],
              loc=frame["loc_id"].to_numpy()[:2000])
    assert np.allclose(model.mu_, X[:2000].mean(0))
    assert not np.allclose(model.mu_, X.mean(0))
