"""The gate-off fast path must be bit-identical to the general path.

`HierarchicalSoftmaxNN.probs_aef_only` skips the detail tower entirely on the
argument that `gated_mean` fusion with `g_t = 0` reduces exactly to the
AlphaEarth tower's representation. "Exactly" is the whole claim -- a fast path
that is merely *close* would silently ship a different model than the one that
was scored, which is the same class of error `infer_s2 --self-check` exists to
prevent. So this asserts bit equality, not a tolerance.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_zoo import HierarchicalSoftmaxNN  # noqa: E402

RNG = np.random.default_rng(0)


def _toy(n=240, d_aef=12, d_s2=9):
    """A small two-tower problem with a realistically sparse detail modality."""
    aef = [f"A{i:02d}" for i in range(d_aef)]
    s2 = [f"S{i:02d}" for i in range(d_s2)]
    frame = pd.DataFrame(
        RNG.normal(size=(n, d_aef + d_s2)).astype("float32"), columns=aef + s2)
    frame["aef_present"] = 1.0
    # ~10% of rows have no detail modality at all, as in the real table.
    frame["s2_present"] = (RNG.random(n) > 0.1).astype("float32")
    y = np.array(["Nature -> Nature", "Cropland -> Cropland",
                  "Nature -> Artificial", "Artificial -> Artificial"])[
        RNG.integers(0, 4, n)]
    return frame, aef, s2, y


def _fit(**over):
    frame, aef, s2, y = _toy()
    kwargs = dict(arch="two_tower", loss="focal", epochs=3, tower_dim=32,
                  aef_columns=aef, tess_columns=s2, mask_column="s2_present",
                  aef_mask_column="aef_present", fusion="gated_mean",
                  modality_dropout=0.5, dropout_tess=0.7, seed=0)
    kwargs.update(over)
    model = HierarchicalSoftmaxNN(aef + s2, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, y)
    return model, frame, aef, s2


@pytest.mark.parametrize("fusion", ["gated_mean", "additive"])
def test_fastpath_is_bit_identical_to_gate_off(fusion):
    model, frame, _, _ = _fit(fusion=fusion)

    off = frame.copy()
    off["s2_present"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        slow_fine, slow_merged = model._probs(off)
    fast_fine, fast_merged = model.probs_aef_only(off)

    assert np.array_equal(slow_fine, fast_fine)
    assert np.array_equal(slow_merged, fast_merged)


def test_fastpath_ignores_detail_columns_entirely():
    """The point of the exercise: serve without ever computing S2 features."""
    model, frame, aef, s2 = _fit()

    off = frame.copy()
    off["s2_present"] = 0.0
    expected_fine, expected_merged = model.probs_aef_only(off)

    # A frame that does not carry the detail block at all must give the same
    # answer -- that is what lets the AOI path skip the composite fetch.
    without = off.drop(columns=s2)
    got_fine, got_merged = model.probs_aef_only(without)
    assert np.array_equal(expected_fine, got_fine)
    assert np.array_equal(expected_merged, got_merged)

    # And the detail values must not influence it: garbage in the S2 block
    # changes nothing while the gate is off.
    garbage = off.copy()
    garbage[s2] = RNG.normal(size=(len(off), len(s2))).astype("float32") * 1e3
    g_fine, g_merged = model.probs_aef_only(garbage)
    assert np.array_equal(expected_fine, g_fine)
    assert np.array_equal(expected_merged, g_merged)


def test_fastpath_refuses_configurations_it_is_not_proved_for():
    """A learned gate is not provably zero, so it must not take this path."""
    model, frame, _, _ = _fit(tess_gate="learned")
    with pytest.raises(ValueError, match="tess_gate"):
        model.probs_aef_only(frame)
    with pytest.raises(ValueError, match="tess_gate"):
        model.probs_aef_only_matrix(np.zeros((3, len(model.aef_columns)), "float32"))


def test_matrix_path_matches_the_frame_path_bit_for_bit():
    """Serving reads a raster, not a table, and must not pay for the difference.

    `probs_aef_only_matrix` standardises on the device instead of in numpy and
    never builds a DataFrame. Same arithmetic, same dtype, so the same bits --
    which is what allows the map to be produced by the cheaper call.
    """
    model, frame, aef, _ = _fit()

    expected_fine, expected_merged = model.probs_aef_only(frame)
    got_fine, got_merged = model.probs_aef_only_matrix(
        frame[aef].to_numpy("float32"))

    assert np.array_equal(expected_fine, got_fine)
    assert np.array_equal(expected_merged, got_merged)


def test_matrix_path_maps_absent_values_to_the_column_mean():
    """Non-finite input must land on 0 after centring, as the frame path does.

    A nodata AlphaEarth pixel arrives as NaN. The frame path maps it with
    `np.where(isfinite)` *after* standardising; the device path uses
    `nan_to_num` in the same position. If the two ever diverged, nodata pixels
    would be classified from a different value than the one the model was fitted
    to treat as "absent" -- and nodata is the majority of many tiles' edges.
    """
    model, frame, aef, _ = _fit()

    holed = frame.copy()
    holed.loc[holed.index[:20], aef[0]] = np.nan
    holed.loc[holed.index[20:30], aef[1]] = np.inf

    expected = model.probs_aef_only(holed)
    got = model.probs_aef_only_matrix(holed[aef].to_numpy("float32"))
    assert np.array_equal(expected[0], got[0])
    assert np.array_equal(expected[1], got[1])
    assert np.isfinite(got[0]).all()


def test_band_stacking_matches_the_pixel_major_stack():
    """The cheap band-major layout must hold exactly the old numbers.

    `stack_aef_bands` replaces `np.stack([...], -1).reshape(n, d)`, which is the
    one place in the pipeline where an embedding channel could be silently
    permuted. Equality against the construction it replaced is the guard.
    """
    import infer_s2

    n_band, h, w = 5, 4, 3
    e18 = RNG.normal(size=(n_band, h, w)).astype("float32")
    e24 = RNG.normal(size=(n_band, h, w)).astype("float32")
    cols = sorted(f"A{i:02d}_{s}" for i in range(n_band)
                  for s in ("2018", "2024", "diff"))

    named = {}
    for i in range(n_band):
        named[f"A{i:02d}_2018"] = e18[i]
        named[f"A{i:02d}_2024"] = e24[i]
        named[f"A{i:02d}_diff"] = e24[i] - e18[i]
    old = np.stack([named[c] for c in cols], -1).reshape(h * w, len(cols))

    new = infer_s2.stack_aef_bands(e18, e24, cols)
    assert new.shape == (len(cols), h * w)
    assert np.array_equal(new.T, old)
