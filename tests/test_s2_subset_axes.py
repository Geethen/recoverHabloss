"""The channel axis must not be able to move the deployed detail tower.

`build_s2_features` gained four more 10 m indices (EVI2, GRVI, BSI, CI) so that
a difference-driven detail tower could be tried. Every column name embeds its
channel, so a family selector that only filtered on the family prefix would have
widened `s2off_centre_m3s3_bf` from 78 columns to 114 the moment the feature
table was rebuilt -- silently serving a different model than the one CLAUDE.md
pins, with no error anywhere.

`s2_subset_columns` therefore defaults its channel filter to `CHANNELS_BASE`.
These tests are that default's guard: they run against a synthetic column list
built from the real channel and family names, so they fail if the default is
removed, if a new channel is appended to `CHANNELS_BASE` rather than to
`CHANNELS_10M`, or if the year axis stops filtering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_s2_features import CHANNELS, CHANNELS_10M, CHANNELS_BASE  # noqa: E402
from twotower_lab import (S2_SUBSET_CHANNELS, S2_SUBSETS, s2_base_columns,  # noqa: E402
                          s2_channel_of, s2_subset_columns, s2_year_of)

YEARS = ("2018", "2024", "diff")
FAMILIES = ("S2c", "S2m3", "S2m9", "S2m25", "S2s3", "S2s9", "S2s25", "S2lc", "S2g")


def _stat_block() -> list[str]:
    """The stat block as `build_s2_features` writes it, names only."""
    cols = [f"{fam}_{ch}_{yr}"
            for fam in FAMILIES for ch in CHANNELS for yr in YEARS]
    cols += [f"S2bf{w}_{yr}" for w in (3, 5, 9, 25, 64) for yr in YEARS]
    return cols


def test_channel_lists_do_not_overlap():
    assert not set(CHANNELS_BASE) & set(CHANNELS_10M)
    assert CHANNELS[:len(CHANNELS_BASE)] == CHANNELS_BASE


def test_deployed_subset_is_78_columns_whatever_the_channel_set():
    """The number CLAUDE.md pins, recomputed from the live definitions."""
    cols = s2_subset_columns(_stat_block(), "centre_m3s3_bf")
    assert len(cols) == 78, f"deployed detail tower moved to {len(cols)} columns"
    assert {s2_channel_of(c) for c in cols} == {None} | set(CHANNELS_BASE)


@pytest.mark.parametrize("name,expect", [("full", 204), ("centre_m3s3_bf", 78),
                                         ("centre_s3_bf", 57), ("bf", 15)])
def test_published_subset_sizes_are_unchanged(name, expect):
    assert len(s2_subset_columns(_stat_block(), name)) == expect


def test_the_added_indices_reach_only_the_subsets_that_ask_for_them():
    stat = _stat_block()
    for name in S2_SUBSETS:
        channels = {s2_channel_of(c) for c in s2_subset_columns(stat, name)}
        extra = channels & set(CHANNELS_10M)
        assert bool(extra) == (name in S2_SUBSET_CHANNELS), name


def test_diff_subsets_carry_no_endpoint_state():
    stat = _stat_block()
    for name in ("diff_centre_m3s3_bf", "diff10_centre_m3s3_bf"):
        assert {s2_year_of(c) for c in s2_subset_columns(stat, name)} == {"diff"}
    # ...except the one arm that deliberately keeps built fraction as a state.
    cols = s2_subset_columns(stat, "diff10_bfstate_centre_m3s3")
    bf = [c for c in cols if c.startswith("S2bf")]
    assert {s2_year_of(c) for c in bf} == set(YEARS)
    assert {s2_year_of(c) for c in cols if c not in bf} == {"diff"}


def test_s2_base_columns_pins_the_published_block():
    stat = _stat_block()
    base = s2_base_columns(stat)
    assert len(base) == 204
    assert not {s2_channel_of(c) for c in base} & set(CHANNELS_10M)


def test_a_subset_asking_for_absent_channels_raises():
    """The failure that would otherwise be a silently smaller feature set."""
    with pytest.raises(ValueError, match="does not carry"):
        s2_subset_columns(s2_base_columns(_stat_block()),
                          "diff10_centre_m3s3_bf")


def test_built_fraction_has_no_channel_and_survives_every_filter():
    assert s2_channel_of("S2bf3_2018") is None
    assert s2_channel_of("S2c_ndvi_diff") == "ndvi"
    assert s2_year_of("S2bf25_diff") == "diff"
    for name in ("bf", "centre_m3s3_bf", "diff10_centre_m3s3_bf"):
        cols = s2_subset_columns(_stat_block(), name)
        assert any(c.startswith("S2bf") for c in cols), name


# ---------------------------------------------------------------------------
# the index arithmetic
# ---------------------------------------------------------------------------
def test_added_indices_are_computed_on_the_right_scale():
    """EVI2's soil term is absolute; the others are ratios and must be bounded.

    Reflectance is stored as L2A DN (x10000). Feeding DN straight into
    2.5(N-R)/(N+2.4R+1) makes the +1 negligible, which turns EVI2 into a plain
    ratio index and loses the one property it was added for -- and nothing about
    the resulting column looks wrong.
    """
    import numpy as np
    from build_s2_features import _channels

    # blue, green, red, nir for a vegetated pixel, in DN.
    patch = np.array([[[[500.0]], [[800.0]], [[600.0]], [[3500.0]]]])
    cube = _channels(patch)
    got = {name: float(cube[0, i, 0, 0]) for i, name in enumerate(CHANNELS)}

    n, r = 0.35, 0.06
    assert got["evi2"] == pytest.approx(2.5 * (n - r) / (n + 2.4 * r + 1.0), rel=1e-6)
    # A ratio-scale EVI2 would land near NDVI; the soil term must separate them.
    assert abs(got["evi2"] - got["ndvi"]) > 0.2

    assert got["grvi"] == pytest.approx((800 - 600) / (800 + 600), rel=1e-4)
    assert got["bsi"] == pytest.approx(((600 + 500) - (3500 + 800))
                                       / ((600 + 500) + (3500 + 800)), rel=1e-4)
    assert got["ci"] == pytest.approx((600 - 500) / (600 + 500), rel=1e-4)
    for name in ("ndvi", "ndwi", "grvi", "bsi", "ci"):
        assert -1.0 <= got[name] <= 1.0, name


def test_every_added_index_uses_only_the_10_m_bands():
    """The point of the set: no 20 m band, so no resample and no SWIR fetch."""
    import inspect

    import build_s2_features as BSF

    src = inspect.getsource(BSF._channels)
    body = src.split('"""')[-1]
    for band in ("swir", "b11", "b12", "rededge", "b05", "b06", "b07", "b8a"):
        assert band not in body.lower(), f"{band} is not a 10 m band"
