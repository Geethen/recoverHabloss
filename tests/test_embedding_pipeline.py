import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "R" / "src"))
from extract_embeddings_gee import (  # noqa: E402
    assign_blocks,
    block_id,
    check_label_columns,
    difference_bands,
    embedding_bands,
    polygon_centroid,
)
from model_transitions import build_target, feature_columns  # noqa: E402


def square(lon: float, lat: float, size: float = 0.001) -> str:
    ring = [
        [lon, lat],
        [lon + size, lat],
        [lon + size, lat + size],
        [lon, lat + size],
        [lon, lat],
    ]
    return str({"type": "Polygon", "coordinates": [ring]})


def test_polygon_centroid_parses_stored_geojson_string():
    lon, lat = polygon_centroid(square(10.0, -5.0, size=2.0))
    assert lon == pytest.approx(10.8, abs=1e-6)
    assert lat == pytest.approx(-4.2, abs=1e-6)


def test_polygon_centroid_returns_none_for_unparseable_geometry():
    assert polygon_centroid("not a geometry") == (None, None)
    assert polygon_centroid(None) == (None, None)


def test_block_id_floors_toward_negative_infinity():
    # Points either side of the meridian must land in different blocks.
    assert block_id(5.0, 5.0, 20.0) != block_id(-5.0, 5.0, 20.0)
    assert block_id(1.0, 1.0, 20.0) == block_id(19.9, 19.9, 20.0)


def test_assign_blocks_groups_nearby_points_and_separates_distant_ones():
    frame = pd.DataFrame(
        {
            "PLOTID": ["a", "b", "c"],
            "geo": [square(10.0, 10.0), square(11.0, 11.0), square(-120.0, -40.0)],
        }
    )

    result = assign_blocks(frame, block_size=20.0)

    assert result.loc[0, "block_id"] == result.loc[1, "block_id"]
    assert result.loc[2, "block_id"] != result.loc[0, "block_id"]
    assert result["lon"].notna().all()


def test_assign_blocks_rejects_unparseable_geometry():
    frame = pd.DataFrame({"PLOTID": ["a"], "geo": ["garbage"]})
    with pytest.raises(ValueError, match="unparseable geometry"):
        assign_blocks(frame, block_size=20.0)


def test_embedding_band_names_are_distinct_per_year():
    assert embedding_bands(2018)[0] == "A00_2018"
    assert len(embedding_bands(2018)) == 64
    assert not set(embedding_bands(2018)) & set(embedding_bands(2024))
    assert difference_bands()[63] == "A63_diff"


def test_check_label_columns_aborts_when_transition_fields_absent():
    # This is the failure that silently produced an unlabelled predictor table.
    with pytest.raises(RuntimeError, match="missing transition label"):
        check_label_columns(["PLOTID", "r", "stratum"])


def test_check_label_columns_keeps_present_fields():
    keep = check_label_columns(["PLOTID", "lc_2018", "lc_2024", "r"])
    assert keep == ["PLOTID", "lc_2018", "lc_2024", "r"]


def test_build_target_pools_rare_transitions():
    frame = pd.DataFrame(
        {
            "lc_2018": ["nature"] * 5 + ["nature"],
            "lc_2024": ["crop"] * 5 + ["artificial"],
        }
    )

    target, summary = build_target(frame, min_count=2)

    assert list(target) == ["nature -> crop"] * 5 + ["other"]
    assert summary.loc[summary["transition"] == "nature -> crop", "retained"].item()


def test_build_target_requires_transition_columns():
    with pytest.raises(ValueError, match="lc_2018"):
        build_target(pd.DataFrame({"lc_2024": ["crop"]}), min_count=1)


def test_feature_columns_can_exclude_difference_bands():
    frame = pd.DataFrame(
        columns=["A00_2018", "A00_2024", "A00_diff", "PLOTID", "lc_2018"]
    )

    assert feature_columns(frame) == ["A00_2018", "A00_2024", "A00_diff"]
    assert feature_columns(frame, use_diff=False) == ["A00_2018", "A00_2024"]


def test_feature_columns_raises_when_no_embeddings_present():
    with pytest.raises(ValueError, match="No embedding feature columns"):
        feature_columns(pd.DataFrame(columns=["PLOTID", "lc_2018"]))
