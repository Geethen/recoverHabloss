import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from design_analysis import read_stratum_areas_parallel, stratified_ratio
from analyse_transition_composition import first_mode


def test_stratified_ratio_uses_area_weights_and_paired_covariance():
    frame = pd.DataFrame({"stratum": ["a", "a", "b", "b"]})
    numerator = np.array([1, 0, 0, 0])
    denominator = np.array([1, 1, 1, 0])
    areas = {"a": 1.0, "b": 3.0}

    result, detail = stratified_ratio(
        frame, areas, numerator, denominator
    )

    # Weighted numerator = 1*.5 + 3*0 = .5; denominator = 1*1 + 3*.5 = 2.5.
    assert np.isclose(result["estimate"], 0.2)
    assert result["raw_numerator"] == 1
    assert result["raw_denominator"] == 3
    assert len(detail) == 2
    assert result["se"] > 0


def test_parallel_area_reader_sums_shards(tmp_path):
    (tmp_path / "a.csv").write_text(
        "stratum,area\n1,1000000\n2,2000000\n", encoding="utf-8"
    )
    (tmp_path / "b.csv").write_text(
        "stratum,area\n1,3000000\n", encoding="utf-8"
    )

    result = read_stratum_areas_parallel(
        tmp_path, {1: "one", 2: "two"}, workers=2
    )

    assert result == {"one": 4.0, "two": 2.0}


def test_first_mode_matches_r_first_value_tie_break():
    assert first_mode(pd.Series(["nature", "crop", "crop", "nature"])) == "nature"
