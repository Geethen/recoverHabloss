from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import infer_s2  # noqa: E402
from model_zoo import HierarchicalSoftmaxNN  # noqa: E402


def test_cached_two_tower_entry_round_trips_probabilities(tmp_path):
    rng = np.random.default_rng(1)
    aef = [f"A{i:02d}" for i in range(8)]
    s2 = [f"S{i:02d}" for i in range(6)]
    frame = pd.DataFrame(
        rng.normal(size=(180, len(aef) + len(s2))).astype("float32"),
        columns=aef + s2,
    )
    frame["aef_present"] = 1.0
    frame["s2_present"] = (rng.random(len(frame)) > 0.15).astype("float32")
    target = np.array([
        "Artificial -> Artificial",
        "Cropland -> Cropland",
        "Nature -> Nature",
        "Nature -> Artificial",
    ])[rng.integers(0, 4, len(frame))]

    model = HierarchicalSoftmaxNN(
        aef + s2,
        arch="two_tower",
        loss="focal",
        epochs=2,
        tower_dim=16,
        aef_columns=aef,
        tess_columns=s2,
        mask_column="s2_present",
        aef_mask_column="aef_present",
        fusion="gated_mean",
        modality_dropout=0.5,
        dropout_tess=0.7,
        seed=1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, target)

    entry = {
        "models": [model],
        "spec": {"columns": aef + s2, "kwargs": {}, "deploy": "aef_only"},
    }
    metadata = {"recipe": "toy", "seed": 1, "n_seeds": 1}
    path = tmp_path / "toy.pt"

    expected = model.probs_aef_only_matrix(frame[aef].to_numpy("float32"))
    infer_s2._save_cached_entry(path, metadata, entry)
    loaded = infer_s2._load_cached_entry(path, metadata)
    got = loaded["models"][0].probs_aef_only_matrix(frame[aef].to_numpy("float32"))

    assert np.array_equal(expected[0], got[0])
    assert np.array_equal(expected[1], got[1])


def test_cache_load_refuses_metadata_mismatch(tmp_path):
    path = tmp_path / "toy.pt"
    infer_s2._save_cached_entry(path, {"recipe": "toy"}, {"models": [], "spec": {}})

    assert infer_s2._load_cached_entry(path, {"recipe": "other"}) is None


def test_cache_metadata_fingerprints_model_code(tmp_path, monkeypatch):
    aef_path = tmp_path / "aef.parquet"
    s2_path = tmp_path / "s2.parquet"
    aef_path.write_bytes(b"aef")
    s2_path.write_bytes(b"s2")
    monkeypatch.setattr(infer_s2, "DEFAULT_INPUT", aef_path)
    monkeypatch.setattr(
        infer_s2,
        "_model_code_fingerprints",
        lambda: {"model_zoo.py": {"sha256": "old"}},
    )

    metadata = infer_s2._cache_metadata(
        "toy", {"columns": ["A00_2018"], "kwargs": {}}, s2_path, seed=0, n_seeds=1
    )
    path = tmp_path / "toy.pt"
    infer_s2._save_cached_entry(path, metadata, {"models": [], "spec": {}})

    changed = dict(metadata)
    changed["code"] = {"model_zoo.py": {"sha256": "new"}}
    assert infer_s2._load_cached_entry(path, changed) is None