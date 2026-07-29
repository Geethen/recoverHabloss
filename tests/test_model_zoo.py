import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from model_zoo import (  # noqa: E402
    COARSE,
    HierarchicalChange,
    LabelEncoded,
    PostClassification,
    Ravelled,
    TimeeAdapter,
    build_registry,
    coarsen,
    gated_targets,
    is_change_label,
    make_splitter,
    scores,
)


COLUMNS = ["A00_2018", "A01_2018", "A00_2024", "A01_2024", "A00_diff"]


def postclass_frame() -> pd.DataFrame:
    """Two separable clusters per date; plot 2 moves Nature -> Artificial."""
    return pd.DataFrame(
        {
            "A00_2018": [0.0, 0.0, 1.0, 1.0],
            "A01_2018": [0.0, 0.1, 1.0, 1.1],
            "A00_2024": [0.0, 1.0, 1.0, 0.0],
            "A01_2024": [0.1, 1.0, 1.1, 0.0],
            "A00_diff": [0.0, 1.0, 0.0, -1.0],
            "lc_2018": ["Nature", "Nature", "Artificial", "Artificial"],
            "lc_2024": ["Nature", "Artificial", "Artificial", "Nature"],
        }
    )


def test_coarsen_harmonises_the_two_source_legends():
    # HABLOSS splits Nature; RECOVER does not. Both must land on one class.
    series = pd.Series(
        ["Nature - forest", "Nature - other", "Nature", "Cropland", "Artificial"]
    )
    assert list(coarsen(series)) == [
        "Nature", "Nature", "Nature", "Cropland", "Artificial"
    ]


def test_coarsen_is_case_and_whitespace_insensitive():
    assert list(coarsen(pd.Series(["  CROPLAND ", "nature - Other"]))) == [
        "Cropland",
        "Nature",
    ]


def test_coarsen_rejects_labels_outside_the_legend():
    # Silently mapping an unknown class to NaN would quietly shrink the sample.
    with pytest.raises(ValueError, match="outside the coarse legend"):
        coarsen(pd.Series(["Cropland", "Wetland"]))


def test_recover_long_form_nature_maps_to_nature():
    assert COARSE["nature (not cropland / artificial)"] == "Nature"


def test_blocked_splitter_never_splits_a_block_across_folds():
    rng = np.random.default_rng(0)
    n = 200
    groups = pd.Series(rng.integers(0, 10, n)).astype(str)
    target = pd.Series(rng.integers(0, 2, n)).astype(str)
    features = pd.DataFrame(rng.normal(size=(n, 3)))

    splitter = make_splitter("blocked", 5)
    for train_idx, test_idx in splitter.split(features, target, groups):
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])


def test_random_splitter_ignores_blocks():
    # The contrast with blocked CV is the whole point of offering both.
    rng = np.random.default_rng(0)
    n = 200
    groups = pd.Series(rng.integers(0, 10, n)).astype(str)
    target = pd.Series(rng.integers(0, 2, n)).astype(str)
    features = pd.DataFrame(rng.normal(size=(n, 3)))

    splitter = make_splitter("random", 5)
    shared = [
        bool(set(groups.iloc[tr]) & set(groups.iloc[te]))
        for tr, te in splitter.split(features, target, groups)
    ]
    assert any(shared)


def test_make_splitter_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown cv mode"):
        make_splitter("spatial-ish", 5)


def test_label_encoded_round_trips_string_classes():
    class Recorder:
        def fit(self, X, y):
            self.seen = y
            return self

        def predict(self, X):
            return self.seen[: len(X)]

    y = np.array(["b -> a", "a -> a", "b -> a"])
    model = LabelEncoded(Recorder).fit(np.zeros((3, 2)), y)
    # The wrapped estimator must see integers, the caller must get strings back.
    assert model.model.seen.dtype.kind in "iu"
    assert list(model.predict(np.zeros((3, 2)))) == list(y)


def test_ravelled_flattens_a_column_vector_prediction():
    class Column:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.array([["a"], ["b"]])

    assert list(Ravelled(Column).fit(None, None).predict(np.zeros((2, 1)))) == ["a", "b"]


def test_timee_adapter_emits_univariate_series():
    # timee-ts 0.1.0 unpacks exactly 3 dims in its forward pass, so the
    # (n, 64 channels, 2 timepoints) framing raises. Only (n, 1, L) runs.
    columns = ["A00_2018", "A00_2024", "A00_diff"]
    frame = pd.DataFrame(
        {"A00_2018": [0.1, 0.2], "A00_2024": [0.3, 0.4], "A00_diff": [0.2, 0.2]}
    )

    series = TimeeAdapter(columns)._series(frame)

    assert series.ndim == 3
    assert series.shape == (2, 1, 3)
    assert not TimeeAdapter.MULTICHANNEL_SUPPORTED


def test_postclassification_reads_the_transition_off_the_two_dates():
    from sklearn.neighbors import KNeighborsClassifier

    frame = postclass_frame()
    target = pd.Series(
        ["Nature -> Nature", "Nature -> Artificial",
         "Artificial -> Artificial", "Artificial -> Nature"]
    )

    model = PostClassification(
        COLUMNS, lambda: KNeighborsClassifier(n_neighbors=1)
    ).fit(frame, target.to_numpy())

    assert list(model.predict(frame)) == list(target)


def test_postclassification_never_sees_the_difference_bands():
    # A difference band is a two-date feature; a single-date classifier that
    # reads it is being handed the change signal it is supposed to derive.
    model = PostClassification(COLUMNS, lambda: None)
    assert model.columns_2018 == ["A00_2018", "A01_2018"]
    assert model.columns_2024 == ["A00_2024", "A01_2024"]


def test_postclassification_pools_transitions_outside_the_target():
    from sklearn.neighbors import KNeighborsClassifier

    frame = postclass_frame()
    # Artificial -> Nature was rare enough to be pooled; the cross-product
    # must not resurrect it as a class the direct models cannot predict.
    target = np.array(
        ["Nature -> Nature", "Nature -> Artificial",
         "Artificial -> Artificial", "other"]
    )

    model = PostClassification(
        COLUMNS, lambda: KNeighborsClassifier(n_neighbors=1)
    ).fit(frame, target)

    assert list(model.predict(frame)) == list(target)


def test_postclassification_shared_model_trains_on_both_epochs():
    class Recorder:
        def fit(self, X, y):
            self.n_rows = len(X)
            return self

        def predict(self, X):
            return np.array(["Nature"] * len(X))

    frame = postclass_frame()
    target = np.array(["Nature -> Nature"] * 4)

    shared = PostClassification(COLUMNS, Recorder, shared=True).fit(frame, target)
    separate = PostClassification(COLUMNS, Recorder, shared=False).fit(frame, target)

    assert shared.model_2018 is shared.model_2024
    assert shared.model_2018.n_rows == 8
    assert separate.model_2018 is not separate.model_2024
    assert separate.model_2018.n_rows == 4


def test_postclassification_requires_both_dates():
    with pytest.raises(ValueError, match="per-date embedding columns"):
        PostClassification(["A00_2018", "A00_diff"], lambda: None)


def test_is_change_label_treats_rare_pool_as_change():
    assert is_change_label("Nature -> Artificial")
    assert not is_change_label("Nature -> Nature")
    # The rare pool carries no '->' but collects off-diagonal transitions.
    assert is_change_label("other")


def hier_frame():
    """Two stable clusters and one change cluster, linearly separable."""
    X = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],      # Nature -> Nature
         [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],      # Artificial -> Artificial
         [0.0, 5.0], [0.1, 5.0], [0.0, 5.1]],     # Nature -> Artificial (change)
        dtype="float64",
    )
    y = np.array(
        ["Nature -> Nature"] * 3
        + ["Artificial -> Artificial"] * 3
        + ["Nature -> Artificial"] * 3
    )
    return X, y


def test_hierarchical_rebuilds_the_full_transition_label():
    from sklearn.neighbors import KNeighborsClassifier

    X, y = hier_frame()
    model = HierarchicalChange(lambda: KNeighborsClassifier(n_neighbors=1)).fit(X, y)
    # Separable by construction, so the gate + both branches recover every label.
    assert list(model.predict(X)) == list(y)


def test_hierarchical_change_branch_never_emits_a_stable_label():
    from sklearn.neighbors import KNeighborsClassifier

    X, y = hier_frame()
    model = HierarchicalChange(lambda: KNeighborsClassifier(n_neighbors=1)).fit(X, y)
    # The change classifier is trained on change plots only, so its label space
    # cannot contain a stable 'X -> X'.
    assert set(model.change_classes_ if hasattr(model, "change_classes_")
               else np.unique(model.change.predict(X))) <= {"Nature -> Artificial"}


def test_hierarchical_degenerate_branch_falls_back_to_constant():
    from sklearn.neighbors import KNeighborsClassifier

    # No change plots at all: the change branch sees zero classes and must not
    # raise -- it collapses to a constant that is simply never selected.
    X = np.array([[0.0], [0.1], [5.0], [5.1]], dtype="float64")
    y = np.array(["Nature -> Nature", "Nature -> Nature",
                  "Artificial -> Artificial", "Artificial -> Artificial"])
    model = HierarchicalChange(lambda: KNeighborsClassifier(n_neighbors=1)).fit(X, y)
    assert list(model.predict(X)) == list(y)


def test_hierarchical_registered_as_hier_base():
    columns = ["A00_2018", "A00_2024", "A00_diff"]
    registry = build_registry(columns, hier_bases=["lda"])
    assert "hier_lda" in registry
    factory, needs_frame = registry["hier_lda"]
    # A direct pixel model: it consumes the feature matrix, not the frame.
    assert needs_frame is False
    assert isinstance(factory(), HierarchicalChange)


def test_hierarchical_rejects_unknown_base():
    with pytest.raises(ValueError, match="hier base 'nope'"):
        build_registry(["A00_2018", "A00_2024"], hier_bases=["nope"])


def test_gated_targets_masks_each_head_to_its_rows():
    y = np.array(["Veg -> Veg", "Veg -> Artificial", "Artificial -> Artificial",
                  "other"])
    change_mask, stable, change, y_stable, y_change = gated_targets(y)

    assert list(change_mask) == [False, True, False, True]
    assert stable == ["Artificial", "Veg"]
    assert change == ["Veg -> Artificial", "other"]
    # Stable head ignores change rows; change head ignores stable rows.
    assert list(y_stable) == [stable.index("Veg"), -1, stable.index("Artificial"), -1]
    assert list(y_change) == [-1, change.index("Veg -> Artificial"), -1,
                              change.index("other")]


def test_hierarchical_nn_gate_threshold_moves_the_change_share():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalTorchNN

    X, y = hier_frame()
    X = np.repeat(X, 6, axis=0)
    y = np.repeat(y, 6)
    model = HierarchicalTorchNN(epochs=120, balanced=True).fit(X, y)
    p_change, change_label, stable_label = model.predict_parts(X)

    # A stricter gate can only shrink the set of plots called change.
    lenient = HierarchicalTorchNN.combine(p_change, change_label, stable_label, 0.2)
    strict = HierarchicalTorchNN.combine(p_change, change_label, stable_label, 0.8)
    n_change = lambda labels: sum(is_change_label(v) for v in labels)  # noqa: E731
    assert n_change(strict) <= n_change(lenient)


def test_temporal_model_needs_at_least_two_years():
    from model_zoo import TemporalTransitionNN

    with pytest.raises(ValueError, match="needs >=2"):
        TemporalTransitionNN._year_columns(["A00_2018", "A00_diff"])


def test_temporal_model_orders_the_year_sequence():
    from model_zoo import TemporalTransitionNN

    columns = ["A01_2024", "A00_2018", "A00_2024", "A01_2018", "A00_diff"]
    year_columns = TemporalTransitionNN._year_columns(columns)
    # Years time-ordered, channels in A00..A63 order, diff bands excluded.
    assert list(year_columns) == ["2018", "2024"]
    assert year_columns["2018"] == ["A00_2018", "A01_2018"]
    assert year_columns["2024"] == ["A00_2024", "A01_2024"]


def test_temporal_model_runs_end_to_end_on_a_trajectory():
    pytest.importorskip("torch")
    from model_zoo import TemporalTransitionNN

    # Three-year toy trajectory: a change cluster drifts, stable clusters do not.
    rng = np.random.default_rng(0)
    years = ["2018", "2019", "2020"]
    columns = [f"A{i:02d}_{y}" for y in years for i in range(2)]
    n = 60
    data = {}
    label = np.array((["Veg -> Veg"] * 2 + ["Veg -> Artificial"]) * (n // 3))
    for y in years:
        for i in range(2):
            base = np.where(label == "Veg -> Artificial", 3.0, 0.0)
            data[f"A{i:02d}_{y}"] = base + rng.normal(0, 0.1, n)
    frame = pd.DataFrame(data)

    model = TemporalTransitionNN(columns, epochs=60).fit(frame, label)
    predicted = model.predict(frame)
    valid = set(model.change_classes_) | {f"{c} -> {c}" for c in model.stable_classes_}
    assert set(predicted) <= valid
    assert len(predicted) == n


def test_to_merged_label_collapses_vegetation():
    from model_zoo import to_merged_label

    assert to_merged_label("Nature -> Artificial") == "Vegetation -> Artificial"
    assert to_merged_label("Cropland -> Artificial") == "Vegetation -> Artificial"
    # The interpreter-noise boundary collapses to a stable class.
    assert to_merged_label("Nature -> Cropland") == "Vegetation -> Vegetation"
    assert to_merged_label("Artificial -> Artificial") == "Artificial -> Artificial"


def test_class_weights_modes():
    from model_zoo import class_weights

    t = np.array([0, 0, 0, 0, 1])  # class 1 is rare
    inv = class_weights(t, 2, "inverse")
    assert inv[1] > inv[0]                      # rare class up-weighted
    assert np.allclose(class_weights(t, 2, "none"), [1.0, 1.0])
    eff = class_weights(t, 2, "effective")
    assert eff[1] > eff[0]


def test_hierarchical_softmax_levels_are_nested_and_consistent():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN, to_merged_label

    columns = ["A00_2018", "A00_2024", "A00_diff"]
    rng = np.random.default_rng(0)
    # Nature/Cropland separable from Artificial; the Nature/Cropland split is noise.
    labels = np.array((["Nature -> Nature", "Cropland -> Cropland",
                        "Nature -> Artificial", "Artificial -> Artificial"]) * 20)
    frame = pd.DataFrame({
        "A00_2018": np.where(np.isin(labels, ["Artificial -> Artificial"]), 5.0, 0.0)
        + rng.normal(0, 0.1, len(labels)),
        "A00_2024": np.where(
            [l.endswith("Artificial") for l in labels], 5.0, 0.0
        ) + rng.normal(0, 0.1, len(labels)),
        "A00_diff": rng.normal(0, 0.1, len(labels)),
    })
    model = HierarchicalSoftmaxNN(columns, arch="mlp", loss="ce", epochs=60).fit(
        frame, labels
    )
    fine = model.predict(frame)
    merged = model.predict_merged(frame)
    # merged2 read is available and every fine label lives under its merged parent.
    assert set(merged) <= set(model.merged_classes_)
    valid_parent = {c: to_merged_label(c) for c in model.fine_classes_}
    assert all(m in model.merged_classes_ for m in valid_parent.values())


@pytest.mark.parametrize("arch,loss", [
    ("mlp", "focal"), ("wide", "cb_focal"), ("deep_res", "weighted_ce"),
])
def test_hierarchical_softmax_variants_run(arch, loss):
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN

    columns = ["A00_2018", "A00_2024", "A00_diff"]
    rng = np.random.default_rng(1)
    labels = np.array((["Nature -> Nature", "Nature -> Artificial",
                        "Artificial -> Artificial"]) * 20)
    frame = pd.DataFrame({c: rng.normal(0, 1, len(labels)) for c in columns})
    model = HierarchicalSoftmaxNN(columns, arch=arch, loss=loss, epochs=40).fit(
        frame, labels
    )
    assert len(model.predict(frame)) == len(labels)
    assert len(model.predict_merged(frame)) == len(labels)


def _softmax_frame(n_per=20):
    """Toy coarse3 frame: Artificial separable, Nature/Cropland overlapping."""
    rng = np.random.default_rng(3)
    labels = np.array((["Nature -> Nature", "Cropland -> Cropland",
                        "Nature -> Artificial", "Artificial -> Artificial"]) * n_per)
    to_art = np.array([l.endswith("Artificial") for l in labels], float)
    frame = pd.DataFrame({
        "A00_2018": np.where(labels == "Artificial -> Artificial", 4.0, 0.0)
        + rng.normal(0, 0.2, len(labels)),
        "A00_2024": 4.0 * to_art + rng.normal(0, 0.2, len(labels)),
        "A00_diff": rng.normal(0, 0.2, len(labels)),
    })
    return ["A00_2018", "A00_2024", "A00_diff"], frame, labels


def test_bilinear_head_runs_and_predicts():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN

    columns, frame, labels = _softmax_frame()
    model = HierarchicalSoftmaxNN(columns, head="bilinear", epochs=40).fit(frame, labels)
    # Factorised head still emits the full transition space, and merged reads.
    assert model.base_classes_ == ["Artificial", "Cropland", "Nature"]
    assert len(model.predict(frame)) == len(labels)
    assert set(model.predict_merged(frame)) <= set(model.merged_classes_)


def test_noise_matrix_is_a_valid_confusion_matrix():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN

    # A flip-closed set (all 9 Nature/Cropland/Artificial transitions), as at the
    # default min-class-count, so no mass leaks outside the modelled classes.
    bases = ["Nature", "Cropland", "Artificial"]
    labels = np.array([f"{a} -> {b}" for a in bases for b in bases])
    model = HierarchicalSoftmaxNN(["A00_2018", "A00_2024"], noise_rate=0.2, epochs=1)
    model.device = "cpu"
    model._build_hierarchy(labels)
    T = model._noise_matrix()
    # Rows are conditional distributions P(observe . | true k): non-negative, sum 1.
    assert np.all(T >= 0)
    assert np.allclose(T.sum(1), 1.0)
    # Artificial->Artificial cannot be observed as anything else (no flip).
    aa = model.fine_classes_.index("Artificial -> Artificial")
    assert T[aa, aa] == pytest.approx(1.0)
    # Nature->Nature leaks to Cropland->Cropland via two independent flips.
    nn_i = model.fine_classes_.index("Nature -> Nature")
    cc_i = model.fine_classes_.index("Cropland -> Cropland")
    assert T[nn_i, cc_i] == pytest.approx(0.2 * 0.2)


def test_forward_correction_trains_and_predicts():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN

    columns, frame, labels = _softmax_frame()
    model = HierarchicalSoftmaxNN(columns, noise_rate=0.2, loss="ce",
                                  epochs=40).fit(frame, labels)
    assert len(model.predict(frame)) == len(labels)
    # At inference the fine head is the clean posterior (T not applied).
    assert model._T is not None


def test_semi_supervised_pseudo_labels_the_unlabelled_frame():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN

    columns, frame, labels = _softmax_frame()
    unlabelled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    model = HierarchicalSoftmaxNN(columns, ssl="fixmatch", ssl_threshold=0.6,
                                  epochs=40)
    model.fit(frame, labels, unlabelled_frame=unlabelled)
    assert len(model.predict_merged(unlabelled)) == len(unlabelled)


def test_hierarchical_softmax_gru_uses_the_trajectory():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalSoftmaxNN

    years = ["2018", "2019", "2020"]
    columns = [f"A{i:02d}_{y}" for y in years for i in range(2)]
    rng = np.random.default_rng(2)
    labels = np.array((["Nature -> Nature", "Nature -> Artificial",
                        "Artificial -> Artificial"]) * 20)
    frame = pd.DataFrame({c: rng.normal(0, 1, len(labels)) for c in columns})
    model = HierarchicalSoftmaxNN(columns, arch="gru", loss="ce", epochs=40).fit(
        frame, labels
    )
    assert list(model.year_columns) == years
    assert len(model.predict_merged(frame)) == len(labels)


def test_hierarchical_nn_end_to_end_gate_then_transition():
    pytest.importorskip("torch")
    from model_zoo import HierarchicalTorchNN

    X, y = hier_frame()
    # Repeat so batch-norm has enough rows and the clusters stay separable.
    X = np.repeat(X, 6, axis=0)
    y = np.repeat(y, 6)
    model = HierarchicalTorchNN(epochs=150, balanced=True).fit(X, y)

    predicted = model.predict(X)
    # Every prediction is a valid transition label from the two head spaces.
    valid = set(model.change_classes_) | {f"{c} -> {c}" for c in model.stable_classes_}
    assert set(predicted) <= valid
    # The change cluster must be found -- that is the whole point of the gate.
    assert (np.array([is_change_label(p) for p in predicted])).any()


def test_scores_reports_change_detection_separately():
    stable_fn = lambda label: label.split(" -> ")[0] != label.split(" -> ")[1]  # noqa: E731
    truth = np.array(["A -> A", "A -> B", "B -> B", "A -> B"])
    # Predicts stable everywhere: high accuracy, zero change recall.
    predicted = np.array(["A -> A", "A -> A", "B -> B", "B -> B"])

    result = scores(truth, predicted, stable_fn)

    assert result["accuracy"] == pytest.approx(0.5)
    assert result["change_recall"] == pytest.approx(0.0)
    assert result["change_f1"] == pytest.approx(0.0)
