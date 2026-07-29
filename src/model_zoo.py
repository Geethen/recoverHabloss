"""Broad model comparison for transition classification from embeddings.

Extends ``model_transitions.py`` from two models to a full leaderboard: the
classical sweep a PyCaret ``compare_models`` call would run, plus the tabular
foundation models (TabPFN, TabICL), a time-series in-context model (TIMEE), and
a torch DNN. Every model is scored under the same spatially blocked CV, so the
numbers are comparable to the two already reported.

Three things make this comparison different from a default tabular benchmark:

* **The legend is harmonised to the coarse 3-class one.** HABLOSS splits Nature
  into forest/other; RECOVER does not. Only Nature / Cropland / Artificial is
  common to all three sample sources, and it is the legend the 3x3 transition
  matrix is defined on.
* **Duplicate PLOTIDs are dropped.** 76 plots appear twice -- 54 RECOVER
  reverifications and 22 HABLOSS dual-frame overlaps -- at identical
  coordinates. Keeping both puts the same location in train and test.
* **Change detection is scored separately.** Accuracy is dominated by the
  stable diagonal; the estimators only consume the off-diagonal, so the
  leaderboard reports change recall/precision/F1 alongside the usual metrics.

Models whose package is not installed are skipped and recorded as such, so the
script runs to completion on a partial environment.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

try:  # keeps the script runnable from outside src
    from project_paths import project_data_dir
except ImportError:  # pragma: no cover
    def project_data_dir(*parts: str) -> Path:
        return Path(__file__).resolve().parents[1].joinpath("data", *parts)


DEFAULT_INPUT = project_data_dir("embeddings", "embeddings_habloss_recover.parquet")
DEFAULT_OUTPUT = project_data_dir("analysis_results")
RARE_LABEL = "other"
SEED = 20250717

# The only legend common to habloss_main, habloss_landwater and recover.
COARSE = {
    "nature": "Nature",
    "nature - forest": "Nature",
    "nature - other": "Nature",
    "nature (not cropland / artificial)": "Nature",
    "cropland": "Cropland",
    "artificial": "Artificial",
}


def coarsen(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(cleaned) - set(COARSE))
    if unknown:
        raise ValueError(f"Land-cover values outside the coarse legend: {unknown}")
    return cleaned.map(COARSE)


def feature_columns(frame: pd.DataFrame, use_diff: bool = True) -> list[str]:
    columns = [c for c in frame.columns if c.startswith("A") and "_" in c]
    if not use_diff:
        columns = [c for c in columns if not c.endswith("_diff")]
    if not columns:
        raise ValueError("No embedding feature columns found")
    return sorted(columns)


def load(path: Path, min_count: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    frame = pd.read_parquet(path)

    before, after = coarsen(frame["lc_2018"]), coarsen(frame["lc_2024"])
    frame = frame.assign(transition=before + " -> " + after)

    # Identical coordinates in train and test would leak; drop the repeats.
    duplicated = int(frame["PLOTID"].duplicated().sum())
    if duplicated:
        print(f"Dropping {duplicated} duplicate PLOTID rows (reverified / dual-frame)")
        frame = frame.drop_duplicates("PLOTID").reset_index(drop=True)

    columns = feature_columns(frame)
    complete = frame[columns].notna().all(axis=1)
    if not complete.all():
        print(f"Dropping {int((~complete).sum())} plots with missing embeddings")
        frame = frame.loc[complete].reset_index(drop=True)

    # Parquet round-trips these as nullable extension dtypes, which reach
    # sklearn as object arrays and break any estimator that calls isnan.
    frame[columns] = frame[columns].astype("float64")

    counts = frame["transition"].value_counts()
    keep = set(counts[counts >= min_count].index)
    target = frame["transition"].where(frame["transition"].isin(keep), RARE_LABEL)
    return frame, target, frame["block_id"]


def make_splitter(cv: str, n_splits: int):
    """Blocked CV holds whole spatial blocks out; random CV does not.

    The sample is stratified random, so random CV is a legitimate estimate of
    performance at sampled locations. The gap between the two is the spatial
    autocorrelation optimism -- how much of a random-CV score comes from having
    a near neighbour of the test plot in training.
    """
    if cv == "blocked":
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED
        )
    if cv == "random":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    raise ValueError(f"Unknown cv mode: {cv}")


class Ravelled:
    """Flatten a predictor that returns an (n, 1) column instead of (n,)."""

    def __init__(self, factory):
        self.model = factory()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()


class LabelEncoded:
    """Wrap an estimator that requires contiguous integer class labels."""

    def __init__(self, factory):
        self.model = factory()

    def fit(self, X, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.model.fit(X, encoded)
        return self

    def predict(self, X):
        return self.classes_[self.model.predict(X).astype(int)]


class TunedLDA(LinearDiscriminantAnalysis):
    """LDA carrying the F1-tuned settings from ``tune_lda.py``.

    Two things the plain ``LinearDiscriminantAnalysis()`` default leaves on the
    table, both found by the sweep under the same spatially blocked CV:

    * **Covariance shrinkage.** The SVD solver cannot shrink; ``lsqr`` with a
      non-zero shrinkage regularises the per-class covariance, which is what
      stabilises the discriminant directions given 192 correlated channels
      relative to the per-class sample size.
    * **Uniform priors.** Selected with ``prior_mode="uniform"`` and resolved to
      an explicit ``priors`` array once the classes are known at fit time, since
      the class count is not known before then. Uniform priors stop the
      stable-diagonal majority from swamping the rare change classes, which is
      where macro-F1 and change-F1 are won.

    Subclasses ``LinearDiscriminantAnalysis`` so it drops straight into an
    sklearn ``Pipeline`` (get_params/clone and the estimator tags come for
    free); only the prior resolution is added on top.
    """

    def __init__(self, solver="lsqr", shrinkage=0.3, prior_mode=None):
        self.prior_mode = prior_mode
        super().__init__(solver=solver, shrinkage=shrinkage)

    def fit(self, X, y):
        if self.prior_mode == "uniform":
            n_classes = len(np.unique(y))
            self.priors = np.full(n_classes, 1.0 / n_classes)
        else:
            self.priors = None
        return super().fit(X, y)


def is_change_label(label: str) -> bool:
    """A transition is change if the two dates differ, or it is the rare pool.

    ``RARE_LABEL`` collects the off-diagonal transitions too thin to model as
    their own class, so it counts as change even though it carries no ``->``.
    """
    return label == RARE_LABEL or label.split(" -> ")[0] != label.split(" -> ")[1]


def gated_targets(y: np.ndarray):
    """Split a transition target into the three sub-targets a gated model needs.

    Shared by every detect-then-name network so the gate, stable-class head and
    change-transition head are defined identically wherever they appear.
    Returns ``(change_mask, stable_classes, change_classes, y_stable,
    y_change)`` where the two head targets carry ``-1`` on the rows the head
    does not apply to (masked out of its loss via ``ignore_index``).
    """
    y = np.asarray(y)
    change_mask = np.array([is_change_label(t) for t in y])
    stable_classes = sorted({t.split(" -> ")[0] for t in y[~change_mask]})
    change_classes = sorted(set(y[change_mask]))
    stable_code = {c: i for i, c in enumerate(stable_classes)}
    change_code = {c: i for i, c in enumerate(change_classes)}

    stable_lc = np.array([t.split(" -> ")[0] for t in y])
    y_stable = np.array(
        [stable_code[c] if not chg else -1
         for c, chg in zip(stable_lc, change_mask)]
    )
    y_change = np.array(
        [change_code[t] if chg else -1 for t, chg in zip(y, change_mask)]
    )
    return change_mask, stable_classes, change_classes, y_stable, y_change


def class_weights(target: np.ndarray, n_classes: int, mode: str) -> np.ndarray:
    """Per-class loss weights: inverse frequency, or class-balanced effective number.

    ``inverse`` is ``N / (C * n_c)``; ``effective`` is the Cui et al. (2019)
    class-balanced weight ``(1 - beta) / (1 - beta^{n_c})`` (beta 0.999),
    normalised to mean 1 so it does not also rescale the learning rate. ``none``
    returns ones.
    """
    counts = np.bincount(target[target >= 0], minlength=n_classes).astype("float64")
    if mode == "inverse":
        w = len(target[target >= 0]) / (n_classes * np.maximum(counts, 1))
    elif mode == "effective":
        beta = 0.999
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / np.maximum(eff, 1e-8)
        w = w / w.mean()
    else:
        w = np.ones(n_classes)
    return w.astype("float32")


def level_loss(probs, target, mode: str, weight=None, gamma: float = 2.0):
    """Cross-entropy or focal loss taken over already-aggregated probabilities.

    The hierarchical model reads its coarse levels off group-summed softmax
    probabilities, not raw logits, so the loss is built from ``probs`` directly:
    ``-log p_t`` for cross-entropy, ``-(1 - p_t)^gamma log p_t`` for focal.
    ``weight`` (per class) multiplies the per-sample loss when supplied.
    """
    import torch

    pt = probs.gather(1, target[:, None]).squeeze(1).clamp_min(1e-8)
    loss = -pt.log()
    if mode in ("focal", "cb_focal"):
        loss = ((1.0 - pt) ** gamma) * loss
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean()


def scores(truth, predicted, is_change_fn) -> dict:
    """Standard multiclass metrics plus the collapsed change-detection view."""
    truth_change = np.array([is_change_fn(t) for t in truth])
    pred_change = np.array([is_change_fn(p) for p in predicted])
    return {
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "f1_macro": f1_score(truth, predicted, average="macro", zero_division=0),
        "change_recall": recall_score(truth_change, pred_change, zero_division=0),
        "change_precision": precision_score(truth_change, pred_change, zero_division=0),
        "change_f1": f1_score(truth_change, pred_change, zero_division=0),
    }


# --------------------------------------------------------------------------
# Model constructors. Each returns (name, factory) or None when unavailable.
# --------------------------------------------------------------------------

def classical_models(balanced: bool) -> dict:
    """The sweep PyCaret's compare_models covers, built directly on sklearn.

    PyCaret itself is not used: it pins older numpy/pandas than this project
    runs, and compare_models is a loop over these same estimators.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    weight = "balanced" if balanced else None
    scaled = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731

    models = {
        "dummy_majority": lambda: DummyClassifier(strategy="most_frequent"),
        "logistic": lambda: scaled(
            LogisticRegression(max_iter=2000, C=1.0, class_weight=weight,
                               random_state=SEED)
        ),
        "ridge": lambda: scaled(RidgeClassifier(class_weight=weight, random_state=SEED)),
        # F1-tuned LDA (tune_lda.py). The direct 7-way problem and the per-date
        # 3-class post-classification problem regularise differently, so they
        # carry different optima: 'lda' is the direct winner (0.3 shrinkage,
        # empirical priors unless --balanced); 'lda_pc' is the post-class winner
        # (Ledoit-Wolf shrinkage, uniform priors) and is the default post-class
        # base below.
        "lda": lambda: scaled(
            TunedLDA(solver="lsqr", shrinkage=0.3,
                     prior_mode="uniform" if balanced else None)
        ),
        "lda_pc": lambda: scaled(
            TunedLDA(solver="lsqr", shrinkage="auto", prior_mode="uniform")
        ),
        "naive_bayes": lambda: scaled(GaussianNB()),
        "knn": lambda: scaled(KNeighborsClassifier(n_neighbors=15)),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=12, class_weight=weight, random_state=SEED
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, n_jobs=-1, class_weight=weight, random_state=SEED
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=500, n_jobs=-1, class_weight=weight, random_state=SEED
        ),
        "hist_gbm": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, early_stopping=True,
            validation_fraction=0.15, random_state=SEED,
            class_weight=weight,
        ),
        # Label-encoded: with early_stopping, sklearn 1.8 scores the validation
        # split via np.isnan(y_pred), which raises on string class labels.
        "mlp_sklearn": lambda: LabelEncoded(
            lambda: scaled(
                MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=600,
                              early_stopping=True, random_state=SEED)
            )
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = lambda: LabelEncoded(
            # device is pinned to CPU: XGBoost 3.x otherwise auto-selects the
            # GPU, and this host's vGPU does not implement the CUDA memory
            # granularity call it needs (CUDA_ERROR_NOT_SUPPORTED).
            lambda: XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                device="cpu", n_jobs=-1, random_state=SEED,
            )
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            class_weight=weight, n_jobs=-1, random_state=SEED, verbose=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = lambda: Ravelled(
            lambda: CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.06, verbose=0,
                random_seed=SEED,
                auto_class_weights="Balanced" if balanced else None,
            )
        )
    except ImportError:
        pass

    return models


class TorchDNN:
    """Residual MLP over the embedding vector, with class-weighted loss.

    Deliberately small: 6k plots and 192 dense features do not support a large
    network, and the point of including it is to test whether depth adds
    anything over the linear probe, not to win a leaderboard.
    """

    def __init__(self, epochs: int = 120, balanced: bool = True):
        self.epochs = epochs
        self.balanced = balanced

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = ((X - self.mu) / self.sd).astype("float32")

        torch.manual_seed(SEED)
        d, k = Xs.shape[1], len(self.classes_)
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, k),
        ).to(self.device)

        weight = None
        if self.balanced:
            counts = np.bincount(y_idx, minlength=k).astype("float32")
            weight = torch.tensor(
                (len(y_idx) / (k * np.maximum(counts, 1))), device=self.device
            )
        loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-2)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2e-3, total_steps=self.epochs
        )

        Xt = torch.tensor(Xs, device=self.device)
        yt = torch.tensor(y_idx, device=self.device, dtype=torch.long)
        self.net.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            loss = loss_fn(self.net(Xt), yt)
            loss.backward()
            optimiser.step()
            schedule.step()
        return self

    def predict(self, X):
        import torch

        Xs = ((X - self.mu) / self.sd).astype("float32")
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(Xs, device=self.device))
        return self.classes_[logits.argmax(1).cpu().numpy()]


class TimeeAdapter:
    """TIMEE, a pretrained in-context time-series classifier.

    The framing TIMEE was picked for -- 64 AlphaEarth channels observed at two
    timepoints, shape (n, 64, 2) -- is **not supported by timee-ts 0.1.0**. Its
    forward pass unpacks exactly three dimensions from the batched input, so any
    multichannel series raises "too many values to unpack"; only univariate
    (n, 1, seq_len) input runs, despite the docstring advertising n_channels.

    The fallback flattens the whole feature vector into one length-192
    pseudo-series. That axis is feature order, not time, so this row tests the
    pretrained model's general in-context ability on this data, **not** whether
    a temporal framing helps. A weak score here is not evidence against
    time-series models for this problem.

    Memory is the other constraint: the full ~5,100-plot fold as in-context
    training data exceeds 24 GB of GPU. The context is therefore capped by
    class-stratified subsampling, which is a real handicap relative to the
    models that train on the whole fold -- another reason to read a weak TIMEE
    score as "not tested fairly" rather than "does not work".
    """

    #: Whether the extractor could use the real (channels, time) framing.
    MULTICHANNEL_SUPPORTED = False

    def __init__(self, columns: list[str], context: int = 1200):
        self.columns = columns
        self.context = context

    def _series(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[self.columns].to_numpy("float32")
        return values[:, None, :]  # (n, 1, seq_len)

    def _subsample(self, frame: pd.DataFrame, y: np.ndarray):
        """Class-stratified cap on the in-context training set."""
        if len(frame) <= self.context:
            return frame, y
        rng = np.random.default_rng(SEED)
        share = self.context / len(frame)
        keep = []
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            # At least two per class, so no class drops out of the context.
            take = max(2, int(round(len(idx) * share)))
            keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
        selected = np.sort(np.concatenate(keep))
        return frame.iloc[selected], y[selected]

    def fit(self, frame, y):
        from timee import TimeeClassifier

        self.clf = TimeeClassifier.from_pretrained(use_ensemble=False)
        frame, y = self._subsample(frame, np.asarray(y))
        self.X, self.y = self._series(frame), y
        return self

    def predict(self, frame):
        out = self.clf.predict(self.X, self.y, self._series(frame))
        return np.asarray(out[0] if isinstance(out, tuple) else out)


class PostClassification:
    """Classify each date on its own, then read the transition off the pair.

    The direct models above learn the transition as one 7-way label from the
    joint 2018+2024+difference vector. This is the classical remote-sensing
    alternative: label 2018 from the 2018 embedding, label 2024 from the 2024
    embedding, and declare change wherever the two labels differ.

    Two things follow from that framing, and they cut in opposite directions:

    * The per-date problem is a 3-class one with the full sample behind every
      class, so ``Nature -> Artificial`` is no longer a 383-example class --
      it is 6,000 examples of Nature and 6,000 of Artificial.
    * Errors compound on the diagonal. A stable plot is called changed if
      *either* date is wrong, so a per-date accuracy of ``p`` gives roughly
      ``p^2`` on the stable class, which is where most plots are. Post
      classification is the textbook example of a method with good per-date
      accuracy and poor change precision.

    With ``shared=True`` (the default) one classifier is trained on both epochs
    stacked -- the usual practice, and it doubles the training rows -- which
    also means the same spectral signature gets the same label in both years.
    ``shared=False`` fits an independent model per date, which can absorb
    sensor/phenology drift between the epochs at the cost of half the data and
    of letting the two dates disagree for reasons other than real change.

    The difference bands are deliberately unused: they are a two-date feature,
    and feeding them to a single-date classifier is the leak this comparison
    exists to avoid.
    """

    def __init__(self, columns: list[str], base_factory, shared: bool = True):
        self.columns_2018 = [c for c in columns if c.endswith("_2018")]
        self.columns_2024 = [c for c in columns if c.endswith("_2024")]
        if not self.columns_2018 or not self.columns_2024:
            raise ValueError(
                "Post classification needs per-date embedding columns "
                "('A00_2018' / 'A00_2024'); found "
                f"{len(self.columns_2018)} and {len(self.columns_2024)}"
            )
        if len(self.columns_2018) != len(self.columns_2024):
            raise ValueError(
                "The two dates carry different numbers of bands "
                f"({len(self.columns_2018)} vs {len(self.columns_2024)}); a "
                "shared classifier would see mismatched features"
            )
        self.base_factory = base_factory
        self.shared = shared

    def __getstate__(self):
        # ``base_factory`` is a construction-time lambda (often a closure over
        # classical_models locals, which pickle cannot reach). Prediction only
        # needs the fitted per-date models + retained set, so drop it: a
        # persisted PostClassification predicts but cannot be re-fit.
        return {**self.__dict__, "base_factory": None}

    def fit(self, frame: pd.DataFrame, y):
        x18 = frame[self.columns_2018].to_numpy()
        x24 = frame[self.columns_2024].to_numpy()
        y18, y24 = coarsen(frame["lc_2018"]), coarsen(frame["lc_2024"])

        if self.shared:
            # Column names differ by date, so stack as arrays: position i is
            # the same embedding channel in both epochs.
            model = self.base_factory()
            model.fit(np.vstack([x18, x24]), np.concatenate([y18, y24]))
            self.model_2018 = self.model_2024 = model
        else:
            self.model_2018 = self.base_factory().fit(x18, y18.to_numpy())
            self.model_2024 = self.base_factory().fit(x24, y24.to_numpy())

        # Transitions the pooled target does not carry as their own class have
        # to land on RARE_LABEL, or the cross-product would score against
        # labels the direct models were never allowed to predict.
        self.retained_ = set(np.unique(y)) - {RARE_LABEL}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_from_arrays(
            frame[self.columns_2018].to_numpy(),
            frame[self.columns_2024].to_numpy(),
        )

    def predict_from_arrays(self, x18: np.ndarray, x24: np.ndarray) -> np.ndarray:
        """Transition labels from raw per-date feature arrays.

        The array entry point ``predict`` funnels through, so wall-to-wall pixel
        inference can pass the 2018 / 2024 embedding matrices directly (columns
        in ``columns_2018`` / ``columns_2024`` order) without materialising a
        DataFrame of millions of rows. Pairs the pooled target never carried as
        their own class collapse to ``RARE_LABEL``, exactly as in training.
        """
        before = self.model_2018.predict(x18)
        after = self.model_2024.predict(x24)
        pairs = np.char.add(np.char.add(before.astype(str), " -> "), after.astype(str))
        return np.array(
            [p if p in self.retained_ else RARE_LABEL for p in pairs], dtype=object
        )


class _Constant:
    """Predict a fixed label -- the fallback when a branch sees one class.

    A spatially blocked fold can hand a sub-problem a single class (all its
    change plots share one transition, say). Most sklearn estimators refuse to
    fit on one class, so a degenerate branch collapses to this instead of
    sinking the whole model.
    """

    def __init__(self, value):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([self.value] * len(X), dtype=object)


def _fit_branch(factory, X, y):
    """Fit ``factory()`` on a sub-problem, or a constant if it is degenerate."""
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return _Constant(classes[0] if len(classes) else RARE_LABEL)
    return factory().fit(X, y)


class HierarchicalChange:
    """Detect change first, then name the transition -- a two-stage classifier.

    The rare-class problem is that the off-diagonal transitions are swamped by
    the stable diagonal: ``Nature -> Artificial`` is a few-hundred-example class
    competing against thousands of stable plots. Splitting the decision defers
    that imbalance. Stage 1 is a binary change / no-change model trained on
    every plot -- a balanced-enough problem. Stage 2 only ever sees the plots
    stage 1 called change, where the off-diagonal transitions are no longer rare
    *relative to each other*, so the model separating them is not also fighting
    the stable majority.

    A change / no-change split alone cannot rebuild the full transition label:
    the no-change branch still has to say *which* stable class. A small 3-way
    stable-class head does that (``Nature`` -> ``Nature -> Nature``), so every
    plot lands on exactly one of the transition classes the direct models emit
    and the leaderboard scores the two framings on identical labels.

    One ``base_factory`` builds all three sub-models (binary gate, change
    classifier, stable classifier), which is what makes ``hier_<base>``
    comparable to the direct ``<base>``: same estimator, restructured decision.
    """

    def __init__(self, base_factory):
        self.base_factory = base_factory

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        change_mask = np.array([is_change_label(t) for t in y])

        binary = np.where(change_mask, "change", "stable")
        self.binary = _fit_branch(self.base_factory, X, binary)

        # Change branch: the off-diagonal transitions (+ RARE_LABEL) only.
        self.change = _fit_branch(self.base_factory, X[change_mask], y[change_mask])

        # Stable branch: the single land cover of each no-change plot, which the
        # gate cannot supply. before == after there, so either token works.
        stable_lc = np.array([t.split(" -> ")[0] for t in y[~change_mask]])
        self.stable = _fit_branch(self.base_factory, X[~change_mask], stable_lc)
        return self

    def predict(self, X):
        X = np.asarray(X)
        gate = np.asarray(self.binary.predict(X))
        out = np.empty(len(X), dtype=object)

        change_mask = gate == "change"
        if change_mask.any():
            out[change_mask] = np.asarray(
                self.change.predict(X[change_mask]), dtype=object
            )
        if (~change_mask).any():
            lc = np.asarray(self.stable.predict(X[~change_mask]))
            out[~change_mask] = np.array([f"{c} -> {c}" for c in lc], dtype=object)
        return out


class HierarchicalTorchNN:
    """The two-stage detect-then-name process as one end-to-end network.

    ``HierarchicalChange`` fits three independent models; this collapses them
    into a single network so the representation is *shared*. A trunk feeds three
    heads -- a binary change gate, a stable-class head, and a change-transition
    head -- and the loss is the sum of the three, each masked to the plots it
    applies to (the stable head is scored only on stable plots, the change head
    only on change plots). The trunk therefore learns one embedding that serves
    both the change decision and the transition it implies.

    At inference the gate picks the head: gate says change -> the change head
    names the transition; gate says stable -> the stable head names the land
    cover, rebuilt as ``X -> X``. That is the same detect-then-assign flow as the
    two-stage model, but in one forward pass with a jointly trained trunk, so any
    gain over ``hier_<base>`` is attributable to sharing the representation.

    The change head carries the inverse-frequency class weights (``balanced``),
    since that is the head where the rare transitions live; the gate uses a
    positive-class weight so change is not drowned by the stable majority.
    """

    def __init__(self, epochs: int = 200, balanced: bool = True,
                 gate_threshold: float = 0.5):
        self.epochs = epochs
        self.balanced = balanced
        self.gate_threshold = gate_threshold

    def _build(self, d: int):
        import torch.nn as nn

        trunk = nn.Sequential(
            nn.Linear(d, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(),
        )
        gate_head = nn.Linear(128, 1)
        stable_head = nn.Linear(128, len(self.stable_classes_))
        change_head = nn.Linear(128, len(self.change_classes_))
        return trunk, gate_head, stable_head, change_head

    def _heads(self, Xt):
        h = self.trunk(Xt)
        return self.gate_head(h), self.stable_head(h), self.change_head(h)

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        (change_mask, self.stable_classes_, self.change_classes_,
         y_stable, y_change) = gated_targets(y)

        self.mu, self.sd = X.mean(0), X.std(0) + 1e-8
        Xs = ((X - self.mu) / self.sd).astype("float32")

        torch.manual_seed(SEED)
        self.trunk, self.gate_head, self.stable_head, self.change_head = self._build(
            Xs.shape[1]
        )
        for module in (self.trunk, self.gate_head, self.stable_head, self.change_head):
            module.to(self.device)

        change_weight = None
        pos_weight = None
        if self.balanced:
            k = len(self.change_classes_)
            counts = np.bincount(y_change[y_change >= 0], minlength=k).astype("float32")
            change_weight = torch.tensor(
                len(y_change) / (k * np.maximum(counts, 1)), device=self.device
            )
            n_pos = float(change_mask.sum())
            n_neg = float((~change_mask).sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)

        gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        stable_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
        change_loss_fn = nn.CrossEntropyLoss(
            weight=change_weight, ignore_index=-1, label_smoothing=0.05
        )

        params = [
            p for m in (self.trunk, self.gate_head, self.stable_head, self.change_head)
            for p in m.parameters()
        ]
        optimiser = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2e-3, total_steps=self.epochs
        )

        Xt = torch.tensor(Xs, device=self.device)
        gate_t = torch.tensor(change_mask.astype("float32"), device=self.device)
        stable_t = torch.tensor(y_stable, device=self.device, dtype=torch.long)
        change_t = torch.tensor(y_change, device=self.device, dtype=torch.long)

        for module in (self.trunk, self.gate_head, self.stable_head, self.change_head):
            module.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            gate_logit, stable_logits, change_logits = self._heads(Xt)
            loss = (
                gate_loss_fn(gate_logit.squeeze(1), gate_t)
                + stable_loss_fn(stable_logits, stable_t)
                + change_loss_fn(change_logits, change_t)
            )
            loss.backward()
            optimiser.step()
            schedule.step()
        return self

    def predict_parts(self, X):
        """Gate probability and each head's label, before the gate decides.

        Returned as ``(p_change, change_label, stable_label)`` so a threshold
        sweep can recombine the transition at any operating point without
        refitting -- ``experiment_gate_threshold.py`` collects these
        out-of-fold, then relabels at each candidate threshold.
        """
        import torch

        Xs = ((np.asarray(X, dtype="float32") - self.mu) / self.sd).astype("float32")
        for module in (self.trunk, self.gate_head, self.stable_head, self.change_head):
            module.eval()
        with torch.no_grad():
            gate_logit, stable_logits, change_logits = self._heads(
                torch.tensor(Xs, device=self.device)
            )
            p_change = torch.sigmoid(gate_logit.squeeze(1)).cpu().numpy()
            stable_idx = stable_logits.argmax(1).cpu().numpy()
            change_idx = change_logits.argmax(1).cpu().numpy()

        stable = np.array(self.stable_classes_, dtype=object)
        change = np.array(self.change_classes_, dtype=object)
        change_label = change[change_idx]
        stable_label = np.array(
            [f"{c} -> {c}" for c in stable[stable_idx]], dtype=object
        )
        return p_change, change_label, stable_label

    @staticmethod
    def combine(p_change, change_label, stable_label, threshold: float):
        """Pick each plot's label by the gate at a given threshold."""
        is_change = p_change > threshold
        out = np.where(is_change, change_label, stable_label)
        return out.astype(object)

    def predict(self, X):
        p_change, change_label, stable_label = self.predict_parts(X)
        return self.combine(p_change, change_label, stable_label, self.gate_threshold)


class TemporalTransitionNN:
    """Gated change-then-transition network over the annual embedding trajectory.

    ``HierarchicalTorchNN`` feeds a flat 2018/2024/diff vector to an MLP trunk.
    This feeds the full per-year sequence -- the ``(T years, 64)`` trajectory --
    to a GRU, then the same three gated heads (change gate, stable-class,
    change-transition). It is the model the multi-year extraction exists for: as
    flat columns the intermediate years are collinear with the endpoints and
    hurt a linear model, but as an *ordered sequence* they carry when and how a
    plot changed, which a recurrent trunk can use and a flat one cannot. If the
    annual data adds nothing over the two endpoints, this is where it shows --
    same task, same heads, only the trunk and its input differ.

    Needs the frame: it selects the per-year embedding columns and reshapes them
    to ``(n, T, 64)``. The endpoint difference bands are ignored -- they are a
    function of the sequence the GRU already sees, so including them would just
    re-feed the first and last step.
    """

    def __init__(self, columns: list[str], epochs: int = 200,
                 balanced: bool = True, hidden: int = 128,
                 gate_threshold: float = 0.5):
        self.year_columns = self._year_columns(columns)
        self.epochs = epochs
        self.balanced = balanced
        self.hidden = hidden
        self.gate_threshold = gate_threshold

    @staticmethod
    def _year_columns(columns: list[str]) -> dict:
        """Per-year embedding columns in channel order, e.g. {'2018': [A00_2018..]}."""
        per_year: dict[str, list[str]] = {}
        for c in columns:
            parts = c.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                per_year.setdefault(parts[1], []).append(c)
        years = sorted(per_year, key=int)
        if len(years) < 2:
            raise ValueError(
                "TemporalTransitionNN needs >=2 per-year embedding blocks; found "
                f"{years}. Extract the annual trajectory (extract_embeddings_gee "
                "--years 2018 .. 2024)."
            )
        counts = {y: len(per_year[y]) for y in years}
        if len(set(counts.values())) != 1:
            raise ValueError(f"Uneven channels across years: {counts}")
        return {y: sorted(per_year[y]) for y in years}

    def _sequence(self, frame: pd.DataFrame) -> np.ndarray:
        """Stack per-year embeddings into ``(n, T, C)`` (time-ordered)."""
        years = list(self.year_columns)
        return np.stack(
            [frame[self.year_columns[y]].to_numpy("float32") for y in years], axis=1
        )

    def _encode(self, Xt):
        import torch

        _, hidden = self.gru(Xt)  # hidden: (num_directions, n, hidden)
        return torch.cat([hidden[0], hidden[1]], dim=1)  # (n, 2*hidden)

    def _heads(self, Xt):
        rep = self._encode(Xt)
        return self.gate_head(rep), self.stable_head(rep), self.change_head(rep)

    def fit(self, frame, y):
        import torch
        import torch.nn as nn

        y = np.asarray(y)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        seq = self._sequence(frame)  # (n, T, C)

        # Standardise per channel across plots and time, so a channel's scale is
        # not learned as spurious temporal structure.
        self.mu = seq.mean((0, 1))
        self.sd = seq.std((0, 1)) + 1e-8
        seq = ((seq - self.mu) / self.sd).astype("float32")

        (change_mask, self.stable_classes_, self.change_classes_,
         y_stable, y_change) = gated_targets(y)

        torch.manual_seed(SEED)
        channels = seq.shape[2]
        self.gru = nn.GRU(channels, self.hidden, batch_first=True,
                          bidirectional=True).to(self.device)
        rep_dim = 2 * self.hidden
        self.gate_head = nn.Linear(rep_dim, 1).to(self.device)
        self.stable_head = nn.Linear(rep_dim, len(self.stable_classes_)).to(self.device)
        self.change_head = nn.Linear(rep_dim, len(self.change_classes_)).to(self.device)
        self._modules_ = (self.gru, self.gate_head, self.stable_head, self.change_head)

        change_weight = pos_weight = None
        if self.balanced:
            k = len(self.change_classes_)
            counts = np.bincount(y_change[y_change >= 0], minlength=k).astype("float32")
            change_weight = torch.tensor(
                len(y_change) / (k * np.maximum(counts, 1)), device=self.device
            )
            n_pos = float(change_mask.sum())
            n_neg = float((~change_mask).sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=self.device)

        gate_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        stable_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.05)
        change_loss_fn = nn.CrossEntropyLoss(
            weight=change_weight, ignore_index=-1, label_smoothing=0.05
        )

        params = [p for m in self._modules_ for p in m.parameters()]
        optimiser = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2e-3, total_steps=self.epochs
        )

        Xt = torch.tensor(seq, device=self.device)
        gate_t = torch.tensor(change_mask.astype("float32"), device=self.device)
        stable_t = torch.tensor(y_stable, device=self.device, dtype=torch.long)
        change_t = torch.tensor(y_change, device=self.device, dtype=torch.long)

        for module in self._modules_:
            module.train()
        for _ in range(self.epochs):
            optimiser.zero_grad()
            gate_logit, stable_logits, change_logits = self._heads(Xt)
            loss = (
                gate_loss_fn(gate_logit.squeeze(1), gate_t)
                + stable_loss_fn(stable_logits, stable_t)
                + change_loss_fn(change_logits, change_t)
            )
            loss.backward()
            optimiser.step()
            schedule.step()
        return self

    def predict_parts(self, frame):
        import torch

        seq = ((self._sequence(frame) - self.mu) / self.sd).astype("float32")
        for module in self._modules_:
            module.eval()
        with torch.no_grad():
            gate_logit, stable_logits, change_logits = self._heads(
                torch.tensor(seq, device=self.device)
            )
            p_change = torch.sigmoid(gate_logit.squeeze(1)).cpu().numpy()
            stable_idx = stable_logits.argmax(1).cpu().numpy()
            change_idx = change_logits.argmax(1).cpu().numpy()

        change = np.array(self.change_classes_, dtype=object)
        stable = np.array(self.stable_classes_, dtype=object)
        change_label = change[change_idx]
        stable_label = np.array(
            [f"{c} -> {c}" for c in stable[stable_idx]], dtype=object
        )
        return p_change, change_label, stable_label

    def predict(self, frame):
        p_change, change_label, stable_label = self.predict_parts(frame)
        return HierarchicalTorchNN.combine(
            p_change, change_label, stable_label, self.gate_threshold
        )


def to_merged_label(label: str) -> str:
    """Collapse a coarse3 transition to the merged2 (Vegetation/Artificial) legend."""
    if label == RARE_LABEL:
        return RARE_LABEL
    before, after = label.split(" -> ")
    veg = lambda c: "Vegetation" if c in ("Nature", "Cropland") else c  # noqa: E731
    return f"{veg(before)} -> {veg(after)}"


class HierarchicalSoftmaxNN:
    """Coarse-to-fine transition network with merged2 as the middle level.

    Three strictly nested levels, each a grouping of the one below::

        gate (change / no-change)
          <-  merged2 (Veg/Art transitions)     <- the intermediate the user named
                <-  coarse3 (Nature/Cropland/Art transitions)   <- the fine head

    A single fine head emits the coarse3 logits; the merged2 and gate
    probabilities are *exact group-sums* of the fine softmax (via fixed 0/1
    aggregation matrices), so the three predictions are always mutually
    consistent and there is only one head to train. The loss supervises all
    three levels, which injects the clean, well-supported merged2 signal --
    where the Cropland/Nature interpreter noise (see analyse_label_noise.py)
    has collapsed into a stable Veg->Veg -- straight into the representation,
    while the fine head still attempts the noisy split. Deploy the merged2 read
    (robust); report the coarse3 read (informative).

    The point of the class is comparison: ``arch`` swaps the trunk (flat MLP,
    residual MLP, wide MLP, or a GRU over the annual trajectory) and ``loss``
    swaps the objective (ce / weighted_ce / focal / cb_focal), so the variants
    are scored on one footing by experiment_hier_variants.py. Always takes the
    frame -- the GRU trunk needs the per-year sequence, the flat trunks take
    ``frame[columns]``.
    """

    #: Default epochs kept deliberately low. The fine coarse3 target carries the
    #: Cropland/Nature interpreter noise, and training past ~30-50 epochs overfits
    #: it -- both the merged (deploy) and fine change-F1 peak early and then
    #: decay (experiment_hier_variants.py epochs sweep). More is not better here.
    def __init__(self, columns: list[str], arch: str = "mlp", loss: str = "ce",
                 epochs: int = 50, level_weights=(1.0, 1.0, 1.0),
                 hidden: int = 128, gamma: float = 2.0, head: str = "flat",
                 noise_rate: float = 0.0, ssl: str | None = None,
                 ssl_weight: float = 1.0, ssl_threshold: float = 0.95,
                 seed: int = SEED, batch_size: int | None = None,
                 early_stop: bool = False, val_fraction: float = 0.2,
                 patience: int = 8, mixup_alpha: float = 0.0,
                 sampler: str = "shuffle", gh_m: int = 8,
                 n_experts: int = 4, moe_k: int = 0, moe_aux: float = 0.0,
                 expert_dim: int = 256, noise_std: float = 0.0,
                 noise_schedule: str = "off", noise_period: int = 1,
                 noise_sites=("input",), noise_gradscale: bool = False,
                 aef_columns=None, tess_columns=None,
                 mask_column: str = "tess_present",
                 modality_dropout: float = 0.5, tower_dim: int = 256,
                 aef_mask_column: str | None = None, fusion: str = "additive",
                 tess_gate: str = "mask", dropout_tess: float | None = None,
                 tess_width: float = 1.0, align_weight: float = 0.0,
                 align_temperature: float = 0.1,
                 distill_weight: float = 0.0, distill_temperature: float = 1.0,
                 endpoint_weight: float = 0.0):
        self.columns = list(columns)
        self.arch = arch
        self.loss = loss
        self.epochs = epochs
        self.level_weights = level_weights
        self.hidden = hidden
        self.gamma = gamma
        # --- opt-in training controls (all default to the original full-batch,
        # fixed-seed, fixed-epoch behaviour; see experiment_hier_improve.py) ---
        # seed: per-model torch seed, so an ensemble can average across seeds.
        self.seed = seed
        # batch_size: None keeps full-batch GD (one step/epoch); an int switches
        # to shuffled minibatches, giving many more optimiser steps per epoch and
        # SGD gradient noise as a regulariser against the endpoint label noise.
        self.batch_size = batch_size
        # early_stop: hold out val_fraction of train, monitor merged2 change-F1
        # each epoch, restore the best-epoch weights (patience epochs w/o gain).
        self.early_stop = early_stop
        self.val_fraction = val_fraction
        self.patience = patience
        # mixup_alpha: >0 mixes input/label pairs (Beta(a,a)); noise-robust reg.
        self.mixup_alpha = mixup_alpha
        # sampler: 'shuffle' (random minibatches / full-batch) or 'gh' -- the
        # Global-Hierarchical sampler of Zhang et al. (2022, ISPRS Siam-GL). Each
        # gradient step draws gh_m rows *per fine transition class*, so every rare
        # change class is present in every step (rare classes are cycled with
        # reshuffling to fill the quota). This attacks the sample-imbalance the
        # paper identifies -- change classes drowned out by the stable majority --
        # from the sampling side, orthogonally to the loss-side weighted_ce/focal.
        # gh_m sets the per-class quota; the number of batches per epoch is
        # ceil(max_class_count / gh_m). Pair with loss='ce' for the pure
        # sampling-only test, or loss='weighted_ce' to also inject the paper's
        # per-class w_j equalisation into the loss.
        self.sampler = sampler
        self.gh_m = gh_m
        # head: 'flat' one 9-way head, or 'bilinear' factorised endpoint head.
        self.head = head
        # noise_rate: symmetric Cropland<->Nature endpoint flip probability used
        # to forward-correct the fine loss (0 disables). The merged/gate levels
        # need no correction -- the confusion collapses into a stable Veg->Veg.
        self.noise_rate = noise_rate
        # ssl: unlabelled-loss method ('fixmatch' pseudo-label + consistency),
        # applied to the unlabelled frame passed to fit(); None = supervised.
        self.ssl = ssl
        self.ssl_weight = ssl_weight
        self.ssl_threshold = ssl_threshold
        # --- mixture-of-experts trunk (arch='moe') ---------------------------
        # n_experts parallel MLP experts over the input, combined by a learned
        # softmax gate. moe_k>0 makes it a sparse top-k gate (only the k highest-
        # weighted experts contribute per plot); moe_k=0 is the dense soft gate.
        # moe_aux>0 adds Shazeer et al.'s importance-balancing penalty (squared
        # coefficient of variation of per-expert gate mass) to stop the gate
        # collapsing onto one expert. expert_dim is each expert's output width and
        # the trunk representation width.
        self.n_experts = n_experts
        self.moe_k = moe_k
        self.moe_aux = moe_aux
        self.expert_dim = expert_dim
        # --- interleaved noise injection (Wiemann et al. 2026, arXiv:2607.14466)
        # Gaussian noise added to the (standardised) trunk input and/or the
        # representation during *training only*. noise_std is the base std;
        # noise_schedule sets when it is on across epochs:
        #   'off'        -- disabled.
        #   'constant'   -- every epoch (plain noise injection).
        #   'interleaved'-- alternate clean and noisy epoch-blocks of noise_period
        #                   epochs each (the paper's headline schedule: repeatedly
        #                   switching lets the optimiser explore in noisy epochs
        #                   without forgetting the clean-epoch features).
        #   'anneal'     -- monotone noisy->clean ramp (the curriculum the paper
        #                   argues interleaving beats), std scaled by 1 - e/(E-1).
        #   'warmup'     -- monotone clean->noisy ramp, std scaled by e/(E-1).
        # noise_sites picks the injection points ('input', 'rep', or both).
        # noise_gradscale co-scales the noise with the running gradient norm --
        # the paper's "gradient-norm scaling", which keeps the random-walk and
        # drift components balanced as the gradient shrinks.
        self.noise_std = noise_std
        self.noise_schedule = noise_schedule
        self.noise_period = noise_period
        self.noise_sites = tuple(noise_sites)
        self.noise_gradscale = noise_gradscale
        # --- two-tower late fusion (arch='two_tower') ------------------------
        # An always-on AlphaEarth tower over aef_columns (guarantees a prediction
        # for every plot) plus a Tessera tower over tess_columns that is gated by
        # the per-row mask_column (1 = Tessera present, 0 = absent). The fused
        # representation is rep_aef + gate * rep_tess, so the Tessera term is
        # zeroed exactly where Tessera is missing -- the right structure for a
        # second modality present for only ~36% of plots. modality_dropout
        # randomly zeroes the gate on *present* rows during training so the trunk
        # never becomes dependent on a feature absent for most change points.
        # tower_dim is each tower's output width and the fused rep width. See
        # experiment_hier_tessera.py (mode 'twotower') and the coverage caveat:
        # the mask correlates with geography under blocked CV.
        self.aef_columns = list(aef_columns) if aef_columns is not None else None
        self.tess_columns = list(tess_columns) if tess_columns is not None else None
        self.mask_column = mask_column
        self.modality_dropout = modality_dropout
        self.tower_dim = tower_dim
        # aef_mask_column: when None the AlphaEarth tower is *always on* (the
        # original Plan B: AlphaEarth guaranteed, Tessera the sparse bonus). Set
        # it to a per-row 0/1 column to make the AlphaEarth tower mask-gated too,
        # so the model also handles rows with NO AlphaEarth (and, via modality
        # dropout on both towers, learns to predict from whichever modality is
        # present). fusion: 'additive' (rep = g_a*rep_aef + g_t*rep_tess -- the
        # original when aef is always on) or 'gated_mean' (divide by the number of
        # present-and-kept modalities), which keeps the fused rep's scale constant
        # whether one or both towers fire -- the right fusion when *either* side
        # can be missing, since a single-modality row is then not down-scaled
        # relative to a both-present one.
        self.aef_mask_column = aef_mask_column
        self.fusion = fusion
        # tess_gate: 'mask' trusts present Tessera as much as AlphaEarth;
        # 'learned' scales the mask by a per-plot reliability read off both
        # representations, so a Tessera vector the AlphaEarth context disagrees
        # with is discounted rather than averaged in at full weight. fusion='film'
        # lets AlphaEarth emit (gamma, beta) that modulate the Tessera rep --
        # context conditioning detail, not just weighting it. dropout_tess sets a
        # separate dropout rate inside the Tessera tower (None = same as
        # AlphaEarth's), the asymmetric-regularisation knob for the noisier
        # modality. See _TwoTowerTrunk.
        self.tess_gate = tess_gate
        self.dropout_tess = dropout_tess
        # tess_width scales the Tessera tower's hidden widths (1.0 = the same
        # 1024/512 as AlphaEarth). The second, independent way to give the
        # noisier modality less room to memorise: capacity rather than dropout.
        self.tess_width = tess_width
        # align_weight > 0 adds a CLIP-style InfoNCE between the two towers'
        # representations on rows where both modalities are real, pulling the
        # AlphaEarth tower toward the Tessera manifold. The hope is that the
        # AlphaEarth tower then carries some of Tessera's structure on the ~64%
        # of plots where no Tessera vector exists to gate in.
        self.align_weight = align_weight
        self.align_temperature = align_temperature
        # --- cross-modal distillation ---------------------------------------
        # distill_weight > 0 adds a cross-entropy against soft merged2 targets
        # supplied to fit() as ``soft_merged`` (a frame whose columns are merged
        # class names, NaN on rows with no teacher). The motivating use is
        # transferring what a Tessera-fused teacher knows into an AlphaEarth-only
        # student, so the detail modality's contribution survives on the ~64% of
        # plots that have no Tessera at all. distill_temperature > 1 softens the
        # teacher (more dark knowledge, less of its arg-max).
        self.distill_weight = distill_weight
        self.distill_temperature = distill_temperature
        # endpoint_weight: auxiliary loss on the merged2 STATE marginals ("was
        # this Artificial in 2018 / in 2024") in addition to the three nested
        # levels. 0 disables, keeping the original three-level objective exactly.
        self.endpoint_weight = endpoint_weight

    # -- trunks --------------------------------------------------------------
    def _flat_trunk(self, d: int):
        import torch.nn as nn

        if self.arch == "mlp":
            trunk = nn.Sequential(
                nn.Linear(d, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.GELU(),
            )
            return trunk, 128
        if self.arch == "wide":
            trunk = nn.Sequential(
                nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.GELU(),
            )
            return trunk, 256
        if self.arch == "wide_se":
            # Wide trunk with a Squeeze-and-Excitation gate on the input: the gate
            # re-weights the ~193 channels per plot before the dense layers mix
            # them, a cheap test of whether feature selection helps at this width.
            trunk = nn.Sequential(
                _SEInput(d, reduction=16),
                nn.Linear(d, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.GELU(),
            )
            return trunk, 256
        if self.arch == "deep_res":
            return _ResidualMLP(d, width=256, blocks=4, dropout=0.3), 256
        if self.arch == "layernorm":
            # Same shape as mlp but LayerNorm, which (unlike BatchNorm) does not
            # depend on batch statistics -- the natural norm for the full-batch,
            # ~30-step regime.
            trunk = nn.Sequential(
                nn.Linear(d, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.GELU(),
            )
            return trunk, 128
        if self.arch == "snn":
            # Self-normalising net: SELU + AlphaDropout + LeCun-normal init keep
            # activations at unit variance without an explicit norm layer.
            def selu_linear(i, o):
                lin = nn.Linear(i, o)
                nn.init.normal_(lin.weight, 0.0, (1.0 / i) ** 0.5)
                nn.init.zeros_(lin.bias)
                return lin

            trunk = nn.Sequential(
                selu_linear(d, 512), nn.SELU(), nn.AlphaDropout(0.1),
                selu_linear(512, 256), nn.SELU(), nn.AlphaDropout(0.1),
                selu_linear(256, 128), nn.SELU(),
            )
            return trunk, 128
        if self.arch == "glu":
            # Gated linear units: each block multiplies a linear projection by a
            # learned sigmoid gate, so the trunk can suppress channels per plot.
            trunk = nn.Sequential(
                nn.Linear(d, 512), nn.GLU(dim=1), nn.BatchNorm1d(256), nn.Dropout(0.3),
                nn.Linear(256, 512), nn.GLU(dim=1), nn.BatchNorm1d(256), nn.Dropout(0.3),
                nn.Linear(256, 256), nn.GLU(dim=1),
            )
            return trunk, 128
        if self.arch == "ft":
            # Feature-tokeniser Transformer: every embedding channel becomes a
            # token, a [CLS] token attends over them, its output is the rep. The
            # only trunk here with cross-feature *attention* rather than fixed
            # dense mixing.
            return _FTTransformer(d, d_token=16, heads=2, layers=1, dropout=0.1), 16
        if self.arch == "moe":
            # Mixture of experts: n_experts parallel MLP experts + a softmax gate
            # (dense, or sparse top-k when moe_k>0). The representation is the
            # gate-weighted sum of the expert outputs, so different experts can
            # specialise on different regions of the 192-D embedding space --
            # e.g. the change vs stable manifolds -- instead of one trunk having
            # to fit all of them. Load balancing (moe_aux) is added in fit().
            trunk = _MoETrunk(d, n_experts=self.n_experts, expert_dim=self.expert_dim,
                              hidden=max(256, self.hidden), dropout=0.3,
                              top_k=self.moe_k)
            return trunk, self.expert_dim
        if self.arch == "two_tower":
            # Input packed by _prepare as [aef | tess | mask_aef | mask_tess]; the
            # trunk splits it, runs the two mask-gated towers, and fuses them
            # (additive, or gated_mean). rep_dim is tower_dim.
            trunk = _TwoTowerTrunk(
                d_aef=len(self.aef_columns), d_tess=len(self.tess_columns),
                out_dim=self.tower_dim, modality_dropout=self.modality_dropout,
                dropout=0.4, aef_maskable=self.aef_mask_column is not None,
                fusion=self.fusion, tess_gate=self.tess_gate,
                dropout_tess=self.dropout_tess, tess_width=self.tess_width,
            )
            return trunk, self.tower_dim
        raise ValueError(f"Unknown flat arch: {self.arch}")

    def _encode(self, Xt):
        import torch

        if self.arch == "gru":
            _, hidden = self.gru(Xt)
            return torch.cat([hidden[0], hidden[1]], dim=1)
        return self.trunk(Xt)

    # -- interleaved noise injection ----------------------------------------
    def _noise_factor(self, epoch: int) -> float:
        """Per-epoch noise multiplier for the configured schedule (0 = clean)."""
        if self.noise_std <= 0 or self.noise_schedule == "off":
            return 0.0
        span = max(1, self.epochs - 1)
        if self.noise_schedule == "constant":
            return 1.0
        if self.noise_schedule == "interleaved":
            # Alternate clean/noisy blocks of ``noise_period`` epochs; noisy on
            # the odd blocks. period=1 gives strict clean/noisy alternation.
            return 1.0 if (epoch // self.noise_period) % 2 == 1 else 0.0
        if self.noise_schedule == "anneal":       # noisy -> clean (curriculum)
            return max(0.0, 1.0 - epoch / span)
        if self.noise_schedule == "warmup":       # clean -> noisy
            return epoch / span
        raise ValueError(f"Unknown noise_schedule: {self.noise_schedule}")

    def _inject(self, x, site: str):
        """Add Gaussian noise to ``x`` at ``site`` when the epoch is a noisy one."""
        import torch

        if site not in self.noise_sites:
            return x
        std = self.noise_std * self._noise_factor_cur * self._noise_scale
        if std <= 0:
            return x
        return x + std * torch.randn_like(x)

    def _prepare(self, frame, fit: bool):
        """Standardised model input: flat matrix, or (n, T, C) sequence for gru."""
        if self.arch == "two_tower":
            # Pack [aef | tess | mask_aef | mask_tess]. Each feature block is
            # standardised *separately* over its own present rows (stats never see
            # an absent row), a missing value -> 0 = the column mean after
            # centring, and the two masks are carried raw (0/1) as the last two
            # columns for the trunk to gate on. When aef_mask_column is None the
            # AlphaEarth tower is always on, so its mask is a synthesised all-ones
            # column and the standardisation is over every row (the original
            # behaviour); set it to gate the AlphaEarth tower on coverage too.
            Xa = frame[self.aef_columns].to_numpy("float32")
            Xt = frame[self.tess_columns].to_numpy("float32")
            mt = frame[self.mask_column].to_numpy("float32").reshape(-1, 1)
            if self.aef_mask_column is not None:
                ma = frame[self.aef_mask_column].to_numpy("float32").reshape(-1, 1)
            else:
                ma = np.ones((len(frame), 1), "float32")

            def block_stats(X, mask):
                present = mask.ravel() > 0.5
                if present.any():
                    return np.nanmean(X[present], 0), np.nanstd(X[present], 0) + 1e-8
                return np.zeros(X.shape[1], "float32"), np.ones(X.shape[1], "float32")

            if fit:
                self.mu_a, self.sd_a = block_stats(Xa, ma)
                self.mu_t, self.sd_t = block_stats(Xt, mt)
            Xa = np.where(np.isfinite((Xa - self.mu_a) / self.sd_a),
                          (Xa - self.mu_a) / self.sd_a, 0.0)  # absent -> 0 = mean
            Xt = np.where(np.isfinite((Xt - self.mu_t) / self.sd_t),
                          (Xt - self.mu_t) / self.sd_t, 0.0)
            return np.concatenate([Xa, Xt, ma, mt], axis=1).astype("float32")
        if self.arch == "gru":
            seq = np.stack(
                [frame[self.year_columns[y]].to_numpy("float32")
                 for y in self.year_columns], axis=1
            )
            if fit:
                self.mu, self.sd = seq.mean((0, 1)), seq.std((0, 1)) + 1e-8
            return ((seq - self.mu) / self.sd).astype("float32")
        X = frame[self.columns].to_numpy("float32")
        # NaN-safe, matching the two_tower branch above: a modality that is
        # genuinely absent for some rows (Sentinel-2 reaches 99.3% of plots, not
        # 100%) must not poison the column statistics for everyone. Plain
        # mean/std propagate NaN into mu/sd and collapse the entire matrix to
        # NaN, which trains a degenerate single-class model. nanmean/nanstd are
        # identical to mean/std on NaN-free input, so every existing result is
        # untouched; the final where() maps an absent value to 0, i.e. the
        # column mean after centring -- the same convention _prepare already
        # uses for an absent tower block.
        if fit:
            self.mu = np.nanmean(X, 0)
            self.sd = np.nanstd(X, 0) + 1e-8
        Z = (X - self.mu) / self.sd
        return np.where(np.isfinite(Z), Z, 0.0).astype("float32")

    # -- aggregation ---------------------------------------------------------
    def _build_hierarchy(self, y: np.ndarray):
        """Fine/merged/gate class lists and the 0/1 aggregation matrices."""
        self.fine_classes_ = sorted(set(y))
        merged = [to_merged_label(c) for c in self.fine_classes_]
        self.merged_classes_ = sorted(set(merged))
        gate = ["change" if is_change_label(m) else "stable"
                for m in self.merged_classes_]
        self.gate_classes_ = sorted(set(gate))

        m_idx = {c: i for i, c in enumerate(self.merged_classes_)}
        g_idx = {c: i for i, c in enumerate(self.gate_classes_)}
        fine_to_merged = np.array([m_idx[to_merged_label(c)] for c in self.fine_classes_])
        merged_to_gate = np.array(
            [g_idx["change" if is_change_label(c) else "stable"]
             for c in self.merged_classes_]
        )

        # Endpoint (base-class) factorisation for the bilinear head and the
        # noise matrix: every fine transition is a (from-class, to-class) pair.
        endpoints = [c.split(" -> ") for c in self.fine_classes_]
        self.base_classes_ = sorted({e for pair in endpoints for e in pair})
        b_idx = {c: i for i, c in enumerate(self.base_classes_)}
        self.from_idx_ = np.array([b_idx[a] for a, _ in endpoints])
        self.to_idx_ = np.array([b_idx[b] for _, b in endpoints])

        # Merged2 endpoint marginals: which STATE each date is in, ignoring the
        # other date. A merged2 label is exactly a (state_2018, state_2024) pair,
        # so "Artificial in 2018" is a group-sum of the merged probabilities in
        # the same way merged2 is a group-sum of the fine ones -- no new head,
        # no new parameters, just two more 0/1 aggregation matrices. Supervising
        # these pools every Artificial-at-that-date plot into one decision
        # instead of splitting them across two thin transition classes, which is
        # the supervision the stable-Artificial-read-as-stable-Vegetation
        # confusion is starving for.
        merged_ends = [c.split(" -> ") if " -> " in c else (RARE_LABEL, RARE_LABEL)
                       for c in self.merged_classes_]
        self.state_classes_ = sorted({e for pair in merged_ends for e in pair})
        s_idx = {c: i for i, c in enumerate(self.state_classes_)}
        self.merged_from_idx_ = np.array([s_idx[a] for a, _ in merged_ends])
        self.merged_to_idx_ = np.array([s_idx[b] for _, b in merged_ends])
        return fine_to_merged, merged_to_gate

    def _noise_matrix(self) -> np.ndarray:
        """Forward-correction matrix ``T[k, k'] = P(observe k' | true k)``.

        Only the Cropland<->Nature endpoints flip (symmetric rate ``noise_rate``,
        estimated from the RECOVER reverifications); Artificial never flips. The
        transition-level matrix is the product of the two independent endpoint
        flips, so ``Nature -> Artificial`` leaks only to ``Cropland -> Artificial``
        and vice versa. Applied to the *fine* loss so the fine head learns the
        clean posterior while still explaining the noisy labels.
        """
        rho = self.noise_rate
        base = self.base_classes_
        flips = {"Cropland": "Nature", "Nature": "Cropland"}
        Mb = np.eye(len(base), dtype="float32")
        for i, c in enumerate(base):
            if c in flips and flips[c] in base:
                Mb[i, i] = 1.0 - rho
                Mb[i, base.index(flips[c])] = rho
        n = len(self.fine_classes_)
        T = np.zeros((n, n), "float32")
        for k in range(n):
            for kp in range(n):
                T[k, kp] = (Mb[self.from_idx_[k], self.from_idx_[kp]]
                            * Mb[self.to_idx_[k], self.to_idx_[kp]])
        # Renormalise rows: when the retained class set is not closed under the
        # Cropland<->Nature flips (a transition was rare-pooled), some flipped
        # mass lands on a class the model does not carry. Conditioning on the
        # observable classes keeps each row a valid P(observe . | true k).
        return T / T.sum(1, keepdims=True)

    def _build_head(self, rep_dim: int, n_fine: int):
        """Fine-logit head: a flat linear, or a factorised bilinear endpoint head."""
        import torch
        import torch.nn as nn

        if self.head == "flat":
            self.fine_head = nn.Linear(rep_dim, n_fine).to(self.device)
            return (self.fine_head,)
        if self.head == "bilinear":
            n_base = len(self.base_classes_)
            self.from_head = nn.Linear(rep_dim, n_base).to(self.device)
            self.to_head = nn.Linear(rep_dim, n_base).to(self.device)
            # Per-transition residual bias, so the factorisation is not forced to
            # explain every interaction additively.
            self.fine_bias = nn.Parameter(torch.zeros(n_fine, device=self.device))
            return (self.from_head, self.to_head)
        raise ValueError(f"Unknown head: {self.head}")

    def _fine_logits(self, rep):
        """Coarse3 logits from the representation, per the configured head."""
        if self.head == "flat":
            return self.fine_head(rep)
        # Bilinear: logit(A -> B) = from[A] + to[B] + bias(A -> B). Shares every
        # transition that shares a from- or to-class, so rare transitions borrow
        # strength from the common ones on the same endpoint.
        from_logits = self.from_head(rep)[:, self._from_idx]
        to_logits = self.to_head(rep)[:, self._to_idx]
        return from_logits + to_logits + self.fine_bias

    def _levels(self, p_fine, fine_t, merged_t, gate_t,
                w_fine, w_merged, w_gate, wf, wm, wg):
        """Weighted sum of the fine/merged/gate losses for one batch of probs."""
        p_merged = p_fine @ self._M
        p_gate = p_merged @ self._G
        # Forward correction: the fine head predicts the *clean* posterior and is
        # pushed through T to explain the noisy labels (identity if noise_rate=0).
        p_fine_obs = p_fine @ self._T if self._T is not None else p_fine
        loss = (wf * level_loss(p_fine_obs, fine_t, self.loss, w_fine, self.gamma)
                + wm * level_loss(p_merged, merged_t, self.loss, w_merged, self.gamma)
                + wg * level_loss(p_gate, gate_t, self.loss, w_gate, self.gamma))
        if self.endpoint_weight > 0:
            loss = loss + self.endpoint_weight * self._endpoint_loss(p_merged, merged_t)
        return loss

    def _endpoint_loss(self, p_merged, merged_t):
        """State-marginal loss: what each date *is*, not what the pair does.

        Targets are read off ``merged_t`` through the same index vectors that
        build the aggregation matrices, so a caller that permutes the merged
        target (mixup) permutes these for free and no signature has to change.
        """
        p_from = p_merged @ self._E_from
        p_to = p_merged @ self._E_to
        from_t = self._m_from[merged_t]
        to_t = self._m_to[merged_t]
        return 0.5 * (level_loss(p_from, from_t, self.loss, self._w_from, self.gamma)
                      + level_loss(p_to, to_t, self.loss, self._w_to, self.gamma))

    def fit(self, frame, y, unlabelled_frame=None, soft_merged=None):
        import torch
        import torch.nn as nn

        y = np.asarray(y)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.arch == "gru":
            self.year_columns = TemporalTransitionNN._year_columns(self.columns)
        Xs = self._prepare(frame, fit=True)

        fine_to_merged, merged_to_gate = self._build_hierarchy(y)
        f_code = {c: i for i, c in enumerate(self.fine_classes_)}
        fine_t = np.array([f_code[c] for c in y])
        merged_t = fine_to_merged[fine_t]
        gate_t = merged_to_gate[merged_t]

        # 0/1 aggregation matrices: p_merged = p_fine @ M, p_gate = p_merged @ G.
        n_fine, n_merged, n_gate = (
            len(self.fine_classes_), len(self.merged_classes_), len(self.gate_classes_)
        )
        M = np.zeros((n_fine, n_merged), "float32"); M[np.arange(n_fine), fine_to_merged] = 1
        G = np.zeros((n_merged, n_gate), "float32"); G[np.arange(n_merged), merged_to_gate] = 1
        self._M = torch.tensor(M, device=self.device)
        self._G = torch.tensor(G, device=self.device)

        if self.endpoint_weight > 0:
            n_state = len(self.state_classes_)
            E_from = np.zeros((n_merged, n_state), "float32")
            E_to = np.zeros((n_merged, n_state), "float32")
            E_from[np.arange(n_merged), self.merged_from_idx_] = 1
            E_to[np.arange(n_merged), self.merged_to_idx_] = 1
            self._E_from = torch.tensor(E_from, device=self.device)
            self._E_to = torch.tensor(E_to, device=self.device)
            self._m_from = torch.tensor(self.merged_from_idx_, device=self.device,
                                        dtype=torch.long)
            self._m_to = torch.tensor(self.merged_to_idx_, device=self.device,
                                      dtype=torch.long)
            state_mode = {"ce": "none", "weighted_ce": "inverse",
                          "focal": "none", "cb_focal": "effective"}[self.loss]
            self._w_from = torch.tensor(
                class_weights(self.merged_from_idx_[merged_t], n_state, state_mode),
                device=self.device)
            self._w_to = torch.tensor(
                class_weights(self.merged_to_idx_[merged_t], n_state, state_mode),
                device=self.device)

        torch.manual_seed(self.seed)
        if self.arch == "gru":
            self.gru = nn.GRU(Xs.shape[2], self.hidden, batch_first=True,
                              bidirectional=True).to(self.device)
            rep_dim = 2 * self.hidden
        else:
            self.trunk, rep_dim = self._flat_trunk(Xs.shape[1])
            self.trunk.to(self.device)
        head_modules = self._build_head(rep_dim, n_fine)
        self._from_idx = torch.tensor(self.from_idx_, device=self.device)
        self._to_idx = torch.tensor(self.to_idx_, device=self.device)
        trunk = (self.gru,) if self.arch == "gru" else (self.trunk,)
        self._modules_ = trunk + head_modules

        # Forward-correction matrix on the fine level (identity if noise_rate=0).
        self._T = (torch.tensor(self._noise_matrix(), device=self.device)
                   if self.noise_rate > 0 else None)

        weight_mode = {"ce": "none", "weighted_ce": "inverse",
                       "focal": "none", "cb_focal": "effective"}[self.loss]
        w_fine = torch.tensor(class_weights(fine_t, n_fine, weight_mode),
                              device=self.device)
        w_merged = torch.tensor(class_weights(merged_t, n_merged, weight_mode),
                                device=self.device)
        w_gate = torch.tensor(class_weights(gate_t, n_gate, weight_mode),
                              device=self.device)

        Xt = torch.tensor(Xs, device=self.device)
        fine_tt = torch.tensor(fine_t, device=self.device, dtype=torch.long)
        merged_tt = torch.tensor(merged_t, device=self.device, dtype=torch.long)
        gate_tt = torch.tensor(gate_t, device=self.device, dtype=torch.long)
        wg, wm, wf = self.level_weights

        # Teacher posteriors, aligned to this fit's merged class order by NAME (a
        # teacher trained on a different subset may not carry every class) and
        # renormalised after temperature. Rows the teacher could not score are
        # masked out rather than dropped, so the hard-label loss still sees them.
        self._soft = self._soft_mask = None
        if self.distill_weight > 0 and soft_merged is not None:
            q = pd.DataFrame(soft_merged).reindex(columns=self.merged_classes_)
            q = q.to_numpy("float64")
            keep = np.isfinite(q).all(1) & (np.nansum(q, 1) > 0)
            q = np.where(np.isfinite(q), q, 0.0)
            if self.distill_temperature != 1.0:
                q = np.power(np.clip(q, 1e-8, None), 1.0 / self.distill_temperature)
            row = q.sum(1, keepdims=True)
            q = np.where(row > 0, q / np.clip(row, 1e-12, None), 0.0)
            self._soft = torch.tensor(q.astype("float32"), device=self.device)
            self._soft_mask = torch.tensor(keep.astype("float32"), device=self.device)

        Xu = None
        if unlabelled_frame is not None and self.ssl:
            Xu = torch.tensor(self._prepare(unlabelled_frame, fit=False),
                              device=self.device)

        # Optional within-fold hold-out for early stopping. The outer CV is
        # already spatially blocked, so this random split only picks the stopping
        # epoch -- it never enters the reported (out-of-fold) score.
        rng = np.random.default_rng(self.seed)
        n = len(fine_t)
        if self.early_stop and n > 20:
            perm = rng.permutation(n)
            n_val = max(1, int(self.val_fraction * n))
            val_idx = np.sort(perm[:n_val])
            tr_idx = np.sort(perm[n_val:])
        else:
            tr_idx, val_idx = np.arange(n), None
        merged_is_change = np.array([is_change_label(c) for c in self.merged_classes_])

        # Global-Hierarchical sampler: fixed per-class index pools over the
        # training rows. Each epoch draws gh_m rows per fine class (reshuffled and
        # cycled for classes short of the quota), so every batch spans all classes
        # -- Algorithm 1 of Zhang et al. (2022). n_batches is the paper's number
        # of hierarchical subsets R = ceil(max class count / gh_m).
        gh_pools, gh_n_batches = None, 1
        if self.sampler == "gh":
            tr_fine = fine_t[tr_idx]
            gh_pools = [tr_idx[tr_fine == c] for c in range(n_fine)]
            gh_pools = [p for p in gh_pools if len(p) > 0]
            gh_n_batches = max(1, int(np.ceil(
                max(len(p) for p in gh_pools) / self.gh_m)))

        def gh_batches():
            need = gh_n_batches * self.gh_m
            filled = []
            for p in gh_pools:
                idx = p[rng.permutation(len(p))]
                reps = int(np.ceil(need / len(idx)))
                filled.append(np.tile(idx, reps)[:need])
            return [np.concatenate([f[b * self.gh_m:(b + 1) * self.gh_m]
                                    for f in filled])
                    for b in range(gh_n_batches)]

        params = [p for m in self._modules_ for p in m.parameters()]
        if self.head == "bilinear":
            params.append(self.fine_bias)
        optimiser = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        if self.sampler == "gh":
            steps_per_epoch = gh_n_batches
        else:
            steps_per_epoch = (1 if self.batch_size is None
                               else max(1, int(np.ceil(len(tr_idx) / self.batch_size))))
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2e-3, total_steps=self.epochs * steps_per_epoch
        )

        def batch_loss(idx):
            """Summed 3-level loss over training rows ``idx`` (mixup if enabled)."""
            it = torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)
            fb, mb, gb = fine_tt[it], merged_tt[it], gate_tt[it]
            Xb = Xt[it]
            if self.mixup_alpha > 0 and len(idx) > 1:
                lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
                order = torch.randperm(len(idx), device=self.device)
                Xb = lam * Xb + (1.0 - lam) * Xb[order]
                rep = self._inject(self._encode(self._inject(Xb, "input")), "rep")
                p_fine = torch.softmax(self._fine_logits(rep), dim=1)
                return (lam * self._levels(p_fine, fb, mb, gb, w_fine, w_merged,
                                           w_gate, wf, wm, wg)
                        + (1.0 - lam) * self._levels(p_fine, fb[order], mb[order],
                                                     gb[order], w_fine, w_merged,
                                                     w_gate, wf, wm, wg))
            rep = self._inject(self._encode(self._inject(Xb, "input")), "rep")
            p_fine = torch.softmax(self._fine_logits(rep), dim=1)
            loss = self._levels(p_fine, fb, mb, gb, w_fine, w_merged, w_gate,
                                wf, wm, wg)
            if self.align_weight > 0 and self.arch == "two_tower":
                loss = loss + self.align_weight * self._align_loss()
            if self._soft is not None:
                # Soft cross-entropy at the merged2 level, averaged over the rows
                # the teacher actually scored.
                p_merged = p_fine @ self._M
                ce = -(self._soft[it] * torch.log(p_merged.clamp_min(1e-8))).sum(1)
                keep = self._soft_mask[it]
                loss = loss + self.distill_weight * (
                    (ce * keep).sum() / keep.sum().clamp_min(1.0))
            return loss

        def snapshot():
            state = [{k: v.detach().clone() for k, v in m.state_dict().items()}
                     for m in self._modules_]
            return (state, self.fine_bias.detach().clone()
                    if self.head == "bilinear" else None)

        def restore(snap):
            state, bias = snap
            for module, st in zip(self._modules_, state):
                module.load_state_dict(st)
            if bias is not None:
                with torch.no_grad():
                    self.fine_bias.copy_(bias)

        # Interleaved-noise state: _noise_scale co-scales the injected std with the
        # running gradient norm when noise_gradscale is set (1.0 = off/unscaled).
        self._noise_scale = 1.0
        self._noise_factor_cur = 0.0
        grad_ema = [None]  # running gradient-norm EMA; ref = first observed value
        grad_ref = [None]

        best_f1, best_snap, since_best = -1.0, None, 0
        for _epoch in range(self.epochs):
            for module in self._modules_:
                module.train()
            self._noise_factor_cur = self._noise_factor(_epoch)
            if self.sampler == "gh":
                chunks = gh_batches()
            else:
                order = (tr_idx if self.batch_size is None
                         else tr_idx[rng.permutation(len(tr_idx))])
                chunks = ([order] if self.batch_size is None
                          else [order[i:i + self.batch_size]
                                for i in range(0, len(order), self.batch_size)])
            for chunk in chunks:
                optimiser.zero_grad()
                loss = batch_loss(chunk)
                if self.arch == "moe" and self.moe_aux > 0 and \
                        getattr(self.trunk, "last_importance", None) is not None:
                    imp = self.trunk.last_importance
                    loss = loss + self.moe_aux * (imp.std() / (imp.mean() + 1e-9)) ** 2
                if Xu is not None:
                    loss = loss + self.ssl_weight * self._ssl_loss(Xu)
                loss.backward()
                if self.noise_gradscale and self.noise_std > 0:
                    with torch.no_grad():
                        total = sum(float(p.grad.norm()) ** 2
                                    for p in params if p.grad is not None) ** 0.5
                    grad_ema[0] = (total if grad_ema[0] is None
                                   else 0.9 * grad_ema[0] + 0.1 * total)
                    if grad_ref[0] is None:
                        grad_ref[0] = grad_ema[0]
                    self._noise_scale = grad_ema[0] / (grad_ref[0] + 1e-12)
                optimiser.step()
                schedule.step()
            if val_idx is not None:
                for module in self._modules_:
                    module.eval()
                vt = torch.as_tensor(val_idx, device=self.device, dtype=torch.long)
                with torch.no_grad():
                    pv = (torch.softmax(self._fine_logits(self._encode(Xt[vt])), 1)
                          @ self._M).argmax(1).cpu().numpy()
                f1 = f1_score(merged_is_change[merged_t[val_idx]],
                              merged_is_change[pv], zero_division=0)
                if f1 > best_f1:
                    best_f1, best_snap, since_best = f1, snapshot(), 0
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break
        if best_snap is not None:
            restore(best_snap)
        return self

    def _ssl_loss(self, Xu):
        """FixMatch-style unlabelled loss on the robust merged2 level.

        A confident pseudo-label is read from a weakly-noised view (detached),
        and cross-entropy against it is taken on a strongly-noised view. The
        merged2 level -- not the noisy fine level -- provides the pseudo-label,
        so semi-supervision propagates the *clean* Vegetation/Artificial signal
        rather than amplifying the Cropland/Nature label noise.
        """
        import torch

        weak = Xu + 0.05 * torch.randn_like(Xu)
        strong = Xu + 0.20 * torch.randn_like(Xu)
        with torch.no_grad():
            p_weak = torch.softmax(self._fine_logits(self._encode(weak)), 1) @ self._M
            conf, pseudo = p_weak.max(1)
            keep = conf >= self.ssl_threshold
        if keep.sum() == 0:
            return Xu.new_tensor(0.0)
        p_strong = torch.softmax(self._fine_logits(self._encode(strong)), 1) @ self._M
        return level_loss(p_strong[keep], pseudo[keep], "ce")

    def _align_loss(self):
        """Symmetric InfoNCE between the two towers on both-present rows."""
        import torch
        import torch.nn.functional as F

        ra, rt = self.trunk.last_ra, self.trunk.last_rt
        both = self.trunk.last_both
        if both.sum() < 2:
            return ra.new_tensor(0.0)
        a = F.normalize(ra[both], dim=1)
        t = F.normalize(rt[both], dim=1)
        logits = a @ t.T / self.align_temperature
        target = torch.arange(len(a), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, target)
                      + F.cross_entropy(logits.T, target))

    def _probs(self, frame):
        import torch

        Xs = self._prepare(frame, fit=False)
        for module in self._modules_:
            module.eval()
        with torch.no_grad():
            # The fine head is the clean posterior (T is a training-time
            # correction, not applied at inference).
            p_fine = torch.softmax(self._fine_logits(self._encode(
                torch.tensor(Xs, device=self.device))), dim=1)
            p_merged = p_fine @ self._M
        return p_fine.cpu().numpy(), p_merged.cpu().numpy()

    def probs_aef_only(self, frame):
        """``_probs`` for the deployed gate-OFF read, without touching the detail tower.

        The deployment chosen for this model trains with both towers and reads
        with the detail gate forced to zero. Under that read the second tower is
        provably irrelevant, not merely small: ``_TwoTowerTrunk`` fuses as
        ``(g_a*r_a + g_t*r_t) / max(g_a + g_t, 1)``, and with ``g_t = 0`` and
        ``g_a = 1`` this is exactly ``r_a``. The detail tower's output is
        multiplied by zero, so computing it at all is waste -- and so is
        standardising, materialising and shipping its columns.

        This path therefore builds only the AlphaEarth block and runs only the
        AlphaEarth tower, and is **bit-identical** to calling ``_probs`` with the
        mask column zeroed (asserted in ``tests/test_s2off_fastpath.py``). The
        caller may pass a frame that does not carry the detail columns at all,
        which is what makes the Sentinel-2 fetch and feature computation
        skippable at inference.

        Restricted to the configuration it is proved for; anything else must go
        through ``_probs`` rather than silently take a different code path.
        """
        self._assert_aef_only_ok()
        return self.probs_aef_only_matrix(
            frame[self.aef_columns].to_numpy("float32"))

    def _assert_aef_only_ok(self) -> None:
        """The three structural conditions under which the fast path is exact."""
        if self.arch != "two_tower":
            raise ValueError("probs_aef_only requires arch='two_tower'")
        if self.fusion not in ("additive", "gated_mean"):
            raise ValueError(f"probs_aef_only not proved for fusion={self.fusion!r}")
        if self.tess_gate != "mask":
            raise ValueError("probs_aef_only requires tess_gate='mask' "
                             "(a learned gate could be non-zero)")

    def probs_aef_only_matrix(self, Xa):
        """``probs_aef_only`` from a raw AlphaEarth matrix -- no pandas anywhere.

        ``Xa`` is ``(n, len(aef_columns))`` **unstandardised** values in
        ``self.aef_columns`` order, as a numpy array or a torch tensor that is
        already on the device. A caller that maps a raster already holds exactly
        that matrix; routing it through a DataFrame so ``_prepare`` can pull the
        columns back out again cost more than the forward pass it fed (measured:
        611 ms of DataFrame construction and 373 ms of numpy standardisation per
        200k-pixel batch, against 98 ms for the tower itself).

        Standardisation moves to the device with the data. It is the same
        arithmetic in the same dtype -- ``(x - mu) / sd`` elementwise in float32,
        non-finite mapped to 0, which is the column mean after centring -- so the
        result is bit-identical to the pandas path (``tests/test_s2off_fastpath.py``)
        while the raw block crosses the PCIe bus once per batch instead of once
        per ensemble member.
        """
        import torch

        self._assert_aef_only_ok()
        if isinstance(Xa, torch.Tensor):
            x = Xa.to(self.device, torch.float32, non_blocking=True)
        else:
            x = torch.as_tensor(np.ascontiguousarray(Xa, dtype="float32"),
                                device=self.device)
        mu = torch.as_tensor(self.mu_a, device=self.device, dtype=torch.float32)
        sd = torch.as_tensor(self.sd_a, device=self.device, dtype=torch.float32)

        net = self.trunk
        for module in self._modules_:
            module.eval()
        with torch.no_grad():
            # nan_to_num after the division, not before: an absent value must
            # land on the column mean *after* centring (0), which is what the
            # frame path's np.where(isfinite) does. inf is mapped the same way.
            z = torch.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
            rep = net.aef_tower(z)
            # g_a is 1 (present) and, for `additive` with a non-maskable
            # AlphaEarth tower, hard-wired to 1. gated_mean then divides by
            # max(1 + 0, 1) = 1. Either way the fused rep is r_a untouched.
            p_fine = torch.softmax(self._fine_logits(rep), dim=1)
            p_merged = p_fine @ self._M
        return p_fine.cpu().numpy(), p_merged.cpu().numpy()

    def predict(self, frame):
        """Fine (coarse3) transition label -- the informative read."""
        p_fine, _ = self._probs(frame)
        return np.array(self.fine_classes_, dtype=object)[p_fine.argmax(1)]

    def merged_labels_from_probs(self, p_merged, change_threshold=None):
        """Merged2 labels from the merged probabilities at a change operating point.

        ``change_threshold=None`` is the plain arg-max (the implicit 0.5 gate).
        Otherwise a plot is called *change* when its total change mass
        ``P(Veg->Art) + P(Art->Veg) >= change_threshold`` and named by the arg-max
        within the chosen side -- the tunable gate from experiment_hier_change_recall.py
        (t~0.45 maximises change-F1, t~0.30 maximises stratification efficiency /
        change recall). Deploy at a threshold to trade change precision for the
        recall the design-based area estimate rewards.
        """
        classes = np.array(self.merged_classes_, dtype=object)
        if change_threshold is None:
            return classes[p_merged.argmax(1)]
        is_chg = np.array([is_change_label(c) for c in self.merged_classes_])
        chg_cols = np.where(is_chg)[0]
        stab_cols = np.where(~is_chg)[0]
        if len(chg_cols) == 0 or len(stab_cols) == 0:
            return classes[p_merged.argmax(1)]  # nothing to gate
        p_change = p_merged[:, chg_cols].sum(1)
        chg_pick = classes[chg_cols][p_merged[:, chg_cols].argmax(1)]
        stab_pick = classes[stab_cols][p_merged[:, stab_cols].argmax(1)]
        return np.where(p_change >= change_threshold, chg_pick, stab_pick)

    def predict_merged(self, frame, change_threshold=None):
        """Merged2 (Vegetation/Artificial) transition label -- the deploy read.

        Pass ``change_threshold`` to call change at a non-default gate; see
        ``merged_labels_from_probs``.
        """
        _, p_merged = self._probs(frame)
        return self.merged_labels_from_probs(p_merged, change_threshold)


class _ResidualMLP:
    """Pre-activation residual MLP trunk, as an nn.Module built lazily.

    Kept out of module import time so ``torch`` is only required when a torch
    model is actually constructed, matching the rest of this file.
    """

    def __new__(cls, d: int, width: int, blocks: int, dropout: float):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Sequential(nn.Linear(d, width), nn.GELU())
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.BatchNorm1d(width), nn.Linear(width, width), nn.GELU(),
                        nn.Dropout(dropout), nn.Linear(width, width),
                    ) for _ in range(blocks)
                ])

            def forward(self, x):
                h = self.stem(x)
                for block in self.blocks:
                    h = h + block(h)
                return h

        return _Net()


class _FTTransformer:
    """Feature-tokeniser Transformer trunk (Gorishniy et al. 2021), built lazily.

    Each scalar embedding channel ``x_i`` becomes a token ``x_i * w_i + b_i`` in
    ``d_token`` dims; a learned ``[CLS]`` token is prepended and the encoder lets
    it attend over all channels. The ``[CLS]`` output is the representation. This
    is the only trunk with genuine cross-feature attention rather than fixed dense
    mixing -- the test of whether attention buys anything on 192 AEF channels.
    """

    def __new__(cls, d_features: int, d_token: int, heads: int, layers: int,
                dropout: float):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(d_features, d_token) * 0.02)
                self.bias = nn.Parameter(torch.zeros(d_features, d_token))
                self.cls = nn.Parameter(torch.randn(1, 1, d_token) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_token, nhead=heads, dim_feedforward=d_token * 4,
                    dropout=dropout, batch_first=True, activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
                self.norm = nn.LayerNorm(d_token)

            def forward(self, x):
                tokens = x.unsqueeze(-1) * self.weight + self.bias  # (n, d, d_token)
                cls = self.cls.expand(x.shape[0], -1, -1)
                sequence = torch.cat([cls, tokens], dim=1)          # (n, d+1, d_token)
                encoded = self.encoder(sequence)
                return self.norm(encoded[:, 0])                     # (n, d_token)

        return _Net()


class _SEInput:
    """Squeeze-and-Excitation gate over the input features (Hu et al. 2018).

    The AEF feature vector already *is* the channel descriptor (no spatial dims to
    squeeze), so this is the "excite" step applied directly: a bottleneck MLP
    ``d -> d/r -> d`` produces a per-feature sigmoid gate that rescales the input,
    ``x * sigmoid(W2 GELU(W1 x))``. With ~193 partly redundant channels (the 64
    diff bands duplicate the endpoint signal) the gate can learn to down-weight
    the noisy/redundant ones per plot before the dense trunk mixes them.
    """

    def __new__(cls, d: int, reduction: int = 16, min_bottleneck: int = 8):
        import torch
        import torch.nn as nn

        bottleneck = max(min_bottleneck, d // reduction)

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(d, bottleneck)
                self.fc2 = nn.Linear(bottleneck, d)
                self.act = nn.GELU()

            def forward(self, x):
                gate = torch.sigmoid(self.fc2(self.act(self.fc1(x))))
                return x * gate

        return _Net()


class _MoETrunk:
    """Mixture-of-experts trunk (Shazeer et al. 2017 style), built lazily.

    ``n_experts`` parallel MLP experts each map the input to ``expert_dim``; a
    linear gate produces a softmax weight per expert and the representation is
    the weighted sum of the expert outputs. With ``top_k>0`` only the k highest-
    weighted experts contribute per plot (renormalised), a sparse conditional
    computation; ``top_k=0`` keeps the dense soft mixture. The forward pass
    stashes ``last_importance`` -- the mean gate mass per expert over the batch --
    so ``HierarchicalSoftmaxNN.fit`` can add the importance-balancing penalty
    that keeps the gate from collapsing onto a single expert.
    """

    def __new__(cls, d: int, n_experts: int, expert_dim: int, hidden: int,
                dropout: float, top_k: int):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate = nn.Linear(d, n_experts)
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
                        nn.Dropout(dropout), nn.Linear(hidden, expert_dim), nn.GELU(),
                    ) for _ in range(n_experts)
                ])
                self.n_experts = n_experts
                self.top_k = top_k
                self.last_importance = None

            def forward(self, x):
                w = torch.softmax(self.gate(x), dim=1)              # (n, E)
                if 0 < self.top_k < self.n_experts:
                    _, top_i = w.topk(self.top_k, dim=1)
                    mask = torch.zeros_like(w).scatter(1, top_i, 1.0)
                    w = w * mask
                    w = w / (w.sum(1, keepdim=True) + 1e-9)
                outs = torch.stack([e(x) for e in self.experts], dim=1)  # (n, E, dim)
                self.last_importance = w.mean(0)                    # (E,)
                return (w.unsqueeze(-1) * outs).sum(1)              # (n, expert_dim)

        return _Net()


class _TwoTowerTrunk:
    """Symmetric two-tower late-fusion trunk for two sparse modalities, built lazily.

    The input is ``[aef | tess | mask_aef | mask_tess]`` (packed by ``_prepare``):
    ``d_aef`` AlphaEarth columns, ``d_tess`` Tessera columns, then two 0/1 mask
    columns (1 = that modality present). Two wide MLP towers of matching shape map
    each block to ``out_dim`` and the fused representation gates each tower by its
    mask::

        additive:    rep = g_a * rep_aef + g_t * rep_tess
        gated_mean:  rep = (g_a * rep_aef + g_t * rep_tess) / max(g_a + g_t, 1)

    A tower contributes exactly zero where its modality is absent (its zero-imputed
    columns never reach the fused rep), so the model produces a prediction from
    whichever modalities *are* present -- AlphaEarth only, Tessera only, or both.
    ``gated_mean`` keeps the fused rep at a constant scale regardless of how many
    towers fire, so a single-modality row is not down-scaled relative to a
    both-present one -- the right choice when *either* side can be missing.

    ``aef_maskable`` controls whether the AlphaEarth tower may be gated/dropped.
    When False (the original Plan B) ``g_a`` is forced to 1 and only Tessera is
    ever gated -- AlphaEarth is the guaranteed base and ``additive`` reduces to the
    tested ``rep_aef + g_t * rep_tess``. When True both towers are gated by their
    masks *and* by modality dropout.

    ``modality_dropout`` randomly zeroes a modality's gate on rows where it is
    present during training, so the trunk learns to predict from the other tower
    alone and never becomes dependent on a feature missing for most plots. The
    guard is that a row's *only* present-and-kept modality is never dropped: drop
    is only ever applied where both modalities are present, and at most one of the
    two is dropped per row -- so every row always retains at least one tower and
    the ``gated_mean`` denominator is never zero.

    BatchNorm (not LayerNorm) inside the towers keeps the all-zero absent rows
    finite -- a per-row LayerNorm would divide a constant vector by ~0. Absent rows
    still pass through their tower but are gated out before fusion, so their only
    effect is a small shift in the tower's batch statistics, which the affine
    parameters absorb (and which modality dropout makes the norm robust to).

    ``tess_gate`` chooses what multiplies the Tessera tower. ``'mask'`` is the
    binary availability flag -- present Tessera is trusted exactly as much as
    AlphaEarth. ``'learned'`` multiplies the mask by a per-plot reliability
    ``sigma(MLP([rep_aef, rep_tess]))``: the fused representation can then *discount*
    a Tessera vector that the AlphaEarth context disagrees with, instead of
    averaging it in at full weight. That is the right structure when one modality
    carries more detail but also more error -- availability and trustworthiness are
    different questions. The gate bias starts at +2 (sigma ~ 0.88) so training
    begins near the tested mask-gated behaviour and has to earn any down-weighting.

    ``fusion='film'`` goes further and lets the context *condition* the detail:
    the AlphaEarth representation emits per-channel ``(gamma, beta)`` that modulate
    the Tessera representation (Perez et al. 2018) before fusion, so AlphaEarth
    changes how Tessera is read rather than just how much of it is added. The
    modulation is skipped on rows whose AlphaEarth tower is gated off, which keeps
    the Tessera-only fallback intact. It is applied residually from a zero-initialised
    layer, so an untrained FiLM is exactly the identity and this starts as
    ``gated_mean`` -- any modulation the network adopts has to pay for itself.
    """

    def __new__(cls, d_aef: int, d_tess: int, out_dim: int,
                modality_dropout: float, dropout: float,
                aef_maskable: bool = False, fusion: str = "additive",
                tess_gate: str = "mask", dropout_tess: float | None = None,
                tess_width: float = 1.0):
        import torch
        import torch.nn as nn

        if fusion not in ("additive", "gated_mean", "film"):
            raise ValueError(f"Unknown two-tower fusion: {fusion}")
        if tess_gate not in ("mask", "learned"):
            raise ValueError(f"Unknown two-tower tess_gate: {tess_gate}")
        d_tess_drop = dropout if dropout_tess is None else dropout_tess

        def tower(d_in, p, width=1.0):
            h1, h2 = max(16, int(1024 * width)), max(16, int(512 * width))
            return nn.Sequential(
                nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(p),
                nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(p),
                nn.Linear(h2, out_dim), nn.GELU(),
            )

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_aef, self.d_tess = d_aef, d_tess
                self.aef_tower = tower(d_aef, dropout)
                self.tess_tower = tower(d_tess, d_tess_drop, tess_width)
                self.modality_dropout = modality_dropout
                self.aef_maskable = aef_maskable
                self.fusion = fusion
                self.tess_gate = tess_gate
                if tess_gate == "learned":
                    self.reliability = nn.Sequential(
                        nn.Linear(2 * out_dim, 64), nn.GELU(), nn.Linear(64, 1))
                    nn.init.zeros_(self.reliability[-1].weight)
                    nn.init.constant_(self.reliability[-1].bias, 2.0)
                if fusion == "film":
                    self.film = nn.Linear(out_dim, 2 * out_dim)
                    nn.init.zeros_(self.film.weight)
                    nn.init.zeros_(self.film.bias)

            def forward(self, x):
                xa = x[:, :self.d_aef]
                xt = x[:, self.d_aef:self.d_aef + self.d_tess]
                ma = x[:, self.d_aef + self.d_tess:self.d_aef + self.d_tess + 1]
                mt = x[:, self.d_aef + self.d_tess + 1:self.d_aef + self.d_tess + 2]
                ra = self.aef_tower(xa)
                rt = self.tess_tower(xt)
                # AlphaEarth gate: the mask, or a constant 1 when it is the
                # guaranteed base (never gated, never dropped).
                ga = ma if self.aef_maskable else torch.ones_like(ma)
                gt = mt
                if self.training and self.modality_dropout > 0:
                    # Only drop where a fallback exists (both present), and drop at
                    # most one side per row, so every row keeps >=1 tower.
                    both = (ma > 0.5) & (mt > 0.5)
                    drop_t = (torch.rand_like(mt) < self.modality_dropout) & both
                    gt = gt * (~drop_t).to(gt.dtype)
                    if self.aef_maskable:
                        drop_a = ((torch.rand_like(ma) < self.modality_dropout)
                                  & both & (~drop_t))
                        ga = ga * (~drop_a).to(ga.dtype)
                if self.fusion == "film":
                    # Context conditions detail, but only where context survives:
                    # a row down to the Tessera tower alone reads it unmodulated.
                    gamma, beta = self.film(ra).chunk(2, dim=1)
                    on = (ga > 0.5).to(rt.dtype)
                    rt = rt + on * (gamma * rt + beta)
                if self.tess_gate == "learned":
                    # Availability (mt, hard) times trustworthiness (learned, soft).
                    gt = gt * torch.sigmoid(self.reliability(torch.cat([ra, rt], 1)))
                # Kept for the optional cross-modal alignment loss, which needs
                # both towers' representations and the rows where both are real.
                self.last_ra, self.last_rt = ra, rt
                self.last_both = ((ma > 0.5) & (mt > 0.5)).squeeze(1)
                fused = ga * ra + gt * rt
                if self.fusion in ("gated_mean", "film"):
                    fused = fused / (ga + gt).clamp_min(1.0)
                return fused

        return _Net()


def foundation_models(balanced: bool) -> dict:
    """Tabular / series foundation models, each skipped when not installed.

    ``balanced`` must be honoured here too: giving the DNN a class-weighted loss
    while the classical models run unweighted would flatter it on exactly the
    metric (balanced accuracy) the comparison turns on.
    """
    models = {}

    try:
        from tabpfn import TabPFNClassifier

        models["tabpfn"] = lambda: TabPFNClassifier(
            device="cuda" if _cuda() else "cpu", ignore_pretraining_limits=True
        )
    except ImportError:
        pass

    try:
        from tabicl import TabICLClassifier

        models["tabicl"] = lambda: TabICLClassifier(
            device="cuda" if _cuda() else "cpu"
        )
    except ImportError:
        pass

    try:
        import torch  # noqa: F401

        models["dnn_torch"] = lambda: TorchDNN(balanced=balanced)
    except ImportError:
        pass

    return models


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def build_registry(
    columns: list[str],
    balanced: bool = False,
    postclass_bases: list[str] | None = None,
    postclass_shared: bool = True,
    hier_bases: list[str] | None = None,
) -> dict:
    """Every runnable model, as ``name -> (factory, needs_frame)``.

    Shared with ``map_efficiency.py`` so both scripts score the identical set
    of models under identical configuration.
    """
    registry = {}
    # TIMEE is excluded as a post-classification base: it reshapes the frame
    # itself, so it cannot be handed a per-date column subset.
    bases = {**classical_models(balanced), **foundation_models(balanced)}
    for name, factory in bases.items():
        registry[name] = (factory, False)

    # Post classification is a wrapper, not a model: pairing it with a base
    # that also runs directly is what makes the two framings comparable.
    for base in postclass_bases or []:
        if base == "none":
            continue
        if base not in bases:
            raise ValueError(
                f"postclass base '{base}' is not an available model. "
                f"Available: {sorted(bases)}"
            )
        registry[f"postclass_{base}"] = (
            lambda base=base: PostClassification(
                columns, bases[base], shared=postclass_shared
            ),
            True,
        )

    # Hierarchical: change / no-change gate, then a transition classifier on the
    # change branch. Same base as the direct model, so 'hier_<base>' isolates
    # the effect of restructuring the decision from the effect of the estimator.
    for base in hier_bases or []:
        if base == "none":
            continue
        if base not in bases:
            raise ValueError(
                f"hier base '{base}' is not an available model. "
                f"Available: {sorted(bases)}"
            )
        registry[f"hier_{base}"] = (
            lambda base=base: HierarchicalChange(bases[base]),
            False,
        )

    # The end-to-end network that mimics the same gated process with a shared
    # trunk. Registered directly (not via ``bases``) because it consumes the
    # transition-string target itself and cannot serve as a per-date post-class
    # or hierarchical base.
    try:
        import torch  # noqa: F401

        registry["hier_nn"] = (lambda: HierarchicalTorchNN(balanced=balanced), False)
        # The GRU-trunk variant over the annual trajectory. Registered only when
        # the frame actually carries >=2 per-year embedding blocks; on the
        # two-endpoint frame it would be a GRU over two steps, which is the flat
        # model with extra machinery.
        try:
            TemporalTransitionNN._year_columns(columns)
            registry["temporal_nn"] = (
                lambda: TemporalTransitionNN(columns, balanced=balanced), True
            )
        except ValueError:
            pass
    except ImportError:
        pass

    try:
        import timee  # noqa: F401

        registry["timee"] = (lambda: TimeeAdapter(columns), True)
    except ImportError:
        pass
    return registry


def evaluate(
    name: str,
    factory,
    features: pd.DataFrame,
    frame: pd.DataFrame,
    target: pd.Series,
    groups: pd.Series,
    n_splits: int,
    is_change_fn,
    needs_frame: bool = False,
    cv: str = "blocked",
) -> tuple[dict, np.ndarray]:
    splitter = make_splitter(cv, n_splits)
    oof = np.empty(len(target), dtype=object)
    started = time.time()
    for train_idx, test_idx in splitter.split(features, target, groups):
        model = factory()
        if needs_frame:
            # TIMEE reshapes the columns itself, so it wants the frame.
            fit_data, test_data = frame.iloc[train_idx], frame.iloc[test_idx]
        else:
            fit_data = features.iloc[train_idx].to_numpy()
            test_data = features.iloc[test_idx].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(fit_data, target.iloc[train_idx].to_numpy())
            oof[test_idx] = model.predict(test_data)
    row = {"model": name, **scores(target.to_numpy(), oof, is_change_fn),
           "seconds": round(time.time() - started, 1)}
    return row, oof


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--cv",
        choices=["blocked", "random"],
        default="blocked",
        help=(
            "blocked holds whole spatial blocks out (default); random splits "
            "plots independently, which the stratified-random design permits "
            "but which lets a test plot's neighbour sit in training"
        ),
    )
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--balanced", action="store_true",
                        help="Use class-balanced weights where the model supports it")
    parser.add_argument(
        "--postclass-base",
        nargs="*",
        default=["lda_pc"],
        help=(
            "Models to also run as post-classification "
            "(per-date labels, transition read off the pair). Registered as "
            "'postclass_<name>'; 'none' disables. Default 'lda_pc' is the "
            "F1-tuned post-classification LDA (see tune_lda.py)"
        ),
    )
    parser.add_argument(
        "--postclass-per-date",
        action="store_true",
        help=(
            "Fit an independent classifier per date instead of one shared "
            "classifier trained on both epochs stacked"
        ),
    )
    parser.add_argument(
        "--hier-base",
        nargs="*",
        default=["lda"],
        help=(
            "Models to also run as a two-stage hierarchical classifier "
            "(change/no-change gate, then a transition model on the change "
            "branch). Registered as 'hier_<name>'; 'none' disables. The "
            "end-to-end 'hier_nn' network is always registered when torch is "
            "present"
        ),
    )
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these model names")
    parser.add_argument("--tag", default="", help="Suffix for the output files")
    args = parser.parse_args()

    frame, target, groups = load(args.input, args.min_class_count)
    columns = feature_columns(frame)
    features = frame[columns]
    is_change_fn = is_change_label

    print(f"{len(frame):,} plots | {len(columns)} features | "
          f"{target.nunique()} classes | {groups.nunique()} blocks | "
          f"{int(sum(is_change_fn(t) for t in target)):,} change plots")

    registry = build_registry(
        columns,
        balanced=args.balanced,
        postclass_bases=args.postclass_base,
        postclass_shared=not args.postclass_per_date,
        hier_bases=args.hier_base,
    )

    if args.only:
        missing = set(args.only) - set(registry)
        if missing:
            print(f"Not available, skipping: {sorted(missing)}")
        registry = {k: v for k, v in registry.items() if k in args.only}

    rows, failures = [], {}
    for name, (factory, needs_frame) in registry.items():
        try:
            row, _ = evaluate(name, factory, features, frame, target, groups,
                              args.n_splits, is_change_fn, needs_frame, args.cv)
            row["cv"] = args.cv
            rows.append(row)
            print(f"{name:16s} acc={row['accuracy']:.3f} bal={row['balanced_accuracy']:.3f} "
                  f"f1={row['f1_macro']:.3f} chg_recall={row['change_recall']:.3f} "
                  f"chg_f1={row['change_f1']:.3f} ({row['seconds']}s)", flush=True)
        except Exception as error:  # a broken model must not sink the sweep
            failures[name] = f"{type(error).__name__}: {error}"
            print(f"{name:16s} FAILED — {type(error).__name__}: {error}", flush=True)

    board = pd.DataFrame(rows).sort_values("change_f1", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    board.to_csv(args.output_dir / f"model_zoo_leaderboard{tag}.csv", index=False)
    (args.output_dir / f"model_zoo_meta{tag}.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "n_plots": len(frame),
                "n_features": len(columns),
                "classes": sorted(target.unique()),
                "n_blocks": int(groups.nunique()),
                "n_splits": args.n_splits,
                "cv": args.cv,
                "balanced_weights": args.balanced,
                "postclass_base": args.postclass_base,
                "postclass_shared_model": not args.postclass_per_date,
                "hier_base": args.hier_base,
                "failures": failures,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(board.to_string(index=False))


if __name__ == "__main__":
    main()
