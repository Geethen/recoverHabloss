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
import contextlib
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


def level_loss(probs, target, mode: str, weight=None, gamma: float = 2.0, *,
               reduce: bool = True, robust: str = "none", robust_q: float = 0.7,
               robust_alpha: float = 0.1, robust_beta: float = 1.0,
               robust_a: float = -4.0):
    """Cross-entropy or focal loss taken over already-aggregated probabilities.

    The hierarchical model reads its coarse levels off group-summed softmax
    probabilities, not raw logits, so the loss is built from ``probs`` directly:
    ``-log p_t`` for cross-entropy, ``-(1 - p_t)^gamma log p_t`` for focal.
    ``weight`` (per class) multiplies the per-sample loss when supplied.

    ``reduce=False`` returns the per-sample vector instead of its mean, which is
    what a sample-selection method (co-teaching) needs and what nothing in this
    file needed before section T.

    ``robust`` swaps the **core** ``-log p_t`` for a bounded surrogate from the
    learning-with-noisy-labels literature, all of which need no estimate of the
    noise rate:

    * ``gce`` -- generalised cross-entropy ``(1 - p_t^q) / q`` (Zhang & Sabuncu
      2018). ``q -> 0`` is CE, ``q = 1`` is unhulled MAE; ``q = 0.7`` is the
      paper's default and interpolates between CE's convergence and MAE's
      noise-robustness.
    * ``sce`` -- symmetric cross-entropy ``alpha*CE + beta*RCE`` (Wang et al.
      2019), with reverse CE ``-A * (1 - p_t)`` for the one-hot label under the
      convention ``log 0 = A``.
    * ``boot_hard`` / ``boot_soft`` -- bootstrapping (Reed et al. 2015): mix the
      given label with the model's own arg-max (hard) or full posterior (soft)
      at weight ``1 - robust_beta``.

    **The focal modulation is dropped whenever a robust core is in use**, and
    that is deliberate rather than an omission. ``(1 - p_t)^gamma`` upweights
    exactly the high-loss samples every one of these surrogates exists to bound,
    so composing them would cancel the mechanism under test. It also means the
    matched reference for a robust arm is the ``loss='ce'`` recipe, not the
    ``loss='focal'`` one.
    """
    import torch

    pt = probs.gather(1, target[:, None]).squeeze(1).clamp_min(1e-8)
    if robust == "none":
        loss = -pt.log()
        if mode in ("focal", "cb_focal"):
            loss = ((1.0 - pt) ** gamma) * loss
    elif robust == "gce":
        loss = (1.0 - pt.pow(robust_q)) / robust_q
    elif robust == "sce":
        loss = robust_alpha * (-pt.log()) + robust_beta * (-robust_a) * (1.0 - pt)
    elif robust in ("boot_hard", "boot_soft"):
        p = probs.clamp_min(1e-8)
        if robust == "boot_hard":
            # Detached so the target is the model's current belief, not a path
            # the gradient can move to make itself right.
            extra = -p.log().gather(1, p.detach().argmax(1)[:, None]).squeeze(1)
        else:
            extra = -(p.detach() * p.log()).sum(1)
        loss = robust_beta * (-pt.log()) + (1.0 - robust_beta) * extra
    else:
        raise ValueError(f"Unknown robust loss: {robust}")
    if weight is not None:
        loss = loss * weight[target]
    return loss.mean() if reduce else loss


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
                 endpoint_weight: float = 0.0,
                 siam_columns_18=None, siam_columns_24=None,
                 siam_extra_columns=None, siam_dim: int = 128,
                 siam_combine: str = "conc", siam_year_adapter: str = "none",
                 siam_crfe: str = "none", siam_pyramid: bool = False,
                 siam_fiim: str = "none", siam_hidden=(512, 256),
                 deep_sup_weight: float = 0.0,
                 dice_weight: float = 0.0, dice_level: str = "gate",
                 set_ce_weight: float = 0.0, set_ce_level: str = "fine",
                 set_ce_alpha: float = 0.10, set_ce_random: bool = False,
                 robust_loss: str = "none", robust_q: float = 0.7,
                 robust_alpha: float = 0.1, robust_beta: float = 1.0,
                 robust_a: float = -4.0, robust_levels: str = "all",
                 cb_levels: str = "all",
                 elr_weight: float = 0.0, elr_beta: float = 0.7,
                 coteach: str = "off", coteach_forget: float = 0.10,
                 coteach_warmup: int = 10, coteach_ramp: int = 10,
                 coteach_level: str = "fine", coteach_beta_a: float = 32.0,
                 coteach_beta_b: float = 2.0, coteach_min_keep: float = 0.10,
                 coteach_thresh_per: str = "batch",
                 coteach_stratify: bool = False,
                 siam_cos_weight: float = 0.0,
                 siam_cos_margin: float = 0.0,
                 siam_cos_stable_margin: float = 0.0,
                 siam_mssm_weight: float = 0.0, siam_mssm_scales: str = "stages",
                 siam_mssm_metric: str = "cos",
                 siam_mssm_margin: float | None = None,
                 siam_mssm_stable_margin: float = 0.0,
                 siam_barlow_weight: float = 0.0,
                 siam_barlow_lambda: float = 0.005,
                 siam_unlabelled_weight: float = 0.0,
                 siam_unlabelled_batch: int = 4096,
                 siam_state_weight: float = 0.0,
                 siam_state_source: str = "external",
                 siam_state_class_weight: str | None = None,
                 siam_state_batch: int = 2048,
                 siam_state_pretrain: int = 0,
                 crt_epochs: int = 0, crt_lr: float = 2e-4,
                 patch_tensor=None, patch_ids=None, patch_augment: bool = True,
                 patch_dim: int = 64,
                 aef_siam: bool = False):
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
        # --- siamese endpoint towers (arch='siamese') ------------------------
        # siam_columns_18 and siam_columns_24 are the SAME features at the two
        # dates, given in the same order, so one shared encoder can read both
        # (see _SiameseTrunk). siam_extra_columns are the per-plot features that
        # are not per-year -- change scalars, an S2 detail block -- and bypass
        # the encoder into the mixer. siam_dim is the endpoint embedding width
        # the cosine and Barlow losses act on; siam_combine picks what the head
        # reads ('diff' or 'conc').
        self.siam_columns_18 = list(siam_columns_18) if siam_columns_18 else None
        self.siam_columns_24 = list(siam_columns_24) if siam_columns_24 else None
        self.siam_extra_columns = list(siam_extra_columns or [])
        self.siam_dim = siam_dim
        self.siam_combine = siam_combine
        # siam_year_adapter frees per-year calibration without unsharing the
        # encoder: 'none' is fully siamese, 'input'/'output' add an
        # identity-initialised diagonal affine per year before/after it. See
        # _SiameseTrunk.
        self.siam_year_adapter = siam_year_adapter
        # --- section Q: modules transcribed from Zhang et al.'s burned-area
        # Swin network (see _SiameseTrunk for what each one becomes without an
        # image grid, and SIAMESE_RESEARCH.md section Q for the verdicts).
        # siam_crfe: 'sum' | 'attn' | 'full' -- their CRFE fusion / channel
        # attention. siam_pyramid: their pyramid decoder, as depth fusion.
        self.siam_crfe = siam_crfe
        self.siam_pyramid = siam_pyramid
        # Hidden widths of the shared encoder. Never swept before section Y --
        # 512/256 was chosen once and inherited by every siamese run since. The
        # two ends are NOT free: the input is one date's columns and the output
        # is `siam_dim`, because the mixer's `d_comb` and every pair loss are
        # computed from it. Only the middle moves.
        self.siam_hidden = tuple(siam_hidden)
        # --- section Q10: modules transcribed from SNIIF-Net (Sci Rep 2025,
        # s41598-025-15468-w), whose siamese layout is this one's with an image
        # grid under it. siam_fiim is their Feature Information Interaction
        # Module: each date's embedding is re-weighted by a gate that reads BOTH
        # dates, so the pair interacts before the difference is taken rather than
        # after it. That is the one thing section Q's CRFE gate does not do --
        # `crfe='attn'` gates the ASSEMBLED block, downstream of the subtraction,
        # so it cannot change z18/z24 themselves and therefore cannot change what
        # the cosine losses read. See _SiameseTrunk.
        self.siam_fiim = siam_fiim
        # deep_sup_weight > 0 hangs an auxiliary coarse3 head off EVERY hidden
        # stage of the shared encoder and supervises it through the same three
        # nested levels. Discarded at predict time -- like the state head, it
        # exists to put gradient into f and costs nothing at serving.
        self.deep_sup_weight = deep_sup_weight
        # dice_weight > 0 adds a soft-Dice term to the focal objective (their
        # "hybrid loss"). dice_level='gate' takes it on the change class, which
        # is a differentiable relaxation of the headline change-F1;
        # dice_level='fine' takes an unweighted mean over the nine coarse3
        # classes, which is the relaxation of `focus_macro_f1` instead.
        if dice_level not in ("gate", "fine"):
            raise ValueError(f"Unknown dice_level: {dice_level}")
        self.dice_weight = dice_weight
        self.dice_level = dice_level
        if set_ce_level not in ("fine", "merged", "both"):
            raise ValueError(f"Unknown set_ce_level: {set_ce_level}")
        self.set_ce_weight = set_ce_weight
        self.set_ce_level = set_ce_level
        self.set_ce_alpha = set_ce_alpha
        self.set_ce_random = set_ce_random
        # --- section T: learning with noisy labels ---------------------------
        # Every knob here addresses the same measured fact from a different
        # direction: the coarse3 target carries interpreter disagreement on the
        # Cropland/Nature boundary (analyse_label_noise.py), and the model is
        # trained to fit it. `robust_loss` bounds the per-sample loss so a
        # mislabelled row cannot dominate the gradient; `elr_weight` holds the
        # model to what it believed before it memorised that row; `coteach`
        # declines to train on it at all.
        #
        # robust_loss / robust_levels: see level_loss. The focal modulation is
        # dropped wherever a robust core applies, so the matched reference for
        # these arms is loss='ce'.
        if robust_loss not in ("none", "gce", "sce", "boot_hard", "boot_soft"):
            raise ValueError(f"Unknown robust_loss: {robust_loss}")
        if robust_levels not in ("fine", "all"):
            raise ValueError(f"Unknown robust_levels: {robust_levels}")
        self.robust_loss = robust_loss
        self.robust_q = robust_q
        self.robust_alpha = robust_alpha
        self.robust_beta = robust_beta
        self.robust_a = robust_a
        self.robust_levels = robust_levels
        # cb_levels: which levels the `weighted_ce` / `cb_focal` class weights
        # reach. 'all' is the historical behaviour -- gate, merged2 and fine all
        # get them. 'fine' puts them on the nine coarse3 classes only, where the
        # 46-plot transitions live, and leaves the 2-class gate unweighted: the
        # gate is ~4:1, and reweighting it trades change precision for recall,
        # which N13 established this product does not want. Without this the two
        # effects are confounded in one `loss=` string.
        if cb_levels not in ("fine", "all"):
            raise ValueError(f"Unknown cb_levels: {cb_levels}")
        self.cb_levels = cb_levels
        # elr_weight > 0 adds early-learning regularisation (Liu et al. 2020):
        # an EMA of the model's own coarse3 posterior per training row, and a
        # term that penalises moving away from it. The premise is the empirical
        # early-learning phase -- a network fits the clean majority first and
        # memorises the mislabelled rows later -- so the EMA target is a record
        # of what the model believed before memorisation, and it needs no noise
        # estimate. elr_beta is the EMA momentum.
        self.elr_weight = elr_weight
        self.elr_beta = elr_beta
        # coteach: two independently initialised networks, each selecting the
        # rows the OTHER one trains on for that step.
        #   'stochastic'  Bertels et al. (2023) -- keep a row when its
        #                 ground-truth-label posterior under the peer clears a
        #                 threshold drawn from Beta(a, b), ramped in over
        #                 coteach_warmup/coteach_ramp epochs. **Needs no
        #                 forget-rate and so no estimate of the noise level**,
        #                 which is the property that makes it the arm worth
        #                 running here: this project has no clean validation set
        #                 to fit a forget rate against.
        #   'classic'     Han et al. (2018) -- keep the (1 - forget) fraction of
        #                 SMALLEST per-sample loss. Registered as the contrast
        #                 that shows what the estimate buys.
        #   'random'      the control: keep the same fraction, chosen at random.
        #                 Without it, a gain from either arm above is
        #                 indistinguishable from the regularisation of training
        #                 on a random subsample.
        # Only network A is ever served, so no arm can win by being a two-model
        # ensemble; the peer exists to filter, and is discarded after fit().
        if coteach not in ("off", "stochastic", "classic", "random"):
            raise ValueError(f"Unknown coteach mode: {coteach}")
        if coteach_level not in ("fine", "merged", "gate"):
            raise ValueError(f"Unknown coteach_level: {coteach_level}")
        if coteach_thresh_per not in ("batch", "instance"):
            raise ValueError(f"Unknown coteach_thresh_per: {coteach_thresh_per}")
        self.coteach = coteach
        self.coteach_forget = coteach_forget
        self.coteach_warmup = coteach_warmup
        self.coteach_ramp = coteach_ramp
        self.coteach_level = coteach_level
        self.coteach_beta_a = coteach_beta_a
        self.coteach_beta_b = coteach_beta_b
        self.coteach_min_keep = coteach_min_keep
        self.coteach_thresh_per = coteach_thresh_per
        # coteach_stratify applies the keep rule WITHIN each coarse3 class, so
        # the forget budget is spent at the same rate on every class instead of
        # almost entirely on the rare ones. Measured on this target, an
        # unstratified forget rate of 0.10 keeps 24% of the 46-plot
        # `Artificial -> Cropland` steps and 99% of `Nature -> Nature`
        # (coteach_diagnostics.py) -- a small-loss criterion ranks rarity, and
        # rarity and mislabelling are not the same thing. This is the same
        # correction Mondrian conformal makes to a pooled cut in section R, and
        # it is what separates "selection does not work here" from "selection
        # was never given a chance to filter noise rather than difficulty".
        self.coteach_stratify = coteach_stratify
        # Diagnostics, filled by fit(). keep_counts_ is the per-training-row
        # count of epochs the row survived selection -- the object the section's
        # relabelling queue is built from, and the reason to keep a rejected
        # row's identity rather than only its count.
        self.coteach_keep_rate_ = None
        self.coteach_guard_rate_ = None
        self.coteach_keep_counts_ = None
        self.coteach_rows_ = None
        # siam_cos_weight > 0 supervises the endpoint cosine directly with the
        # gate label: a stable plot is one land cover measured twice and its two
        # embeddings should point the same way, a change plot's should not.
        # siam_cos_margin is the cosine a change pair is allowed to keep before
        # it is penalised (0 = push to orthogonal).
        self.siam_cos_weight = siam_cos_weight
        self.siam_cos_margin = siam_cos_margin
        # siam_cos_stable_margin is the SNIIF-Net double-margin form of the same
        # term: the stable side gets slack too, so a pair that is already within
        # `1 - cos <= eps` stops being pulled. The published loss is
        # max(D - eps, 0)^2 on the unchanged side against max(theta - D, 0)^2 on
        # the changed one, i.e. both sides hinged; this project's term hinges
        # only the change side and drives every stable pair at 1 - cos towards
        # exactly 1. 0.0 is the old behaviour exactly (cos <= 1, so the hinge is
        # never active).
        self.siam_cos_stable_margin = siam_cos_stable_margin
        # siam_mssm_weight > 0 adds SNIIF-Net's Multi-Scale Supervision Method:
        # the SAME pair objective as siam_cos_weight, repeated at every hidden
        # stage of the shared encoder rather than at the final embedding alone.
        # This is deep supervision (Q5) with the pair geometry as the auxiliary
        # objective instead of a classification head, and that distinction is
        # what makes it a separate question: Q5 was flat because the three nested
        # levels already supervise the representation at full depth, but NOTHING
        # currently constrains the pair at intermediate depth -- the cosine term
        # reads z and only z. No parameters, so serving cost is unchanged (Q5's
        # heads at least existed).
        #
        # siam_mssm_scales: 'stages' = the hidden stages only (the final
        # embedding is already covered by siam_cos_weight); 'all' adds the final
        # embedding, which is the paper's literal reading -- it constrains all
        # four of its decoder scales including the one the prediction is made
        # from. siam_mssm_metric: 'cos' reuses the hinge above; 'euclid' is the
        # paper's squared-hinge Euclidean form. siam_mssm_margin defaults to
        # siam_cos_margin ('cos') or 1.2 ('euclid').
        if siam_mssm_scales not in ("stages", "all"):
            raise ValueError(f"Unknown siam_mssm_scales: {siam_mssm_scales}")
        if siam_mssm_metric not in ("cos", "euclid"):
            raise ValueError(f"Unknown siam_mssm_metric: {siam_mssm_metric}")
        self.siam_mssm_weight = siam_mssm_weight
        self.siam_mssm_scales = siam_mssm_scales
        self.siam_mssm_metric = siam_mssm_metric
        self.siam_mssm_margin = (
            siam_mssm_margin if siam_mssm_margin is not None
            else (1.2 if siam_mssm_metric == "euclid" else siam_cos_margin))
        self.siam_mssm_stable_margin = siam_mssm_stable_margin
        # siam_barlow_weight > 0 adds Barlow Twins redundancy reduction over the
        # STABLE endpoint pairs only, which supply two genuine views of one
        # unchanged patch without an augmentation policy. siam_barlow_lambda is
        # the paper's off-diagonal trade-off.
        self.siam_barlow_weight = siam_barlow_weight
        self.siam_barlow_lambda = siam_barlow_lambda
        # siam_unlabelled_weight > 0 applies the SAME Barlow term to a minibatch
        # of the unlabelled endpoint pool passed to fit() as ``unlabelled_frame``
        # (build_unlabelled_aef.py), every pixel of which is assumed stable. This
        # is the only term in the section that adds information rather than
        # rearranging the labelled plots; it is separate from siam_barlow_weight
        # so the labelled and unlabelled halves can be weighted apart.
        self.siam_unlabelled_weight = siam_unlabelled_weight
        self._siam_unlabelled_batch = siam_unlabelled_batch
        # siam_state_weight > 0 adds an auxiliary SINGLE-DATE land-cover state
        # head on the shared encoder: g(f(x)) -> {Nature, Cropland, Artificial}.
        # This is the one place external single-date labels can enter the model
        # at all -- f is a function of one date, while the flat trunk's first
        # layer eats [2018 | 2024 | diff] and has no single-date input. N4 showed
        # more *unlabelled* AlphaEarth buys nothing and concluded the shortage is
        # labels; a state label is a label.
        #
        # siam_state_source picks where the supervision comes from:
        #   'endogenous' -- the training plots' own endpoints. A transition
        #                   label From -> To is two free state labels, so this
        #                   adds no data and tests the MECHANISM alone. Expect
        #                   little: F1 (TWOTOWER_RESEARCH) supervised the same
        #                   marginals at the softmax and came back flat.
        #   'external'   -- a separate single-date pool (build_state_labels.py).
        #                   This is the only setting that adds information.
        #   'both'       -- the two terms summed, equally weighted.
        # Running 'endogenous' beside 'external' is what separates "the head
        # helps" from "the new labels help"; neither number means much alone.
        self.siam_state_weight = siam_state_weight
        self.siam_state_source = siam_state_source
        # 'balanced' re-weights the state head's cross-entropy by 1/class
        # frequency, exactly as `statepre.models._MLP(class_weight="balanced")`
        # does. STATE_PRETRAIN_RESEARCH.md section Y3 is why: it is the only
        # thing tested there that moves the *ratio* of two errors rather than
        # their sum -- cropland grows (`f1_cropland` +0.006/+0.004), the growth
        # comes out of Nature (`nature_as_cropland` +0.008/+0.009), and built-up
        # leaks LESS into cropland (`artificial_as_cropland` -0.004/-0.004), at
        # both fold counts. That is the user's Oslo constraint stated as metrics.
        # It costs state-level macro-F1 (-0.001/-0.005), so it is a trade and has
        # to be judged on the map, which is what this option exists to produce.
        self.siam_state_class_weight = siam_state_class_weight
        self._siam_state_batch = siam_state_batch
        # siam_state_pretrain > 0 runs the SAME state supervision as a separate
        # phase BEFORE the transition loss is ever seen, instead of alongside it:
        # n epochs of g(f(x)) -> state on the pool alone, then the usual fit from
        # those weights with the auxiliary term off. Every arm of sections N14
        # and P weighted the two objectives against each other at one optimiser
        # step -- discarded side-loss, output parameterisation and input feature
        # all landed flat -- and joint training is the one thing they share. It
        # is a distinguishable experiment for two reasons: the encoder gets the
        # pool's full capacity rather than a 0.3-weighted share, and the
        # transition loss starts from a state-organised representation rather
        # than from noise. It is NOT expected to break the section-P finding
        # (GLanCE's errors are correlated with AlphaEarth's however they are fed)
        # and it is registered so that stays a measurement rather than a guess.
        #
        # siam_state_source selects the pool exactly as above, and the
        # endogenous arm is the control that matters: pretraining on the training
        # plots' OWN endpoints adds no data, so it separates "a state-organised
        # initialisation helps" from "GLanCE's labels help".
        self.siam_state_pretrain = siam_state_pretrain
        # crt_epochs > 0 runs Kang et al.'s classifier re-training after the main
        # loop: the trunk is frozen at its served eval-mode behaviour and only
        # the head is retrained, on a class-balanced draw. No new parameters and
        # no serving cost -- the served graph is identical, only the head's
        # weights differ. See _retrain_classifier for why this is not the
        # already-negative G-H sampler.
        self.crt_epochs = crt_epochs
        self.crt_lr = crt_lr
        self._crt_pair = None
        # patch_tensor turns the two-tower's DETAIL tower into a small conv
        # encoder over the stored (2 years x C bands x H x W) Sentinel-2 patches,
        # instead of a wide MLP over hand-built statistics. S3 tested a FLATTENED
        # 8x8 pooled patch (1,344 raw columns) and it was the worst result on the
        # board; this tests the two things that test did not have -- weight
        # sharing across the image, and the eight free dihedral augmentations
        # that satellite patches admit and tabular columns do not. Privileged
        # exactly like the columnar detail tower: mask-gated, dropped at serving,
        # so no patch is ever read at inference.
        self.patch_tensor = patch_tensor
        self.patch_ids = patch_ids
        self.patch_augment = patch_augment
        self.patch_dim = patch_dim   # dropout comes from dropout_tess, as columnar does
        self._patch_pos = ({pid: i for i, pid in enumerate(patch_ids)}
                           if patch_ids is not None else None)
        # aef_siam makes the two-tower's AlphaEarth tower a shared endpoint
        # encoder (section N). The 2018/2024 split is derived from the COLUMN
        # NAMES, so aef_columns may arrive in any order -- see
        # _aef_siam_permutation.
        self.aef_siam = aef_siam

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
                aef_siam_perm=self._aef_siam_permutation() if self.aef_siam else None,
                aef_siam_dim=self.siam_dim, aef_siam_combine=self.siam_combine,
                aef_siam_crfe=self.siam_crfe, aef_siam_pyramid=self.siam_pyramid,
                aef_siam_hidden=self.siam_hidden,
                aef_siam_fiim=self.siam_fiim,
                patch_tensor=self.patch_tensor, patch_augment=self.patch_augment,
                patch_dim=self.patch_dim,
            )
            return trunk, self.tower_dim
        if self.arch == "siamese":
            # d is the packed width [x18 | x24 | extra]; the two endpoint blocks
            # are equal by construction in _prepare, so d_end fixes the split.
            d_end = len(self.siam_columns_18)
            trunk = _SiameseTrunk(
                d_end=d_end, d_extra=d - 2 * d_end, out_dim=self.tower_dim,
                siam_dim=self.siam_dim, dropout=0.4, combine=self.siam_combine,
                year_adapter=self.siam_year_adapter,
                crfe=self.siam_crfe, pyramid=self.siam_pyramid,
                fiim=self.siam_fiim, hidden=self.siam_hidden,
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
            if self.patch_tensor is not None:
                # A raw, UNstandardised row-index column appended after the
                # masks. The detail tower gathers its 64x64 patches with it
                # rather than reading them out of the frame, because a
                # (2 x 4 x 64 x 64) image per plot cannot be a set of DataFrame
                # columns. Carrying an index instead of the pixels keeps every
                # other path -- standardisation, minibatching, the mask gate --
                # exactly as it is for a columnar detail tower.
                idx = self._patch_index(frame).reshape(-1, 1).astype("float32")
                return np.concatenate([Xa, Xt, ma, mt, idx], axis=1).astype("float32")
            return np.concatenate([Xa, Xt, ma, mt], axis=1).astype("float32")
        if self.arch == "siamese":
            # Pack [x18 | x24 | extra]. The two endpoint blocks are standardised
            # with POOLED statistics -- one mu/sd per feature computed over both
            # years stacked -- because a shared encoder that is handed two
            # differently-centred versions of the same measurement has had its
            # weight sharing undone before the first layer. It also keeps
            # z24 - z18 and cos(z18, z24) meaningful: with per-year statistics a
            # feature that simply drifted between 2018 and 2024 would be
            # re-centred away and read as no change. ``extra`` is standardised on
            # its own, NaN-safe, exactly as the flat branch below.
            X18 = frame[self.siam_columns_18].to_numpy("float32")
            X24 = frame[self.siam_columns_24].to_numpy("float32")
            Xe = (frame[self.siam_extra_columns].to_numpy("float32")
                  if self.siam_extra_columns
                  else np.zeros((len(frame), 0), "float32"))
            if fit:
                pooled = np.concatenate([X18, X24], axis=0)
                self.mu_end = np.nanmean(pooled, 0)
                self.sd_end = np.nanstd(pooled, 0) + 1e-8
                self.mu_extra = (np.nanmean(Xe, 0) if Xe.shape[1]
                                 else np.zeros(0, "float32"))
                self.sd_extra = (np.nanstd(Xe, 0) + 1e-8 if Xe.shape[1]
                                 else np.ones(0, "float32"))

            def norm(X, mu, sd):
                Z = (X - mu) / sd
                return np.where(np.isfinite(Z), Z, 0.0)

            return np.concatenate(
                [norm(X18, self.mu_end, self.sd_end),
                 norm(X24, self.mu_end, self.sd_end),
                 norm(Xe, self.mu_extra, self.sd_extra)],
                axis=1).astype("float32")
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
        # Where 'stable' landed in the sorted gate list -- the siamese auxiliary
        # losses split the batch on it and must not assume the ordering. -1 when
        # a fold is single-sided, which both losses treat as "no pairs".
        self._stable_gate_idx = (self.gate_classes_.index("stable")
                                 if "stable" in self.gate_classes_ else -1)

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

    def _build_network(self, in_shape, n_fine: int, seed: int):
        """Trunk plus fine head, seeded, registered in ``self._modules_``.

        Split out of ``fit`` so that a **second, independently initialised copy**
        of the same architecture can be built for co-teaching without
        duplicating any of fit's target, hierarchy or standardisation setup --
        the peer shares all of that by construction (see ``_make_peer``) and
        differs only in its weights and in the rows it is shown.
        """
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        if self.arch == "gru":
            self.gru = nn.GRU(in_shape[2], self.hidden, batch_first=True,
                              bidirectional=True).to(self.device)
            rep_dim = 2 * self.hidden
        else:
            self.trunk, rep_dim = self._flat_trunk(in_shape[1])
            self.trunk.to(self.device)
        head_modules = self._build_head(rep_dim, n_fine)
        trunk = (self.gru,) if self.arch == "gru" else (self.trunk,)
        self._modules_ = trunk + head_modules

    def _make_peer(self, in_shape, n_fine: int):
        """The co-teaching partner: same configuration, different initialisation.

        A shallow copy, so every fitted-but-shared piece -- the standardisation
        statistics, the class lists, the 0/1 level-aggregation matrices, the
        endpoint index tensors -- is the *same object* on both networks and
        cannot drift apart. Only the parameters are rebuilt, under a different
        torch seed, which is the entire source of the two networks' disagreement
        and therefore of the method: Han et al.'s premise is that two nets
        initialised apart filter different errors for each other, and a peer that
        shared the initialisation would filter nothing.
        """
        import copy

        peer = copy.copy(self)
        peer._build_network(in_shape, n_fine, self.seed + 7919)
        return peer

    def _coteach_keep(self, p_true, per_loss, epoch: int, rng, groups=None):
        """Rows this network selects for its partner to train on this step.

        Returns ``(keep, guard_fired)``. ``p_true`` is the posterior of the
        *given* label at ``coteach_level``, ``per_loss`` the per-row three-level
        loss; which one is read depends on the mode. ``groups`` is the per-row
        coarse3 class, read only when ``coteach_stratify`` is set.
        """
        import torch

        n = int(p_true.numel())
        device = p_true.device
        if self.coteach in ("classic", "random"):
            # Han et al.'s schedule: forget nothing at epoch 0, ramp the forget
            # rate linearly to `coteach_forget` over `coteach_ramp` epochs.
            tau = self.coteach_forget * min(1.0, (epoch + 1) / max(self.coteach_ramp, 1))
            keep = torch.zeros(n, dtype=torch.bool, device=device)
            # Stratified: the same forget rate applied inside every class, so
            # the budget cannot be spent on rarity. Unstratified: one pooled
            # ranking, which is the published method.
            blocks = ([torch.arange(n, device=device)] if not self.coteach_stratify
                      else [torch.nonzero(groups == c, as_tuple=True)[0]
                            for c in torch.unique(groups)])
            for block in blocks:
                if len(block) == 0:
                    continue
                k = max(1, int(round((1.0 - tau) * len(block))))
                if self.coteach == "classic":
                    keep[block[per_loss[block].argsort()[:k]]] = True
                else:
                    pick = rng.choice(len(block), size=k, replace=False)
                    keep[block[torch.as_tensor(pick, device=device,
                                               dtype=torch.long)]] = True
            return keep, False
        if self.coteach_stratify:
            raise ValueError(
                "coteach_stratify is defined for the rank-based selectors "
                "(classic/random). The stochastic threshold is a cut on the "
                "posterior itself, and stratifying it means calibrating a "
                "per-class cut -- which is Mondrian conformal, already measured "
                "post-hoc in section R, not this method.")

        # Stochastic co-teaching (Bertels et al. 2023, Sci Rep 13:16875). The
        # threshold is a draw from Beta(a, b) -- mass near 1, with a tail that
        # occasionally admits a low-posterior row -- scaled by the ramp
        # eta = clip((epoch - warmup) / ramp, 0, 1) so early epochs (eta = 0,
        # threshold 0) train on everything and selection phases in. There is no
        # forget rate anywhere in that, which is the point.
        eta = float(np.clip((epoch - self.coteach_warmup) / max(self.coteach_ramp, 1),
                            0.0, 1.0))
        size = n if self.coteach_thresh_per == "instance" else 1
        for _ in range(5):
            draw = np.clip(rng.beta(self.coteach_beta_a, self.coteach_beta_b,
                                    size=size), 0.01, 0.99) * eta
            thresh = torch.as_tensor(draw, device=device, dtype=p_true.dtype)
            keep = p_true > thresh
            if float(keep.float().mean()) >= self.coteach_min_keep:
                return keep, False
        # The paper's guard resamples the *mini-batch* after five failed draws.
        # This project trains full-batch by default, so there is no other batch
        # to draw; the fallback keeps the `coteach_min_keep` highest-posterior
        # rows instead. Recorded as `coteach_guard_rate_` because a high rate
        # means the Beta prior is mismatched to this model's confidence, not
        # that the data is clean -- the paper's alpha=32, beta=2 assumes a model
        # whose true-class posterior reaches ~0.94, and a 9-class hierarchical
        # head on 6.5k plots does not.
        k = max(1, int(round(self.coteach_min_keep * n)))
        keep = torch.zeros(n, dtype=torch.bool, device=device)
        keep[p_true.argsort(descending=True)[:k]] = True
        return keep, True

    def _elr_loss(self, p_fine, idx):
        """Early-learning regularisation (Liu et al. 2020) on the coarse3 head.

        ``log(1 - <t_i, p_i>)`` against an EMA ``t_i`` of the model's own
        posterior for that row. Early in training the EMA records the clean
        early-learning fit; later it opposes the gradient that would memorise a
        mislabelled row. Needs the *global* row index, which is why it lives in
        ``batch_loss`` rather than in ``_levels``.
        """
        import torch

        p = p_fine.clamp(1e-6, 1.0 - 1e-6)
        with torch.no_grad():
            q = p.detach()
            q = q / q.sum(1, keepdim=True).clamp_min(1e-8)
            self._elr_target[idx] = (self.elr_beta * self._elr_target[idx]
                                     + (1.0 - self.elr_beta) * q)
        inner = (self._elr_target[idx] * p).sum(1).clamp_max(1.0 - 1e-4)
        return torch.log(1.0 - inner).mean()

    def _build_head(self, rep_dim: int, n_fine: int):
        """Fine-logit head: flat linear, bilinear, or endpoint-tied factorised."""
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
        if self.head == "proto":
            # Cosine (prototype) classifier: logit_k = s * cos(rep, w_k). One
            # learnable prototype per coarse3 class plus one learnable scale.
            #
            # The mechanism is long-tail specific and is NOT what focal loss
            # already does. A plain linear head trained on a 4,200-vs-46 split
            # develops class weight NORMS proportional to class frequency, so the
            # rare class loses on magnitude before its direction is ever
            # consulted; focal reweights the loss but leaves that geometry in
            # place. Normalising both sides removes the magnitude channel
            # entirely and leaves only the direction -- which is the same
            # decoupling insight as O2, applied to the parameterisation rather
            # than to the training schedule (Kang et al.'s tau-normalisation).
            self.proto = nn.Parameter(
                torch.randn(n_fine, rep_dim, device=self.device) * 0.02)
            self.proto_scale = nn.Parameter(torch.tensor(10.0, device=self.device))
            return ()
        if self.head in ("endpoint", "endpoint_pure"):
            if self.arch != "siamese" and not self.aef_siam:
                raise ValueError(
                    f"head={self.head!r} reads the endpoint pair (z18, z24) and "
                    "so needs a shared encoder -- set arch='siamese', or "
                    "arch='two_tower' with aef_siam=True. On a flat trunk the "
                    "factorisation has no date structure to tie the two state "
                    "reads to, which is `head='bilinear'` and is already "
                    "tested-negative.")
            n_base = len(self.base_classes_)
            # ONE state head, applied to both endpoint embeddings. This is the
            # whole difference from `bilinear` and it is only expressible on a
            # siamese trunk: there, f is a function of a single date, so g(z18)
            # and g(z24) are the same question asked twice and the two marginals
            # pool their evidence. `bilinear` reads two SEPARATE heads off the
            # one fused representation, which has no date structure to tie them
            # to -- and it is on the tested-negative list.
            self.state_head = nn.Linear(self.siam_dim, n_base).to(self.device)
            # Learned per-transition log-prior, 9 scalars. Without it the head
            # asserts from and to are conditionally independent given the pair,
            # which is false: the transition matrix is strongly diagonal.
            self.trans_prior = nn.Parameter(torch.zeros(n_fine, device=self.device))
            mods = (self.state_head,)
            if self.head == "endpoint":
                # Zero-initialised residual over the FUSED representation, so
                # training starts at exactly the pure factorised model and the
                # residual has to earn every logit it moves. It is also the only
                # path by which the privileged Sentinel-2 detail tower reaches
                # the fine head at all -- the factorised term reads the
                # AlphaEarth endpoint embeddings and nothing else, so without
                # this `endpoint_pure` on a two-tower silently discards the
                # detail tower from the coarse3 read.
                self.fine_head = nn.Linear(rep_dim, n_fine).to(self.device)
                nn.init.zeros_(self.fine_head.weight)
                nn.init.zeros_(self.fine_head.bias)
                mods = mods + (self.fine_head,)
            return mods
        raise ValueError(f"Unknown head: {self.head}")

    def _patch_index(self, frame) -> np.ndarray:
        """Row of ``patch_tensor`` for every row of ``frame``, -1 where absent.

        Resolved by PLOTID rather than by position, because the frame is built by
        merges and reindexed by the CV split; a positional assumption would
        silently pair a plot with another plot's imagery, which no metric in this
        project would catch.
        """
        if "PLOTID" not in frame.columns:
            raise ValueError("a patch detail tower needs PLOTID on the frame")
        return np.array([self._patch_pos.get(p, -1) for p in frame["PLOTID"]],
                        dtype="int64")

    def _loose_params(self) -> list:
        """Head parameters that live outside any ``nn.Module``.

        ``fine_bias`` and ``trans_prior`` are bare ``nn.Parameter``s, so they
        reach neither ``_modules_.parameters()`` nor ``state_dict()``. Collected
        in one place because forgetting them in *either* the optimiser or the
        early-stopping snapshot fails silently -- an unoptimised prior stays at
        its zero init and simply looks like the idea not working.
        """
        loose = []
        if self.head == "bilinear":
            loose.append(self.fine_bias)
        if self.head in ("endpoint", "endpoint_pure"):
            loose.append(self.trans_prior)
        if self.head == "proto":
            loose.extend([self.proto, self.proto_scale])
        return loose

    def _fine_logits(self, rep):
        """Coarse3 logits from the representation, per the configured head."""
        import torch.nn.functional as F

        if self.head == "flat":
            return self.fine_head(rep)
        if self.head == "proto":
            return self.proto_scale * F.normalize(rep, dim=1) @ \
                F.normalize(self.proto, dim=1).T
        if self.head in ("endpoint", "endpoint_pure"):
            # logit(A -> B) = log P(from=A | z18) + log P(to=B | z24) + prior(A -> B)
            #
            # The point is N0: `Artificial -> Cropland` has 46 plots and every
            # model in section N returns it at 0.000, because a 9-way softmax
            # has to find a decision boundary for it against 4,200 stable plots.
            # This head never draws that boundary. Its 46 plots are read through
            # `from=Artificial` (~1k plots) and `to=Cropland` (~1.5k plots) plus
            # one scalar, so the rare cell inherits the marginals' support.
            #
            # log_softmax rather than raw logits, so the two terms are genuine
            # log-probabilities and the sum is a proper factorised model up to
            # the prior -- with raw logits the scale of each head would be free
            # and the factorisation would carry no probabilistic meaning.
            z18, z24 = self._siam_pair()
            lp_from = F.log_softmax(self.state_head(z18), dim=1)
            lp_to = F.log_softmax(self.state_head(z24), dim=1)
            logits = (lp_from[:, self._from_idx] + lp_to[:, self._to_idx]
                      + self.trans_prior)
            if self.head == "endpoint":
                logits = logits + self.fine_head(rep)
            return logits
        # Bilinear: logit(A -> B) = from[A] + to[B] + bias(A -> B). Shares every
        # transition that shares a from- or to-class, so rare transitions borrow
        # strength from the common ones on the same endpoint.
        from_logits = self.from_head(rep)[:, self._from_idx]
        to_logits = self.to_head(rep)[:, self._to_idx]
        return from_logits + to_logits + self.fine_bias

    def _levels(self, p_fine, fine_t, merged_t, gate_t,
                w_fine, w_merged, w_gate, wf, wm, wg, extras: bool = True,
                per_sample: bool = False):
        """Weighted sum of the fine/merged/gate losses for one batch of probs.

        ``extras=False`` drops the terms that are properties of the *objective*
        rather than of the three levels -- the endpoint marginals and the Dice
        term. The deep-supervision heads pass it, so that composing deep
        supervision with either of those does not silently apply them once per
        auxiliary head as well as once on the main output.

        ``per_sample=True`` returns the three-level sum as a per-row vector and
        implies ``extras=False``: every extra term is a statistic of the *batch*
        (a Dice overlap, a conformal quantile) and has no per-row value to
        return. It is the read a sample-selection method scores rows with.
        """
        p_merged = p_fine @ self._M
        p_gate = p_merged @ self._G
        # Forward correction: the fine head predicts the *clean* posterior and is
        # pushed through T to explain the noisy labels (identity if noise_rate=0).
        p_fine_obs = p_fine @ self._T if self._T is not None else p_fine
        rob = dict(reduce=not per_sample, robust=self.robust_loss,
                   robust_q=self.robust_q, robust_alpha=self.robust_alpha,
                   robust_beta=self.robust_beta, robust_a=self.robust_a)
        # `robust_levels='fine'` puts the bounded surrogate only where this
        # project measures the noise -- the coarse3 Cropland/Nature boundary --
        # and leaves the merged2/gate levels, which the legend already absorbs
        # the noise on, at the configured loss.
        coarse = dict(rob) if self.robust_levels == "all" else dict(
            rob, robust="none")
        loss = (wf * level_loss(p_fine_obs, fine_t, self.loss, w_fine, self.gamma,
                                **rob)
                + wm * level_loss(p_merged, merged_t, self.loss, w_merged,
                                  self.gamma, **coarse)
                + wg * level_loss(p_gate, gate_t, self.loss, w_gate, self.gamma,
                                  **coarse))
        if per_sample:
            return loss
        if not extras:
            return loss
        if self.endpoint_weight > 0:
            loss = loss + self.endpoint_weight * self._endpoint_loss(p_merged, merged_t)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self._dice_loss(p_fine, p_gate,
                                                             fine_t, gate_t)
        if self.set_ce_weight > 0:
            terms = []
            if self.set_ce_level in ("fine", "both"):
                terms.append(self._set_ce_loss(p_fine, fine_t, level_offset=0))
            if self.set_ce_level in ("merged", "both"):
                terms.append(self._set_ce_loss(p_merged, merged_t, level_offset=1000003))
            loss = loss + self.set_ce_weight * sum(terms)
        return loss

    def _set_ce_loss(self, probs, target, level_offset: int = 0):
        """Cross-entropy renormalised to the conformal prediction set (section S).

        ``-log( p_y / sum_{k in S} p_k )`` -- *given* the answer is one of these
        k classes, be right. The set comes from :meth:`_conformal_sets` and is a
        constant mask, so this reweights which pairs of logits get pushed apart
        without differentiating through the quantile.
        """
        import torch

        built = self._conformal_sets(probs, target, level_offset)
        if built is None:
            return probs.new_tensor(0.0)
        mask, score_idx, _cal_idx, _q = built
        with torch.no_grad():
            # A singleton restricted CE is identically zero: no gradient, and
            # averaging over those rows would only dilute the term.
            keep = mask.sum(1) >= 2
        if not bool(keep.any()):
            return probs.new_tensor(0.0)
        p = probs[score_idx][keep].clamp_min(1e-8)
        y = target[score_idx][keep]
        chosen = mask[keep]
        numer = p.gather(1, y[:, None]).squeeze(1)
        denom = (p * chosen.to(p.dtype)).sum(1).clamp_min(1e-8)
        return -(torch.log(numer) - torch.log(denom)).mean()

    def _conformal_sets(self, probs, target, level_offset: int = 0):
        """Mondrian LAC sets over the scored half of a batch, as a constant mask.

        ConfTr's split construction: half the rows calibrate the per-class
        quantiles and the *other* half is scored, alternating each epoch, or the
        term is trivially satisfied by the rows that set the threshold. The
        quantile is a hard order statistic, so the whole construction runs under
        ``no_grad`` and the set reaches the loss as a 0/1 mask.

        Returns ``(mask, score_idx, cal_idx, q)`` or ``None`` when the batch is
        too small to split. ``mask`` is over ``score_idx`` rows only.
        """
        import torch

        n, n_classes = probs.shape
        if n < 2 or n_classes < 2:
            return None
        epoch = int(getattr(self, "_epoch", 0))
        generator = torch.Generator(device=probs.device)
        generator.manual_seed(int(self.seed) + level_offset)
        perm = torch.randperm(n, device=probs.device, generator=generator)
        mid = n // 2
        if epoch % 2 == 0:
            cal_idx, score_idx = perm[:mid], perm[mid:]
        else:
            cal_idx, score_idx = perm[mid:], perm[:mid]
        if len(cal_idx) == 0 or len(score_idx) == 0:
            return None

        with torch.no_grad():
            cal_p = probs[cal_idx]
            cal_y = target[cal_idx]
            q = torch.empty(n_classes, device=probs.device, dtype=probs.dtype)
            for cls in range(n_classes):
                scores = 1.0 - cal_p[cal_y == cls, cls]
                count = int(scores.numel())
                rank = int(np.ceil((count + 1) * (1.0 - self.set_ce_alpha)))
                if count == 0 or rank > count:
                    q[cls] = 1.0
                else:
                    q[cls] = scores.sort().values[rank - 1].clamp_max(1.0)

            score_p = probs[score_idx]
            score_y = target[score_idx]
            mask = (1.0 - score_p) <= q[None, :]
            mask.scatter_(1, score_y[:, None], True)

            if self.set_ce_random:
                sizes = mask.sum(1).clamp(1, n_classes)
                rand_gen = torch.Generator(device=probs.device)
                rand_gen.manual_seed(int(self.seed) + level_offset + 7919 * (epoch + 1))
                noise = torch.rand(len(score_idx), n_classes, device=probs.device,
                                   generator=rand_gen)
                noise.scatter_(1, score_y[:, None], float("inf"))
                order = noise.argsort(dim=1)
                rank = torch.empty_like(order)
                positions = torch.arange(n_classes, device=probs.device)[None, :].expand_as(order)
                rank.scatter_(1, order, positions)
                mask = rank < (sizes[:, None] - 1)
                mask.scatter_(1, score_y[:, None], True)

        return mask, score_idx, cal_idx, q

    def _dice_loss(self, p_fine, p_gate, fine_t, gate_t):
        """Soft-Dice over the change class, or macro over the coarse3 classes.

        The hybrid half of Zhang et al.'s "deep supervision and hybrid loss":
        focal is a per-sample objective and cannot see the *set* overlap the
        model is scored on, while Dice on the change class is exactly a
        differentiable change-F1 -- ``2|P n T| / (|P| + |T|)`` with soft
        memberships. Taken over the whole batch, which under this project's
        full-batch default is the whole training fold, so the statistic is the
        one the metric computes rather than a minibatch estimate of it.

        ``dice_level='fine'`` averages the per-class Dice over all nine coarse3
        classes UNWEIGHTED, which makes it the relaxation of ``focus_macro_f1``:
        the 46-plot transition contributes as much as the 4,200-plot stable one,
        and unlike focal loss that weighting is on the overlap rather than on
        the per-sample gradient.
        """
        import torch

        eps = 1.0
        if self.dice_level == "gate":
            if self._stable_gate_idx < 0:
                return p_gate.new_tensor(0.0)      # single-sided fold, no pairs
            change_idx = 1 - self._stable_gate_idx
            p = p_gate[:, change_idx]
            y = (gate_t != self._stable_gate_idx).to(p.dtype)
            return 1.0 - (2.0 * (p * y).sum() + eps) / (p.sum() + y.sum() + eps)
        onehot = torch.zeros_like(p_fine)
        onehot.scatter_(1, fine_t[:, None], 1.0)
        inter = (p_fine * onehot).sum(0)
        denom = p_fine.sum(0) + onehot.sum(0)
        return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()

    def _siam_stages(self):
        """Per-stage ``[(h18, h24), ...]`` from the shared encoder's last pass.

        Same lookup as ``_siam_pair``: the encoder sits at the trunk for
        ``arch='siamese'`` and inside the AlphaEarth tower for the two-tower.
        """
        for module in (self.trunk, getattr(self.trunk, "aef_tower", None)):
            if module is not None and getattr(module, "last_h", None) is not None:
                return module.last_h
        raise RuntimeError(
            "deep_sup_weight > 0 but no shared encoder produced intermediates "
            "-- arch must be 'siamese', or 'two_tower' with aef_siam=True")

    def _deep_sup_loss(self, fine_t, merged_t, gate_t,
                       w_fine, w_merged, w_gate, wf, wm, wg):
        """The three-level loss, repeated at every hidden stage of the encoder.

        Each stage's pair is combined the way the head combines the final one --
        ``[h18, h24, h24-h18, |h24-h18|]`` -- and read by its own linear head
        into coarse3 logits, whose softmax is then group-summed to merged2 and
        gate by the same fixed matrices. So the auxiliary signal is not a
        different objective at a shallower depth; it is the *same* objective,
        which is what deep supervision means and what makes the weight
        interpretable against the main term.
        """
        import torch

        total = None
        for head, (h18, h24) in zip(self.deep_heads, self._siam_stages()):
            d = h24 - h18
            p = torch.softmax(head(torch.cat([h18, h24, d, d.abs()], dim=1)), dim=1)
            term = self._levels(p, fine_t, merged_t, gate_t, w_fine, w_merged,
                                w_gate, wf, wm, wg, extras=False)
            total = term if total is None else total + term
        return total / max(len(self.deep_heads), 1)

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

    def fit(self, frame, y, unlabelled_frame=None, soft_merged=None,
            state_frame=None):
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
            # The endpoint heads classify *state*, not transition, so they sit
            # with the coarse levels under cb_levels='fine'. Inert on the
            # siamese recipes, which carry endpoint_weight=0.
            if self.cb_levels != "all":
                state_mode = "none"
            self._w_from = torch.tensor(
                class_weights(self.merged_from_idx_[merged_t], n_state, state_mode),
                device=self.device)
            self._w_to = torch.tensor(
                class_weights(self.merged_to_idx_[merged_t], n_state, state_mode),
                device=self.device)

        self._build_network(Xs.shape, n_fine, self.seed)
        self._from_idx = torch.tensor(self.from_idx_, device=self.device)
        self._to_idx = torch.tensor(self.to_idx_, device=self.device)

        # Deep supervision: one auxiliary coarse3 head per hidden stage of the
        # shared encoder, reading that stage's pair the way the real head reads
        # the final one. Built here rather than in _build_head because they are
        # not a head choice -- they ride whatever head is configured.
        self.deep_heads = None
        if self.deep_sup_weight > 0:
            enc = self._siam_encoder()
            self.deep_heads = nn.ModuleList(
                [nn.Linear(4 * dim, n_fine) for dim in enc.stage_dims]
            ).to(self.device)
            self._modules_ = self._modules_ + (self.deep_heads,)

        # Auxiliary single-date state head on the shared encoder embedding. One
        # linear layer, discarded at predict time -- it exists to put gradient
        # into f, never to be read. Serving cost is therefore unchanged, which is
        # the property the whole s2off line is built around.
        if self.head not in ("endpoint", "endpoint_pure"):
            self.state_head = None
        self._X_state = self._y_state = None
        if self.siam_state_weight > 0 or self.siam_state_pretrain > 0:
            if (self.siam_state_source in ("endogenous", "both")
                    and self.siam_unlabelled_weight > 0):
                raise ValueError(
                    "the endogenous state term reads last_z18/last_z24, which "
                    "the unlabelled Barlow pass overwrites -- combining them "
                    "would silently supervise the pool's embeddings with the "
                    "plots' labels. Use siam_state_source='external'.")
            if self.state_head is None:
                self.state_head = nn.Linear(
                    self.siam_dim, len(self.base_classes_)).to(self.device)
                self._modules_ = self._modules_ + (self.state_head,)
            # With an endpoint head the state head is ALREADY the output
            # parameterisation, so the auxiliary term supervises the same
            # parameters directly rather than a discarded copy -- which is
            # exactly what N14b was missing when it threw the head away.
            if self.siam_state_source in ("external", "both"):
                if state_frame is None:
                    raise ValueError(
                        "siam_state_source includes 'external' but fit() got no "
                        "state_frame")
                self._X_state, self._y_state = self._prepare_state_pool(state_frame)

        # Forward-correction matrix on the fine level (identity if noise_rate=0).
        self._T = (torch.tensor(self._noise_matrix(), device=self.device)
                   if self.noise_rate > 0 else None)

        weight_mode = {"ce": "none", "weighted_ce": "inverse",
                       "focal": "none", "cb_focal": "effective"}[self.loss]
        # `cb_levels='fine'` keeps the class weights off the gate and merged2
        # levels -- see the constructor.
        coarse_mode = weight_mode if self.cb_levels == "all" else "none"
        w_fine = torch.tensor(class_weights(fine_t, n_fine, weight_mode),
                              device=self.device)
        w_merged = torch.tensor(class_weights(merged_t, n_merged, coarse_mode),
                                device=self.device)
        w_gate = torch.tensor(class_weights(gate_t, n_gate, coarse_mode),
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

        # Unlabelled endpoint pool for the siamese Barlow term. Prepared with the
        # LABELLED fold's standardisation (fit=False) -- the pool must land in the
        # same space as the training rows or the shared encoder sees two
        # distributions and the cross-correlation is measuring the shift between
        # them rather than year-invariance. Every pooled pixel is *assumed*
        # stable, which is wrong on the AOI's change fraction (~0.5%); see
        # build_unlabelled_aef.py for why that is the whole approximation.
        self._Xu_siam = None
        if (unlabelled_frame is not None and self.arch == "siamese"
                and self.siam_barlow_weight > 0 and self.siam_unlabelled_weight > 0):
            self._Xu_siam = torch.tensor(
                self._prepare(unlabelled_frame, fit=False), device=self.device)

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
        params.extend(self._loose_params())
        optimiser = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        if self.sampler == "gh":
            steps_per_epoch = gh_n_batches
        else:
            steps_per_epoch = (1 if self.batch_size is None
                               else max(1, int(np.ceil(len(tr_idx) / self.batch_size))))
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2e-3, total_steps=self.epochs * steps_per_epoch
        )

        # Early-learning regularisation keeps one EMA row per TRAINING row, so it
        # is indexed by position in this fold's frame and dies with the fold.
        self._elr_target = None
        if self.elr_weight > 0:
            self._elr_target = torch.zeros(len(fine_t), n_fine, device=self.device)

        # The co-teaching partner, built last so the shallow copy carries every
        # piece of fit-time state assigned above (standardisation, teacher
        # posteriors, unlabelled pool). It is trained, read for its selections,
        # and then discarded -- `self` alone is what predict() serves.
        peer = peer_optimiser = peer_schedule = None
        if self.coteach != "off":
            if self.elr_weight > 0:
                raise ValueError(
                    "co-teaching and ELR would share one _elr_target tensor "
                    "through the shallow copy, so the peer's beliefs would be "
                    "written into the EMA the first network is held to. Run them "
                    "as separate arms.")
            if self.crt_epochs > 0 or self.sampler == "gh":
                raise ValueError(
                    "co-teaching selects rows per step; cRT retraining and the "
                    "G-H sampler both re-choose the rows themselves.")
            peer = self._make_peer(Xs.shape, n_fine)
            peer_params = [p for m in peer._modules_ for p in m.parameters()]
            peer_params.extend(peer._loose_params())
            peer_optimiser = torch.optim.AdamW(peer_params, lr=2e-3,
                                               weight_decay=1e-2)
            peer_schedule = torch.optim.lr_scheduler.OneCycleLR(
                peer_optimiser, max_lr=2e-3,
                total_steps=self.epochs * steps_per_epoch)

        def batch_loss(idx, net=None):
            """Summed 3-level loss over training rows ``idx`` (mixup if enabled).

            ``net`` selects *which* network's parameters the loss is built from
            and defaults to this one. Co-teaching passes the peer here so both
            networks are trained under a single definition of the objective --
            every auxiliary term, the Sentinel-2 gate, the cosine loss and the
            standardisation included -- rather than a second, drifting copy of
            it. The two differ only in their weights and in ``idx``.
            """
            net = self if net is None else net
            it = torch.as_tensor(np.asarray(idx), device=self.device, dtype=torch.long)
            fb, mb, gb = fine_tt[it], merged_tt[it], gate_tt[it]
            Xb = Xt[it]
            if self.mixup_alpha > 0 and len(idx) > 1:
                lam = float(rng.beta(self.mixup_alpha, self.mixup_alpha))
                order = torch.randperm(len(idx), device=self.device)
                Xb = lam * Xb + (1.0 - lam) * Xb[order]
                rep = net._inject(net._encode(net._inject(Xb, "input")), "rep")
                p_fine = torch.softmax(net._fine_logits(rep), dim=1)
                return (lam * net._levels(p_fine, fb, mb, gb, w_fine, w_merged,
                                          w_gate, wf, wm, wg)
                        + (1.0 - lam) * net._levels(p_fine, fb[order], mb[order],
                                                    gb[order], w_fine, w_merged,
                                                    w_gate, wf, wm, wg))
            rep = net._inject(net._encode(net._inject(Xb, "input")), "rep")
            p_fine = torch.softmax(net._fine_logits(rep), dim=1)
            loss = net._levels(p_fine, fb, mb, gb, w_fine, w_merged, w_gate,
                               wf, wm, wg)
            if self.elr_weight > 0:
                loss = loss + self.elr_weight * net._elr_loss(p_fine, it)
            if self.deep_sup_weight > 0:
                # Reads the intermediates from the forward pass just above, so it
                # must come before any auxiliary term that runs the encoder again
                # (the unlabelled Barlow pass and the external state pass both
                # overwrite them).
                loss = loss + self.deep_sup_weight * self._deep_sup_loss(
                    fb, mb, gb, w_fine, w_merged, w_gate, wf, wm, wg)
            if self.align_weight > 0 and self.arch == "two_tower":
                loss = loss + self.align_weight * net._align_loss()
            if (self.arch in ("siamese", "two_tower")
                    and (self.siam_cos_weight > 0 or self.siam_barlow_weight > 0
                         or self.siam_mssm_weight > 0)):
                # gb is the gate target for these rows: index of 'stable' in the
                # sorted gate class list, so 'change' < 'stable' and stable == 1.
                stable = gb == self._stable_gate_idx
                if self.siam_mssm_weight > 0:
                    # Reads the per-stage intermediates from the forward pass
                    # above, so it goes before the unlabelled/state passes that
                    # overwrite them -- same constraint as _deep_sup_loss.
                    loss = loss + self.siam_mssm_weight * net._siam_mssm_loss(stable)
                if self.siam_cos_weight > 0:
                    loss = loss + self.siam_cos_weight * net._siam_cos_loss(stable)
                if self.siam_barlow_weight > 0:
                    loss = loss + self.siam_barlow_weight * net._siam_barlow_loss(stable)
                if self._Xu_siam is not None:
                    # Same term on a minibatch of unlabelled pixel pairs. Drawn
                    # fresh each step so the 30 epochs see far more of the pool
                    # than one pass would, and run through the trunk in its own
                    # forward pass -- last_z18/last_z24 are overwritten, so this
                    # must come AFTER the labelled Barlow term above reads them.
                    #
                    # The BatchNorm running statistics are FROZEN for this pass.
                    # Without that freeze the extra forward pass folds the pool's
                    # distribution into the running mean/var that eval() then uses
                    # to normalise labelled test plots -- and the pool is one
                    # city's pixels while the labelled plots are spread across
                    # the sample, so the two are not the same distribution.
                    # Measured: running |mean| 0.058 -> 0.189 and running var
                    # 0.368 -> 0.270, which cost -0.053 change-F1 and collapsed
                    # stable-Artificial recall from 0.638 to 0.443. The gradient
                    # signal from the pool is wanted; its batch statistics are
                    # not.
                    sub = torch.randint(len(self._Xu_siam),
                                        (min(self._siam_unlabelled_batch,
                                             len(self._Xu_siam)),),
                                        device=self.device)
                    with net._frozen_bn_stats():
                        net._encode(self._Xu_siam[sub])
                    all_stable = torch.ones(len(sub), dtype=torch.bool,
                                            device=self.device)
                    loss = loss + self.siam_unlabelled_weight * \
                        net._siam_barlow_loss(all_stable)
            if self.siam_state_weight > 0:
                # Must come after the cosine/Barlow terms above: the external
                # pass overwrites last_z18/last_z24, and those read the pair.
                loss = loss + self.siam_state_weight * net._siam_state_loss(
                    self._from_idx[fb], self._to_idx[fb])
            if self._soft is not None:
                # Soft cross-entropy at the merged2 level, averaged over the rows
                # the teacher actually scored.
                p_merged = p_fine @ self._M
                ce = -(self._soft[it] * torch.log(p_merged.clamp_min(1e-8))).sum(1)
                keep = self._soft_mask[it]
                loss = loss + self.distill_weight * (
                    (ce * keep).sum() / keep.sum().clamp_min(1.0))
            return loss

        # Section T bookkeeping: how often each training row survived selection
        # for the served network, and how often the stochastic guard had to fire.
        keep_counts = np.zeros(len(fine_t), dtype="int64")
        keep_tally = [0.0, 0, 0]                  # sum(rate), steps, guard hits

        def coteach_step(chunk, epoch):
            """One co-teaching step: each network trains on its partner's picks."""
            it = torch.as_tensor(np.asarray(chunk), device=self.device,
                                 dtype=torch.long)
            fb, mb, gb = fine_tt[it], merged_tt[it], gate_tt[it]
            picks = {}
            for tag, net in (("A", self), ("B", peer)):
                # Selection reads the network in eval mode. Under this project's
                # recipe the training forward drops a whole modality half the
                # time (modality_dropout=0.5) and 70% of the detail tower's
                # units, so a train-mode posterior is a draw from a wide
                # distribution rather than the network's belief about the row --
                # and the whole method is that belief.
                for module in net._modules_:
                    module.eval()
                with torch.no_grad():
                    p_fine = torch.softmax(net._fine_logits(net._encode(Xt[it])), 1)
                    p_merged = p_fine @ self._M
                    if self.coteach_level == "fine":
                        p_true = p_fine.gather(1, fb[:, None]).squeeze(1)
                    elif self.coteach_level == "merged":
                        p_true = p_merged.gather(1, mb[:, None]).squeeze(1)
                    else:
                        p_true = (p_merged @ self._G).gather(1, gb[:, None]).squeeze(1)
                    per_loss = net._levels(p_fine, fb, mb, gb, w_fine, w_merged,
                                           w_gate, wf, wm, wg, per_sample=True)
                keep, guard = net._coteach_keep(p_true, per_loss, epoch, rng,
                                                groups=fb)
                picks[tag] = keep
                keep_tally[2] += int(guard)
                for module in net._modules_:
                    module.train()
            # The cross: A trains on what B kept, B on what A kept. Training each
            # on its own picks is self-paced learning, a different (and weaker)
            # method -- the error a network makes is invisible to itself.
            for net, opt, sch, keep in ((self, optimiser, schedule, picks["B"]),
                                        (peer, peer_optimiser, peer_schedule,
                                         picks["A"])):
                kept = it[keep]
                if len(kept) == 0:
                    sch.step()
                    continue
                opt.zero_grad()
                loss = batch_loss(kept.detach().cpu().numpy(), net=net)
                loss.backward()
                opt.step()
                sch.step()
            served = picks["B"].detach().cpu().numpy()
            keep_counts[np.asarray(chunk)[served]] += 1
            keep_tally[0] += float(served.mean())
            keep_tally[1] += 1

        def snapshot():
            state = [{k: v.detach().clone() for k, v in m.state_dict().items()}
                     for m in self._modules_]
            return (state, [p.detach().clone() for p in self._loose_params()])

        def restore(snap):
            state, loose = snap
            for module, st in zip(self._modules_, state):
                module.load_state_dict(st)
            with torch.no_grad():
                for param, saved in zip(self._loose_params(), loose):
                    param.copy_(saved)

        # Interleaved-noise state: _noise_scale co-scales the injected std with the
        # running gradient norm when noise_gradscale is set (1.0 = off/unscaled).
        self._noise_scale = 1.0
        self._noise_factor_cur = 0.0
        grad_ema = [None]  # running gradient-norm EMA; ref = first observed value
        grad_ref = [None]

        if self.siam_state_pretrain > 0:
            self._pretrain_state(Xs, fine_t, tr_idx)

        best_f1, best_snap, since_best = -1.0, None, 0
        for _epoch in range(self.epochs):
            self._epoch = _epoch
            for module in self._modules_:
                module.train()
            self._noise_factor_cur = self._noise_factor(_epoch)
            if peer is not None:
                # The peer is a distinct object, so the per-epoch state the loss
                # terms read off `self` has to be mirrored onto it.
                peer._epoch = _epoch
                peer._noise_factor_cur = self._noise_factor_cur
                peer._noise_scale = self._noise_scale
                for module in peer._modules_:
                    module.train()
            if self.sampler == "gh":
                chunks = gh_batches()
            else:
                order = (tr_idx if self.batch_size is None
                         else tr_idx[rng.permutation(len(tr_idx))])
                chunks = ([order] if self.batch_size is None
                          else [order[i:i + self.batch_size]
                                for i in range(0, len(order), self.batch_size)])
            for chunk in chunks:
                if peer is not None:
                    coteach_step(chunk, _epoch)
                    continue
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
        if peer is not None:
            # Keep what the selection did, not the network that did it. The
            # counts are per training row of THIS fold; the caller maps them
            # back to plots (see twotower_lab.coteach_keep_table).
            steps = max(keep_tally[1], 1)
            self.coteach_keep_rate_ = keep_tally[0] / steps
            self.coteach_guard_rate_ = keep_tally[2] / (2 * steps)
            self.coteach_keep_counts_ = keep_counts
            # Rows actually offered to the selector: with early stopping on, the
            # held-out split never enters a chunk, and a zero count there means
            # "never seen", not "always rejected".
            self.coteach_rows_ = tr_idx
            self.coteach_steps_ = steps
            del peer
        self._elr_target = None
        if self.crt_epochs > 0:
            self._retrain_classifier(Xt, tr_idx, fine_tt, merged_tt, gate_tt,
                                     w_fine, w_merged, w_gate, wf, wm, wg, rng)
        return self

    def _retrain_classifier(self, Xt, tr_idx, fine_tt, merged_tt, gate_tt,
                            w_fine, w_merged, w_gate, wf, wm, wg, rng):
        """cRT: freeze the representation, retrain only the head, class-balanced.

        Kang et al. (2020) decouple long-tailed recognition into representation
        learning and classifier learning, and their central finding is that
        class-balanced *sampling* damages the representation while helping the
        classifier. That is the reading of this project's own G-H sampling
        negative (`gh-sampler-does-not-help`): the balanced sampler was applied
        during joint training, which is precisely the configuration Kang et al.
        report as the losing one. Decoupling is the version that has not run.

        The trunk goes to ``eval()`` and its gradients off, so BatchNorm uses its
        running statistics and the representation is frozen *exactly* as it will
        be at predict time -- retraining a head against train-mode batch
        statistics would tune it for a representation the model never serves.
        The representation is therefore constant, so it is computed once and the
        head is retrained on the cached matrix: the whole pass costs a few
        seconds regardless of ``crt_epochs``.

        Resampling is on the FINE target, since that is where the imbalance the
        head is meant to see actually lives (46 plots against 4,200).
        """
        import numpy as np
        import torch

        for module in self._modules_:
            module.eval()
        trunk = self.gru if self.arch == "gru" else self.trunk
        head_modules = tuple(m for m in self._modules_ if m is not trunk)
        # The endpoint head reads (z18, z24) off the trunk's last forward pass,
        # and this loop never runs the trunk again -- so the pair is cached with
        # the representation and replayed through _crt_pair. Without that the
        # head would retrain against whichever chunk happened to be last, which
        # is wrong silently rather than loudly.
        reps, pairs = [], []
        with torch.no_grad():
            for i in range(0, len(tr_idx), 4096):
                it = torch.as_tensor(tr_idx[i:i + 4096], device=self.device,
                                     dtype=torch.long)
                reps.append(self._encode(Xt[it]))
                if self.head in ("endpoint", "endpoint_pure"):
                    pairs.append(tuple(z.detach() for z in self._siam_pair()))
        rep = torch.cat(reps)
        pair = ((torch.cat([p[0] for p in pairs]),
                 torch.cat([p[1] for p in pairs])) if pairs else None)
        for module in head_modules:
            module.train()
        params = [p for m in head_modules for p in m.parameters()]
        params.extend(self._loose_params())
        optimiser = torch.optim.AdamW(params, lr=self.crt_lr, weight_decay=1e-2)

        fine_local = fine_tt[torch.as_tensor(tr_idx, device=self.device,
                                             dtype=torch.long)]
        merged_local = merged_tt[torch.as_tensor(tr_idx, device=self.device,
                                                 dtype=torch.long)]
        gate_local = gate_tt[torch.as_tensor(tr_idx, device=self.device,
                                             dtype=torch.long)]
        # Class-balanced draw: every fine class contributes the same number of
        # rows per epoch, so the 46-plot transition is seen as often as the
        # 4,200-plot stable one. Sampled with replacement, which is what makes
        # the rare classes reachable at all.
        pools = [np.flatnonzero(fine_local.cpu().numpy() == k)
                 for k in range(len(self.fine_classes_))]
        pools = [p for p in pools if len(p)]
        per_class = max(1, len(tr_idx) // max(len(pools), 1))
        for _ in range(self.crt_epochs):
            draw = np.concatenate([p[rng.integers(len(p), size=per_class)]
                                   for p in pools])
            it = torch.as_tensor(rng.permutation(draw), device=self.device,
                                 dtype=torch.long)
            optimiser.zero_grad()
            self._crt_pair = (pair[0][it], pair[1][it]) if pair else None
            p_fine = torch.softmax(self._fine_logits(rep[it]), dim=1)
            loss = self._levels(p_fine, fine_local[it], merged_local[it],
                                gate_local[it], w_fine, w_merged, w_gate,
                                wf, wm, wg)
            loss.backward()
            optimiser.step()
        self._crt_pair = None
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

    def _aef_siam_permutation(self):
        """Indices viewing ``aef_columns`` as ``[all _2018 | all _2024 | rest]``.

        Derived from the column *names*, and the two year blocks are sorted by
        the same band stem, so ``A07_2018`` and ``A07_2024`` land at the same
        offset in their blocks whatever order the caller supplied. That is the
        one property the shared encoder needs: position *i* must be the same
        measurement at both dates.

        Raises rather than truncating when the endpoints do not pair up 1:1 --
        a mismatch would silently hand the encoder two different feature sets
        and train something that looks fine and means nothing.
        """
        cols = self.aef_columns
        stem = lambda i: cols[i].rsplit("_", 1)[0]  # noqa: E731
        i18 = sorted((i for i, c in enumerate(cols) if c.endswith("_2018")), key=stem)
        i24 = sorted((i for i, c in enumerate(cols) if c.endswith("_2024")), key=stem)
        rest = [i for i, c in enumerate(cols)
                if not c.endswith(("_2018", "_2024"))]
        if not i18 or [stem(i) for i in i18] != [stem(i) for i in i24]:
            raise ValueError(
                "aef_siam needs matching _2018/_2024 columns for every band; "
                f"got {len(i18)} and {len(i24)}")
        return i18, i24, rest

    def _siam_pair(self):
        """The last endpoint embedding pair, wherever the siamese encoder sits.

        `arch='siamese'` puts it at the trunk; `arch='two_tower'` with
        `aef_siam` puts it inside the AlphaEarth tower. The auxiliary
        losses are the same either way, so they look the pair up rather than
        assuming a location -- and a missing pair is a configuration error worth
        raising on, not a zero worth silently adding.
        """
        if getattr(self, "_crt_pair", None) is not None:
            return self._crt_pair          # replayed cache, see _retrain_classifier
        for module in (self.trunk, getattr(self.trunk, "aef_tower", None)):
            if module is not None and hasattr(module, "last_z18"):
                return module.last_z18, module.last_z24
        raise RuntimeError(
            "siamese auxiliary loss requested but no endpoint pair was produced "
            "-- arch must be 'siamese', or 'two_tower' with aef_siam=True")

    def _siam_encoder(self):
        """The module holding the shared encoder, wherever it sits.

        Same lookup as ``_siam_pair``: ``arch='siamese'`` puts it at the trunk,
        ``arch='two_tower'`` with ``aef_siam`` nests it in the AlphaEarth tower.
        """
        for module in (self.trunk, getattr(self.trunk, "aef_tower", None)):
            if module is not None and hasattr(module, "encode_single"):
                return module
        raise RuntimeError(
            "siam_state_weight > 0 but no shared encoder was built -- arch must "
            "be 'siamese', or 'two_tower' with aef_siam=True")

    def _state_endpoint_columns(self) -> tuple[list, object, object]:
        """The 2018 endpoint columns and the statistics that standardise them.

        The external pool must land in exactly the space the paired path puts
        the 2018 block in, or the shared encoder is handed two distributions and
        the state head learns the shift between them instead of land cover. The
        two arches reach the same block by different routes, so the statistics
        are looked up per arch rather than assumed.
        """
        if self.arch == "siamese":
            return list(self.siam_columns_18), self.mu_end, self.sd_end
        if self.arch == "two_tower" and self.aef_siam:
            i18, _, _ = self._aef_siam_permutation()
            cols = [self.aef_columns[i] for i in i18]
            return cols, self.mu_a[i18], self.sd_a[i18]
        raise RuntimeError(
            f"no single-date endpoint block for arch={self.arch!r} "
            f"(aef_siam={self.aef_siam})")

    def _state_endpoint_slices(self) -> tuple[object, object]:
        """Column indices into the prepared matrix for the 2018 and 2024 blocks.

        The index-level twin of ``_state_endpoint_columns``, for the *endogenous*
        pool -- which is not a frame to be looked up by name but the fold's own
        already-standardised ``Xs``, sliced into its two dates.

        The two arches pack that matrix differently and the difference is not
        cosmetic. ``siamese`` packs ``[x18 | x24 | extra]``, so the dates are
        contiguous halves. ``two_tower`` packs ``[aef | detail | masks]`` with
        the AlphaEarth block in the caller's sorted column order, and the shared
        encoder gathers the year blocks *inside* the tower -- so the 2018 columns
        are scattered through the first block and `siam_columns_18` is None.
        Slicing ``[:d_end]`` there, as this used to, either raises (it did) or
        would silently hand the encoder half of 2018 and half of 2024.
        """
        if self.arch == "siamese":
            d_end = len(self.siam_columns_18)
            return slice(0, d_end), slice(d_end, 2 * d_end)
        if self.arch == "two_tower" and self.aef_siam:
            # Indices into `aef_columns`, which `_prepare` writes as the leading
            # block of Xs, so they are indices into Xs unchanged.
            i18, i24, _ = self._aef_siam_permutation()
            return np.asarray(i18), np.asarray(i24)
        raise RuntimeError(
            f"no single-date endpoint block for arch={self.arch!r} "
            f"(aef_siam={self.aef_siam})")

    def _prepare_state_pool(self, state_frame):
        """(X, y) for the external pool, standardised like the 2018 block.

        Raises on a state outside the model's own endpoint vocabulary rather
        than dropping it: a pool whose legend does not match is the failure this
        whole line of work is gated on (N14), and it must not pass silently.
        """
        import numpy as np
        import torch

        cols, mu, sd = self._state_endpoint_columns()
        missing = [c for c in cols if c not in state_frame.columns]
        if missing:
            raise ValueError(
                f"state pool is missing {len(missing)} endpoint columns, e.g. "
                f"{missing[:3]} -- it must carry the same 2018 block the model "
                "is trained on")
        X = state_frame[cols].to_numpy("float32")
        Z = (X - mu) / sd
        Z = np.where(np.isfinite(Z), Z, 0.0).astype("float32")

        vocab = {c.lower(): i for i, c in enumerate(self.base_classes_)}
        raw = state_frame["state"].astype(str).str.strip().str.lower()
        unknown = sorted(set(raw) - set(vocab))
        if unknown:
            raise ValueError(
                f"state pool has states outside the model's endpoint classes "
                f"{self.base_classes_}: {unknown}")
        y = raw.map(vocab).to_numpy("int64")
        return (torch.tensor(Z, device=self.device),
                torch.tensor(y, device=self.device))

    def _pretrain_state(self, Xs, fine_t, tr_idx):
        """Train the shared encoder on single-date states, before the main loop.

        The same objective ``_siam_state_loss`` adds as a weighted term, run
        instead as its own phase: ``siam_state_pretrain`` epochs of
        ``g(f(x)) -> state`` over the pool, updating only the encoder and the
        state head, after which ``fit`` proceeds normally from those weights.
        Sections N14 and P varied *where* the state path lands and got the same
        flat answer three times; this varies *when*, which is the one axis they
        share.

        Three properties are deliberate.

        * **The endogenous arm reads the standardised endpoint blocks directly**
          -- ``Xs[:, :d]`` is the 2018 block and ``Xs[:, d:2d]`` the 2024 block,
          both already pooled-standardised by ``_prepare`` -- rather than the
          forward pass's ``last_z18``/``last_z24`` the joint term uses. There is
          no forward pass to read yet, and a ``From -> To`` label is two free
          state labels, so this is the no-new-data control.
        * **Only training rows enter**, via ``tr_idx``. With early stopping on,
          the held-out split picks the stopping epoch, and pretraining on it
          would leak that choice into the representation.
        * **BatchNorm running statistics are frozen**, for the reason N4
          recorded: the pool is not the labelled plots' distribution, and
          letting it write the running mean/var that ``eval()`` later uses on
          test plots cost -0.053 change-F1 there. The gradient is wanted; the
          batch statistics are not. The 30 main epochs would partly overwrite
          them anyway, which is exactly what makes an unfrozen pass a silent,
          seed-dependent contaminant rather than an honest one.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        # Its own stream, not the caller's: the two arms hold different numbers
        # of pool rows, so drawing the pretrain shuffles from `fit`'s rng would
        # leave the main loop's minibatch order at a different point in the same
        # stream for each arm, and the comparison would carry that as well as the
        # pool.
        pre_rng = np.random.default_rng(self.seed + 1_000_003)

        blocks = []                      # (X, year_tag, y)
        if self.siam_state_source in ("endogenous", "both"):
            # `tr_idx` is this fold's training rows and nothing else, which is
            # what makes the endogenous pool leak-free: the labels are the two
            # halves of the transition target the fit is already supervised on,
            # never a held-out plot's. See tests/test_state_pool_leak.py for the
            # sibling guarantee on the *external* file path, which needs the
            # block filter to get there.
            s18, s24 = self._state_endpoint_slices()
            X = torch.tensor(Xs[tr_idx], device=self.device)
            y_fine = fine_t[tr_idx]
            blocks.append((X[:, s18], "2018",
                           torch.tensor(self.from_idx_[y_fine],
                                        device=self.device, dtype=torch.long)))
            blocks.append((X[:, s24], "2024",
                           torch.tensor(self.to_idx_[y_fine],
                                        device=self.device, dtype=torch.long)))
        if self.siam_state_source in ("external", "both"):
            blocks.append((self._X_state, "2018", self._y_state))

        # Class weights over the *concatenated* pool, so a 'both' run balances
        # the states it actually sees rather than each source separately.
        class_weight = None
        if self.siam_state_class_weight == "balanced":
            counts = torch.zeros(len(self.base_classes_), device=self.device)
            for _, _, y in blocks:
                counts += torch.bincount(y, minlength=len(self.base_classes_))
            # N / (C * n_c), the sklearn convention `statepre` also uses. Mean
            # weight over the pool is exactly 1, so the loss scale is unchanged.
            present = counts > 0
            class_weight = torch.zeros_like(counts)
            class_weight[present] = (counts.sum()
                                     / (present.sum() * counts[present]))

        n = sum(len(X) for X, _, _ in blocks)
        batch = max(1, min(self._siam_state_batch, n))
        steps_per_epoch = max(1, int(np.ceil(n / batch)))
        encoder = self._siam_encoder()
        params = list(encoder.parameters()) + list(self.state_head.parameters())
        optimiser = torch.optim.AdamW(params, lr=2e-3, weight_decay=1e-2)
        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=2e-3,
            total_steps=self.siam_state_pretrain * steps_per_epoch)

        # One flat index over the concatenated blocks, so a 'both' run shuffles
        # the two sources together instead of alternating pure-source steps.
        bounds = np.cumsum([0] + [len(X) for X, _, _ in blocks])
        for module in self._modules_:
            module.train()
        for _ in range(self.siam_state_pretrain):
            order = pre_rng.permutation(n)
            for start in range(0, n, batch):
                chunk = order[start:start + batch]
                optimiser.zero_grad()
                loss, rows = 0.0, 0
                for b, (X, year, y) in enumerate(blocks):
                    take = chunk[(chunk >= bounds[b]) & (chunk < bounds[b + 1])]
                    # A single row cannot go through BatchNorm in train mode, and
                    # each block can leave at most one such remainder in the last
                    # chunk of an epoch. Dropped rather than padded or run in
                    # eval mode: one row of 13,118 per epoch is not worth either
                    # a duplicate gradient or a second normalisation regime.
                    if len(take) < 2:
                        continue
                    sub = torch.tensor(take - bounds[b], device=self.device,
                                       dtype=torch.long)
                    with self._frozen_bn_stats():
                        z = encoder.encode_single(X[sub], year)
                    loss = loss + F.cross_entropy(
                        self.state_head(z), y[sub], reduction="sum",
                        weight=class_weight)
                    # Normalise by the weight actually applied, not the row
                    # count, so the batch loss stays a weighted *mean* and the
                    # reweighting cannot change the effective learning rate as a
                    # side effect. With no weighting w == 1 and this is the row
                    # count, i.e. byte-identical to the unweighted path.
                    rows += (len(take) if class_weight is None
                             else float(class_weight[y[sub]].sum()))
                if not rows:
                    continue
                (loss / rows).backward()
                optimiser.step()
                schedule.step()
        self.state_pretrain_loss_ = float(loss / max(rows, 1))

    def _siam_state_loss(self, from_b, to_b):
        """Cross-entropy of the state head on single-date encoder embeddings.

        The endogenous half reads the training rows' own endpoints -- ``z18``
        should say ``From`` and ``z24`` should say ``To`` -- and reuses the pair
        the forward pass already produced, so it costs one linear layer.

        The external half draws a fresh minibatch from the pool each step and
        runs its own forward pass. **BatchNorm running statistics are frozen for
        that pass** (N4's silent defect: the extra pass folded an out-of-
        distribution pool into the running mean/var that ``eval()`` then used on
        test plots, costing -0.053 change-F1 and collapsing stable-Artificial
        recall from 0.638 to 0.443). The gradient from the pool is wanted; its
        batch statistics are not.
        """
        import torch
        import torch.nn.functional as F

        terms = []
        if self.siam_state_source in ("endogenous", "both"):
            z18, z24 = self._siam_pair()
            terms.append(0.5 * (F.cross_entropy(self.state_head(z18), from_b)
                                + F.cross_entropy(self.state_head(z24), to_b)))
        if self.siam_state_source in ("external", "both"):
            if self._X_state is None:
                raise RuntimeError(
                    "siam_state_source includes 'external' but no state pool was "
                    "passed to fit(state_frame=...)")
            sub = torch.randint(len(self._X_state),
                                (min(self._siam_state_batch, len(self._X_state)),),
                                device=self.device)
            encoder = self._siam_encoder()
            with self._frozen_bn_stats():
                z = encoder.encode_single(self._X_state[sub], "2018")
            terms.append(F.cross_entropy(self.state_head(z), self._y_state[sub]))
        return sum(terms) / max(len(terms), 1)

    @contextlib.contextmanager
    def _frozen_bn_stats(self):
        """Forward through the trunk without updating BatchNorm running stats.

        ``momentum = 0`` leaves ``running_mean``/``running_var`` untouched
        (PyTorch updates them as ``(1 - m) * running + m * batch``) while the
        layer still normalises by the *batch* statistics, so the pass is
        numerically what training would do and only the eval-time state is
        protected. Restores whatever momentum each module had, so a module
        configured differently is not silently rewritten.
        """
        import torch.nn as nn

        saved = [(m, m.momentum) for m in self.trunk.modules()
                 if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        for module, _ in saved:
            module.momentum = 0.0
        try:
            yield
        finally:
            for module, momentum in saved:
                module.momentum = momentum

    def _siam_cos_loss(self, stable):
        """Gate-supervised cosine between the two endpoint embeddings.

        A stable plot is one piece of ground measured twice, so its 2018 and
        2024 embeddings should point the same way; a change plot's should not.
        This states that directly on the representation instead of hoping the
        classifier head induces it, which is the whole argument for the siamese
        layout -- and it is supervision the *stable majority* carries, so it
        costs nothing on the rare transitions that are the actual target.

        The two group terms are averaged with equal weight, not pooled: stable
        plots outnumber change plots roughly 4:1 here and a pooled mean would be
        almost entirely the stable term, i.e. a plain "make everything similar"
        regulariser. ``siam_cos_margin`` is the cosine a change pair may keep
        before it is penalised -- 0 pushes change pairs to orthogonal, which is
        stronger than the task needs (a Nature -> Cropland plot is not the
        opposite of itself), so a positive margin is the sane setting.
        """
        import torch
        import torch.nn.functional as F

        z18, z24 = self._siam_pair()
        return self._pair_margin_term(
            z18, z24, stable, metric="cos", margin=self.siam_cos_margin,
            stable_margin=self.siam_cos_stable_margin)

    def _pair_margin_term(self, a, b, stable, *, metric, margin, stable_margin):
        """One scale's contrastive term: pull stable pairs together, push change apart.

        Factored out of ``_siam_cos_loss`` so the multi-scale term applies the
        *identical* objective at the hidden stages -- if the two drifted, a flat
        MSSM result would be about the drift rather than about the depth.

        ``metric='cos'`` is this project's form: ``1 - cos`` on the stable side
        (hinged at ``stable_margin``, which is 0 by default and therefore inert,
        since ``cos <= 1``) against ``max(cos - margin, 0)`` on the change side.
        ``metric='euclid'`` is SNIIF-Net's: a squared hinge on Euclidean distance
        both ways. That distance is taken on **L2-normalised** features, i.e.
        ``D = sqrt(2(1 - cos)) in [0, 2]``, which is the only way one pair of
        margins can be legal at three stages of different width and scale --
        the raw activations coming out of a 512-wide BatchNorm+GELU stage and a
        128-wide linear one do not live on a common scale, and a fixed theta
        applied to both would be a different objective at each.

        Both groups are averaged with equal weight, not pooled: stable plots
        outnumber change plots roughly 4:1 and a pooled mean would be almost
        entirely the stable term, i.e. a plain "make everything similar"
        regulariser.
        """
        import torch
        import torch.nn.functional as F

        cos = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        change = ~stable
        terms = []
        if metric == "euclid":
            d = (2.0 * (1.0 - cos)).clamp_min(0.0).sqrt()
            if stable.any():
                terms.append(torch.clamp(d[stable] - stable_margin,
                                         min=0.0).pow(2).mean())
            if change.any():
                terms.append(torch.clamp(margin - d[change],
                                         min=0.0).pow(2).mean())
        else:
            if stable.any():
                terms.append(torch.clamp(1.0 - cos[stable] - stable_margin,
                                         min=0.0).mean())
            if change.any():
                terms.append(torch.clamp(cos[change] - margin, min=0.0).mean())
        if not terms:
            return cos.new_tensor(0.0)
        return torch.stack(terms).mean()

    def _siam_mssm_loss(self, stable):
        """SNIIF-Net's multi-scale supervision: the pair term at every depth.

        The paper constrains its feature pair at 1/8, 1/4, 1/2 and 1/1 of the
        input resolution rather than at the output alone, on the argument that a
        change/no-change geometry imposed only where the prediction is made
        leaves the intermediate features free to be organised by anything. The
        tabular form of "scale" is encoder *stage*: ``[h1 (512), h2 (256)]`` and
        optionally ``z (128)``.

        Distinct from Q5's deep supervision in the one way that matters here.
        Q5 hung a *classification* head off each stage and was flat, and the
        reading was that a three-layer encoder under a three-level nested loss
        is already supervised at depth. That reading says nothing about this
        term: no loss in the model currently touches the pair geometry anywhere
        but at ``z``, so the stages are free to interleave the two dates however
        they like as long as the final layer can separate them. Unlike Q5 this
        also adds **no parameters at all** -- Q5's heads existed and had to be
        discarded at predict time; there is nothing here to discard.

        Reads the intermediates cached by the forward pass, so like
        ``_deep_sup_loss`` it must be called before any auxiliary term that runs
        the encoder again.
        """
        import torch

        pairs = list(self._siam_stages())
        if self.siam_mssm_scales == "all":
            pairs = pairs + [self._siam_pair()]
        terms = [self._pair_margin_term(
                     a, b, stable, metric=self.siam_mssm_metric,
                     margin=self.siam_mssm_margin,
                     stable_margin=self.siam_mssm_stable_margin)
                 for a, b in pairs]
        return torch.stack(terms).mean()

    def _siam_barlow_loss(self, stable):
        """Barlow Twins redundancy reduction over the STABLE endpoint pairs.

        Zbontar et al. (2021) need two augmented views of one sample and spend a
        lot of design effort on the augmentation policy. A stable plot supplies
        two views for free and without any policy at all: 2018 and 2024 are two
        genuine observations of the same unchanged ground, differing by
        acquisition, phenology and whatever else the embedding did not manage to
        abstract away. Driving the cross-correlation of the two views to the
        identity asks the encoder for features that are (a) invariant to those
        nuisances and (b) mutually decorrelated -- so what survives in
        ``z24 - z18`` is change rather than acquisition, and the embedding does
        not spend half its dimensions on one duplicated factor.

        Stable rows only. A change plot's two years are *not* two views of one
        thing, and pulling them together is precisely what ``_siam_cos_loss``
        pushes apart. Note this term needs no labels beyond stable/change, which
        is the cheapest label there is -- the property that makes it extendable
        to an unlabelled pool later.
        """
        import torch

        z18, z24 = self._siam_pair()
        a, b = z18[stable], z24[stable]
        n, d = a.shape
        if n < 2:
            return z18.new_tensor(0.0)
        # Standardise each view along the BATCH dimension (the paper's
        # normalisation), so the cross-correlation entries are correlations.
        a = (a - a.mean(0)) / (a.std(0) + 1e-6)
        b = (b - b.mean(0)) / (b.std(0) + 1e-6)
        c = (a.T @ b) / n
        on = torch.diagonal(c).add(-1.0).pow(2).sum()
        off = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
        return (on + self.siam_barlow_lambda * off) / d

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


class _PatchTower:
    """Small conv encoder over the stacked (years x bands) Sentinel-2 patch.

    Deliberately tiny. S3's verdict was that 1,344 raw pooled pixel values on
    6,414 plots is squarely the overfitting regime and that hand-built
    statistics beat a learned texture by a wide margin; a large CNN here would
    re-run that experiment with more parameters and lose harder. What this tests
    is the two things S3 did not have -- *weight sharing across the image*, so
    the parameter count is set by the filters rather than by the pixel count,
    and the dihedral augmentations that only exist once the patch is treated as
    an image.

    The two years enter as channels of one tensor rather than through a shared
    encoder, so the network sees both dates in the first convolution and can
    form a difference in filter space. That is the opposite choice from the
    AlphaEarth tower, and deliberately: the siamese argument is about sharing
    weights across two *comparable embedding vectors*, while here the useful
    structure is local and spatial.

    Global average pooling, not a flatten: a flatten would reintroduce exactly
    the per-pixel parameter explosion that sank S3.
    """

    def __new__(cls, in_ch: int, out_dim: int, dim: int, dropout: float):
        import torch.nn as nn

        def block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.GELU())

        return nn.Sequential(
            block(in_ch, dim), block(dim, dim * 2), block(dim * 2, dim * 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(dim * 2, out_dim), nn.GELU(),
        )


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
                tess_width: float = 1.0, aef_siam_perm=None,
                aef_siam_dim: int = 128, aef_siam_combine: str = "conc",
                aef_siam_crfe: str = "none", aef_siam_pyramid: bool = False,
                aef_siam_fiim: str = "none", aef_siam_hidden=(512, 256),
                patch_tensor=None, patch_augment: bool = True,
                patch_dim: int = 64):
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
                # aef_siam_perm swaps the AlphaEarth tower for a shared endpoint
                # encoder (section N). Its contract is identical -- d_aef in,
                # out_dim out -- so the gating, the modality dropout and the
                # gate-off serving path are all untouched, and Sentinel-2 stays
                # privileged information that is never read at inference.
                #
                # The permutation is carried INTO the tower rather than imposed
                # on `aef_columns`. The shared encoder needs the block as
                # [all _2018 | all _2024 | rest], but `feature_columns` returns
                # the sorted order, which interleaves the years by band. Making
                # the caller reorder would put a silent correctness requirement
                # on every path that builds an AlphaEarth matrix -- including
                # `stack_aef_bands`, whose raster row order is checked against
                # the *spec* rather than against the model. Gathering here means
                # any column order works and there is nothing to get wrong.
                if aef_siam_perm is not None:
                    i18, i24, rest = aef_siam_perm
                    self.aef_tower = _SiameseTrunk(
                        d_end=len(i18), d_extra=len(rest), out_dim=out_dim,
                        siam_dim=aef_siam_dim, dropout=dropout,
                        combine=aef_siam_combine, crfe=aef_siam_crfe,
                        pyramid=aef_siam_pyramid, fiim=aef_siam_fiim,
                        hidden=aef_siam_hidden,
                        reorder=torch.tensor(list(i18) + list(i24) + list(rest),
                                             dtype=torch.long))
                else:
                    self.aef_tower = tower(d_aef, dropout)
                if patch_tensor is None:
                    self.tess_tower = tower(d_tess, d_tess_drop, tess_width)
                else:
                    self.tess_tower = _PatchTower(
                        in_ch=patch_tensor.shape[1] * patch_tensor.shape[2],
                        out_dim=out_dim, dim=patch_dim, dropout=d_tess_drop)
                    # Plain attribute, not a buffer: this is fixed data, not a
                    # parameter, and registering ~200M values would put them in
                    # every state_dict and snapshot the early-stopping loop takes.
                    self.patches = patch_tensor
                    self.patch_augment = patch_augment
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
                if getattr(self, "patches", None) is None:
                    rt = self.tess_tower(xt)
                else:
                    # Last column is the raw row index into `patches` (-1 = this
                    # plot has no imagery). Absent rows are gathered from row 0
                    # and then gated out by mt exactly as a zero-imputed columnar
                    # block would be, so they cost a forward pass and nothing else.
                    # Gather on the CPU where `patches` lives, then move only the
                    # batch to the device -- indexing a CPU tensor with a CUDA
                    # index is an error, and moving the whole array to the GPU
                    # would defeat holding it as uint8 in the first place.
                    idx = x[:, -1].long().clamp_min(0).cpu()
                    img = self.patches[idx].to(x.device, torch.float32) / 255.0
                    img = img.flatten(1, 2)          # (n, years*bands, H, W)
                    if self.training and self.patch_augment:
                        # The eight dihedral symmetries. A land-cover transition
                        # is invariant to them and the labels are unchanged, so
                        # this is free extra data of exactly the kind S3's
                        # flattened-pixel test could not use -- a flat column
                        # vector has no geometry to reflect. Applied per batch
                        # rather than per row: one gather stays contiguous, and
                        # over 30 epochs the batch still sees every orientation.
                        if torch.rand(()) < 0.5:
                            img = torch.flip(img, dims=[-1])
                        if torch.rand(()) < 0.5:
                            img = torch.flip(img, dims=[-2])
                        k = int(torch.randint(4, ()))
                        if k:
                            img = torch.rot90(img, k, dims=[-2, -1])
                    rt = self.tess_tower(img)
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


class _SiameseTrunk:
    """Shared-encoder trunk over the two endpoint years, built lazily.

    The input is ``[x18 | x24 | extra]`` (packed by ``_prepare``): ``d_end``
    columns for 2018, *the same features in the same order* for 2024, then
    ``d_extra`` columns that are not per-year (change scalars, a Sentinel-2
    detail block) and bypass the encoder. One encoder ``f`` reads both dates::

        z18 = f(x18)        z24 = f(x24)

    the Siamese change-detection structure of Daudt et al. (2018) written for
    tabular embeddings rather than image patches. Two properties follow, and
    they are the reason to try it against the flat ``wide`` trunk:

    * **Half the encoder parameters at the same input width.** The flat trunk
      learns separate first-layer weights for the 2018 block, the 2024 block and
      the diff block; this learns one set used three times. With 6.4k plots and
      46 examples of the rarest transition, parameter count is a live constraint
      -- the learning curves put the model at +0.026 change-F1 per doubling of
      labels, i.e. squarely in the data-limited regime where sharing pays.
    * **Year symmetry by construction.** A flat trunk can at best *learn* that
      ``A07_2018`` and ``A07_2024`` are one measurement at two dates. Here it is
      told. ``_prepare`` standardises both years with pooled statistics for the
      same reason.

    Both dates go through the encoder in a **single stacked call**, so its
    BatchNorm sees one distribution over 2018-and-2024 rows rather than two
    per-year distributions. Running them separately would give each year its own
    batch statistics and silently re-centre a real between-year shift to zero --
    which is the signal. The last encoder layer is deliberately **linear**: a
    GELU there would push the embedding almost entirely into the non-negative
    orthant and compress the cosine the auxiliary losses act on into a narrow
    positive band.

    ``combine`` sets what the head reads::

        diff:  [z24 - z18, |z24 - z18|, cos(z18, z24)]
        conc:  [z18, z24, z24 - z18, |z24 - z18|, cos(z18, z24)]

    ``diff`` is the pure change representation -- a stable plot maps near zero
    and the head cannot tell stable Nature from stable Artificial. That is fatal
    for *this* target, whose classes are ``from -> to`` and not merely
    changed/unchanged, so ``conc`` keeps the endpoint states as well. The cosine
    is carried as an explicit feature because it is the statistic the auxiliary
    losses supervise, and because the analogous raw-embedding scalar already
    earned its place (``diff+cos`` beat plain ``diff``, 0.6639 vs 0.6567).

    ``last_z18`` / ``last_z24`` are kept for the cosine and Barlow losses, which
    need the pair before it is combined. ``last_h`` keeps the per-stage
    intermediates for the deep-supervision term (section Q).

    Three options here transcribe modules from Zhang et al.'s burned-area Swin
    network onto this tabular encoder, which is the only form they can take
    without an image grid (section Q):

    ``crfe`` -- their Change-Region Feature Enhancement. ``"sum"`` adds the
    *elementwise sum* ``z18 + z24`` to the head's block, so the pair is read
    through both of their fusion operators (add and subtract) rather than the
    difference alone; ``"attn"`` puts a squeeze-and-excitation gate over the
    assembled block, which is their channel attention with the spatial-squeeze
    step dropped because there are no spatial dims to squeeze; ``"full"`` is
    both, i.e. their module as published minus the spatial branch.

    ``pyramid`` -- their pyramid up-sampling decoder. There is no resolution to
    recover here, so what survives is the *depth* half of the idea: the two
    hidden stages are projected to ``siam_dim`` and folded into the embedding
    bottom-to-top (``z + P2 h2 + P1 h1``), zero-initialised so an untrained
    pyramid is exactly the plain encoder and every logit it moves has to be paid
    for. Note this changes ``z`` itself, which is what the cosine and Barlow
    terms read -- deliberately, since the paper predicts from the fused map.

    ``fiim`` is from a different paper (SNIIF-Net, Sci Rep 2025) and is section
    Q10: their Feature Information Interaction Module, which lets the two
    branches exchange information *before* the difference is taken. Their form
    is spatial attention over feature maps and has no tabular analogue, but the
    cross-branch part does -- ``z18 * (1 + tanh(W [z18 | z24]))``, one shared
    ``W`` used with the inputs swapped for the other date, zero-initialised. The
    property being tested is placement, not the gate: ``crfe='attn'`` already
    gates, but downstream of the subtraction, where it can no longer change
    ``z24 - z18``, the cosine feature or what the pair losses see. Note the
    single-date entry point ``encode_single`` cannot apply it (there is no
    pair), which is correct rather than a gap -- the state-pretraining phase
    trains ``enc``, and the interaction is not part of ``enc``.
    ``fiim='self'`` is its control: the same gate at the same size reading its
    own date twice, so only the cross-branch information is removed.
    """

    def __new__(cls, d_end: int, d_extra: int, out_dim: int, siam_dim: int,
                dropout: float, combine: str = "conc", year_adapter: str = "none",
                reorder=None, crfe: str = "none", pyramid: bool = False,
                fiim: str = "none", hidden=(512, 256)):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        if combine not in ("diff", "conc"):
            raise ValueError(f"Unknown siamese combine: {combine}")
        if year_adapter not in ("none", "input", "output"):
            raise ValueError(f"Unknown siamese year_adapter: {year_adapter}")
        if crfe not in ("none", "sum", "attn", "full", "rand", "randattn"):
            raise ValueError(f"Unknown siamese crfe: {crfe}")
        if fiim not in ("none", "cross", "self"):
            raise ValueError(f"Unknown siamese fiim: {fiim}")

        d_comb = (2 if combine == "diff" else 4) * siam_dim + 1
        if crfe in ("sum", "full", "rand", "randattn"):
            d_comb += siam_dim

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_end, self.d_extra, self.combine = d_end, d_extra, combine
                self.year_adapter = year_adapter
                # Buffer, not a plain attribute, so it moves with .to(device)
                # and survives a state_dict round-trip with the model.
                if reorder is None:
                    self.reorder = None
                else:
                    self.register_buffer("reorder", reorder)
                # Pseudo-siamese calibration (Daudt et al.'s siamese vs
                # pseudo-siamese distinction, at the smallest size that tests it).
                # A fully shared encoder must read an absolute land-cover state
                # through one map for both dates; a flat trunk gets a separate
                # first-layer weight block per year and can absorb a sensor or
                # phenology offset there. N9 traced the stable-built-up
                # regression to exactly that. This restores per-year calibration
                # with a DIAGONAL affine -- 2 x d_end parameters per year, not a
                # second encoder -- so the weight sharing that produced the
                # focus-class gains is kept and only the calibration is freed.
                # Identity-initialised, so an untrained adapter is exactly the
                # fully-shared model and any deviation has to pay for itself.
                if year_adapter == "input":
                    self.g18 = nn.Parameter(torch.ones(d_end))
                    self.b18 = nn.Parameter(torch.zeros(d_end))
                    self.g24 = nn.Parameter(torch.ones(d_end))
                    self.b24 = nn.Parameter(torch.zeros(d_end))
                elif year_adapter == "output":
                    self.g18 = nn.Parameter(torch.ones(siam_dim))
                    self.b18 = nn.Parameter(torch.zeros(siam_dim))
                    self.g24 = nn.Parameter(torch.ones(siam_dim))
                    self.b24 = nn.Parameter(torch.zeros(siam_dim))
                stack, prev = [], d_end
                for width in hidden:
                    stack += [nn.Linear(prev, width), nn.BatchNorm1d(width),
                              nn.GELU(), nn.Dropout(dropout)]
                    prev = width
                stack.append(nn.Linear(prev, siam_dim))  # linear on purpose
                self.enc = nn.Sequential(*stack)
                self.mixer = nn.Sequential(
                    nn.Linear(d_comb + d_extra, 512), nn.BatchNorm1d(512),
                    nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(512, out_dim), nn.GELU(),
                )
                # Everything below is built AFTER enc and mixer on purpose: the
                # modules above then draw the same initialisation from the seeded
                # RNG whether or not the section-Q options are on, so a flat
                # result is the option's and not a reshuffled init's.
                self.crfe, self.pyramid = crfe, pyramid
                #: Hidden widths of `enc`, in order. Read by the deep-supervision
                #: heads, which live on the model rather than in here.
                self.stage_dims = tuple(hidden)
                if crfe in ("attn", "full", "randattn"):
                    self.se = _SEInput(d_comb + d_extra)
                if crfe in ("rand", "randattn"):
                    # The CONTROL for the `sum` arm. z18 + z24 is a fixed linear
                    # map of a block that already contains z18 and z24, so it
                    # cannot change what the mixer is able to compute -- only how
                    # the optimiser gets there, and how wide the first layer is.
                    # A fixed random linear view of the same pair at the same
                    # width is the same kind of object with none of CRFE's
                    # meaning, so if it reproduces the effect then the effect is
                    # width and conditioning, not the sum operator.
                    self.register_buffer("rand_mix",
                                         torch.randn(2 * siam_dim, siam_dim)
                                         / (2 * siam_dim) ** 0.5)
                if pyramid:
                    # Zero-initialised, so training starts at the plain encoder.
                    self.pyr = nn.ModuleList(
                        [nn.Linear(dim, siam_dim) for dim in self.stage_dims])
                    for layer in self.pyr:
                        nn.init.zeros_(layer.weight)
                        nn.init.zeros_(layer.bias)
                self.fiim = fiim
                if fiim != "none":
                    # ONE gate, applied to both dates with the two inputs
                    # swapped, so the module is siamese in the same sense the
                    # encoder is: swapping 2018 and 2024 swaps the outputs and
                    # nothing else. Zero-initialised -> tanh(0) = 0 -> the
                    # multiplier is exactly 1, so an untrained FIIM is exactly
                    # the plain encoder and anything it moves has to be paid for
                    # (the Q4 convention).
                    self.fiim_gate = nn.Linear(2 * siam_dim, siam_dim)
                    nn.init.zeros_(self.fiim_gate.weight)
                    nn.init.zeros_(self.fiim_gate.bias)

            def _run_enc(self, x):
                """``enc`` run stage by stage: (z, [h1, h2]).

                Sliced rather than restructured into a ModuleList, so the layers,
                their init order and the dropout draws are bit-identical to the
                published trunk and only the extra terms differ.
                """
                h1 = self.enc[0:4](x)
                h2 = self.enc[4:8](h1)
                z = self.enc[8:](h2)
                if self.pyramid:
                    # Bottom-to-top, deepest first: z + P2 h2, then + P1 h1.
                    z = z + self.pyr[1](h2) + self.pyr[0](h1)
                return z, [h1, h2]

            def encode_single(self, x, year: str = "2018"):
                """Encode ONE date's block -- the single-date entry point.

                This is what makes an external state label usable: ``f`` never
                needed the pair, only the classifier head above it did. The
                caller is responsible for handing over a block already
                standardised with the endpoint statistics the paired path uses,
                or the encoder sees a different distribution than it was
                trained on.
                """
                if self.year_adapter == "input":
                    g, b = ((self.g18, self.b18) if year == "2018"
                            else (self.g24, self.b24))
                    x = g * x + b
                z, _ = self._run_enc(x)
                if self.year_adapter == "output":
                    g, b = ((self.g18, self.b18) if year == "2018"
                            else (self.g24, self.b24))
                    z = g * z + b
                return z

            def forward(self, x):
                n = x.shape[0]
                if self.reorder is not None:
                    x = x[:, self.reorder]
                x18 = x[:, :self.d_end]
                x24 = x[:, self.d_end:2 * self.d_end]
                if self.year_adapter == "input":
                    x18 = self.g18 * x18 + self.b18
                    x24 = self.g24 * x24 + self.b24
                z, hs = self._run_enc(torch.cat([x18, x24], dim=0))  # one BN pop.
                z18, z24 = z[:n], z[n:]
                # Per-stage pair, for the deep-supervision heads.
                self.last_h = [(h[:n], h[n:]) for h in hs]
                if self.year_adapter == "output":
                    # After the encoder the cosine and Barlow terms read these,
                    # so a per-year affine here also rescales what those losses
                    # see -- the reason 'input' is the default reading of N10.
                    z18 = self.g18 * z18 + self.b18
                    z24 = self.g24 * z24 + self.b24
                if self.fiim != "none":
                    # Feature information interaction: each date re-weighted by a
                    # gate that has seen BOTH dates. Deliberately upstream of the
                    # subtraction and of last_z18/last_z24, so it changes the
                    # difference, the cosine feature AND what the pair losses
                    # read -- which is the whole difference from crfe='attn',
                    # whose gate sits on the assembled block downstream of all
                    # three. The paper predicts from its interacted features for
                    # the same reason.
                    #
                    # 'self' is the CONTROL: each date's gate reads that date
                    # twice instead of the pair. Identical parameter count,
                    # identical nonlinearity, identical init draw -- the only
                    # thing removed is the cross-branch information, which is the
                    # one thing the module claims to add.
                    other18 = z18 if self.fiim == "self" else z24
                    other24 = z24 if self.fiim == "self" else z18
                    e18 = z18 * (1.0 + torch.tanh(
                        self.fiim_gate(torch.cat([z18, other18], dim=1))))
                    e24 = z24 * (1.0 + torch.tanh(
                        self.fiim_gate(torch.cat([z24, other24], dim=1))))
                    z18, z24 = e18, e24
                d = z24 - z18
                cos = F.cosine_similarity(z18, z24, dim=1, eps=1e-8).unsqueeze(1)
                parts = ([z18, z24, d, d.abs(), cos] if self.combine == "conc"
                         else [d, d.abs(), cos])
                if self.crfe in ("rand", "randattn"):
                    parts.append(torch.cat([z18, z24], dim=1) @ self.rand_mix)
                if self.crfe in ("sum", "full"):
                    # CRFE's other fusion operator. Linear in (z18, z24) and so
                    # spanned by the block already -- exactly as the raw AlphaEarth
                    # `diff` block is, which is worth -0.048 change-F1 to remove
                    # from the flat trunk. That is the precedent for testing it.
                    parts.append(z18 + z24)
                if self.d_extra:
                    parts.append(x[:, 2 * self.d_end:])
                self.last_z18, self.last_z24 = z18, z24
                block = torch.cat(parts, dim=1)
                if self.crfe in ("attn", "full", "randattn"):
                    block = self.se(block)
                return self.mixer(block)

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
