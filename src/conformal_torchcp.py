"""TorchCP conformal methods compared on the model behind the deployed raster.

The model is `siam_s2off_state_pre` (`infer_s2.fit_models`), the recipe that
wrote `data/inference/s2_20260731_120223/oslo_siam_s2off_state_pre_*`. The
raster names the model; it cannot score one -- Oslo has **zero** labelled plots
inside the AOI. Everything with a truth in it is therefore computed on that
model's cached out-of-fold plot posteriors (15 seeds, `twotower_lab_oof/`), on
the same spatially blocked folds the ledger uses, so the numbers here sit beside
`conformal_report.py` rather than in a different universe. `--map-probs` adds the
one thing the raster *can* answer on its own: set size on 2.95M unlabelled
pixels, which is efficiency at the deployment prior rather than the plot prior.

## The thing to understand before reading the table

**ECE, Brier and CRPS are properties of the probability vector. Conformal set
constructors do not touch the probability vector.** LAC, APS, RAPS, SAPS, Margin
and TopK all consume the same softmax and emit a set; split vs class-conditional
vs clustered vs RC3P only changes where the threshold is cut. So those three
columns are *constant* down any block of rows that share a calibrator, and a
table that varies only the score function has nothing to say about them. That is
not a bug in the comparison, it is the comparison: asking "which conformal method
has the best ECE" is a category error, and the answer is "whichever probability
calibrator it was handed".

The grid is therefore two axes, and each axis owns some of the metrics:

* **calibrator** (`raw` / `temp` / `costgate`) -- transforms the posterior. Owns
  ECE, Brier, CRPS. Moves coverage and efficiency only indirectly, by changing
  the scores the threshold is cut on.
* **score x predictor** -- the conformal method proper. Owns coverage and
  efficiency. Cannot move ECE, Brier or CRPS at all.

`costgate` is the cost vector in `coarse3_costs__siam_s2off_state_pre.json`, the
one that turns `*_coarse3.tif` into `*_coarse3_gated.tif`. It is in the grid
because it is the transform the named raster actually applies, and because it is
a genuine probability transform -- so it is the arm that shows what the shipped
decision rule costs in calibration terms. It is fitted on these same OOF rows
(that is what `fit_coarse3_costs.py` is for), so its prob-side numbers are
mildly optimistic; `temp` is fitted per fold and is not.

## CRPS

CRPS on *nominal* categories is degenerate: with the discrete metric it equals
exactly half the multiclass Brier score (`crps_nominal` is reported only to make
that visible). It carries information only against an ordering, so the ordinal
one used here is stated rather than assumed --

    naturalness: Artificial = 0, Cropland = 1, Nature = 2   (Vegetation = 1 at merged2)
    severity delta = naturalness(end) - naturalness(start)

giving five ordered levels at coarse3 (-2 = Nature -> Artificial, the habitat
loss this project exists to map, ... +2 = Artificial -> Nature) and three at
merged2. `crps_ord` is the ranked probability score of that collapsed ordinal
forecast, normalised by (levels - 1) so it stays in [0, 1]. It penalises putting
mass two rungs from the truth more than one rung, which plain Brier does not --
predicting `Nature -> Cropland` when the truth is `Nature -> Artificial` is a
better wrong answer than predicting `Nature -> Nature`, and only `crps_ord` says
so.

## Exchangeability

The folds are spatial blocks, so exchangeability between calibration and test is
violated by construction and every coverage guarantee here is nominal. Realised
coverage is a measurement, not a formality. Read it first.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python   # the .venv is broken
    $P src/conformal_torchcp.py                                   # 5 seeds, both levels
    $P src/conformal_torchcp.py --n-seeds 15 --alphas 0.10
    $P src/conformal_torchcp.py --map-probs data/inference/<run>/oslo_..._coarse3_probs.tif
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# torchcp import
# ---------------------------------------------------------------------------
# `torchcp/__init__.py` eagerly imports its graph/ and llm/ subpackages, which
# drag in torch_geometric, torchvision and transformers. None of that is needed
# for classification, and installing three heavy trees into a shared pixi env to
# satisfy an unused import is the wrong trade. A meta-path finder fabricates any
# module under those roots, so the import succeeds and nothing we call touches
# the stubs. Only classification/ is used below.
_STUB_ROOTS = ("torch_geometric", "transformers", "torchvision")


def _stub_attr(name):
    # Dunders must still raise: `inspect.getmodule` walks every entry in
    # sys.modules and asks each one for `__file__`, and a stub that answers with
    # a class there breaks any later traceback or `torch.library` registration
    # that happens to import while these are installed.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return type(name, (), {})


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__file__ = "<torchcp-unused-dependency-stub>"
        module.__getattr__ = _stub_attr
        return module

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUB_ROOTS and fullname not in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(),
                                                  is_package=True)
        return None


if "torchcp" not in sys.modules:
    sys.meta_path.insert(0, _StubFinder())

from torchcp.classification.predictor import (  # noqa: E402
    ClassConditionalPredictor, ClusteredPredictor, RC3PPredictor, SplitPredictor)
from torchcp.classification.score import (  # noqa: E402
    APS, LAC, RAPS, SAPS, Margin, TOPK)
from torchcp.classification.trainer.ts_trainer import _ECELoss  # noqa: E402
from torchcp.classification.utils.metrics import (  # noqa: E402
    CovGap, SSCV, average_size, coverage_rate, singleton_hit_ratio)

import twotower_lab as lab  # noqa: E402
from project_paths import project_data_dir  # noqa: E402

OUT = project_data_dir("analysis_results") / "conformal_torchcp.csv"
COSTS_DIR = project_data_dir("analysis_results")

#: The recipe that wrote the raster this was asked about. `infer_s2.py` reads
#: `siam_s2off_state_pre` through the `siam_s2off_state_pre` cost vector, so the
#: OOF cache under that name is the model, and `c3gate_...` is the same
#: posteriors with the gate applied at arg-max time (identical `fine_probs`).
DEFAULT_SOURCE = "siam_s2off_state_pre"

#: Ordinal axis for CRPS. Not an ordering of the legend -- an ordering of what
#: the transition does to habitat. See the module docstring.
NATURALNESS = {"Artificial": 0, "Cropland": 1, "Vegetation": 1, "Nature": 2}


# ---------------------------------------------------------------------------
# the conformal grid
# ---------------------------------------------------------------------------
def score_functions() -> dict:
    """The six TorchCP nonconformity scores, with their published defaults.

    `LAC` is Sadinle's least-ambiguous score and the one `twotower_lab` already
    implements, so it is the bridge to the existing ledger. `APS` is Romano's
    adaptive score; `RAPS` is APS with a rank penalty that truncates the tail of
    the set, and `SAPS` replaces all but the top probability with a constant --
    both exist to stop APS paying for conditional coverage in set size, which is
    the exact trade this legend (4,200-plot class against a 46-plot one) forces.
    `Margin` and `TOPK` are the cheap baselines: without them there is no way to
    tell whether an adaptive score earned anything.

    RAPS `penalty`/`kreg` are left at the small-K defaults (0.1, 1): with 9
    classes a rank penalty tuned on a 1000-class benchmark is meaningless, and
    tuning it here on the same rows it is scored on would be the mistake
    `fit_coarse3_costs.py` exists to avoid.
    """
    return {
        "lac": lambda: LAC(score_type="softmax"),
        "aps": lambda: APS(score_type="softmax", randomized=True),
        "raps": lambda: RAPS(score_type="softmax", randomized=True,
                             penalty=0.1, kreg=1),
        "saps": lambda: SAPS(score_type="softmax", randomized=True, weight=0.2),
        "margin": lambda: Margin(score_type="softmax"),
        "topk": lambda: TOPK(score_type="softmax", randomized=True),
    }


def predictors() -> dict:
    """Threshold placements, from one cut to one cut per class.

    `split` is a single pooled quantile -- marginal coverage only, and over a
    legend that is 66% stable Vegetation it can be met while a rare transition is
    never covered at all. `classwise` is the Mondrian cut `nested_conformal`
    already uses. `cluster` (Ding et al.) groups classes by the shape of their
    score distribution and shares a quantile inside a group, which is the
    designed answer to "the rare class has too few calibration rows to hold its
    own quantile". `rc3p` calibrates a label-rank threshold alongside the score
    threshold, and is the one method in the library built specifically for the
    long-tailed class-conditional case this legend is.
    """
    return {
        "split": lambda score, alpha: SplitPredictor(score, alpha=alpha,
                                                     device="cpu"),
        "classwise": lambda score, alpha: ClassConditionalPredictor(
            score, alpha=alpha, device="cpu"),
        "cluster": lambda score, alpha: ClusteredPredictor(
            score, alpha=alpha, device="cpu"),
        "rc3p": lambda score, alpha: RC3PPredictor(score, alpha=alpha,
                                                   device="cpu"),
    }


# ---------------------------------------------------------------------------
# probability-side calibrators
# ---------------------------------------------------------------------------
def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Guo et al. temperature, LBFGS on calibration NLL.

    One parameter, so it cannot reorder a row's classes: accuracy and every
    arg-max metric are invariant under it, and only the sharpness moves. That is
    what makes it the clean arm for showing which metrics a calibrator owns.
    """
    log_t = torch.zeros(1, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_t.exp().item())


def calibrated_probs(probs: np.ndarray, y: np.ndarray, ok: np.ndarray,
                     folds: list, mode: str, costs: np.ndarray | None):
    """Apply a probability transform out-of-fold, returning ``(probs, note)``.

    `temp` is fitted on each fold's calibration rows and applied to its test
    rows, so no row is transformed by a parameter that saw it. `costgate` is the
    shipped vector and is applied whole -- refitting it per fold would be a
    different (and unshipped) object.
    """
    if mode == "raw":
        return probs, ""
    if mode == "costgate":
        if costs is None:
            return None, ""
        scaled = probs * costs[None, :]
        return scaled / scaled.sum(1, keepdims=True), ""
    if mode != "temp":
        raise ValueError(f"unknown calibrator: {mode}")

    logits_all = torch.as_tensor(np.log(np.clip(probs, 1e-12, None)),
                                 dtype=torch.float64)
    out = np.array(probs, dtype=float)
    temps = []
    for train, test in folds:
        cal = train[ok[train]]
        temperature = fit_temperature(logits_all[cal].float(),
                                      torch.as_tensor(y[cal]))
        temps.append(temperature)
        block = torch.softmax(logits_all[test] / temperature, dim=1)
        out[test] = block.numpy()
    return out, f"T={np.mean(temps):.3f}"


# ---------------------------------------------------------------------------
# probability-side metrics
# ---------------------------------------------------------------------------
def severity_delta(classes: list) -> np.ndarray:
    """Habitat-severity rung for each transition class, or NaN if unmappable."""
    delta = np.full(len(classes), np.nan)
    for i, cls in enumerate(classes):
        parts = str(cls).split(" -> ")
        if len(parts) != 2:
            continue
        start, end = (NATURALNESS.get(p.strip()) for p in parts)
        if start is not None and end is not None:
            delta[i] = end - start
    return delta


def brier_score(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier, the sum-over-classes convention (range 0..2)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def crps_ordinal(probs: np.ndarray, y: np.ndarray, delta: np.ndarray) -> float:
    """Ranked probability score on the collapsed severity axis, in [0, 1].

    Classes sharing a rung are summed before the CDF is taken, so this scores
    the forecast of *how much habitat the transition costs*, not of which of the
    nine cells it is. Returns NaN when the level has no usable ordering.
    """
    if np.isnan(delta).any():
        return float("nan")
    levels = np.unique(delta)
    if len(levels) < 2:
        return float("nan")
    collapsed = np.stack([probs[:, delta == lvl].sum(1) for lvl in levels], 1)
    truth_level = np.searchsorted(levels, delta[y])
    onehot = np.zeros_like(collapsed)
    onehot[np.arange(len(y)), truth_level] = 1.0
    diff = collapsed.cumsum(1) - onehot.cumsum(1)
    return float((diff ** 2).sum(1).mean() / (len(levels) - 1))


def classwise_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Mean over classes of the binned |predicted probability - frequency|.

    Top-label ECE is dominated by the two majority classes here and can look
    healthy while a 46-plot transition's probabilities mean nothing. This reads
    every column, which is the calibration question a per-class map product
    actually asks.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    per_class = []
    for k in range(probs.shape[1]):
        p = probs[:, k]
        hit = (y == k).astype(float)
        gap = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            rows = (p > lo) & (p <= hi)
            if not rows.any():
                continue
            gap += abs(p[rows].mean() - hit[rows].mean()) * rows.mean()
        per_class.append(gap)
    return float(np.mean(per_class))


def probability_metrics(probs: np.ndarray, y: np.ndarray,
                        delta: np.ndarray) -> dict:
    """The calibrator-owned block: ECE, Brier, CRPS, and the two references."""
    logits = torch.as_tensor(np.log(np.clip(probs, 1e-12, None)),
                             dtype=torch.float32)
    labels = torch.as_tensor(y, dtype=torch.long)
    brier = brier_score(probs, y)
    return {
        "ece": float(_ECELoss(n_bins=15)(logits, labels).item()),
        "ece_classwise": classwise_ece(probs, y),
        "brier": brier,
        # Reported to make the degeneracy explicit rather than to add evidence:
        # for nominal categories CRPS with the discrete metric IS half the Brier
        # score, exactly. `crps_ord` is the only informative reading.
        "crps_nominal": 0.5 * brier,
        "crps_ord": crps_ordinal(probs, y, delta),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12,
                                     None)).mean()),
        "acc": float((probs.argmax(1) == y).mean()),
    }


# ---------------------------------------------------------------------------
# set-side metrics
# ---------------------------------------------------------------------------
def set_metrics(sets: np.ndarray, y: np.ndarray, classes: list, alpha: float,
                argmax: np.ndarray) -> dict:
    """Coverage and efficiency, from TorchCP's own metric implementations."""
    tensor = torch.as_tensor(sets.astype(np.int64))
    labels = torch.as_tensor(y, dtype=torch.long)
    size = sets.sum(1)
    single = size == 1
    covered = sets[np.arange(len(y)), y]
    correct = argmax == y

    row = {
        "coverage": coverage_rate(tensor, labels),
        "cov_macro": coverage_rate(tensor, labels, coverage_type="macro",
                                   num_classes=len(classes)),
        # Mean |per-class coverage - nominal|, in percentage points. The single
        # number for "is this conditionally valid", and the one the rare classes
        # move.
        "covgap": CovGap(tensor, labels, alpha, len(classes)),
        # Size-stratified coverage violation: does coverage hold *within* each
        # set-size band, or is the marginal number a trade between confident
        # rows over-covered and ambiguous rows under-covered?
        "sscv": SSCV(tensor, labels, alpha,
                     stratified_size=[[0, 1], [2, 2], [3, 3], [4, 10]]),
        "set_size": average_size(tensor),
        "singleton_frac": float(single.mean()),
        # Accuracy on the rows the method is willing to commit to. The selective
        # -prediction read: this pair is the abstention curve in two numbers.
        "singleton_hit": singleton_hit_ratio(tensor, labels),
        "empty_frac": float((size == 0).mean()),
        # Arg-max accuracy on the rows the method commits to, against arg-max
        # accuracy on the rows it hedges. Not coverage -- the abstention curve
        # is about whether the *point* prediction is trustworthy where the set
        # is a singleton, which is the question a map uncertainty band asks.
        "acc_singleton": float(correct[single].mean()) if single.any() else np.nan,
        "acc_hedged": float(correct[~single].mean()) if (~single).any() else np.nan,
        "size_p90": float(np.quantile(size, 0.90)),
    }
    for k, cls in enumerate(classes):
        rows = y == k
        slug = str(cls).lower().replace(" -> ", "_to_").replace(" ", "")
        row[f"cov_{slug}"] = float(covered[rows].mean()) if rows.any() else np.nan
        row[f"n_{slug}"] = int(rows.sum())
    return row


def _prepare(predictor, n_classes: int):
    """Work around RC3P setting ``num_classes`` only on the dataloader path.

    `RC3PPredictor.calibrate` assigns it from the batch; `calculate_threshold`,
    the documented entry point for precomputed logits, reads it without ever
    assigning it and dies on ``torch.full(size=(None,))``. Setting it here is the
    smallest correct fix and keeps the library unpatched.
    """
    if getattr(predictor, "num_classes", "absent") is None:
        predictor.num_classes = n_classes
    return predictor


def conformal_sets(probs: np.ndarray, y: np.ndarray, ok: np.ndarray,
                   folds: list, make_score, make_predictor, alpha: float,
                   seed: int) -> np.ndarray | None:
    """Fold-wise sets: each row's threshold is cut on folds it is not in.

    Same protocol as `twotower_lab.nested_conformal`, so a TorchCP `lac`/`split`
    row and the existing ledger's marginal LAC row are the same experiment run
    through two implementations, and disagreeing is informative.
    """
    logits = torch.as_tensor(np.log(np.clip(probs, 1e-12, None)),
                             dtype=torch.float32)
    labels = torch.as_tensor(y, dtype=torch.long)
    sets = np.zeros(probs.shape, dtype=bool)
    for fold, (train, test) in enumerate(folds):
        cal = train[ok[train]]
        # APS/RAPS/SAPS/TopK randomise inside the score. Seeded per fold so a
        # rerun reproduces, and so two calibrators see the same draw.
        torch.manual_seed(seed * 1000 + fold)
        predictor = _prepare(make_predictor(make_score(), alpha),
                             probs.shape[1])
        try:
            predictor.calculate_threshold(logits[cal], labels[cal], alpha)
            block = predictor.predict_with_logits(logits[test])
        except (ValueError, RuntimeError, IndexError):
            # Clustered and RC3P both refuse configurations they cannot support
            # (too few rows for the rarest class to embed or to rank-calibrate).
            # That is a real property of this legend at this alpha, not a bug to
            # paper over -- the cell is dropped and the CSV shows the gap.
            return None
        sets[test] = np.asarray(block.cpu()).astype(bool)
    return sets


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def load_level(source: str, level: str, seed: int):
    """``(probs, classes, truth)`` for one seed at one level, or None."""
    cached = lab.load_oof(source, "full", seed)
    if cached is None:
        return None
    if level == "merged2":
        return cached[0], None, "merged"
    fine = lab.load_oof_fine(source, "full", seed)
    if fine is None:
        return None
    return fine[0], list(fine[1]), "fine"


def load_costs(source: str, classes: list) -> np.ndarray | None:
    """The shipped coarse3 decision-cost vector, aligned to `classes`."""
    path = COSTS_DIR / f"coarse3_costs__{source}.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    order = {c: i for i, c in enumerate(blob["fine_classes"])}
    if any(c not in order for c in classes):
        return None
    return np.array([blob["costs"][order[c]] for c in classes], dtype=float)


def run_level(ctx, source: str, level: str, seeds, alphas, calibrators,
              wanted_scores, wanted_predictors) -> list:
    view = ctx.view("full")
    scores, preds = score_functions(), predictors()
    rows = []
    for seed in seeds:
        loaded = load_level(source, level, seed)
        if loaded is None:
            continue
        probs, fine_classes, kind = loaded
        if kind == "merged":
            classes, truth = view.merged_classes, view.truth_merged
        else:
            classes, truth = fine_classes, view.truth_fine
        index = {c: i for i, c in enumerate(classes)}
        raw_idx = np.array([index.get(t, -1) for t in truth])
        ok = raw_idx >= 0
        y = np.clip(raw_idx, 0, None)
        delta = severity_delta(classes)
        costs = load_costs(source, classes) if kind == "fine" else None

        for calib in calibrators:
            adjusted, note = calibrated_probs(probs, y, ok, view.folds, calib,
                                              costs)
            if adjusted is None:
                continue
            prob_block = probability_metrics(adjusted[ok], y[ok], delta)
            prob_block["note"] = note
            for score_name in wanted_scores:
                for pred_name in wanted_predictors:
                    for alpha in alphas:
                        sets = conformal_sets(adjusted, y, ok, view.folds,
                                              scores[score_name],
                                              preds[pred_name], alpha, seed)
                        if sets is None:
                            continue
                        row = dict(source=source, level=level, calibrator=calib,
                                   score=score_name, predictor=pred_name,
                                   alpha=alpha, seed=seed)
                        row.update(prob_block)
                        row.update(set_metrics(sets[ok], y[ok], classes, alpha,
                                               adjusted[ok].argmax(1)))
                        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# the map-side read
# ---------------------------------------------------------------------------
def map_efficiency(prob_raster: Path, source: str, ctx, seeds, alphas,
                   calibrators, wanted_scores, wanted_predictors) -> list:
    """Set size on the raster's own pixels, at a single deployment threshold.

    Coverage cannot be read here -- Oslo has no labelled plots -- but efficiency
    can, and it is not the same number as the plot-level one: the map is 65%
    stable Nature against the plot pool's mix, so a method whose sets are small
    where the model is confident wins by more on the map than on the plots. This
    fits ONE threshold on every labelled plot (the `fit_coarse3_costs.py`
    argument: a raster has no folds, so it ships an instance, and the fold-wise
    runs above stay the evidence that the procedure generalises).

    The calibration probabilities are the **seed-averaged** OOF, not one seed's.
    The raster is a 5-seed ensemble mean, so a threshold cut on a single seed's
    posteriors would be placed on a sharper-tailed distribution than the pixels
    it is applied to, and every set size here would be wrong in the same
    direction. The folds are deterministic in `block_id`, so the seed-mean of
    the per-seed OOF matrices *is* the ensemble's out-of-fold posterior. `seeds`
    is then reused only to replicate the randomised scores' own RNG draw.
    """
    import rasterio

    with rasterio.open(prob_raster) as src:
        stack = src.read().astype(np.float64)
        names = [src.descriptions[i] or f"band{i + 1}" for i in range(src.count)]
    flat = stack.reshape(len(names), -1).T
    valid = np.isfinite(flat).all(1)
    pixels = flat[valid]

    view = ctx.view("full")
    scores, preds = score_functions(), predictors()
    stackable, classes = [], None
    for seed in seeds:
        fine = lab.load_oof_fine(source, "full", seed)
        if fine is None:
            continue
        stackable.append(fine[0])
        classes = list(fine[1])
    if not stackable:
        return []
    if [str(c) for c in classes] != [str(n) for n in names]:
        raise SystemExit("raster band order does not match the OOF classes; "
                         "refusing to guess the mapping")
    probs = np.mean(stackable, axis=0)
    print(f"map: {pixels.shape[0]:,} valid pixels x {len(names)} classes from "
          f"{prob_raster.name}; threshold calibrated on the {len(stackable)}-seed "
          f"mean OOF over {len(probs):,} plots")

    index = {c: i for i, c in enumerate(classes)}
    raw_idx = np.array([index.get(t, -1) for t in view.truth_fine])
    ok = raw_idx >= 0
    y = np.clip(raw_idx, 0, None)
    costs = load_costs(source, classes)

    rows = []
    for calib in calibrators:
            # ONE transform for both sides. The fold-wise fit used above is the
            # honest way to *score* a calibrator; it is not a thing that can be
            # shipped, because a raster has no folds. So the deployment instance
            # is fitted once on every labelled plot and applied to the plots
            # (to place the threshold) and to the pixels (to be thresholded)
            # identically -- anything else is train/serve skew in the threshold.
            if calib == "costgate":
                if costs is None:
                    continue
                plot_probs = probs * costs[None, :]
                map_probs = pixels * costs[None, :]
            elif calib == "temp":
                logits = torch.as_tensor(np.log(np.clip(probs[ok], 1e-12, None)),
                                         dtype=torch.float32)
                temperature = fit_temperature(logits, torch.as_tensor(y[ok]))
                plot_probs = np.exp(np.log(np.clip(probs, 1e-12, None))
                                    / temperature)
                map_probs = np.exp(
                    (np.log(np.clip(pixels, 1e-12, None)) / temperature)
                    - (np.log(np.clip(pixels, 1e-12, None)) / temperature
                       ).max(1, keepdims=True))
            else:
                plot_probs, map_probs = probs, pixels
            plot_probs = plot_probs / plot_probs.sum(1, keepdims=True)
            map_probs = map_probs / map_probs.sum(1, keepdims=True)

            cal_logits = torch.as_tensor(
                np.log(np.clip(plot_probs[ok], 1e-12, None)), dtype=torch.float32)
            cal_labels = torch.as_tensor(y[ok], dtype=torch.long)
            map_logits_t = torch.as_tensor(
                np.log(np.clip(map_probs, 1e-12, None)), dtype=torch.float32)

            for score_name in wanted_scores:
                for pred_name in wanted_predictors:
                    for alpha, seed in ((a, s) for a in alphas for s in seeds):
                        # `seed` replicates only the randomised scores' draw
                        # here -- the posteriors are already the seed mean.
                        torch.manual_seed(seed)
                        predictor = _prepare(
                            preds[pred_name](scores[score_name](), alpha),
                            map_probs.shape[1])
                        try:
                            predictor.calculate_threshold(cal_logits, cal_labels,
                                                          alpha)
                            sets = np.zeros(map_probs.shape, dtype=bool)
                            # RC3P materialises a (batch, K, K) rank tensor, so
                            # the chunk is sized for that, not for the logits.
                            for start in range(0, len(map_probs), 200_000):
                                stop = start + 200_000
                                block = predictor.predict_with_logits(
                                    map_logits_t[start:stop])
                                sets[start:stop] = np.asarray(
                                    block.cpu()).astype(bool)
                        except (ValueError, RuntimeError, IndexError):
                            continue
                        size = sets.sum(1)
                        rows.append(dict(
                            source=source, level="coarse3_map",
                            calibrator=calib, score=score_name,
                            predictor=pred_name, alpha=alpha, seed=seed,
                            set_size=float(size.mean()),
                            singleton_frac=float((size == 1).mean()),
                            empty_frac=float((size == 0).mean()),
                            size_p90=float(np.quantile(size, 0.90)),
                            size_max=int(size.max()),
                            # The product read: how much of the map the method
                            # is willing to call unambiguously, and how much it
                            # hands back as "one of these several transitions".
                            frac_le2=float((size <= 2).mean()),
                            n_pixels=int(len(size))))
    return rows


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def summarise(frame: pd.DataFrame, alpha: float) -> None:
    for level in ("merged2", "coarse3", "coarse3_map"):
        part = frame[(frame.level == level) & (frame.alpha == alpha)]
        if part.empty:
            continue
        keys = ["calibrator", "score", "predictor"]
        agg = part.groupby(keys).mean(numeric_only=True).reset_index()
        n_seeds = part.groupby(keys)["seed"].nunique().to_numpy()

        print(f"\n{'=' * 78}\n{level} @ alpha={alpha:.2f}   "
              f"(nominal coverage {1 - alpha:.2f}, {n_seeds.min()}-"
              f"{n_seeds.max()} seeds)\n{'=' * 78}")
        if level == "coarse3_map":
            cols = ["calibrator", "score", "predictor", "set_size",
                    "singleton_frac", "frac_le2", "empty_frac", "size_p90"]
            print("no truth on this raster -- efficiency only")
            print(agg[cols].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
            continue

        print("\n-- CALIBRATOR-OWNED (identical down every score x predictor "
              "block; that is the point) --")
        prob_cols = ["ece", "ece_classwise", "brier", "crps_ord",
                     "crps_nominal", "nll", "acc"]
        prob = agg.groupby("calibrator")[prob_cols].mean().reset_index()
        print(prob.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        print("\n-- CONFORMAL-METHOD-OWNED: coverage and efficiency --")
        cols = ["calibrator", "score", "predictor", "coverage", "cov_macro",
                "covgap", "sscv", "set_size", "singleton_frac", "singleton_hit",
                "empty_frac"]
        show = agg[cols].sort_values(["calibrator", "set_size"])
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        cov_cols = [c for c in agg.columns if c.startswith("cov_")
                    and c not in ("cov_macro",)]
        if cov_cols:
            print("\n-- PER-CLASS coverage (the number cov_macro and covgap "
                  "are made of) --")
            print(agg[["calibrator", "score", "predictor"] + cov_cols].to_string(
                index=False, float_format=lambda v: f"{v:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--levels", default="merged2,coarse3")
    parser.add_argument("--alphas", default="")
    parser.add_argument("--calibrators", default="raw,temp,costgate")
    parser.add_argument("--scores", default="lac,aps,raps,saps,margin,topk")
    parser.add_argument("--predictors", default="split,classwise,cluster,rc3p")
    parser.add_argument("--map-probs", default="",
                        help="coarse3 probability raster from infer_s2.py "
                             "--save-probs; adds the unlabelled efficiency read")
    parser.add_argument("--report-alpha", type=float, default=0.10)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    alphas = ([float(a) for a in args.alphas.split(",") if a]
              or list(lab.CONFORMAL_ALPHAS))
    levels = [s for s in args.levels.split(",") if s]
    calibrators = [s for s in args.calibrators.split(",") if s]
    wanted_scores = [s for s in args.scores.split(",") if s]
    wanted_predictors = [s for s in args.predictors.split(",") if s]

    ctx = lab.load_context()
    seeds = list(range(args.n_seeds))

    rows = []
    for level in levels:
        rows += run_level(ctx, args.source, level, seeds, alphas, calibrators,
                          wanted_scores, wanted_predictors)
    if args.map_probs:
        rows += map_efficiency(Path(args.map_probs), args.source, ctx, seeds,
                               alphas, calibrators, wanted_scores,
                               wanted_predictors)

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("nothing cached for the requested source")
        return
    frame.to_csv(args.out, index=False)
    summarise(frame, args.report_alpha)
    print(f"\n{len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
