"""Research harness for lifting the AlphaEarth+Tessera two-tower change-F1.

Everything the two-tower search needs on one footing: one data build, one set of
spatially blocked folds, one metric, one append-only ledger, and out-of-fold
merged2 probabilities cached to disk so post-hoc ideas (threshold tuning,
blending, stacking, distillation) can reuse a fit instead of repeating it.

Reads
-----
``full``    all ~6.4k plots -- the deploy metric. AlphaEarth is dense, Tessera is
            present for only ~36%, so this is what would actually ship and is the
            number every prior memory quotes (AlphaEarth-only = 0.660).
``subset``  the ~2.3k both-years-covered plots -- the only read where the fusion
            is actually exercised on every row (AlphaEarth-only = 0.661, the
            fused two-tower = 0.679-0.683). A win here is real signal; a win on
            ``full`` is real *deployment* value. Ideas are judged on both.

An idea is a function that returns out-of-fold merged2 probabilities for a read;
most are just a (columns, model-kwargs) pair, so ``model_idea`` builds those.
Ideas that need more (distillation across reads, stacking over cached OOF probs,
blending) get the full context and write their own CV loop.

Metrics are averaged over ``--n-seeds`` torch seeds because sub-1pt differences
here are inside seed noise (see memory: seed-average before believing a win).
``change_f1`` is the arg-max read -- directly comparable to every prior number.
``change_f1_bestt`` re-labels the same probabilities at the change-mass threshold
that maximises F1; it is an optimistic (tuned-on-OOF) upper bound, reported to
show the headroom in the operating point, never as the headline.

Usage::

    python twotower_lab.py --list
    python twotower_lab.py --ideas baseline_aef,tt_symmetric_md0.5
    python twotower_lab.py --all --n-seeds 3
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from experiment_hier_tessera import BASE, TESSERA, attach_tessera
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from model_zoo import (
    DEFAULT_INPUT,
    HierarchicalSoftmaxNN,
    is_change_label,
    make_splitter,
    to_merged_label,
)
from project_paths import project_data_dir
from twotower_metrics import extended_metrics, per_class

AEF_MASK = "aef_present"
MIN_COUNT = 20
N_SPLITS = 5
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

LEDGER = project_data_dir("analysis_results") / "twotower_lab_ledger.csv"
OOF_DIR = project_data_dir("analysis_results") / "twotower_lab_oof"


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
@dataclass
class View:
    """One read (``full`` or ``subset``): frame, target, folds and truths."""

    name: str
    frame: pd.DataFrame
    target: pd.Series
    folds: list
    truth_merged: np.ndarray
    truth_fine: np.ndarray
    merged_classes: list

    @property
    def tess_present(self) -> np.ndarray:
        """Row mask of real (both-year) Tessera -- the omission-split axis."""
        return self.frame["tess_present"].to_numpy() > 0.5


@dataclass
class Context:
    frame: pd.DataFrame
    aef_cols: list
    tess_cols: list
    views: dict = field(default_factory=dict)
    aef_scalars: list = field(default_factory=list)
    tess_scalars: list = field(default_factory=list)
    tess24_cols: list = field(default_factory=list)
    agree_scalars: list = field(default_factory=list)
    # Sentinel-2 VNIR detail modality (build_s2_features.py). Empty when the
    # feature table has not been built, so every Tessera idea still runs.
    s2_stat_cols: list = field(default_factory=list)
    s2_texture_cols: list = field(default_factory=list)
    s2_patch_cols: list = field(default_factory=list)
    s2_scalars: list = field(default_factory=list)
    s2_built_cols: list = field(default_factory=list)

    def view(self, read: str) -> View:
        return self.views[read]


def _make_view(name: str, frame: pd.DataFrame, aef_cols: list) -> View:
    frame = frame.reset_index(drop=True)
    target = target_for_legend(frame, LEGENDS["coarse3"], MIN_COUNT)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    folds = list(make_splitter("blocked", N_SPLITS).split(
        frame[aef_cols], target, frame["block_id"]))
    return View(name, frame, target, folds, truth_merged, truth_fine,
                sorted(set(truth_merged)))


def change_scalar_arrays(x1: np.ndarray, x2: np.ndarray, prefix: str):
    """The five change-magnitude scalars for an ``(n, d)`` endpoint pair.

    Array-level so training (a plot frame) and inference (a raster of pixels)
    compute them from one implementation -- a scalar defined twice is a
    train/serve skew waiting to happen.
    """
    d = x2 - x1
    n1 = np.linalg.norm(x1, axis=1)
    n2 = np.linalg.norm(x2, axis=1)
    cos = (x1 * x2).sum(1) / np.clip(n1 * n2, 1e-12, None)
    return {
        f"{prefix}_cosdist": 1.0 - cos,
        f"{prefix}_l2": np.linalg.norm(d, axis=1),
        f"{prefix}_l1": np.abs(d).sum(1),
        f"{prefix}_cheb": np.abs(d).max(1),
        f"{prefix}_dnorm": n2 - n1,
    }


def change_scalars(frame: pd.DataFrame, c18: list, c24: list, prefix: str):
    """Per-plot change-magnitude scalars for one modality's endpoint pair.

    ``diff+cos`` beat plain ``diff`` on AlphaEarth (0.6639 vs 0.6567) because a
    whole-vector magnitude is a different statistic from 64 per-band differences.
    Tessera never got the same treatment, so both modalities get the same five
    here. NaN where an endpoint is missing, which ``_prepare`` maps to the block
    mean behind the tower's gate.
    """
    cols = change_scalar_arrays(frame[c18].to_numpy("float64"),
                                frame[c24].to_numpy("float64"), prefix)
    out = pd.DataFrame(cols, index=frame.index)
    return out, list(out.columns)


S2_FEATURES = "s2_features_habloss_recover.parquet"
# Column-prefix families written by build_s2_features.py. The split matters: the
# `texture` families are the ones AlphaEarth structurally cannot carry (a smooth
# context embedding has no within-pixel heterogeneity), so an idea that wins on
# `texture` alone is evidence about *detail*, not just about extra features.
S2_TEXTURE_PREFIXES = ("S2s3_", "S2s9_", "S2s25_", "S2lc_", "S2g_")
# Built fraction at a calibrated NDVI cut -- F6's named lever for stable-Artificial.
S2_BUILT_PREFIXES = ("S2bf3_", "S2bf5_", "S2bf9_", "S2bf25_", "S2bf64_")
S2_STAT_PREFIXES = (("S2c_", "S2m3_", "S2m9_", "S2m25_")
                    + S2_TEXTURE_PREFIXES + S2_BUILT_PREFIXES)
S2_PATCH_PREFIX = "S2p_"
S2_MASK = "s2_present"


def s2_families(stat: list[str]) -> dict[str, list[str]]:
    """The Sentinel-2 stat block split into the families `build_s2_features` builds.

    One entry per stage of ``features_for_year``, so a family dropped here is a
    stage that would not be computed there. The full block is
    ``7 channels x 9 families + 5 built fractions = 68 per year``, times
    2018 / 2024 / diff = 204.
    """
    def fam(*prefixes):
        return [c for c in stat if c.startswith(prefixes)]

    return {
        "c": fam("S2c_"),                       # centre reflectance, 7 channels
        "m3": fam("S2m3_"), "m9": fam("S2m9_"), "m25": fam("S2m25_"),
        "s3": fam("S2s3_"), "s9": fam("S2s9_"), "s25": fam("S2s25_"),
        "lc": fam("S2lc_"),                     # centre minus 9 px mean
        "g": fam("S2g_"),                       # Sobel gradient
        "bf": fam("S2bf"),                      # built fraction, 5 radii
    }


#: Named detail-tower feature sets, ordered by how much of the block they keep.
#: The names are the point: each one is a sentence in a methods section, which
#: "204 engineered Sentinel-2 features" is not. Selected on map detail and IoU
#: against the full block by `compare_s2_subsets.py` -- see S18.
S2_SUBSETS: dict[str, tuple[str, ...]] = {
    "bf": ("bf",),
    "centre_bf": ("c", "bf"),
    "centre_s3_bf": ("c", "s3", "bf"),
    "centre_m3s3_bf": ("c", "m3", "s3", "bf"),
    "centre_3px_lc_bf": ("c", "m3", "s3", "lc", "bf"),
    "fine_scales_bf": ("c", "m3", "s3", "m9", "s9", "lc", "g", "bf"),
    "full": ("c", "m3", "m9", "m25", "s3", "s9", "s25", "lc", "g", "bf"),
}

#: One-line descriptions, for the methods section and the comparison table.
S2_SUBSET_DESC = {
    "bf": "built fraction at five radii",
    "centre_bf": "per-channel reflectance at the plot + built fraction",
    "centre_s3_bf": "per-channel reflectance and its 3 px standard deviation, "
                    "+ built fraction",
    "centre_m3s3_bf": "per-channel reflectance, 3 px mean and standard deviation, "
                      "+ built fraction",
    "centre_3px_lc_bf": "per-channel reflectance, its 3 px mean/std and local "
                        "contrast, + built fraction",
    "fine_scales_bf": "everything except the 25 px scale",
    "full": "all ten families at three scales (the deployed block)",
}


def s2_subset_columns(stat: list[str], name: str) -> list[str]:
    """Columns of a named subset, in the stat block's own order."""
    fams = s2_families(stat)
    wanted = set()
    for key in S2_SUBSETS[name]:
        wanted.update(fams[key])
    return [c for c in stat if c in wanted]


def attach_s2(frame: pd.DataFrame, s2_path: Path):
    """Join the Sentinel-2 detail features on by PLOTID; tolerate absence.

    Returns the frame plus the three column families. When the feature table has
    not been built the families come back empty and only the S2 ideas are
    unavailable -- the Tessera search must keep running unchanged.
    """
    if not Path(s2_path).exists():
        return frame, [], [], [], [], []
    s2 = pd.read_parquet(s2_path)
    stat = [c for c in s2.columns if c.startswith(S2_STAT_PREFIXES)]
    built = [c for c in s2.columns if c.startswith(S2_BUILT_PREFIXES)]
    texture = [c for c in s2.columns if c.startswith(tuple(S2_TEXTURE_PREFIXES))]
    patch = [c for c in s2.columns if c.startswith(S2_PATCH_PREFIX)]
    merged = frame.merge(s2, on="PLOTID", how="left", validate="many_to_one")
    # A plot with no S2 reading gets mask 0, which gates its tower off exactly
    # the way a missing Tessera tile does -- one mechanism for both modalities.
    merged[S2_MASK] = merged[S2_MASK].fillna(0.0).astype("float32")

    # The same five whole-vector change scalars the other modalities get, over
    # the per-band centre reflectance. `diff+cos` beat plain `diff` twice now.
    c18 = sorted(c for c in stat if c.startswith("S2c_") and c.endswith("_2018"))
    c24 = sorted(c for c in stat if c.startswith("S2c_") and c.endswith("_2024"))
    extra, names = change_scalars(merged, c18, c24, "s2S")
    merged = pd.concat([merged, extra], axis=1).copy()
    return merged, stat, texture, patch, names, built


def load_context(input_path: Path = DEFAULT_INPUT,
                 tess_path: Path | None = None,
                 s2_path: Path | None = None) -> Context:
    """Build the frame once: AlphaEarth + Tessera + S2 + masks + both reads' folds."""
    tess_path = tess_path or project_data_dir("embeddings", TESSERA)
    s2_path = s2_path or project_data_dir("embeddings", S2_FEATURES)
    frame, aef_cols = build_frame(input_path)
    frame, groups, _ = attach_tessera(frame, tess_path)
    tess_cols = groups["tess_2yr"]
    tess24 = groups["tess_2024"]

    aef18 = sorted(c for c in aef_cols if c.endswith("_2018"))
    aef24 = sorted(c for c in aef_cols if c.endswith("_2024"))
    te18 = sorted(c for c in tess_cols if c.endswith("_2018"))
    a_extra, a_names = change_scalars(frame, aef18, aef24, "aefS")
    t_extra, t_names = change_scalars(frame, te18, tess24, "tesS")

    # Cross-modal agreement: the two modalities have different dimensionalities,
    # so they are compared through their change *magnitudes* rather than
    # directly. Each modality's cosine change distance is rank-normalised over
    # the plots that have it (unsupervised, no label involved), and the signed
    # and absolute gaps say whether AlphaEarth and Tessera tell the same story
    # about this plot -- a per-plot proxy for how far this Tessera reading can be
    # trusted, which is exactly what the availability mask cannot express.
    ra = a_extra["aefS_cosdist"].rank(pct=True)
    rt = t_extra["tesS_cosdist"].rank(pct=True)
    agree = pd.DataFrame({
        "agree_signed": rt - ra,
        "agree_abs": (rt - ra).abs(),
        "agree_min": np.minimum(ra, rt),
    }, index=frame.index)
    agree_names = list(agree.columns)

    extra = pd.concat([a_extra, t_extra, agree], axis=1)
    # AlphaEarth covers every plot here; the mask column exists so the symmetric
    # two-tower can gate it (and so an eval masking can switch it off). The 2024
    # Tessera mask is its own thing: single-date Tessera reaches ~99% of plots
    # while the 2018 pair reaches 36%, so a 2024-only tower is nearly always on.
    extra[AEF_MASK] = 1.0
    extra["tess24_present"] = frame["_eff24"].astype("float32")
    frame = pd.concat([frame, extra], axis=1).copy()

    (frame, s2_stat, s2_texture, s2_patch, s2_names,
     s2_built) = attach_s2(frame, s2_path)

    ctx = Context(frame, aef_cols, tess_cols, aef_scalars=a_names,
                  tess_scalars=t_names, tess24_cols=tess24,
                  agree_scalars=agree_names, s2_stat_cols=s2_stat,
                  s2_texture_cols=s2_texture, s2_patch_cols=s2_patch,
                  s2_scalars=s2_names, s2_built_cols=s2_built)
    ctx.views["full"] = _make_view("full", frame, aef_cols)
    ctx.views["subset"] = _make_view(
        "subset", frame.loc[frame["_effboth"]].copy(), aef_cols)
    return ctx


# ---------------------------------------------------------------------------
# CV + metrics
# ---------------------------------------------------------------------------
def cv_probs(view: View, cols: list, kwargs: dict, seed: int,
             fit_frame_fn: Callable | None = None):
    """Blocked-CV out-of-fold merged2 probabilities and fine labels.

    Fold-local class lists are mapped into the view's global merged class order,
    so a fold that never sees a rare transition simply leaves that column at 0
    instead of silently shifting every column.
    """
    n = len(view.target)
    classes = view.merged_classes
    probs = np.zeros((n, len(classes)), dtype="float64")
    fine = np.empty(n, dtype=object)
    for tr, te in view.folds:
        frame = view.frame if fit_frame_fn is None else fit_frame_fn(view.frame, tr)
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame.iloc[tr], view.target.iloc[tr].to_numpy())
            p_fine, p_merged = model._probs(frame.iloc[te])
        idx = [classes.index(c) for c in model.merged_classes_]
        probs[np.ix_(te, idx)] = p_merged
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


def labels_from_probs(probs: np.ndarray, classes: list, threshold=None):
    """Merged2 labels at the arg-max (``threshold=None``) or a change gate."""
    arr = np.array(classes, dtype=object)
    if threshold is None:
        return arr[probs.argmax(1)]
    chg = np.array([is_change_label(c) for c in classes])
    if not chg.any() or chg.all():
        return arr[probs.argmax(1)]
    p_change = probs[:, chg].sum(1)
    pick_c = arr[chg][probs[:, chg].argmax(1)]
    pick_s = arr[~chg][probs[:, ~chg].argmax(1)]
    return np.where(p_change >= threshold, pick_c, pick_s)


def change_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Binary change-detection F1/recall/precision on the merged2 labels."""
    t = np.array([is_change_label(x) for x in truth])
    p = np.array([is_change_label(x) for x in pred])
    tp = int((t & p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"change_f1": f1, "change_precision": prec, "change_recall": rec}


def macro_f1(truth: np.ndarray, pred: np.ndarray, classes: list) -> float:
    """Unweighted mean merged2 F1 -- the objective for the cost-sensitive gate."""
    return float(per_class(truth, pred, classes)["macro_f1"])


def bal_acc(truth: np.ndarray, pred: np.ndarray) -> float:
    accs = [float((pred[truth == c] == c).mean()) for c in sorted(set(truth))]
    return float(np.mean(accs))


def score_probs(view: View, probs: np.ndarray, fine: np.ndarray | None,
                labels: np.ndarray | None = None) -> dict:
    """Every metric for one seed's OOF probabilities.

    ``labels`` overrides the arg-max read when an idea decides its own operating
    point (the nested gate); the threshold sweep still runs on the probabilities
    so the tuned-gate column stays comparable across every idea.

    Beyond the binary change columns this records the per-class merged2 metrics
    and the Tessera-availability split (see ``twotower_metrics``), because the
    two failures the deployed maps show -- stable Artificial read as stable
    Vegetation, and omission concentrated on the rows where Tessera fires -- are
    both invisible to change-F1.
    """
    pred = labels_from_probs(probs, view.merged_classes) if labels is None else labels
    out = extended_metrics(view.truth_merged, pred, view.merged_classes,
                           view.tess_present)
    out["bal_acc"] = bal_acc(view.truth_merged, pred)
    grid = [(t, change_metrics(view.truth_merged,
                               labels_from_probs(probs, view.merged_classes, t))["change_f1"])
            for t in THRESHOLD_GRID]
    best_t, best_f1 = max(grid, key=lambda kv: kv[1])
    out["best_t"] = float(best_t)
    out["change_f1_bestt"] = float(best_f1)
    out["fine_change_f1"] = (
        change_metrics(view.truth_fine, fine)["change_f1"] if fine is not None else np.nan)
    return out


# ---------------------------------------------------------------------------
# idea registry
# ---------------------------------------------------------------------------
@dataclass
class Idea:
    name: str
    fn: Callable
    reads: tuple
    group: str
    desc: str
    params: dict


IDEAS: dict[str, Idea] = {}


def register(name, *, reads=("full", "subset"), group="misc", desc="", params=None):
    def deco(fn):
        IDEAS[name] = Idea(name, fn, tuple(reads), group, desc, params or {})
        return fn
    return deco


def model_idea(name, *, cols_fn, kwargs_fn, reads=("full", "subset"), group="arch",
               desc="", params=None):
    """Register the common case: a column set + ``HierarchicalSoftmaxNN`` kwargs."""
    def fn(ctx, view, seed):
        return cv_probs(view, cols_fn(ctx), kwargs_fn(ctx), seed)
    return register(name, reads=reads, group=group, desc=desc, params=params)(fn)


def two_tower_kwargs(ctx, *, modality_dropout=0.5, fusion="additive",
                     aef_maskable=False, **extra):
    """The winning wide/focal/30-epoch recipe wearing a two-tower trunk."""
    kwargs = dict(
        arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"],
        aef_columns=ctx.aef_cols, tess_columns=ctx.tess_cols,
        mask_column="tess_present", modality_dropout=modality_dropout,
        tower_dim=256, fusion=fusion,
        aef_mask_column=AEF_MASK if aef_maskable else None,
    )
    kwargs.update(extra)  # e.g. a different tess block or a different mask
    return kwargs


# -- reference rungs ---------------------------------------------------------
model_idea(
    "baseline_aef",
    cols_fn=lambda c: c.aef_cols, kwargs_fn=lambda c: dict(BASE),
    group="baseline",
    desc="AlphaEarth-only wide/focal/30ep hier NN -- the incumbent deploy model.",
)

model_idea(
    "tt_additive_md0.5",
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(c, modality_dropout=0.5, fusion="additive"),
    group="baseline",
    desc="Plan B two-tower: always-on AlphaEarth base + mask-gated Tessera, md=0.5.",
)

model_idea(
    "tt_symmetric_md0.5",
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(c, modality_dropout=0.5, fusion="gated_mean",
                                         aef_maskable=True),
    group="baseline",
    desc="Symmetric two-tower: both towers mask-gated, gated_mean fusion, md=0.5.",
)

model_idea(
    "tess_only",
    cols_fn=lambda c: c.tess_cols, kwargs_fn=lambda c: dict(BASE),
    reads=("subset",), group="baseline",
    desc="Tessera-only hier NN on the covered subset -- the detail modality alone.",
)


# -- D. features -------------------------------------------------------------
model_idea(
    "tt_scalars",                                                        # D1
    cols_fn=lambda c: (c.aef_cols + c.aef_scalars + c.tess_cols + c.tess_scalars),
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        aef_columns=c.aef_cols + c.aef_scalars,
        tess_columns=c.tess_cols + c.tess_scalars),
    group="features",
    desc="D1: whole-vector change scalars (cos-dist, L2, L1, Chebyshev, norm change) "
         "added to BOTH towers -- diff+cos beat diff on AlphaEarth; Tessera never got it.",
)

model_idea(
    "tt_tess2024",                                                       # D2
    cols_fn=lambda c: c.aef_cols + c.tess24_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        tess_columns=c.tess24_cols, mask_column="tess24_present"),
    group="features",
    desc="D2: Tessera tower on the DENSE 2024 date only (99% coverage) instead of the "
         "sparse 2018/2024 pair -- trades change signal for a tower that is nearly always on.",
)

model_idea(
    "tt_tess2024_plus2yr",                                               # D2b
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        tess_columns=c.tess24_cols + c.tess_cols, mask_column="tess24_present"),
    group="features",
    desc="D2b: Tessera tower gated on the DENSE 2024 mask but carrying the 2018/diff bands "
         "too (mean-filled where absent) -- the tower fires on 99% of plots, richer where it can.",
)


# -- A / E. post-hoc combinations over cached or in-fold fits ----------------
def _fit(view, cols, kwargs, tr, seed):
    model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
    return model


def _place(probs, classes, model, te, p_merged):
    idx = [classes.index(c) for c in model.merged_classes_]
    probs[np.ix_(te, idx)] = p_merged


@register("router_covered", reads=("full",), group="harvest",
          desc="A4: hard router -- the fused two-tower predicts the covered plots, the "
               "AlphaEarth model predicts the rest. Banks the covered-subset gain "
               "without asking one model to serve both regimes.")
def router_covered(ctx, view, seed):
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    tt_kw = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                             aef_maskable=True)
    cols = ctx.aef_cols + ctx.tess_cols
    for tr, te in view.folds:
        covered_tr = tr[view.frame["_effboth"].to_numpy()[tr]]
        aef_m = _fit(view, ctx.aef_cols, dict(BASE), tr, seed)
        tt_m = _fit(view, cols, tt_kw, covered_tr, seed)
        covered_te = view.frame["_effboth"].to_numpy()[te]
        for model, sel in ((aef_m, ~covered_te), (tt_m, covered_te)):
            if not sel.any():
                continue
            rows = te[sel]
            p_fine, p_merged = model._probs(view.frame.iloc[rows])
            _place(probs, classes, model, rows, p_merged)
            fine[rows] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


@register("tta_modality_views", group="ensemble",
          desc="E2: test-time averaging over the mask-gated views (both / AlphaEarth-only / "
               "Tessera-only) of ONE symmetric two-tower -- a free ensemble the gating "
               "already supports, averaged only over the views a plot actually has.")
def tta_modality_views(ctx, view, seed):
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    kw = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                          aef_maskable=True)
    cols = ctx.aef_cols + ctx.tess_cols
    for tr, te in view.folds:
        model = _fit(view, cols, kw, tr, seed)
        te_frame = view.frame.iloc[te]
        has_tess = te_frame["tess_present"].to_numpy() > 0.5
        acc = np.zeros((len(te), len(classes)))
        n_views = np.zeros(len(te))
        fine_acc = None
        for aef_on, tess_on in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0)):
            v = te_frame.copy()
            v[AEF_MASK] = aef_on
            v["tess_present"] = v["tess_present"] * tess_on
            # A Tessera-only view is meaningless on a plot with no Tessera.
            usable = np.ones(len(te), bool) if aef_on else has_tess
            p_fine, p_merged = model._probs(v)
            idx = [classes.index(c) for c in model.merged_classes_]
            block = np.zeros((len(te), len(classes)))
            block[:, idx] = p_merged
            acc[usable] += block[usable]
            n_views[usable] += 1
            if aef_on and tess_on:
                fine_acc = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
        probs[te] = acc / n_views[:, None]
        fine[te] = fine_acc
    return probs, fine


def _stack(ctx, view, seed, bases):
    """Logistic meta-learner over cached OOF probabilities + the coverage mask."""
    from sklearn.linear_model import LogisticRegression

    if view.name == "subset" and "tess_only" not in bases:
        bases = bases + ["tess_only"]
    loaded = [load_oof(b, view.name, seed) for b in bases]
    if any(x is None for x in loaded):
        raise RuntimeError(f"stack_oof needs cached OOF for {bases} on "
                           f"{view.name} seed {seed} -- run those ideas first")
    mask = view.frame["tess_present"].to_numpy("float64").reshape(-1, 1)
    X = np.hstack([p for p, _ in loaded] + [mask])
    y = view.truth_merged
    classes = view.merged_classes
    probs = np.zeros((len(y), len(classes)))
    for tr, te in view.folds:
        meta = LogisticRegression(max_iter=2000, C=1.0)
        meta.fit(X[tr], y[tr])
        cols = [classes.index(c) for c in meta.classes_]
        probs[np.ix_(te, cols)] = meta.predict_proba(X[te])
    # The fine read is inherited from the strongest base -- the meta-learner
    # works at the merged2 (deploy) level only.
    fine = np.asarray(loaded[1][1], dtype=object)
    return probs, fine


register("stack_oof", group="ensemble",
         desc="E3: logistic meta-learner over the cached out-of-fold merged2 probabilities "
              "of the AlphaEarth model and the symmetric two-tower, with the Tessera "
              "availability mask as a meta-feature -- lets the blend depend on coverage."
         )(lambda ctx, view, seed: _stack(
             ctx, view, seed, ["baseline_aef", "tt_symmetric_md0.5"]))

WIDE_BASES = ["baseline_aef", "tt_symmetric_md0.5", "tt_scalars", "tt_film",
              "tt_learned_gate"]

WIDE2_BASES = WIDE_BASES + ["tt_tessdrop0.6", "tt_tessdrop0.7", "tt_tess_pca32"]

register("stack_wide2", group="ensemble",
         desc="E3c: the diverse stack extended with the asymmetric-dropout towers and the "
              "PCA-projected tower -- the members that actually beat the baseline on their "
              "own, not just members that differ from it."
         )(lambda ctx, view, seed: _stack(ctx, view, seed, WIDE2_BASES))

register("stack_wide", group="ensemble",
         desc="E3b: the same stack over a deliberately DIVERSE base set -- AlphaEarth-only, "
              "the mask-gated two-tower, the change-scalar tower, the FiLM-conditioned tower "
              "and the learned-reliability tower. Individually within noise of each other, "
              "but they make different mistakes, which is what a stack monetises."
         )(lambda ctx, view, seed: _stack(ctx, view, seed, WIDE_BASES))


# -- A1. cross-modal distillation --------------------------------------------
def _distil(ctx, view, seed, *, teacher_on_covered: bool, weight: float,
            temperature: float):
    """Fused teacher -> AlphaEarth-only student, refit inside every fold.

    The teacher is the symmetric two-tower (it sees Tessera); the student sees
    only AlphaEarth and so can score every plot. Distilling the teacher's merged2
    posteriors into the student is the one route that moves Tessera's contribution
    onto the 64% of plots that have no Tessera vector of their own -- the teacher's
    *soft* ranking of a plot carries detail the hard label does not.

    Both teacher variants are fold-safe: the teacher only ever sees training rows,
    and the student is scored out of fold as usual.
    """
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    tt_kw = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                             aef_maskable=True)
    cols = ctx.aef_cols + ctx.tess_cols
    covered = view.frame["_effboth"].to_numpy()
    for tr, te in view.folds:
        teach_rows = tr[covered[tr]] if teacher_on_covered else tr
        teacher = _fit(view, cols, tt_kw, teach_rows, seed)
        _, q = teacher._probs(view.frame.iloc[tr])
        soft = pd.DataFrame(q, columns=teacher.merged_classes_, index=tr)
        # Only distil where the teacher had real Tessera to be a teacher about.
        soft.loc[~covered[tr]] = np.nan

        student = HierarchicalSoftmaxNN(
            ctx.aef_cols, seed=seed, distill_weight=weight,
            distill_temperature=temperature, **BASE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pass the frame, not its values: fit aligns the teacher's columns to
            # the student's merged class order BY NAME.
            student.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy(),
                        soft_merged=soft)
            p_fine, p_merged = student._probs(view.frame.iloc[te])
        _place(probs, classes, student, te, p_merged)
        fine[te] = np.array(student.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


def distil_idea(name, *, teacher_on_covered, weight, temperature, desc):
    return register(name, group="harvest", desc=desc)(
        lambda ctx, view, seed: _distil(
            ctx, view, seed, teacher_on_covered=teacher_on_covered,
            weight=weight, temperature=temperature))


distil_idea("distil_w1", teacher_on_covered=True, weight=1.0, temperature=1.0,
            desc="A1: AlphaEarth-only student distilled from a two-tower teacher trained on "
                 "the covered plots -- moves Tessera's contribution onto plots that have none.")
distil_idea("distil_w3_T2", teacher_on_covered=True, weight=3.0, temperature=2.0,
            desc="A1: same, weight 3 and temperature 2 -- a louder, softer teacher (more of "
                 "its dark knowledge, less of its arg-max).")
distil_idea("distil_fullteacher", teacher_on_covered=False, weight=1.0,
            temperature=1.0,
            desc="A1: student distilled from a two-tower teacher trained on ALL training "
                 "plots (mask-gated), not just the covered ones -- more teacher data, "
                 "weaker Tessera signal per row.")


# -- C. treat Tessera as the noisy modality ----------------------------------
model_idea(
    "tt_tessdrop0.6",                                                    # C2
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        dropout_tess=0.6),
    group="noisy-modality",
    desc="C2: asymmetric regularisation -- heavier dropout (0.6 vs 0.4) inside the Tessera "
         "tower only, on the theory that the noisier modality should be regularised harder.",
)

model_idea(
    "tt_tessdrop0.2",
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        dropout_tess=0.2),
    group="noisy-modality",
    desc="C2 control: LIGHTER dropout (0.2) in the Tessera tower -- the other direction, to "
         "tell a real asymmetry effect from a dropout-rate effect.",
)


def _pca_frame_fn(ctx, n_components):
    """Fold-safe PCA of the Tessera block, fitted on covered TRAIN rows only."""
    def fn(frame, tr):
        from sklearn.decomposition import PCA

        cols = ctx.tess_cols
        X = frame[cols].to_numpy("float64")
        covered = frame["_effboth"].to_numpy()
        fit_rows = np.zeros(len(frame), bool)
        fit_rows[tr] = True
        fit_rows &= covered
        pca = PCA(n_components=n_components, random_state=0)
        pca.fit(np.nan_to_num(X[fit_rows]))
        Z = pca.transform(np.nan_to_num(X))
        Z[~covered] = np.nan          # absent stays absent; the mask still gates it
        names = [f"TP{i:03d}" for i in range(n_components)]
        return pd.concat([frame.drop(columns=[c for c in names if c in frame]),
                          pd.DataFrame(Z, columns=names, index=frame.index)], axis=1)
    return fn


def pca_idea(name, n_components, desc):
    def fn(ctx, view, seed):
        names = [f"TP{i:03d}" for i in range(n_components)]
        kw = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                              aef_maskable=True, tess_columns=names)
        return cv_probs(view, ctx.aef_cols + names, kw, seed,
                        fit_frame_fn=_pca_frame_fn(ctx, n_components))
    return register(name, group="noisy-modality", desc=desc)(fn)


pca_idea("tt_tess_pca64", 64,                                            # C3
         "C3: Tessera projected to 64 PCA components (fitted on covered train rows only) "
         "before the tower -- 384 raw dims on 2.3k covered plots is the overfitting regime, "
         "and the leading components should keep signal while dropping per-band noise.")
pca_idea("tt_tess_pca32", 32,
         "C3: Tessera projected to 32 PCA components -- a harder squeeze.")

def tt_variant(name, desc, *, scalars=False, md=0.5, group="noisy-modality", **kw):
    """Register a two-tower variant on the winning asymmetric-dropout recipe.

    Every knob defaults to the symmetric ``gated_mean`` two-tower, so a variant
    differs from the tested incumbent only in what it names.
    """
    cols = ((lambda c: c.aef_cols + c.aef_scalars + c.tess_cols + c.tess_scalars)
            if scalars else (lambda c: c.aef_cols + c.tess_cols))
    blocks = ((lambda c: dict(aef_columns=c.aef_cols + c.aef_scalars,
                              tess_columns=c.tess_cols + c.tess_scalars))
              if scalars else (lambda c: {}))
    return model_idea(
        name, cols_fn=cols,
        kwargs_fn=lambda c: two_tower_kwargs(
            c, fusion="gated_mean", aef_maskable=True, modality_dropout=md,
            **blocks(c), **kw),
        group=group, desc=desc)


tt_variant("tt_drop0.7_w0.5", tess_width=0.5, dropout_tess=0.7,
           desc="C2b: heavier dropout AND half-width Tessera tower -- tests whether the win "
                "is regularisation specifically or capacity reduction generally.")
tt_variant("tt_drop0.7_w0.25", tess_width=0.25, dropout_tess=0.7,
           desc="C2b: quarter-width Tessera tower at dropout 0.7 -- a hard capacity squeeze.")
tt_variant("tt_w0.25", tess_width=0.25,
           desc="C2b control: quarter-width Tessera tower at the DEFAULT dropout 0.4 -- "
                "isolates capacity from dropout.")
tt_variant("tt_drop0.7_md0.3", dropout_tess=0.7, md=0.3,
           desc="C2c: does the Tessera-tower dropout optimum shift the modality-dropout "
                "optimum? Less modality dropout, more within-tower dropout.")
tt_variant("tt_drop0.7_md0.7", dropout_tess=0.7, md=0.7,
           desc="C2c: more modality dropout on top of the heavier within-tower dropout.")
tt_variant("tt_drop0.7_scalars", dropout_tess=0.7, scalars=True,
           desc="D1+C2: the change scalars under the winning asymmetric dropout -- the test "
                "of whether the two independent gains compose.")
for _w in (0.1, 0.5):                                                    # A3
    tt_variant(f"tt_align{_w}", dropout_tess=0.7, align_weight=_w,
               desc=f"A3: CLIP-style InfoNCE between the two towers (weight {_w}) on rows "
                    "where both modalities are real -- pulls the AlphaEarth tower toward the "
                    "Tessera manifold so it carries some detail structure where Tessera is absent.")


# -- S. Sentinel-2 VNIR as the detail modality -------------------------------
# Tessera's ceiling was never architectural: 2018 tiles reach 35.8% of plots and
# every idea in sections A-E was an attempt to spend a gain confined to a third
# of the data. Raw S2 L2A has no such hole -- both endpoints, every plot, 10 m,
# four bands, free off the AWS open-data COGs. The tower slot is identical, so
# these ideas are the Tessera ideas with the modality swapped, which is what
# makes them comparable to the whole existing ledger.
def s2_variant(name, desc, *, family="stat", scalars=False, md=0.5,
               group="s2-detail", **kw):
    """Register a two-tower variant whose detail tower is Sentinel-2, not Tessera.

    ``family`` selects the feature block: ``stat`` (centre + neighbourhood mean
    and std + local contrast + gradient), ``texture`` (only what AlphaEarth
    cannot express), ``patch`` (the 8x8 mean-pooled image), or ``all``.
    """
    def detail(c):
        block = {"stat": c.s2_stat_cols, "texture": c.s2_texture_cols,
                 "patch": c.s2_patch_cols,
                 "all": c.s2_stat_cols + c.s2_patch_cols}[family]
        return block + (c.s2_scalars if scalars else [])

    return model_idea(
        name,
        cols_fn=lambda c: c.aef_cols + (c.aef_scalars if scalars else []) + detail(c),
        kwargs_fn=lambda c: two_tower_kwargs(
            c, fusion="gated_mean", aef_maskable=True, modality_dropout=md,
            aef_columns=c.aef_cols + (c.aef_scalars if scalars else []),
            tess_columns=detail(c), mask_column=S2_MASK, **kw),
        group=group, desc=desc)


model_idea(
    "s2_only",
    cols_fn=lambda c: c.s2_stat_cols, kwargs_fn=lambda c: dict(BASE),
    group="s2-detail",
    desc="S0: Sentinel-2 detail features alone on the wide/focal hier NN -- does raw VNIR "
         "carry the transition signal at all, before any fusion question is asked?",
)

s2_variant("tt_s2_stat",
           desc="S1: the headline candidate -- AlphaEarth context tower + a Sentinel-2 tower "
                "over centre/neighbourhood/contrast/gradient features, gated on a mask that "
                "is ~1 everywhere instead of Tessera's 0.358.")
s2_variant("tt_s2_texture", family="texture",
           desc="S2: the detail hypothesis in isolation -- the S2 tower sees ONLY within-"
                "neighbourhood std, local contrast and edge gradient, none of which a smooth "
                "context embedding can represent. A win here is evidence about detail, not "
                "about extra features.")
s2_variant("tt_s2_patch", family="patch",
           desc="S3: give the tower the 8x8 mean-pooled image instead of hand-built "
                "statistics and let it learn its own texture.")
s2_variant("tt_s2_drop0.7", dropout_tess=0.7,
           desc="S4: transfer C2, the one architectural lever that ever moved this metric -- "
                "heavier dropout on the detail tower (0.7) than the context tower (0.4). S2 "
                "is dense where Tessera was sparse, so the optimum may well move.")
s2_variant("tt_s2_scalars", scalars=True, dropout_tess=0.7,
           desc="S5: D1+C2 transferred -- per-modality change scalars on both towers under "
                "the asymmetric dropout, the composition that gave the deployed model.")


model_idea(
    "aef_builtfrac",                                                     # S8
    cols_fn=lambda c: c.aef_cols + c.s2_built_cols,
    kwargs_fn=lambda c: dict(BASE),
    group="s2-detail",
    desc="S8: AlphaEarth + the NDVI built-fraction covariate ONLY (5 radii x 2 years + "
         "diff), flat -- no second tower. F6 closed section F saying the lever for "
         "stable-Artificial was 'a built-fraction covariate or a sub-pixel label, not "
         "another fusion'; this is the cheapest possible test of that sentence, and it "
         "avoids the gated tower that S1/S2t/S4 showed suppresses change.")


# -- S10/S11: the two knobs mc_s2_drop0.7 inherited from Tessera, never swept --
# The user ranks this model best on the map. Both knobs control the same thing --
# how far the noisy detail tower is trusted -- and neither was ever tuned for
# Sentinel-2, which is denser and cleaner than the Tessera it was copied from.
for _d in (0.4, 0.5, 0.6, 0.8):
    register(f"mc_s2_drop{_d}", group="s2-detail",
             desc=f"S10: detail-tower dropout {_d} under the MC gate. C2's dose-response "
                  "was the single biggest architectural lever on Tessera and 0.7 was "
                  "simply carried over; S2 is denser and less noisy, so its optimum "
                  "may sit lower.")(
        (lambda d: (lambda ctx, view, seed: _mc_s2(ctx, view, seed, dropout_tess=d)))(_d))

for _k in (0.3, 0.7, 0.9):
    register(f"mc_s2_keep{_k}", group="s2-detail",
             desc=f"S11: MC keep-probability {_k} at dropout 0.7 -- how often the detail "
                  "tower is trusted per pass. 1.0 is the deterministic gate that "
                  "suppressed change, 0.0 is AlphaEarth alone; 0.5 was inherited, not "
                  "chosen. This is the detail/suppression dial the map responds to.")(
        (lambda k: (lambda ctx, view, seed: _mc_s2(ctx, view, seed, keep_prob=k,
                                                   dropout_tess=0.7)))(_k))


def _mc_s2_split(ctx, view, seed, **extra):
    """The two S2 wins composed, with each feature family on the tower it suits.

    `aef_builtfrac` (N4) put the NDVI built fraction flat beside AlphaEarth and
    gave the lowest `art_stable_as_veg` on the board (0.192); `mc_s2_drop0.7`
    (S6) put the S2 texture behind a stochastic gate and gave the best S2
    change-F1 (0.6656). They target different errors, so composing them is worth
    a test -- but F7 is the warning: the seed ensemble and the cost gate also
    looked independent and turned out to correct the same under-confidence, so
    their composition landed *between* its parts rather than above them.

    The split is principled rather than arbitrary. Built fraction is dense
    (99.3%), physically interpretable and reliable, so it rides the **always-on**
    AlphaEarth tower where it is never gated away. Texture is the noisy detail
    that S1/S2t/S4 showed must not be trusted deterministically, so it stays on
    the **stochastically gated** tower. Putting the covariate behind the gate
    would throw away half its evidence on every MC pass.
    """
    built = ctx.s2_built_cols
    detail = [c for c in ctx.s2_stat_cols if c not in set(built)]
    aef = ctx.aef_cols + built
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    kwargs = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                              aef_maskable=True, aef_columns=aef,
                              tess_columns=detail, mask_column=S2_MASK, **extra)
    rng = np.random.default_rng(seed)
    for tr, te in view.folds:
        model = _fit(view, aef + detail, kwargs, tr, seed)
        te_frame = view.frame.iloc[te]
        has_s2 = te_frame[S2_MASK].to_numpy() > 0.5
        acc = np.zeros((len(te), len(classes)))
        for _ in range(16):
            v = te_frame.copy()
            # is literally the detail/noise dial: 1.0 is the deterministic gate
            # that suppressed change, 0.0 is AlphaEarth alone. 0.5 was inherited
            # from the Tessera recipe and has never been swept for S2, which is
            # a denser and cleaner modality and may well want to be trusted more.
            keep = rng.random(len(te)) >= 0.5
            v[S2_MASK] = np.where(has_s2 & keep, 1.0, 0.0)
            _, p_merged = model._probs(v)
            block = np.zeros((len(te), len(classes)))
            block[:, [classes.index(c) for c in model.merged_classes_]] = p_merged
            acc += block
        probs[te] = acc / 16
        p_fine, _ = model._probs(te_frame)
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


register("mc_s2_bf", group="s2-detail",
         desc="S9: compose the two S2 wins -- NDVI built fraction flat on the always-on "
              "AlphaEarth tower (N4's lever, best art->veg 0.192) plus S2 texture behind "
              "the stochastic gate (S6's lever, best S2 change-F1 0.6656). Tests whether "
              "they add or, like F7's pair, merely overlap.")(
    lambda ctx, view, seed: _mc_s2_split(ctx, view, seed, dropout_tess=0.7))


def _mc_s2(ctx, view, seed, scalars=False, keep_prob=0.5, passes=16, **extra):
    """E4 transferred to the Sentinel-2 tower, aimed at the measured failure mode.

    S1/S2/S4 all lose change-F1 the same way: recall collapses (0.713 -> 0.63-0.65)
    while precision rises. The gated detail tower makes the model conservative
    about change. The ledger already contains the antidote for exactly that
    symptom on the Tessera tower -- `mc_dropout_scalars` carries change recall
    0.7146 where its own deterministic parent `tt_symmetric_md0.5` sits at 0.6899,
    i.e. keeping the gate stochastic at test time and averaging recovers the
    recall the gate suppresses. If that mechanism is general, it should transfer;
    if it does not, the suppression is a property of what S2 *says*, not of how
    the gate reads it -- which is the same distinction G1 drew for Tessera.
    """
    detail = ctx.s2_stat_cols + (ctx.s2_scalars if scalars else [])
    aef = ctx.aef_cols + (ctx.aef_scalars if scalars else [])
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    kwargs = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                              aef_maskable=True, aef_columns=aef,
                              tess_columns=detail, mask_column=S2_MASK, **extra)
    rng = np.random.default_rng(seed)
    for tr, te in view.folds:
        model = _fit(view, aef + detail, kwargs, tr, seed)
        te_frame = view.frame.iloc[te]
        has_s2 = te_frame[S2_MASK].to_numpy() > 0.5
        acc = np.zeros((len(te), len(classes)))
        for _ in range(passes):
            v = te_frame.copy()
            # keep_prob is how often the detail tower is trusted on a pass, so it
            # is literally the detail/suppression dial: 1.0 is the deterministic
            # gate that cost change recall, 0.0 is AlphaEarth alone. 0.5 came
            # across from the Tessera recipe and was never chosen for S2.
            keep = rng.random(len(te)) < keep_prob
            v[S2_MASK] = np.where(has_s2 & keep, 1.0, 0.0)
            _, p_merged = model._probs(v)
            block = np.zeros((len(te), len(classes)))
            block[:, [classes.index(c) for c in model.merged_classes_]] = p_merged
            acc += block
        probs[te] = acc / passes
        p_fine, _ = model._probs(te_frame)
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


register("mc_s2_dropout", group="s2-detail",
         desc="S6: E4 on the Sentinel-2 tower. S1/S2/S4 all fail by suppressing change "
              "RECALL (0.713 -> 0.63-0.65) while precision rises; MC modality dropout is "
              "the one lever already shown to reverse that exact symptom on Tessera "
              "(recall 0.6899 -> 0.7146). Tests whether the suppression is the gate or "
              "the modality.")(lambda ctx, view, seed: _mc_s2(ctx, view, seed))

register("mc_s2_drop0.7", group="s2-detail",
         desc="S6b: as mc_s2_dropout with the asymmetric detail-tower dropout (0.7) that "
              "was the only architectural lever ever to move this metric.")(
    lambda ctx, view, seed: _mc_s2(ctx, view, seed, dropout_tess=0.7))


def _mc_idea(name, scalars, desc, group="ensemble", **extra):
    def fn(ctx, view, seed):
        return _mc(ctx, view, seed, scalars, **extra)
    return register(name, group=group, desc=desc)(fn)


@register("mc_modality_dropout", group="ensemble",                        # E4
          desc="E4: Monte-Carlo modality dropout -- keep the gate stochastic at test time and "
               "average 16 passes, so the prediction integrates over 'trust Tessera' and "
               "'ignore Tessera' instead of committing to one.")
def mc_modality_dropout(ctx, view, seed):
    return _mc(ctx, view, seed, scalars=False)


def _mc(ctx, view, seed, scalars, **extra):
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    blocks = dict(aef_columns=ctx.aef_cols + ctx.aef_scalars,
                  tess_columns=ctx.tess_cols + ctx.tess_scalars) if scalars else {}
    kw = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                          aef_maskable=True, dropout_tess=0.7, **blocks, **extra)
    cols = ((ctx.aef_cols + ctx.aef_scalars + ctx.tess_cols + ctx.tess_scalars)
            if scalars else ctx.aef_cols + ctx.tess_cols)
    rng = np.random.default_rng(seed)
    for tr, te in view.folds:
        model = _fit(view, cols, kw, tr, seed)
        te_frame = view.frame.iloc[te]
        has_tess = te_frame["tess_present"].to_numpy() > 0.5
        acc = np.zeros((len(te), len(classes)))
        for _ in range(16):
            v = te_frame.copy()
            # Drop the Tessera gate at random on the rows that have it; every row
            # keeps AlphaEarth, so no row is ever left with nothing.
            keep = rng.random(len(te)) >= 0.5
            v["tess_present"] = np.where(has_tess & keep, 1.0, 0.0)
            _, p_merged = model._probs(v)
            block = np.zeros((len(te), len(classes)))
            block[:, [classes.index(c) for c in model.merged_classes_]] = p_merged
            acc += block
        probs[te] = acc / 16
        p_fine, _ = model._probs(te_frame)
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


_mc_idea("mc_dropout_scalars", True,
         "E4+D1+C2: Monte-Carlo modality dropout on the change-scalar, asymmetric-dropout "
         "two-tower -- the three independent full-set gains stacked into one model.")


def _hallucinate_fn(ctx):                                                 # A2
    """Fold-safe ridge imputation of Tessera from AlphaEarth, with a synthetic flag.

    Fitted on covered TRAIN rows only, applied to the uncovered ones, and the
    Tessera mask is switched on for the imputed rows so the tower actually fires
    -- the whole point is to give the 64% of plots with no Tessera *something*
    for the detail tower to read. ``tess_synthetic`` lets the network learn how
    far to trust a hallucinated vector.
    """
    def fn(frame, tr):
        from sklearn.linear_model import Ridge

        covered = frame["_effboth"].to_numpy()
        fit_rows = np.zeros(len(frame), bool)
        fit_rows[tr] = True
        fit_rows &= covered
        X = frame[ctx.aef_cols].to_numpy("float64")
        Y = np.nan_to_num(frame[ctx.tess_cols].to_numpy("float64"))
        model = Ridge(alpha=10.0).fit(X[fit_rows], Y[fit_rows])
        pred = model.predict(X)
        out = frame.copy()
        block = np.array(out[ctx.tess_cols].to_numpy("float64"), copy=True)
        block[~covered] = pred[~covered]
        out[ctx.tess_cols] = block
        out["tess_present"] = 1.0
        out["tess_synthetic"] = (~covered).astype("float32")
        return out
    return fn


@register("tt_hallucinated", group="harvest",
          desc="A2: Tessera regressed from AlphaEarth on covered plots and imputed for the "
               "rest, mask switched on, with a `tess_synthetic` flag -- gives the detail "
               "tower an input on every plot instead of gating it off for 64% of them.")
def tt_hallucinated(ctx, view, seed):
    tess = ctx.tess_cols + ["tess_synthetic"]
    kw = two_tower_kwargs(ctx, modality_dropout=0.5, fusion="gated_mean",
                          aef_maskable=True, dropout_tess=0.7, tess_columns=tess)
    return cv_probs(view, ctx.aef_cols + tess, kw, seed,
                    fit_frame_fn=_hallucinate_fn(ctx))


model_idea(                                                              # C4
    "tt_xmodal_agree",
    cols_fn=lambda c: (c.aef_cols + c.aef_scalars + c.agree_scalars
                       + c.tess_cols + c.tess_scalars),
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        dropout_tess=0.7,
        aef_columns=c.aef_cols + c.aef_scalars + c.agree_scalars,
        tess_columns=c.tess_cols + c.tess_scalars),
    group="noisy-modality",
    desc="C4: cross-modal agreement scalars -- the rank-normalised gap between the AlphaEarth "
         "and Tessera change magnitudes, a per-plot proxy for how far this Tessera reading can "
         "be trusted. Carried on the always-present AlphaEarth tower, so it informs the model "
         "even on rows where the Tessera tower is gated off.",
)

tt_variant("tt_drop0.65", dropout_tess=0.65,
           desc="C2: filling in the dropout dose-response between the subset optimum (0.6) "
                "and the deploy optimum (0.7).")


for _p in (0.7, 0.8):
    model_idea(
        f"tt_tessdrop{_p}",
        cols_fn=lambda c: c.aef_cols + c.tess_cols,
        kwargs_fn=(lambda p: lambda c: two_tower_kwargs(
            c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
            dropout_tess=p))(_p),
        group="noisy-modality",
        desc=f"C2: Tessera-tower dropout {_p} -- pushing the asymmetry further to find "
             "where over-regularising the detail modality starts to cost signal.",
    )


# -- E1. honest operating point ----------------------------------------------
def nested_gate(view: View, probs: np.ndarray) -> np.ndarray:
    """Labels whose change gate is chosen without ever seeing the fold it scores.

    For each outer fold the threshold that maximises change-F1 on the *other*
    folds' out-of-fold predictions is selected, then applied to the held-out
    fold. This is the honest version of the ``change_f1_bestt`` column -- that one
    picks its threshold on the same rows it reports, which flatters any model
    whose probabilities are mis-centred but well-ordered.
    """
    labels = np.empty(len(view.truth_merged), dtype=object)
    for tr, te in view.folds:
        best_t, best_f1 = None, -1.0
        for t in THRESHOLD_GRID:
            f1 = change_metrics(
                view.truth_merged[tr],
                labels_from_probs(probs[tr], view.merged_classes, t))["change_f1"]
            if f1 > best_f1:
                best_f1, best_t = f1, t
        labels[te] = labels_from_probs(probs[te], view.merged_classes, best_t)
    return labels


def gate_idea(name, source, desc):
    """Register the nested-gate re-read of an already-cached idea's OOF probs."""
    def fn(ctx, view, seed):
        cached = load_oof(source, view.name, seed)
        if cached is None:
            raise RuntimeError(f"gate_{source} needs cached OOF for {source} on "
                               f"{view.name} seed {seed} -- run it first")
        probs, fine = cached
        return probs, fine, nested_gate(view, probs)
    return register(name, group="operating-point", desc=desc)(fn)


gate_idea("gate_baseline_aef", "baseline_aef",
          "E1: AlphaEarth-only model read at a nested-CV-selected change gate "
          "instead of the implicit 0.5 arg-max.")
gate_idea("gate_tt_symmetric", "tt_symmetric_md0.5",
          "E1: symmetric two-tower read at a nested-CV-selected change gate.")
gate_idea("gate_tt_scalars", "tt_scalars",
          "E1: the change-scalar two-tower read at a nested-CV-selected change gate.")
gate_idea("gate_stack_wide2", "stack_wide2",
          "E1+E3c: the extended diverse stack read at a nested-CV-selected change gate.")
gate_idea("gate_stack_wide", "stack_wide",
          "E1+E3b: the diverse stack read at a nested-CV-selected change gate.")
gate_idea("gate_stack_oof", "stack_oof",
          "E1: the stacked blend read at a nested-CV-selected change gate -- the test of "
          "whether its strong ranking survives an honestly chosen operating point.")


# -- B. fusion that lets context arbitrate detail ----------------------------
model_idea(
    "tt_learned_gate",                                                   # B2
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        tess_gate="learned"),
    group="fusion",
    desc="B2: per-plot Tessera reliability gate, sigma(MLP([rep_aef, rep_tess])), on top of "
         "the availability mask -- lets the model discount a noisy Tessera vector instead "
         "of trusting every present one equally.",
)

model_idea(
    "tt_film",                                                           # B1
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="film", aef_maskable=True),
    group="fusion",
    desc="B1: FiLM conditioning -- the AlphaEarth context emits per-channel (gamma, beta) "
         "that modulate the Tessera representation before fusion, so context changes how "
         "detail is read, not just how much of it is added.",
)

model_idea(
    "tt_film_learned_gate",                                              # B1+B2
    cols_fn=lambda c: c.aef_cols + c.tess_cols,
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="film", aef_maskable=True,
        tess_gate="learned"),
    group="fusion",
    desc="B1+B2: FiLM conditioning and the learned reliability gate together -- context both "
         "reshapes the detail and decides how far to trust it.",
)

model_idea(
    "tt_scalars_learned_gate",                                           # D1+B2
    cols_fn=lambda c: (c.aef_cols + c.aef_scalars + c.tess_cols + c.tess_scalars),
    kwargs_fn=lambda c: two_tower_kwargs(
        c, modality_dropout=0.5, fusion="gated_mean", aef_maskable=True,
        tess_gate="learned",
        aef_columns=c.aef_cols + c.aef_scalars,
        tess_columns=c.tess_cols + c.tess_scalars),
    group="fusion",
    desc="D1+B2: the change scalars (best full-set feature set so far) under the learned "
         "reliability gate.",
)


# -- F. the stable-Artificial confusion --------------------------------------
# Every idea above was judged on binary change-F1, under which "Artificial ->
# Artificial" and "Vegetation -> Vegetation" score identically -- both are "no
# change". Rescoring the whole cache (rescore_ledger.py) shows 20-25% of stable
# Artificial plots come back as stable Vegetation in EVERY one of the ~40 ideas
# tested, a wall nothing has moved because nothing has aimed at it. These do.

for _w in (0.3, 1.0):                                                     # F1
    tt_variant(f"tt_endpoint{_w}", dropout_tess=0.7, scalars=True,
               endpoint_weight=_w, group="state",
               desc=f"F1: state-marginal supervision at weight {_w} -- an auxiliary loss on "
                    "'was this Artificial in 2018' and 'in 2024' as group-sums of the same "
                    "softmax. Pools 1,148 Artificial-in-2018 and 1,695 Artificial-in-2024 "
                    "plots into one built-up decision each, instead of splitting them across "
                    "two thin transition classes.")

_mc_idea("mc_endpoint", True,
         "F1+E4: the deployed mc_dropout_scalars recipe with state-marginal supervision -- "
         "the fix applied to the model that is actually shipping.",
         group="state", endpoint_weight=1.0)


def _seed_pool(view: View, source: str, seed: int):
    """The cached seeds of ``source`` for this read, holding one out when asked.

    Seed 0 pools every cached seed; higher seeds drop one, so a ledger row's seed
    spread measures the ensemble's sensitivity to *membership* rather than
    re-running an ensemble that has no randomness left in it.
    """
    have = [(s, c) for s, c in ((s, load_oof(source, view.name, s)) for s in range(5))
            if c is not None]
    if not have:
        raise RuntimeError(f"needs cached {source} OOF for {view.name}")
    members = [c for s, c in have if seed == 0 or s != (seed - 1) % len(have)]
    return np.mean([m[0] for m in members], axis=0), have[0][1][1]


@register("seed_ensemble_mc", group="ensemble",                           # F2
          desc="F2: average the deployed model's OOF probabilities across its 5 torch seeds "
               "and read once. Free at inference (the seeds are already trained) and worth "
               "+0.005 change-F1 / +0.004 macro-F1 on the deploy read.")
def seed_ensemble_mc(ctx, view, seed):
    return _seed_pool(view, "mc_dropout_scalars", seed)


def _cost_labels(view: View, probs: np.ndarray, costs: np.ndarray) -> np.ndarray:
    arr = np.array(view.merged_classes, dtype=object)
    return arr[(probs * costs).argmax(1)]


COST_GRID = np.round(np.arange(0.8, 3.01, 0.2), 2)


def nested_cost_gate(view: View, probs: np.ndarray,
                     target: str | None = "Artificial -> Artificial",
                     passes: int = 2) -> np.ndarray:
    """Labels whose per-class decision cost is chosen without seeing the fold.

    The arg-max is the Bayes rule for *accuracy*; it is the wrong rule for a
    metric that weights a 979-plot class like a 4,550-plot one. For each outer
    fold the multiplier that maximises macro-F1 on the other folds is selected
    and applied to the held-out one -- the honest version of the prior
    correction, which un-nested buys Artificial recall by spending change-F1.

    ``target`` names the one class to reweight; ``None`` reweights **every**
    class by coordinate ascent over ``passes`` sweeps. The wide search has four
    free parameters fitted on four folds, so it is as much a test of whether the
    inner folds can support that many as of whether the extra freedom pays.
    """
    classes = view.merged_classes
    targets = ([i for i, c in enumerate(classes) if c == target] if target
               else list(range(len(classes))))
    labels = np.empty(len(view.truth_merged), dtype=object)
    for tr, te in view.folds:
        costs = np.ones(len(classes))
        best_score = macro_f1(view.truth_merged[tr],
                              _cost_labels(view, probs[tr], costs), classes)
        for _ in range(passes if target is None else 1):
            for j in targets:
                for m in COST_GRID:
                    trial = costs.copy()
                    trial[j] = m
                    score = macro_f1(view.truth_merged[tr],
                                     _cost_labels(view, probs[tr], trial), classes)
                    if score > best_score:
                        best_score, costs = score, trial
        labels[te] = _cost_labels(view, probs[te], costs)
    return labels


def costgate_idea(name, desc, *, source="mc_dropout_scalars", ensemble=False,
                  target="Artificial -> Artificial"):
    """Register a nested cost-gate read over one cached model's OOF probabilities."""
    def fn(ctx, view, seed):
        if ensemble:
            probs, fine = _seed_pool(view, source, seed)
        else:
            cached = load_oof(source, view.name, seed)
            if cached is None:
                raise RuntimeError(f"{name} needs cached {source} OOF")
            probs, fine = cached
        return probs, fine, nested_cost_gate(view, probs, target)
    return register(name, group="operating-point", desc=desc)(fn)


costgate_idea(                                                            # F3
    "costgate_macro",
    "F3: per-class decision costs chosen on inner folds to maximise MACRO-F1, then "
    "applied to the held-out fold. The arg-max is the Bayes rule for accuracy, not for "
    "a metric that weights a 979-plot class like a 4,550-plot one -- this is the honest "
    "nested version of that correction.")

costgate_idea(                                                            # F7
    "costgate_ensemble", ensemble=True,
    desc="F7: F2 and F3 composed -- the nested macro-F1 cost gate chosen on the "
         "SEED-ENSEMBLED probabilities rather than one seed's. The gate is a decision "
         "rule over probabilities and the ensemble makes those probabilities better "
         "calibrated, so the two gains should compose rather than overlap.")

costgate_idea(                                                            # F8
    "costgate_wide", target=None,
    desc="F8: the cost search widened from one class to ALL FOUR by coordinate ascent on "
         "the inner folds. F3 found +0.035 stable-Artificial recall by reweighting one "
         "class; this asks whether macro-F1 has more to give, or whether four free "
         "parameters fitted on four folds is simply more than the data supports.")


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
def oof_path(idea: str, read: str, seed: int) -> Path:
    return OOF_DIR / f"{idea}__{read}__seed{seed}.npz"


def load_oof(idea: str, read: str, seed: int):
    """Cached OOF probabilities for a previously run idea (None if absent)."""
    path = oof_path(idea, read, seed)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return z["probs"], z["fine"]


def run_idea(ctx: Context, idea: Idea, read: str, seeds: list, notes: str = "") -> dict:
    """Run one idea on one read over every seed and append a ledger row."""
    view = ctx.view(read)
    per_seed, t0 = [], time.time()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = idea.fn(ctx, view, seed)
        probs, fine = result[0], result[1]
        labels = result[2] if len(result) > 2 else None
        np.savez_compressed(oof_path(idea.name, read, seed), probs=probs,
                            fine=np.asarray(fine, dtype=object),
                            classes=np.asarray(view.merged_classes, dtype=object))
        per_seed.append(score_probs(view, probs, fine, labels))
    elapsed = time.time() - t0

    row = {"idea": idea.name, "read": read, "group": idea.group,
           "n_seeds": len(seeds), "n_plots": len(view.target),
           "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
           "seconds": round(elapsed, 1), "desc": idea.desc, "notes": notes,
           "params": json.dumps(idea.params)}
    for key in per_seed[0]:
        vals = [s[key] for s in per_seed]
        with warnings.catch_warnings():
            # The Tessera split is all-NaN on the covered subset by construction.
            warnings.simplefilter("ignore", RuntimeWarning)
            row[f"{key}_mean"] = round(float(np.nanmean(vals)), 4)
            row[f"{key}_std"] = round(float(np.nanstd(vals)), 4)
    append_ledger(row)
    return row


def append_ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if LEDGER.exists():
        old = pd.read_csv(LEDGER)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(LEDGER, index=False)


def ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()


@register("seed_ensemble_s2_d07", group="s2-detail",                      # S12b
          desc="S12b: the seed ensemble over mc_s2_drop0.7 specifically -- the model the "
               "user picked from the maps. S12 measured the gain on drop0.6; S10 showed "
               "0.6 and 0.7 are statistically tied, but the deployed map should be scored "
               "on the recipe it actually uses, not on its twin.")
def seed_ensemble_s2_d07(ctx, view, seed):
    return _seed_pool(view, "mc_s2_drop0.7", seed)


@register("seed_ensemble_s2", group="s2-detail",                          # S12
          desc="S12: F2 transferred -- average the S2 model's OOF probabilities across its "
               "cached torch seeds. F2 was a clean win on the Tessera two-tower (+0.005 "
               "change-F1, +0.007 artStab, tightest variance on the board) and is free at "
               "inference because the seeds are already trained. Never tried on S2, and the "
               "deployed Oslo map currently runs a SINGLE seed.")
def seed_ensemble_s2(ctx, view, seed):
    return _seed_pool(view, "mc_s2_drop0.6", seed)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--tessera", type=Path, default=None)
    parser.add_argument("--s2", type=Path, default=None,
                        help="Sentinel-2 feature table (build_s2_features.py)")
    parser.add_argument("--ideas", default="", help="comma-separated idea names")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--group", default="", help="run every idea in a group")
    parser.add_argument("--read", choices=["full", "subset", "both"], default="both")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--notes", default="")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for name, idea in IDEAS.items():
            print(f"{name:28s} [{idea.group:10s}] reads={','.join(idea.reads):13s} "
                  f"{idea.desc}")
        return

    names = [n for n in args.ideas.split(",") if n]
    if args.all:
        names = list(IDEAS)
    if args.group:
        names = [n for n, i in IDEAS.items() if i.group == args.group]
    if not names:
        parser.error("nothing to run: pass --ideas, --group or --all")
    unknown = [n for n in names if n not in IDEAS]
    if unknown:
        parser.error(f"unknown ideas: {unknown} (see --list)")

    ctx = load_context(args.input, args.tessera, args.s2)
    seeds = list(range(args.n_seeds))
    s2_note = (f" s2={len(ctx.s2_stat_cols)}+{len(ctx.s2_patch_cols)}p "
               f"({ctx.frame['s2_present'].mean():.1%} covered)"
               if ctx.s2_stat_cols else " s2=absent")
    print(f"{len(ctx.frame):,} plots | aef={len(ctx.aef_cols)} "
          f"tess={len(ctx.tess_cols)}{s2_note} | "
          f"covered={len(ctx.view('subset').target):,} | seeds={seeds}", flush=True)

    for name in names:
        idea = IDEAS[name]
        reads = idea.reads if args.read == "both" else (
            (args.read,) if args.read in idea.reads else ())
        for read in reads:
            row = run_idea(ctx, idea, read, seeds, args.notes)
            print(f"  {name:26s} [{read:6s}] change_f1="
                  f"{row['change_f1_mean']:.4f}±{row['change_f1_std']:.3f}  "
                  f"macro_f1={row['macro_f1_mean']:.4f}  "
                  f"artStab={row['art_stable_recall_mean']:.3f} "
                  f"(as_veg={row['art_stable_as_veg_mean']:.3f})  "
                  f"({row['seconds']:.0f}s)", flush=True)
    print(f"\nledger -> {LEDGER}")


if __name__ == "__main__":
    main()
