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

from build_s2_features import CHANNELS_10M, CHANNELS_BASE
from experiment_hier_tessera import BASE, TESSERA, attach_tessera
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from model_zoo import (
    DEFAULT_INPUT,
    RARE_LABEL,
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


# The 10 m channel table (build_s2_features.py). Renamed from
# `s2_features_habloss_recover.parquet` when the channel set went from seven to
# eleven -- and then kept renamed, because the old path is UNREADABLE on this
# CIFS share: after being rewritten it opens with EIO and cannot even be
# unlinked, while an identical file under any other name reads fine. Do not
# rewrite a large parquet in place here; write a new name.
S2_FEATURES = "s2_features_habloss_recover_10m.parquet"
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
#:
#: A subset is a choice on THREE axes, not one: which families, which channels,
#: and which years. This dict is the family axis; `S2_SUBSET_CHANNELS` and
#: `S2_SUBSET_YEARS` are the other two, and both default so that every subset
#: written before the channel expansion still means exactly what it meant.
S2_SUBSETS: dict[str, tuple[str, ...]] = {
    "bf": ("bf",),
    "centre_bf": ("c", "bf"),
    "centre_s3_bf": ("c", "s3", "bf"),
    "centre_m3s3_bf": ("c", "m3", "s3", "bf"),
    "centre_3px_lc_bf": ("c", "m3", "s3", "lc", "bf"),
    "fine_scales_bf": ("c", "m3", "s3", "m9", "s9", "lc", "g", "bf"),
    "full": ("c", "m3", "m9", "m25", "s3", "s9", "s25", "lc", "g", "bf"),
    # -- S19: difference-driven detail, the deployed families throughout ------
    # The deployed tower reads each statistic three times: at 2018, at 2024 and
    # as their difference. Two thirds of it is therefore a *state* read, and a
    # state read is exactly what AlphaEarth already supplies -- what it cannot
    # supply is 10 m within-pixel structure and how that structure moved. These
    # arms drop the endpoints and keep the movement.
    "diff_centre_m3s3_bf": ("c", "m3", "s3", "bf"),
    "diff10_centre_m3s3_bf": ("c", "m3", "s3", "bf"),
    "diff10_bfstate_centre_m3s3": ("c", "m3", "s3", "bf"),
    "x10_centre_m3s3_bf": ("c", "m3", "s3", "bf"),
}

#: The channel axis. Absent = `CHANNELS_BASE`, the deployed seven -- so a channel
#: added to `build_s2_features.CHANNELS` reaches a model only when a subset names
#: it here. Without this default, appending an index would have silently widened
#: `s2off_centre_m3s3_bf` past the 78 columns CLAUDE.md pins it to.
S2_ALL_CHANNELS = tuple(CHANNELS_BASE) + tuple(CHANNELS_10M)
S2_SUBSET_CHANNELS: dict[str, tuple[str, ...]] = {
    "diff10_centre_m3s3_bf": S2_ALL_CHANNELS,
    "diff10_bfstate_centre_m3s3": S2_ALL_CHANNELS,
    "x10_centre_m3s3_bf": S2_ALL_CHANNELS,
}

#: The year axis. Absent = all three blocks (2018, 2024 and their difference).
#: A per-family override is allowed, keyed by family, for the one arm that keeps
#: built fraction as a state while every other family is a difference.
S2_YEARS_ALL = ("2018", "2024", "diff")
S2_SUBSET_YEARS: dict[str, tuple[str, ...] | dict[str, tuple[str, ...]]] = {
    "diff_centre_m3s3_bf": ("diff",),
    "diff10_centre_m3s3_bf": ("diff",),
    "diff10_bfstate_centre_m3s3": {"bf": S2_YEARS_ALL, None: ("diff",)},
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
    "diff_centre_m3s3_bf": "the deployed families and channels, but only the "
                           "2024-2018 difference of each",
    "diff10_centre_m3s3_bf": "2024-2018 differences only, over eleven 10 m "
                             "channels (four bands + seven indices)",
    "diff10_bfstate_centre_m3s3": "as diff10, but built fraction kept as a "
                                  "state at both years",
    "x10_centre_m3s3_bf": "the deployed year structure over eleven 10 m "
                          "channels -- the control that isolates the channels",
}


def s2_base_columns(stat: list[str]) -> list[str]:
    """The stat block restricted to `CHANNELS_BASE` -- the published 204.

    Every number in the ledger written before the 10 m index expansion was
    measured on seven channels. Anything that consumes the *whole* stat block
    rather than a named subset therefore has to say which block it means, or a
    re-run would quietly compare 312 columns against a 204-column row and read
    the difference as an effect. Named subsets do not need this: they default to
    `CHANNELS_BASE` in `s2_subset_columns` already.
    """
    return [c for c in stat if s2_channel_of(c) in (None,) + tuple(CHANNELS_BASE)]


def s2_channel_of(col: str) -> str | None:
    """The channel a Sentinel-2 column belongs to, or None if it has no channel.

    Columns are ``S2<fam>_<channel>_<year>`` except built fraction, which is
    ``S2bf<radius>_<year>`` and is a property of the window rather than of a
    channel. Parsing from the right is what makes this safe: channel names
    contain no underscore, but a family prefix may end in a digit.
    """
    if col.startswith("S2bf"):
        return None
    parts = col.split("_")
    return parts[-2] if len(parts) >= 3 else None


def s2_year_of(col: str) -> str:
    """``"2018"``, ``"2024"`` or ``"diff"`` -- the block a column belongs to."""
    return col.rsplit("_", 1)[-1]


def s2_subset_columns(stat: list[str], name: str) -> list[str]:
    """Columns of a named subset, in the stat block's own order.

    Filters on all three axes. The channel and year filters default to the
    deployed set, so this returns byte-identical columns for every subset that
    predates them however many channels `build_s2_features` grows.
    """
    fams = s2_families(stat)
    channels = set(S2_SUBSET_CHANNELS.get(name, CHANNELS_BASE))
    have = {s2_channel_of(c) for c in stat} - {None}
    if have and not channels <= have:
        raise ValueError(
            f"subset {name!r} wants channels {sorted(channels - have)} that the "
            f"column list handed in does not carry. Either the feature table "
            f"predates them (rebuild: `python src/build_s2_features.py`) or the "
            f"caller passed `s2_base_columns(...)`, which pins to the deployed "
            f"seven -- pass the whole stat block and let the subset filter it.")
    years = S2_SUBSET_YEARS.get(name, S2_YEARS_ALL)
    wanted = set()
    for key in S2_SUBSETS[name]:
        keep = years.get(key, years.get(None, S2_YEARS_ALL)) if isinstance(
            years, dict) else years
        for col in fams[key]:
            chan = s2_channel_of(col)
            if (chan is None or chan in channels) and s2_year_of(col) in keep:
                wanted.add(col)
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
    # Pinned to `CHANNELS_BASE`: these five scalars are a fixed published
    # quantity, and letting a newly added index widen the vector they are
    # computed over would move them for every model that reads them.
    def _c(year):
        return sorted(f"S2c_{ch}_{year}" for ch in CHANNELS_BASE
                      if f"S2c_{ch}_{year}" in set(stat))

    c18, c24 = _c(2018), _c(2024)
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
    ctx.views["lumped"] = _lumped_view(ctx.views["full"])
    return ctx


#: The lumped legend's single stable class. Written as a transition so that
#: ``is_change_label`` and ``to_merged_label`` -- which both parse ``"A -> B"`` --
#: keep working unmodified: it reads as stable, and it collapses to itself at the
#: merged level, so the three-level hierarchy stays strictly nested.
STABLE_LABEL = "Stable -> Stable"


def _lumped_view(full: View) -> View:
    """The user's MVP legend: every stable transition collapsed into one class.

    Nine coarse3 classes become seven -- ``Stable -> Stable`` (5,172 plots) plus
    the six change transitions unchanged. The hypothesis is that the fine head is
    spending capacity separating stable Nature from stable Cropland, the noisiest
    boundary in the whole legend (``analyse_label_noise.py``), and that the
    change classes are paying for it.

    **The folds are `full`'s folds, reused verbatim.** Re-running the splitter on
    the lumped target would restratify and hand this view a different partition,
    and every difference in the result would then be partly the split rather than
    the legend. Same plots, same blocks, same rows -- only the target changes.

    What stays comparable: binary ``change_f1`` (the change/stable partition is
    identical) and every ``focus_metrics`` transition (those classes are
    untouched). What does not: the merged2 per-class metrics, since
    ``Artificial -> Artificial`` no longer exists as a class. Judge this view on
    the first two only.
    """
    target = full.target.map(
        lambda c: STABLE_LABEL if c != RARE_LABEL and not is_change_label(c) else c)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    return View("lumped", full.frame, target, full.folds, truth_merged,
                truth_fine, sorted(set(truth_merged)))


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
    out.update(focus_metrics(view.truth_fine, fine))
    return out


#: The coarse3 transitions the map is actually commissioned to find, named so a
#: run can be judged on them and not only on the binary gate. The first two are
#: habitat loss, the second two recovery -- which is what this project is for and
#: what the aggregate change-F1 hides, since Nature -> Artificial (383 plots) and
#: Artificial -> Cropland (46) contribute almost nothing to a metric dominated by
#: the 4.2k stable plots.
FOCUS_TRANSITIONS = ("Nature -> Artificial", "Cropland -> Artificial",
                     "Artificial -> Nature", "Artificial -> Cropland")


def focus_metrics(truth_fine: np.ndarray, fine: np.ndarray | None) -> dict:
    """Per-class recall/precision/F1 on the commissioned coarse3 transitions.

    NaN rather than 0 for a class absent from this read, so a missing class never
    silently drags the focus macro down. ``focus_macro_f1`` is the unweighted
    mean over the classes present -- a 46-plot transition weighs the same as a
    383-plot one, which is the point of tracking it separately from change-F1.
    """
    if fine is None:
        return {}
    out, f1s = {}, []
    for cls in FOCUS_TRANSITIONS:
        slug = cls.lower().replace(" -> ", "_to_").replace(" ", "")
        present = truth_fine == cls
        if not present.any():
            out[f"fine_recall_{slug}"] = np.nan
            out[f"fine_f1_{slug}"] = np.nan
            continue
        picked = fine == cls
        tp = int((present & picked).sum())
        rec = tp / max(int(present.sum()), 1)
        prec = tp / max(int(picked.sum()), 1) if picked.any() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[f"fine_recall_{slug}"] = float(rec)
        out[f"fine_f1_{slug}"] = float(f1)
        f1s.append(f1)
    out["focus_macro_f1"] = float(np.mean(f1s)) if f1s else np.nan
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
    group="baseline", reads=("full", "subset", "lumped"),
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
        # `s2_base_columns`: the S-section numbers in the ledger are seven-
        # channel numbers, so these blocks stay seven-channel however many
        # 10 m indices `build_s2_features` grows.
        stat = s2_base_columns(c.s2_stat_cols)
        block = {"stat": stat, "texture": s2_base_columns(c.s2_texture_cols),
                 "patch": c.s2_patch_cols,
                 "all": stat + c.s2_patch_cols}[family]
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
    cols_fn=lambda c: s2_base_columns(c.s2_stat_cols),
    kwargs_fn=lambda c: dict(BASE),
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
    detail = [c for c in s2_base_columns(ctx.s2_stat_cols) if c not in set(built)]
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
    detail = s2_base_columns(ctx.s2_stat_cols) + (ctx.s2_scalars if scalars else [])
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

costgate_idea(                                                            # N12
    "costgate_siam", source="siam_cos",
    desc="N12: the F3 nested cost gate applied to the siamese. N11 showed the siamese sits "
         "at a DIFFERENT POINT on the stable Artificial/Vegetation boundary rather than on "
         "a worse boundary -- art->veg rose while veg->art fell and false-change-on-built-up "
         "fell by a quarter. If that reading is right the trade is recoverable post-hoc by "
         "reweighting one class, and the user can pick which error they want without a "
         "retrain. Free: a different arg-max over probabilities already cached.")

costgate_idea(                                                            # N12b
    "costgate_s2off", source="s2off_centre_m3s3_bf",
    desc="N12b: the same gate on the DEPLOYED model, so N12 is compared against an "
         "incumbent that has had the same post-hoc treatment rather than against its raw "
         "arg-max. Comparing a gated challenger to an ungated incumbent is how a free "
         "decision-rule gain gets misattributed to the architecture.")

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


def load_oof_fine(idea: str, read: str, seed: int):
    """Cached OOF *coarse3* probabilities and class order, or None.

    Separate from ``load_oof`` because the cache only carries these for ideas
    run through a fine-probability CV path -- everything before section O stored
    the coarse3 arg-max label and threw the distribution away, which is why the
    coarse3 cost gate (N6) could not be run post-hoc on the existing cache.
    """
    path = oof_path(idea, read, seed)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    if "fine_probs" not in z.files:
        return None
    return z["fine_probs"], list(z["fine_classes"])


def run_idea(ctx: Context, idea: Idea, read: str, seeds: list, notes: str = "") -> dict:
    """Run one idea on one read over every seed and append a ledger row."""
    view = ctx.view(read)
    per_seed, t0 = [], time.time()
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        result = idea.fn(ctx, view, seed)
        probs, fine = result[0], result[1]
        labels = result[2] if len(result) > 2 else None
        extra = {}
        # Optional 4th element: (coarse3 probabilities, coarse3 class order).
        # Only the section-O paths produce it, and only it makes the coarse3
        # cost gate runnable over the cache instead of by refitting.
        if len(result) > 3 and result[3] is not None:
            fine_probs, fine_classes = result[3]
            extra = {"fine_probs": fine_probs,
                     "fine_classes": np.asarray(fine_classes, dtype=object)}
        np.savez_compressed(oof_path(idea.name, read, seed), probs=probs,
                            fine=np.asarray(fine, dtype=object),
                            classes=np.asarray(view.merged_classes, dtype=object),
                            **extra)
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
# N. Siamese endpoint towers  (docs/research/SIAMESE_RESEARCH.md)
#
# A different question from every idea above. Sections A-G asked how to fuse two
# *modalities*; this asks how to structure the two *dates*. Every model in the
# ledger flattens [2018 | 2024 | diff] into one vector and lets a wide trunk work
# out that the first two blocks are the same 64 measurements six years apart. A
# siamese encoder is told instead, which halves the encoder parameters -- the
# lever the learning curves say this problem is short of -- and puts the endpoint
# pair somewhere an auxiliary objective can reach it.
# ---------------------------------------------------------------------------
def siam_columns(ctx):
    """The endpoint blocks and the extras, in the order ``_prepare`` expects."""
    a18 = sorted(c for c in ctx.aef_cols if c.endswith("_2018"))
    a24 = sorted(c for c in ctx.aef_cols if c.endswith("_2024"))
    diff = sorted(c for c in ctx.aef_cols if c.endswith("_diff"))
    assert len(a18) == len(a24), "endpoint blocks must pair up 1:1"
    # a18[i] and a24[i] are the same band by construction of the sort (A00_2018 /
    # A00_2024), which is what the shared encoder relies on.
    assert [c[:-5] for c in a18] == [c[:-5] for c in a24]
    return a18, a24, diff


def siam_kwargs(ctx, *, extra_cols=None, **overrides):
    """The wide/focal/30-epoch recipe wearing a siamese endpoint trunk.

    ``extra_cols`` defaults to the raw AlphaEarth ``diff`` block, so the first
    comparison against ``baseline_aef`` is information-matched: the flat model
    reads 2018+2024+diff, and so does this one -- the only difference is that the
    endpoints go through a shared encoder rather than three separate first-layer
    weight blocks. Dropping the raw diff is a separate hypothesis (N2), not a
    handicap to bundle into the first run, given that removing it from the flat
    model costs -0.048 change-F1.
    """
    a18, a24, diff = siam_columns(ctx)
    kwargs = dict(
        arch="siamese", loss=BASE["loss"], epochs=BASE["epochs"],
        siam_columns_18=a18, siam_columns_24=a24,
        siam_extra_columns=diff if extra_cols is None else extra_cols,
        siam_dim=128, siam_combine="conc", tower_dim=256,
    )
    kwargs.update(overrides)
    return kwargs


def siam_all_cols(ctx, extra_cols=None):
    a18, a24, diff = siam_columns(ctx)
    return a18 + a24 + (diff if extra_cols is None else list(extra_cols))


model_idea(
    "siam_shared",                                                        # N1
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c),
    group="siamese", reads=("full",),
    desc="N1: shared-encoder siamese over the 2018 and 2024 AlphaEarth blocks, head on "
         "[z18, z24, z24-z18, |z24-z18|, cos] plus the raw diff block. Information-matched "
         "to baseline_aef; the only change is weight sharing across the two dates. The "
         "gate question for the whole section -- if a shared encoder cannot match a flat "
         "trunk on the same inputs, the cosine and Barlow objectives that need it have "
         "nowhere to live.")


def s2off_cv(view, cols, kwargs, seed):
    """Out-of-fold probabilities under the DEPLOYED gate-off read.

    The deployed model trains with both towers and serves with the Sentinel-2
    gate forced to zero, so scoring it with the gate on would score a model that
    is never run. Zeroing ``S2_MASK`` on the test frame is exactly what
    ``optimise_s2off.gate_off_cv`` does and what produced the 15-seed table in
    ``S2_DETAIL_RESEARCH.md``; it is reproduced here rather than imported so the
    deployed recipe lands in *this* ledger, on these folds, with the focus
    metrics section N is judged on.
    """
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    for tr, te in view.folds:
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
            te_frame = view.frame.iloc[te].copy()
            te_frame[S2_MASK] = 0.0            # the deployed read
            p_fine, p_merged = model._probs(te_frame)
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


def s2off_deployed_kwargs(ctx):
    """`s2off_centre_m3s3_bf` exactly as `infer_s2.fit_models` builds it."""
    detail = s2_subset_columns(ctx.s2_stat_cols, "centre_m3s3_bf")
    return detail, dict(
        arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"], tower_dim=256,
        aef_columns=ctx.aef_cols, tess_columns=detail,
        mask_column=S2_MASK, aef_mask_column=AEF_MASK,
        fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
    )


register("s2off_centre_m3s3_bf", reads=("full",), group="siamese",
         desc="THE DEPLOYED MODEL (CLAUDE.md), scored on section N's footing: the "
              "AlphaEarth two-tower trained with the 78-column Sentinel-2 detail tower as "
              "privileged information and read with the detail gate OFF. Registered here "
              "only so the siamese line has an honest incumbent on the same folds, seeds "
              "and focus metrics -- it is the comparison target, not a candidate."
         )(lambda ctx, view, seed: s2off_cv(view, ctx.aef_cols + s2off_deployed_kwargs(ctx)[0],
                                            s2off_deployed_kwargs(ctx)[1], seed))


def siam_s2off_kwargs(ctx, **overrides):
    """The deployed gate-off recipe with a SIAMESE AlphaEarth tower.

    Everything that makes the deployment work is kept: the 78-column Sentinel-2
    detail tower is still privileged information, still mask-gated, still forced
    off at serving, so no Sentinel-2 is read at inference. Only the AlphaEarth
    tower changes, from a flat MLP over 192 interleaved columns to the shared
    endpoint encoder of N1/N2.

    ``aef_columns`` stays in its usual sorted order. The shared encoder needs
    the block as ``[all _2018 | all _2024 | rest]``, but it derives that
    permutation from the column names itself (`_aef_siam_permutation`) and
    gathers inside the tower, so nothing upstream -- including the raster path's
    `stack_aef_bands` -- has to know or care.
    """
    aef = ctx.aef_cols
    detail = s2_subset_columns(ctx.s2_stat_cols, "centre_m3s3_bf")
    kwargs = dict(
        arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"], tower_dim=256,
        aef_columns=aef, tess_columns=detail,
        mask_column=S2_MASK, aef_mask_column=AEF_MASK,
        fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
        aef_siam=True, siam_dim=128, siam_combine="conc",
    )
    kwargs.update(overrides)
    return aef + detail, kwargs


def _siam_s2off_idea(name, desc, **overrides):
    def fn(ctx, view, seed):
        cols, kwargs = siam_s2off_kwargs(ctx, **overrides)
        return s2off_cv(view, cols, kwargs, seed)
    return register(name, reads=("full",), group="siamese", desc=desc)(fn)


_siam_s2off_idea(
    "siam_s2off",                                                         # N8
    "N8: the DEPLOYED recipe with its AlphaEarth tower replaced by the shared endpoint "
    "encoder -- same 78-column privileged Sentinel-2 tower, same gate-off serving, same "
    "modality dropout. N12 established that stable built-up is genuinely the deployed "
    "model's win rather than an operating-point artefact, and N2 that the focus classes "
    "are the siamese's. This is the only construction that could hold both, and it costs "
    "nothing at inference because Sentinel-2 is still never read.")

_siam_s2off_idea(
    "siam_s2off_cos", siam_cos_weight=0.3, siam_cos_margin=0.3,           # N8b
    desc="N8b: N8 plus the N2 cosine objective, which is where the siamese's gain actually "
         "came from -- N1 alone was a tie on change-F1. Run as a pair with N8 so the "
         "architecture and the objective are not confounded.")


#: Auxiliary-loss weights, scaled off the measured loss magnitudes rather than
#: guessed. Under active optimisation the three terms sit on one scale -- the
#: three-level supervised loss runs 2.49 -> 0.71 over the 30 epochs, the cosine
#: term 0.36 -> 0.18, the Barlow term 0.64 -> 0.11 -- so 0.3 makes an auxiliary
#: roughly 10% of the objective at convergence: a regulariser, not a co-objective.
#: That is the conservative first test; 1.0 is the co-objective reading and is
#: registered so the strength question is a preregistered follow-up rather than a
#: post-hoc rescue of a flat result.
SIAM_AUX = 0.3


model_idea(
    "siam_cos",                                                           # N2
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c, siam_cos_weight=SIAM_AUX,
                                    siam_cos_margin=0.3),
    group="siamese", reads=("full", "lumped"),
    desc="N2: N1 plus the gate-supervised cosine -- pull z18/z24 together on stable plots, "
         "push them apart past a 0.3 margin on change plots, the two group terms weighted "
         "equally so the 4:1 stable majority cannot turn it into a plain 'make everything "
         "similar' regulariser. States on the representation what the classifier head is "
         "otherwise left to induce, and the supervision is carried by the stable majority "
         "so it costs nothing on the rare transitions that are the target.")

model_idea(
    "siam_cos_strong",                                                    # N2b
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c, siam_cos_weight=1.0,
                                    siam_cos_margin=0.3),
    group="siamese", reads=("full",),
    desc="N2b: the same cosine objective at weight 1.0 -- a co-objective rather than a "
         "regulariser. Preregistered so that if N2 comes back flat the strength question "
         "is answered rather than assumed.")

model_idea(
    "siam_barlow",                                                        # N3
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c, siam_barlow_weight=SIAM_AUX),
    group="siamese", reads=("full",),
    desc="N3: N1 plus Barlow Twins redundancy reduction between z18 and z24 on STABLE "
         "pairs only. Zbontar et al. need two augmented views and a hand-tuned "
         "augmentation policy; a stable plot supplies two genuine views of one unchanged "
         "patch for free. Driving their cross-correlation to the identity asks for "
         "features invariant to acquisition and phenology and mutually decorrelated, so "
         "what survives in z24-z18 is change rather than nuisance.")

model_idea(
    "siam_cos_barlow",                                                    # N3b
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c, siam_cos_weight=SIAM_AUX,
                                    siam_cos_margin=0.3,
                                    siam_barlow_weight=SIAM_AUX),
    group="siamese", reads=("full",),
    desc="N3b: both auxiliaries together. They are not the same statement -- the cosine "
         "acts on the ANGLE of one pair and needs the change label; Barlow acts on the "
         "cross-correlation ACROSS the batch and needs only stable/change. Run only if N2 "
         "and N3 each move something, and read against both rather than against N1.")

UNLABELLED_AEF = "unlabelled_aef_oslo.parquet"


def _unlabelled_pool():
    """The unlabelled endpoint pool, or None when it has not been built."""
    path = project_data_dir("embeddings", UNLABELLED_AEF)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    return frame.astype({c: "float64" for c in frame.columns
                         if frame[c].dtype.kind == "f"})


def siam_ssl_idea(name, weight, desc):
    """A Barlow idea that also sees the unlabelled pool at ``weight``."""
    def fn(ctx, view, seed):
        pool = _unlabelled_pool()
        if pool is None:
            raise SystemExit(
                f"{name} needs {UNLABELLED_AEF}; run build_unlabelled_aef.py first")
        kwargs = siam_kwargs(ctx, siam_barlow_weight=SIAM_AUX,
                             siam_unlabelled_weight=weight)
        return cv_probs_unlabelled(view, siam_all_cols(ctx), kwargs, seed, pool)
    return register(name, reads=("full",), group="siamese", desc=desc)(fn)


def cv_probs_unlabelled(view, cols, kwargs, seed, pool):
    """``cv_probs`` with an unlabelled frame handed to every fold's fit.

    The pool is passed whole to each fold rather than split: it carries no
    labels, so there is nothing in it to leak, and none of its pixels can be a
    test plot -- zero labelled plots fall inside either AOI bbox (G3/G4).
    """
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    for tr, te in view.folds:
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy(),
                      unlabelled_frame=pool)
            p_fine, p_merged = model._probs(view.frame.iloc[te])
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


siam_ssl_idea(
    "siam_barlow_ssl", 0.3,                                               # N4
    "N4: the N3 Barlow term extended to 200k UNLABELLED Oslo endpoint pairs, every one "
    "assumed stable. N3 showed the term reaches the same accuracy as the cosine from "
    "strictly less supervision -- never needing to know which transition a plot is -- "
    "which is exactly what lets it transfer to pixels carrying no label. The only idea "
    "in section N that adds information rather than rearranging the same 6,414 plots. "
    "The assumption is wrong on ~0.5% of sampled pairs at the deployed map's base rate.")

siam_ssl_idea(
    "siam_barlow_ssl_strong", 1.0,                                        # N4b
    "N4b: the same with the unlabelled term at weight 1.0 -- if 200k pixels are worth "
    "having, they may be worth more than a 0.3 regulariser's share of the objective. "
    "Preregistered so a flat N4 is a statement about the pool rather than about a weight.")


# -- N14. external single-date state labels ----------------------------------
# N4 tested the unlabelled route and it failed, concluding that what this problem
# is short of is *labels*. The shared encoder is a function of ONE date, so a
# single-date land-cover state label is a valid input to it -- the flat trunk has
# no such entry point. N14a (`diagnose_state_pools.py`) cleared GLanCE's legend
# against the RECOVER interpreters' at the self-floor and did NOT clear LUCAS as
# a global pool; these ideas are the model-side test that follows.
STATE_POOL = "state_labels_glance_strict_2018.parquet"


def _state_pool(name=STATE_POOL):
    """The external single-date state pool, or None when it has not been built."""
    path = project_data_dir("embeddings", name)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    return frame.astype({c: "float64" for c in frame.columns
                         if frame[c].dtype.kind == "f"})


def cv_probs_state(view, cols, kwargs, seed, pool, s2off=False):
    """``cv_probs`` with an external state pool handed to every fold's fit.

    **The pool is cut to each fold's training blocks**, unlike
    ``cv_probs_unlabelled``. That pool could be passed whole because no labelled
    plot falls inside either AOI (G3/G4); this one is global and covers all 83
    blocks, so a whole-pool pass would let the encoder see embeddings from the
    held-out block. No plot-level leak is possible either way -- zero pool points
    sit within 100 m of a RECOVER plot and 9 of 6,414 within 1 km -- but the
    blocked CV exists to measure spatial generalisation, and handing the model
    the test block's feature distribution would quietly weaken exactly that.
    """
    classes = view.merged_classes
    probs = np.zeros((len(view.target), len(classes)))
    fine = np.empty(len(view.target), dtype=object)
    for tr, te in view.folds:
        fold_pool = None
        if pool is not None:
            train_blocks = set(view.frame.iloc[tr]["block_id"].unique())
            fold_pool = pool[pool["block_id"].isin(train_blocks)]
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy(),
                      state_frame=fold_pool)
            te_frame = view.frame.iloc[te]
            if s2off:
                te_frame = te_frame.copy()
                te_frame[S2_MASK] = 0.0        # the deployed gate-off read
            p_fine, p_merged = model._probs(te_frame)
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine


def siam_state_idea(name, desc, *, base="cos", weight=SIAM_AUX,
                    source="external", **overrides):
    """A state-head idea over either N2 (`cos`) or N8b (`s2off`)."""
    def fn(ctx, view, seed):
        pool = None
        if source in ("external", "both"):
            pool = _state_pool()
            if pool is None:
                raise SystemExit(
                    f"{name} needs {STATE_POOL}; run build_state_labels.py first")
        extra = dict(siam_state_weight=weight, siam_state_source=source,
                     **overrides)
        if base == "s2off":
            cols, kwargs = siam_s2off_kwargs(
                ctx, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3, **extra)
            return cv_probs_state(view, cols, kwargs, seed, pool, s2off=True)
        kwargs = siam_kwargs(ctx, siam_cos_weight=SIAM_AUX,
                             siam_cos_margin=0.3, **extra)
        return cv_probs_state(view, siam_all_cols(ctx), kwargs, seed, pool)
    return register(name, reads=("full",), group="siamese", desc=desc)(fn)


siam_state_idea(
    "siam_cos_state_endo", source="endogenous",                           # N14b
    desc="N14b: the CONTROL. The state head fed only from the plots' own endpoints -- a "
         "From -> To label is two free state labels, so this adds no data and tests the "
         "mechanism alone. F1 supervised the same marginals at the softmax and came back "
         "flat, so a flat result here is the expectation, not a surprise; its job is to "
         "make N14c's number attributable to the new labels rather than to the head.")

siam_state_idea(
    "siam_cos_state", weight=SIAM_AUX,                                    # N14c
    desc="N14c: N2 plus a single-date state head supervised by 13,118 GLanCE units at "
         "2018, harmonised to coarse3 and cleared against the RECOVER legend in N14a "
         "(macro-F1 0.733 against a 0.740 self-floor, and BETTER than the floor on "
         "crop->nature and art->nature). Weight 0.3 to match the cosine's preregistered "
         "regulariser share. The first idea in section N to add labels rather than "
         "rearrange the 6,414.")

siam_state_idea(
    "siam_cos_state_strong", weight=1.0,                                  # N14d
    desc="N14d: the same at weight 1.0, a co-objective rather than a regulariser. "
         "Preregistered because N4 found the unlabelled term bought change-F1 at 1.0 by "
         "trading Nature -> Artificial away; if that is what external data does at "
         "strength, this is where it shows.")

model_idea(
    "siam_pseudo",                                                        # N10
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c, siam_cos_weight=SIAM_AUX,
                                    siam_cos_margin=0.3,
                                    siam_year_adapter="input"),
    group="siamese", reads=("full",),
    desc="N10: N2 with a per-year diagonal affine on the encoder input -- Daudt et al.'s "
         "pseudo-siamese, at the smallest size that tests the mechanism. N9 ruled out a "
         "feature gap for the stable-built-up regression and left one explanation: a flat "
         "trunk gets a separate first-layer weight block per year and can absorb a "
         "sensor/phenology offset there, one shared map cannot. 2 x 64 parameters per "
         "year, identity-initialised, so the fully-shared model is the starting point and "
         "any per-year deviation has to pay for itself.")

model_idea(
    "siam_pseudo_out",                                                    # N10b
    cols_fn=siam_all_cols,
    kwargs_fn=lambda c: siam_kwargs(c, siam_cos_weight=SIAM_AUX,
                                    siam_cos_margin=0.3,
                                    siam_year_adapter="output"),
    group="siamese", reads=("full",),
    desc="N10b: the same adapter placed AFTER the encoder instead of before. Not "
         "equivalent -- the cosine and Barlow losses read z18/z24, so an output adapter "
         "also rescales what those objectives see, and a degenerate solution is available "
         "(shrink one year's scale to raise every cosine). Registered to bound that, and "
         "expected to be the worse of the two for exactly that reason.")

model_idea(
    "siam_builtfrac",                                                     # N9
    cols_fn=lambda c: siam_all_cols(
        c, extra_cols=sorted(x for x in c.aef_cols if x.endswith("_diff")) + c.s2_built_cols),
    kwargs_fn=lambda c: siam_kwargs(
        c, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,
        extra_cols=sorted(x for x in c.aef_cols if x.endswith("_diff")) + c.s2_built_cols),
    group="siamese", reads=("full",),
    desc="N9: N2 plus the NDVI built-fraction block as flat extras. Diagnostic, not a "
         "candidate: the siamese beats the deployed model on every commissioned "
         "transition but returns 23.4% of stable built-up as stable Vegetation against "
         "the deployed model's 19.6%, and built fraction is the known lever for exactly "
         "that error (S8/aef_builtfrac: art->veg 0.1916, the best on the board). This "
         "asks whether the regression is fixable at all. NOTE it reads Sentinel-2 at "
         "INFERENCE and so gives up the s2off property -- if it works, the deployable "
         "form is a siamese AlphaEarth tower inside the gated two-tower, not this.")


model_idea(
    "siam_diffonly",                                                      # N5
    cols_fn=lambda c: siam_all_cols(c, extra_cols=[]),
    kwargs_fn=lambda c: siam_kwargs(c, extra_cols=[]),
    group="siamese", reads=("full",),
    desc="N5: drop the raw AlphaEarth diff block and let the learned z24-z18 carry it. "
         "Removing diff from the FLAT model costs -0.048 change-F1, but that model has no "
         "learned difference to fall back on. Tests whether the siamese has actually "
         "internalised it or is quietly leaning on the raw block.")


# ---------------------------------------------------------------------------
# Section O -- output structure and decision rule, over the section-N encoders.
#
# Section N closed on ARCHITECTURE: every trunk change ran and the verdict was
# that a shared encoder plus any year-invariance pressure is the whole gain.
# Section O keeps that encoder fixed and changes what sits on top of it -- how
# the 9 coarse3 logits are PARAMETERISED, and how the arg-max over them is
# TAKEN. Both are aimed at the thing section N could not move: `Artificial ->
# Cropland` at 0.000 for every model in the table, and `focus_macro_f1` more
# generally, which N12 showed the existing merged2 cost gate cannot touch.
#
# Everything here runs through `cv_probs_fine`/`s2off_cv_fine`, which cache the
# coarse3 DISTRIBUTION rather than only its arg-max. That is not a detail: it is
# what makes O3 a post-hoc read over the cache instead of a refit, and the
# reason the pre-section-O cache cannot answer it.
# ---------------------------------------------------------------------------
def cv_probs_fine(view: View, cols: list, kwargs: dict, seed: int):
    """``cv_probs`` that also returns the coarse3 probabilities.

    Fold-local coarse3 classes are mapped into a global sorted class list the
    same way ``cv_probs`` maps the merged ones, so a fold that never sees the
    46-plot transition leaves that column at 0 rather than shifting every column
    one to the left. Getting that wrong would be invisible in the aggregate and
    fatal in exactly the rare class this section is about.
    """
    n = len(view.target)
    classes = view.merged_classes
    fine_classes = sorted(set(view.truth_fine))
    probs = np.zeros((n, len(classes)), dtype="float64")
    fine_probs = np.zeros((n, len(fine_classes)), dtype="float64")
    fine = np.empty(n, dtype=object)
    for tr, te in view.folds:
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
            p_fine, p_merged = model._probs(view.frame.iloc[te])
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine_probs[np.ix_(te, [fine_classes.index(c)
                               for c in model.fine_classes_])] = p_fine
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine, None, (fine_probs, fine_classes)


def s2off_cv_fine(view: View, cols: list, kwargs: dict, seed: int):
    """``s2off_cv`` that also returns the coarse3 probabilities.

    Keeps the deployed gate-off read exactly -- ``S2_MASK`` zeroed on the test
    frame -- because a model scored with the detail gate on is a model that is
    never served.
    """
    n = len(view.target)
    classes = view.merged_classes
    fine_classes = sorted(set(view.truth_fine))
    probs = np.zeros((n, len(classes)), dtype="float64")
    fine_probs = np.zeros((n, len(fine_classes)), dtype="float64")
    fine = np.empty(n, dtype=object)
    for tr, te in view.folds:
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
            te_frame = view.frame.iloc[te].copy()
            te_frame[S2_MASK] = 0.0            # the deployed read
            p_fine, p_merged = model._probs(te_frame)
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine_probs[np.ix_(te, [fine_classes.index(c)
                               for c in model.fine_classes_])] = p_fine
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine, None, (fine_probs, fine_classes)


def siam_idea(name, desc, **overrides):
    """An AlphaEarth-only siamese idea on the N2 base, with fine probs cached."""
    def fn(ctx, view, seed):
        return cv_probs_fine(view, siam_all_cols(ctx),
                             siam_kwargs(ctx, siam_cos_weight=SIAM_AUX,
                                         siam_cos_margin=0.3, **overrides), seed)
    return register(name, reads=("full",), group="section-o", desc=desc)(fn)


def s2off_siam_idea(name, desc, **overrides):
    """A deployed-recipe siamese idea (N8b base), gate-off, with fine probs cached."""
    def fn(ctx, view, seed):
        cols, kwargs = siam_s2off_kwargs(ctx, siam_cos_weight=0.3,
                                         siam_cos_margin=0.3, **overrides)
        return s2off_cv_fine(view, cols, kwargs, seed)
    return register(name, reads=("full",), group="section-o", desc=desc)(fn)


# -- O0: the three incumbents re-run so the coarse3 gate has something to read --
register("base_siam_cos_fine", reads=("full",), group="section-o",
         desc="O0a: `siam_cos` (N2) re-run through the fine-probability path. Numerically "
              "the same model on the same folds and seeds -- it exists so O3 has a cached "
              "coarse3 DISTRIBUTION for the AlphaEarth-only siamese, which the pre-O cache "
              "does not carry."
         )(lambda ctx, view, seed: cv_probs_fine(
             view, siam_all_cols(ctx),
             siam_kwargs(ctx, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3), seed))

register("base_siam_s2off_cos_fine", reads=("full",), group="section-o",
         desc="O0b: `siam_s2off_cos` (N8b, the section's best model) re-run through the "
              "fine-probability path, for the same reason as O0a."
         )(lambda ctx, view, seed: s2off_cv_fine(
             view, *siam_s2off_kwargs(ctx, siam_cos_weight=0.3, siam_cos_margin=0.3),
             seed))

register("base_deployed_fine", reads=("full",), group="section-o",
         desc="O0c: the DEPLOYED model `s2off_centre_m3s3_bf` re-run through the "
              "fine-probability path. The incumbent has to be gated by the same instrument "
              "as the challengers or a free decision-rule gain gets misattributed to an "
              "architecture -- which is the mistake N12 was designed to avoid."
         )(lambda ctx, view, seed: s2off_cv_fine(
             view, ctx.aef_cols + s2off_deployed_kwargs(ctx)[0],
             s2off_deployed_kwargs(ctx)[1], seed))


# -- O1: endpoint-tied factorised head ---------------------------------------
_ENDPOINT_DESC = (
    "the coarse3 logits reparameterised as log P(from | z18) + log P(to | z24) + a learned "
    "9-scalar transition prior, with ONE shared state head g read at both dates. N0's "
    "finding is that `Artificial -> Cropland` (46 plots) cannot support a decision "
    "boundary against 4,200 stable plots under a 9-way softmax; this head never draws "
    "that boundary -- the rare cell is a product of two well-supported marginals. NOT the "
    "tested-negative `head='bilinear'`, which reads two separate heads off one fused "
    "representation with no date structure tying them together; the shared encoder is "
    "what makes g(z18) and g(z24) the same question asked twice.")

siam_idea("siam_endpoint_pure", head="endpoint_pure",                     # O1
          desc="O1: " + _ENDPOINT_DESC + " `_pure` is the strong form: the factorisation "
               "IS the head, with no residual escape, so a gain cannot be a wider head "
               "in disguise.")

siam_idea("siam_endpoint", head="endpoint",                               # O1b
          desc="O1b: O1 plus a ZERO-initialised linear residual over the fused "
               "representation, so training starts at exactly the pure factorised model "
               "and the residual has to earn what it moves. Run as a pair with O1 because "
               "a full-rank residual can in principle drown the factorisation out, and "
               "the pair is what says whether it did.")

s2off_siam_idea("siam_s2off_endpoint", head="endpoint",                   # O1c
                desc="O1c: the endpoint head on the section's best model (N8b). The "
                     "residual form is mandatory here, not a choice: the factorised term "
                     "reads the AlphaEarth endpoint embeddings only, so `endpoint_pure` on "
                     "a two-tower would silently discard the privileged Sentinel-2 detail "
                     "tower from the coarse3 read.")

siam_idea("siam_endpoint_state", head="endpoint", siam_state_weight=0.3,  # O1d
          siam_state_source="endogenous",
          desc="O1d: O1b with the ENDOGENOUS state head of N14b -- which, once the head is "
               "the output parameterisation rather than a discarded auxiliary, supervises "
               "the same parameters directly. N14b found the endogenous control (not the "
               "external GLanCE pool) was what moved `focus_macro_f1`; this is that finding "
               "wired to the output instead of thrown away at predict time.")


# -- O2: decoupled classifier retraining (cRT) -------------------------------
_CRT_DESC = (
    "Kang et al. (2020): freeze the representation, retrain ONLY the head on a "
    "class-balanced draw. No new parameters and no serving cost -- the served graph is "
    "identical and only the head's weights differ. This project's G-H sampling negative is "
    "the argument FOR it rather than against: Kang et al.'s central finding is that "
    "balanced sampling damages the representation while helping the classifier, and G-H "
    "applied it during JOINT training, which is the losing configuration.")

siam_idea("siam_cos_crt", crt_epochs=30,                                  # O2
          desc="O2: " + _CRT_DESC)

s2off_siam_idea("siam_s2off_crt", crt_epochs=30,                          # O2b
                desc="O2b: O2 on the section's best model, so the cheapest lever is "
                     "measured on the model it would actually be applied to.")


# -- O2c: prototype (cosine) classifier --------------------------------------
_PROTO_DESC = (
    "replace the linear fine head with a cosine classifier: logit_k = s * cos(rep, w_k), "
    "one learnable prototype per coarse3 class and one learnable scale. A linear head "
    "trained on a 4,200-vs-46 split develops class weight NORMS proportional to class "
    "frequency, so the rare class loses on magnitude before its direction is consulted -- "
    "focal loss reweights the loss but leaves that geometry untouched. Normalising both "
    "sides deletes the magnitude channel. Same decoupling insight as O2, moved from the "
    "training schedule into the parameterisation (Kang et al.'s tau-normalisation).")

siam_idea("siam_cos_proto", head="proto", desc="O2c: " + _PROTO_DESC)     # O2c

s2off_siam_idea("siam_s2off_proto", head="proto",                         # O2d
                desc="O2d: O2c on the section's best model.")


# -- O5: a conv detail tower over the stored 64x64 patches -------------------
S2_SHARDS = "s2_shards"
#: Centre crop taken from each stored 64x64 (640 m) patch. 32 px is 320 m, still
#: ten times the 3 px window the deployed detail statistics are computed over,
#: and it quarters the tensor -- which matters because the recipe trains
#: FULL-BATCH, so the whole training block's imagery is resident at once.
PATCH_CROP = 32


def load_patch_tensor(crop: int = PATCH_CROP):
    """(N, years, bands, crop, crop) uint8 patch tensor + its PLOTID order.

    Stored as uint16 reflectance; rescaled per band to [0, 1] against a fixed
    10,000 divisor (the Sentinel-2 L2A convention) and held as **uint8** so the
    whole array is ~100 MB rather than ~1.7 GB as float32. The conv tower casts
    a batch to float on gather. Quantisation to 256 levels is far below the
    noise floor of the measurement and is what makes full-batch training of a
    conv tower possible at all here.
    """
    import torch

    shards = sorted(project_data_dir("embeddings", S2_SHARDS).glob("shard_*.npz"))
    if not shards:
        raise SystemExit(f"no patch shards under {S2_SHARDS}")
    tiles, ids = [], []
    for path in shards:
        z = np.load(path, allow_pickle=True)
        arr = z["patches"]                       # (n, years, bands, 64, 64)
        lo = (arr.shape[-1] - crop) // 2
        arr = arr[..., lo:lo + crop, lo:lo + crop]
        # Non-finite pixels are stored as a separate mask; zero them so a bad
        # pixel reads as "no reflectance" rather than as whatever uint16 garbage
        # happened to be written there.
        finite = z["finite"][..., lo:lo + crop, lo:lo + crop]
        arr = np.where(finite, arr, 0)
        scaled = np.clip(arr.astype("float32") / 10000.0, 0, 1) * 255.0
        tiles.append(scaled.astype("uint8"))
        ids.append(z["plotid"])
    return torch.from_numpy(np.concatenate(tiles)), np.concatenate(ids)


_PATCH_CACHE = {}


def _patches():
    if "t" not in _PATCH_CACHE:
        _PATCH_CACHE["t"] = load_patch_tensor()
    return _PATCH_CACHE["t"]


def patch_idea(name, desc, **overrides):
    """A two-tower idea whose DETAIL tower is a conv encoder over the patches."""
    def fn(ctx, view, seed):
        tensor, ids = _patches()
        aef = ctx.aef_cols
        kwargs = dict(
            arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"],
            tower_dim=256, aef_columns=aef,
            # The detail tower reads the patch tensor, not these columns, so the
            # block is a single harmless column carried only to keep the packed
            # layout the trunk expects. Everything else is the deployed recipe.
            tess_columns=[S2_MASK],
            mask_column=S2_MASK, aef_mask_column=AEF_MASK,
            fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
            aef_siam=True, siam_dim=128, siam_combine="conc",
            siam_cos_weight=0.3, siam_cos_margin=0.3,
            patch_tensor=tensor, patch_ids=ids,
        )
        kwargs.update(overrides)
        return s2off_cv_fine(view, aef + [S2_MASK], kwargs, seed)
    return register(name, reads=("full",), group="section-o", desc=desc)(fn)


patch_idea(                                                               # O5
    "siam_s2off_patch",
    "O5: the section's best model with its 78-column hand-built detail tower replaced by a "
    "small conv encoder over the stored 32x32 Sentinel-2 patches, with the eight dihedral "
    "augmentations. S3 tested a FLATTENED 8x8 pooled patch (1,344 raw columns) and it was "
    "the worst result on the board -- but that tested flattening, not a spatial encoder: "
    "no weight sharing across the image, and no geometry to augment. Still privileged and "
    "still gate-off served, so no patch is read at inference.")

patch_idea(                                                               # O5b
    "siam_s2off_patch_noaug", patch_augment=False,
    desc="O5b: O5 without the dihedral augmentations, so a result is attributable to the "
         "spatial encoder or to the extra data and not to both at once.")


# -- O3: the coarse3 cost gate (N6) ------------------------------------------
def coarse3_cost_gate(view: View, fine_probs: np.ndarray, fine_classes: list,
                      passes: int = 2) -> np.ndarray:
    """Nested per-class decision costs on the COARSE3 level.

    N12's instrument limitation, answered. The existing cost gate reweights
    *merged2* classes, so it leaves every commissioned transition exactly
    unchanged to four decimals and cannot buy the classes section N is scored
    on. This is the same construction one level down: for each outer fold the
    per-class multipliers that maximise ``focus_macro_f1`` on the *other* folds
    are selected and applied to the held-out one, so the operating point is
    never chosen on the data it is scored on.

    Only the four commissioned transitions are given a free multiplier. The
    alternative -- all nine -- is nine parameters fitted on four inner folds,
    and F3 already found the wide search is as much a test of whether the folds
    can support the freedom as of whether the freedom pays.
    """
    arr = np.array(fine_classes, dtype=object)
    targets = [i for i, c in enumerate(fine_classes) if c in FOCUS_TRANSITIONS]
    labels = np.empty(len(view.truth_fine), dtype=object)

    def score(idx, costs):
        picked = arr[(fine_probs[idx] * costs).argmax(1)]
        return focus_metrics(view.truth_fine[idx], picked)["focus_macro_f1"]

    for tr, te in view.folds:
        costs = np.ones(len(fine_classes))
        best = score(tr, costs)
        for _ in range(passes):
            for j in targets:
                for m in COST_GRID:
                    trial = costs.copy()
                    trial[j] = m
                    got = score(tr, trial)
                    if got > best:
                        best, costs = got, trial
        labels[te] = arr[(fine_probs[te] * costs).argmax(1)]
    return labels


def coarse3_gate_idea(name, source, desc):
    """Register a coarse3 cost-gate re-read over one cached model's fine probs."""
    def fn(ctx, view, seed):
        cached = load_oof(source, view.name, seed)
        fine_cached = load_oof_fine(source, view.name, seed)
        if cached is None or fine_cached is None:
            raise RuntimeError(
                f"{name} needs cached fine probabilities for {source} on "
                f"{view.name}: run --ideas {source} first")
        probs, _ = cached
        fine_probs, fine_classes = fine_cached
        gated = coarse3_cost_gate(view, fine_probs, fine_classes)
        # The merged2 probabilities are returned untouched, so every aggregate
        # column is identical to the source model by construction and any move
        # in the focus columns is attributable to the gate alone.
        return probs, gated, None, (fine_probs, fine_classes)
    return register(name, reads=("full",), group="section-o", desc=desc)(fn)


coarse3_gate_idea(                                                        # O3
    "c3gate_siam_cos", "base_siam_cos_fine",
    "O3 (= N6, the one modelling item section N left open): a cost gate at the COARSE3 "
    "level over `siam_cos`. N12 established the merged2 gate cannot touch the commissioned "
    "transitions; this is the instrument that can. Costs nothing -- a different arg-max "
    "over probabilities already computed.")

coarse3_gate_idea(                                                        # O3b
    "c3gate_siam_s2off_cos", "base_siam_s2off_cos_fine",
    "O3b: O3 on the section's best model.")

coarse3_gate_idea(                                                        # O3c
    "c3gate_deployed", "base_deployed_fine",
    "O3c: O3 on the DEPLOYED model, so the incumbent is gated by the same instrument as "
    "the challengers. Without this row a free decision-rule gain would read as an "
    "architecture win -- the misattribution N12 was built to prevent.")

# -- O4: the composition. O1 and O3 both break N0's dead class, by different --
# means: O1 by making the rare cell a product of two supported marginals, O3 by
# moving the decision threshold on the existing distribution. Whether they
# compose is not predictable from either -- F7 and N3b both found that two
# mechanisms correcting the SAME thing land between their parts. If that is what
# happens here, the two are one lever and the free one wins on cost alone.
coarse3_gate_idea(                                                        # O4
    "c3gate_endpoint_pure", "siam_endpoint_pure",
    "O4: the coarse3 gate over the endpoint head. Preregistered prediction: they do NOT "
    "compose, because O1 already spends its Artificial -> Cropland budget and the gate has "
    "nothing left to buy -- the N3b/F7 signature.")

coarse3_gate_idea(                                                        # O4b
    "c3gate_s2off_endpoint", "siam_s2off_endpoint",
    "O4b: O4 on the two-tower endpoint head, which is the better of the O1 rungs on the "
    "aggregates.")


# ===========================================================================
# Section P -- single-date auxiliary paths that are READ AT PREDICT TIME
# ===========================================================================
# N14 put external single-date state labels into the encoder as a *discarded*
# auxiliary and came back flat, with the endogenous control beating the external
# pool. Section P changes what the single-date path IS, on two axes N14 did not
# vary:
#
#   * **Where it lands.** O1's endpoint head makes the state read the output
#     parameterisation rather than a side-loss thrown away at predict time, so
#     external `from`/`to` supervision trains the exact head that decides the
#     transition. O1 was only ever run with endogenous supervision (O1d) and
#     N14 was only ever run with a flat head -- the cell where the two meet is
#     empty, and it is the cell the "a cropland model verifies from-Cropland
#     and to-Cropland" framing actually points at.
#   * **Whether it is a loss at all.** P3 does not touch the encoder: it fits
#     the external state model separately, applies it to *both* endpoint blocks,
#     and hands the model nine posterior columns. That is the literal reading of
#     "use a cropland model to verify" -- and unlike a state head it needs no
#     shared encoder, so it is the one form of this idea that would also work on
#     the flat trunk.
#
# P0 (`diagnose_state_year_transfer.py`) is the gate both rest on, and it
# overturns the structural limit N14 recorded. GLanCE ends in 2020, so the pool
# is 2018-only -- but AlphaEarth's space is year-stable enough that the same
# 2018-fitted probe scores the 2024 block as well as the 2018 one (pool ->
# RECOVER macro-F1 0.735 at 2024 against 0.733 at 2018, against a 2024 self-floor
# of 0.744). The `-> Cropland in 2024` half is therefore reachable from the pool
# that is already built, and no 2024 label source is required.


def cv_probs_state_fine(view: View, cols: list, kwargs: dict, seed: int, pool,
                        s2off: bool = False):
    """``cv_probs_state`` that also caches the coarse3 probabilities.

    Section O's gate is free and is the standing recommendation on the
    commissioned transitions, so every section-P idea has to arrive with a fine
    distribution or it cannot be compared against a gated incumbent -- N12's
    misattribution trap, one level down.

    The pool is cut to each fold's training blocks exactly as in
    ``cv_probs_state``; see that docstring for why a global pool must be split
    where the Oslo unlabelled pool did not have to be.
    """
    n = len(view.target)
    classes = view.merged_classes
    fine_classes = sorted(set(view.truth_fine))
    probs = np.zeros((n, len(classes)), dtype="float64")
    fine_probs = np.zeros((n, len(fine_classes)), dtype="float64")
    fine = np.empty(n, dtype=object)
    for tr, te in view.folds:
        fold_pool = None
        if pool is not None:
            train_blocks = set(view.frame.iloc[tr]["block_id"].unique())
            fold_pool = pool[pool["block_id"].isin(train_blocks)]
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy(),
                      state_frame=fold_pool)
            te_frame = view.frame.iloc[te]
            if s2off:
                te_frame = te_frame.copy()
                te_frame[S2_MASK] = 0.0        # the deployed gate-off read
            p_fine, p_merged = model._probs(te_frame)
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine_probs[np.ix_(te, [fine_classes.index(c)
                               for c in model.fine_classes_])] = p_fine
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine, None, (fine_probs, fine_classes)


def state_endpoint_idea(name, desc, *, base="cos", weight=SIAM_AUX,
                        source="external", head="endpoint", **overrides):
    """A state-supervised idea whose state head IS the output parameterisation."""
    def fn(ctx, view, seed):
        pool = None
        if source in ("external", "both"):
            pool = _state_pool()
            if pool is None:
                raise SystemExit(
                    f"{name} needs {STATE_POOL}; run build_state_labels.py first")
        extra = dict(siam_state_weight=weight, siam_state_source=source,
                     head=head, **overrides)
        if base == "s2off":
            cols, kwargs = siam_s2off_kwargs(
                ctx, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3, **extra)
            return cv_probs_state_fine(view, cols, kwargs, seed, pool, s2off=True)
        kwargs = siam_kwargs(ctx, siam_cos_weight=SIAM_AUX,
                             siam_cos_margin=0.3, **extra)
        return cv_probs_state_fine(view, siam_all_cols(ctx), kwargs, seed, pool)
    return register(name, reads=("full",), group="section-p", desc=desc)(fn)


def state_pretrain_idea(name, desc, *, base="cos", epochs=30,
                        source="external", shuffle_labels=False, **overrides):
    """Pretrain the shared encoder on single-date states, then fit normally.

    Sections N14 and P varied WHERE the state path lands -- discarded side-loss,
    output parameterisation, input feature -- and all three came back flat. Every
    one of them trained the state objective JOINTLY with the transition loss, so
    "when" is the axis none of them moved. Here the pool trains the encoder on
    its own first (`siam_state_pretrain` epochs), and the transition fit then
    starts from those weights with the auxiliary term switched OFF -- no
    weighting between the two objectives at all, and the encoder gets the pool's
    full capacity rather than a 0.3-weighted share of a gradient step.

    P5 predicts this is flat for a reason that has nothing to do with the
    schedule: GLanCE's errors are correlated with AlphaEarth's whatever route
    they take, and error independence -- not accuracy, not placement -- is what a
    single-date path is bought on. Registered so that prediction is measured
    rather than assumed, and with the endogenous control beside it because P's
    fourth finding is that the naive reading without one would have been
    "+0.002 from GLanCE" twice over.
    """
    def fn(ctx, view, seed):
        pool = None
        if source in ("external", "both"):
            pool = _state_pool()
            if pool is None:
                raise SystemExit(
                    f"{name} needs {STATE_POOL}; run build_state_labels.py first")
            if shuffle_labels:
                # Same rows, same 83 blocks, same per-fold split, same number of
                # gradient steps -- only the label-to-embedding correspondence
                # destroyed. Permuted once per seed rather than per fold so the
                # arm is one consistent wrong pool, not a fresh one each fold.
                pool = pool.copy()
                pool["state"] = (pool["state"]
                                 .sample(frac=1.0, random_state=seed)
                                 .to_numpy())
        # The cosine term's own settings go in the dict rather than beside it, so
        # an arm may OVERRIDE them (Q10c raises the weight) instead of colliding
        # with a duplicate keyword. Unchanged for every arm that does not.
        extra = dict(siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,
                     siam_state_weight=0.0, siam_state_source=source,
                     siam_state_pretrain=epochs)
        extra.update(overrides)
        if base == "s2off":
            cols, kwargs = siam_s2off_kwargs(ctx, **extra)
            return cv_probs_state_fine(view, cols, kwargs, seed, pool, s2off=True)
        kwargs = siam_kwargs(ctx, **extra)
        return cv_probs_state_fine(view, siam_all_cols(ctx), kwargs, seed, pool)
    return register(name, reads=("full",), group="section-p", desc=desc)(fn)


state_pretrain_idea(                                                      # P7a
    "siam_cos_state_pre",
    desc="P7a: 30 epochs of g(f(x)) -> state on the GLanCE pool alone, then the N2 fit "
         "from those weights with the state term off. The pretrain-then-finetune cell "
         "sections N14 and P never entered: all six of their arms weighted the two "
         "objectives against each other inside one optimiser step.")

state_pretrain_idea(                                                      # P7b
    "siam_cos_state_pre_endo", source="endogenous",
    desc="P7b: the CONTROL. The identical phase, the identical epoch count, pretrained on "
         "the TRAINING PLOTS' own endpoints -- a From -> To label is two free state labels, "
         "so it adds no data. Separates 'a state-organised initialisation helps' from "
         "'GLanCE's 13,118 labels help'. N14b and P3b both had the control win.")

state_pretrain_idea(                                                      # P7c
    "siam_cos_state_pre_long", epochs=100,
    desc="P7c: P7a at 100 pretrain epochs. Preregistered so a flat P7a cannot be dismissed "
         "as an undertrained phase -- 30 epochs over 13k rows at batch 2048 is ~210 steps, "
         "and the main fit gets 30. If the pool has anything to give the encoder, more of "
         "it is where that shows.")


state_pretrain_idea(                                                      # P7g
    "siam_cos_state_pre_shuf", shuffle_labels=True,
    desc="P7g: the SECOND control, and the one P7b leaves open. The identical pool, epochs "
         "and step count with the state labels permuted, so the phase still moves the "
         "encoder the same distance for the same cost but the labels carry no land cover. "
         "P7b (endogenous, no new data) came back at 0.044 on `Artificial -> Cropland` "
         "rather than 0.000, so 'any pretraining at all' is a live alternative explanation "
         "for P7a and this is what rules it in or out.")

state_pretrain_idea(                                                      # P7e
    "siam_s2off_state_pre", base="s2off",
    desc="P7e: P7a on the deployed recipe's base (N8b), still gate-off, so no Sentinel-2 "
         "is read at inference and serving cost is unchanged. A result on one base is a "
         "result on one base -- N18's false verdict came from not asking this -- and this "
         "is the base a deployment decision would actually be taken on.")

state_pretrain_idea(                                                      # Y3
    "siam_s2off_state_pre_cw", base="s2off",
    siam_state_class_weight="balanced",
    desc="Y3: P7e with the pretraining head's cross-entropy weighted by 1/class frequency, "
         "and nothing else changed. From the user's read of the Oslo map -- cropland has "
         "room to grow but must take pixels from Nature, not from the built-up classes. "
         "STATE_PRETRAIN_RESEARCH section Y added `artificial_as_cropland` to measure that "
         "and swept encoder capacity looking for it: four arms up to 3.8x the parameters "
         "moved the SUM of the two cropland errors and never the ratio, and `mlp_cw` moved "
         "the ratio at both fold counts -- `f1_cropland` +0.006/+0.004 and "
         "`nature_as_cropland` +0.008/+0.009 up, `artificial_as_cropland` -0.004/-0.004 "
         "down. It costs state macro-F1 (-0.001/-0.005), so it is a trade, and the two "
         "state-level gains it is bought on were not measured when V3 first recommended "
         "it. Judge on `art_stable_recall` and the map against change-F1, not on either "
         "alone; P7i's lesson is that a state-level gain need not survive the map.")

state_pretrain_idea(                                                      # P7k
    "siam_s2off_state_pre_endo", base="s2off", source="endogenous",
    desc="P7k: P7b's endogenous control moved onto the deployed base, because P7b ran on "
         "`cos` and P7e on `s2off` and the two were never put on one base. Without it the "
         "'both' arm below has no control that shares its trunk: a difference between P7e "
         "and P7i could be the pool or could be the endogenous half's presence at all.")

state_pretrain_idea(                                                      # P7i
    "siam_s2off_state_pre_both", base="s2off", source="both",
    desc="P7i: the pretrain pool is GLanCE AND the training plots' own endpoints, not "
         "either alone. STATE_PRETRAIN_RESEARCH U1c/U3 measure that union at +0.0234 "
         "paired over GLanCE alone on an LLTO *state* read, 24/25 folds, reproducing at "
         "20 folds under V0's fixed geometry -- where P7e's external-only pool and P7h's "
         "endogenous-only control are the two halves it beats. This is the transition-level "
         "test that section's recommendation was waiting on, and section W's lesson applies "
         "in full: a plot-level state gain need not survive the map's 0.5% base rate. "
         "Leak-free by construction -- the endogenous half is read from each fold's own "
         "`tr_idx` inside `fit`, never from a pool file (tests/test_state_pool_leak.py).")

coarse3_gate_idea(                                                        # P7m
    "c3gate_siam_s2off_state_pre_both", "siam_s2off_state_pre_both",
    "P7m: O3's free coarse3 gate over P7i, so the union pool is read on the same decision "
    "rule P7f gives P7e. The arg-max and the gate break `Artificial -> Cropland` by "
    "different routes and the deployment comparison is gate-to-gate.")

coarse3_gate_idea(                                                        # P7f
    "c3gate_siam_s2off_state_pre", "siam_s2off_state_pre",
    "P7f: the gate over P7e, against `c3gate_siam_s2off_cos` -- the two-tower half of the "
    "P7d question, and the comparison a deployment would be decided on.")

coarse3_gate_idea(                                                        # P7d
    "c3gate_siam_cos_state_pre", "siam_cos_state_pre",
    "P7d: O3's free coarse3 gate over the state-pretrained encoder. P7a breaks `Artificial "
    "-> Cropland` at the arg-max, which is the same class and the same failure O3 breaks "
    "with a decision rule -- so the question is whether a representation that has seen "
    "single-date states and a gate that reweights the softmax are two routes to one gain or "
    "two gains. The incumbent on the commissioned transitions is O3 at 0.4412, not the "
    "arg-max, and P7a's 0.4307 does not reach it.")


_P1_DESC = (
    "the O1b endpoint head -- logit(A -> B) = log P(from = A | z18) + log P(to = B | z24) "
    "+ prior -- with its ONE shared state head g supervised by the external GLanCE pool. "
    "N14 put the same labels into a head that was thrown away at predict time and the "
    "endogenous control beat them; O1d wired the head to the output but fed it only the "
    "plots' own endpoints. Here the external labels train the head that actually decides "
    "the transition, which is what 'a cropland model verifies from-Cropland and "
    "to-Cropland' means if it means anything. P0 is what makes the `to` half legitimate: "
    "the 2018 pool reads the 2024 block as well as it reads 2018.")

state_endpoint_idea("siam_endpoint_state_ext",                            # P1
                    desc="P1: " + _P1_DESC)

state_endpoint_idea("siam_endpoint_state_both", source="both",            # P1b
                    desc="P1b: P1 with the endogenous term kept alongside the external "
                         "one. N14b found the endogenous half was carrying the whole "
                         "`focus_macro_f1` gain, so dropping it to add GLanCE would trade "
                         "the known effect for the unknown one; this is the row that says "
                         "whether they add.")

state_endpoint_idea("siam_endpoint_state_ext_strong", weight=1.0,         # P1c
                    desc="P1c: P1 at weight 1.0, a co-objective rather than a "
                         "regulariser. Preregistered because both N4 and N14d found "
                         "external supervision at strength buys aggregates by trading the "
                         "commissioned classes away -- `Cropland -> Artificial` and "
                         "`art_stable_recall` declining monotonically with the weight is "
                         "the signature to check for, not a surprise to discover later.")

state_endpoint_idea("siam_s2off_endpoint_state", base="s2off",            # P2
                    desc="P2: P1 on the section's best model (N8b), still gate-off, so no "
                         "Sentinel-2 is read at inference and serving cost is unchanged. "
                         "The residual form of the head is mandatory on a two-tower -- "
                         "`endpoint_pure` would silently discard the privileged detail "
                         "tower from the coarse3 read (O1c).")


# -- P3: the external state model as INPUT, not as a loss --------------------
#: Nine columns: P(state | 2018), P(state | 2024) and the difference. The diff is
#: carried explicitly for the same reason the raw AlphaEarth `diff` block is
#: (N5): the model can form it, but not for free, and this is the term the whole
#: idea is about -- "was Cropland, is not any more".
#:
#: The ``_y18``/``_y24`` suffixes are load-bearing and must NOT be ``_2018`` /
#: ``_2024``: ``_aef_siam_permutation`` splits the AlphaEarth tower's columns on
#: exactly those endings, so a year-suffixed prior column would pair up cleanly,
#: pass every assertion, and be fed silently through the shared *encoder*
#: instead of arriving as a flat extra. That would still train -- and it would
#: no longer be the same idea as the AlphaEarth-only rung it is compared against.
PRIOR_STATES = ("artificial", "cropland", "nature")
PRIOR_COLS = ([f"pstate_{s}_y18" for s in PRIOR_STATES]
              + [f"pstate_{s}_y24" for s in PRIOR_STATES]
              + [f"pstate_{s}_delta" for s in PRIOR_STATES])


def _state_probe():
    """The same linear probe N14a cleared the pool's legend with.

    Deliberately not a second neural network. The point of P3 is to test whether
    an external state *reading* helps once it is available at both endpoints,
    and a probe with the same form as the diagnostic keeps the number
    comparable to N14a's transfer table instead of introducing a new model whose
    own capacity is a confound.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def attach_state_prior(ctx, frame, X_fit, y_fit):
    """``frame`` plus the probe's posterior read at BOTH endpoint blocks.

    One probe, applied to the 2018 block and then to the 2024 block -- the same
    "one function of a single date asked twice" structure the siamese encoder
    has, which is what lets a 2018-only pool speak about 2024 (P0).
    """
    a18, a24, _ = siam_columns(ctx)
    probe = _state_probe().fit(X_fit, y_fit)
    # Column order is pinned to PRIOR_STATES rather than inherited from
    # ``classes_``: a fold whose probe never saw a state would otherwise shift
    # every column silently, which is the failure cv_probs_fine guards against
    # one level up.
    missing = [s for s in PRIOR_STATES if s not in set(probe.classes_)]
    if missing:
        raise ValueError(f"state probe never saw {missing}")
    order = [list(probe.classes_).index(s) for s in PRIOR_STATES]
    p18 = probe.predict_proba(frame[a18].astype("float64").to_numpy())[:, order]
    p24 = probe.predict_proba(frame[a24].astype("float64").to_numpy())[:, order]
    out = frame.copy()
    for col, values in zip(PRIOR_COLS, np.hstack([p18, p24, p24 - p18]).T):
        out[col] = values
    return out


def _endogenous_state_fit(ctx, view, tr):
    """(X, y) for a probe trained on the TRAINING plots' own two endpoints.

    The control that makes P3 readable, built the same way N14b's was: a
    ``From -> To`` label is two free state labels, so this probe has exactly the
    same form and the same access to both years as the external one and differs
    only in where its labels came from. Test-fold rows are never in ``tr``, so
    the probe applied to them has seen neither their labels nor their block.
    """
    a18, a24, _ = siam_columns(ctx)
    frame = view.frame.iloc[tr]
    # Rare-pooled rows carry no " -> " and so name no endpoint state; they are
    # dropped from the probe's training set rather than guessed at.
    pairs = [c.split(" -> ") if " -> " in c else None for c in view.truth_fine[tr]]
    keep = np.array([p is not None for p in pairs])
    X = np.vstack([frame[a18].astype("float64").to_numpy()[keep],
                   frame[a24].astype("float64").to_numpy()[keep]])
    y = np.array([p[0].lower() for p, k in zip(pairs, keep) if k]
                 + [p[1].lower() for p, k in zip(pairs, keep) if k])
    return X, y


def simulated_state_prior(view: View, accuracy: float, seed: int):
    """``view.frame`` plus a synthetic single-date state read of known accuracy.

    **A ceiling, never a candidate.** P1-P3 answer "does the pool we have help";
    this answers the question underneath, which is the one that decides whether
    to go looking for a better pool at all: *how accurate would a single-date
    product have to be before it is worth acquiring?* P0 puts both GLanCE and
    the plots' own labels at ~0.74 state accuracy, so the interesting part of
    the curve is above that.

    The reader is corrupted independently at each date -- with probability
    ``accuracy`` it names the true coarse3 state, otherwise it picks uniformly
    among the other two -- which is what a real single-date product does and is
    why the sweep is not degenerate below 1.0. It **is** degenerate at 1.0: two
    exact endpoint states determine the coarse3 transition by definition, so
    that end of the curve measures the legend, not a model.

    Applied identically to training and held-out rows, because a real product
    would be. Drawn off the model seed, so the corruption is redrawn per seed
    and the reported spread includes it rather than hiding it.
    """
    rng = np.random.default_rng(1000 + seed)
    states = np.array(PRIOR_STATES)
    pairs = [c.split(" -> ") if " -> " in c else None for c in view.truth_fine]
    out = view.frame.copy()
    reads = []
    for side in (0, 1):
        true_idx = np.array([list(states).index(p[side].lower()) if p else -1
                             for p in pairs])
        keep = rng.random(len(true_idx)) < accuracy
        # The wrong answer is drawn from the OTHER two states, so `accuracy` is
        # the reader's accuracy exactly rather than 1/3 of the way to it.
        offset = rng.integers(1, len(states), size=len(true_idx))
        read = np.where(keep, true_idx, (true_idx + offset) % len(states))
        onehot = np.zeros((len(true_idx), len(states)))
        known = true_idx >= 0
        onehot[known] = np.eye(len(states))[read[known]]
        # A rare-pooled row names no endpoint state; it gets a flat read rather
        # than a guess, which is what a product would return for "unknown".
        onehot[~known] = 1.0 / len(states)
        reads.append(onehot)
    for col, values in zip(PRIOR_COLS,
                           np.hstack([reads[0], reads[1],
                                      reads[1] - reads[0]]).T):
        out[col] = values
    return out


def cv_probs_prior_fine(ctx, view, kwargs_fn, seed, source, s2off=False):
    """Blocked CV where each fold's frame carries a fold-local state posterior.

    The probe is refitted per fold -- on the pool rows in that fold's training
    blocks, or on that fold's training plots -- and only then applied to the
    held-out rows. Precomputing the nine columns once over the whole pool would
    be a block-level leak of exactly the kind ``cv_probs_state`` exists to
    avoid, and it would be invisible in every metric.
    """
    n = len(view.target)
    classes = view.merged_classes
    fine_classes = sorted(set(view.truth_fine))
    probs = np.zeros((n, len(classes)), dtype="float64")
    fine_probs = np.zeros((n, len(fine_classes)), dtype="float64")
    fine = np.empty(n, dtype=object)
    pool = _state_pool() if source == "external" else None
    if source == "external" and pool is None:
        raise SystemExit(f"needs {STATE_POOL}; run build_state_labels.py first")
    a18, _, _ = siam_columns(ctx)

    for tr, te in view.folds:
        if source.startswith("oracle:"):
            frame = simulated_state_prior(view, float(source.split(":")[1]), seed)
        elif source == "external":
            train_blocks = set(view.frame.iloc[tr]["block_id"].unique())
            fold_pool = pool[pool["block_id"].isin(train_blocks)]
            X_fit = fold_pool[a18].astype("float64").to_numpy()
            y_fit = fold_pool["state"].astype(str).to_numpy()
            frame = attach_state_prior(ctx, view.frame, X_fit, y_fit)
        else:
            X_fit, y_fit = _endogenous_state_fit(ctx, view, tr)
            frame = attach_state_prior(ctx, view.frame, X_fit, y_fit)
        cols, kwargs = kwargs_fn(ctx)
        model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame.iloc[tr], view.target.iloc[tr].to_numpy())
            te_frame = frame.iloc[te]
            if s2off:
                te_frame = te_frame.copy()
                te_frame[S2_MASK] = 0.0
            p_fine, p_merged = model._probs(te_frame)
        probs[np.ix_(te, [classes.index(c) for c in model.merged_classes_])] = p_merged
        fine_probs[np.ix_(te, [fine_classes.index(c)
                               for c in model.fine_classes_])] = p_fine
        fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
    return probs, fine, None, (fine_probs, fine_classes)


def prior_idea(name, desc, *, base="cos", source="external", **overrides):
    """A state-posterior-as-input idea over N2 (`cos`) or N8b (`s2off`)."""
    def fn(ctx, view, seed):
        def kwargs_fn(c):
            _, _, diff = siam_columns(c)
            if base == "s2off":
                cols, kwargs = siam_s2off_kwargs(
                    c, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3, **overrides)
                # The nine columns join the AlphaEarth side, not the privileged
                # detail tower: they are read at inference and must survive the
                # gate-off serve, which anything in `tess_columns` does not.
                kwargs = dict(kwargs, aef_columns=kwargs["aef_columns"] + PRIOR_COLS)
                return cols + PRIOR_COLS, kwargs
            extras = diff + PRIOR_COLS
            return (siam_all_cols(c, extra_cols=extras),
                    siam_kwargs(c, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,
                                extra_cols=extras, **overrides))
        return cv_probs_prior_fine(ctx, view, kwargs_fn, seed, source,
                                   s2off=(base == "s2off"))
    return register(name, reads=("full",), group="section-p", desc=desc)(fn)


prior_idea(                                                               # P3
    "siam_cos_prior",
    "P3: N2 plus nine columns -- P(Artificial/Cropland/Nature | 2018), the same at 2024, "
    "and the difference -- from a linear probe fitted on the GLanCE pool and applied to "
    "both endpoint blocks. No auxiliary loss and no shared parameters: the external "
    "cropland reading enters as evidence the transition head can consult, which is what "
    "'a model that predicts croplands verifies from-Cropland and to-Cropland' says "
    "literally. The one form of this idea that does not need a siamese encoder, so a win "
    "here transfers to the flat trunk and a state head never could.")

prior_idea(                                                               # P3b
    "siam_cos_prior_endo",
    "P3b: the CONTROL, and the row that makes P3 readable. The identical nine columns from "
    "a probe fitted on the TRAINING PLOTS' own two endpoints -- same form, same access to "
    "both years, no new labels. N14b's design, and its lesson: reported without this "
    "control a gain would read as '+x from GLanCE' and, last time, that reading was wrong.",
    source="endogenous")

prior_idea(                                                               # P3c
    "siam_s2off_prior",
    "P3c: P3 on the section's best model (N8b), gate-off. The nine columns ride the "
    "AlphaEarth tower rather than the privileged detail tower, so they survive the "
    "serve -- and they cost one 64-column logistic regression per date at inference, "
    "which is the whole added serving cost of the idea.", base="s2off")


# -- P5: how good would a single-date source have to be? ---------------------
# The question P1-P3 cannot answer. They say the pool that exists does not help;
# they do not say whether a better one would, and that is the decision the user
# actually faces -- whether to go and get one. P0 measures both GLanCE and the
# plots' own labels at ~0.74 state accuracy on these embeddings, so the sweep
# starts just above where every source currently available sits and runs to the
# degenerate end.
for _acc in (0.74, 0.75, 0.85, 0.95, 1.00):
    prior_idea(
        f"prior_oracle_{int(_acc * 100)}",
        f"P5 (a CEILING, not a candidate): P3's nine columns replaced by a synthetic "
        f"single-date reader that is correct {_acc:.0%} of the time at each date "
        f"independently. Read as a curve, not as rows: the slope between 0.75 and 0.95 is "
        f"what a better single-date product would buy, and 1.00 is degenerate by "
        f"construction because two exact endpoint states ARE the coarse3 label. The point "
        f"of the sweep is the threshold -- P0 puts every source currently available, "
        f"GLanCE and the RECOVER plots alike, at ~0.74.",
        source=f"oracle:{_acc}")

coarse3_gate_idea(                                                        # P4
    "c3gate_endpoint_state_ext", "siam_endpoint_state_ext",
    "P4: the free coarse3 gate over P1. O4 found the gate does NOT compose with the "
    "endpoint head -- both spend the same `Artificial -> Cropland` budget -- so this is "
    "registered to check whether external state supervision changes that, not because a "
    "gain is expected.")

coarse3_gate_idea(                                                        # P4b
    "c3gate_siam_cos_prior", "siam_cos_prior",
    "P4b: the free coarse3 gate over P3. Unlike P4 there is no reason to expect "
    "interference here -- P3 changes the evidence, O3 changes the arg-max over it -- so "
    "this is the row that would actually ship if P3 clears.")


# ===========================================================================
# Section Q -- the burned-area Swin network's modules, transcribed
# ===========================================================================
# Zhang et al. (RSE 2025) report five design choices behind a Swin-Transformer
# burned-area change detector. Three of them are architecture-for-image-grids and
# have no form here (see the section head in SIAMESE_RESEARCH.md); the ones below
# are the parts that survive the translation to a plot-level tabular encoder.
# Each is registered over `siam_cos` (N2), which is the section-N base every
# other idea in N/O/P was measured against, so the comparison is on the folds
# and seeds the whole ledger uses.


def _siam_q_idea(name, desc, **overrides):
    """A section-Q variant of N2: the cosine objective plus one paper module."""
    return model_idea(
        name, cols_fn=siam_all_cols,
        kwargs_fn=lambda c: siam_kwargs(c, siam_cos_weight=SIAM_AUX,
                                        siam_cos_margin=0.3, **overrides),
        group="section-q", reads=("full",), desc=desc)


# -- Q1: does channel count buy anything? (their "multi-band input") ---------
def s2_channel_columns(stat: list, subset: str, channels: tuple) -> list:
    """A named detail-tower subset restricted to some of the seven channels.

    Built fraction is kept in every rung whatever ``channels`` says. It is a
    spatial statistic (the fraction of an NDVI-thresholded neighbourhood) and so
    is not derivable from any of the per-channel columns, which means dropping it
    would confound a channel-count ladder with a families change. Keeping it
    makes the ladder *conservative* for the paper's claim -- the three-band rung
    still gets a near-infrared-derived summary through the back door, so the
    marginal value this measures for adding a band is a lower bound.
    """
    keep = []
    for col in s2_subset_columns(stat, subset):
        if col.startswith("S2bf") or col.split("_")[1] in channels:
            keep.append(col)
    return keep


#: The band ladder, ordered by channel count. `S2c_bright` is the mean of the
#: four reflectance bands, so it only exists once all four are in.
S2_CHANNEL_RUNGS = {
    "b3": ("blue", "green", "red"),
    "b4": ("blue", "green", "red", "nir"),
    "b7": ("blue", "green", "red", "nir", "ndvi", "ndwi", "bright"),
}


def _siam_s2off_bands_idea(name, rung, desc):
    def fn(ctx, view, seed):
        detail = s2_channel_columns(ctx.s2_stat_cols, "centre_m3s3_bf",
                                    S2_CHANNEL_RUNGS[rung])
        cols, kwargs = siam_s2off_kwargs(ctx, siam_cos_weight=SIAM_AUX,
                                         siam_cos_margin=0.3)
        return s2off_cv(view, ctx.aef_cols + detail,
                        dict(kwargs, tess_columns=detail), seed)
    return register(name, reads=("full",), group="section-q", desc=desc)(fn)


for _rung, _n in (("b3", 3), ("b4", 4), ("b7", 7)):
    _siam_s2off_bands_idea(
        f"siam_s2off_{_rung}", _rung,
        f"Q1: N8b's privileged detail tower restricted to {_n} of its seven channels "
        f"({', '.join(S2_CHANNEL_RUNGS[_rung])}), built fraction kept throughout. Read as "
        f"a LADDER, not as rows. The paper's first design choice is that more spectral "
        f"bands of input improve burned-area detection, and its selected set (B12, B11, "
        f"B8A, B4, B3, B2) is half SWIR -- which this project has never extracted, because "
        f"the extractor takes only the four bands native at 10 m. Whether to go and get "
        f"SWIR is a download decision, and the ladder is the cheap gate on it: b3 -> b4 "
        f"adds a genuine band (near-infrared) and b4 -> b7 adds only nonlinear "
        f"recombinations of bands already present, so the two steps separate 'more bands' "
        f"from 'more derived columns'. A flat b3 -> b4 step says channel count is not the "
        f"lever on this target and no extraction is warranted.")


# -- Q3: CRFE, the two halves that have a tabular form ----------------------
_siam_q_idea(
    "siam_cos_crfe_sum",                                                  # Q2
    siam_crfe="sum",
    desc="Q2: CRFE's feature fusion. The paper builds its feature-rich block from four "
         "things -- pre, post, their SUM and their DIFFERENCE -- where the head here reads "
         "[z18, z24, z24-z18, |z24-z18|, cos] and has no sum. Adding z18+z24 is linear in "
         "a block that already contains z18 and z24, so it carries no new information in "
         "principle; the precedent for testing it anyway is that exactly the same is true "
         "of the raw AlphaEarth diff block, whose removal costs -0.048 change-F1 on the "
         "flat trunk and -0.004 here (N5). Cheapest idea in the section and the one with "
         "the clearest prior against it.")

_siam_q_idea(
    "siam_cos_crfe_attn",                                                 # Q3
    siam_crfe="attn",
    desc="Q3: CRFE's channel attention. A squeeze-and-excitation gate over the assembled "
         "block, so the head can down-weight parts of it per plot before the mixer "
         "dense-mixes them. The spatial-squeeze step of Woo et al. is dropped because there "
         "are no spatial dims to squeeze; the excite step is the whole module here. Note "
         "the precedent is mixed rather than absent: an SE gate on the RAW 193-column input "
         "was tried on the flat trunk (experiment_hier_se.py) -- this one sits on the "
         "siamese block, where the redundancy is structural (z18, z24 and their difference) "
         "rather than incidental.")

_siam_q_idea(
    "siam_cos_crfe_full",                                                 # Q3b
    siam_crfe="full",
    desc="Q3b: both CRFE halves, i.e. the published module minus its spatial branch. "
         "Registered so 'CRFE does/does not help' is answerable about the module and not "
         "only about its parts -- and preregistered with the F7 / N3b / O4 prediction that "
         "two mechanisms acting on the same block land between their parts.")

# -- Q4: the pyramid decoder, as depth fusion -------------------------------
_siam_q_idea(
    "siam_cos_pyramid",                                                   # Q4
    siam_pyramid=True,
    desc="Q4: the pyramid up-sampling decoder. Its stated purpose is that small burned "
         "areas lose their features to repeated down-sampling, so shallow high-detail maps "
         "are folded back into the deep semantic one. There is no spatial resolution to "
         "recover in a tabular encoder, so what transfers is the depth half: project both "
         "hidden stages to the endpoint width and fold them into z bottom-to-top, "
         "zero-initialised so training starts at exactly the plain encoder. The mechanism "
         "to watch is the rare transitions -- if the analogy holds at all, the classes that "
         "gain are the small ones (`Artificial -> Cropland`, 46 plots).")

# -- Q5: deep supervision ---------------------------------------------------
_siam_q_idea(
    "siam_cos_deepsup",                                                   # Q5
    deep_sup_weight=SIAM_AUX,
    desc="Q5: deep supervision. An auxiliary coarse3 head on every hidden stage of the "
         "shared encoder, supervised through the same three nested levels and discarded at "
         "predict time, so serving cost is unchanged. Note this model ALREADY has deep "
         "supervision in one sense -- the loss supervises gate, merged2 and coarse3 -- but "
         "that is three reads of one output, all at full depth. This is the paper's sense: "
         "the same objective at shallower DEPTHS. Weight 0.3, the section's preregistered "
         "regulariser strength, with 1.0 registered separately rather than reached for "
         "afterwards.")

_siam_q_idea(
    "siam_cos_deepsup_strong",                                            # Q5b
    deep_sup_weight=1.0,
    desc="Q5b: deep supervision at weight 1.0 -- the paper's own reading, where the "
         "auxiliary losses are co-objectives rather than a regulariser. Preregistered so a "
         "flat Q5 answers the strength question instead of leaving it open (the N2/N2b "
         "pattern).")

# -- Q6: the hybrid loss ----------------------------------------------------
_siam_q_idea(
    "siam_cos_dice",                                                      # Q6
    dice_weight=SIAM_AUX, dice_level="gate",
    desc="Q6: the hybrid loss. Focal is a per-sample objective and cannot see the set "
         "overlap the model is actually scored on; soft-Dice over the change class IS "
         "change-F1 with soft memberships, and under this project's full-batch default it "
         "is computed over the whole training fold rather than a minibatch estimate. The "
         "closest thing already on the board is the tuned change threshold, which moves the "
         "same trade-off post-hoc -- so the counter-check that matters is precision/recall: "
         "N13 established this product does not want more change called, it wants the "
         "calls to be right.")

_siam_q_idea(
    "siam_cos_dice_fine",                                                 # Q6b
    dice_weight=SIAM_AUX, dice_level="fine",
    desc="Q6b: the same hybrid loss taken as an UNWEIGHTED mean of the per-class Dice over "
         "all nine coarse3 classes, which makes it the differentiable relaxation of "
         "`focus_macro_f1` -- the metric this section is scored on -- rather than of "
         "change-F1. The 46-plot transition then contributes as much overlap as the "
         "4,200-plot stable one, and it does so on the SET rather than on the per-sample "
         "gradient, which is the one long-tail lever the section has not pulled: focal "
         "(everywhere), class-balanced sampling (G-H, negative), cRT (O2, negative) and "
         "tau-normalisation (O2c, negative) are all per-sample or per-parameter.")

# -- Q7: the one non-flat result, on the section's best base ----------------
_siam_s2off_idea(
    "siam_s2off_crfe_attn", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,   # Q7
    siam_crfe="attn",
    desc="Q7: CRFE channel attention over N8b, the best model in the ledger. Q3/Q3b are "
         "flat-to-negative on every aggregate and move exactly one thing -- stable "
         "built-up, which is the frontier four ideas in section N (N9 built fraction, N10 "
         "per-year adapters, N12 the merged2 gate, the N8 tower swap) all failed to close "
         "and which SIAMESE_RESEARCH.md records as still the deployed model's. Composed "
         "here because the detail tower is the other thing that partly recovers it, so "
         "whether the two are the same lever is the question -- and the F7 / N3b / O4 "
         "signature says two mechanisms correcting the same thing land between their parts.")

_siam_s2off_idea(
    "siam_s2off_crfe_full", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,   # Q7b
    siam_crfe="full",
    desc="Q7b: the full CRFE module over N8b. Q3b is where the built-up move is largest, "
         "and it is the arm that pays for it in change-F1, so this is the row that decides "
         "whether the trade is worth taking at all.")

# -- Q8: the control that says whether CRFE is the mechanism -----------------
_siam_s2off_idea(
    "siam_s2off_crfe_rand", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,   # Q8
    siam_crfe="rand",
    desc="Q8: the CONTROL for Q2/Q7b, and the row that decides what the built-up move is "
         "made of. `z18 + z24` is a fixed linear map of a block that already carries z18 "
         "and z24, so it cannot enlarge what the mixer can compute -- it can only widen the "
         "first layer and recondition the optimisation. This replaces it with a FIXED RANDOM "
         "linear view of the same pair at the same width: the same kind of object, none of "
         "CRFE's meaning. If it reproduces the move, the finding is about width and "
         "conditioning and CRFE is not the mechanism; if it is flat, the sum operator is "
         "carrying it. Same design as the endogenous controls that inverted the readings in "
         "N14b and P3.")

_siam_s2off_idea(
    "siam_s2off_crfe_sum", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,    # Q8c
    siam_crfe="sum",
    desc="Q8c: the sum arm on the s2off base, which completes the 2x2 the control needs -- "
         "{nothing, sum, random mix} x {gate, no gate}, all on one base, one set of folds "
         "and one set of seeds. Without this cell the sum arm and its control sit on "
         "different bases and the comparison is not matched.")

_siam_s2off_idea(
    "siam_s2off_crfe_randattn", siam_cos_weight=SIAM_AUX,                    # Q8b
    siam_cos_margin=0.3, siam_crfe="randattn",
    desc="Q8b: the control for the full module -- random mix plus the channel-attention "
         "gate, i.e. Q7b with only the sum operator swapped out. Run as a pair with Q8 so "
         "the control is available against both arms.")

# -- Q10: SNIIF-Net, on the state-pretrained base ---------------------------
# Sci Rep 2025 (s41598-025-15468-w), "Siamese change detection based on
# information interaction and fusion network". A second bi-temporal siamese
# paper, and the three modules it introduces map onto this model as follows:
#
#   FPFM (dual-branch fusion, |f1 - f2| and f1 + f2)  -> ALREADY TESTED. It is
#       section Q's CRFE (Q2/Q7b/Q8), 15 seeds, where the sum operator's
#       marginal effect was inside the seed spread once the gate was present.
#       Not re-run.
#   FIIM (cross-branch attention BEFORE the difference) -> Q10f. Section Q's
#       gate sits downstream of the subtraction; this is the same operator moved
#       upstream of it, which is the one placement never tried.
#   MSSM (contrastive supervision at every decoder scale) -> Q10a-e. The
#       headline idea, and the only one with no precedent here at all.
#
# The base is P7e (`siam_s2off_state_pre`), not N8b: it is the section's current
# best on the commissioned transitions, it costs nothing extra at serving, and
# the user asked for these on that base. Every arm is therefore
# state-pretrained, gate-off and reads no Sentinel-2 at inference.

def _q10(name, desc, **overrides):
    return state_pretrain_idea(name, base="s2off", desc=desc, **overrides)


_q10("siam_s2off_state_pre_mssm",                                         # Q10a
     siam_mssm_weight=SIAM_AUX,
     desc="Q10a: MSSM. The pair objective the model already carries at the final embedding "
          "-- pull stable endpoints together, push change apart past a margin -- repeated "
          "at BOTH hidden stages of the shared encoder, weight 0.3, no new parameters. "
          "Distinct from Q5, which was flat: Q5 hung a classification head off each stage "
          "and the reading was that a three-level nested loss already supervises this "
          "encoder at depth. That reading does not cover the pair GEOMETRY, which nothing "
          "constrains anywhere but at z -- the stages are currently free to interleave the "
          "two dates however they like provided the last layer can separate them. "
          "Preregistered prediction: if Q5's explanation is the whole story this is flat "
          "too, and the pair of results then says the encoder is insensitive to auxiliary "
          "depth rather than to auxiliary heads.")

_q10("siam_s2off_state_pre_mssm_all",                                     # Q10b
     siam_mssm_weight=SIAM_AUX, siam_mssm_scales="all",
     desc="Q10b: Q10a extended to the final embedding as well, which is the paper's literal "
          "reading -- it constrains all four decoder scales including the one the "
          "prediction is made from. That doubles the pair term's weight on z (0.3 from the "
          "cosine loss plus 0.3/3 from here), which is exactly why Q10c exists.")

_q10("siam_s2off_state_pre_mssm_ctrl",                                    # Q10c
     siam_cos_weight=2 * SIAM_AUX,
     desc="Q10c: the CONTROL, and the row that decides what Q10a/Q10b are made of. Same "
          "total auxiliary pair weight (0.6), all of it at the final embedding, no stage "
          "term. If the effect is reproduced here then 'multi-scale' is not the mechanism "
          "and the finding is simply that this model wants more of the N2 cosine objective "
          "-- a one-line change with no module attached. Same design as the endogenous and "
          "shuffled controls that inverted the readings in N14b, P3 and Q8.")

_q10("siam_s2off_state_pre_mssm_euclid",                                  # Q10d
     siam_mssm_weight=SIAM_AUX, siam_mssm_scales="all",
     siam_mssm_metric="euclid", siam_mssm_stable_margin=0.5,
     desc="Q10d: the paper's loss FORM rather than this project's -- squared hinges on "
          "Euclidean distance both ways, with slack on the unchanged side, at all scales. "
          "Distance is taken on L2-normalised features (D = sqrt(2(1-cos)), in [0,2]), "
          "because one pair of margins cannot otherwise be legal at a 512-wide BatchNorm "
          "stage and a 128-wide linear one at once. That normalisation makes the metric "
          "choice a reparameterisation and leaves the squared hinge and the double margin "
          "as the real content -- so a difference from Q10b is about loss shape, not about "
          "Euclidean vs cosine, and should be reported that way.")

_q10("siam_s2off_state_pre_dm",                                           # Q10e
     siam_cos_stable_margin=0.1,
     desc="Q10e: the double margin alone, at the final embedding, with no multi-scale term "
          "-- one number changed. The published loss hinges BOTH sides; this project's "
          "hinges only the change side and drives every stable pair towards cos exactly 1. "
          "Stable plots are 4:1 here, so that term is most of the auxiliary gradient and it "
          "is still pulling on pairs that already agree. Slack of 0.1 (stop at cos >= 0.9) "
          "releases that capacity. The counter-check is change RECALL: this is the arm most "
          "likely to move the stable/change geometry, and N13 established the product wants "
          "the calls to be right rather than more change called.")

_q10("siam_s2off_state_pre_fiim",                                         # Q10f
     siam_fiim="cross",
     desc="Q10f: FIIM, the cross-branch interaction. Each date's embedding re-weighted by a "
          "gate that reads BOTH dates, one shared weight matrix used with the inputs "
          "swapped, zero-initialised so training starts at exactly the plain encoder. The "
          "question is PLACEMENT and not the gate: Q7's SE gate was the section's one "
          "non-flat result and it sits on the assembled block, downstream of the "
          "subtraction, where it cannot change z24-z18, the cosine feature, or what the "
          "pair losses read. This is the same operator moved upstream of all three, which "
          "is the paper's own ordering. Watch stable built-up -- if Q10f and Q7 are one "
          "lever, that is where they overlap, and the F7/N3b/O4 signature says two "
          "mechanisms on one block land between their parts.")

_q10("siam_s2off_state_pre_fiim_self",                                    # Q10g
     siam_fiim="self",
     desc="Q10g: the CONTROL for Q10f, and the row that says whether INTERACTION is the "
          "mechanism. The same gate, the same parameter count, the same nonlinearity, the "
          "same init draw -- but each date's gate reads that date twice instead of the "
          "pair, so the only thing removed is the cross-branch information the module "
          "exists to add. Q8's fixed-random control did exactly this job for CRFE's sum "
          "operator and answered it (the sum is not width). If this reproduces Q10f, the "
          "finding is that a per-date multiplicative gate before the subtraction helps and "
          "FIIM is not the mechanism.")

# -- Q11: the composition Q10 left open -------------------------------------
# Q10's finding was PLACEMENT: a multiplicative gate upstream of the endpoint
# subtraction is a different object from one downstream of it, because only the
# upstream one can change z24-z18, the cosine feature and what the pair losses
# read. Section Q only ever tried the downstream one (Q7's SE gate over the
# assembled block), and it was that section's single non-flat result. The two
# have never been in the same model.
#
# The design is a 3x2 on ONE base and one set of folds -- {nothing, self-gate,
# cross-gate} upstream x {nothing, SE gate} downstream -- of which Q10 already
# holds the left-hand column at 15 seeds. Preregistered prediction, the
# F7 / N3b / O4 signature: two mechanisms correcting the same failure land
# BETWEEN their parts, not at their sum. That is the null here. A composition at
# or above the sum says the two gates are genuinely separate levers and the
# placement finding is worth more than the module that produced it.
#
# The crfe='full' rows are included because Q7b -- not Q7 -- is the arm that
# moved BOTH built-up numbers the right way on N8b, and N11 requires as_veg and
# as_chg to be read together. They also carry section Q's own caveat: once the
# gate is present the sum operator's marginal effect was inside the seed spread.

_q10("siam_s2off_state_pre_crfe_attn",                                    # Q11a
     siam_crfe="attn",
     desc="Q11a: Q7's SE gate moved onto the state-pretrained base. The missing cell -- "
          "section Q ran CRFE on N8b and section P7e changed the base under it, so 'the "
          "downstream gate' has never been measured where the upstream one was. Without "
          "this row the composition below is compared against a part measured on a "
          "different model, which is exactly the mismatch N18's false verdict came from.")

_q10("siam_s2off_state_pre_crfe_full",                                    # Q11b
     siam_crfe="full",
     desc="Q11b: the full CRFE module (sum + SE gate) on this base. Q7b is where section "
          "Q's built-up move was largest and is the arm that moved as_veg and as_chg "
          "together, so it is the downstream part worth composing against as well as the "
          "gate alone.")

_q10("siam_s2off_state_pre_fiim_attn",                                    # Q11c
     siam_fiim="cross", siam_crfe="attn",
     desc="Q11c: THE COMPOSITION. Both gates, one upstream of the endpoint subtraction and "
          "one downstream of it, in one model. Q10's whole finding is that these are "
          "different objects; this is the row where that claim pays or does not. Against "
          "Q10f (0.4427 focus, artStab 0.669) and Q11a, with O4's between-the-parts result "
          "as the preregistered null.")

_q10("siam_s2off_state_pre_fiim_self_attn",                               # Q11d
     siam_fiim="self", siam_crfe="attn",
     desc="Q11d: the composition's CONTROL, carried forward because Q10g inverted Q10f on "
          "the built-up numbers -- the self-gate was BETTER there. If the composition is "
          "worth having, the row that says whether it needs the cross-branch half is this "
          "one, and Q10 established it cannot be assumed.")

_q10("siam_s2off_state_pre_fiim_crfe_full",                               # Q11e
     siam_fiim="cross", siam_crfe="full",
     desc="Q11e: the upstream gate against the full downstream module rather than its gate "
          "alone. Q7b's summation branch is FPFM's, so this is the one row in which all "
          "three of SNIIF-Net's modules -- interaction, dual-branch fusion, and the channel "
          "attention they share with Zhang et al. -- are present at once, which is the "
          "closest this tabular encoder gets to the published network.")

coarse3_gate_idea(                                                        # Q11f
    "c3gate_siam_s2off_state_pre_fiim_attn", "siam_s2off_state_pre_fiim_attn",
    "Q11f: O3's gate over the composition. Q10h is the lesson being applied rather than "
    "repeated: FIIM's +0.026 arg-max focus became +0.0035 gate-to-gate, so no arg-max "
    "number in Q11 means anything for a deployment until it is re-read here.")

coarse3_gate_idea(                                                        # Q11g
    "c3gate_siam_s2off_state_pre_crfe_attn", "siam_s2off_state_pre_crfe_attn",
    "Q11g: the same re-read for the downstream gate alone, so the gated comparison is a "
    "complete row and not the composition against an ungated part.")

coarse3_gate_idea(                                                        # Q10h
    "c3gate_siam_s2off_state_pre_fiim", "siam_s2off_state_pre_fiim",
    "Q10h: O3's free coarse3 gate over Q10f. The incumbent on the commissioned transitions "
    "is not an arg-max, it is P7f at focus 0.4383 -- so an arg-max gain of +0.0195 has to "
    "be re-read gate-to-gate before it means anything for a deployment. The two break "
    "`Artificial -> Cropland` by different routes (a representation vs a decision rule) and "
    "O4's signature is that two mechanisms on one failure land between their parts.")


# -- S: set-restricted conformal loss ---------------------------------------
_siam_s2off_idea(
    "siam_setce_fine", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,          # S1
    set_ce_weight=SIAM_AUX, set_ce_level="fine",
    desc="S1: N8b plus the set-restricted loss at the coarse3 head. The auxiliary term "
         "builds Mondrian LAC sets on one half of the training fold, scores the other "
         "half, forces the truth in, and asks the model to be right conditional on the "
         "answer being inside that set. This is the training-side version of section R's "
         "headroom, not the already-negative decode-time re-ranking.")

_siam_s2off_idea(
    "siam_setce_fine_strong", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,   # S1b
    set_ce_weight=1.0, set_ce_level="fine",
    desc="S1b: S1 at weight 1.0, preregistered as the strength question rather than a "
         "post-hoc rescue if the conservative 0.3 regulariser is flat.")

_siam_s2off_idea(
    "siam_setce_both", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,          # S1c
    set_ce_weight=SIAM_AUX, set_ce_level="both",
    desc="S1c: the set-restricted loss at both coarse3 and merged2. The gate level is "
         "intentionally skipped: with two classes the restricted CE is either plain CE or "
         "a constant, so it does not test the proposed mechanism.")

_siam_s2off_idea(
    "siam_setce_rand", siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3,          # S2
    set_ce_weight=SIAM_AUX, set_ce_level="fine", set_ce_random=True,
    desc="S2: the required size-matched random-set control for S1. It keeps the conformal "
         "set sizes, draws random class subsets of those sizes with the true class forced "
         "in, and decides whether any S1 movement is calibration-specific or just extra "
         "contrastive gradient on the tail -- the control pattern that inverted N14b, P3 "
         "and Q8.")


# ---------------------------------------------------------------------------
# R. Conformal prediction  (docs/research/SIAMESE_RESEARCH.md)
#
# Every operating point in this ledger is chosen by SEARCH: the change gate
# (E1) grid-searches a threshold for change-F1, the cost gates (F3, N12, O3)
# grid-search per-class multipliers for macro- or focus-F1. Conformal prediction
# chooses thresholds by CALIBRATION instead -- each class gets the threshold at
# which its own held-out scores reach a stated coverage level -- which has three
# consequences worth testing here and nowhere else in the ledger:
#
#   * it needs no target metric, so it needs no FOCUS SET. P6 ended on exactly
#     that question -- widening the gate's four commissioned transitions to six
#     nearly doubled `Nature -> Cropland` for free, and which classes belong in
#     the set is a product decision. A calibrated threshold per class does not
#     ask.
#   * the rare-class thresholds come from the rare class's own score
#     distribution, which is the mechanism O3 found by search. Class-conditional
#     (Mondrian) conformal is that correction in closed form.
#   * it produces SETS, not labels -- so it also answers a question no ledger
#     row has answered: where on the map does the model not know? CLAUDE.md's
#     standing verdict is that spatial smoothing removes change pixels first and
#     that the lever is inputs or uncertainty. This is the uncertainty.
#
# All of it is post-hoc over the cached OOF probabilities, so every row below is
# free and none of them touch serving cost. Thresholds are calibrated per outer
# fold on the OTHER folds and applied to the held-out one -- the nested_gate
# discipline, for the same reason.
# ---------------------------------------------------------------------------
#: Coverage levels reported by ``conformal_report.py``. 0.10 is the level the
#: ledger ideas pin, chosen a priori rather than swept-then-quoted.
CONFORMAL_ALPHAS = (0.05, 0.10, 0.20, 0.30)
CONFORMAL_ALPHA = 0.10


def conformal_score_matrix(probs: np.ndarray, kind: str = "lac",
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """``(n, K)`` nonconformity scores -- low means the class fits the row.

    ``lac`` is the least-ambiguous score ``1 - p_k`` (Sadinle et al.): smallest
    possible sets at a given marginal coverage, and the score whose per-class
    threshold is exactly a probability cut, so it composes with everything else
    in this file. ``aps`` is the adaptive score (Romano et al.) -- the cumulative
    mass down to class ``k``, randomised within its own probability so the sets
    are exactly rather than conservatively valid. APS trades set size for better
    conditional coverage, which is the property this problem's 4.2k-vs-46 class
    imbalance should reward if anything does.
    """
    if kind == "lac":
        return 1.0 - probs
    if kind != "aps":
        raise ValueError(f"unknown conformal score: {kind}")
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    u = 1.0 if rng is None else rng.random((len(probs), 1))
    scores = np.empty_like(cum)
    np.put_along_axis(scores, order, cum - u * sorted_p, axis=1)
    return scores


def conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """The split-conformal threshold: the ``ceil((n+1)(1-alpha))``-th smallest.

    Returns ``inf`` when the calibration set is too small to support the level
    (``ceil((n+1)(1-alpha)) > n``), which is the honest answer for a 46-plot
    class at a tight alpha -- the class is then always in the set rather than
    silently thresholded at its own maximum.
    """
    n = len(cal_scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return np.inf if k > n else float(np.sort(cal_scores)[k - 1])


def nested_conformal(view: View, probs: np.ndarray, classes: list,
                     truth: np.ndarray, alpha: float = CONFORMAL_ALPHA, *,
                     kind: str = "lac", mondrian: bool = True,
                     seed: int = 0) -> tuple:
    """Fold-wise conformal thresholds, calibrated without seeing the fold.

    Returns ``(scores, q, sets)``, all ``(n, K)``: the nonconformity scores, the
    threshold each row's class faced, and the set membership ``scores <= q``.

    ``mondrian=True`` calibrates a separate threshold per class on the
    calibration rows of that class, giving per-class coverage; ``False`` pools
    them into one threshold and gives marginal coverage only. The distinction is
    the whole experiment: marginal coverage over a legend that is 66% stable
    Vegetation can be met while a 46-plot transition is never covered at all,
    and that is the failure mode section O and P have been chasing by search.
    """
    rng = np.random.default_rng(seed)
    scores = conformal_score_matrix(probs, kind, rng)
    index = {c: i for i, c in enumerate(classes)}
    # Rows whose truth is outside this level's class list (a rare transition
    # lumped to RARE_LABEL) can score no class, so they calibrate nothing.
    true_idx = np.array([index.get(t, -1) for t in truth])
    ok = true_idx >= 0
    true_scores = np.where(ok, scores[np.arange(len(scores)),
                                      np.clip(true_idx, 0, None)], np.nan)
    q = np.zeros_like(scores)
    for tr, te in view.folds:
        cal = tr[ok[tr]]
        if mondrian:
            for k in range(len(classes)):
                rows = cal[true_idx[cal] == k]
                q[te, k] = conformal_quantile(true_scores[rows], alpha)
        else:
            q[te, :] = conformal_quantile(true_scores[cal], alpha)
    return scores, q, scores <= q


def conformal_labels(classes: list, scores: np.ndarray, q: np.ndarray,
                     mode: str = "margin") -> np.ndarray:
    """A point label from calibrated thresholds -- the deepest set member.

    ``margin`` picks ``argmin_k (s_k - q_k)``: the class furthest inside its own
    conformal threshold, and the class closest to entering when the set is
    empty, so this is a total labelling directly comparable to an arg-max and to
    the cost gates. Under the ``lac`` score it is an ADDITIVE per-class shift of
    the probabilities, which makes ``ratio`` the interesting alternative: the
    same thresholds read multiplicatively, ``argmax_k p_k / (1 - q_k)``, which is
    literally ``nested_cost_gate`` with costs derived from calibration instead of
    from a grid search against the metric.
    """
    arr = np.array(classes, dtype=object)
    # An infinite threshold means "always in the set" -- correct for membership,
    # degenerate for ranking: several -inf margins tie and the arg-min silently
    # returns the lowest class index. Both scores here live in [0, 1], so 1 is
    # the weakest finite threshold that still always includes the class, and it
    # keeps the ordering well defined. This bites on the 46-plot coarse3 class
    # at alpha <= 0.02, where it is the difference between a read and an
    # artefact.
    q = np.minimum(q, 1.0)
    if mode == "margin":
        return arr[np.argmin(scores - q, axis=1)]
    if mode == "ratio":
        # 1 - q is the probability cut for class k; inf thresholds (a class too
        # rare to calibrate at this alpha) cut at 0 and are floored so the
        # ratio stays finite and the class stays strongly preferred.
        cut = np.clip(1.0 - q, 1e-3, None)
        return arr[((1.0 - scores) / cut).argmax(1)]
    raise ValueError(f"unknown conformal read: {mode}")


#: Coverage levels the nested-alpha read may choose from. Wide, because the
#: point of the row is that alpha is a free parameter and the cost gates get
#: theirs tuned -- a fixed alpha compared against a tuned multiplier is the
#: N12b mistake in the other direction.
#: Extends below the conformal-native range on purpose. As alpha -> 0 every
#: threshold saturates and the margin read collapses to exactly the arg-max, so
#: an interior optimum is evidence the per-class correction pays and a slide to
#: the bottom of the grid is evidence it does not. The first pass picked 0.02,
#: the old lower edge, unanimously across 25 folds -- which is not readable
#: without the two rungs below it.
ALPHA_GRID = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def nested_alpha_conformal(view: View, probs: np.ndarray, classes: list,
                           truth: np.ndarray, *, kind: str = "lac",
                           mode: str = "margin", objective: str = "macro_f1",
                           seed: int = 0) -> np.ndarray:
    """Conformal labels whose coverage level is also chosen off-fold.

    R1 pins alpha at 0.10 a priori, which is the conformal-native choice but not
    a matched comparison: ``nested_cost_gate`` gets its multipliers tuned on the
    inner folds against the metric, so the calibration route should be allowed
    the one free parameter it has. For each outer fold, each candidate alpha is
    scored by leave-one-fold-out INSIDE the calibration folds -- calibrate on
    three, label the fourth, pool -- and the winner is then calibrated on all
    four and applied to the held-out fold. The test fold is never involved in
    either choice.
    """
    labels = np.empty(len(truth), dtype=object)
    index = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([index.get(t, -1) for t in truth])
    ok = true_idx >= 0
    rng = np.random.default_rng(seed)
    scores = conformal_score_matrix(probs, kind, rng)
    true_scores = np.where(ok, scores[np.arange(len(scores)),
                                      np.clip(true_idx, 0, None)], np.nan)

    def label_block(cal_rows, out_rows, alpha):
        q = np.zeros((len(out_rows), len(classes)))
        for k in range(len(classes)):
            rows = cal_rows[true_idx[cal_rows] == k]
            q[:, k] = conformal_quantile(true_scores[rows], alpha)
        return conformal_labels(classes, scores[out_rows], q, mode)

    def score_inner(pred, rows):
        if objective == "focus_macro_f1":
            got = focus_metrics(truth[rows], pred)["focus_macro_f1"]
        elif objective == "change_f1":
            got = change_metrics(truth[rows], pred)["change_f1"]
        else:
            got = macro_f1(truth[rows], pred, classes)
        return -1.0 if not np.isfinite(got) else float(got)

    for tr, te in view.folds:
        tr_ok = tr[ok[tr]]
        inner = [(np.setdiff1d(tr_ok, np.intersect1d(tr_ok, itr_te)),
                  np.intersect1d(tr_ok, itr_te))
                 for _, itr_te in view.folds]
        inner = [(c, o) for c, o in inner if len(o) and len(c)]
        best_alpha, best = ALPHA_GRID[0], -np.inf
        for alpha in ALPHA_GRID:
            preds = np.empty(0, dtype=object)
            rows = np.empty(0, dtype=int)
            for cal_rows, out_rows in inner:
                preds = np.concatenate([preds, label_block(cal_rows, out_rows, alpha)])
                rows = np.concatenate([rows, out_rows])
            got = score_inner(preds, rows)
            if got > best:
                best, best_alpha = got, alpha
        labels[te] = label_block(tr_ok, te, best_alpha)
    return labels


def conformal_change_labels(classes: list, probs: np.ndarray,
                            sets: np.ndarray) -> np.ndarray:
    """The precautionary set read: change if any change class survives the set.

    A set-valued prediction has to be collapsed before it can be scored on this
    ledger's metrics, and how it is collapsed IS the decision. This is the
    collapse the product implies -- N13 and Q9 both end on change suppression as
    the risk, so a plot whose conformal set still admits a transition is called
    a transition, at the arg-max within the set. An empty set (the row conforms
    to nothing) falls back to the plain arg-max rather than to either side.
    """
    arr = np.array(classes, dtype=object)
    chg = np.array([is_change_label(c) for c in classes])
    if not chg.any() or chg.all():
        return arr[probs.argmax(1)]
    masked = np.where(sets, probs, -np.inf)
    any_change = sets[:, chg].any(1)
    any_stable = sets[:, ~chg].any(1)
    pick_change = arr[chg][masked[:, chg].argmax(1)]
    pick_stable = arr[~chg][masked[:, ~chg].argmax(1)]
    out = np.where(any_change, pick_change, np.where(any_stable, pick_stable,
                                                    arr[probs.argmax(1)]))
    return out.astype(object)


def _binom_ucb(k: int, n: int, delta: float = 0.1) -> float:
    """Clopper-Pearson upper confidence bound on a rate of ``k/n``."""
    from scipy.stats import beta
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - delta, k + 1, n - k))


def crc_change_gate(view: View, probs: np.ndarray, classes: list,
                    alpha: float = 0.25, delta: float = 0.1) -> np.ndarray:
    """The change gate that GUARANTEES recall instead of maximising F1.

    Conformal risk control (Angelopoulos et al.): missed change is monotone
    increasing in the threshold, so the largest threshold whose upper confidence
    bound on the calibration miss-rate stays under ``alpha`` is the tightest gate
    that still promises change recall >= 1 - alpha on the held-out fold. Chosen
    per outer fold on the others, like every other gate here.

    This is a different KIND of row from the rest of the ledger. It does not
    claim to win change-F1 -- it converts "what threshold should the map use",
    which no amount of Oslo inspection can settle (G3/G4: zero labelled plots),
    into "state the recall you require and this is what it costs in precision".
    """
    labels = np.empty(len(view.truth_merged), dtype=object)
    truth_chg = np.array([is_change_label(x) for x in view.truth_merged])
    chg_cols = np.array([is_change_label(c) for c in classes])
    p_change = probs[:, chg_cols].sum(1)
    for tr, te in view.folds:
        cal = tr[truth_chg[tr]]
        best = float(THRESHOLD_GRID.min())
        for t in THRESHOLD_GRID:
            missed = int((p_change[cal] < t).sum())
            if _binom_ucb(missed, len(cal), delta) <= alpha:
                best = max(best, float(t))
        labels[te] = labels_from_probs(probs[te], classes, best)
    return labels


def conformal_idea(name, source, desc, *, level="merged2", mode="margin",
                   kind="lac", mondrian=True, alpha=CONFORMAL_ALPHA,
                   read_fn=None):
    """Register a conformal re-read of one cached model's OOF probabilities.

    ``level="merged2"`` re-reads the four-class probabilities and moves every
    aggregate column; ``level="coarse3"`` re-reads the nine-class ones and moves
    only the focus columns, leaving the aggregates identical to the source by
    construction -- the same separation ``coarse3_gate_idea`` relies on, so a
    conformal row and an O3 row are directly comparable.

    ``alpha="nested"`` hands the coverage level to ``nested_alpha_conformal``
    instead of pinning it, which is the comparison matched against a tuned cost
    gate; a float pins it a priori.
    """
    def fn(ctx, view, seed):
        cached = load_oof(source, view.name, seed)
        if cached is None:
            raise RuntimeError(f"{name} needs cached OOF for {source} on "
                               f"{view.name} seed {seed} -- run it first")
        probs, fine = cached
        if level == "merged2":
            classes, truth = view.merged_classes, view.truth_merged
            if alpha == "nested":
                return probs, fine, nested_alpha_conformal(
                    view, probs, classes, truth, kind=kind, mode=mode,
                    objective="macro_f1", seed=seed)
            scores, q, sets = nested_conformal(view, probs, classes, truth,
                                               alpha, kind=kind,
                                               mondrian=mondrian, seed=seed)
            labels = (read_fn(classes, probs, scores, q, sets) if read_fn
                      else conformal_labels(classes, scores, q, mode))
            return probs, fine, labels
        fine_cached = load_oof_fine(source, view.name, seed)
        if fine_cached is None:
            raise RuntimeError(f"{name} needs cached COARSE3 probabilities for "
                               f"{source}: run --ideas {source} first")
        fine_probs, fine_classes = fine_cached
        if alpha == "nested":
            picked = nested_alpha_conformal(
                view, fine_probs, fine_classes, view.truth_fine, kind=kind,
                mode=mode, objective="focus_macro_f1", seed=seed)
        else:
            scores, q, _ = nested_conformal(view, fine_probs, fine_classes,
                                            view.truth_fine, alpha, kind=kind,
                                            mondrian=mondrian, seed=seed)
            picked = conformal_labels(fine_classes, scores, q, mode)
        return probs, picked, None, (fine_probs, fine_classes)

    return register(name, reads=("full",), group="section-r", desc=desc)(fn)


# -- R1: the merged2 Mondrian read, against costgate_siam (N12) ---------------
conformal_idea(                                                           # R1
    "conf_siam_cos", "siam_s2off_cos",
    "R1: class-conditional (Mondrian) conformal thresholds over the section's best "
    "model, read at the deepest set member. The head-to-head with N12's cost gate: "
    "both give every merged2 class its own decision threshold, N12 by grid-searching "
    "macro-F1 on the inner folds, this by asking each class's own held-out scores "
    "where its 90% coverage lies. If they land together the calibration route is worth "
    "having anyway -- it has no target metric to overfit and no focus set to choose; if "
    "conformal wins, the search was fitting the metric's noise.")

conformal_idea(                                                           # R1b
    "conf_siam_cos_ratio", "siam_s2off_cos", mode="ratio",
    desc="R1b: the same calibrated thresholds read MULTIPLICATIVELY -- argmax p_k/(1-q_k) "
         "-- which is exactly `nested_cost_gate` with conformal costs. R1's additive read "
         "and this one differ only in functional form, so the pair separates 'calibration "
         "beats search' from 'the additive shift happens to suit this legend'.")

conformal_idea(                                                           # R1c
    "conf_siam_cos_marginal", "siam_s2off_cos", mondrian=False,
    desc="R1c: the CONTROL. One pooled threshold for all four classes instead of one per "
         "class, i.e. marginal coverage only. A pooled LAC threshold cannot reorder the "
         "arg-max at all, so this row should reproduce the source model exactly -- and it "
         "is here because if it does NOT, the fold-wise machinery is doing something "
         "other than what R1 claims and every other row in the section is suspect.")

conformal_idea(                                                           # R1d
    "conf_deployed", "s2off_centre_m3s3_bf",
    desc="R1d: R1 on the DEPLOYED model. The N12b lesson, applied: a challenger read at a "
         "calibrated operating point must be compared against an incumbent read at one, or "
         "a free decision rule gets misattributed to the architecture.")

conformal_idea(                                                           # R1e
    "conf_siam_cos_nested", "siam_s2off_cos", alpha="nested",
    desc="R1e: R1 with the coverage level itself chosen on the inner folds against macro-F1 "
         "-- the MATCHED comparison against N12, which gets its four multipliers tuned the "
         "same way. A pinned alpha against a tuned multiplier decides the head-to-head on "
         "who was allowed a free parameter rather than on calibration vs search. The "
         "counter-question this row asks of R1: if the tuned alpha lands far from 0.10, the "
         "conformal-native level is not the right operating point for this legend and the "
         "calibration is buying its per-class SHAPE, not its coverage.")

costgate_idea(                                                            # R1f
    "costgate_siam_s2off", source="siam_s2off_cos",
    desc="R1f: the missing cell. N12 ran the cost gate over `siam_cos` and N12b over the "
         "deployed model, so there was no SEARCH row on `siam_s2off_cos` -- the model every "
         "conformal row in section R re-reads. Without it, calibration-vs-search is a "
         "comparison across two base models. Registered before reading R1e, not after.")

conformal_idea(                                                           # R1g
    "conf_crfe", "siam_s2off_crfe_full", alpha="nested",
    desc="R1g: the nested conformal read over Q7b, the other thing in this project that "
         "moves stable built-up (art_stable_recall 0.669 against the siamese's 0.644). Both "
         "act on the same error, so the F7 / N3b / O4 signature predicts they land BETWEEN "
         "their parts rather than adding -- and if they do, the free post-hoc one wins on "
         "cost, because Q7b pays for its move in change-F1 and a decision rule does not.")

conformal_idea(                                                           # R2
    "conf_siam_aps", "siam_s2off_cos", kind="aps",
    desc="R2: the adaptive (APS) score instead of LAC. APS thresholds a cumulative mass "
         "rather than a single probability, so its per-class correction depends on how the "
         "rest of the distribution is shaped -- the one conformal variant that is not a "
         "reparameterised cost gate. Bigger sets at the same coverage; the question is "
         "whether the better conditional coverage shows up as stable-built-up recall.")

conformal_idea(                                                           # R3
    "conf_siam_setchange", "siam_s2off_cos",
    read_fn=lambda classes, probs, scores, q, sets:
        conformal_change_labels(classes, probs, sets),
    desc="R3: the precautionary SET read -- a plot is change if its 90% conformal set still "
         "admits a transition. This is the only row in the ledger where the decision comes "
         "from a set rather than a ranking, and it targets the failure both N13 and Q9 end "
         "on: the siamese line calls ~0.3% of Oslo change against the deployed model's "
         "0.56% and nothing labelled can adjudicate it. Expect recall up and precision "
         "down; the number that matters is the exchange rate.")


# -- R4: conformal risk control on change recall ------------------------------
def _crc_idea(name, source, alpha, desc):
    def fn(ctx, view, seed):
        cached = load_oof(source, view.name, seed)
        if cached is None:
            raise RuntimeError(f"{name} needs cached OOF for {source}")
        probs, fine = cached
        return probs, fine, crc_change_gate(view, probs, view.merged_classes, alpha)
    return register(name, reads=("full",), group="section-r", desc=desc)(fn)


_crc_idea("crc_siam_r75", "siam_s2off_cos", 0.25,
          "R4: conformal risk control at a required change recall of 0.75 -- the largest "
          "threshold whose upper confidence bound on missed change clears 25%, chosen per "
          "fold. The siamese sits at recall 0.670 by arg-max, so this row prices the "
          "difference: what does the eighth of the change class the model currently misses "
          "cost in precision, stated as a guarantee rather than as a tuned threshold.")

_crc_idea("crc_siam_r90", "siam_s2off_cos", 0.10,
          "R4b: the same instrument at a required recall of 0.90. Registered as a pair so "
          "the exchange rate is a curve rather than a point, and because a guarantee this "
          "tight may not be reachable on this grid at all -- in which case the honest "
          "output is the degenerate gate and the reason for it.")

_crc_idea("crc_deployed_r75", "s2off_centre_m3s3_bf", 0.25,
          "R4c: R4 on the deployed model, whose arg-max recall is already 0.726. The "
          "matched-incumbent row: if a guaranteed 0.75 costs the siamese more precision "
          "than it costs the deployed model, that is an argument about which model to ship "
          "that no aggregate in this ledger makes.")


# -- R5: the coarse3 level, against the O3 cost gate --------------------------
conformal_idea(                                                           # R5
    "conf_c3_siam_cos", "base_siam_s2off_cos_fine", level="coarse3",
    desc="R5: Mondrian conformal at the COARSE3 level -- the direct competitor to O3b's "
         "cost gate, on the same cached probabilities and the same folds. Two differences "
         "in kind, and they are the reason this section exists. O3 tunes multipliers for "
         "four COMMISSIONED transitions and P6 showed the choice of four is doing real "
         "work (widening to six nearly doubled `Nature -> Cropland` for free); conformal "
         "gives all nine a calibrated threshold and never names a focus set. And O3's "
         "search sees `focus_macro_f1` on the inner folds, so it can fit its noise; "
         "calibration never looks at the metric.")

conformal_idea(                                                           # R5b
    "conf_c3_siam_cos_ratio", "base_siam_s2off_cos_fine", level="coarse3",
    mode="ratio",
    desc="R5b: the multiplicative read of R5 -- conformal costs at the coarse3 level, which "
         "is O3's instrument with its grid search replaced by calibration and nothing else "
         "changed.")

conformal_idea(                                                           # R5c
    "conf_c3_base_siam", "base_siam_cos_fine", level="coarse3",
    desc="R5c: R5 on `siam_cos`, the base O3 itself was run on (0.4412 focus macro), so the "
         "conformal-vs-search comparison has a cell where neither side changes model.")

conformal_idea(                                                           # R5d
    "conf_c3_deployed", "base_deployed_fine", level="coarse3",
    desc="R5d: R5 on the deployed model, matching O3c. Completes the 2x2 of "
         "{search, calibration} x {siamese, deployed} at the coarse3 level.")

conformal_idea(                                                           # R5f
    "conf_c3_siam_nested", "base_siam_s2off_cos_fine", level="coarse3",
    alpha="nested",
    desc="R5f: the coarse3 conformal read with alpha tuned on the inner folds against "
         "`focus_macro_f1` -- the matched comparison against O3b, which tunes its "
         "multipliers on that same metric. Note what this row concedes: tuning on the focus "
         "macro puts the focus set back in, so R5 and this one bracket the question. R5 is "
         "the version that never names the four; this is the version that gets the same "
         "freedom O3 has.")

# -- R6: conformal chooses the focus set, search tunes it --------------------
def conformal_focus_gate(view: View, fine_probs: np.ndarray, fine_classes: list,
                         alpha: float = CONFORMAL_ALPHA, passes: int = 2,
                         n_focus: int = 4) -> np.ndarray:
    """The O3 cost gate with its target classes CHOSEN BY CALIBRATION.

    R5 and O3 break different dead classes -- calibration finds
    ``Nature -> Cropland`` (+0.100 F1) and search finds ``Artificial -> Cropland``
    (+0.212) -- so they are not competing instruments, and this is the
    composition that follows: for each outer fold, the classes whose calibrated
    coverage falls furthest short of ``1 - alpha`` are the ones given a free
    multiplier, and coordinate ascent then tunes them exactly as O3 does.

    It is aimed at what P6 left open. ``FOCUS_TRANSITIONS`` is a commissioned
    list, widening it to six nearly doubled ``Nature -> Cropland`` for free, and
    P6 declined to register that because which classes belong in the set is a
    product decision rather than a modelling one. This does not decide it either:
    it replaces the list with a measurement of which classes the model is
    failing to cover, taken on the calibration folds only. The scoring metric is
    still ``focus_macro_f1`` over the commissioned four, so the row cannot win by
    quietly changing what it is judged on.
    """
    arr = np.array(fine_classes, dtype=object)
    labels = np.empty(len(view.truth_fine), dtype=object)
    index = {c: i for i, c in enumerate(fine_classes)}
    true_idx = np.array([index.get(t, -1) for t in view.truth_fine])
    ok = true_idx >= 0
    scores = conformal_score_matrix(fine_probs, "lac")
    true_scores = np.where(ok, scores[np.arange(len(scores)),
                                      np.clip(true_idx, 0, None)], np.nan)

    def score(idx, costs):
        picked = arr[(fine_probs[idx] * costs).argmax(1)]
        got = focus_metrics(view.truth_fine[idx], picked)["focus_macro_f1"]
        return -1.0 if not np.isfinite(got) else got

    for tr, te in view.folds:
        cal = tr[ok[tr]]
        # Coverage each class actually achieves, measured OUT OF SAMPLE. The
        # first version of this calibrated the cut and checked coverage on the
        # same rows, where the conformal quantile puts it above 1 - alpha by
        # construction: every shortfall came out negative and the nomination
        # ranked the three majority classes. So the split is not optional here.
        # It is leave-one-fold-out inside the calibration folds -- the same
        # inner structure `nested_alpha_conformal` uses, and the held-out fold
        # is still never involved.
        shortfall = {k: [] for k in range(len(fine_classes))}
        for _, inner_te in view.folds:
            inner_out = np.intersect1d(cal, inner_te)
            inner_cal = np.setdiff1d(cal, inner_out)
            if not len(inner_out) or not len(inner_cal):
                continue
            for k in range(len(fine_classes)):
                fit_rows = inner_cal[true_idx[inner_cal] == k]
                out_rows = inner_out[true_idx[inner_out] == k]
                if not len(fit_rows) or not len(out_rows):
                    continue
                q_k = min(conformal_quantile(true_scores[fit_rows], alpha), 1.0)
                shortfall[k].append(float((scores[out_rows, k] <= q_k).mean()))
        ranked = {k: (1.0 - alpha) - float(np.mean(v))
                  for k, v in shortfall.items() if v}
        targets = [k for k, _ in sorted(ranked.items(),
                                        key=lambda kv: -kv[1])[:n_focus]]
        costs = np.ones(len(fine_classes))
        best = score(tr, costs)
        for _ in range(passes):
            for j in targets:
                for m in COST_GRID:
                    trial = costs.copy()
                    trial[j] = m
                    got = score(tr, trial)
                    if got > best:
                        best, costs = got, trial
        labels[te] = arr[(fine_probs[te] * costs).argmax(1)]
    return labels


def conformal_focus_idea(name, source, desc, **kw):
    def fn(ctx, view, seed):
        cached = load_oof(source, view.name, seed)
        fine_cached = load_oof_fine(source, view.name, seed)
        if cached is None or fine_cached is None:
            raise RuntimeError(f"{name} needs cached coarse3 probs for {source}")
        fine_probs, fine_classes = fine_cached
        picked = conformal_focus_gate(view, fine_probs, fine_classes, **kw)
        return cached[0], picked, None, (fine_probs, fine_classes)
    return register(name, reads=("full",), group="section-r", desc=desc)(fn)


conformal_focus_idea(                                                     # R6
    "conf_focus_gate_siam", "base_siam_s2off_cos_fine",
    "R6: the composition. Calibration nominates the four classes it cannot cover, the O3 "
    "search tunes their multipliers, and the row is still scored on the commissioned four. "
    "Preregistered prediction, against the F7 / N3b / O4 precedent that two mechanisms on "
    "one error land between their parts: these two do NOT act on one error -- R5 moves "
    "`Nature -> Cropland` and O3 moves `Artificial -> Cropland`, and neither touches the "
    "other's class -- so this should ADD. If it lands between, the mechanism is the "
    "reweighting and not the nomination, and O3 keeps the field.")

conformal_focus_idea(                                                     # R6b
    "conf_focus_gate_deployed", "base_deployed_fine",
    "R6b: R6 on the deployed model, matching O3c, so the composition is not judged on one "
    "base.")

conformal_idea(                                                           # R5e
    "conf_c3_siam_aps", "base_siam_s2off_cos_fine", level="coarse3", kind="aps",
    desc="R5e: APS at the coarse3 level. Nine classes is where an adaptive score has room "
         "to differ from LAC -- with four it mostly does not -- and the two dead classes "
         "(`Artificial -> Cropland`, `Cropland -> Nature`) are exactly the rows whose "
         "conditional coverage LAC is known to sacrifice.")


# ---------------------------------------------------------------------------
# T. Learning with noisy labels  (docs/research/NOISY_LABEL_RESEARCH.md)
#
# Every section before this one has taken the labels as the target and asked
# what to fit them with. This one takes the ledger's own repeated conclusion
# seriously -- that the ceiling is interpreter disagreement on the
# Cropland/Nature boundary (`analyse_label_noise.py`, `cropland-nature-label-
# noise`, N0, P6, R7) -- and asks whether a training procedure that expects
# some labels to be wrong beats one that does not.
#
# Three families, and the axis that separates them is what each needs to be
# TOLD about the noise:
#
#   * nothing at all -- the stochastic co-teaching selector, and the bounded
#     losses (GCE, SCE, bootstrapping) and ELR;
#   * a noise RATE -- classic co-teaching's forget rate;
#   * a noise MATRIX -- forward loss correction, which is already on the
#     tested-negative list at the foot of TWOTOWER_RESEARCH.md, together with
#     confident-learning cleaning and mixup. None of the three is re-run here.
#
# That the first family is the live one is not a preference: this project has
# **no clean validation set** to fit a rate or a matrix against, and the one
# direct measurement of interpreter disagreement it does have (54 reverified
# plots) is change-enriched by construction and cannot be read as a population
# noise rate. A method that needs no estimate is the only kind that can be
# deployed here honestly.
#
# Base: `siam_s2off_cos` (N8b), section R and S's base and the ledger's best
# aggregate model, so every row below is comparable to the standing table
# without a new baseline. **Only network A is served** under co-teaching, so no
# arm can bank a two-model ensemble as if it were noise robustness.
# ---------------------------------------------------------------------------
def _lnl_idea(name, desc, **overrides):
    """A section-T arm: `siam_s2off_cos` plus one noisy-label mechanism."""
    def fn(ctx, view, seed):
        cols, kwargs = siam_s2off_kwargs(
            ctx, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3, **overrides)
        return s2off_cv(view, cols, kwargs, seed)
    return register(name, reads=("full",), group="section-t", desc=desc)(fn)


# -- T1: the selector that needs no noise estimate --------------------------
_lnl_idea(
    "siam_sct",                                                            # T1
    "T1: stochastic co-teaching (Bertels et al. 2023). Two networks, independently "
    "initialised; each keeps the rows whose GIVEN-label posterior clears a threshold drawn "
    "from Beta(32, 2), and trains its partner on them. The threshold is a draw rather than "
    "a rank, so there is no forget rate and nothing has to be estimated about the noise "
    "level -- the property that makes it runnable on a project with no clean validation "
    "set. Read at the coarse3 level, where this project's measured disagreement lives.",
    coteach="stochastic", coteach_level="fine")

_lnl_idea(
    "siam_sct_gate",                                                       # T1b
    "T1b: the same selector reading the GATE posterior (change / no-change) instead of the "
    "coarse3 one. Preregistered as the confidence-regime check, not as a rescue: Beta(32, "
    "2) has mean 0.94 and the paper's settings are ones where a trained net reaches that "
    "on its true class, which a 9-way head on 6.4k plots does not. If T1's guard rate is "
    "high and T1b's is low, the difference is the prior's fit to the head, not the data.",
    coteach="stochastic", coteach_level="gate")

_lnl_idea(
    "siam_sct_merged",                                                     # T1c
    "T1c: the same selector on the merged2 posterior -- the deploy level, and the level "
    "the project's legend has already absorbed the Cropland/Nature noise into. Completes "
    "the level sweep so the choice of reading level is measured rather than assumed.",
    coteach="stochastic", coteach_level="merged")

# -- T2: what a noise-rate estimate buys ------------------------------------
_lnl_idea(
    "siam_coteach10",                                                      # T2
    "T2: classic co-teaching (Han et al. 2018) at forget rate 0.10 -- keep the 90% of rows "
    "with smallest loss under the partner. The contrast T1 exists for: this needs the "
    "noise level as a hyperparameter, and Bertels et al.'s stated motivation is that a "
    "misspecified forget rate costs real accuracy. 0.10 is registered as the a-priori "
    "value, not a swept one.",
    coteach="classic", coteach_forget=0.10)

_lnl_idea(
    "siam_coteach20",                                                      # T2b
    "T2b: classic co-teaching at forget rate 0.20, registered up front as the sensitivity "
    "question rather than reached for if 0.10 is flat. The pair T2/T2b is the measurement "
    "of how much the forget rate matters here, which is exactly what T1 claims to avoid "
    "having to know.",
    coteach="classic", coteach_forget=0.20)

# -- T3: the controls, registered before any of the above is read -----------
_lnl_idea(
    "siam_cotrand10",                                                      # T3
    "T3: THE CONTROL for T2. Same two networks, same schedule, same number of rows "
    "dropped per step -- chosen at RANDOM instead of by loss. Training on a random 90% "
    "subsample is itself a regulariser, and without this row a T2 gain is not attributable "
    "to noise filtering. The pattern that inverted N14b, P3 and Q8.",
    coteach="random", coteach_forget=0.10)

# -- T4: bounded losses, no estimate, no second network ---------------------
_lnl_idea(
    "siam_ce",                                                             # T4ref
    "T4ref: the base with plain cross-entropy instead of focal. Not a noisy-label method "
    "-- it is the REFERENCE the T4 rows have to be read against, because a bounded loss "
    "replaces the CE core and drops the focal modulation with it. Comparing GCE against a "
    "focal baseline would confound the two changes, and focal is itself the ledger's least "
    "noise-robust choice: (1-p)^gamma upweights exactly the rows a mislabel produces.",
    loss="ce")

_lnl_idea(
    "siam_gce",                                                            # T4
    "T4: generalised cross-entropy, q=0.7 (Zhang & Sabuncu 2018) at all three levels. The "
    "bounded-loss answer to the same problem: a mislabelled row's gradient is capped "
    "rather than unbounded, so it cannot dominate. No second network and no estimate; "
    "compare against `siam_ce`, not against the focal base.",
    loss="ce", robust_loss="gce", robust_q=0.7)

_lnl_idea(
    "siam_gce_fine",                                                       # T4b
    "T4b: GCE at the COARSE3 level only, leaving merged2 and the gate on cross-entropy. "
    "The hierarchy is what makes this expressible: this project's measured noise is a "
    "coarse3 phenomenon that the merged2 legend already absorbs, so bounding the loss "
    "where the noise is and nowhere else is the targeted version of T4.",
    loss="ce", robust_loss="gce", robust_q=0.7, robust_levels="fine")

_lnl_idea(
    "siam_sce",                                                            # T4c
    "T4c: symmetric cross-entropy (Wang et al. 2019), alpha=0.1, beta=1.0. The other "
    "standard bounded objective; it differs from GCE in keeping a CE term for "
    "convergence and adding the reverse-KL term for the robustness, so the two are not "
    "the same hypothesis wearing different constants.",
    loss="ce", robust_loss="sce", robust_alpha=0.1, robust_beta=1.0)

_lnl_idea(
    "siam_boot_soft",                                                      # T4d
    "T4d: soft bootstrapping (Reed et al. 2015), beta=0.95 -- mix 5% of the model's own "
    "posterior into the target. The weakest and oldest of the three, registered because it "
    "is the one that acts on the TARGET rather than on the loss shape, which is a "
    "different mechanism even where the effect size is similar.",
    loss="ce", robust_loss="boot_soft", robust_beta=0.95)

# -- T5: early-learning regularisation --------------------------------------
_lnl_idea(
    "siam_elr",                                                            # T5
    "T5: early-learning regularisation (Liu et al. 2020), lambda=3, beta=0.7. Holds the "
    "coarse3 head to an EMA of its own earlier posterior, on the premise that the clean "
    "majority is fitted before the mislabelled rows are memorised. It needs no noise "
    "estimate and no second network, and it is the closest thing in the literature to a "
    "principled version of `early stopping`, which is on this project's tested-negative "
    "list as a blunt instrument.",
    elr_weight=3.0, elr_beta=0.7)

_lnl_idea(
    "siam_coteach10_strat",                                                # T6
    "T6: classic co-teaching at forget rate 0.10, applied WITHIN each coarse3 class. "
    "The diagnostic that motivates it is measured, not assumed: unstratified, the 10% "
    "budget is spent almost entirely on the rare transitions -- 24% of `Artificial -> "
    "Cropland` steps kept against 99% of `Nature -> Nature` (coteach_diagnostics.py) -- so "
    "T2 tested a rarity filter, not a noise filter. Per-class selection is the same "
    "correction Mondrian conformal makes to a pooled cut in R7, and it is what decides "
    "whether small-loss selection has anything to offer this target once imbalance is "
    "taken away from it.",
    coteach="classic", coteach_forget=0.10, coteach_stratify=True)

_lnl_idea(
    "siam_cotrand10_strat",                                                # T6b
    "T6b: the matched random control for T6 -- same per-class budget, rows drawn at "
    "random inside each class. T3 established that the unstratified random drop is free; "
    "this establishes the same for the stratified one, so T6 is read against a control at "
    "its own selection rate rather than against T3's.",
    coteach="random", coteach_forget=0.10, coteach_stratify=True)

_lnl_idea(
    "siam_coteach30_strat",                                                # T6c
    "T6c: T6 at forget rate 0.30. Registered as the strength question up front. If "
    "per-class selection is genuinely finding mislabelled rows then a larger budget "
    "should help before it hurts, and if 0.10 and 0.30 are both flat the mechanism has no "
    "purchase here at any rate -- which is the stronger negative and the one worth "
    "recording.",
    coteach="classic", coteach_forget=0.30, coteach_stratify=True)

_lnl_idea(
    "siam_elr_aux",                                                        # T5b
    "T5b: ELR at the project's preregistered auxiliary weight 0.3 rather than the paper's "
    "3.0. The N2/N2b pattern -- the strength question registered up front, so a flat "
    "primary is not rescued post hoc by a weight sweep.",
    elr_weight=SIAM_AUX, elr_beta=0.7)


# ---------------------------------------------------------------------------
# V. Frequency-split specialists  (docs/research/SPECIALIST_SPLIT_RESEARCH.md)
#
# Two models instead of one: a HEAD model over the classes that already reach an
# acceptable accuracy, and a TAIL model over the ones that do not. Measured on
# `base_siam_s2off_cos_fine`, the split is not a judgement call --
#
#     Nature -> Nature      2532  F1 0.766     | Nature -> Cropland   243  0.272
#     Cropland -> Cropland  1661     0.729     | Artificial -> Nature 123  0.440
#     Artificial -> Art.     979     0.700     | Cropland -> Nature   114  0.000
#     Nature -> Artificial   383     0.509     | Artificial -> Crop.   46  0.000
#     Cropland -> Artificial 333     0.611     |
#
# -- there is a clean gap at 333 plots / F1 0.51, and the cut is taken there and
# not moved afterwards.
#
# **The mechanism is N0's own diagnosis, taken at its word.** N0 states that the
# rare transitions die because "a class that small cannot support a decision
# boundary against 4,200 stable plots under focal loss". Every rare-class idea
# since has kept all nine classes in one softmax and attacked the loss (focal,
# cb_focal, Dice), the sampler (G-H), the parameterisation (proto, tau-norm,
# cRT) or the decision rule (O3's cost gate, R5's conformal cuts). A specialist
# trained on the 526 tail plots alone never draws that boundary at all: within
# the tail, `Artificial -> Cropland` faces 480 rivals rather than 6,368.
#
# **The composition is exact and needs no new decision rule.** For a partition
# of the coarse3 classes into blocks B,
#
#     P(k) = P_base(B) * P_spec(k | B)          k in B
#
# which is the chain rule, sums to one, and -- this is the property that makes
# the section clean -- **collapses to the base model exactly** when the
# specialist reproduces the base's own within-block conditional. So there is no
# composition-rule confound to control for: the base *is* the control, and any
# difference is attributable to the specialist's conditional alone. The block
# masses are untouched, so the merged2 and gate reads are untouched too, and
# every movement lands on `focus_macro_f1` and the coarse3 per-class table.
#
# **The preregistered tension, and it is quantified on both sides.** This trades
# imbalance for sample size. The tail specialist sees 526 plots where the base
# sees 6,414 -- 3.6 doublings down, and S19's learning curve prices a doubling at
# +0.026 change-F1. Against that, the tail's imbalance falls from 55:1 to 5:1.
# The ledger has a measured exchange rate for the first and none for the second,
# so the honest prior is that this loses; what makes it worth running anyway is
# that no idea in ~50 has moved these classes and every one of them left the
# imbalance in place.
#
# **The incumbent to beat is O3, not the arg-max.** `c3gate_siam_s2off_cos`
# reaches focus_macro_f1 0.4318 post-hoc for zero training cost; the arg-max is
# 0.3847. A two-model system that lands between them has lost.
# ---------------------------------------------------------------------------
#: The frequency cut, taken once on the table above and fixed. `tail4` is the
#: primary split (F1 <= 0.44 against >= 0.51); `tail6` is the alternative that
#: puts every transition in the tail and keeps only the three stable classes in
#: the head, registered so the choice of cut is measured rather than assumed.
FREQ_BLOCKS: dict[str, list[str]] = {
    "tail4": ["Nature -> Cropland", "Artificial -> Nature",
              "Cropland -> Nature", "Artificial -> Cropland"],
    "head5": ["Nature -> Nature", "Cropland -> Cropland",
              "Artificial -> Artificial", "Nature -> Artificial",
              "Cropland -> Artificial"],
    "tail6": ["Nature -> Artificial", "Cropland -> Artificial",
              "Nature -> Cropland", "Artificial -> Nature",
              "Cropland -> Nature", "Artificial -> Cropland"],
    "head3": ["Nature -> Nature", "Cropland -> Cropland",
              "Artificial -> Artificial"],
}


def specialist_cv(view: View, cols: list, kwargs: dict, seed: int,
                  base_source: str, blocks: list[str]):
    """Compose a cached base model's block masses with per-block specialists.

    The base supplies ``P(B)`` for every block and its own answer for any class
    outside one; each specialist is fitted **on the training fold's rows of its
    own block only** and supplies ``P(k | B)`` for every test row. Nothing here
    is fitted on the held-out fold, and the base's probabilities come from the
    cache, so the two halves are on identical folds by construction.

    A specialist that has not seen one of its block's classes in a fold simply
    does not emit it; the row is renormalised over what the specialist did emit
    rather than being silently rescaled, which would leak the missing class's
    mass into its neighbours.
    """
    cached = load_oof_fine(base_source, view.name, seed)
    if cached is None:
        raise RuntimeError(f"specialist composition needs cached coarse3 probs "
                           f"for {base_source} seed {seed}")
    p_base, fine_classes = cached
    fine_classes = list(fine_classes)
    truth = view.truth_fine
    composed = np.asarray(p_base, dtype="float64").copy()

    for block_name in blocks:
        block = FREQ_BLOCKS[block_name]
        idx = [fine_classes.index(c) for c in block]
        p_spec = np.zeros((len(truth), len(block)), dtype="float64")
        for tr, te in view.folds:
            rows = tr[np.isin(truth[tr], block)]
            model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(view.frame.iloc[rows],
                          view.target.iloc[rows].to_numpy())
                te_frame = view.frame.iloc[te].copy()
                te_frame[S2_MASK] = 0.0        # the deployed gate-off read
                p_fine, _ = model._probs(te_frame)
            p_spec[np.ix_(te, [block.index(c) for c in model.fine_classes_])] = p_fine
        p_spec /= np.clip(p_spec.sum(1, keepdims=True), 1e-12, None)
        composed[:, idx] = p_base[:, idx].sum(1, keepdims=True) * p_spec

    merged_classes = view.merged_classes
    M = np.zeros((len(fine_classes), len(merged_classes)))
    for i, c in enumerate(fine_classes):
        M[i, merged_classes.index(to_merged_label(c))] = 1.0
    fine = np.array(fine_classes, dtype=object)[composed.argmax(1)]
    return composed @ M, fine, None, (composed, fine_classes)


def _siam_spec_kwargs(ctx, **overrides):
    return siam_s2off_kwargs(ctx, siam_cos_weight=SIAM_AUX,
                             siam_cos_margin=0.3, **overrides)


def _deployed_spec_kwargs(ctx, **overrides):
    """The specialist built from the DEPLOYED recipe rather than the siamese one."""
    detail, kwargs = s2off_deployed_kwargs(ctx)
    kwargs.update(overrides)
    return ctx.aef_cols + detail, kwargs


def specialist_idea(name, desc, *, blocks, base="base_siam_s2off_cos_fine",
                    kwargs_fn=_siam_spec_kwargs, **overrides):
    """A section-V arm: the base's block masses plus one or two specialists."""
    def fn(ctx, view, seed):
        cols, kwargs = kwargs_fn(ctx, **overrides)
        return specialist_cv(view, cols, kwargs, seed, base, blocks)
    return register(name, reads=("full",), group="section-v", desc=desc)(fn)


specialist_idea(                                                          # V1
    "spec_tail4", blocks=["tail4"],
    desc="V1: the primary split. One specialist over the four classes the base returns at "
         "F1 <= 0.44 (526 plots, 4-way, imbalance 5:1), composed with the base's own mass "
         "on that block. The head classes keep the base's answer untouched, so this is a "
         "pure test of whether a model that never has to separate the rare transitions "
         "from 4,200 stable plots names them better -- N0's stated reason they die.")

specialist_idea(                                                          # V2
    "spec_tail6", blocks=["tail6"],
    desc="V2: the alternative cut -- every transition in the tail, only the three stable "
         "classes left to the base. 1,242 plots and 6 classes, so it is less starved than "
         "V1 and less balanced (8:1). Registered up front as the cut question: V1 and V2 "
         "bracket where the frequency line should fall, and running only one of them would "
         "make the cut look like a choice made after seeing a result.")

specialist_idea(                                                          # V3
    "spec_split45", blocks=["head5", "tail4"],
    desc="V3: the user's construction in full -- a head model over the five acceptable "
         "classes AND a tail model over the four, each reading only its own block, with "
         "the base reduced to a router that supplies the two block masses. If V1 moves and "
         "V3 does not, the head specialist is the cost: a model trained without the rare "
         "classes present cannot be worse at the head classes for any reason except lost "
         "data, and that is the same 3.6-doublings argument turned on the majority.")

specialist_idea(                                                          # V4
    "spec_split36", blocks=["head3", "tail6"],
    desc="V4: the V3 construction at the V2 cut, so the two-specialist question and the "
         "cut question do not have to be answered from one cell each.")


specialist_idea(                                                          # V7
    "spec_tail4_deployed", blocks=["tail4"], base="base_deployed_fine",
    kwargs_fn=_deployed_spec_kwargs,
    desc="V7: V1 rebuilt end to end on the DEPLOYED model -- its cached block masses, its "
         "recipe for the specialist. The O3c pattern, and it is not optional here: a "
         "construction measured on one base is an observation about that base until a "
         "second one carries it. It also answers the deployment question directly, since "
         "`s2off_centre_m3s3_bf` is what `infer_s2.py` actually serves.")


def sharpen_cv(view: View, seed: int, base_source: str, blocks: list[str],
               power: float):
    """THE CONTROL: concentrate the block mass without training anything.

    A specialist over four classes is *sharper* than a nine-way softmax on the
    same rows for reasons that have nothing to do with what it learned -- fewer
    rivals, a flatter prior, a balanced training set. Under a composition that
    fixes the block mass, sharpening alone raises the block's arg-max
    probability and can win the class outright. That is a decision-rule effect,
    it is free, and it is available from the cached base probabilities.

    So the base's own within-block conditional is raised to ``power`` and
    renormalised. ``power=inf`` is the limit case -- all the block mass on the
    base's own within-block arg-max -- which no specialist can beat by
    sharpening and which needs no parameter chosen. If the specialists do not
    clear this row, section V is a rediscovery of O3 with a training cost.
    """
    cached = load_oof_fine(base_source, view.name, seed)
    if cached is None:
        raise RuntimeError(f"the sharpening control needs cached probs for "
                           f"{base_source} seed {seed}")
    p_base, fine_classes = cached
    fine_classes = list(fine_classes)
    composed = np.asarray(p_base, dtype="float64").copy()

    for block_name in blocks:
        idx = [fine_classes.index(c) for c in FREQ_BLOCKS[block_name]]
        sub = np.asarray(p_base, dtype="float64")[:, idx]
        mass = sub.sum(1, keepdims=True)
        if np.isinf(power):
            cond = np.zeros_like(sub)
            cond[np.arange(len(sub)), sub.argmax(1)] = 1.0
        else:
            cond = np.power(np.clip(sub, 1e-12, None), power)
            cond /= np.clip(cond.sum(1, keepdims=True), 1e-12, None)
        composed[:, idx] = mass * cond

    merged_classes = view.merged_classes
    M = np.zeros((len(fine_classes), len(merged_classes)))
    for i, c in enumerate(fine_classes):
        M[i, merged_classes.index(to_merged_label(c))] = 1.0
    fine = np.array(fine_classes, dtype=object)[composed.argmax(1)]
    return composed @ M, fine, None, (composed, fine_classes)


def sharpen_idea(name, desc, *, blocks, power,
                 base="base_siam_s2off_cos_fine"):
    def fn(ctx, view, seed):
        return sharpen_cv(view, seed, base, blocks, power)
    return register(name, reads=("full",), group="section-v", desc=desc)(fn)


sharpen_idea(                                                             # V0
    "spec_sharp_tail4_max", blocks=["tail4"], power=float("inf"),
    desc="V0: THE CONTROL for V1, registered before V1 is read at more than one seed. All "
         "of the base's tail mass placed on the base's own within-tail arg-max: the "
         "sharpest any composition over this block can be, achieved without training "
         "anything. It isolates the part of a specialist's gain that is concentration "
         "rather than knowledge -- the same role the random-drop arm played in section T, "
         "and the reason that section's verdict held.")

sharpen_idea(                                                             # V0b
    "spec_sharp_tail4_t4", blocks=["tail4"], power=4.0,
    desc="V0b: the same control at power 4 rather than the limit, so the mechanism is "
         "measured as a curve and not only at its endpoint. If V0b already reaches V1, the "
         "specialist is a temperature setting.")

sharpen_idea(                                                             # V0c
    "spec_sharp_tail6_max", blocks=["tail6"], power=float("inf"),
    desc="V0c: the V0 control at the V2 cut, so both cuts have a matched free baseline.")

coarse3_gate_idea(                                                        # V5
    "c3gate_spec_tail4", "spec_tail4",
    "V5: the O3 cost gate over V1's composed probabilities. Both instruments break the "
    "same class -- O3 by moving a threshold on the base distribution, V1 by replacing the "
    "distribution -- so the F7 / N3b / O4 precedent predicts they land BETWEEN their "
    "parts rather than adding. Registered with that prediction stated, because a "
    "composition that lands between is evidence the two are one mechanism and a "
    "composition that adds is evidence they are not.")

conformal_idea(                                                           # V6
    "conf_c3_spec_tail4", "spec_tail4", level="coarse3", alpha="nested",
    desc="V6: R5f's Mondrian conformal cuts over V1's composed probabilities. Unlike V5 "
         "this is preregistered to ADD, on R5's own finding: search breaks `Artificial -> "
         "Cropland` and calibration breaks `Nature -> Cropland`, and neither touches the "
         "other's class. V1 owns the first outright (0.314 against O3's 0.211) and moves "
         "the second not at all, so the two instruments are disjoint here in a way V5's "
         "pair is not. The counter-check is R5f's own failure mode: it bought its focus "
         "macro by spending `Nature -> Nature` 0.763 -> 0.684, so read the majority "
         "classes beside the focus four or this row means nothing.")


# ---------------------------------------------------------------------------
# Section W -- loss-side class weighting, which this model has never carried
# ---------------------------------------------------------------------------
# Every siamese recipe inherits `BASE["loss"] = "focal"`, and focal maps to a
# flat ones-vector of class weights at all three levels. So the long-tail levers
# this project has tested -- G-H class-balanced sampling, cRT (O2), tau-norm
# (O2c), set-level Dice (Q6b), the merged2 cost gate (F3/N12) and the coarse3
# one (O3) -- act on the sampler, on a re-fitted classifier, on the set, or
# post-hoc on the decision rule. None of them reweights the per-class term of
# the loss itself, and that is the gap here.
#
# It looks tested and is not. `hier_variants_2yr.csv` swept the four loss modes
# ONCE, at ONE seed, on the 2023-era flat `wide` trunk, and `focal` won on
# change-F1 by 0.0046 -- inside the +/-0.005 band, below AUTORESEARCH's 3-seed
# floor. On the two balanced-accuracy columns `cb_focal` won: merged 0.6705 vs
# 0.6594, fine 0.4604 vs 0.4470. That is the shape a rare-class reweighting
# should have, and `focus_macro_f1` -- the metric sections N-V are scored on,
# which did not exist when that sweep ran -- is far closer to fine balanced
# accuracy than to change-F1. The incumbent loss was therefore selected against
# the wrong objective, underpowered, on a different architecture. TWOTOWER F5
# still carries it as an open TODO.
#
# `cb_focal` is Cui et al. (2019) effective-number weights (beta 0.999)
# multiplying the SAME focal modulation, so it is a one-factor change against
# N8b rather than the two-factor swap `weighted_ce` would be -- the T4ref
# precedent, where dropping focal cost -0.030 change-F1 on its own.


def _siam_s2off_w_idea(name, desc, **overrides):
    """A section-W variant of N8b: the best model, plus one weighting change."""
    def fn(ctx, view, seed):
        cols, kwargs = siam_s2off_kwargs(
            ctx, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3, **overrides)
        return s2off_cv(view, cols, kwargs, seed)
    return register(name, reads=("full",), group="section-w", desc=desc)(fn)


_siam_s2off_w_idea(
    "siam_s2off_cb",                                                      # W1
    loss="cb_focal",
    desc="W1: N8b with class-balanced focal in place of plain focal -- effective-number "
         "weights (Cui et al., beta 0.999, normalised to mean 1 so the learning rate is "
         "not also rescaled) on the gate, merged2 and coarse3 levels. The direct test of "
         "the one long-tail lever never pulled on this model. Read it on `focus_macro_f1` "
         "and the four commissioned transitions, NOT on change-F1: change-F1 is what "
         "selected plain focal in the first place, at one seed, and re-reading the same "
         "metric would only reproduce that choice. The counter-check is N13 -- if this "
         "buys focus macro by calling more change at lower precision, it is the tuned "
         "threshold in a costume and should be compared against it, not against N8b.")

_siam_s2off_w_idea(
    "siam_s2off_cb_fine",                                                 # W1b
    loss="cb_focal", cb_levels="fine",
    desc="W1b: the same weights on the COARSE3 level only, with the 2-class gate and "
         "merged2 left unweighted. Preregistered beside W1 rather than as a rescue: the "
         "gate is ~4:1 and reweighting it is a change/stable precision-recall trade, while "
         "the 4,200-vs-46 imbalance this section is about lives entirely at the fine "
         "level. W1 confounds the two in one `loss=` string; this arm separates them, so a "
         "flat W1 with a positive W1b would say the gate reweighting is what cancels the "
         "gain rather than that class weighting does not reach the tail.")


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
    parser.add_argument("--read", choices=["full", "subset", "lumped", "both"],
                        default="both")
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
            # The stable-Artificial columns do not exist on the lumped read --
            # that legend has no `Artificial -> Artificial` class by design -- so
            # report what the read actually carries rather than assuming a fixed
            # metric set. `focus_macro_f1` is the column every read shares.
            line = (f"  {name:26s} [{read:6s}] change_f1="
                    f"{row['change_f1_mean']:.4f}±{row['change_f1_std']:.3f}  "
                    f"macro_f1={row['macro_f1_mean']:.4f}  "
                    f"focus={row.get('focus_macro_f1_mean', float('nan')):.4f}")
            if "art_stable_recall_mean" in row:
                line += (f"  artStab={row['art_stable_recall_mean']:.3f} "
                         f"(as_veg={row['art_stable_as_veg_mean']:.3f})")
            print(line + f"  ({row['seconds']:.0f}s)", flush=True)
    print(f"\nledger -> {LEDGER}")


# -- W2 / W3: the composition ------------------------------------------------
# Section W left one thing open. W1 owns stable built-up (`art_stable_as_veg`
# 0.151 against the deployed model's 0.196 and the gated 0.165) and is the only
# instrument on the board that can move it at all, because it RETRAINS and the
# metric is read at merged2. O3 and V1 own `Artificial -> Cropland` and are free,
# but they are post-hoc re-reads of the coarse3 arg-max and leave every merged2
# metric bit-identical. The two therefore act at different levels on different
# metrics, which is why the F7 / N3b / O4 "two instruments on one class land
# BETWEEN their parts" precedent is predicted NOT to apply here -- stated before
# the run, as V5 was.
#
# The base changes and the instrument does not: `spec_tail4_cb` keeps V1's plain
# focal specialist and V1's fixed `FREQ_BLOCKS["tail4"]` cut, so the only moving
# part is which model's block masses it composes with. That is the `spec_tail4_deployed`
# (V7) pattern, for the same reason -- a construction measured on one base is an
# observation about that base until a second one carries it.

register("base_siam_s2off_cb_fine", reads=("full",), group="section-w",
         desc="W1c: W1 (`siam_s2off_cb`) re-run through the fine-probability path. "
              "Numerically the same model on the same folds and seeds -- it exists so O3 "
              "and V1 have a cached coarse3 DISTRIBUTION for the class-weighted model, "
              "which the W1 cache does not carry. The O0 pattern."
         )(lambda ctx, view, seed: s2off_cv_fine(
             view, *siam_s2off_kwargs(ctx, siam_cos_weight=0.3, siam_cos_margin=0.3,
                                      loss="cb_focal"),
             seed))

coarse3_gate_idea(                                                        # W2
    "c3gate_siam_s2off_cb", "base_siam_s2off_cb_fine",
    "W2: the O3 coarse3 cost gate over W1's distribution. O3 leaves every merged2 metric "
    "untouched by construction, so `art_stable_as_veg` MUST come through at W1's 0.151 -- "
    "that is a plumbing check, not a result. The result is whether the gate still finds "
    "the +0.047 focus macro it found on the plain-focal base, or whether W1's fine-level "
    "weights have already spent the headroom the gate was moving a threshold into.")

specialist_idea(                                                          # W3
    "spec_tail4_cb", blocks=["tail4"], base="base_siam_s2off_cb_fine",
    desc="W3: V1's tail-4 specialist over W1's block masses. V1 is the stronger of the two "
         "free instruments (focus macro 0.4581 against O3's 0.4318) and W1 is the only "
         "model that moves built-up, so this is the arm that could hold both. Read it "
         "against V1 on `focus_macro_f1` and against W1 on `art_stable_as_veg`; it has to "
         "clear BOTH parents on their own metric to be worth its retrain, and the honest "
         "failure mode is that it clears neither because W1's distribution is a worse "
         "thing to compose with than N8b's.")



if __name__ == "__main__":
    main()
