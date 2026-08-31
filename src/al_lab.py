"""Active-learning replay lab -- simulate an acquisition campaign on the 6,414
labelled plots and read the tradeoffs between setups.

Why a replay and not the real thing
-----------------------------------
The real question is "which acquisition surface should steer the next ~1,250
patches", and the real answer costs months of interpreter time. This harness
answers the *cheap* half of it, which `ACTIVE_LEARNING.md` build order lists as
step 2 and says to run before anything else is built:

    hide most of the existing labels, let a strategy ask for them back a batch at
    a time, and read what it bought per acquisition.

**What it measures is ranking quality, and nothing else.** The 6,414 plots are
not a random sample of the globe -- change classes are enriched ~30x over their
share of land -- so a realised "plots per acquisition" here does NOT transfer to
a global draw. What does transfer is the *ordering* of strategies and the shape
of the tradeoff between them. Any absolute yield quoted from this file is wrong
by the enrichment factor and must be labelled as such.

The protocol, and the three ways it could have been rigged
----------------------------------------------------------
1. **The test set is a held-out spatial region**, k-means on the unit sphere
   (the `statepre/llto.py` geometry), never a random row split. With a random
   split an acquisition function that simply picks plots *near* the test set
   wins on spatial autocorrelation alone, and every diversity method would have
   been buried by it. `STATE_PRETRAIN_RESEARCH.md` has 20-degree blocks reading
   0.024 high and *reversing* an ordering, so the fold rule is not a detail.
2. **Every strategy at one (seed, fold) starts from the same seed set and is
   scored on the same test rows**, so arms are compared *paired*. Seed sd on
   this target is ~0.02 and the effects here are ~0.01; unpaired it is unreadable.
   `AL0` measures the paired floor explicitly -- run it before believing any gap.
3. **A strategy may not look at a label it has not acquired.** Scores take the
   pool's features and the current model, never `view.target[pool]`. The one
   exception is `oracle_*`, which is clearly named and exists only as a ceiling.

Run
---
    G=/home/geethen.singh/.pixi/envs/geo
    PROJ_DATA=$G/share/proj PROJ_LIB=$G/share/proj GDAL_DATA=$G/share/gdal \\
    /home/geethen.singh/.cache/phoenix-test/venv/bin/python src/al_lab.py \\
        --list
        --strategies random bald kcenter --seeds 5

Appends one row per (arm, seed, fold, round) to
``data/analysis_results/al_lab_ledger.csv``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import acquisition as acq
from model_zoo import HierarchicalSoftmaxNN, is_change_label
from project_paths import project_data_dir
from twotower_lab import (AEF_MASK, S2_MASK, load_context, s2_subset_columns)
from twotower_metrics import prf

LEDGER = project_data_dir("analysis_results") / "al_lab_ledger.csv"

#: The deployed recipe (CLAUDE.md), gate-off read. Held identical to
#: `learning_curves.DEPLOYED` so a curve from this file and a curve from that one
#: are the same model at the same sizes.
DEPLOYED = dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
                fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
                mask_column=S2_MASK, aef_mask_column=AEF_MASK)
SUBSET = "centre_m3s3_bf"

#: Floor of training rows per class, so the fine head always emits nine classes.
#: Same constant and same reason as `learning_curves.MIN_PER_CLASS`.
MIN_PER_CLASS = 2

#: The six change transitions, in the order the ledger reports them.
CHANGE_CLASSES = ("Artificial -> Cropland", "Artificial -> Nature",
                  "Cropland -> Artificial", "Cropland -> Nature",
                  "Nature -> Artificial", "Nature -> Cropland")


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------
@dataclass
class AlContext:
    """Everything a simulation needs, built once and reused across arms."""

    frame: pd.DataFrame          # compact: model cols + lon/lat only
    target: np.ndarray           # coarse3 transition, object
    truth_merged: np.ndarray
    cols: list                   # model input columns
    kwargs: dict                 # HierarchicalSoftmaxNN kwargs
    fine_classes: list
    merged_classes: list
    spaces: dict = field(default_factory=dict)   # name -> (n, d) float64
    #: Row-aligned terrain covariates (`extract_terrain_gee.py`). Used ONLY to
    #: define a coverage gap and to read whether one was closed -- never as a
    #: model input and never by a strategy, which would be an oracle.
    terrain: pd.DataFrame | None = None
    #: Boolean per row: is this plot in the stratum a `biased_terrain` start
    #: deliberately withholds? See `common_terrain`.
    gap: np.ndarray | None = None


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), acq.EPS, None)


def build_context(s2_subset: str = SUBSET) -> AlContext:
    """Load the modelling frame and precompute the model-free feature spaces."""
    ctx = load_context()
    view = ctx.view("full")
    s2 = s2_subset_columns(ctx.s2_stat_cols, s2_subset)
    cols = ctx.aef_cols + s2
    kwargs = dict(DEPLOYED, aef_columns=ctx.aef_cols, tess_columns=s2)

    keep = list(dict.fromkeys(cols + [S2_MASK, AEF_MASK, "lon", "lat"]))
    frame = view.frame[keep].copy()
    # pandas 3.x round-trips parquet floats as nullable extension dtypes that
    # reach sklearn as object arrays (CLAUDE.md).
    for c in cols + [S2_MASK, AEF_MASK]:
        frame[c] = frame[c].astype("float64")

    a18 = np.asarray(view.frame[[f"A{i:02d}_2018" for i in range(64)]], float)
    a24 = np.asarray(view.frame[[f"A{i:02d}_2024" for i in range(64)]], float)
    spaces = {
        # "what kind of place is this" -- the coverage/diversity space.
        "state": _l2(np.hstack([a18, a24])),
        # "what happened here" -- the retrieval direction measured at AUC 0.915
        # for Artificial -> Cropland (ACTIVE_LEARNING.md). Different question
        # from the tested-negative per-band ND *classifier features*.
        "delta": _l2(a24 - a18),
    }
    terrain, gap = _terrain(view.frame)
    return AlContext(terrain=terrain, gap=gap,
                     frame=frame.reset_index(drop=True),
                     target=view.target.to_numpy(),
                     truth_merged=np.asarray(view.truth_merged),
                     cols=cols, kwargs=kwargs,
                     fine_classes=sorted(set(view.truth_fine)),
                     merged_classes=list(view.merged_classes),
                     spaces=spaces)


#: The withheld stratum for a `biased_terrain` start: steep ground, plus the
#: WorldCover classes the user's two map errors sit on -- bare/snow/moss for the
#: "mountains read as Artificial" error, wetland/mangrove for "wetlands read as
#: Cropland". 1,202 of 6,414 plots (18.7%), which is the size this test wants:
#: large enough to measure recovery against, small enough that a random draw
#: returns it slowly. Widening it to slope > 3 takes it to 49.5%, at which point
#: random recovers it as fast as anything else and the test says nothing.
GAP_WORLDCOVER = (60, 70, 90, 95, 100)   # bare, snow/ice, wetland, mangrove, moss
GAP_SLOPE_DEG = 8.0


def _terrain(frame: pd.DataFrame) -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Row-aligned terrain, and the boolean 'is in the withheld stratum' mask."""
    path = project_data_dir("analysis_results") / "terrain_plots.parquet"
    if not path.exists():
        return None, None
    t = pd.read_parquet(path)
    t["PLOTID"] = t["PLOTID"].astype(str)
    aligned = (pd.DataFrame({"PLOTID": frame["PLOTID"].astype(str).to_numpy()})
               .merge(t.drop_duplicates("PLOTID"), on="PLOTID", how="left"))
    steep = aligned["slope"].fillna(0.0) > GAP_SLOPE_DEG
    odd = aligned["worldcover"].isin(GAP_WORLDCOVER)
    return aligned, np.asarray(steep | odd)


def spatial_folds(lon: np.ndarray, lat: np.ndarray, n_folds: int,
                  seed: int) -> np.ndarray:
    """k-means on the unit sphere -> one fold id per row.

    The `statepre/llto.py` geometry, minus the location/time machinery this frame
    does not need (one row per plot, one plot per location). Folds are relabelled
    west-to-east so a fold id means the same region at every seed.
    """
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    xyz = np.column_stack([np.cos(lat_r) * np.cos(lon_r),
                           np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    lab = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit_predict(xyz)
    order = (pd.DataFrame({"f": lab, "lon": lon, "lat": lat})
             .groupby("f")[["lon", "lat"]].mean()
             .sort_values(["lon", "lat"]).index)
    return np.asarray(pd.Series(lab).map({o: i for i, o in enumerate(order)}))


# ---------------------------------------------------------------------------
# the model step
# ---------------------------------------------------------------------------
def _place(block: np.ndarray, local: list, classes: list) -> np.ndarray:
    """Fold-local probability block -> global class order, matched by NAME.

    Positional assignment silently permutes classes exactly where a rare class
    drops out of a small training draw, which is every early round here.
    """
    out = np.zeros((len(block), len(classes)))
    out[:, [classes.index(c) for c in local]] = block
    return out


def fit_predict(ctx: AlContext, train: np.ndarray, score_on: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit on ``train`` rows, return (fine, merged) probabilities on ``score_on``.

    Predictions are taken with the detail gate forced off -- the deployed serving
    configuration (`infer_s2.probs_aef_only_matrix`).
    """
    model = HierarchicalSoftmaxNN(ctx.cols, seed=seed, **ctx.kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(ctx.frame.iloc[train], ctx.target[train])
        te = ctx.frame.iloc[score_on].copy()
        te[S2_MASK] = 0.0
        pf, pm = model._probs(te)
    return (_place(pf, list(model.fine_classes_), ctx.fine_classes),
            _place(pm, list(model.merged_classes_), ctx.merged_classes))


def evaluate(ctx: AlContext, pf: np.ndarray, pm: np.ndarray,
             test: np.ndarray) -> dict:
    """Every metric the loop reads, from one pair of probability blocks."""
    fine = np.asarray(ctx.fine_classes, dtype=object)[pf.argmax(1)]
    merged = np.asarray(ctx.merged_classes, dtype=object)[pm.argmax(1)]
    t_fine, t_merged = ctx.target[test], ctx.truth_merged[test]

    p, r, f = prf(np.array([is_change_label(x) for x in t_merged]),
                  np.array([is_change_label(x) for x in merged]))
    out = dict(change_precision=p, change_recall=r, change_f1=f)

    f1s, chg = [], []
    for c in ctx.fine_classes:
        _, _, cf = prf(t_fine == c, fine == c)
        f1s.append(cf)
        key = c.replace(" -> ", "2").replace(" ", "")
        out[f"f1__{key}"] = cf
        if c in CHANGE_CLASSES:
            chg.append(cf)
    out["macro_f1"] = float(np.mean(f1s))
    out["change_macro_f1"] = float(np.mean(chg)) if chg else np.nan

    # The stable-class confusions the user reports seeing on the map. They cost
    # nothing on change-F1 by construction -- both sides are stable -- so they
    # are invisible to every aggregate above and have to be carried separately.
    nat = t_fine == "Nature -> Nature"
    out["natStab_as_art"] = float((fine[nat] == "Artificial -> Artificial").mean()) \
        if nat.any() else np.nan
    out["natStab_as_crop"] = float((fine[nat] == "Cropland -> Cropland").mean()) \
        if nat.any() else np.nan
    art = t_fine == "Artificial -> Artificial"
    out["artStab_recall"] = float((fine[art] == "Artificial -> Artificial").mean()) \
        if art.any() else np.nan
    crop = t_fine == "Cropland -> Cropland"
    out["cropStab_recall"] = float((fine[crop] == "Cropland -> Cropland").mean()) \
        if crop.any() else np.nan
    return out


# ---------------------------------------------------------------------------
# acquisition strategies
# ---------------------------------------------------------------------------
@dataclass
class Round:
    """What a strategy is allowed to see when it scores the pool."""

    ctx: AlContext
    labelled: np.ndarray        # row indices already acquired
    pool: np.ndarray            # row indices still hidden -- the candidates
    seed: int
    round_idx: int
    n_rounds: int
    _probs: dict = field(default_factory=dict)   # lazy model reads, cached
    cache: dict = field(default_factory=dict)   # clusterings, one per round

    def probs(self, n_members: int = 1) -> np.ndarray:
        """``(n_members, len(pool), n_fine)`` posteriors from the current model.

        Cached per member count, because within one round `bald` and `entropy`
        would otherwise refit the identical network.
        """
        if n_members not in self._probs:
            mem = [fit_predict(self.ctx, self.labelled, self.pool,
                               self.seed * 100 + m)[0] for m in range(n_members)]
            self._probs[n_members] = np.stack(mem)
        return self._probs[n_members]

    @property
    def frac_done(self) -> float:
        return self.round_idx / max(1, self.n_rounds)


STRATEGIES: dict[str, tuple] = {}


def strategy(name: str, *, group: str, desc: str):
    def deco(fn):
        STRATEGIES[name] = (fn, group, desc)
        return fn
    return deco


# --- the baseline ----------------------------------------------------------
@strategy("random", group="baseline",
          desc="Equal weight. The baseline every method must beat; also the "
               "arm AL0 runs against itself to get the paired noise floor.")
def _random(rd: Round) -> np.ndarray:
    return np.random.default_rng(rd.seed * 1000 + rd.round_idx).random(len(rd.pool))


def _random_stream(offset: int):
    def fn(rd: Round) -> np.ndarray:
        return np.random.default_rng(
            offset + rd.seed * 1000 + rd.round_idx).random(len(rd.pool))
    return fn


#: `random` at a fixed (seed, fold) is deterministic, so it cannot be run
#: against itself. These are the same strategy on a different RNG stream, and
#: the spread between the three IS the paired floor -- the smallest gap this
#: harness can resolve. Every verdict below is quoted against it, the same way
#: map comparisons are quoted against the ~0.84 self-IoU floor (CLAUDE.md).
for _i, _n in ((7_000_000, "random_b"), (13_000_000, "random_c")):
    STRATEGIES[_n] = (_random_stream(_i), "baseline",
                      "`random` on an independent RNG stream -- the paired "
                      "noise floor, not a method.")


# --- model-in-the-loop: read the posterior ---------------------------------
@strategy("entropy", group="uncertainty",
          desc="Normalised Shannon entropy of the nine-class posterior.")
def _entropy(rd: Round) -> np.ndarray:
    return acq.normalised_entropy(rd.probs(1)[0])


@strategy("margin", group="uncertainty",
          desc="1 - (p1 - p2). Knife-edge decisions, which entropy cannot "
               "separate from mass smeared over nine classes.")
def _margin(rd: Round) -> np.ndarray:
    return acq.margin(rd.probs(1)[0])


@strategy("least_conf", group="uncertainty",
          desc="Probability mass not on the arg-max.")
def _least_conf(rd: Round) -> np.ndarray:
    return acq.least_confidence(rd.probs(1)[0])


@strategy("bald", group="uncertainty",
          desc="H(mean p) - mean H(p) over 3 members: the reducible part of the "
               "uncertainty only. The one uncertainty score that can decline to "
               "buy another argument about the Cropland/Nature label boundary.")
def _bald(rd: Round) -> np.ndarray:
    return acq.bald(rd.probs(3))


@strategy("vote_entropy", group="uncertainty",
          desc="Disagreement on the arg-max alone over 3 members -- the one "
               "disagreement measure a miscalibrated member cannot corrupt.")
def _vote(rd: Round) -> np.ndarray:
    return acq.vote_entropy(rd.probs(3))


@strategy("conformal_size", group="uncertainty",
          desc="Mondrian LAC set size: how many classes cannot be ruled out at "
               "90%. Calibrated on a third of the labelled set.")
def _conformal(rd: Round) -> np.ndarray:
    qhat = _mondrian_qhat(rd)
    return acq.conformal_set_size(rd.probs(1)[0], qhat)


def _mondrian_qhat(rd: Round, alpha: float = 0.10) -> np.ndarray:
    """Per-class LAC quantiles from a third of the labelled set.

    A scalar (marginal) qhat reintroduces exactly the blindness the channel
    exists to fix: `CONFORMAL_TORCHCP.md` has the marginal SplitPredictor reading
    0.8999 coverage while covering `Cropland -> Nature` 13% of the time.
    """
    rng = np.random.default_rng(rd.seed * 7 + rd.round_idx)
    perm = rng.permutation(len(rd.labelled))
    n_cal = max(len(rd.ctx.fine_classes) * 2, len(perm) // 3)
    cal, fit = rd.labelled[perm[:n_cal]], rd.labelled[perm[n_cal:]]
    if len(fit) < len(rd.ctx.fine_classes) * MIN_PER_CLASS:
        return np.ones(len(rd.ctx.fine_classes))
    pf, _ = fit_predict(rd.ctx, fit, cal, rd.seed)
    y = rd.ctx.target[cal]
    qhat = np.ones(len(rd.ctx.fine_classes))
    for k, c in enumerate(rd.ctx.fine_classes):
        m = y == c
        if m.sum() < 2:
            continue
        scores = 1.0 - pf[m, k]
        n = m.sum()
        lvl = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        qhat[k] = float(np.quantile(scores, lvl, method="higher"))
    return qhat


# --- model-free: read the embedding ----------------------------------------
def nearest_distance(x: np.ndarray, ref: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Per-row cosine distance to the nearest row of ``ref``.

    `acq.novelty_to_reference` is the same quantity reduced at p90 over the
    pixels of a *patch*; a plot is a single point, so the reduction is the
    identity and the per-row form is what a plot-level replay needs. Chunked
    because the pool x labelled product is ~5,000 x 2,000 every round.

    Both arguments are re-normalised even though `AlContext.spaces` already
    holds unit vectors, so this is a no-op on the replay's own inputs. It is
    here because without it the function returns a dot product on anything else
    -- values outside [0, 2], silently -- and a caller passing a raw embedding
    would get a plausible-looking ranking that is not a distance.
    """
    x = _l2(np.asarray(x, dtype=np.float64))
    ref = _l2(np.asarray(ref, dtype=np.float64))
    out = np.empty(len(x))
    for i in range(0, len(x), chunk):
        out[i:i + chunk] = 1.0 - (x[i:i + chunk] @ ref.T).max(axis=1)
    return out


@strategy("novelty", group="diversity",
          desc="Cosine distance to the nearest already-labelled plot in the "
               "state space. Coverage, referenced to the LABEL set, not the pool.")
def _novelty(rd: Round) -> np.ndarray:
    x = rd.ctx.spaces["state"]
    return nearest_distance(x[rd.pool], x[rd.labelled])


@strategy("kcenter", group="diversity",
          desc="k-center greedy seeded with the labelled set: take the point "
               "furthest from everything selected, repeat. Batch-aware, so it "
               "cannot spend the batch on one unrepresented biome.")
def _kcenter(rd: Round) -> np.ndarray:
    return None     # handled specially -- returns an ORDER, not a score


@strategy("rarity", group="diversity",
          desc="Zaytar eq. 1: cluster the pool, make every cluster equally "
               "likely, spread its mass over its members. Rare-looking terrain "
               "gets weight proportional to how rare it looks. No labels.")
def _rarity(rd: Round) -> np.ndarray:
    lab = _cluster(rd, "state")
    return acq.cluster_inverse_size(lab)


@strategy("delta_rarity", group="diversity",
          desc="`rarity` in the change-vector space instead of the state space "
               "-- rare *transitions* rather than rare places.")
def _delta_rarity(rd: Round) -> np.ndarray:
    lab = _cluster(rd, "delta")
    return acq.cluster_inverse_size(lab)


@strategy("fa", group="diversity",
          desc="Core-set Feature Activation. Untested on a signed modality: "
               "their features are ReLU-nonnegative and AlphaEarth's are not.")
def _fa(rd: Round) -> np.ndarray:
    return acq.feature_activation(rd.ctx.spaces["state"][rd.pool], scale=True)


def _cluster(rd: Round, space: str, k: int = 40) -> np.ndarray:
    """Bisecting-KMeans-flavoured labels over the pool. Cached per round.

    The bootstrapping paper's measured ordering is Bisecting KMeans >> DBSCAN,
    and it is a large gap, not a preference. sklearn's BisectingKMeans is used
    where available; plain KMeans is the fallback and is noted in the ledger.
    """
    key = f"clu_{space}_{k}"
    if key not in rd.cache:
        try:
            from sklearn.cluster import BisectingKMeans as _KM
        except ImportError:
            _KM = KMeans
        x = rd.ctx.spaces[space][rd.pool]
        rd.cache[key] = _KM(n_clusters=max(2, min(k, len(x) // 5)),
                            random_state=rd.seed).fit_predict(x)
    return rd.cache[key]


# --- class-targeted retrieval ----------------------------------------------
@strategy("proto_sim", group="retrieval",
          desc="Cosine similarity to the mean change-vector of the change "
               "classes already acquired. The model-free channel that reaches a "
               "class the posterior never exceeds 0.191 on.")
def _proto_sim(rd: Round) -> np.ndarray:
    x = rd.ctx.spaces["delta"]
    seen = rd.ctx.target[rd.labelled]
    mask = np.array([is_change_label(t) for t in seen])
    if mask.sum() < 5:
        return np.zeros(len(rd.pool))
    proto = _l2(x[rd.labelled[mask]].mean(0, keepdims=True))
    return (x[rd.pool] @ proto.T).ravel()


@strategy("delta_mag", group="retrieval",
          desc="Plain cosine change magnitude, no labels at all. The control "
               "`proto_sim` has to beat: is the prototype direction doing "
               "anything a bare 'something moved here' score does not?")
def _delta_mag(rd: Round) -> np.ndarray:
    a = rd.ctx.spaces["state"][rd.pool]
    d = a.shape[1] // 2
    return 1.0 - (a[:, :d] * a[:, d:]).sum(1) / (
        np.linalg.norm(a[:, :d], axis=1) * np.linalg.norm(a[:, d:], axis=1) + acq.EPS)


@strategy("pred_change", group="retrieval",
          desc="Posterior mass on the six change classes. The model-in-the-loop "
               "twin of `delta_mag`, and the arm that shows what the 0.191 "
               "ceiling costs a retrieval channel.")
def _pred_change(rd: Round) -> np.ndarray:
    p = rd.probs(1)[0]
    idx = [i for i, c in enumerate(rd.ctx.fine_classes) if is_change_label(c)]
    return p[:, idx].sum(1)


@strategy("pred_balance", group="composition",
          desc="Core-set Class Balance on PREDICTED class: take the candidate "
               "that leaves the predicted-class histogram of the labelled set "
               "flattest. Chases whichever class is scarcest with no quotas.")
def _pred_balance(rd: Round) -> np.ndarray:
    return None     # handled specially -- greedy, returns an ORDER


# --- composed --------------------------------------------------------------
@strategy("bald_novelty", group="hybrid",
          desc="rank_mean(bald, novelty). Average the percentile RANKS: a "
               "cosine distance and an entropy are on incomparable scales.")
def _bald_novelty(rd: Round) -> np.ndarray:
    return acq.rank_mean(_bald(rd), _novelty(rd))


@strategy("switch_div_unc", group="hybrid",
          desc="The user's hypothesis, made falsifiable: novelty for the first "
               "half of the campaign, BALD for the second. If diversity-then-"
               "complexity is real this beats both of its halves.")
def _switch(rd: Round) -> np.ndarray:
    return _novelty(rd) if rd.frac_done < 0.5 else _bald(rd)


@strategy("entropy_novelty", group="hybrid",
          desc="rank_mean(entropy, novelty). The blend built on the arm that "
               "actually won AL1, rather than on BALD.")
def _ent_nov(rd: Round) -> np.ndarray:
    return acq.rank_mean(_entropy(rd), _novelty(rd))


@strategy("switch_rand_unc", group="hybrid",
          desc="Random until the campaign is half done, then entropy. The "
               "schedule AL1 implies rather than the one it was asked about: "
               "uncertainty is WORSE than random below ~1,200 labels and better "
               "above it, and diversity is flat throughout, so the thing to "
               "delay is uncertainty, not the thing to lead with diversity.")
def _switch_rand(rd: Round) -> np.ndarray:
    return _random(rd) if rd.frac_done < 0.5 else _entropy(rd)


@strategy("switch_unc_div", group="hybrid",
          desc="The same switch, reversed. Present because a hybrid that beats "
               "both halves in EITHER order is a batch-size artefact, not a "
               "regime effect -- this is the control that tells them apart.")
def _switch_rev(rd: Round) -> np.ndarray:
    return _bald(rd) if rd.frac_done < 0.5 else _novelty(rd)


# --- ceilings (see a label they have not acquired -- clearly named) ---------
@strategy("oracle_change", group="oracle",
          desc="CEILING, NOT A METHOD. Acquires change plots first. The most "
               "any retrieval channel could buy if it were perfect.")
def _oracle_change(rd: Round) -> np.ndarray:
    return np.array([1.0 if is_change_label(t) else 0.0
                     for t in rd.ctx.target[rd.pool]])


@strategy("oracle_balance", group="oracle",
          desc="CEILING, NOT A METHOD. Perfect class balance on the true label.")
def _oracle_balance(rd: Round) -> np.ndarray:
    y = rd.ctx.target[rd.pool]
    have = pd.Series(rd.ctx.target[rd.labelled]).value_counts()
    n = np.array([have.get(t, 0) for t in y], float)
    return 1.0 / (1.0 + n)


# ---------------------------------------------------------------------------
# the greedy strategies, which pick a batch rather than score it
# ---------------------------------------------------------------------------
def _pick(rd: Round, name: str, batch: int) -> np.ndarray:
    """Positions *within* ``rd.pool`` that this strategy acquires this round."""
    if name == "kcenter":
        return acq.kcenter_greedy(
            np.vstack([rd.ctx.spaces["state"][rd.pool],
                       rd.ctx.spaces["state"][rd.labelled]]),
            batch, selected=np.arange(len(rd.pool), len(rd.pool) + len(rd.labelled)))
    if name == "pred_balance":
        p = rd.probs(1)[0]
        counts = np.eye(len(rd.ctx.fine_classes))[p.argmax(1)]
        have = pd.Series(rd.ctx.target[rd.labelled]).value_counts()
        prior = np.array([have.get(c, 0) for c in rd.ctx.fine_classes], float)
        return acq.class_balance_greedy(counts, batch, prior=prior)
    scores = STRATEGIES[name][0](rd)
    # Ties are everywhere at small pools (`fa` saturates, `vote_entropy` is
    # integer-valued); break them at random rather than by row order, which is
    # PLOTID order and correlates with source.
    jitter = np.random.default_rng(rd.seed * 31 + rd.round_idx).random(len(scores))
    return np.lexsort((jitter, -np.asarray(scores, float)))[:batch]


# ---------------------------------------------------------------------------
# one trajectory
# ---------------------------------------------------------------------------
def simulate(ctx: AlContext, name: str, *, seed: int, fold: int, folds: np.ndarray,
             n_seed_set: int, batch: int, n_rounds: int,
             seed_mode: str = "random") -> list[dict]:
    """One (strategy, seed, fold) campaign. Returns one row per round."""
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(folds == fold)
    avail = np.flatnonzero(folds != fold)

    # The starting labels. `random` is the honest cold start; `stratified`
    # guarantees the fine head can emit all nine classes from round 0 and is the
    # right control when the question is about acquisition, not about warm-up.
    # `biased_terrain` is the arm that makes the replay able to answer the
    # question it otherwise cannot. In every other seed mode the starting labels
    # and the pool are draws from the SAME distribution, so there is no coverage
    # gap for a diversity score to close and "diversity does nothing" is
    # guaranteed by the design rather than measured. Here the start is drawn
    # only from ordinary low-slope vegetated ground and the whole awkward
    # stratum -- steep, bare, wet -- is left in the pool.
    if seed_mode == "biased_terrain":
        if ctx.gap is None:
            raise SystemExit("biased_terrain needs terrain_plots.parquet; "
                             "run src/extract_terrain_gee.py first")
        ok = avail[~ctx.gap[avail]]
        labelled = np.sort(rng.permutation(ok)[:n_seed_set])
    elif seed_mode == "stratified":
        start = []
        for c in ctx.fine_classes:
            pool_c = avail[ctx.target[avail] == c]
            k = max(MIN_PER_CLASS,
                    int(round(n_seed_set * len(pool_c) / len(avail))))
            start.append(rng.permutation(pool_c)[:min(k, len(pool_c))])
        labelled = np.sort(np.concatenate(start))
    else:
        labelled = np.sort(rng.permutation(avail)[:n_seed_set])
    pool = np.setdiff1d(avail, labelled)

    rows, t0 = [], time.time()
    for r in range(n_rounds + 1):
        pf, pm = fit_predict(ctx, labelled, test, seed)
        got = ctx.target[labelled]
        row = dict(arm=name, seed=seed, fold=fold, round=r,
                   n_labelled=len(labelled), n_pool=len(pool),
                   n_test=len(test), batch=batch, n_rounds=n_rounds,
                   n_seed_set=n_seed_set, seed_mode=seed_mode)
        row.update(evaluate(ctx, pf, pm, test))
        row["acq_change_n"] = int(sum(is_change_label(t) for t in got))
        row["acq_change_frac"] = row["acq_change_n"] / len(got)
        for c in CHANGE_CLASSES:
            row[f"n__{c.replace(' -> ', '2').replace(' ', '')}"] = int((got == c).sum())
        # Effective number of distinct places the campaign has bought so far.
        # Computable with no labels at all, which makes it the only read here
        # that survives the move to the real global pool.
        if ctx.gap is not None:
            # How much of the withheld stratum the campaign has bought back.
            # Under a `biased_terrain` start this is the coverage read; under
            # any other start it is simply the stratum's share and is expected
            # to sit at the frame's own rate, which is the control.
            row["acq_gap_n"] = int(ctx.gap[labelled].sum())
            row["acq_gap_frac"] = float(ctx.gap[labelled].mean())
        row["vendi_state"] = acq.vendi_score(ctx.spaces["state"][labelled])
        row["vendi_delta"] = acq.vendi_score(ctx.spaces["delta"][labelled])
        row["secs"] = round(time.time() - t0, 1)
        rows.append(row)

        if r == n_rounds or len(pool) == 0:
            break
        rd = Round(ctx, labelled, pool, seed, r, n_rounds)
        take = _pick(rd, name, min(batch, len(pool)))
        labelled = np.sort(np.concatenate([labelled, pool[take]]))
        pool = np.setdiff1d(pool, pool[take])
    return rows


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------
def append_ledger(rows: list[dict], path: Path = LEDGER) -> None:
    df = pd.DataFrame(rows)
    df["stamp"] = time.strftime("%Y%m%d_%H%M%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def ledger(path: Path = LEDGER) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--strategies", nargs="+", default=["random"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--use-folds", nargs="+", type=int, default=None,
                    help="which fold ids to hold out (default: all)")
    ap.add_argument("--n-seed-set", type=int, default=400)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--seed-mode",
                    choices=("random", "stratified", "biased_terrain"),
                    default="stratified")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.list:
        for n, (_, g, d) in sorted(STRATEGIES.items(), key=lambda kv: kv[1][1]):
            print(f"{g:<12} {n:<16} {d}")
        return

    unknown = set(args.strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies: {sorted(unknown)}")

    ctx = build_context()
    print(f"{len(ctx.frame):,} plots | {len(ctx.cols)} model cols | "
          f"{len(ctx.fine_classes)} classes", flush=True)

    rows = []
    for seed in range(args.seeds):
        folds = spatial_folds(ctx.frame["lon"].to_numpy(),
                              ctx.frame["lat"].to_numpy(), args.folds, seed)
        use = args.use_folds if args.use_folds is not None else range(args.folds)
        for fold in use:
            for name in args.strategies:
                t0 = time.time()
                r = simulate(ctx, name, seed=seed, fold=fold, folds=folds,
                             n_seed_set=args.n_seed_set, batch=args.batch,
                             n_rounds=args.rounds, seed_mode=args.seed_mode)
                for row in r:
                    row["tag"] = args.tag
                rows.extend(r)
                print(f"  seed {seed} fold {fold} {name:<16} "
                      f"final change-F1 {r[-1]['change_f1']:.4f} "
                      f"chg-n {r[-1]['acq_change_n']:4d} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    if args.dry_run:
        print("\n(dry run -- ledger not written)")
    else:
        append_ledger(rows)
        print(f"\n-> {LEDGER}  (+{len(rows)} rows)")


if __name__ == "__main__":
    main()
