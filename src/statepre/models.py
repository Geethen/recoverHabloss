"""Architecture choices for the pretraining phase.

Every model here answers the same question -- one 64-D AlphaEarth vector, which
of ``artificial`` / ``cropland`` / ``nature`` -- and exposes the same two calls,
so ``llto.py`` can swap them without knowing which is which::

    model.fit(X, y, year=..., loc=..., weight=...)   -> self
    model.predict(X, year=...)                       -> array of state strings

``year`` and ``loc`` are handed to every model and ignored by most. They are in
the signature because a multi-year pool is the first thing in this project that
*could* use them, and the point of the package is to find out whether anything
should.

The ladder
----------
``linear``
    ``StandardScaler`` + multinomial logistic regression. The probe
    ``diagnose_state_pools`` and ``diagnose_state_year_transfer`` already report,
    so a number here is directly comparable to P0's 0.751 / 0.740.
``rf``
    Random forest, the model in the user's LLTO sketch. Present as a second
    non-parametric read: if a dataset effect shows up under the probe and not
    under the forest, it is a linear-separability effect, not a data effect.
``mlp``
    **The one that matters.** Byte-for-byte the encoder
    ``model_zoo._pretrain_state`` trains -- ``64 -> 512 -> 256 -> 128`` with
    BatchNorm, GELU and dropout 0.4, a linear ``128 -> 3`` state head, AdamW at
    2e-3 under OneCycle for 30 epochs at batch 2048. Anything measured on this
    transfers to the deployed phase without a re-derivation.
``mlp_year``
    ``mlp`` plus a per-year diagonal affine on the input -- ``model_zoo``'s
    ``year_adapter="input"``, which exists for two years, generalised to seven.
    Identity-initialised, so an untrained adapter *is* ``mlp`` and any deviation
    has to pay for itself. Tests the opposite of invariance: maybe the encoder
    should be told the year and allowed to calibrate it away.
``mlp_inv``
    ``mlp`` plus an explicit invariance term. Two rows of the same location in
    the same batch are pulled together in embedding space, weighted by
    ``inv_weight``. This is the user's temporal-robustness hypothesis written as
    a loss instead of as data, and the pair only exists in a multi-year pool.
``mlp_yeardrop``
    ``mlp`` where each epoch draws **one** row per location, year chosen
    uniformly. Same location budget as ``endpoints`` per epoch, same total
    variety as ``stable_years`` across epochs. It separates "the extra rows are
    extra gradient steps" from "the extra rows are extra views".
``mlp_wide`` / ``mlp_deep`` / ``mlp_xl`` / ``mlp_narrow``
    **Capacity**, the one axis of this encoder nobody has moved. Only ``hidden``
    changes, so every arm remains a drop-in for the deployed encoder -- the input
    is still one date, the output is still ``siam_dim``, and the last layer is
    still linear, which is what ``encode_single``, ``_SiameseTrunk``'s mixer
    width and the pyramid heads all depend on. ``mlp_narrow`` is the control that
    says whether the current 512/256 is a peak or merely a point on a slope.

All four ``mlp`` variants standardise the input with statistics fitted on the
**training rows only**, pooled across years. Pooling is not a detail: a per-year
standardisation would remove exactly the between-year offset the invariance
question is about, and ``model_zoo._prepare`` pools for the same reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from statepre.data import YEARS

#: name -> (factory(seed, **kw), one-line description)
ARCHS: dict[str, tuple] = {}


def arch(name: str, desc: str):
    def register(fn):
        ARCHS[name] = (fn, desc)
        return fn
    return register


class _Sklearn:
    """Adapter: an sklearn estimator behind the year/loc/weight signature."""

    def __init__(self, estimator, supports_weight: bool = True):
        self.estimator = estimator
        self.supports_weight = supports_weight

    def fit(self, X, y, *, year=None, loc=None, weight=None):
        kw = {}
        if weight is not None and self.supports_weight:
            step = self.estimator.steps[-1][0] if hasattr(self.estimator, "steps") \
                else None
            kw = {f"{step}__sample_weight": weight} if step else {
                "sample_weight": weight}
        self.estimator.fit(X, np.asarray(y, dtype=str), **kw)
        return self

    def predict(self, X, *, year=None):
        return self.estimator.predict(X)


@arch("linear", "StandardScaler + multinomial logistic regression (the project probe).")
def _linear(seed: int, **kw):
    return _Sklearn(make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed)))


@arch("rf", "random forest, 300 trees -- the model in the user's LLTO sketch.")
def _rf(seed: int, n_estimators: int = 300, **kw):
    return _Sklearn(RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=seed))


class _MLP:
    """``model_zoo._pretrain_state``'s encoder and head, standing alone.

    Held deliberately close to the original: the same widths, the same
    BatchNorm/GELU/dropout order, the same optimiser, schedule, epoch count and
    batch size. The variants below turn exactly one thing on at a time, so a
    difference measured here is a difference the deployed phase would also see.
    """

    def __init__(self, seed: int, *, epochs: int = 30, batch: int = 2048,
                 dropout: float = 0.4, siam_dim: int = 128, lr: float = 2e-3,
                 hidden: tuple[int, ...] = (512, 256),
                 year_adapter: bool = False, inv_weight: float = 0.0,
                 year_drop: bool = False, class_weight: str | None = None,
                 triplet_weight: float = 0.0, triplet_margin: float = 0.2,
                 device: str | None = None):
        self.seed = seed
        self.epochs, self.batch, self.dropout = epochs, batch, dropout
        self.siam_dim, self.lr, self.hidden = siam_dim, lr, tuple(hidden)
        self.triplet_weight, self.triplet_margin = triplet_weight, triplet_margin
        self.year_adapter, self.inv_weight, self.year_drop = (
            year_adapter, inv_weight, year_drop)
        self.class_weight = class_weight
        self._device = device

    # -- construction ----------------------------------------------------
    def _build(self, d_in: int, n_class: int):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        # `hidden` is the only thing the capacity arms move, and the two ends are
        # fixed by the siamese contract: the input is one date's `d_in` columns
        # (so `encode_single` still applies) and the output is exactly
        # `siam_dim`, because `_SiameseTrunk`'s mixer width `d_comb` is computed
        # from it. The last layer stays linear on purpose, as in `model_zoo`.
        layers, prev = [], d_in
        for width in self.hidden:
            layers += [nn.Linear(prev, width), nn.BatchNorm1d(width), nn.GELU(),
                       nn.Dropout(self.dropout)]
            prev = width
        layers.append(nn.Linear(prev, self.siam_dim))
        self.enc = nn.Sequential(*layers).to(device)
        self.head = nn.Linear(self.siam_dim, n_class).to(device)
        modules = [self.enc, self.head]
        if self.year_adapter:
            # One diagonal affine per year, identity-initialised. 2 x 64
            # parameters a year, not a second encoder.
            self.gain = nn.Parameter(torch.ones(len(YEARS), d_in, device=device))
            self.bias = nn.Parameter(torch.zeros(len(YEARS), d_in, device=device))
            self.params = list(self.enc.parameters()) + list(
                self.head.parameters()) + [self.gain, self.bias]
        else:
            self.gain = self.bias = None
            self.params = list(self.enc.parameters()) + list(self.head.parameters())
        self.modules_ = modules

    def _encode(self, X, year_idx):
        if self.gain is not None:
            X = self.gain[year_idx] * X + self.bias[year_idx]
        return self.enc(X)

    # -- fitting ---------------------------------------------------------
    def fit(self, X, y, *, year=None, loc=None, weight=None):
        import torch
        import torch.nn.functional as F

        X = np.asarray(X, dtype="float64")
        y = np.asarray(y, dtype=str)
        self.classes_ = np.array(sorted(set(y)))
        code = {c: i for i, c in enumerate(self.classes_)}

        # Pooled over years and fitted on the training rows only.
        self.mu_ = X.mean(0)
        self.sd_ = np.where(X.std(0) > 1e-8, X.std(0), 1.0)
        self._build(X.shape[1], len(self.classes_))

        year = (np.zeros(len(X), dtype=int) if year is None
                else np.searchsorted(np.asarray(YEARS), np.asarray(year)))
        loc = np.arange(len(X)) if loc is None else np.asarray(loc)
        weight = np.ones(len(X)) if weight is None else np.asarray(weight,
                                                                  dtype="float64")
        if self.class_weight == "balanced":
            # Folded into the row weight rather than passed to cross_entropy, so
            # it composes with `per_cell`/`per_source` instead of overriding
            # them: the batch loss is already a weighted mean and one more factor
            # is one more factor. The pools are unbalanced by construction --
            # LUCAS is 5,000/5,000/2,360 and the RECOVER endpoints are ~81%
            # nature -- and `artificial` is the state the project's open frontier
            # (stable built-up) is made of.
            counts = pd.Series(y).value_counts()
            weight = weight * (len(y) / (len(counts) * pd.Series(y).map(counts))
                               ).to_numpy()

        dev = self.device
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32, device=dev)
        yt = torch.tensor([code[v] for v in y], dtype=torch.long, device=dev)
        yr = torch.tensor(year, dtype=torch.long, device=dev)
        wt = torch.tensor(weight, dtype=torch.float32, device=dev)
        # Locations as contiguous integers, so the invariance term can find
        # same-location pairs in a batch with one sort.
        _, loc_code = np.unique(loc, return_inverse=True)
        lc = torch.tensor(loc_code, dtype=torch.long, device=dev)

        rng = np.random.default_rng(self.seed + 1_000_003)
        if self.year_drop:
            by_loc = _group_indices(loc_code)

        n = len(X)
        steps = max(1, int(np.ceil(
            (len(by_loc) if self.year_drop else n) / self.batch)))
        opt = torch.optim.AdamW(self.params, lr=self.lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, total_steps=self.epochs * steps)

        for module in self.modules_:
            module.train()
        for _ in range(self.epochs):
            if self.year_drop:
                pool = np.array([g[rng.integers(len(g))] for g in by_loc])
            else:
                pool = np.arange(n)
            order = rng.permutation(pool)
            for start in range(0, len(order), self.batch):
                chunk = order[start:start + self.batch]
                # A single row cannot pass BatchNorm in train mode; one leftover
                # row an epoch is dropped rather than padded (model_zoo does the
                # same, for the same reason).
                if len(chunk) < 2:
                    continue
                idx = torch.tensor(chunk, dtype=torch.long, device=dev)
                opt.zero_grad()
                z = self._encode(Xt[idx], yr[idx])
                w = wt[idx]
                loss = (F.cross_entropy(self.head(z), yt[idx], reduction="none")
                        * w).sum() / w.sum()
                if self.inv_weight > 0:
                    loss = loss + self.inv_weight * _invariance(z, lc[idx])
                if self.triplet_weight > 0:
                    loss = loss + self.triplet_weight * _batch_hard_triplet(
                        z, yt[idx], self.triplet_margin)
                loss.backward()
                opt.step()
                sched.step()
        self.train_loss_ = float(loss.detach())
        return self

    def predict(self, X, *, year=None):
        import torch

        X = np.asarray(X, dtype="float64")
        year = (np.zeros(len(X), dtype=int) if year is None
                else np.searchsorted(np.asarray(YEARS), np.asarray(year)))
        dev = self.device
        for module in self.modules_:
            module.eval()
        out = []
        with torch.no_grad():
            Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32,
                              device=dev)
            yr = torch.tensor(year, dtype=torch.long, device=dev)
            for start in range(0, len(X), 8192):
                sl = slice(start, start + 8192)
                logits = self.head(self._encode(Xt[sl], yr[sl]))
                out.append(logits.argmax(1).cpu().numpy())
        return self.classes_[np.concatenate(out)]


def _group_indices(codes: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by location code, for the year-drop sampler."""
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
    return np.split(order, bounds)


def _invariance(z, loc_code):
    """Mean squared distance between same-location embeddings in the batch.

    Normalised to unit length first, so the term asks for the same *direction*
    and cannot be satisfied by shrinking the whole embedding -- the collapse this
    kind of term is famous for. Returns 0 when the batch happens to contain no
    repeated location, which is what makes it safe to leave switched on for a
    single-year pool: the arm then reduces exactly to ``mlp``.
    """
    import torch
    import torch.nn.functional as F

    order = torch.argsort(loc_code)
    codes = loc_code[order]
    same = codes[1:] == codes[:-1]
    if not bool(same.any()):
        return torch.zeros((), device=z.device)
    zn = F.normalize(z[order], dim=1)
    pairs = (zn[1:][same] - zn[:-1][same]).pow(2).sum(1)
    return pairs.mean()


def _batch_hard_triplet(z, y, margin: float):
    """Batch-hard triplet (Hermans et al. 2017) on the unit sphere.

    For every anchor in the batch: the *furthest* same-state row and the
    *nearest* different-state row, then ``relu(d_pos - d_neg + margin)``. Batch
    hard rather than random triplets because with three classes and a batch of
    2048 almost every random triplet is already separated and contributes exactly
    zero gradient -- the term would be a no-op dressed as an experiment.

    Distances are cosine, on L2-normalised embeddings, matching ``_invariance``
    here and ``model_zoo._siam_cos_loss`` in the deployed model. That also caps
    ``d`` in [0, 2], which is what makes a fixed ``margin`` mean the same thing
    at every epoch instead of drifting with the embedding's scale.

    Anchors whose state is alone in the batch are dropped: they have no positive,
    and a "hardest positive" taken from another class would invert the objective.
    """
    import torch
    import torch.nn.functional as F

    zn = F.normalize(z, dim=1)
    dist = torch.cdist(zn, zn)
    same = y[:, None] == y[None, :]
    eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
    pos_ok = same & ~eye
    usable = pos_ok.any(1) & (~same).any(1)
    if not bool(usable.any()):
        return torch.zeros((), device=z.device)
    # Masked extremes: -inf where a positive is impossible so `max` cannot pick
    # it, +inf where a negative is impossible so `min` cannot.
    d_pos = dist.masked_fill(~pos_ok, float("-inf")).max(1).values
    d_neg = dist.masked_fill(same, float("inf")).min(1).values
    return F.relu(d_pos[usable] - d_neg[usable] + margin).mean()


@arch("mlp", "model_zoo._pretrain_state's own encoder: 64-512-256-128 + linear head.")
def _mlp(seed: int, **kw):
    return _MLP(seed, **kw)


@arch("mlp_year", "mlp + a per-year diagonal affine on the input (year_adapter).")
def _mlp_year(seed: int, **kw):
    return _MLP(seed, year_adapter=True, **kw)


@arch("mlp_inv", "mlp + a same-location embedding-invariance term (weight 0.1).")
def _mlp_inv(seed: int, inv_weight: float = 0.1, **kw):
    return _MLP(seed, inv_weight=inv_weight, **kw)


@arch("mlp_cw", "mlp with class-balanced row weights on the state head.")
def _mlp_cw(seed: int, **kw):
    return _MLP(seed, class_weight="balanced", **kw)


@arch("mlp_yeardrop", "mlp where each epoch draws one random year per location.")
def _mlp_yeardrop(seed: int, **kw):
    return _MLP(seed, year_drop=True, **kw)


# -- metric learning (section Z) ---------------------------------------------
#
# The reason hcropland30 is worth a second look. Section X could only ever
# *concatenate* it, because a pool of one class cannot train a softmax head, and
# its verdict was redundancy: resampling the cropland GLanCE already had matched
# it fold for fold. A triplet term asks a different question -- not "which state
# is this row" but "which rows are alike" -- and under that question a
# single-class pool stops being degenerate and becomes a **positive bank**:
# 32,343 globally-spread cropland points are 32,343 chances to say "cropland in
# Kenya and cropland in Iowa belong together", which is the invariance the
# Cropland/Nature boundary actually lacks.
#
# So section X's control has to be run again rather than inherited. If
# `glance_cropdup_endpoints` matches the hcropland arm here too, the pool is
# redundant under both objectives and the matter is closed for good.

@arch("mlp_trip", "mlp + a batch-hard triplet term on the embedding (weight 0.3).")
def _mlp_trip(seed: int, triplet_weight: float = 0.3, **kw):
    return _MLP(seed, triplet_weight=triplet_weight, **kw)


@arch("mlp_trip_hi", "mlp_trip at weight 1.0 -- the term as a peer of the CE, "
                     "not a garnish.")
def _mlp_trip_hi(seed: int, **kw):
    return _MLP(seed, triplet_weight=1.0, **kw)


@arch("mlp_trip_cw", "mlp_trip + the class-balanced head (Y3), since the two "
                     "act on different halves of the same objective.")
def _mlp_trip_cw(seed: int, **kw):
    return _MLP(seed, triplet_weight=0.3, class_weight="balanced", **kw)


# -- capacity (section Y) ----------------------------------------------------
#
# The one axis of this encoder nobody has moved. `mlp` is 64->512->256->128 in
# both this package and `model_zoo._SiameseTrunk`, and those widths have never
# been swept -- they were chosen once and inherited. The arms below change only
# `hidden`, so each stays a drop-in for the deployed encoder: same single-date
# input, same `siam_dim` output, same linear last layer, so `encode_single`,
# the pyramid heads (which read `stage_dims`) and the mixer are all unaffected.
#
# Read them on `artificial_as_cropland` as well as `macro_f1`. Extra capacity
# spent on separating Cropland from Nature is the point; extra capacity that
# also starts calling built-up "cropland" is the Oslo regression and is not a
# win however the aggregate moves.

@arch("mlp_wide", "capacity: 64-1024-512-128. Same depth, double the width.")
def _mlp_wide(seed: int, **kw):
    return _MLP(seed, hidden=(1024, 512), **kw)


@arch("mlp_deep", "capacity: 64-512-512-256-128. Same width, one more stage.")
def _mlp_deep(seed: int, **kw):
    return _MLP(seed, hidden=(512, 512, 256), **kw)


@arch("mlp_xl", "capacity: 64-1024-512-256-128. Wider and deeper.")
def _mlp_xl(seed: int, **kw):
    return _MLP(seed, hidden=(1024, 512, 256), **kw)


@arch("mlp_narrow", "capacity control: 64-256-128-128. The other direction.")
def _mlp_narrow(seed: int, **kw):
    return _MLP(seed, hidden=(256, 128), **kw)


def factory(name: str, **kw):
    """``factory(name)(seed)`` -> a fresh model. The callable ``llto.run`` wants."""
    if name not in ARCHS:
        raise KeyError(f"unknown architecture {name!r}; have {sorted(ARCHS)}")
    build = ARCHS[name][0]
    return lambda seed: build(seed, **kw)
