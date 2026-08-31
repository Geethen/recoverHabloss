"""Leave-Location-Time-Out validation, and the two weaker splits it is read against.

The problem this solves is created by the augmentation it is meant to judge. A
stable plot contributes seven rows -- the same 10 m footprint, seven consecutive
AlphaEarth years -- and consecutive years of unchanged ground are very nearly the
same vector. Split those rows at random and a model is scored on rows whose
near-duplicates it was trained on, so **every year-augmented arm wins by
construction** and the size of the win grows with the augmentation. That is not a
subtle effect and it is the whole reason this module exists.

LLTO removes it by cutting folds on **location**, so a held-out plot is held out
at *every* year, and by cutting those folds **spatially**, so the neighbouring
ground goes with it. ``assert_no_leak`` enforces the first as a hard invariant on
every fold, and the third split in the ladder below exists so the cost of not
having it is on the record rather than assumed.

The ladder
----------
``random``
    Rows split at random. **Leaks, on purpose** -- it is the number an
    unsuspecting run would report, and the gap to ``llto`` is the size of the
    trap.
``loc``
    Folds cut on location, assigned at random. Time-out but no space-out: fixes
    the near-duplicate leak, leaves spatial autocorrelation between a held-out
    plot and its neighbours.
``llto``
    Folds cut on location and *placed* by k-means over the coordinates, so a fold
    is a contiguous region. The default and the one verdicts are read on.
``block``
    Folds cut on the project's existing 20-degree ``block_id``, grouped into
    ``n_folds``. The comparison to everything already in the ledger, which is
    blocked on exactly this.

Two details are deliberate.

* **k-means runs on the unit sphere, not on (lon, lat).** The plots span
  lon -167..178 and lat -55..80. Euclidean k-means on degrees makes Alaska and
  Kamchatka the two furthest points on Earth and stretches every high-latitude
  fold east-west; on ``xyz`` the folds are actual caps. Pass
  ``space="lonlat"`` for the naive version if you want to see the difference.
* **External pool rows are folded too.** A GLanCE unit inside the held-out
  region is removed with the RECOVER plots there. It is never *scored* -- the
  test read is always RECOVER's own legend -- but leaving it in training would
  let an external pool answer a held-out region from labels drawn inside it,
  which is the same leak one source removed.

Whose coordinates place the folds (``fold_ref``)
------------------------------------------------
``reference`` (the default)
    The fold rule is fitted on the **RECOVER plots alone** and then applied to
    every pool row by nearest centroid. The fold geography is therefore a
    property of the protocol, and two arms at one seed hold out the same ground
    by construction.
``union``
    k-means over the training pool *and* the test pool together -- what this
    module did before 2026-07-31, and what every ledger row written before then
    used.

``union`` is not a paired protocol and section V0 measures the size of the
problem: adding LUCAS's 12,360 Europe-only rows drags the cluster centres into
Europe, and only 60% of the test plots keep their fold id at 5 folds, 29% at 20.
A "paired" difference across that is two arms scored on two different partitions
of the world, on folds whose between-fold spread is ~16 points. The effect is
small but not nil for the globally-spread GLanCE pool (99.6% agreement at 5
folds, 84% at 20), which is why the 5-fold numbers in sections U1-U3 survive and
the 10-fold ones are the weaker read.

Pseudo-labels
-------------
``kind == "pseudo"`` rows carry no state. They are filled **inside each fold** by
a stage-1 model fitted on that fold's genuinely-labelled training rows, then the
final model is fitted on everything -- the user's Common Ground bridge, with the
constraint that nothing outside the training folds is ever consulted. A pseudo
row never enters a test set.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from diagnose_state_pools import metrics as state_metrics
from statepre.data import FEATURES

SPLITS = ("llto", "loc", "random", "block")
FOLD_REFS = ("reference", "union")


def _unit_sphere(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat_r) * np.cos(lon_r),
                            np.cos(lat_r) * np.sin(lon_r),
                            np.sin(lat_r)])


def _coords(locs: pd.DataFrame, space: str) -> np.ndarray:
    return (_unit_sphere(locs["lon"].to_numpy(), locs["lat"].to_numpy())
            if space == "sphere" else locs[["lon", "lat"]].to_numpy())


def location_folds(frame: pd.DataFrame, n_folds: int, seed: int,
                   split: str = "llto", space: str = "sphere",
                   reference: pd.DataFrame | None = None) -> dict:
    """``loc_id -> fold`` for every location in ``frame``.

    ``reference``, when given, is the point cloud the fold rule is **fitted** on
    -- the RECOVER plots. Every other location is then assigned to the fold it
    falls in, so the partition is a property of the protocol rather than of
    whichever pool is under test, and two arms at one seed hold out the same
    ground. Without it the rule is fitted on ``frame`` itself, which is the old
    ``union`` behaviour and is not paired; see the module docstring.

    ``random`` has no location map by definition; ``run`` handles it separately.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "random":
        raise ValueError("the 'random' split is per-row and has no location map")

    cols = ["loc_id", "lon", "lat", "block_id"]
    locs = frame[cols].drop_duplicates("loc_id").reset_index(drop=True)
    fit_on = (locs if reference is None
              else reference[cols].drop_duplicates("loc_id").reset_index(drop=True))

    if split == "loc":
        # Hashed rather than drawn, so a location's fold does not depend on how
        # many other locations the arm happens to carry or on their order. The
        # drawn version made `loc` un-paired for exactly the reason `union` is.
        salt = str(seed).encode()
        locs["fold"] = [
            int.from_bytes(hashlib.blake2b(salt + str(v).encode(),
                                           digest_size=8).digest(), "big") % n_folds
            for v in locs["loc_id"]]
    elif split == "block":
        # Blocks are wildly uneven in size, so they are packed largest-first into
        # the currently-emptiest fold rather than dealt round-robin -- otherwise
        # one fold takes the 900-plot block and the fold means are incomparable.
        # Packed on the *reference* sizes: an external pool's own block counts
        # must not be able to re-pack the folds it is being judged on.
        sizes = fit_on["block_id"].value_counts()
        load = np.zeros(n_folds)
        assignment = {}
        for block, n in sizes.items():
            target = int(np.argmin(load))
            assignment[block] = target
            load[target] += n
        # A pool block the reference never reaches still has to go somewhere; it
        # joins the emptiest fold and is therefore held out with it.
        unseen = sorted(set(locs["block_id"]) - set(assignment))
        for block in unseen:
            assignment[block] = int(np.argmin(load))
        locs["fold"] = locs["block_id"].map(assignment).to_numpy()
    else:
        km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10).fit(
            _coords(fit_on, space))
        locs["fold"] = km.predict(_coords(locs, space))
        fit_on = fit_on.assign(fold=km.predict(_coords(fit_on, space)))

    return dict(zip(locs["loc_id"], _canonical(locs, fit_on)))


def _canonical(locs: pd.DataFrame, fit_on: pd.DataFrame | None = None) -> np.ndarray:
    """Relabel folds west-to-east so a fold id means the same region every seed.

    k-means returns the *same* five continental clusters at every seed on this
    point cloud but numbers them arbitrarily, so grouping the per-fold ledger by
    ``fold`` across seeds would average South America into Asia. Pairing within a
    seed is unaffected either way -- both arms see the same labelling -- but the
    per-fold read is only interpretable once the id is anchored to geography.

    The ordering is taken from ``fit_on`` (the reference cloud) when there is
    one, for the same reason the folds are: a pool concentrated in one region
    would otherwise drag a fold's mean longitude and permute the ids.
    """
    order = ((fit_on if fit_on is not None else locs)
             .groupby("fold")[["lon", "lat"]].mean()
             .sort_values(["lon", "lat"]).index)
    mapping = {old: new for new, old in enumerate(order)}
    return locs["fold"].map(mapping).to_numpy()


def shared_locations(train_locs: np.ndarray, test_locs: np.ndarray) -> set:
    return set(train_locs) & set(test_locs)


def assert_no_leak(train_locs: np.ndarray, test_locs: np.ndarray,
                   test_kind: np.ndarray) -> None:
    """No location may appear in both halves, at any year. The LLTO invariant."""
    shared = shared_locations(train_locs, test_locs)
    if shared:
        raise AssertionError(
            f"LLTO violated: {len(shared)} locations are in both the training "
            f"and the test half, e.g. {sorted(shared)[:3]}")
    if (test_kind != "observed").any():
        raise AssertionError("a non-observed row reached the test set")


WEIGHTS = ("none", "per_loc", "per_source", "per_cell")


def _row_weights(frame: pd.DataFrame, idx: np.ndarray, scheme: str) -> np.ndarray | None:
    """Undo a vote distortion: equalise the total weight per ``scheme``'s unit.

    ``per_loc``
        Every location the same total weight regardless of how many years it
        contributes. Without it, ``stable_years`` hands a stable plot seven votes
        and a changed plot two, which shifts both the class balance and the
        stable/changed balance of the training set. Then a difference against
        ``endpoints`` is partly a difference in sample weight, and the arm cannot
        be read as "temporal diversity helps". Section U1b.
    ``per_source``
        Every source -- ``recover``, ``glance``, ``lucas`` -- the same total
        weight. A pool added to another does not get to outvote it by being
        larger.
    ``per_cell``
        Every 20-degree block the same total weight. The **geographic** analogue
        of ``per_loc``, and the one the LUCAS arms are about: LUCAS puts 12,360
        rows into 8 blocks where GLanCE puts 13,118 into 83, so concatenating
        them does not add Europe to the pool, it makes the pool mostly Europe.
        ``per_loc``'s lesson was that a pool's *composition* is a lever
        independent of its content; this is that lever pointed at space.
    """
    if scheme == "none":
        return None
    if scheme not in WEIGHTS:
        raise ValueError(f"unknown row-weight scheme {scheme!r}, have {WEIGHTS}")
    column = {"per_loc": "loc_id", "per_source": "source",
              "per_cell": "block_id"}[scheme]
    unit = pd.Series(frame[column].to_numpy()[idx])
    # 1/count normalises each unit to total weight 1; the mean-scaling keeps the
    # summed weight equal to the row count, so a scheme cannot change the
    # effective learning rate as a side effect of changing the balance.
    weight = 1.0 / unit.map(unit.value_counts()).to_numpy()
    return weight / weight.mean()


def _read(truth: np.ndarray, guess: np.ndarray) -> dict:
    """``state_metrics``, but an empty subset reports ``n=0`` instead of raising.

    A stable-only or changed-only test pool is a legitimate configuration -- and
    a fold can legitimately contain no changed plots -- so an empty read is a
    missing number, not an error.
    """
    if not len(truth):
        return {"n": 0}
    return state_metrics(truth, guess)


class _Pool:
    """A long frame's columns pulled out once as arrays, since folds re-index it."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.X = frame[FEATURES].to_numpy("float64")
        self.year = frame["year"].to_numpy()
        self.loc = frame["loc_id"].to_numpy()
        self.state = frame["state"].to_numpy(dtype=object)
        self.kind = frame["kind"].to_numpy()
        self.changed = frame["changed"].to_numpy()


def _fit_predict(factory, pool: _Pool, test_pool: _Pool, train, test, seed,
                 weight_scheme):
    labelled = train[pool.kind[train] != "pseudo"]
    unlabelled = train[pool.kind[train] == "pseudo"]
    y = pool.state.copy()

    if len(unlabelled):
        # Stage 1: the bridge. Fitted on this fold's real labels only, used only
        # to fill this fold's pseudo rows.
        bridge = factory(seed)
        bridge.fit(pool.X[labelled], y[labelled], year=pool.year[labelled],
                   loc=pool.loc[labelled],
                   weight=_row_weights(pool.frame, labelled, weight_scheme))
        y[unlabelled] = bridge.predict(pool.X[unlabelled],
                                       year=pool.year[unlabelled])

    model = factory(seed)
    model.fit(pool.X[train], y[train], year=pool.year[train],
              loc=pool.loc[train],
              weight=_row_weights(pool.frame, train, weight_scheme))
    return model.predict(test_pool.X[test], year=test_pool.year[test])


def run(frame: pd.DataFrame, factory, *, test_frame: pd.DataFrame,
        n_folds: int = 5, seed: int = 0, split: str = "llto",
        space: str = "sphere", test_year: int = 2024, weight: str = "none",
        fold_ref: str = "reference", verbose: bool = False) -> dict:
    """One validation pass. Returns the reads keyed by name, plus per-fold detail.

    ``frame`` is the **training pool** -- whichever dataset arm is under test --
    and ``test_frame`` is held separate and fixed: RECOVER's own observed
    endpoint rows. Keeping them apart is what lets an arm that contains no
    RECOVER rows at all (``glance``) be scored on the same test set as one that
    is nothing but RECOVER rows, which a single-frame construction cannot do.

    Folds are cut once, on the test pool's locations under the default
    ``fold_ref="reference"``, and both pools are mapped onto them -- so every arm
    at a given seed holds out the same ground. ``fold_ref="union"`` restores the
    pre-2026-07-31 behaviour, where the cut was fitted on both pools together and
    an arm could therefore move its own fold boundaries.

    Three reads come off the test set: everything, and then split by whether the
    plot changed state across the window, because those are different questions.
    A stable plot's 2024 state is its 2018 state, so a model can score it by
    memorising 2018; the changed plots are where a 2024 read does 2024 work, and
    they are where the commissioned transitions live (P0 measured a 4.3 pt
    cross-year penalty there against none in the aggregate).
    """
    pool, test_pool = _Pool(frame), _Pool(test_frame)
    is_test_row = ((test_pool.kind == "observed")
                   & (test_pool.year == test_year))

    if split == "random":
        # Deliberately leaky: rows, not locations. The reported `leaked_locations`
        # is the count that makes the number below unreadable as a generalisation
        # estimate, and it is carried out of here rather than asserted away.
        rng = np.random.default_rng(seed)
        train_fold = rng.integers(0, n_folds, len(frame))
        test_fold = rng.integers(0, n_folds, len(test_frame))
    else:
        if fold_ref not in FOLD_REFS:
            raise ValueError(f"fold_ref must be one of {FOLD_REFS}, got {fold_ref!r}")
        cols = ["loc_id", "lon", "lat", "block_id"]
        mapping = location_folds(
            pd.concat([frame[cols], test_frame[cols]], ignore_index=True),
            n_folds, seed, split, space,
            reference=test_frame[cols] if fold_ref == "reference" else None)
        train_fold = pool.frame["loc_id"].map(mapping).to_numpy()
        test_fold = test_pool.frame["loc_id"].map(mapping).to_numpy()

    pred = np.empty(len(test_frame), dtype=object)
    scored = np.zeros(len(test_frame), dtype=bool)
    per_fold, leaked = [], 0
    for k in range(n_folds):
        test = np.flatnonzero(is_test_row & (test_fold == k))
        train = np.flatnonzero(train_fold != k)
        if not len(test) or not len(train):
            continue
        if split == "random":
            leaked += len(shared_locations(pool.loc[train], test_pool.loc[test]))
        else:
            assert_no_leak(pool.loc[train], test_pool.loc[test],
                           test_pool.kind[test])
        pred[test] = _fit_predict(factory, pool, test_pool, train, test,
                                  seed + k, weight)
        scored[test] = True
        row = _read(test_pool.state[test].astype(str), pred[test].astype(str))
        row.update(fold=k, n_train=len(train))
        per_fold.append(row)
        if verbose:
            print(f"    fold {k}: n_test={row['n']:>5,} n_train={row['n_train']:>6,} "
                  f"macro_f1={row['macro_f1']:.4f}", flush=True)

    idx = np.flatnonzero(scored)
    if not len(idx):
        raise RuntimeError("no test rows were scored -- check test_year")
    truth, guess = test_pool.state[idx].astype(str), pred[idx].astype(str)
    changed = test_pool.changed[idx]
    reads = {
        "t1_all": _read(truth, guess),
        "t1_changed": _read(truth[changed], guess[changed]),
        "t1_stable": _read(truth[~changed], guess[~changed]),
    }
    return {"reads": reads, "per_fold": per_fold, "leaked_locations": leaked,
            "n_folds_run": len(per_fold),
            "fold_sizes": [int(r["n"]) for r in per_fold],
            # The out-of-fold predictions themselves. Every scalar above is a
            # projection of these, and the ledger cannot hold a 3x3 table, so a
            # confusion matrix has to be built from the arrays or not at all.
            "truth": truth, "pred": guess, "changed": changed}


def test_pool_frame(plots: pd.DataFrame) -> pd.DataFrame:
    """The fixed test pool: RECOVER's observed endpoint rows, every plot.

    Deliberately *not* one of ``data.ARMS`` -- the test set is a property of the
    question, not of the arm under test, and building it here keeps it impossible
    to vary by accident.
    """
    from statepre.data import build

    return build("endpoints", plots)
