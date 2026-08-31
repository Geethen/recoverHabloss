"""Dataset choices for the state-pretraining phase: which single-date rows to feed it.

Every pool this module builds is a **long** frame -- one row per (location,
year) -- with the columns ``llto.py`` and ``models.py`` agree on:

===============  ==========================================================
``loc_id``       the location. Folds are cut on this and nothing else.
``lon``/``lat``  degrees, for the spatial fold assignment.
``block_id``     the project's 20-degree block, for the ``block`` fold scheme.
``year``         int, 2018..2024. A model may condition on it or ignore it.
``state``        ``artificial`` / ``cropland`` / ``nature``, or NA if pseudo.
``kind``         how the label was obtained -- see below.
``source``       ``recover`` / ``glance`` / ``lucas``.
``changed``      the location changed state 2018->2024 (RECOVER rows only).
``F00``..``F63`` the AlphaEarth embedding for that location at that year.
===============  ==========================================================

``kind`` is the honesty column and it is load-bearing:

``observed``
    A label somebody assigned to that location at that year. Only these are ever
    scored -- ``llto.py`` refuses to put anything else in a test fold.
``assumed``
    The user's hypothesis, made explicit. A plot whose coarse3 state is the same
    at 2018 and at 2024 is *assumed* to hold that state at 2019..2023 as well.
    Cheap and almost always right; wrong exactly when a plot was cleared and
    regrew inside the window, which the endpoints cannot see. It is training
    data, never test data.
``synthetic``
    A control row: the same location's endpoint embedding, duplicated or
    jittered. Carries no year information by construction.
``pseudo``
    ``state`` is NA and is filled *inside each fold* by a stage-1 model
    (``llto.py``). The intermediate years of a **changed** plot, which have no
    assumable state -- the plot was one thing and became another and the
    endpoints do not say when. This is the user's Common Ground bridge.

The arms
--------
``endpoints``
    2018 -> ``lc_2018`` and 2024 -> ``lc_2024``, every plot. The no-augmentation
    baseline, and the pool ``model_zoo._pretrain_state`` already builds for
    ``siam_state_source="endogenous"``.
``stable_years``
    ``endpoints`` plus 2019..2023 for the stable plots. **The hypothesis.**
``stable_years_dup``
    ``endpoints`` plus five copies of each stable plot's *2018* row. Identical
    row count, identical label distribution, zero temporal content. If this
    matches ``stable_years`` then the effect was sample weight, not time.
``stable_years_jit``
    As ``_dup``, but each copy gets Gaussian noise whose per-band standard
    deviation is the *observed* inter-annual spread of the stable plots. The
    harder control: it perturbs the input by the right amount in the wrong
    (isotropic, memoryless) way, so a gap over it is evidence that real
    inter-annual variation is structured rather than merely large.
``all_years_pseudo``
    ``stable_years`` plus the changed plots' 2019..2023 as ``pseudo``.
``glance``
    The external GLanCE pool at 2018 alone -- what P7's phase actually trains on
    today, restated in this frame so it is comparable.
``glance_stable_years``
    Both. The question P7 could not ask: does a temporally diverse endogenous
    pool make the external one redundant, or are they additive?
``hcrop_endpoints`` / ``glance_hcrop_endpoints`` / ``glance_hcropall_endpoints``
    The hybrid 30 m cropland map's 2020 points, **cropland only** -- its
    non-cropland class is "not cropland", which is ``{artificial, nature}`` plus
    the water/ice/barren coarse3 has no home for, and is dropped for that reason
    (``build_state_labels.hcropland_points``). A single-state pool cannot be
    trained alone, so there is no ``hcropland`` arm; it is only ever concatenated.

    These are aimed at a diagnosis, not at a gap in the map: V1b found LUCAS
    negative *because* it returns 66% of RECOVER's Cropland as Nature, and
    ``cropland_as_nature`` is the read they have to move. Two properties decide
    whether they can. The pool reaches **60 of the 83 blocks**, so unlike LUCAS
    it does not have to be read regionally. And it is a **map product** -- another
    model's decision boundary, the concern already on the record against GLanCE's
    GLC30 and MapBiomas components -- which the six-map unanimity filter
    (``strict``, 11,411 of 32,343) mitigates and ``_all`` ablates.
``glance_cropdup_endpoints``
    The control the arms above are unreadable without. See ``_resample_state``.
``lucas`` / ``lucas_endpoints`` / ``glance_lucas`` / ``glance_lucas_endpoints``
    The same ladder with LUCAS added. LUCAS is **12,360 in-situ field points and
    all of them are in EU-27** (lon -10..34, lat 35..70, 8 of the project's 83
    blocks), against GLanCE's 13,118 spread over all 83 -- comparable in size,
    incomparable in reach. Two consequences govern how these arms may be read:

    * At **5 folds LUCAS falls entirely inside one fold**, so ``lucas`` alone has
      no training rows at all when that fold is held out and its aggregate is
      computed over four folds against everything else's five. Read these arms at
      **20 folds**, where LUCAS spans five and 1,015 RECOVER plots sit inside its
      footprint.
    * The arms are only paired if the folds are, which is why section V0 had to
      fix the fold geography first: on the old ``union`` rule, adding LUCAS's
      12,360 European rows moved 40% of the test plots into a different fold.

Usage::

    python -m statepre.data --list
    python -m statepre.data --describe stable_years
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_merged_legend import LEGENDS
from project_paths import project_data_dir

YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)
ENDPOINTS = (2018, 2024)
INTERMEDIATE = tuple(y for y in YEARS if y not in ENDPOINTS)
STATES = ("artificial", "cropland", "nature")

RECOVER_FRAME = "embeddings_habloss_recover_annual.parquet"
FEATURES = [f"F{i:02d}" for i in range(64)]
META = ["loc_id", "lon", "lat", "block_id", "year", "state", "kind", "source",
        "changed"]


def bands(year: int) -> list[str]:
    return [f"A{i:02d}_{year}" for i in range(64)]


def load_plots(path: Path | None = None) -> pd.DataFrame:
    """RECOVER plots, deduped, complete at **every** year, with coarse3 endpoints.

    Completeness is required at all seven years rather than at the endpoints
    only. It costs 2 plots of 6,416 and it buys the property the whole comparison
    rests on: every arm below is built on the *same* locations, so a difference
    between two arms cannot be a difference in which plots they saw.
    """
    path = path or project_data_dir("embeddings") / RECOVER_FRAME
    frame = pd.read_parquet(path).drop_duplicates("PLOTID").reset_index(drop=True)
    complete = np.logical_and.reduce(
        [frame[bands(y)].notna().all(axis=1).to_numpy() for y in YEARS])
    frame = frame.loc[complete].reset_index(drop=True)

    legend = LEGENDS["coarse3"]
    out = pd.DataFrame({
        "loc_id": frame["PLOTID"].astype(str).to_numpy(),
        "lon": frame["lon"].astype("float64").to_numpy(),
        "lat": frame["lat"].astype("float64").to_numpy(),
        "block_id": frame["block_id"].to_numpy(),
    })
    for year in ENDPOINTS:
        raw = frame[f"lc_{year}"].astype(str).str.strip().str.lower()
        unknown = sorted(set(raw) - set(legend))
        if unknown:
            raise ValueError(f"lc_{year} outside the coarse3 legend: {unknown}")
        out[f"state_{year}"] = raw.map(legend).str.lower().to_numpy()
    out["changed"] = (out["state_2018"] != out["state_2024"]).to_numpy()
    # One concat rather than seven assignments: 448 columns inserted one block at
    # a time fragments the frame and pandas says so loudly.
    # pandas 3.x round-trips parquet floats as nullable extension dtypes, which
    # reach sklearn as object arrays (CLAUDE.md), hence the explicit cast.
    blocks = [pd.DataFrame(frame[bands(year)].astype("float64").to_numpy(),
                           columns=[f"{c}_{year}" for c in FEATURES])
              for year in YEARS]
    return pd.concat([out] + blocks, axis=1)


def _rows(plots: pd.DataFrame, mask, year: int, state, kind: str,
          features: np.ndarray | None = None) -> pd.DataFrame:
    """One long block: the masked plots at ``year``, labelled ``state``."""
    sub = plots.loc[mask]
    frame = pd.DataFrame({
        "loc_id": sub["loc_id"].to_numpy(),
        "lon": sub["lon"].to_numpy(),
        "lat": sub["lat"].to_numpy(),
        "block_id": sub["block_id"].to_numpy(),
        "year": year,
        "state": (sub[state].to_numpy() if isinstance(state, str)
                  else np.asarray(state)),
        "kind": kind,
        "source": "recover",
        "changed": sub["changed"].to_numpy(),
    })
    if features is None:
        features = sub[[f"{c}_{year}" for c in FEATURES]].to_numpy()
    frame[FEATURES] = features
    return frame


def _endpoints(plots: pd.DataFrame) -> list[pd.DataFrame]:
    everyone = np.ones(len(plots), dtype=bool)
    return [_rows(plots, everyone, y, f"state_{y}", "observed") for y in ENDPOINTS]


def _stable_years(plots: pd.DataFrame) -> list[pd.DataFrame]:
    stable = ~plots["changed"].to_numpy()
    return [_rows(plots, stable, y, "state_2018", "assumed") for y in INTERMEDIATE]


def _year_spread(plots: pd.DataFrame) -> np.ndarray:
    """Per-band SD of a stable plot's embedding around its own 7-year mean.

    The magnitude the jitter control has to match. Measured on the stable plots
    only -- a changed plot's spread across the window contains the change, which
    is the one thing the control must not import.
    """
    stable = plots.loc[~plots["changed"].to_numpy()]
    stack = np.stack([stable[[f"{c}_{y}" for c in FEATURES]].to_numpy()
                      for y in YEARS])                      # (year, plot, band)
    return stack.std(axis=0).mean(axis=0)                   # (band,)


def _stable_dup(plots: pd.DataFrame, jitter: bool, seed: int) -> list[pd.DataFrame]:
    stable = ~plots["changed"].to_numpy()
    base = plots.loc[stable, [f"{c}_2018" for c in FEATURES]].to_numpy()
    sd = _year_spread(plots) if jitter else None
    rng = np.random.default_rng(seed)
    out = []
    for year in INTERMEDIATE:
        X = base if sd is None else base + rng.normal(0.0, sd, base.shape)
        # The row is tagged with the year it stands in for, so a year-conditioned
        # architecture sees the same year vector it would see under `stable_years`
        # and the control stays a control for those too.
        out.append(_rows(plots, stable, year, "state_2018", "synthetic", X))
    return out


def _pseudo_years(plots: pd.DataFrame) -> list[pd.DataFrame]:
    changed = plots["changed"].to_numpy()
    n = int(changed.sum())
    return [_rows(plots, changed, y, np.full(n, None, dtype=object), "pseudo")
            for y in INTERMEDIATE]


#: The year each external pool's labels -- and therefore its embeddings -- were
#: extracted at. ``build_state_labels.SOURCE_YEAR`` is the same fact on the
#: writing side; it lives twice because the two scripts must not import each
#: other (this one has no ``ee`` dependency and must stay that way).
POOL_YEAR = {"glance_strict": 2018, "glance_broad": 2018, "lucas": 2018,
             "hcropland_all": 2020, "hcropland_strict": 2020}


def load_glance(pool: str = "glance_strict",
                embeddings_dir: Path | None = None) -> pd.DataFrame:
    """An external single-date pool, restated in the long frame.

    ``loc_id`` is prefixed so a pool unit can never collide with a PLOTID, and
    ``changed`` is False rather than NA: a GLanCE ``strict`` unit is by
    construction a *stable* segment covering its year, which is what the flag
    means here. The rows are ``observed`` -- somebody interpreted them -- but they
    never enter a test fold, because the test read is always RECOVER's legend.

    The pool's year comes from ``POOL_YEAR`` rather than being fixed at 2018:
    ``hcropland_*`` is a 2020 map and its rows carry ``year=2020``, which the
    frame has always had a column for and which the phase's single-date encoder
    is indifferent to.
    """
    return _long_pool(_read_pool(pool, embeddings_dir), pool)


def _read_pool(pool: str, embeddings_dir: Path | None = None) -> pd.DataFrame:
    """The pool's parquet as written, columns and all. Year comes from the name."""
    directory = embeddings_dir or project_data_dir("embeddings")
    if pool not in POOL_YEAR:
        raise KeyError(f"unknown pool {pool!r}; have {sorted(POOL_YEAR)}")
    path = directory / f"state_labels_{pool}_{POOL_YEAR[pool]}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run build_state_labels.py first")
    return pd.read_parquet(path).reset_index(drop=True)


def _long_pool(frame: pd.DataFrame, pool: str) -> pd.DataFrame:
    year = POOL_YEAR[pool]
    out = pd.DataFrame({
        "loc_id": pool + ":" + frame["sid"].astype(str),
        "lon": frame["lon"].astype("float64").to_numpy(),
        "lat": frame["lat"].astype("float64").to_numpy(),
        "block_id": frame["block_id"].to_numpy(),
        "year": year,
        "state": frame["state"].astype(str).str.lower().to_numpy(),
        "kind": "observed",
        "source": pool.split("_")[0],
        "changed": False,
    })
    out[FEATURES] = frame[bands(year)].astype("float64").to_numpy()
    return out


def load_hcropland(quality: str = "strict",
                   embeddings_dir: Path | None = None) -> pd.DataFrame:
    """The hybrid-cropland-map pool at 2020: cropland rows and nothing else.

    ``strict`` is cut from the ``all`` extraction on the ``uncertainty`` column
    rather than being its own file, so the two qualities are the same rows read
    two ways and a difference between them is the unanimity filter alone. A null
    uncertainty is not unanimity and does not survive the cut.
    """
    if quality not in ("strict", "all"):
        raise ValueError(f"quality must be 'strict' or 'all', got {quality!r}")
    frame = _read_pool("hcropland_all", embeddings_dir)
    if quality == "strict":
        keep = frame["uncertainty"].astype("float64").eq(0.0).fillna(False)
        frame = frame.loc[keep.to_numpy()].reset_index(drop=True)
    return _long_pool(frame, "hcropland_all")


#: name -> (builder, one-line description). The builder takes the wide plot
#: frame and a seed (used only by the jitter control) and returns a long frame.
ARMS: dict[str, tuple] = {}


def arm(name: str, desc: str):
    def register(fn):
        ARMS[name] = (fn, desc)
        return fn
    return register


@arm("endpoints", "2018 and 2024 observed states, every plot. The baseline.")
def _arm_endpoints(plots, seed, **kw):
    return pd.concat(_endpoints(plots), ignore_index=True)


@arm("stable_years",
     "endpoints + 2019..2023 for stable plots (assumed). THE HYPOTHESIS.")
def _arm_stable_years(plots, seed, **kw):
    return pd.concat(_endpoints(plots) + _stable_years(plots), ignore_index=True)


@arm("stable_years_dup",
     "row-count control: the extra rows are copies of the 2018 embedding.")
def _arm_stable_dup(plots, seed, **kw):
    return pd.concat(_endpoints(plots) + _stable_dup(plots, False, seed),
                     ignore_index=True)


@arm("stable_years_jit",
     "noise control: copies + Gaussian noise at the observed inter-year SD.")
def _arm_stable_jit(plots, seed, **kw):
    return pd.concat(_endpoints(plots) + _stable_dup(plots, True, seed),
                     ignore_index=True)


@arm("all_years_pseudo",
     "stable_years + the changed plots' intermediate years, pseudo-labelled.")
def _arm_all_years(plots, seed, **kw):
    return pd.concat(_endpoints(plots) + _stable_years(plots)
                     + _pseudo_years(plots), ignore_index=True)


@arm("glance", "the external GLanCE 2018 pool alone -- what P7's phase trains on.")
def _arm_glance(plots, seed, pool="glance_strict", **kw):
    return load_glance(pool)


@arm("glance_endpoints", "GLanCE 2018 + the endpoints. P7's phase, restated.")
def _arm_glance_endpoints(plots, seed, pool="glance_strict", **kw):
    return pd.concat([load_glance(pool)] + _endpoints(plots), ignore_index=True)


@arm("glance_stable_years", "GLanCE 2018 + the full year-augmented endogenous pool.")
def _arm_glance_stable(plots, seed, pool="glance_strict", **kw):
    return pd.concat([load_glance(pool)] + _endpoints(plots)
                     + _stable_years(plots), ignore_index=True)


@arm("lucas", "the LUCAS 2018 in-situ pool alone -- 12,360 rows, EU-27 only.")
def _arm_lucas(plots, seed, **kw):
    return load_glance("lucas")


@arm("lucas_endpoints", "LUCAS 2018 + the endpoints. The regional analogue of "
                        "glance_endpoints.")
def _arm_lucas_endpoints(plots, seed, **kw):
    return pd.concat([load_glance("lucas")] + _endpoints(plots),
                     ignore_index=True)


@arm("glance_lucas", "both external pools, no RECOVER rows.")
def _arm_glance_lucas(plots, seed, pool="glance_strict", **kw):
    return pd.concat([load_glance(pool), load_glance("lucas")],
                     ignore_index=True)


@arm("glance_lucas_endpoints",
     "U3's winning union + LUCAS. THE HYPOTHESIS OF SECTION V.")
def _arm_glance_lucas_endpoints(plots, seed, pool="glance_strict", **kw):
    return pd.concat([load_glance(pool), load_glance("lucas")]
                     + _endpoints(plots), ignore_index=True)


def _resample_footprint(pool: pd.DataFrame, footprint: pd.DataFrame,
                        seed: int) -> pd.DataFrame:
    """``pool``'s rows inside ``footprint``'s blocks, resampled to its row count.

    The density control for a regional pool, and the geographic twin of
    ``stable_years_dup``. LUCAS adds 12,360 rows to eight blocks; if that helps,
    the reading could be "in-situ European labels are better" or merely "eight
    blocks got twelve thousand more rows". This arm supplies the second without
    the first -- same count, same blocks, no information GLanCE did not already
    have -- so a gap over it is evidence about the labels rather than the volume.
    """
    blocks = set(footprint["block_id"])
    inside = pool.loc[pool["block_id"].isin(blocks)]
    if inside.empty:
        raise ValueError("the pool has no rows inside the footprint's blocks")
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(inside), len(footprint))
    out = inside.iloc[take].copy()
    # Distinct loc_ids or the fold map collapses the copies onto one location and
    # `per_loc` would silently down-weight the whole control to a single vote.
    out["loc_id"] = [f"{v}#dup{i}" for i, v in enumerate(out["loc_id"])]
    out["kind"] = "synthetic"
    return out.reset_index(drop=True)


def _resample_state(pool: pd.DataFrame, state: str, n: int,
                    seed: int) -> pd.DataFrame:
    """``n`` rows drawn with replacement from ``pool``'s rows of one state.

    The **class-density** control, and the reason ``glance_hcrop_endpoints`` is
    readable. hcropland adds 11,411 cropland rows to a 13,118-row pool that held
    5,000 -- it more than trebles one class and takes the pool from 38% cropland
    to 63% before a single new label is consulted. Section V3 already showed that
    moving the class balance alone moves this read, so a win over
    ``glance_endpoints`` could be either thing. This arm supplies the balance
    without the labels: same count, same state, same source, no information
    GLanCE did not already have.
    """
    inside = pool.loc[pool["state"] == state]
    if inside.empty:
        raise ValueError(f"the pool has no {state} rows to resample")
    rng = np.random.default_rng(seed)
    out = inside.iloc[rng.integers(0, len(inside), n)].copy()
    # Distinct loc_ids or the fold map collapses the copies onto one location.
    out["loc_id"] = [f"{v}#dup{i}" for i, v in enumerate(out["loc_id"])]
    out["kind"] = "synthetic"
    return out.reset_index(drop=True)


@arm("hcrop_endpoints",
     "hcropland30 cropland (unanimous) + the endpoints. The pool with no "
     "external Nature or Artificial at all.")
def _arm_hcrop_endpoints(plots, seed, hcrop_quality="strict", **kw):
    return pd.concat([load_hcropland(hcrop_quality)] + _endpoints(plots),
                     ignore_index=True)


@arm("glance_hcrop_endpoints",
     "U3's winning union + hcropland30 cropland. THE HYPOTHESIS OF SECTION W.")
def _arm_glance_hcrop_endpoints(plots, seed, pool="glance_strict",
                                hcrop_quality="strict", **kw):
    return pd.concat([load_glance(pool), load_hcropland(hcrop_quality)]
                     + _endpoints(plots), ignore_index=True)


@arm("glance_hcropall_endpoints",
     "as glance_hcrop_endpoints but every cropland point, unanimous or not -- "
     "the unanimity filter's own ablation.")
def _arm_glance_hcropall_endpoints(plots, seed, pool="glance_strict", **kw):
    return _arm_glance_hcrop_endpoints(plots, seed, pool=pool,
                                       hcrop_quality="all")


@arm("glance_cropdup_endpoints",
     "class-density control: glance_endpoints + GLanCE's own cropland resampled "
     "to hcropland30's count. Same balance, no new labels.")
def _arm_glance_cropdup_endpoints(plots, seed, pool="glance_strict",
                                  hcrop_quality="strict", **kw):
    glance = load_glance(pool)
    n = len(load_hcropland(hcrop_quality))
    return pd.concat([glance, _resample_state(glance, "cropland", n, seed)]
                     + _endpoints(plots), ignore_index=True)


@arm("glance_eudup_endpoints",
     "density control: glance_endpoints + GLanCE resampled to LUCAS's count "
     "inside LUCAS's blocks. Same rows added, no new labels.")
def _arm_glance_eudup_endpoints(plots, seed, pool="glance_strict", **kw):
    glance = load_glance(pool)
    return pd.concat([glance,
                      _resample_footprint(glance, load_glance("lucas"), seed)]
                     + _endpoints(plots), ignore_index=True)


def build(name: str, plots: pd.DataFrame | None = None, seed: int = 0,
          **kw) -> pd.DataFrame:
    """The named arm as a long frame, columns ``META + FEATURES``."""
    if name not in ARMS:
        raise KeyError(f"unknown dataset arm {name!r}; have {sorted(ARMS)}")
    plots = load_plots() if plots is None else plots
    frame = ARMS[name][0](plots, seed, **kw)
    missing = [c for c in META + FEATURES if c not in frame.columns]
    if missing:
        raise AssertionError(f"{name} is missing columns {missing}")
    observed = frame["kind"] != "pseudo"
    bad = sorted(set(frame.loc[observed, "state"].dropna()) - set(STATES))
    if bad:
        raise AssertionError(f"{name} carries states outside coarse3: {bad}")
    return frame[META + FEATURES].reset_index(drop=True)


def describe(name: str, frame: pd.DataFrame) -> str:
    lines = [f"{name}: {len(frame):,} rows, "
             f"{frame['loc_id'].nunique():,} locations, "
             f"years {sorted(frame['year'].unique())}"]
    kinds = frame.groupby(["source", "kind"]).size()
    for (source, kind), n in kinds.items():
        lines.append(f"  {source:<8s} {kind:<10s} {n:>7,}")
    counts = frame["state"].value_counts(dropna=False).to_dict()
    lines.append("  states  " + ", ".join(
        f"{k}={v:,}" for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--describe", nargs="*", default=None)
    args = parser.parse_args()

    if args.list or args.describe is None:
        for name, (_, desc) in ARMS.items():
            print(f"  {name:<22s} {desc}")
        return
    plots = load_plots()
    print(f"plots: {len(plots):,} complete at all 7 years, "
          f"{int(plots['changed'].sum()):,} changed at coarse3\n")
    for name in (args.describe or list(ARMS)):
        print(describe(name, build(name, plots)))


if __name__ == "__main__":
    main()
