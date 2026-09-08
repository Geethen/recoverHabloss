"""Candidates for the **coverage** channel: terrain-stratified, novelty-ranked.

What this channel is for, and what it is not for
------------------------------------------------
`ACTIVE_LEARNING.md` AL1/AL3 measured coverage sampling against random on five
metrics. It is worth **nothing** on change-F1 -- flat at every start size, with
or without a deliberate coverage gap. What it does buy, in 21 of 25 spatial
folds and at twice the paired floor, is the two stable-class errors that are
visible on the map and invisible to every aggregate in the ledger:

    Nature -> Nature read as Artificial -> Artificial   -0.020  (`novelty`)
    ... at the cost of Nature -> Nature read as Cropland  +0.016

**That is a ratio trade and both halves must be reported.** A round of these
points that fixes the bare-ground error and quietly worsens the wetland one has
not obviously helped, and the only way anyone finds out is if both are quoted.

Two ideas, and they do different jobs
-------------------------------------
1. **Stratify** on the terrain the label set is missing. AL-T measured the label
   set against a 30,000-point equal-area draw of global land: `bare` is 17.4% of
   land and 6.6% of the labels, `moss/lichen` 2.5% against 0.5%, `wetland` 1.6%
   against 0.9%. That is the targeting -- it says *where* to look.
2. **Rank by novelty within the stratum.** `novelty` = cosine distance to the
   nearest already-labelled plot in AlphaEarth state space, the exact arm AL3
   measured. That is the ordering -- it stops 75 points landing in one desert.

Doing only (1) buys the stratum and a lot of redundancy; doing only (2) is the
AL3 arm and does not know which terrain is short. The batch does both.

Two-stage sampling, because the obvious one-stage version is slow
-----------------------------------------------------------------
AlphaEarth is 64 bands per year, so a 128-band sample over 40,000 points is a
large `getInfo` payload and most of it is thrown away. Stage 1 samples only the
four cheap covariates and keeps the rows in an under-sampled stratum; stage 2
samples the embeddings for the survivors only, which is ~5x less traffic.

Run
---
    G=/home/geethen.singh/.pixi/envs/geo
    PROJ_DATA=$G/share/proj PROJ_LIB=$G/share/proj GDAL_DATA=$G/share/gdal \\
    /home/geethen.singh/.cache/phoenix-test/venv/bin/python \\
        src/build_coverage_candidates.py --n-select 75

Writes ``data/analysis_results/coverage_candidates.csv``, ranked, ready for
``build_label_batches.py --channel coverage``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir

RESULTS = project_data_dir("analysis_results")
LAND_SHARE = RESULTS / "terrain_land_share.json"
OUT = RESULTS / "coverage_candidates.csv"

#: WorldCover v200 codes -> the names AL-T's land-share table is keyed on.
WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 70: "snow/ice", 80: "water", 90: "wetland",
              95: "mangrove", 100: "moss/lichen"}

#: Strata a point must be in to be a coverage candidate: those the label set
#: under-samples relative to their share of global land (ratio < 1 in AL-T's
#: table). `water` is excluded even though it is under-sampled -- open water is
#: not land and the legend has nowhere to put it.
EXCLUDE_FROM_STRATA = ("water",)

#: A stratum has to be worth a whole point before it gets one. `snow/ice` is
#: under-sampled 125x but is 1.9% of land and mostly uninterpretable at 10 m;
#: this floor keeps the allocation from spending the batch on it.
MIN_LAND_SHARE = 0.005


def init_gee(project: str = "ee-gsingh"):
    import ee
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=project)
    return ee


# ---------------------------------------------------------------------------
def equal_area_draw(n: int, seed: int) -> pd.DataFrame:
    """Uniform in longitude and in **sin(latitude)** -- the `global_patches.py`
    frame, so a candidate and a land share are draws from one population."""
    rng = np.random.default_rng(seed)
    s_lo, s_hi = np.sin(np.radians([-60.0, 84.0]))
    return pd.DataFrame({
        "id": [f"c{i:06d}" for i in range(n)],
        "lon": rng.uniform(-180.0, 180.0, n),
        "lat": np.degrees(np.arcsin(rng.uniform(s_lo, s_hi, n))),
    })


def sample_points(ee, img, frame: pd.DataFrame, scale: int,
                  chunk: int, label: str) -> pd.DataFrame:
    """`sampleRegions` in chunks, with the retry the network needs."""
    parts, t0 = [], time.time()
    for i in range(0, len(frame), chunk):
        sub = frame.iloc[i:i + chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                       {"id": r.id}) for r in sub.itertuples()])
        for attempt in range(4):
            try:
                rows = img.sampleRegions(collection=fc, scale=scale,
                                         geometries=False).getInfo()["features"]
                parts.append(pd.DataFrame([f["properties"] for f in rows]))
                break
            except Exception as exc:                          # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"    retry {attempt + 1}: {type(exc).__name__} "
                      f"{str(exc)[:100]}", flush=True)
                time.sleep(5 * (attempt + 1))
        print(f"  {label} {min(i + chunk, len(frame)):6d}/{len(frame)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(parts, ignore_index=True)


def covariate_image(ee):
    """Terrain and land cover -- the cheap stage-1 bands."""
    srtm = ee.Image("USGS/SRTMGL1_003")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()
    # unmask everything: `sampleRegions` DROPS a point whose bands are all
    # masked, and JRC masks all land that has never held water -- which silently
    # returned 95 of 6,414 rows the first time (see extract_terrain_gee.py).
    return (ee.Terrain.slope(srtm).unmask(0).rename("slope")
            .addBands(srtm.unmask(0).rename("elevation"))
            .addBands(wc.unmask(0).rename("worldcover"))
            .addBands(ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
                      .select("occurrence").unmask(0).rename("water_occurrence")))


def embedding_image(ee, year_a: int, year_b: int):
    """AlphaEarth at both endpoints, band-renamed to match the local frame."""
    coll = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")

    def one(year, suffix):
        img = coll.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
        return img.rename([f"A{i:02d}_{suffix}" for i in range(64)])

    return one(year_a, "2018").addBands(one(year_b, "2024"))


# ---------------------------------------------------------------------------
def label_state_space() -> np.ndarray:
    """The 6,414 labelled plots as unit vectors in concat(2018, 2024) space.

    Novelty is referenced to the LABEL set, not to the candidate pool. The
    distinction is the whole point: a place that is ordinary globally but unlike
    every plot anyone has labelled is exactly the one worth an afternoon, and a
    pool-referenced score ranks it as ordinary.
    """
    from twotower_lab import load_context
    frame = load_context().view("full").frame
    cols = ([f"A{i:02d}_2018" for i in range(64)]
            + [f"A{i:02d}_2024" for i in range(64)])
    x = np.asarray(frame[cols], dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


#: The three stable transitions. A confusion *among* these is the error family
#: this channel exists to fix -- and the one `change_f1` cannot see.
STABLE = ("Nature -> Nature", "Cropland -> Cropland", "Artificial -> Artificial")

#: Below this many stable plots a stratum's rate is noise; use the global one.
MIN_PLOTS_FOR_RATE = 30


def error_rate_by_stratum() -> dict:
    """Rate of **stable-class confusion** per kind of ground, from the OOF cache.

    Deficit alone is the wrong allocation and the failure is concrete: `tree` is
    34.6% of land against 24.4% of labels, the second-largest absolute deficit
    there is, and a deficit-proportional split spends a third of the batch on
    forest -- where the model is already right. Buying labels where the label set
    is thin AND the model is wrong is what this channel is for, so the weight is
    the product.

    The rate is deliberately **not** the overall error rate. Overall error on
    `grass` and `crop` is dominated by the Cropland/Nature boundary, which the
    ledger says is a **label-noise ceiling** -- buying more of it buys more
    argument, not more signal, which is the failure BALD exists to avoid. What is
    counted here is: truth is a stable class, prediction is a *different* stable
    class. That is exactly the family the user sees on the map (mountains read as
    built-up, wetlands as fields) and exactly the family no aggregate can see.
    """
    path = RESULTS / "oof_terrain.parquet"
    if not path.exists():
        print("  (no oof_terrain.parquet -- allocating on deficit alone)")
        return {}
    d = pd.read_parquet(path)
    d["wc"] = d["worldcover"].map(WORLDCOVER)
    d = d[d["truth"].isin(STABLE)]
    confused = d["pred"].isin(STABLE) & (d["pred"] != d["truth"])
    rate = confused.groupby(d["wc"]).mean()
    n = d["wc"].value_counts()
    overall = float(confused.mean())
    return {k: (float(v) if n.get(k, 0) >= MIN_PLOTS_FOR_RATE else overall)
            for k, v in rate.items()}


def allocate(strata: pd.Series, land: dict, labels: dict, n_select: int,
             errors: dict | None = None) -> dict:
    """How many points each stratum gets: ``deficit x error rate``.

    ``deficit = land_share - label_share`` in absolute percentage points, not the
    ratio. The ratio is the right thing to *notice* a gap with (snow/ice reads
    125x under-sampled) and the wrong thing to spend a budget on, because a
    stratum can be enormously under-sampled and still be a rounding error of the
    world. Bare ground is 17.4% of land against 6.6% of labels: a 10.8-point
    deficit, and the largest single one there is.

    The error rate is the second factor and it is what stops the batch going to
    forest -- see `error_rate_by_stratum`. Passing ``errors=None`` recovers the
    deficit-only allocation, which is the control if anyone wants to test that
    this weighting earns its complexity.
    """
    weights = {}
    for name in strata.unique():
        if name in EXCLUDE_FROM_STRATA or name is None:
            continue
        share_land = land.get(name)
        if not share_land or share_land < MIN_LAND_SHARE:
            continue
        deficit = share_land - labels.get(name, 0.0)
        if deficit > 0:
            weights[name] = deficit * (errors or {}).get(name, 1.0)
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("no under-sampled stratum found -- check the land shares")
    # Largest-remainder, so the allocation sums to exactly n_select rather than
    # to n_select +- rounding.
    exact = {k: n_select * v / total for k, v in weights.items()}
    out = {k: int(np.floor(v)) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True):
        if sum(out.values()) >= n_select:
            break
        out[k] += 1
    return {k: v for k, v in out.items() if v > 0}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-draw", type=int, default=40000,
                    help="equal-area candidates before the stratum filter")
    ap.add_argument("--n-select", type=int, default=75)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--emb-chunk", type=int, default=500)
    ap.add_argument("--min-sep-km", type=float, default=25.0,
                    help="thin selected points closer than this (spatial "
                         "autocorrelation: two points 2 km apart are close to "
                         "one observation)")
    ap.add_argument("--pool-per-point", type=int, default=40,
                    help="stage-2 candidates kept per selected point, per "
                         "stratum -- the novelty ranking's choice set")
    ap.add_argument("--deficit-only", action="store_true",
                    help="allocate on the coverage deficit alone, without the "
                         "model error rate -- the control for that weighting")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not LAND_SHARE.exists():
        raise SystemExit(f"{LAND_SHARE} missing -- run "
                         "src/diagnose_terrain_errors.py --refresh-land first")
    land = json.loads(LAND_SHARE.read_text())["wc"]

    ee = init_gee()
    draw = equal_area_draw(args.n_draw, args.seed)
    print(f"stage 1: {len(draw):,} equal-area points, terrain + land cover",
          flush=True)
    cov = sample_points(ee, covariate_image(ee), draw, 30, args.chunk, "cov")
    cand = draw.merge(cov.drop_duplicates("id"), on="id", how="inner")
    cand["wc"] = cand["worldcover"].map(WORLDCOVER)
    cand = cand[cand["wc"].notna() & ~cand["wc"].isin(EXCLUDE_FROM_STRATA)]
    print(f"  {len(cand):,} on land")

    # The label set's own stratum shares, recomputed here rather than pasted, so
    # the deficit moves as the campaign fills the gap.
    terrain = pd.read_parquet(RESULTS / "terrain_plots.parquet")
    labels = (terrain["worldcover"].map(WORLDCOVER)
              .value_counts(normalize=True).to_dict())

    errors = {} if args.deficit_only else error_rate_by_stratum()
    quota = allocate(cand["wc"], land, labels, args.n_select, errors)
    print("\nallocation (weight = deficit x model error rate):")
    for k, v in sorted(quota.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v:>3}   land {land[k]:.3f}  labels "
              f"{labels.get(k, 0):.3f}  deficit {land[k] - labels.get(k, 0):+.3f}"
              f"  err {errors.get(k, float('nan')):.3f}")

    # Cap the stage-2 pool per stratum. Sampling 128 bands is most of the
    # runtime and the ranking throws nearly all of it away; keeping ~40x the
    # quota leaves the novelty ordering plenty to choose between.
    rng = np.random.default_rng(args.seed)
    pool = pd.concat([
        g.iloc[rng.permutation(len(g))[:max(args.pool_per_point * quota[name],
                                            200)]]
        for name, g in cand[cand["wc"].isin(quota)].groupby("wc")
    ]).reset_index(drop=True)
    print(f"\nstage 2: AlphaEarth for {len(pool):,} in-stratum candidates "
          f"(capped from {int(cand['wc'].isin(quota).sum()):,})", flush=True)
    emb = sample_points(ee, embedding_image(ee, 2018, 2024), pool, 10,
                        args.emb_chunk, "emb")
    pool = pool.merge(emb.drop_duplicates("id"), on="id", how="inner")

    cols = ([f"A{i:02d}_2018" for i in range(64)]
            + [f"A{i:02d}_2024" for i in range(64)])
    pool = pool.dropna(subset=cols)
    x = np.array(pool[cols], dtype=np.float64)      # copy: asarray can alias
    x /= np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    ref = label_state_space()
    nn = np.empty(len(x))
    for i in range(0, len(x), 2048):
        nn[i:i + 2048] = (x[i:i + 2048] @ ref.T).max(axis=1)
    pool["novelty"] = 1.0 - nn
    print(f"  {len(pool):,} with embeddings | novelty "
          f"{pool['novelty'].min():.3f}..{pool['novelty'].max():.3f}")

    # Take the most novel within each stratum, thinning as we go. Thinning has
    # to happen during selection, not after: dropping a near-duplicate
    # afterwards leaves the batch short of its quota.
    chosen, kept_lonlat = [], []
    for stratum, k in quota.items():
        sub = (pool[pool["wc"] == stratum]
               .sort_values("novelty", ascending=False))
        for row in sub.itertuples():
            if len(chosen) >= sum(quota.values()):
                break
            if _too_close(row.lon, row.lat, kept_lonlat, args.min_sep_km):
                continue
            chosen.append(row.Index)
            kept_lonlat.append((row.lon, row.lat))
            if sum(1 for i in chosen if pool.at[i, "wc"] == stratum) >= k:
                break

    out = pool.loc[chosen].copy()
    out = out.sort_values("novelty", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out["score"] = out["novelty"]
    out["channel"] = "coverage"
    out["cell_km"] = 5.0
    keep = ["id", "lon", "lat", "channel", "rank", "score", "cell_km",
            "novelty", "wc", "slope", "elevation", "water_occurrence"]
    out[keep].to_csv(args.out, index=False)

    # The one quality check that needs no labels: the effective number of
    # distinct places the batch actually selected. A surface that has silently
    # collapsed onto one kind of terrain scores near 1 however good its ranking
    # looked (ACTIVE_LEARNING.md, evaluation section).
    import acquisition as acq
    v_sel = acq.vendi_score(np.asarray(out[cols], dtype=np.float64))
    rng = np.random.default_rng(0)
    v_rand = np.mean([acq.vendi_score(
        np.asarray(pool[cols].iloc[rng.choice(len(pool), len(out), False)],
                   dtype=np.float64)) for _ in range(5)])
    print(f"\n-> {args.out}  ({len(out)} points)")
    print(out["wc"].value_counts().to_string())
    print(f"\nVendi (effective distinct places): selected {v_sel:.1f} | "
          f"same-size random from the same pool {v_rand:.1f}")


def _too_close(lon: float, lat: float, kept: list, min_km: float) -> bool:
    if not kept or min_km <= 0:
        return False
    lon0, lat0 = np.array(kept).T
    # Equirectangular is plenty at a 25 km rejection radius and avoids a
    # haversine over a growing list on every candidate.
    dx = (lon - lon0) * np.cos(np.radians(lat)) * 111.32
    dy = (lat - lat0) * 110.57
    return bool(np.min(dx * dx + dy * dy) < min_km * min_km)


if __name__ == "__main__":
    main()
