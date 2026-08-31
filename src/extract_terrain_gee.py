"""Terrain and wetness covariates for the 6,414 labelled plots, from Earth Engine.

Why this exists
---------------
The user reports two errors on the deployed map that no aggregate in the ledger
can see, because both sides of each are *stable* classes and change-F1 is
blind to them by construction:

    mountains  read as  Artificial -> Artificial
    wetlands   read as  Cropland  -> Cropland

Both are plausible from the physics -- bare rock and scree are bright and
low-NDVI, which is what built-up looks like to a 10 m embedding; seasonally
flooded herbaceous vegetation is geometric and green-then-bare, which is what
cropland looks like. Neither is testable without a terrain covariate, and the
modelling frame has none. This script joins the four cheapest ones.

**These are diagnostics, not model inputs.** Nothing here goes into
`HierarchicalSoftmaxNN`; adding terrain as a feature is a separate proposal with
its own control, and it is not what this file is for. The question here is
narrower: *are the errors the user sees concentrated on high-slope and wet
ground, and if so is that terrain under-represented in the label set* -- which
is an acquisition question, answerable once, offline.

Sources
-------
====================  ================================================
`elevation`, `slope`  NASA SRTM v3 (`USGS/SRTMGL1_003`), 30 m, +-60 deg
`elev_glo30`          Copernicus GLO-30, the fill above 60 deg N
`water_occurrence`    JRC Global Surface Water v1.4 -- % of months 1984-2021
                      with surface water. The wetland axis that WorldCover's
                      hard class misses: a plot at 35% occurrence is seasonally
                      flooded and carries no wetland *label* anywhere.
`worldcover`          ESA WorldCover v200 (2021), 10 m. Class 90 is herbaceous
                      wetland, 80 permanent water, 95 mangrove.
====================  ================================================

SRTM stops at 60 deg N and this frame has Nordic plots above it, so elevation is
taken from SRTM where it exists and Copernicus GLO-30 above -- reported as two
columns plus a merged one, so a later reader can see which source a row used
rather than having to trust a silent coalesce.

Run
---
    /home/geethen.singh/.pixi/envs/geo/bin/python src/extract_terrain_gee.py

Writes ``data/analysis_results/terrain_plots.parquet`` -- one row per plot,
keyed on ``PLOTID``.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from project_paths import project_data_dir

PROJECT = "ee-gsingh"
#: EE caps a getInfo payload; 1,000 points per request is comfortably inside it
#: and the whole frame is seven requests.
CHUNK = 1000
OUT = project_data_dir("analysis_results") / "terrain_plots.parquet"


def init_gee(project: str = PROJECT) -> None:
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=project)


def covariate_image() -> ee.Image:
    """One multi-band image; every band is sampled in a single request."""
    # SRTM is masked over water and stops at 60 deg N; both would drop rows.
    srtm = ee.Image("USGS/SRTMGL1_003")
    glo30 = (ee.ImageCollection("COPERNICUS/DEM/GLO30")
             .select("DEM").mosaic()
             # mosaic() drops the default projection and `ee.Terrain` needs one
             # (the same trap the Artificial -> Cropland script hit).
             .setDefaultProjection(ee.Projection("EPSG:4326").atScale(30)))
    terrain = ee.Terrain.products(srtm).unmask(0)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    wc = ee.ImageCollection("ESA/WorldCover/v200").first()

    return (terrain.select(["elevation", "slope", "aspect"])
            .rename(["elev_srtm", "slope_srtm", "aspect_srtm"])
            .addBands(glo30.unmask(0).rename("elev_glo30"))
            # JRC masks everything that has never held surface water, and
            # `sampleRegions` drops a point whose bands are all masked -- that
            # silently returned 95 of 6,414 rows, all of them wet, which is the
            # exact opposite of the sample this diagnostic needs. Unmasking to 0
            # is also the right semantics: never water = 0% occurrence.
            .addBands(jrc.select("occurrence").unmask(0).rename("water_occurrence"))
            .addBands(jrc.select("seasonality").unmask(0)
                      .rename("water_seasonality"))
            .addBands(wc.unmask(0).rename("worldcover")))


def sample_chunk(img: ee.Image, sub: pd.DataFrame, scale: int) -> pd.DataFrame:
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                   {"PLOTID": str(r.PLOTID)})
        for r in sub.itertuples()])
    # `worldcover` is categorical, so the reducer is a first/mode read at the
    # point rather than a mean -- `sampleRegions` at scale takes the pixel the
    # point falls in, which is the right semantics for all seven bands here.
    out = img.sampleRegions(collection=fc, scale=scale, geometries=False)
    rows = out.getInfo()["features"]
    return pd.DataFrame([f["properties"] for f in rows])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=int, default=30)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from twotower_lab import load_context
    frame = load_context().view("full").frame[["PLOTID", "lon", "lat"]].copy()
    frame["PLOTID"] = frame["PLOTID"].astype(str)
    print(f"{len(frame):,} plots", flush=True)

    init_gee()
    img = covariate_image()
    parts, t0 = [], time.time()
    for i in range(0, len(frame), args.chunk):
        sub = frame.iloc[i:i + args.chunk]
        for attempt in range(4):
            try:
                parts.append(sample_chunk(img, sub, args.scale))
                break
            except Exception as exc:                       # noqa: BLE001
                if attempt == 3:
                    raise
                print(f"  retry {attempt + 1}: {type(exc).__name__} "
                      f"{str(exc)[:120]}", flush=True)
                time.sleep(5 * (attempt + 1))
        print(f"  {min(i + args.chunk, len(frame)):5d}/{len(frame)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    out = pd.concat(parts, ignore_index=True)
    # A point off the SRTM footprint returns no `elev_srtm` key at all rather
    # than a null, so the column can be missing from an entire chunk; reindex
    # before coalescing or the Nordic rows silently vanish.
    for c in ("elev_srtm", "slope_srtm", "aspect_srtm", "elev_glo30",
              "water_occurrence", "water_seasonality", "worldcover"):
        if c not in out:
            out[c] = np.nan
    merged = frame.merge(out.drop_duplicates("PLOTID"), on="PLOTID", how="left")
    # Every band is now unmasked, so "off the SRTM footprint" reads as 0 rather
    # than as null and cannot be detected from the value. Latitude is the honest
    # test -- SRTM v3 is 60N..56S -- and the source is recorded per row so a
    # later reader can see which DEM answered instead of trusting a coalesce.
    on_srtm = merged["lat"].between(-56.0, 60.0)
    merged["elev_source"] = np.where(on_srtm, "srtm", "glo30")
    merged["elevation"] = np.where(on_srtm, merged["elev_srtm"],
                                   merged["elev_glo30"])
    # Slope only exists from SRTM here; off-footprint rows get NaN rather than a
    # zero that would read as flat ground.
    merged["slope"] = np.where(on_srtm, merged["slope_srtm"], np.nan)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.out, index=False)
    print(f"\n-> {args.out}  ({len(merged):,} rows, "
          f"{merged['elevation'].notna().sum():,} with elevation)")
    print(merged["elev_source"].value_counts().to_string())
    print(merged[["elevation", "slope", "water_occurrence"]].describe())


if __name__ == "__main__":
    main()
