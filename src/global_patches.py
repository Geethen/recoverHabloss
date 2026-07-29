"""Equal-area random global sample of 5x5 km patches, and the patch geometry.

Why patches and not points
--------------------------
The label frame is 6,492 *plots* and the learning curves say the bottleneck is
label quantity, not architecture (+0.026 change-F1 per doubling, and two coarse3
classes sitting at F1 0.000 on 46 and 114 plots). Labelling more plots means
looking at imagery, and looking at imagery happens over a *scene*, not over a
scattered point. A 5x5 km patch at 10 m is 500x500 px -- one screenful in QGIS,
big enough to hold several independent labels and small enough to fetch and
infer 100 of them.

So the unit of this design is the patch: sample patches, map them, then draw
label points inside the patches that are worth a human's time.

The sampling frame
------------------
Uniform over **land area**, which is two things:

* Equal area on the sphere. Uniform in longitude and in ``sin(latitude)``, not
  in latitude -- sampling latitude uniformly puts ~2x too much weight on the
  poles, which for this product means tundra and ice instead of the agricultural
  and peri-urban frontiers where the transitions live.
* Land is defined as **AlphaEarth coverage in both 2018 and 2024**. This is the
  operational definition rather than a coastline polygon, and deliberately so: a
  patch the embedding index cannot serve for both endpoints cannot be mapped, so
  including it in the frame would silently bias the realised sample away from
  the drawn one. The AEF index is ~33k tiles/year over ~149M km2, i.e. land.

Antarctica (``lat < -60``) is cut. It is ~9% of equal-area land draws and none
of it carries a Nature/Cropland/Artificial transition; keeping it would spend a
tenth of the sample on ice to no purpose. This is the one substantive departure
from "uniform over land" and it is recorded in the manifest so a later run can
undo it (``--min-lat -90``).

Rejection, not stratification
-----------------------------
The draw is unstratified on purpose. The whole point of the first 100 patches is
to *measure* how often each class occurs on the ground so the later, larger draw
can be sized. Stratifying now on anything derived from the map would put the
answer into the question. Later rounds rank an oversample by novelty/entropy
(`plan_patch_sampling.py`), which is a different and explicitly non-random step.

Reproducibility
---------------
Every patch carries the seed, the draw index and its rejection-loop position, so
``--n 400`` with the same seed re-draws the same first 100 patches followed by
300 new ones. The oversample for round two is therefore a superset of this
sample, not a fresh draw that would have to be re-mapped from scratch.

Run
---
    python src/global_patches.py --n 100 --seed 0
    python src/global_patches.py --n 400 --seed 0     # superset, for round two
"""
from __future__ import annotations

import argparse
import asyncio
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir

YEARS = (2018, 2024)
#: Patch edge in metres. 500x500 px at 10 m.
PATCH_M = 5_000.0
#: Target grid resolution, matching every other map in this project.
RESOLUTION = 10.0
#: Equal-area draws below this latitude are discarded (Antarctica).
MIN_LAT = -60.0
#: The AEF index tops out near +/-83.4 deg.
MAX_LAT = 83.0


def _land_footprint(index, years=YEARS):
    """Union-free land test: the tile frames for every year, spatially indexed.

    Returns a list of GeoDataFrames, one per year. A point is "land" only if it
    falls inside a tile in *all* of them -- an endpoint pair with a hole at one
    end is not a mappable patch.
    """
    import geopandas as gpd

    gdf = index._gdf
    if gdf is None:
        index.load()
        gdf = index._gdf
    out = []
    for year in years:
        sub = gdf[gdf["year"] == year]
        if sub.empty:
            raise SystemExit(f"AEF index has no tiles for {year}")
        # The index carries a *column* called "crs" (the tile's UTM string),
        # which collides with the GeoDataFrame's own `.crs`. Rename it before it
        # can be mistaken for the frame's CRS, then assert WGS84 on the geometry
        # -- the index's total bounds are lon/lat, so this is a relabel and not
        # a reprojection.
        sub = gpd.GeoDataFrame(
            sub[["fid", "crs"]].rename(columns={"crs": "tile_crs"}),
            geometry=sub.geometry.values)
        sub = sub.set_crs("EPSG:4326", allow_override=True)
        sub.sindex  # build the R-tree once
        out.append(sub)
    return out


def sample_land_points(index, n, seed=0, min_lat=MIN_LAT, max_lat=MAX_LAT,
                       batch=4096, max_rounds=200):
    """``n`` equal-area random points on AlphaEarth-covered land.

    Rejection sampling in blocks: draw a block uniformly in ``(lon, sin lat)``,
    keep the points that land inside a tile in every year, repeat. Points are
    kept in draw order, so a larger ``n`` extends the sample rather than
    replacing it.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    frames = _land_footprint(index)
    rng = np.random.default_rng(seed)
    s_lo, s_hi = np.sin(np.radians([min_lat, max_lat]))

    lons: list[float] = []
    lats: list[float] = []
    epsgs: list[int] = []
    n_drawn = 0
    n_land = 0
    for _ in range(max_rounds):
        if len(lons) >= n:
            break
        lon = rng.uniform(-180.0, 180.0, batch)
        lat = np.degrees(np.arcsin(rng.uniform(s_lo, s_hi, batch)))
        n_drawn += batch
        pts = gpd.GeoDataFrame(
            {"i": np.arange(batch)},
            geometry=[Point(x, y) for x, y in zip(lon, lat)], crs="EPSG:4326")
        keep = None
        crs_col = None
        for frame in frames:
            hit = gpd.sjoin(pts, frame, how="inner", predicate="within")
            # A point on a tile seam matches two tiles; keep the first.
            hit = hit[~hit.index.duplicated(keep="first")]
            idx = set(hit["i"].tolist())
            keep = idx if keep is None else (keep & idx)
            if crs_col is None:
                crs_col = hit.set_index("i")["tile_crs"]
        n_land += len(keep)
        for i in sorted(keep):
            if len(lons) >= n:
                break
            lons.append(float(lon[i]))
            lats.append(float(lat[i]))
            epsgs.append(_epsg_of(crs_col.loc[i]))
    if len(lons) < n:
        raise SystemExit(f"only {len(lons)} land points after {n_drawn} draws")
    # The land fraction is a check on the *frame*, not a cost: an equal-area
    # draw over the globe minus Antarctica should hit AEF-covered land about a
    # quarter of the time. A number far from that says the footprint test is
    # wrong, and it would be wrong silently.
    return (np.array(lons), np.array(lats), np.array(epsgs, dtype=int),
            n_drawn, n_land)


def _epsg_of(crs_value) -> int:
    """The tile's native UTM EPSG code, however the index spells it."""
    from rasterio.crs import CRS as RioCRS

    if isinstance(crs_value, (int, np.integer)):
        return int(crs_value)
    text = str(crs_value)
    if text.upper().startswith("EPSG:"):
        return int(text.split(":")[1])
    return int(RioCRS.from_string(text).to_epsg())


def patch_geometry(lon, lat, epsg, patch_m=PATCH_M, resolution=RESOLUTION):
    """The patch's UTM box, its WGS84 query bbox, and the target geobox.

    The box is built in the tile's own UTM zone and centred on a point snapped
    to the ``resolution`` lattice, so **every patch is exactly the same number
    of pixels and exactly the same ground area** regardless of latitude. Doing
    this the easy way -- a lon/lat box with a ``cos(lat)`` correction -- gives
    patches whose pixel counts drift with latitude, and pixel counts are the
    thing the whole sizing calculation is denominated in.

    The WGS84 bbox returned for the tile *query* is the reprojection of that
    box, densified and then padded by one tile-seam's worth, because the query
    only has to find the tiles that cover the patch and being generous there is
    free.
    """
    from rasterio.warp import transform as warp_transform
    from rasterio.warp import transform_bounds
    from aef_loader import aoi_geobox

    utm = f"EPSG:{epsg}"
    xs, ys = warp_transform("EPSG:4326", utm, [float(lon)], [float(lat)])
    # Snap the centre so the box edges land on the global 10 m lattice; the
    # 2018 and 2024 reads then share pixel coordinates exactly, as in run_aoi.
    cx = np.round(xs[0] / resolution) * resolution
    cy = np.round(ys[0] / resolution) * resolution
    half = patch_m / 2.0
    utm_bbox = (cx - half, cy - half, cx + half, cy + half)
    geobox = aoi_geobox(utm_bbox, crs=utm, resolution=resolution,
                        bbox_crs=utm)
    wgs = transform_bounds(utm, "EPSG:4326", *utm_bbox, densify_pts=21)
    pad = 0.02
    wgs_bbox = (wgs[0] - pad, wgs[1] - pad, wgs[2] + pad, wgs[3] + pad)
    return utm_bbox, wgs_bbox, geobox


async def draw(n, seed, min_lat, max_lat, out_path: Path):
    from aef_loader import AEFIndex, DataSource

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()

    lon, lat, epsg, n_drawn, n_land = sample_land_points(
        index, n, seed=seed, min_lat=min_lat, max_lat=max_lat)
    print(f"{n} patches kept; {n_land}/{n_drawn} equal-area draws fell on "
          f"AEF-covered land ({n_land / n_drawn:.1%})", flush=True)

    rows = []
    for i, (x, y, e) in enumerate(zip(lon, lat, epsg)):
        utm_bbox, wgs_bbox, geobox = patch_geometry(x, y, e)
        rows.append({
            "patch_id": f"p{i:04d}",
            "draw_index": i,
            "lon": float(x), "lat": float(y), "epsg": int(e),
            "utm_minx": utm_bbox[0], "utm_miny": utm_bbox[1],
            "utm_maxx": utm_bbox[2], "utm_maxy": utm_bbox[3],
            "west": wgs_bbox[0], "south": wgs_bbox[1],
            "east": wgs_bbox[2], "north": wgs_bbox[3],
            "height": int(geobox.shape.y), "width": int(geobox.shape.x),
        })
    frame = pd.DataFrame(rows)
    frame["area_km2"] = frame.height * frame.width * (RESOLUTION ** 2) / 1e6

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n": int(n), "seed": int(seed),
        "min_lat": min_lat, "max_lat": max_lat,
        "patch_m": PATCH_M, "resolution": RESOLUTION,
        "years": list(YEARS),
        "n_equal_area_draws": int(n_drawn),
        "n_on_land": int(n_land),
        "land_fraction": float(n_land / n_drawn),
        "frame": "uniform in (lon, sin lat) over AEF tile coverage in both "
                 "2018 and 2024, Antarctica excluded",
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    px = frame.height * frame.width
    print(f"patch size {px.min():,}-{px.max():,} px "
          f"({frame.area_km2.min():.2f}-{frame.area_km2.max():.2f} km2)")
    print(f"lat {frame.lat.min():.1f}..{frame.lat.max():.1f}, "
          f"{frame.epsg.nunique()} UTM zones")
    print(f"-> {out_path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-lat", type=float, default=MIN_LAT)
    parser.add_argument("--max-lat", type=float, default=MAX_LAT)
    parser.add_argument("--out", type=Path,
                        default=project_data_dir("patches", "patches.parquet"))
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    asyncio.run(draw(args.n, args.seed, args.min_lat, args.max_lat, args.out))


if __name__ == "__main__":
    main()
