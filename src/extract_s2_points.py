"""Sentinel-2 VNIR 10 m patch extraction at plot points, off the AWS open-data COGs.

Why this exists
---------------
Tessera gave the AlphaEarth model real spatial detail (+1.8pt change-F1 where it
fires) but only **35.8% of plots have both 2018 and 2024 tiles**, and the whole
two-tower search concluded the bottleneck was that coverage, not the
architecture. Raw Sentinel-2 has no such hole: L2A COGs cover the globe for both
endpoints, they are 10 m in the VNIR, and they are fast and free to pull from
``s3://sentinel-cogs`` (Element 84 / AWS Open Data registry). This module is the
detail modality's data path -- the S2 analogue of ``extract_tessera_points.py``.

How it reads
------------
STAC search (earth-search v1) per point-year, then **windowed reads straight out
of the COGs**. Element 84's COGs are tiled 1024x1024 DEFLATE, so GDAL fetches one
internal block per read regardless of how small the window is: a 64x64 patch
costs exactly what a 5x5 patch costs. We therefore take 64x64 (640 m) and store
it, so every downstream texture / neighbourhood / segmentation idea is a local
computation rather than a re-extraction. (D3 in TWOTOWER_RESEARCH.md died on
"needs re-extraction"; this is the fix.)

Compositing
-----------
Up to ``--scenes-per-year`` scenes per year, chosen as the least-cloudy scene in
each season so the composite carries phenology rather than one date's weather,
and so 2018 and 2024 are sampled from comparable parts of the year -- a
season-mismatched pair manufactures "change" that is really leaf-on/leaf-off.
Per-pixel SCL masking drops cloud, shadow and cirrus; the composite is the
per-pixel median over surviving scenes. Pixels with no clear observation are
NaN, and ``n_valid`` records the depth behind every pixel so a downstream model
can tell a confident pixel from a lucky one.

Output
------
``s2_shards/shard_XXXXXX.npz``  patches (n, 2, 4, 64, 64) uint16 + counts
``s2_shards/shard_XXXXXX.parquet``  PLOTID/lon/lat/coverage bookkeeping
Both are written per shard and resumed, so an interrupted run continues.

Typical use::

    python extract_s2_points.py --limit 40          # smoke test
    python extract_s2_points.py --workers 48        # full run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# GDAL/rasterio need these before the first open; anonymous access to a public
# bucket, and no directory listing (the COGs are addressed by exact href).
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_REGION", "us-west-2")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")

import rasterio  # noqa: E402
import rasterio.warp  # noqa: E402
from rasterio.windows import Window  # noqa: E402

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
YEARS = (2018, 2024)
# The VNIR quartet: the only S2 bands native at 10 m, and the reason this is
# cheap to deploy -- four bands, no pan-sharpening, no 20 m resampling.
BANDS = ("blue", "green", "red", "nir")
PATCH = 64          # 10 m pixels -> 640 m across, one COG block's worth for free
SCL_BAND = "scl"    # 20 m scene classification, used only as a mask
# SCL classes kept as clear ground: 4 vegetation, 5 not-vegetated, 6 water,
# 7 unclassified, 11 snow. Dropped: 0 nodata, 1 saturated, 2 dark, 3 shadow,
# 8 cloud-medium, 9 cloud-high, 10 cirrus.
SCL_CLEAR = (4, 5, 6, 7, 11)
# Season edges as (month_start, month_end) inclusive. One scene is taken per
# season so the two endpoint years are sampled from comparable phenology.
SEASONS = ((1, 3), (4, 6), (7, 9), (10, 12))
CLOUD_STEPS = (20, 40, 70, 100)  # progressively relaxed until scenes are found

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = REPO_ROOT / "data" / "embeddings" / "embeddings_habloss_recover.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "embeddings" / "s2_shards"


# ---------------------------------------------------------------------------
# scene selection
# ---------------------------------------------------------------------------
def _client():
    from pystac_client import Client

    return Client.open(STAC_URL)


def _stac_cache_path(cache_dir: Path, lon: float, lat: float, year: int) -> Path:
    """Cache file for a point-year's scene list, keyed on a ~1 km rounded cell.

    Two plots 300 m apart see the same granules, so the key is rounded rather
    than exact -- the same reasoning as ``AEFIndex`` querying one downloaded
    index locally instead of hitting the API per lookup.
    """
    key = f"{round(lon, 2)}_{round(lat, 2)}_{year}"
    return cache_dir / f"stac_{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def search_scenes(client, lon: float, lat: float, year: int, max_items: int = 120,
                  cache_dir: Path | None = None):
    """Least-cloudy STAC items for one point-year, relaxing the cloud filter.

    Tropical and high-latitude plots routinely have no <20% scene in a season,
    so the threshold is stepped up rather than returning nothing. The step that
    succeeded is not recorded per scene because ``eo:cloud_cover`` rides on the
    item itself and is what the seasonal pick actually sorts on.
    """
    from pystac import Item

    cached = _stac_cache_path(cache_dir, lon, lat, year) if cache_dir else None
    if cached is not None and cached.exists():
        try:
            return [Item.from_dict(d) for d in json.loads(cached.read_text())]
        except Exception:
            pass  # a bad cache entry is a miss, never a failure

    point = {"type": "Point", "coordinates": [float(lon), float(lat)]}
    items = []
    for limit in CLOUD_STEPS:
        try:
            items = list(client.search(
                collections=[COLLECTION], intersects=point,
                datetime=f"{year}-01-01/{year}-12-31",
                query={"eo:cloud_cover": {"lt": limit}},
                max_items=max_items).items())
        except Exception:
            return []
        if items:
            break

    if cached is not None and items:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(".tmp")
            tmp.write_text(json.dumps([i.to_dict() for i in items]))
            tmp.replace(cached)
        except OSError:
            pass
    return items


def pick_seasonal(items, per_year: int) -> list:
    """Least-cloudy scene per season, then fill by cloud cover up to ``per_year``.

    Taking the ``per_year`` globally-least-cloudy scenes would happily return
    four July scenes; a change model then reads 2018-summer against 2024-summer
    for one plot and 2018-winter against 2024-spring for the next. Seasonal
    stratification makes the composite mean the same thing everywhere.
    """
    if not items:
        return []
    cloud = lambda it: it.properties.get("eo:cloud_cover", 100.0)  # noqa: E731
    chosen, used = [], set()
    for lo, hi in SEASONS:
        season = [it for it in items
                  if lo <= int(it.datetime.month) <= hi and it.id not in used]
        if season:
            best = min(season, key=cloud)
            chosen.append(best)
            used.add(best.id)
        if len(chosen) >= per_year:
            break
    if len(chosen) < per_year:
        rest = sorted((it for it in items if it.id not in used), key=cloud)
        chosen.extend(rest[: per_year - len(chosen)])
    return chosen[:per_year]


# ---------------------------------------------------------------------------
# windowed reads
# ---------------------------------------------------------------------------
_TRANSFORMERS: dict[int, object] = {}


def _to_crs(epsg: int, lon: float, lat: float):
    """Project a lon/lat to a granule's CRS, reusing the pyproj transformer.

    Building a Transformer is expensive enough to matter at ~500k calls, and it
    is the same object for every plot in a UTM zone.
    """
    from pyproj import Transformer

    transformer = _TRANSFORMERS.get(epsg)
    if transformer is None:
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}",
                                           always_xy=True)
        _TRANSFORMERS[epsg] = transformer
    return transformer.transform(lon, lat)


def _read_window_cached(href: str, lon: float, lat: float, size: int,
                        cache_dir: Path):
    """The fast path: cached COG header + a range GET. None means 'use rasterio'."""
    import s2_cog

    try:
        _, _, _, _, _, epsg = s2_cog.metadata(href, cache_dir)
        x, y = _to_crs(epsg, lon, lat)
        return s2_cog.read_window(href, x, y, size, cache_dir)
    except s2_cog.CogFallback:
        return None
    except Exception:
        return None


def _read_window(href: str, lon: float, lat: float, size: int,
                 cache_dir: Path | None = None):
    """A ``size x size`` window centred on (lon, lat), or None if unreadable.

    Tries the cached-header reader first (see ``s2_cog``); falls back to rasterio
    on anything it will not vouch for. Reads whatever the COG can give and pads
    to the requested shape, so a plot near a granule edge yields a partial patch
    instead of being dropped.
    """
    if cache_dir is not None:
        fast = _read_window_cached(href, lon, lat, size, cache_dir)
        if fast is not None:
            return fast
    try:
        with rasterio.open(href) as src:
            xs, ys = rasterio.warp.transform("EPSG:4326", src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            row, col = int(row), int(col)
            half = size // 2
            window = Window(col - half, row - half, size, size)
            arr = src.read(1, window=window, boundless=True, fill_value=0)
    except Exception:
        return None
    if arr.shape != (size, size):
        out = np.zeros((size, size), arr.dtype)
        out[: arr.shape[0], : arr.shape[1]] = arr
        arr = out
    return arr


def read_scene(item, lon: float, lat: float, cache_dir: Path | None = None):
    """One scene's VNIR patch (4, PATCH, PATCH) float32 with cloud pixels NaN.

    SCL is 20 m, so its patch is half the width and is repeated 2x2 back onto
    the 10 m grid -- the mask is coarser than the imagery, which is a property
    of the product, not a shortcut.
    """
    scl = _read_window(item.assets[SCL_BAND].href, lon, lat, PATCH // 2, cache_dir)
    if scl is None:
        return None
    clear = np.isin(scl, SCL_CLEAR)
    clear = np.repeat(np.repeat(clear, 2, axis=0), 2, axis=1)[:PATCH, :PATCH]
    if not clear.any():
        return None

    out = np.full((len(BANDS), PATCH, PATCH), np.nan, np.float32)
    for i, band in enumerate(BANDS):
        arr = _read_window(item.assets[band].href, lon, lat, PATCH, cache_dir)
        if arr is None:
            return None
        values = arr.astype(np.float32)
        values[(~clear) | (arr == 0)] = np.nan  # 0 is L2A nodata
        out[i] = values
    return out


def composite_point(client, lon: float, lat: float, year: int, per_year: int,
                    cache_dir: Path | None = None):
    """Per-pixel median VNIR composite for one plot-year, plus its depth map."""
    items = pick_seasonal(
        search_scenes(client, lon, lat, year, cache_dir=cache_dir), per_year)
    if not items:
        return (np.full((len(BANDS), PATCH, PATCH), np.nan, np.float32),
                np.zeros((PATCH, PATCH), np.uint8), 0)
    stack = [s for s in (read_scene(it, lon, lat, cache_dir) for it in items)
             if s is not None]
    if not stack:
        return (np.full((len(BANDS), PATCH, PATCH), np.nan, np.float32),
                np.zeros((PATCH, PATCH), np.uint8), 0)
    cube = np.stack(stack)                       # (s, b, h, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # all-NaN pixels are expected
        median = np.nanmedian(cube, axis=0).astype(np.float32)
    n_valid = np.isfinite(cube[:, 0]).sum(0).astype(np.uint8)
    return median, n_valid, len(stack)


# ---------------------------------------------------------------------------
# shard driver
# ---------------------------------------------------------------------------
def run_shard(frame: pd.DataFrame, years, per_year: int, workers: int,
              cache_dir: Path | None = None):
    """Extract one shard of plots; threads because GDAL releases the GIL on I/O."""
    client = _client()
    n = len(frame)
    patches = np.full((n, len(years), len(BANDS), PATCH, PATCH), np.nan, np.float32)
    depth = np.zeros((n, len(years), PATCH, PATCH), np.uint8)
    scenes = np.zeros((n, len(years)), np.int16)

    def one(job):
        i, yi, lon, lat, year = job
        median, n_valid, used = composite_point(client, lon, lat, year, per_year,
                                                cache_dir)
        patches[i, yi] = median
        depth[i, yi] = n_valid
        scenes[i, yi] = used

    jobs = [(i, yi, float(r.lon), float(r.lat), y)
            for i, r in enumerate(frame.itertuples(index=False))
            for yi, y in enumerate(years)]
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES",
                      GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                      GDAL_HTTP_MULTIPLEX="YES", GDAL_NUM_THREADS="1"):
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, jobs))

    centre = PATCH // 2
    index = pd.DataFrame({
        "PLOTID": frame["PLOTID"].to_numpy(),
        "lon": frame["lon"].to_numpy(),
        "lat": frame["lat"].to_numpy(),
    })
    for yi, year in enumerate(years):
        index[f"s2_scenes_{year}"] = scenes[:, yi]
        index[f"s2_depth_{year}"] = depth[:, yi, centre, centre]
        index[f"s2_present_{year}"] = np.isfinite(
            patches[:, yi, :, centre, centre]).all(1)
        index[f"s2_patch_frac_{year}"] = np.isfinite(patches[:, yi, 0]).mean((1, 2))
    return patches, depth, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--years", type=int, nargs="+", default=list(YEARS))
    parser.add_argument("--scenes-per-year", type=int, default=4)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="COG-header and STAC cache (default: <out-dir>/cache). "
                             "Persistent and safe to keep: entries are header "
                             "bytes and scene listings for immutable objects.")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the cached-header fast path (rasterio only)")
    args = parser.parse_args()

    frame = pd.read_parquet(args.points, columns=["PLOTID", "lon", "lat"])
    frame = frame.drop_duplicates("PLOTID").reset_index(drop=True)
    if args.limit:
        frame = frame.iloc[: args.limit].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    years = tuple(args.years)
    cache_dir = None if args.no_cache else (args.cache_dir or args.out_dir / "cache")
    print(f"{len(frame):,} plots x years {years} | {args.scenes_per_year} scenes/yr "
          f"| {args.workers} workers | patch {PATCH}x{PATCH} "
          f"| cache {'off' if cache_dir is None else cache_dir}", flush=True)

    started = time.time()
    shard_paths = []
    for start in range(0, len(frame), args.shard_size):
        chunk = frame.iloc[start:start + args.shard_size]
        npz_path = args.out_dir / f"shard_{start:06d}.npz"
        idx_path = args.out_dir / f"shard_{start:06d}.parquet"
        shard_paths.append((npz_path, idx_path))
        if npz_path.exists() and idx_path.exists():
            print(f"  shard {start:06d} cached", flush=True)
            continue
        t0 = time.time()
        patches, depth, index = run_shard(chunk, years, args.scenes_per_year,
                                          args.workers, cache_dir)
        # float32 NaN patches compress poorly; store reflectance as uint16 with a
        # separate finite mask, which is how the product ships anyway.
        finite = np.isfinite(patches)
        stored = np.where(finite, patches, 0).astype(np.uint16)
        np.savez_compressed(npz_path, patches=stored, finite=finite, depth=depth,
                            plotid=index["PLOTID"].to_numpy().astype("U"),
                            years=np.array(years))
        index.to_parquet(idx_path, index=False)
        cov = {y: float(index[f"s2_present_{y}"].mean()) for y in years}
        print(f"  shard {start:06d} ({len(chunk)}) {time.time() - t0:.1f}s cov={cov}",
              flush=True)

    index = pd.concat([pd.read_parquet(p) for _, p in shard_paths], ignore_index=True)
    final = args.out_dir.parent / "s2_index_habloss_recover.parquet"
    index.to_parquet(final, index=False)
    print(f"\nDONE {len(index):,} plots in {(time.time() - started) / 60:.1f} min "
          f"-> {final}")
    for year in years:
        both = index[f"s2_present_{year}"].mean()
        print(f"  {year} centre-pixel coverage: {both:.1%} "
              f"| mean scenes {index[f's2_scenes_{year}'].mean():.1f}")
    both = np.logical_and.reduce([index[f"s2_present_{y}"] for y in years])
    print(f"  both years: {both.mean():.1%}  (Tessera's comparable figure: 35.8%)")


if __name__ == "__main__":
    main()
