"""Extract AlphaEarth satellite embeddings for the labelled RECOVER plots.

The labelled plots are distributed globally, so a single request over the whole
collection is neither reliable nor fast. Points are assigned to fixed longitude
/latitude blocks, and each block is extracted independently and written to its
own Parquet shard. Shards are resumable: an existing shard is reused, so an
interrupted run continues where it stopped.

Embeddings are pulled for the two endpoints of the RECOVER transition period
(2018 and 2024). The per-year 64-D vectors are stored alongside their
difference, which carries the change signal that the transition model needs.

Typical use::

    # inspect the block partition without touching Earth Engine
    python extract_embeddings_gee.py --dry-run

    # smoke test on a couple of blocks before the global run
    python extract_embeddings_gee.py --max-blocks 2

    # full extraction
    python extract_embeddings_gee.py
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import pandas as pd


PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
EMBEDDING_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DEFAULT_OUTPUT = Path("data/embeddings")
EMBEDDING_DIM = 64
EMBEDDING_YEARS = (2018, 2024)
# Fields carrying the interpreted transition. They live on the samples asset;
# the run aborts if they are absent rather than silently producing an
# unlabelled table, which is how the earlier predictor extraction lost them.
LABEL_COLUMNS = ["PLOTID", "stratum", "lc_2018", "lc_2024", "r"]
DEFAULT_BLOCK_SIZE = 20.0  # degrees
# Scale of the embedding product. Plot polygons are ~10 m across, so a first
# reducer at native resolution returns the single covering pixel.
EMBEDDING_SCALE = 10


def init_gee(project: str) -> None:
    try:
        ee.Initialize(
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
    except Exception:
        ee.Initialize(project=project)


def embedding_bands(year: int) -> list[str]:
    return [f"A{i:02d}_{year}" for i in range(EMBEDDING_DIM)]


def difference_bands() -> list[str]:
    return [f"A{i:02d}_diff" for i in range(EMBEDDING_DIM)]


def build_embedding_stack(years: tuple[int, ...] = EMBEDDING_YEARS) -> ee.Image:
    """One image holding each year's 64-D embedding plus their difference."""
    collection = ee.ImageCollection(EMBEDDING_ASSET)
    per_year = {}
    for year in years:
        image = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
        per_year[year] = image.rename(embedding_bands(year))

    layers = [per_year[year] for year in years]
    if len(years) == 2:
        first, last = years
        change = (
            per_year[last]
            .rename(embedding_bands(first))
            .subtract(per_year[first])
            .rename(difference_bands())
        )
        layers.append(change)
    return ee.Image.cat(layers)


def polygon_centroid(geo: object) -> tuple[float, float] | tuple[None, None]:
    """Centroid of a stored GeoJSON geometry, as (lon, lat)."""
    if isinstance(geo, str):
        try:
            geo = ast.literal_eval(geo)
        except (ValueError, SyntaxError):
            return (None, None)
    if not isinstance(geo, dict) or "coordinates" not in geo:
        return (None, None)

    def points(value):
        if value and isinstance(value[0], (int, float)):
            yield value
        else:
            for child in value:
                yield from points(child)

    coords = list(points(geo["coordinates"]))
    if not coords:
        return (None, None)
    return (
        sum(p[0] for p in coords) / len(coords),
        sum(p[1] for p in coords) / len(coords),
    )


def block_id(lon: float, lat: float, block_size: float) -> str:
    """Stable identifier for the block containing a coordinate."""
    lon_min = math.floor(lon / block_size) * block_size
    lat_min = math.floor(lat / block_size) * block_size
    return f"x{lon_min:+07.1f}_y{lat_min:+06.1f}"


def assign_blocks(frame: pd.DataFrame, block_size: float) -> pd.DataFrame:
    """Attach centroid coordinates and a block id to each labelled plot."""
    centroids = frame["geo"].map(polygon_centroid)
    frame = frame.copy()
    frame["lon"] = [c[0] for c in centroids]
    frame["lat"] = [c[1] for c in centroids]
    missing = frame["lon"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} plots have unparseable geometry in 'geo'"
        )
    frame["block_id"] = [
        block_id(lon, lat, block_size)
        for lon, lat in zip(frame["lon"], frame["lat"])
    ]
    return frame


def fetch_dataframe(collection: ee.FeatureCollection) -> pd.DataFrame:
    return ee.data.computeFeatures(
        {"expression": collection, "fileFormat": "PANDAS_DATAFRAME"}
    )


def check_label_columns(available: list[str]) -> list[str]:
    """Keep the label fields present on the asset, requiring the transition."""
    present = [c for c in LABEL_COLUMNS if c in available]
    required = {"lc_2018", "lc_2024"}
    absent = required - set(present)
    if absent:
        raise RuntimeError(
            f"Samples asset {SAMPLES_ASSET} is missing transition label field(s) "
            f"{sorted(absent)}. Available properties: {sorted(available)}. "
            "Extraction aborted: without these the model has no target."
        )
    return present


def extract_block(
    stack: ee.Image,
    plot_ids: list[str],
    keep: list[str],
    scale: int,
    retries: int = 3,
) -> pd.DataFrame:
    """Reduce the embedding stack over one block's plots."""
    plots = ee.FeatureCollection(SAMPLES_ASSET).filter(
        ee.Filter.inList("PLOTID", plot_ids)
    )

    def attach(feature):
        values = stack.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=scale,
            maxPixels=1_000_000,
        )
        return feature.set(values)

    bands = [b for year in EMBEDDING_YEARS for b in embedding_bands(year)]
    if len(EMBEDDING_YEARS) == 2:
        bands += difference_bands()
    selected = plots.map(attach).select([*keep, *bands])

    last_error = None
    for attempt in range(retries):
        try:
            return fetch_dataframe(selected)
        except Exception as error:  # transient GEE failures are common
            last_error = error
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Block extraction failed after retries") from last_error


def run_extraction(
    blocks: dict[str, list[str]],
    stack: ee.Image,
    keep: list[str],
    shard_dir: Path,
    scale: int,
    max_workers: int,
) -> pd.DataFrame:
    shard_dir.mkdir(parents=True, exist_ok=True)

    def one_block(name: str, plot_ids: list[str]) -> pd.DataFrame:
        shard_path = shard_dir / f"block_{name}.parquet"
        if shard_path.exists():
            return pd.read_parquet(shard_path)
        frame = extract_block(stack, plot_ids, keep, scale)
        frame["block_id"] = name
        frame.to_parquet(shard_path, index=False)
        return frame

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(one_block, name, ids): name
            for name, ids in blocks.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            frame = future.result()
            print(f"Block {name}: {len(frame):,} plots", flush=True)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--plots",
        type=Path,
        default=Path("data/ppi_gee/labelled_predictions.csv"),
        help="Table with 'geo' and 'PLOTID', used to build the block partition",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--block-size",
        type=float,
        default=DEFAULT_BLOCK_SIZE,
        help="Block edge length in degrees (default: 20)",
    )
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Extract only the N largest blocks; use for a smoke test",
    )
    parser.add_argument("--scale", type=int, default=EMBEDDING_SCALE)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the block partition without contacting Earth Engine",
    )
    args = parser.parse_args()

    plots = pd.read_csv(args.plots)
    if "geo" not in plots or "PLOTID" not in plots:
        raise ValueError(f"{args.plots} must contain 'geo' and 'PLOTID' columns")
    plots = assign_blocks(plots, args.block_size)

    grouped = plots.groupby("block_id")["PLOTID"].apply(list).to_dict()
    ordered = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    if args.max_blocks is not None:
        ordered = ordered[: args.max_blocks]
    blocks = dict(ordered)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "block_size_deg": args.block_size,
                    "n_plots": len(plots),
                    "n_blocks_total": len(grouped),
                    "n_blocks_selected": len(blocks),
                    "plots_per_block": {k: len(v) for k, v in blocks.items()},
                },
                indent=2,
            )
        )
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    init_gee(args.project)

    available = ee.Feature(
        ee.FeatureCollection(SAMPLES_ASSET).first()
    ).propertyNames().getInfo()
    keep = check_label_columns(available)

    stack = build_embedding_stack()
    result = run_extraction(
        blocks,
        stack,
        keep,
        args.output_dir / "shards",
        args.scale,
        args.max_workers,
    )

    # Re-attach the block partition computed locally, so downstream spatial CV
    # can group on it without recomputing centroids.
    result = result.merge(
        plots[["PLOTID", "lon", "lat", "block_id"]],
        on=["PLOTID", "block_id"],
        how="left",
    )
    suffix = "" if args.max_blocks is None else f"_test{len(blocks)}"
    out_path = args.output_dir / f"embeddings_labelled{suffix}.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Wrote {len(result):,} rows x {result.shape[1]} cols -> {out_path}")

    metadata = {
        "project": args.project,
        "samples_asset": SAMPLES_ASSET,
        "embedding_asset": EMBEDDING_ASSET,
        "embedding_years": list(EMBEDDING_YEARS),
        "embedding_dim": EMBEDDING_DIM,
        "scale_m": args.scale,
        "block_size_deg": args.block_size,
        "n_blocks": len(blocks),
        "n_rows": len(result),
        "label_columns": keep,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
