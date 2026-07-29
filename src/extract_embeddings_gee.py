"""Extract AlphaEarth satellite embeddings for the labelled RECOVER plots.

The labelled plots are distributed globally, so a single request over the whole
collection is neither reliable nor fast. Points are assigned to fixed longitude
/latitude blocks, and each block is extracted independently and written to its
own Parquet shard. Shards are resumable: an existing shard is reused, so an
interrupted run continues where it stopped.

Embeddings are pulled for the two endpoints of the RECOVER transition period
(2018 and 2024). The per-year 64-D vectors are stored alongside their
difference, which carries the change signal that the transition model needs.

The samples asset stores only ``PLOTID`` and the coarse reference label ``r``;
the interpreted ``lc_2018``/``lc_2024`` transition lives in the cleaned RECOVER
sampler CSVs. Labels are therefore joined locally by ``PLOTID`` after
extraction, the same way ``compare_vector_estimators.py`` does it. The run
still aborts rather than write an unlabelled table -- the check just happens
against the joined result instead of against the asset.

Two plot sources are supported. The default addresses the hosted RECOVER
samples asset by ``PLOTID`` and reduces over its stored polygons. Passing
``--shapefile`` instead reads a local point sample file and sends its geometry
and attributes up with the request, which is how sample files that are not
hosted in Earth Engine -- and that already carry their own labels -- are
extracted. Each source writes to its own shard directory, so they never reuse
one another's cached blocks.

Typical use::

    # inspect the block partition without touching Earth Engine
    python extract_embeddings_gee.py --dry-run

    # smoke test on a couple of blocks before the global run
    python extract_embeddings_gee.py --max-blocks 2

    # full extraction
    python extract_embeddings_gee.py

    # a local point shapefile that carries its own transition labels
    python extract_embeddings_gee.py \
        --shapefile /path/to/samples_habloss_recover_for_geethen.shp \
        --output-name embeddings_habloss_recover
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

from project_paths import default_recover_root, project_data_dir


PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
EMBEDDING_ASSET = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
DEFAULT_OUTPUT = project_data_dir("embeddings")
EMBEDDING_DIM = 64
EMBEDDING_YEARS = (2018, 2024)
# Everything the samples asset actually carries. The interpreted transition is
# not here -- it is joined locally from the sampler CSVs (see attach_labels).
LABEL_COLUMNS = ["PLOTID", "r"]
# Joined in from the cleaned RECOVER labels and the design shapefile.
TRANSITION_COLUMNS = ["lc_2018", "lc_2024"]
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
    """One image: each year's 64-D embedding plus the endpoint difference.

    The difference is always ``last - first`` over the sorted year span, so it
    carries the same 2018->2024 change signal whether two endpoints or the full
    annual trajectory (2018..2024) are extracted. The intermediate years ride
    alongside as extra per-year bands for a temporal model to consume; the
    endpoint diff keeps the existing direct/post-classification models working
    unchanged.
    """
    ordered = sorted(years)
    collection = ee.ImageCollection(EMBEDDING_ASSET)
    per_year = {}
    for year in ordered:
        image = collection.filterDate(f"{year}-01-01", f"{year + 1}-01-01").mosaic()
        per_year[year] = image.rename(embedding_bands(year))

    layers = [per_year[year] for year in ordered]
    if len(ordered) >= 2:
        first, last = ordered[0], ordered[-1]
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
    """Attach centroid coordinates and a block id to each labelled plot.

    ``lon``/``lat`` are derived from the stored ``geo`` polygon unless the frame
    already carries them, which is the case for point sample files.
    """
    frame = frame.copy()
    if not {"lon", "lat"}.issubset(frame.columns):
        centroids = frame["geo"].map(polygon_centroid)
        frame["lon"] = [c[0] for c in centroids]
        frame["lat"] = [c[1] for c in centroids]
    missing = frame["lon"].isna() | frame["lat"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} plots have unparseable geometry in 'geo'"
        )
    frame["block_id"] = [
        block_id(lon, lat, block_size)
        for lon, lat in zip(frame["lon"], frame["lat"])
    ]
    return frame


def read_point_shapefile(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read a point sample shapefile into a frame, plus its attribute names.

    Every attribute is carried through to Earth Engine, so a file that already
    holds its interpreted transition needs no label join afterwards.
    """
    import shapefile  # pyshp; imported here so --dry-run stays dependency-light

    reader = shapefile.Reader(str(path))
    attributes = [field[0] for field in reader.fields[1:]]
    frame = pd.DataFrame(reader.records(), columns=attributes)
    shapes = reader.shapes()
    if any(len(shape.points) != 1 for shape in shapes):
        raise ValueError(
            f"{path} is not a single-point-per-record shapefile; "
            "the point extraction path expects one vertex per plot"
        )
    frame["lon"] = [shape.points[0][0] for shape in shapes]
    frame["lat"] = [shape.points[0][1] for shape in shapes]
    if "PLOTID" not in frame:
        raise ValueError(f"{path} has no PLOTID field; got {attributes}")
    return frame, attributes


def fetch_dataframe(collection: ee.FeatureCollection) -> pd.DataFrame:
    return ee.data.computeFeatures(
        {"expression": collection, "fileFormat": "PANDAS_DATAFRAME"}
    )


def check_label_columns(available: list[str]) -> list[str]:
    """Keep the asset fields worth carrying through, requiring the join key."""
    present = [c for c in LABEL_COLUMNS if c in available]
    if "PLOTID" not in present:
        raise RuntimeError(
            f"Samples asset {SAMPLES_ASSET} has no PLOTID field. Available "
            f"properties: {sorted(available)}. Extraction aborted: without it "
            "the interpreted transition labels cannot be joined on."
        )
    return present


def attach_labels(frame: pd.DataFrame, recover_root: Path) -> pd.DataFrame:
    """Join the interpreted transition and design stratum on by PLOTID.

    The asset carries no transition, so this is where the model's target comes
    from. Aborts if the join lands on nothing, which would otherwise produce an
    unlabelled table that only fails later in the model script.
    """
    # Imported here so --dry-run stays free of shapefile/CSV reads.
    from analyse_transition_composition import read_clean_recover_labels
    from compare_ppi_subsets import read_design_strata

    _, labels, _ = read_clean_recover_labels(recover_root)
    design = read_design_strata(recover_root)
    merged = frame.merge(
        labels[["PLOTID", *TRANSITION_COLUMNS]],
        on="PLOTID",
        how="left",
        validate="many_to_one",
    )
    if "stratum" not in merged:
        merged = merged.merge(
            design, on="PLOTID", how="left", validate="many_to_one"
        )

    labelled = merged[TRANSITION_COLUMNS].notna().all(axis=1)
    if not labelled.any():
        raise RuntimeError(
            f"No extracted plot matched the cleaned RECOVER labels under "
            f"{recover_root}. Extraction aborted rather than write an "
            "unlabelled table."
        )
    missing = int((~labelled).sum())
    if missing:
        print(
            f"{missing:,} of {len(merged):,} plots have no interpreted "
            "transition and will be dropped by the model script"
        )
    return merged


def asset_collection(block: pd.DataFrame, keep: list[str]) -> ee.FeatureCollection:
    """One block's plots, taken from the hosted samples asset by PLOTID.

    Reduces over the stored plot polygon, so this is the path to use when the
    asset's own geometry is authoritative.
    """
    return ee.FeatureCollection(SAMPLES_ASSET).filter(
        ee.Filter.inList("PLOTID", block["PLOTID"].tolist())
    )


def point_collection(block: pd.DataFrame, keep: list[str]) -> ee.FeatureCollection:
    """One block's plots, built client-side from a lon/lat point table.

    Used for sample files that are not hosted in Earth Engine. The attributes
    ride along on each feature, so tables that already carry their labels need
    no join afterwards.
    """
    features = []
    for row in block.to_dict("records"):
        properties = {k: str(row[k]) for k in keep}
        features.append(
            ee.Feature(
                ee.Geometry.Point([float(row["lon"]), float(row["lat"])]),
                properties,
            )
        )
    return ee.FeatureCollection(features)


def extract_block(
    stack: ee.Image,
    block: pd.DataFrame,
    keep: list[str],
    scale: int,
    years: tuple[int, ...] = EMBEDDING_YEARS,
    build_collection=asset_collection,
    retries: int = 3,
) -> pd.DataFrame:
    """Reduce the embedding stack over one block's plots."""
    plots = build_collection(block, keep)

    def attach(feature):
        values = stack.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=scale,
            maxPixels=1_000_000,
        )
        return feature.set(values)

    bands = [b for year in sorted(years) for b in embedding_bands(year)]
    if len(years) >= 2:
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
    blocks: dict[str, pd.DataFrame],
    stack: ee.Image,
    keep: list[str],
    shard_dir: Path,
    scale: int,
    max_workers: int,
    years: tuple[int, ...] = EMBEDDING_YEARS,
    build_collection=asset_collection,
) -> pd.DataFrame:
    shard_dir.mkdir(parents=True, exist_ok=True)

    def one_block(name: str, block: pd.DataFrame) -> pd.DataFrame:
        shard_path = shard_dir / f"block_{name}.parquet"
        if shard_path.exists():
            return pd.read_parquet(shard_path)
        frame = extract_block(stack, block, keep, scale, years, build_collection)
        frame["block_id"] = name
        frame.to_parquet(shard_path, index=False)
        return frame

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(one_block, name, block): name
            for name, block in blocks.items()
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
        default=project_data_dir("ppi_gee", "labelled_predictions.csv"),
        help="Table with 'geo' and 'PLOTID', used to build the block partition",
    )
    parser.add_argument(
        "--shapefile",
        type=Path,
        default=None,
        help=(
            "Point sample shapefile to extract instead of --plots. Its "
            "attributes are carried through, so a file holding lc_2018/lc_2024 "
            "needs no label join"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Base name for the merged output (default: embeddings_labelled)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--recover-root",
        type=Path,
        default=default_recover_root(),
        help="Read-only RECOVER project root holding the cleaned label CSVs",
    )
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
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(EMBEDDING_YEARS),
        help=(
            "Embedding years to extract (default: 2018 2024). Pass the full span "
            "'2018 2019 2020 2021 2022 2023 2024' to get the annual trajectory; "
            "the endpoint (first->last) difference bands are always added"
        ),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the block partition without contacting Earth Engine",
    )
    args = parser.parse_args()

    # Two sources: a hosted asset addressed by PLOTID, or a local point
    # shapefile whose geometry and attributes are sent up with the request.
    if args.shapefile is not None:
        plots, attributes = read_point_shapefile(args.shapefile)
        source_label = str(args.shapefile)
    else:
        plots = pd.read_csv(args.plots)
        if "geo" not in plots or "PLOTID" not in plots:
            raise ValueError(
                f"{args.plots} must contain 'geo' and 'PLOTID' columns"
            )
        attributes = None
        source_label = str(args.plots)
    plots = assign_blocks(plots, args.block_size)

    grouped = dict(tuple(plots.groupby("block_id")))
    ordered = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
    if args.max_blocks is not None:
        ordered = ordered[: args.max_blocks]
    blocks = dict(ordered)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "source": source_label,
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

    if args.shapefile is not None:
        keep = attributes
        build_collection = point_collection
    else:
        available = ee.Feature(
            ee.FeatureCollection(SAMPLES_ASSET).first()
        ).propertyNames().getInfo()
        keep = check_label_columns(available)
        build_collection = asset_collection

    # Shards are keyed by block id only, so each plot source needs its own
    # directory -- otherwise a second source silently reuses the first one's
    # cached blocks.
    stem = args.output_name or "embeddings_labelled"
    years = tuple(args.years)
    stack = build_embedding_stack(years)
    result = run_extraction(
        blocks,
        stack,
        keep,
        args.output_dir / "shards" / stem,
        args.scale,
        args.max_workers,
        years,
        build_collection,
    )

    # Re-attach the block partition computed locally, so downstream spatial CV
    # can group on it without recomputing centroids. Deduplicated on the key
    # because sample files may repeat a PLOTID (reverified or dual-frame
    # plots); without it those rows would fan out on the merge.
    coordinates = plots[["PLOTID", "lon", "lat", "block_id"]].drop_duplicates(
        ["PLOTID", "block_id"]
    )
    result = result.merge(
        coordinates, on=["PLOTID", "block_id"], how="left", validate="many_to_one"
    )
    if args.shapefile is None:
        result = attach_labels(result, args.recover_root)
        label_source = str(args.recover_root)
    else:
        # The shapefile carried its own transition up to Earth Engine and back.
        label_source = str(args.shapefile)

    suffix = "" if args.max_blocks is None else f"_test{len(blocks)}"
    out_path = args.output_dir / f"{stem}{suffix}.parquet"
    result.to_parquet(out_path, index=False)
    print(f"Wrote {len(result):,} rows x {result.shape[1]} cols -> {out_path}")

    metadata = {
        "project": args.project,
        "plot_source": source_label,
        "embedding_asset": EMBEDDING_ASSET,
        "embedding_years": list(years),
        "embedding_dim": EMBEDDING_DIM,
        "scale_m": args.scale,
        "block_size_deg": args.block_size,
        "n_blocks": len(blocks),
        "n_rows": len(result),
        "carried_columns": keep,
        "label_source": label_source,
        "n_labelled": int(result[TRANSITION_COLUMNS].notna().all(axis=1).sum()),
        "n_unique_plotids": int(result["PLOTID"].nunique()),
    }
    (args.output_dir / f"metadata_{stem}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
