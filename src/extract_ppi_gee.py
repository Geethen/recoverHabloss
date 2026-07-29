"""Extract RECOVER predictions for 10k/100k PPI experiments from GEE.

The maximum predicted-only sample is drawn once. Smaller requested samples are
deterministic, stratum-preserving subsets of it. The same prediction bands are
attached to the manually interpreted RECOVER points for bias correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd

from project_paths import default_recover_root, project_data_dir


PROJECT = "ee-gsingh"
SAMPLES_ASSET = "projects/nina/RECOVER/samples_recover_w_ref_label"
ECOREGIONS_ASSET = "RESOLVE/ECOREGIONS/2017"
CROP_TREND_ASSET = (
    "projects/gee-zander-nina/assets/HABLOSS/crop_trend_2018_2024_v1"
)
GREY_TREND_ASSET = (
    "projects/gee-zander-nina/assets/HABLOSS/grey_trend_2018_2024_v1"
)
BUILDINGS_TREND_ASSET = (
    "projects/gee-zander-nina/assets/HABLOSS/buildings_trend_2018_2023_v1"
)
DEFAULT_RECOVER = default_recover_root()
DEFAULT_AREAS = DEFAULT_RECOVER / "data/from_gee/areas_recover"
DEFAULT_GRID = DEFAULT_RECOVER / "data/World_UTM_Grid_cleaned.geojson"
DEFAULT_OUTPUT = project_data_dir("ppi_gee")
DEFAULT_CRS = "EPSG:3410"  # NSIDC EASE-Grid Global, equal-area and GEE-supported
PREDICTOR_COLUMNS = [
    "crop_trend",
    "grey_trend",
    "buildings_trend",
    "yhat_hard",
    "yhat_score",
    "yhat_nature",
    "yhat_cropland",
    "yhat_artificial",
]
def init_gee(project: str) -> None:
    try:
        ee.Initialize(
            project=project,
            opt_url="https://earthengine-highvolume.googleapis.com",
        )
    except Exception:
        ee.Initialize(project=project)


def load_stratum_areas(areas_dir: Path, cache_path: Path | None = None) -> dict[int, float]:
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path)
        return dict(zip(cached["stratum_id"].astype(int), cached["area_m2"].astype(float)))
    totals: dict[int, float] = {}
    files = sorted(areas_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No RECOVER area CSV files in {areas_dir}")
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("stratum") or not row.get("area"):
                    continue
                h = int(float(row["stratum"]))
                totals[h] = totals.get(h, 0.0) + float(row["area"])
    result = {h: area for h, area in totals.items() if area > 0}
    if cache_path is not None:
        pd.DataFrame(
            {"stratum_id": result.keys(), "area_m2": result.values()}
        ).sort_values("stratum_id").to_csv(cache_path, index=False)
    return result


def allocate_counts(
    areas: dict[int, float], total_n: int, minimum: int = 50
) -> dict[int, int]:
    """Area-proportional allocation after a minimum allocation per stratum."""
    strata = sorted(areas)
    if total_n < minimum * len(strata):
        raise ValueError(f"total_n must be at least {minimum * len(strata)}")
    remaining = total_n - minimum * len(strata)
    area_total = sum(areas.values())
    exact = {h: remaining * areas[h] / area_total for h in strata}
    counts = {h: minimum + math.floor(exact[h]) for h in strata}
    left = total_n - sum(counts.values())
    order = sorted(strata, key=lambda h: exact[h] - math.floor(exact[h]), reverse=True)
    for h in order[:left]:
        counts[h] += 1
    return counts


def load_cell_areas(areas_dir: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["stratum_id"] = cached["stratum_id"].astype(int)
        return cached
    rows = []
    for path in sorted(areas_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if not row.get("stratum") or not row.get("area"):
                    continue
                rows.append(
                    {
                        "CellCode": row.get("CellCode") or path.stem.removeprefix("areas_"),
                        "stratum_id": int(float(row["stratum"])),
                        "area_m2": float(row["area"]),
                    }
                )
    frame = pd.DataFrame(rows).groupby(
        ["CellCode", "stratum_id"], as_index=False
    )["area_m2"].sum()
    frame.to_csv(cache_path, index=False)
    return frame


def allocate_cells(
    cell_areas: pd.DataFrame, stratum_counts: dict[int, int]
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for h, requested in stratum_counts.items():
        subset = cell_areas.loc[
            (cell_areas["stratum_id"] == h) & (cell_areas["area_m2"] > 0)
        ].copy()
        exact = requested * subset["area_m2"] / subset["area_m2"].sum()
        subset["count"] = exact.map(math.floor).astype(int)
        left = requested - int(subset["count"].sum())
        subset["remainder"] = exact - subset["count"]
        if left:
            add_idx = subset.nlargest(left, "remainder").index
            subset.loc[add_idx, "count"] += 1
        for row in subset.loc[subset["count"] > 0].itertuples():
            result.setdefault(str(row.CellCode), {})[h] = int(row.count)
    return result


def load_grid_bounds(path: Path) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    def points(value):
        if value and isinstance(value[0], (int, float)):
            yield value
        else:
            for child in value:
                yield from points(child)

    bounds = {}
    for feature in data["features"]:
        coords = list(points(feature["geometry"]["coordinates"]))
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        bounds[str(feature["properties"]["CellCode"])] = [
            min(xs), min(ys), max(xs), max(ys)
        ]
    return bounds


def build_stack() -> ee.Image:
    crop = ee.ImageCollection(CROP_TREND_ASSET).mosaic().rename("crop_trend").unmask(0)
    grey = ee.ImageCollection(GREY_TREND_ASSET).mosaic().rename("grey_trend").unmask(0)
    buildings = (
        ee.ImageCollection(BUILDINGS_TREND_ASSET)
        .mosaic()
        .rename("buildings_trend")
        .unmask(0)
    )

    built_loss = grey.lt(-7).Or(buildings.lt(-7)).rename("yhat_hard")
    strongest_built_trend = grey.min(buildings)
    built_score = (
        strongest_built_trend.multiply(-1)
        .subtract(5)
        .divide(10)
        .clamp(0, 1)
        .rename("yhat_score")
    )

    # Per-class destination scores for vector-valued PPI over the coarse
    # classes {Nature, Cropland, Artificial}. Each raw score rises with the
    # evidence for that destination; they are normalised to a simplex so the
    # predicted composition sums to one at every point, matching the one-hot
    # labelled truth. Nature is the residual "no strong loss trend" mass.
    artificial_raw = strongest_built_trend.multiply(-1).subtract(5).divide(10).clamp(0, 1)
    cropland_raw = crop.multiply(-1).subtract(5).divide(10).clamp(0, 1)
    nature_raw = ee.Image(1).subtract(artificial_raw.max(cropland_raw)).clamp(0, 1)
    denom = nature_raw.add(cropland_raw).add(artificial_raw).max(1e-6)
    yhat_nature = nature_raw.divide(denom).rename("yhat_nature")
    yhat_cropland = cropland_raw.divide(denom).rename("yhat_cropland")
    yhat_artificial = artificial_raw.divide(denom).rename("yhat_artificial")

    # Reproduce RECOVER's five map classes. Later where() calls have priority.
    map_class = (
        ee.Image(1)
        .where(crop.lt(-5), 2)
        .where(grey.lt(-5), 3)
        .where(crop.lt(-7), 4)
        .where(built_loss, 5)
        .rename("map_class")
    )

    # RECOVER combines the 14 RESOLVE biomes into eight reporting biomes.
    ecoregions = ee.FeatureCollection(ECOREGIONS_ASSET).filter(
        ee.Filter.neq("BIOME_NAME", "N/A")
    )
    biome14 = ecoregions.reduceToImage(["BIOME_NUM"], ee.Reducer.first()).toInt()
    biome8 = biome14.remap(
        [14, 6, 11, 4, 5, 8, 10, 12, 13, 1, 2, 3, 7, 9],
        [1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 7, 8, 8],
    ).rename("biome_id")
    stratum = map_class.add(biome8.subtract(1).multiply(5)).rename("stratum_id")

    return ee.Image.cat(
        [
            crop,
            grey,
            buildings,
            built_loss.toByte(),
            built_score,
            yhat_nature,
            yhat_cropland,
            yhat_artificial,
            stratum.toInt(),
        ]
    ).updateMask(biome8.mask())


def fetch_dataframe(collection: ee.FeatureCollection) -> pd.DataFrame:
    return ee.data.computeFeatures(
        {"expression": collection, "fileFormat": "PANDAS_DATAFRAME"}
    )


def extract_labelled(stack: ee.Image) -> pd.DataFrame:
    labels = ee.FeatureCollection(SAMPLES_ASSET)
    predictors = stack.select(PREDICTOR_COLUMNS)

    def attach(feature):
        values = predictors.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=feature.geometry(),
            scale=10,
            maxPixels=1_000_000,
        )
        return feature.set(values)

    keep = ["PLOTID", "stratum", "lc_2018", "lc_2024", "r", *PREDICTOR_COLUMNS]
    return fetch_dataframe(labels.map(attach).select(keep))


def extract_max_sample(
    stack: ee.Image,
    counts: dict[int, int],
    cell_areas: pd.DataFrame,
    grid_bounds: dict[str, list[float]],
    seed: int,
    scale: int,
    crs: str,
    shard_dir: Path,
    max_workers: int,
) -> pd.DataFrame:
    shard_dir.mkdir(parents=True, exist_ok=True)
    cell_counts = allocate_cells(cell_areas, counts)

    def sample_cell(cell_code: str, requested: dict[int, int]) -> pd.DataFrame:
        shard_path = shard_dir / f"cell_{cell_code}.csv"
        if shard_path.exists():
            return pd.read_csv(shard_path)
        region = ee.Geometry.Rectangle(grid_bounds[cell_code], geodesic=False)
        sample = stack.stratifiedSample(
            numPoints=0,
            classBand="stratum_id",
            classValues=list(requested),
            classPoints=[requested[h] for h in requested],
            region=region,
            scale=scale,
            projection=ee.Projection(crs),
            seed=seed + int(cell_code[-6:], 16),
            dropNulls=True,
            tileScale=4,
            geometries=False,
        ).map(lambda ft: ft.set("CellCode", cell_code))
        last_error = None
        for attempt in range(3):
            try:
                frame = fetch_dataframe(sample)
                frame.to_csv(shard_path, index=False)
                return frame
            except Exception as error:
                last_error = error
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"Cell {cell_code} failed after retries") from last_error

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(sample_cell, cell_code, requested): cell_code
            for cell_code, requested in cell_counts.items()
        }
        for future in as_completed(futures):
            cell_code = futures[future]
            frame = future.result()
            print(
                f"Cell {cell_code}: received {len(frame):,} predictions",
                flush=True,
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def nested_subset(
    frame: pd.DataFrame,
    counts: dict[int, int],
    seed: int,
) -> pd.DataFrame:
    parts = []
    selected_indexes = []
    frame = frame.copy()
    frame["stratum_id"] = frame["stratum_id"].astype(int)
    if "split_key" not in frame:
        frame["split_key"] = np.random.default_rng(seed).random(len(frame))
    for h, n in counts.items():
        available = frame.loc[frame["stratum_id"] == h]
        part = available.nsmallest(min(n, len(available)), "split_key")
        parts.append(part)
        selected_indexes.extend(part.index.tolist())
    target_n = sum(counts.values())
    deficit = target_n - sum(len(part) for part in parts)
    if deficit:
        remaining = frame.drop(index=selected_indexes).nsmallest(deficit, "split_key")
        if len(remaining) != deficit:
            raise RuntimeError(
                f"Requested nested sample of {target_n}, only {target_n - deficit + len(remaining)} available"
            )
        parts.append(remaining)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--areas-dir", type=Path, default=DEFAULT_AREAS)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--seed", type=int, default=20250717)
    parser.add_argument(
        "--min-per-stratum",
        type=int,
        default=50,
        help="Minimum predicted-only observations per design stratum",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Equal-area sampling resolution in metres (default: 100)",
    )
    parser.add_argument(
        "--crs",
        default=DEFAULT_CRS,
        help="GEE-supported equal-area sampling CRS (default: EPSG:3410)",
    )
    parser.add_argument("--reuse-max", action="store_true")
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sizes = sorted(set(args.sample_sizes))
    if any(n <= 0 for n in sizes):
        raise ValueError("Sample sizes must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    areas = load_stratum_areas(
        args.areas_dir, args.output_dir / "stratum_areas.csv"
    )
    cell_areas = load_cell_areas(
        args.areas_dir, args.output_dir / "cell_stratum_areas.csv"
    )
    allocations = {
        n: allocate_counts(areas, n, args.min_per_stratum) for n in sizes
    }

    if args.dry_run:
        print(json.dumps({"sample_sizes": sizes, "allocations": allocations}, indent=2))
        return

    init_gee(args.project)
    stack = build_stack()
    grid_bounds = load_grid_bounds(args.grid)
    labelled_path = args.output_dir / "labelled_predictions.csv"
    if not (args.reuse_max and labelled_path.exists()):
        labelled = extract_labelled(stack)
        labelled.to_csv(labelled_path, index=False)
        print(f"Wrote {len(labelled):,} labelled predictions -> {labelled_path}")

    max_n = max(sizes)
    max_path = args.output_dir / f"predicted_{max_n}.csv"
    if args.reuse_max and max_path.exists():
        maximum = pd.read_csv(max_path)
        print(f"Reusing {len(maximum):,} predictions from {max_path}")
    else:
        maximum = extract_max_sample(
            stack,
            allocations[max_n],
            cell_areas,
            grid_bounds,
            args.seed,
            args.scale,
            args.crs,
            args.output_dir / f"predicted_{max_n}_shards",
            args.max_workers,
        )
        maximum.to_csv(max_path, index=False)
        print(f"Wrote {len(maximum):,} predictions -> {max_path}")

    actual_sizes = {}
    for n in sizes:
        path = args.output_dir / f"predicted_{n}.csv"
        if n == max_n:
            subset = maximum
        else:
            subset = nested_subset(maximum, allocations[n], args.seed + n)
            subset.to_csv(path, index=False)
        actual_sizes[str(n)] = len(subset)
        print(f"Prepared nested sample n={len(subset):,} -> {path}")

    metadata = {
        "project": args.project,
        "samples_asset": SAMPLES_ASSET,
        "sample_sizes": sizes,
        "actual_sample_sizes": actual_sizes,
        "seed": args.seed,
        "scale_m": args.scale,
        "sampling_crs": args.crs,
        "allocation": "minimum per stratum, then area-proportional",
        "minimum_per_stratum": args.min_per_stratum,
        "areas_dir": str(args.areas_dir),
        "grid": str(args.grid),
        "assets": {
            "crop_trend": CROP_TREND_ASSET,
            "grey_trend": GREY_TREND_ASSET,
            "buildings_trend": BUILDINGS_TREND_ASSET,
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
