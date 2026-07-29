"""Run the F1-tuned LDA transition model over AlphaEarth pixels for two cities.

The leaderboard proves the tuned LDA out-of-fold on the labelled plots; this
script turns it into a *map*. For Johannesburg and Oslo it pulls the AlphaEarth
Foundations 2018 and 2024 embeddings straight from Source Cooperative with the
local ``aef_loader_plus`` fork, rebuilds the exact 192-feature vector the model
was trained on (per channel: 2018, 2024, and their difference), and writes a
per-pixel transition class GeoTIFF plus a changed/stable mask.

Why the plus fork matters here. Three of its changes are on this exact path:

* ``chunks=None`` on the stock loader silently materialises the whole 8192x64
  int8 tile (~100 s/tile) via ``expand_dims``; the fork opens at the native
  block size so the windowed city read stays lazy.
* ``dequantize_aef`` uses a 256-entry int8->float32 LUT, so turning the int8
  codes into the float embeddings the model expects is ~1.7x faster.
* ``aoi_geobox`` snaps the target grid to the global 10 m lattice, so 2018 and
  2024 land on identical pixel coordinates -- required, since the transition
  feature is literally ``embedding_2024 - embedding_2018`` per pixel and a
  half-pixel grid offset would corrupt every difference band.

Reprojection uses nearest resampling on the raw int8 (the only exact resampler
on quantized data), then dequantizes; pixels the loader marks nodata (-128 ->
NaN) are left unclassified in the output.

The model and its feature-column order come from ``train_final_lda.py``'s
artefacts; the column order is honoured exactly, because a pixel matrix stacked
in a different order would be silently misaligned against the standardiser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import joblib
import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS

from aef_loader import (
    AEFIndex,
    DataSource,
    VirtualTiffReader,
    aoi_geobox,
    dequantize_aef,
)
from aef_loader.utils import reproject_datatree
from project_paths import project_data_dir

# AOIs in WGS84 (minx, miny, maxx, maxy). Each sits in a single UTM tile per
# year (Johannesburg 35S / EPSG:32735, Oslo 32N / EPSG:32632), so one tile is
# fetched per city per year.
CITY_AOIS: dict[str, tuple[float, float, float, float]] = {
    "johannesburg": (27.95, -26.30, 28.15, -26.10),
    "oslo": (10.60, 59.85, 10.90, 60.00),
}
EMBEDDING_DIM = 64
YEARS = (2018, 2024)

# A readable colormap: stable classes muted, change classes vivid. Filled in
# against the model's actual class list at write time; anything unlisted falls
# back to grey.
CLASS_COLORS = {
    "Nature -> Nature": (39, 119, 64),
    "Cropland -> Cropland": (222, 199, 120),
    "Artificial -> Artificial": (140, 140, 140),
    "Nature -> Cropland": (230, 160, 40),
    "Nature -> Artificial": (214, 39, 40),
    "Cropland -> Nature": (120, 200, 110),
    "Cropland -> Artificial": (255, 80, 80),
    "Artificial -> Nature": (40, 180, 160),
    "Artificial -> Cropland": (200, 200, 60),
    "other": (120, 80, 160),
}


def is_change(label: str, rare_label: str) -> bool:
    return label == rare_label or label.split(" -> ")[0] != label.split(" -> ")[1]


async def load_year_embeddings(
    index: AEFIndex,
    reader: VirtualTiffReader,
    bbox: tuple[float, float, float, float],
    year: int,
    geobox,
) -> np.ndarray:
    """Dequantized ``(64, H, W)`` embedding stack for one year on ``geobox``.

    Reproject (nearest, exact on int8) then dequantize, so the -128 nodata
    sentinel becomes NaN rather than being blended into valid pixels.
    """
    tiles = await index.query(bbox=bbox, years=year)
    if not tiles:
        raise RuntimeError(f"No AEF tiles for {bbox} in {year}")
    # chunks={"band": -1}: keep all 64 bands on one worker so the per-pixel
    # feature build never triggers a cross-chunk shuffle.
    tree = await reader.open_tiles_by_zone(tiles, chunks={"band": -1})
    combined = reproject_datatree(tree, geobox, resampling="nearest")
    deq = dequantize_aef(combined).compute()
    array = np.asarray(deq["embeddings"].values, dtype="float32")
    # (time=1, band, y, x) -> (band, y, x)
    return array[0]


def build_feature_matrix(
    emb_2018: np.ndarray, emb_2024: np.ndarray, feature_columns: list[str]
) -> np.ndarray:
    """Stack pixels into ``(H*W, n_features)`` in the model's column order.

    Reproduces the training layout exactly: for each channel ``i`` the columns
    ``A{i}_2018``, ``A{i}_2024`` and ``A{i}_diff = 2024 - 2018``. Ordering the
    stack by ``feature_columns`` (not by construction order) guarantees the
    pixel matrix lines up with the fitted standardiser column-for-column.
    """
    bands, height, width = emb_2018.shape
    per_name: dict[str, np.ndarray] = {}
    for i in range(bands):
        per_name[f"A{i:02d}_2018"] = emb_2018[i]
        per_name[f"A{i:02d}_2024"] = emb_2024[i]
        per_name[f"A{i:02d}_diff"] = emb_2024[i] - emb_2018[i]

    missing = [c for c in feature_columns if c not in per_name]
    if missing:
        raise ValueError(f"Feature columns not reconstructable from bands: {missing[:5]}")
    stack = np.stack([per_name[c] for c in feature_columns], axis=-1)  # (H, W, F)
    return stack.reshape(height * width, len(feature_columns))


def write_geotiff(
    path: Path, bands: list[tuple[str, np.ndarray]], geobox, nodata: int,
    colormap: dict[int, tuple[int, int, int, int]] | None = None,
) -> None:
    """Write a multi-band uint8 GeoTIFF; ``bands`` is ``[(description, array)]``.

    Every band carries the same colormap (the two framings share a class-code
    space) and a description so the framing is readable straight from the file.
    """
    height, width = bands[0][1].shape
    epsg = geobox.crs.epsg
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=len(bands),
        dtype="uint8", crs=RioCRS.from_epsg(epsg),
        transform=geobox.transform, nodata=nodata, compress="deflate",
    ) as dst:
        for index, (description, array) in enumerate(bands, start=1):
            dst.write(array.astype("uint8"), index)
            dst.set_band_description(index, description)
            if colormap:
                dst.write_colormap(index, colormap)


def encode(predicted: np.ndarray, class_code: dict, rare_label: str,
           n_pixels: int, valid: np.ndarray, nodata: int):
    """Predicted labels over valid pixels -> full-length code + change arrays.

    A label outside the shared class space (only possible if the post-class
    cross-product resurrects a pooled ``other``) is left as nodata rather than
    guessed onto a colour.
    """
    codes = np.full(n_pixels, nodata, dtype="uint8")
    change = np.full(n_pixels, nodata, dtype="uint8")
    codes[valid] = np.array(
        [class_code.get(p, nodata) for p in predicted], dtype="uint8"
    )
    change[valid] = np.array(
        [1 if is_change(p, rare_label) else 0 for p in predicted], dtype="uint8"
    )
    return codes, change


async def infer_city(
    name: str,
    bbox: tuple[float, float, float, float],
    index: AEFIndex,
    reader: VirtualTiffReader,
    direct_model,
    postclass_model,
    sidecar: dict,
    resolution: float,
    output_dir: Path,
) -> dict:
    feature_columns = sidecar["feature_columns"]
    classes = sidecar["classes"]
    rare_label = sidecar["rare_label"]
    class_code = {c: i for i, c in enumerate(classes)}
    nodata = 255

    # Target grid from the AOI's own UTM zone, snapped to the 10 m lattice so
    # both years share pixel coordinates.
    probe = await index.query(bbox=bbox, years=YEARS[0])
    epsg = probe[0].crs_epsg
    geobox = aoi_geobox(bbox, crs=f"EPSG:{epsg}", resolution=resolution,
                        bbox_crs="EPSG:4326")
    height, width = geobox.shape.y, geobox.shape.x
    print(f"[{name}] grid {height}x{width} px @ {resolution} m in EPSG:{epsg}",
          flush=True)

    emb_2018 = await load_year_embeddings(index, reader, bbox, YEARS[0], geobox)
    emb_2024 = await load_year_embeddings(index, reader, bbox, YEARS[1], geobox)
    print(f"[{name}] embeddings loaded {emb_2018.shape}", flush=True)

    n_pixels = height * width
    features = build_feature_matrix(emb_2018, emb_2024, feature_columns)
    valid = np.isfinite(features).all(axis=1)
    n_valid = int(valid.sum())
    print(f"[{name}] {n_valid:,}/{n_pixels:,} valid pixels", flush=True)

    # Direct: the joint 192-vector, one 7-way call. Post-class: the per-date
    # embeddings in channel order (A00..A63), labelled separately then paired.
    direct_pred = direct_model.predict(features[valid]) if n_valid else np.array([])
    x18 = emb_2018.reshape(EMBEDDING_DIM, n_pixels).T[valid]
    x24 = emb_2024.reshape(EMBEDDING_DIM, n_pixels).T[valid]
    pc_pred = (
        postclass_model.predict_from_arrays(x18, x24) if n_valid else np.array([])
    )

    framings = {"direct": direct_pred, "postclass": pc_pred}
    codes = {}
    changes = {}
    tallies = {}
    changed_counts = {}
    for framing, pred in framings.items():
        code, change = encode(pred, class_code, rare_label, n_pixels, valid, nodata)
        code = code.reshape(height, width)
        change = change.reshape(height, width)
        codes[framing] = code
        changes[framing] = change
        seen = code[code != nodata]
        tallies[framing] = {
            classes[c]: int((seen == c).sum())
            for c in range(len(classes)) if (seen == c).any()
        }
        changed_counts[framing] = int((change == 1).sum())
        print(f"[{name}] {framing:9s} changed: {changed_counts[framing]:,} "
              f"({100 * changed_counts[framing] / max(n_valid, 1):.1f}%)", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    class_colormap = {
        class_code[c]: (*CLASS_COLORS.get(c, (120, 120, 120)), 255) for c in classes
    }
    class_colormap[nodata] = (0, 0, 0, 0)
    change_colormap = {0: (200, 200, 200, 255), 1: (214, 39, 40, 255),
                       nodata: (0, 0, 0, 0)}

    trans_path = output_dir / f"{name}_transition_lda.tif"
    change_path = output_dir / f"{name}_change_lda.tif"
    write_geotiff(
        trans_path,
        [("direct LDA transition", codes["direct"]),
         ("post-classification LDA transition", codes["postclass"])],
        geobox, nodata, class_colormap,
    )
    write_geotiff(
        change_path,
        [("direct LDA change", changes["direct"]),
         ("post-classification LDA change", changes["postclass"])],
        geobox, nodata, change_colormap,
    )

    return {
        "city": name,
        "bbox_wgs84": list(bbox),
        "epsg": int(epsg),
        "resolution_m": resolution,
        "grid": [int(height), int(width)],
        "n_valid_pixels": n_valid,
        "bands": {"1": "direct LDA", "2": "post-classification LDA"},
        "n_changed_pixels": changed_counts,
        "class_pixel_counts": tallies,
        "transition_raster": str(trans_path),
        "change_raster": str(change_path),
    }


async def main_async(args) -> None:
    direct_model = joblib.load(args.model_path)
    postclass_model = joblib.load(args.postclass_path)
    sidecar = json.loads(Path(args.sidecar).read_text())
    print(f"Loaded direct '{sidecar['model']}' + post-class | "
          f"{sidecar['n_features']} features | {len(sidecar['classes'])} classes",
          flush=True)

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()

    results = []
    async with VirtualTiffReader(
        manifest_cache_dir=args.manifest_cache
    ) as reader:
        for name in args.cities:
            if name not in CITY_AOIS:
                raise SystemExit(f"Unknown city '{name}'. Have: {sorted(CITY_AOIS)}")
            results.append(
                await infer_city(
                    name, CITY_AOIS[name], index, reader, direct_model,
                    postclass_model, sidecar, args.resolution, args.output_dir,
                )
            )

    (args.output_dir / "inference_summary.json").write_text(
        json.dumps(
            {
                "direct_model_path": str(args.model_path),
                "postclass_model_path": str(args.postclass_path),
                "bands": {
                    "1": "direct LDA (lsqr, shrinkage=0.3, empirical priors)",
                    "2": "post-classification LDA (lsqr, shrinkage=auto, uniform "
                    "priors; per-date labels paired)",
                },
                "embedding_source": "AlphaEarth Foundations via aef_loader_plus "
                "(Source Cooperative)",
                "attribution": "The AlphaEarth Foundations Satellite Embedding "
                "dataset is produced by Google and Google DeepMind.",
                "years": list(YEARS),
                "cities": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nWrote inference_summary.json")


def main() -> None:
    default_models = project_data_dir("models")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path,
        default=default_models / "transition_lda_direct.joblib",
    )
    parser.add_argument(
        "--sidecar", type=Path,
        default=default_models / "transition_lda_direct.json",
    )
    parser.add_argument(
        "--postclass-path", type=Path,
        default=default_models / "transition_lda_pc_postclass.joblib",
        help="Fitted PostClassification model (band 2 of each raster)",
    )
    parser.add_argument("--cities", nargs="+", default=list(CITY_AOIS))
    parser.add_argument("--resolution", type=float, default=10.0,
                        help="Output pixel size in metres (native AEF is 10 m)")
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("inference"))
    parser.add_argument(
        "--manifest-cache", type=Path,
        default=project_data_dir("inference", "manifest_cache"),
        help="Directory for the plus-fork COG manifest cache",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
