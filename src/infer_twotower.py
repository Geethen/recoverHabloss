"""Map the symmetric two-tower over two AOIs under three modality views.

Turns the symmetric AlphaEarth+Tessera two-tower (``model_zoo._TwoTowerTrunk``,
``fusion='gated_mean'``, both towers mask-gated, modality dropout on both) into a
map. For each AOI it runs the *same trained model* three ways -- the direct test
of the architecture's "handles missing data from either side":

    both       AlphaEarth + Tessera fused        (the deploy prediction)
    aef_only   Tessera masked off                (AlphaEarth tower alone)
    tess_only  AlphaEarth masked off             (Tessera tower alone)

Data. AlphaEarth (64-D x {2018,2024,diff} = 192 features) is fetched with the
``aef_loader_plus`` fork exactly as ``infer_cities.py`` does. Tessera (128-D x
{2018,2024,diff} = 384 features) is loaded densely here: the whole 0.1-degree
tiles overlapping the AOI are pulled from S3 (anonymous byte range via obstore),
dequantised (int8 * per-pixel scale, == geotessera), and reprojected onto the
*identical* AlphaEarth UTM geobox with nearest resampling (exact on the 10 m
lattice both products share). A Tessera pixel is "present" only where BOTH years
are finite -- the same ``tess_present = both-years`` convention the model trained
on, so the 2018->2024 diff is real. Where a modality is absent a pixel's block is
zero-imputed and its mask is 0, so the trunk gates that tower out and predicts
from whatever remains (the model's ``_prepare`` does the mask-aware standardise).

Coverage caveat, surfaced in the summary. Tessera 2018 is globally sparse (~37%).
Oslo has both years, so all three views are genuinely tri-modal there. Johannesburg
has 2024 only, so ``tess_present`` is 0 everywhere -- ``both`` collapses to
``aef_only`` (graceful fallback, the design goal) and ``tess_only`` is degenerate
(no modality present). That asymmetry is the point, not a bug: it is the
coverage-bound regime the symmetric architecture exists to survive.

Outputs are chosen to be light and QGIS-native (see ``write_class_raster``):
single-band **uint8, paletted, tiled, DEFLATE-compressed** GeoTIFFs with an
embedded colour table, internal overviews, and a ``.qml`` sidecar so QGIS shows
named categories straight away. Two reads per view -- the informative coarse3
(9-transition) and the deployed merged2 (Vegetation/Artificial) -- plus a change
mask. Everything lands in ``inference/twotower_<timestamp>/``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.enums import Resampling
from rasterio.warp import reproject

import obstore

from aef_loader import AEFIndex, DataSource, VirtualTiffReader, aoi_geobox
from geotessera.registry import tile_from_world, tile_to_landmask_filename

from experiment_hier_tessera import TESSERA, attach_tessera
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from extract_tessera_points import DIM, HDR, PREFIX, _store, landmask_geo
from infer_cities import (
    CITY_AOIS,
    CLASS_COLORS,
    YEARS,
    build_feature_matrix,
    is_change,
    load_year_embeddings,
)
from model_zoo import DEFAULT_INPUT, RARE_LABEL, HierarchicalSoftmaxNN, to_merged_label
from project_paths import project_data_dir

# merged2 (Vegetation/Artificial) palette; stable muted, change vivid.
MERGED_COLORS = {
    "Vegetation -> Vegetation": (39, 119, 64),
    "Artificial -> Artificial": (140, 140, 140),
    "Vegetation -> Artificial": (214, 39, 40),
    "Artificial -> Vegetation": (40, 180, 160),
    "other": (120, 80, 160),
}
VIEWS = ("both", "aef_only", "tess_only")
NODATA = 255


# --------------------------------------------------------------------------
# Dense Tessera: whole overlapping tiles -> reprojected (128, H, W) on the geobox
# --------------------------------------------------------------------------
def _tile_stems(bbox: tuple[float, float, float, float]) -> list[str]:
    """Grid stems (``grid_<lon>_<lat>``) of every 0.1-degree tile touching bbox."""
    minx, miny, maxx, maxy = bbox
    stems: dict[str, None] = {}
    # Step finer than a tile (0.05 deg) so no overlapping tile is missed.
    xs = np.arange(minx - 0.1, maxx + 0.1, 0.05)
    ys = np.arange(miny - 0.1, maxy + 0.1, 0.05)
    for x in xs:
        for y in ys:
            tl, ta = tile_from_world(float(x), float(y))
            stems.setdefault(tile_to_landmask_filename(tl, ta)[:-5], None)
    return list(stems)


async def _download(key: str) -> bytes | None:
    try:
        r = await obstore.get_async(_store, key)
        return bytes(await r.bytes_async())
    except FileNotFoundError:
        return None


async def _load_tile(stem: str, year: int):
    """Dequantised ``(H, W, 128)`` float32 tile + (crs, transform), or None.

    NaN where the per-pixel scale is non-finite (water / nodata), matching the
    point extractor. Whole-tile read: the dense map wants every pixel, so a single
    object GET per tile beats millions of range GETs.
    """
    geo = await landmask_geo(stem)
    if geo is None:
        return None
    crs, transform, W, H = geo
    base = f"{PREFIX}/global_0.1_degree_representation/{year}/{stem}/{stem}"
    qb, sb = await asyncio.gather(
        _download(f"{base}.npy"), _download(f"{base}_scales.npy")
    )
    if qb is None or sb is None:
        return None
    q = np.frombuffer(qb, np.int8, H * W * DIM, offset=HDR).reshape(H, W, DIM)
    s = np.frombuffer(sb, "<f4", H * W, offset=HDR).reshape(H, W)
    deq = q.astype("float32") * s[:, :, None]  # non-finite scale -> NaN row
    return deq, crs, transform


async def load_tessera_year(bbox, year, geobox) -> np.ndarray:
    """Reproject every overlapping Tessera tile onto ``geobox`` -> ``(128, H, W)``.

    Nearest resampling on the dequantised float (10 m -> 10 m, so effectively a
    grid snap), NaN where absent. Tiles are disjoint, so a later tile only fills
    the still-NaN pixels of the running mosaic.
    """
    Ht, Wt = int(geobox.shape.y), int(geobox.shape.x)
    dst_crs = RioCRS.from_epsg(geobox.crs.epsg)
    out = np.full((DIM, Ht, Wt), np.nan, "float32")
    for stem in _tile_stems(bbox):
        loaded = await _load_tile(stem, year)
        if loaded is None:
            continue
        deq, crs, transform = loaded
        src = np.ascontiguousarray(np.moveaxis(deq, 2, 0))  # (128, H, W)
        del deq
        tmp = np.full_like(out, np.nan)
        reproject(
            source=src, destination=tmp,
            src_transform=transform, src_crs=crs,
            dst_transform=geobox.transform, dst_crs=dst_crs,
            src_nodata=np.nan, dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        del src
        np.copyto(out, tmp, where=np.isnan(out) & np.isfinite(tmp))
        del tmp
    return out


def tessera_matrix(stack18, stack24, tess_columns: list[str]) -> np.ndarray:
    """``(H*W, 384)`` in the model's ``tess_columns`` order from the two stacks."""
    _, Ht, Wt = stack18.shape
    n = Ht * Wt
    diff = stack24 - stack18
    by_name: dict[str, np.ndarray] = {}
    for i in range(DIM):
        by_name[f"TE{i:03d}_2018"] = stack18[i].reshape(n)
        by_name[f"TE{i:03d}_2024"] = stack24[i].reshape(n)
        by_name[f"TE{i:03d}_diff"] = diff[i].reshape(n)
    return np.stack([by_name[c] for c in tess_columns], axis=1)


# --------------------------------------------------------------------------
# Prediction: one trained model, three modality views, batched over pixels
# --------------------------------------------------------------------------
def predict_views(model, aef_mat, tess_mat, aef_cols, tess_cols, batch: int):
    """Fine + merged uint8 code rasters (flat) for each of the three views.

    A view forces a modality on/off through its mask column; absent blocks are
    left as-is (the model zero-imputes NaN and gates the tower out). Prediction is
    batched so a multi-million-pixel AOI never builds one giant frame on the GPU.
    """
    import pandas as pd

    n = aef_mat.shape[0]
    aef_ok = np.isfinite(aef_mat).all(1)
    tess_ok = np.isfinite(tess_mat).all(1)
    view_valid = {
        "both": aef_ok | tess_ok,
        "aef_only": aef_ok,
        "tess_only": tess_ok,
    }
    view_masks = {  # (aef_present, tess_present) per pixel for the view
        "both": (aef_ok.astype("float32"), tess_ok.astype("float32")),
        "aef_only": (np.ones(n, "float32"), np.zeros(n, "float32")),
        "tess_only": (np.zeros(n, "float32"), np.ones(n, "float32")),
    }
    fine_code = {c: i for i, c in enumerate(model.fine_classes_)}
    merged_code = {c: i for i, c in enumerate(model.merged_classes_)}

    out = {}
    for view in VIEWS:
        valid = view_valid[view]
        idx = np.flatnonzero(valid)
        fine = np.full(n, NODATA, "uint8")
        merged = np.full(n, NODATA, "uint8")
        ma, mt = view_masks[view]
        for s0 in range(0, len(idx), batch):
            sel = idx[s0:s0 + batch]
            data = {c: aef_mat[sel, j] for j, c in enumerate(aef_cols)}
            data.update({c: tess_mat[sel, j] for j, c in enumerate(tess_cols)})
            data["aef_present"] = ma[sel]
            data["tess_present"] = mt[sel]
            frame = pd.DataFrame(data)
            pf = model.predict(frame)
            pm = model.predict_merged(frame)
            fine[sel] = [fine_code.get(p, NODATA) for p in pf]
            merged[sel] = [merged_code.get(p, NODATA) for p in pm]
        out[view] = (fine, merged, int(valid.sum()))
    return out


# --------------------------------------------------------------------------
# QGIS-native output: paletted uint8 GeoTIFF + overviews + .qml sidecar
# --------------------------------------------------------------------------
def write_class_raster(path: Path, codes2d, geobox, classes, colors):
    """Single-band paletted uint8 GeoTIFF (tiled/DEFLATE) + overviews + QML.

    Paletted uint8 is the smallest faithful encoding of a class map and the one
    QGIS renders natively; the embedded colour table gives instant colours and
    the ``.qml`` sidecar names each category in the legend.
    """
    height, width = codes2d.shape
    code_of = {c: i for i, c in enumerate(classes)}
    colormap = {i: (*colors.get(c, (120, 120, 120)), 255)
                for c, i in code_of.items()}
    colormap[NODATA] = (0, 0, 0, 0)
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint8", crs=RioCRS.from_epsg(geobox.crs.epsg),
        transform=geobox.transform, nodata=NODATA,
        compress="deflate", predictor=1, tiled=True, blockxsize=256,
        blockysize=256,
    ) as dst:
        dst.write(codes2d.astype("uint8"), 1)
        dst.write_colormap(1, colormap)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")
    _write_qml(path.with_suffix(".qml"), code_of, colors)


def _write_qml(path: Path, code_of: dict, colors: dict) -> None:
    """Minimal QGIS paletted-raster style so categories show with names."""
    entries = "\n".join(
        f'          <paletteEntry value="{i}" '
        f'color="#{colors.get(c, (120, 120, 120))[0]:02x}'
        f'{colors.get(c, (120, 120, 120))[1]:02x}'
        f'{colors.get(c, (120, 120, 120))[2]:02x}" '
        f'alpha="255" label="{c}"/>'
        for c, i in code_of.items()
    )
    path.write_text(
        '<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
        '<qgis version="3.28"><pipe>\n'
        '  <rasterrenderer type="paletted" band="1" opacity="1">\n'
        '    <rasterTransparency/>\n'
        '    <colorPalette>\n' + entries + '\n    </colorPalette>\n'
        '  </rasterrenderer>\n'
        '</pipe><blendMode>0</blendMode></qgis>\n',
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
def fit_model(tessera_path: Path, seed: int):
    """Fit the deployed symmetric two-tower on every labelled plot (no CV)."""
    frame, aef_cols = build_frame(DEFAULT_INPUT)
    frame, tgroups, _ = attach_tessera(frame, tessera_path)
    frame = frame.assign(aef_present=1.0)
    tess_cols = tgroups["tess_2yr"]
    target = target_for_legend(frame, LEGENDS["coarse3"], 20)
    model = HierarchicalSoftmaxNN(
        aef_cols + tess_cols, arch="two_tower", loss="focal", epochs=30,
        aef_columns=aef_cols, tess_columns=tess_cols,
        mask_column="tess_present", aef_mask_column="aef_present",
        fusion="gated_mean", modality_dropout=0.5, tower_dim=256, seed=seed,
    )
    print(f"Fitting symmetric two-tower on {len(frame):,} plots "
          f"(aef={len(aef_cols)} tess={len(tess_cols)}) ...", flush=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, target.to_numpy())
    return model, aef_cols, tess_cols


async def infer_aoi(name, bbox, index, reader, model, aef_cols, tess_cols,
                    resolution, batch, out_dir):
    probe = await index.query(bbox=bbox, years=YEARS[0])
    epsg = probe[0].crs_epsg
    geobox = aoi_geobox(bbox, crs=f"EPSG:{epsg}", resolution=resolution,
                        bbox_crs="EPSG:4326")
    H, W = int(geobox.shape.y), int(geobox.shape.x)
    print(f"\n[{name}] grid {H}x{W} @ {resolution} m EPSG:{epsg}", flush=True)

    emb18 = await load_year_embeddings(index, reader, bbox, YEARS[0], geobox)
    emb24 = await load_year_embeddings(index, reader, bbox, YEARS[1], geobox)
    aef_mat = build_feature_matrix(emb18, emb24, aef_cols)
    del emb18, emb24
    print(f"[{name}] AlphaEarth loaded", flush=True)

    t18 = await load_tessera_year(bbox, YEARS[0], geobox)
    t24 = await load_tessera_year(bbox, YEARS[1], geobox)
    cov18 = float(np.isfinite(t18).all(0).mean())
    cov24 = float(np.isfinite(t24).all(0).mean())
    tess_mat = tessera_matrix(t18, t24, tess_cols)
    del t18, t24
    print(f"[{name}] Tessera loaded | pixel coverage 2018={cov18:.1%} "
          f"2024={cov24:.1%}", flush=True)

    preds = predict_views(model, aef_mat, tess_mat, aef_cols, tess_cols, batch)
    del aef_mat, tess_mat

    fine_classes = list(model.fine_classes_)
    merged_classes = list(model.merged_classes_)
    view_stats = {}
    for view, (fine, merged, n_valid) in preds.items():
        vdir = out_dir / name
        vdir.mkdir(parents=True, exist_ok=True)
        write_class_raster(vdir / f"{name}_{view}_coarse3.tif",
                           fine.reshape(H, W), geobox, fine_classes, CLASS_COLORS)
        write_class_raster(vdir / f"{name}_{view}_merged2.tif",
                           merged.reshape(H, W), geobox, merged_classes,
                           MERGED_COLORS)
        # Change mask from the deployed merged2 read (0 stable, 1 change).
        chg = np.full_like(merged, NODATA)
        valid_m = merged != NODATA
        chg[valid_m] = np.array(
            [1 if is_change(merged_classes[c], RARE_LABEL) else 0
             for c in merged[valid_m]], "uint8")
        write_class_raster(vdir / f"{name}_{view}_change.tif",
                           chg.reshape(H, W), geobox, ["stable", "change"],
                           {"stable": (200, 200, 200), "change": (214, 39, 40)})
        n_change = int((chg == 1).sum())
        view_stats[view] = {
            "n_valid_pixels": n_valid,
            "n_change_pixels": n_change,
            "change_fraction": round(n_change / max(n_valid, 1), 4),
        }
        print(f"[{name}] {view:9s} valid={n_valid:,} change={n_change:,} "
              f"({100 * n_change / max(n_valid, 1):.1f}%)", flush=True)

    return {
        "aoi": name, "bbox_wgs84": list(bbox), "epsg": int(epsg),
        "resolution_m": resolution, "grid": [H, W],
        "tessera_pixel_coverage": {"2018": round(cov18, 4), "2024": round(cov24, 4)},
        "views": view_stats,
    }


async def main_async(args) -> None:
    model, aef_cols, tess_cols = fit_model(args.tessera, args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"twotower_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {out_dir}", flush=True)

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    results = []
    async with VirtualTiffReader(manifest_cache_dir=args.manifest_cache) as reader:
        for name in args.aois:
            if name not in CITY_AOIS:
                raise SystemExit(f"Unknown AOI '{name}'. Have: {sorted(CITY_AOIS)}")
            results.append(await infer_aoi(
                name, CITY_AOIS[name], index, reader, model, aef_cols, tess_cols,
                args.resolution, args.batch, out_dir))

    (out_dir / "inference_summary.json").write_text(json.dumps({
        "model": "HierarchicalSoftmaxNN symmetric two-tower "
                 "(arch=two_tower, fusion=gated_mean, modality_dropout=0.5, "
                 "loss=focal, epochs=30, tower_dim=256)",
        "views": {
            "both": "AlphaEarth + Tessera fused",
            "aef_only": "Tessera masked off (AlphaEarth tower alone)",
            "tess_only": "AlphaEarth masked off (Tessera tower alone)",
        },
        "reads": {
            "coarse3": "9-transition Nature/Cropland/Artificial (informative)",
            "merged2": "Vegetation/Artificial transition (deployed)",
            "change": "merged2 change mask (0 stable, 1 change)",
        },
        "tessera_present_convention": "both 2018 and 2024 finite (diff is real)",
        "years": list(YEARS),
        "attribution": "AlphaEarth Foundations (c) Google / Google DeepMind; "
                       "Tessera embeddings via geotessera.",
        "aois": results,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'inference_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aois", nargs="+", default=list(CITY_AOIS))
    parser.add_argument("--tessera", type=Path,
                        default=project_data_dir("embeddings", TESSERA))
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--batch", type=int, default=200_000,
                        help="Pixels per GPU prediction batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("inference"))
    parser.add_argument("--manifest-cache", type=Path,
                        default=project_data_dir("inference", "manifest_cache"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
