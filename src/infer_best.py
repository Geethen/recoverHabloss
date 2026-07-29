"""Map the best two-tower configuration found by the change-F1 search.

The model is ``mc_dropout_scalars`` -- the top of ``src/TWOTOWER_RESEARCH.md``
at merged2 change-F1 **0.6704 +/-0.003** on the 6,414-plot deploy read (5 seeds,
5-fold spatially blocked CV), against 0.6594 for the previously deployed
symmetric two-tower and 0.6577 for AlphaEarth-only. Three ingredients, all of
which survived a 5-seed re-run:

1. **Asymmetric regularisation.** The Tessera tower gets dropout 0.7 while the
   AlphaEarth tower stays at 0.4. Tessera is the sharper but noisier modality and
   wants harder regularisation; the dose-response over 0.2/0.4/0.6/0.7/0.8 is
   unimodal, and *narrowing* the tower instead makes things worse -- the gain is
   stochastic unit-dropping, not lower capacity.
2. **Per-modality change scalars.** Cosine distance, L2, L1, Chebyshev and norm
   change of each modality's 2018->2024 endpoint pair, appended to that
   modality's own tower. Shared with training via
   ``twotower_lab.change_scalar_arrays`` so the two paths cannot drift.
3. **Monte-Carlo modality dropout at inference.** The Tessera gate stays
   stochastic when predicting: ``--mc-passes`` forward passes per pixel with the
   gate randomly dropped on the pixels that have Tessera, averaged. The map then
   integrates over "trust Tessera" and "ignore Tessera" instead of committing to
   either. This is what lifts 0.6665 to 0.6704, and it is free at train time.

Alongside the headline map the script writes the deterministic ``both`` view (the
same model, gate always on where Tessera exists) so the effect of the averaging is
visible rather than asserted, plus an ``mc_std`` uncertainty band -- the standard
deviation of P(change) across the MC passes, which is a genuine per-pixel
"how much does this depend on trusting Tessera" layer and the natural thing to
stratify field checking on.

Everything else -- AlphaEarth loading, dense Tessera tiles, the UTM geobox, the
QGIS-native paletted uint8 output -- is reused unchanged from ``infer_twotower.py``.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P infer_best.py                       # both AOIs, 16 MC passes
    $P infer_best.py --aois oslo --mc-passes 32
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from aef_loader import AEFIndex, DataSource, VirtualTiffReader, aoi_geobox

from experiment_hier_tessera import TESSERA, attach_tessera
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from infer_cities import (
    CITY_AOIS,
    CLASS_COLORS,
    YEARS,
    build_feature_matrix,
    is_change,
    load_year_embeddings,
)
from infer_twotower import (
    MERGED_COLORS,
    NODATA,
    load_tessera_year,
    tessera_matrix,
    write_class_raster,
)
from model_zoo import DEFAULT_INPUT, RARE_LABEL, HierarchicalSoftmaxNN
from project_paths import project_data_dir
from twotower_lab import change_scalar_arrays

# The winning recipe, one place. Mirrors twotower_lab.tt_variant(
#   "mc_dropout_scalars", dropout_tess=0.7, scalars=True) + MC gate averaging.
RECIPE = dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
              fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7)
UNCERTAINTY_COLORS = {  # sequential single hue: low -> high disagreement
    "0.00-0.05": (222, 235, 247), "0.05-0.10": (158, 202, 225),
    "0.10-0.20": (66, 146, 198), "0.20+": (8, 69, 148),
}
UNCERTAINTY_BINS = (0.05, 0.10, 0.20)


# --------------------------------------------------------------------------
def scalar_columns(mat: np.ndarray, cols: list[str], prefix: str):
    """Change scalars for a pixel matrix, addressed by the model's column names."""
    idx = {c: j for j, c in enumerate(cols)}
    c18 = sorted(c for c in cols if c.endswith("_2018"))
    c24 = sorted(c for c in cols if c.endswith("_2024"))
    x1 = mat[:, [idx[c] for c in c18]].astype("float64")
    x2 = mat[:, [idx[c] for c in c24]].astype("float64")
    return change_scalar_arrays(x1, x2, prefix)


def fit_model(tessera_path: Path, seed: int):
    """Fit the winning two-tower on every labelled plot (no CV -- this is deploy).

    Returns the model plus the four column groups it expects, so the raster path
    builds its frames in exactly the order the towers were standardised on.
    """
    frame, aef_cols = build_frame(DEFAULT_INPUT)
    frame, tgroups, _ = attach_tessera(frame, tessera_path)
    tess_cols = tgroups["tess_2yr"]

    aef18 = sorted(c for c in aef_cols if c.endswith("_2018"))
    aef24 = sorted(c for c in aef_cols if c.endswith("_2024"))
    te18 = sorted(c for c in tess_cols if c.endswith("_2018"))
    a_s = change_scalar_arrays(frame[aef18].to_numpy("float64"),
                               frame[aef24].to_numpy("float64"), "aefS")
    t_s = change_scalar_arrays(frame[te18].to_numpy("float64"),
                               frame[tgroups["tess_2024"]].to_numpy("float64"), "tesS")
    frame = pd.concat(
        [frame, pd.DataFrame({**a_s, **t_s}, index=frame.index)], axis=1)
    frame = frame.assign(aef_present=1.0).copy()

    aef_block = aef_cols + list(a_s)
    tess_block = tess_cols + list(t_s)
    target = target_for_legend(frame, LEGENDS["coarse3"], 20)

    model = HierarchicalSoftmaxNN(
        aef_block + tess_block, seed=seed,
        aef_columns=aef_block, tess_columns=tess_block,
        mask_column="tess_present", aef_mask_column="aef_present", **RECIPE)
    print(f"Fitting mc_dropout_scalars on {len(frame):,} plots "
          f"(aef={len(aef_block)} tess={len(tess_block)}) ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, target.to_numpy())
    return model, aef_cols, tess_cols, list(a_s), list(t_s)


# --------------------------------------------------------------------------
def _mean_probs(models, frame):
    """Merged2 (and fine) posteriors averaged over the seed ensemble.

    Averaging *probabilities* rather than voting on labels is what the plot-level
    evidence measured: pooling the seeds is worth +0.005 change-F1, +0.004
    macro-F1 and +0.007 stable-Artificial recall, and costs only the extra
    forward passes because the seeds are trained once per run either way. A
    single-model list reproduces the old behaviour exactly.
    """
    p_fine = p_merged = None
    for m in models:
        pf, pm = m._probs(frame)
        p_fine = pf if p_fine is None else p_fine + pf
        p_merged = pm if p_merged is None else p_merged + pm
    return p_fine / len(models), p_merged / len(models)


def predict_mc(models, aef_mat, tess_mat, aef_cols, tess_cols, batch, passes,
               seed):
    """Deterministic and MC-averaged predictions over the pixel grid.

    The MC read repeats the forward pass with the Tessera gate dropped at random
    on the pixels that have Tessera (AlphaEarth is never dropped, so no pixel is
    ever left with no modality) and averages the merged2 posteriors. ``mc_std`` is
    the standard deviation of P(change) across those passes.
    """
    n = aef_mat.shape[0]
    aef_ok = np.isfinite(aef_mat).all(1)
    tess_ok = np.isfinite(tess_mat).all(1)
    valid = aef_ok | tess_ok
    idx = np.flatnonzero(valid)

    a_scal = scalar_columns(aef_mat, aef_cols, "aefS")
    t_scal = scalar_columns(tess_mat, tess_cols, "tesS")

    ref = models[0]
    merged_classes = list(ref.merged_classes_)
    fine_code = {c: i for i, c in enumerate(ref.fine_classes_)}
    merged_code = {c: i for i, c in enumerate(merged_classes)}
    is_chg = np.array([is_change(c, RARE_LABEL) for c in merged_classes])

    fine_det = np.full(n, NODATA, "uint8")
    merged_det = np.full(n, NODATA, "uint8")
    merged_mc = np.full(n, NODATA, "uint8")
    # The counterfactual: the same model on the same pixels with the Tessera
    # gate forced off. Comparing this to the deterministic read is the only
    # honest way to ask "does the detail modality omit change here?" -- Oslo
    # against Johannesburg cannot answer it, because those two AOIs differ in
    # city as well as in Tessera coverage.
    merged_aef = np.full(n, NODATA, "uint8")
    p_change = np.full(n, np.nan, "float32")
    mc_std = np.full(n, np.nan, "float32")
    rng = np.random.default_rng(seed)

    for s0 in range(0, len(idx), batch):
        sel = idx[s0:s0 + batch]
        data = {c: aef_mat[sel, j] for j, c in enumerate(aef_cols)}
        data.update({c: tess_mat[sel, j] for j, c in enumerate(tess_cols)})
        data.update({k: v[sel] for k, v in a_scal.items()})
        data.update({k: v[sel] for k, v in t_scal.items()})
        frame = pd.DataFrame(data)
        frame["aef_present"] = aef_ok[sel].astype("float32")
        has_tess = tess_ok[sel].astype("float32")

        # Deterministic read: gate on wherever Tessera is real.
        frame["tess_present"] = has_tess
        p_fine, p_merged = _mean_probs(models, frame)
        fine_det[sel] = [fine_code.get(c, NODATA)
                         for c in np.array(ref.fine_classes_, dtype=object)[
                             p_fine.argmax(1)]]
        merged_det[sel] = [merged_code.get(c, NODATA)
                           for c in np.array(merged_classes, dtype=object)[
                               p_merged.argmax(1)]]

        # AlphaEarth-only counterfactual on the identical pixels.
        frame["tess_present"] = np.zeros(len(sel), "float32")
        _, p_aef = _mean_probs(models, frame)
        merged_aef[sel] = [merged_code.get(c, NODATA)
                           for c in np.array(merged_classes, dtype=object)[
                               p_aef.argmax(1)]]

        frame["tess_present"] = has_tess
        acc = np.zeros_like(p_merged)
        chg_passes = np.empty((passes, len(sel)), "float32")
        for k in range(passes):
            keep = rng.random(len(sel)) >= 0.5
            frame["tess_present"] = has_tess * keep
            _, pm = _mean_probs(models, frame)
            acc += pm
            chg_passes[k] = pm[:, is_chg].sum(1)
        pm_mean = acc / passes
        merged_mc[sel] = [merged_code.get(c, NODATA)
                          for c in np.array(merged_classes, dtype=object)[
                              pm_mean.argmax(1)]]
        p_change[sel] = pm_mean[:, is_chg].sum(1)
        mc_std[sel] = chg_passes.std(0)

    return {
        "fine_det": fine_det, "merged_det": merged_det, "merged_mc": merged_mc,
        "merged_aef": merged_aef, "p_change": p_change, "mc_std": mc_std,
        "n_valid": int(valid.sum()), "n_tess": int((tess_ok & valid).sum()),
        "tess_ok": tess_ok & valid,
    }


def change_codes(merged, merged_classes):
    """0 stable / 1 change / NODATA, from merged2 class codes."""
    out = np.full_like(merged, NODATA)
    ok = merged != NODATA
    out[ok] = np.array([1 if is_change(merged_classes[c], RARE_LABEL) else 0
                        for c in merged[ok]], "uint8")
    return out


def bin_uncertainty(mc_std):
    """MC disagreement binned into the four legend classes (NODATA preserved)."""
    codes = np.full(mc_std.shape, NODATA, "uint8")
    ok = np.isfinite(mc_std)
    codes[ok] = np.digitize(mc_std[ok], UNCERTAINTY_BINS).astype("uint8")
    return codes


# --------------------------------------------------------------------------
async def infer_aoi(name, bbox, index, reader, models, groups, args, out_dir):
    aef_cols, tess_cols = groups
    probe = await index.query(bbox=bbox, years=YEARS[0])
    epsg = probe[0].crs_epsg
    geobox = aoi_geobox(bbox, crs=f"EPSG:{epsg}", resolution=args.resolution,
                        bbox_crs="EPSG:4326")
    H, W = int(geobox.shape.y), int(geobox.shape.x)
    print(f"\n[{name}] grid {H}x{W} @ {args.resolution} m EPSG:{epsg}", flush=True)

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

    out = predict_mc(models, aef_mat, tess_mat, aef_cols, tess_cols,
                     args.batch, args.mc_passes, args.seed)
    del aef_mat, tess_mat

    merged_classes = list(models[0].merged_classes_)
    vdir = out_dir / name
    vdir.mkdir(parents=True, exist_ok=True)

    write_class_raster(vdir / f"{name}_mc_merged2.tif",
                       out["merged_mc"].reshape(H, W), geobox, merged_classes,
                       MERGED_COLORS)
    write_class_raster(vdir / f"{name}_det_merged2.tif",
                       out["merged_det"].reshape(H, W), geobox, merged_classes,
                       MERGED_COLORS)
    write_class_raster(vdir / f"{name}_det_coarse3.tif",
                       out["fine_det"].reshape(H, W), geobox,
                       list(models[0].fine_classes_), CLASS_COLORS)

    chg_mc = change_codes(out["merged_mc"], merged_classes)
    chg_det = change_codes(out["merged_det"], merged_classes)
    chg_aef = change_codes(out["merged_aef"], merged_classes)
    for tag, chg in (("mc", chg_mc), ("det", chg_det), ("aefonly", chg_aef)):
        write_class_raster(vdir / f"{name}_{tag}_change.tif", chg.reshape(H, W),
                           geobox, ["stable", "change"],
                           {"stable": (200, 200, 200), "change": (214, 39, 40)})
    write_class_raster(vdir / f"{name}_mc_uncertainty.tif",
                       bin_uncertainty(out["mc_std"]).reshape(H, W), geobox,
                       list(UNCERTAINTY_COLORS), UNCERTAINTY_COLORS)

    n_valid = out["n_valid"]
    n_mc = int((chg_mc == 1).sum())
    n_det = int((chg_det == 1).sum())
    n_aef = int((chg_aef == 1).sum())
    flipped = int(((chg_mc != chg_det) & (chg_mc != NODATA)
                   & (chg_det != NODATA)).sum())

    # The omission read, restricted to pixels where the gate can actually fire:
    # elsewhere the two views are the same model by construction.
    tok = out["tess_ok"]
    n_det_t = int((chg_det[tok] == 1).sum())
    n_aef_t = int((chg_aef[tok] == 1).sum())
    omission = {
        "tessera_pixels": int(tok.sum()),
        "change_with_tessera": n_det_t,
        "change_without_tessera": n_aef_t,
        "relative_change": round((n_det_t - n_aef_t) / max(n_aef_t, 1), 4),
        "turned_off_by_tessera": int(((chg_aef == 1) & (chg_det == 0) & tok).sum()),
        "turned_on_by_tessera": int(((chg_aef == 0) & (chg_det == 1) & tok).sum()),
    }
    if tok.any():
        print(f"[{name}] omission check on {tok.sum():,} Tessera pixels: "
              f"change {n_aef_t:,} without -> {n_det_t:,} with "
              f"({omission['relative_change']:+.1%}); "
              f"{omission['turned_off_by_tessera']:,} px turned off, "
              f"{omission['turned_on_by_tessera']:,} turned on", flush=True)
    std_ok = np.isfinite(out["mc_std"])
    print(f"[{name}] valid={n_valid:,} tessera={out['n_tess']:,} | "
          f"change mc={n_mc:,} ({100 * n_mc / max(n_valid, 1):.1f}%) "
          f"det={n_det:,} ({100 * n_det / max(n_valid, 1):.1f}%) | "
          f"MC flipped {flipped:,} px", flush=True)

    return {
        "aoi": name, "bbox_wgs84": list(bbox), "epsg": int(epsg),
        "resolution_m": args.resolution, "grid": [H, W],
        "n_valid_pixels": n_valid, "n_tessera_pixels": out["n_tess"],
        "tessera_pixel_coverage": {"2018": round(cov18, 4), "2024": round(cov24, 4)},
        "change_pixels": {"mc": n_mc, "deterministic": n_det, "aef_only": n_aef},
        "change_fraction": {"mc": round(n_mc / max(n_valid, 1), 4),
                            "deterministic": round(n_det / max(n_valid, 1), 4),
                            "aef_only": round(n_aef / max(n_valid, 1), 4)},
        "tessera_omission": omission,
        "pixels_flipped_by_mc": flipped,
        "mc_std_mean": round(float(np.nanmean(out["mc_std"][std_ok])), 4)
        if std_ok.any() else None,
    }


async def main_async(args) -> None:
    # The seed ensemble (backlog G1). Averaging the seeds' posteriors is worth
    # +0.005 change-F1 / +0.004 macro-F1 / +0.007 stable-Artificial recall on the
    # plot read, and costs only extra forward passes -- the AlphaEarth and
    # Tessera rasters are fetched once regardless. --seeds 1 is the old behaviour.
    models = []
    for k in range(args.seeds):
        model, aef_cols, tess_cols, a_names, t_names = fit_model(
            args.tessera, args.seed + k)
        models.append(model)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"best_{stamp}"
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
                name, CITY_AOIS[name], index, reader, models,
                (aef_cols, tess_cols), args, out_dir))

    (out_dir / "inference_summary.json").write_text(json.dumps({
        "model": "mc_dropout_scalars -- symmetric two-tower, Tessera-tower "
                 "dropout 0.7 vs AlphaEarth 0.4, per-modality change scalars, "
                 f"Monte-Carlo modality dropout over {args.mc_passes} passes"
                 + (f", posteriors averaged over {args.seeds} torch seeds"
                    if args.seeds > 1 else ""),
        "recipe": {**RECIPE, "n_aef_features": len(aef_cols) + len(a_names),
                   "n_tess_features": len(tess_cols) + len(t_names),
                   "mc_passes": args.mc_passes, "n_seeds": args.seeds},
        "validation": {
            "metric": "merged2 change-F1, 5-fold spatially blocked CV, 5 seeds",
            "mc_dropout_scalars": 0.6704,
            "symmetric_two_tower_previous_deploy": 0.6594,
            "alphaearth_only": 0.6577,
        },
        "outputs": {
            "mc_merged2": "deployed Vegetation/Artificial map (MC-averaged)",
            "det_merged2": "same model, gate always on -- the comparison read",
            "det_coarse3": "9-transition informative read",
            "mc_change / det_change": "change mask (0 stable, 1 change)",
            "mc_uncertainty": "std of P(change) across MC passes, binned -- "
                              "high = the call depends on trusting Tessera",
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
    parser.add_argument("--mc-passes", type=int, default=16,
                        help="Monte-Carlo modality-dropout passes per pixel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5,
                        help="how many torch seeds to average (G1); 1 = single model")
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("inference"))
    parser.add_argument("--manifest-cache", type=Path,
                        default=project_data_dir("inference", "manifest_cache"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
