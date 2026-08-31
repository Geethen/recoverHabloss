"""Map the deployed model over a global patch sample, and score each patch.

This is `infer_s2.run_aoi` reduced to what a 5x5 km patch needs and extended by
the two per-patch quantities the sampling design turns on. It runs **the
deployed recipe and nothing else** -- `s2off_centre_m3s3_bf`, AlphaEarth-only at
inference with the detail gate off (CLAUDE.md) -- so a patch's class map here is
the same map the AOI runs produce, at the same 10 m grid.

Because the deployed read never touches Sentinel-2, a patch costs exactly two
AlphaEarth reads and one forward pass per ensemble member. There is no composite
fetch, no sliding window, no self-check to run: the whole reason 100 patches is
affordable at all is the serving choice made in S16.

What comes out
--------------
Per patch, two rasters (`*_merged2.tif`, `*_coarse3.tif`, paletted uint8 + .qml,
exactly as the AOI maps) and one row of statistics:

``px_<class>``
    Pixel count per coarse3 class. This is the sampling design's currency: the
    proportion of the map a class occupies is what decides how many patches must
    be drawn to find enough of it to label.

``entropy_mean`` / ``entropy_change``
    Mean Shannon entropy of the nine-class posterior over valid pixels, in nats
    normalised by ``log 9`` so 0 is a one-hot map and 1 is uniform. The second
    restricts to pixels the map calls change, which is the population the
    labelling effort is actually about -- a patch can be confidently stable
    everywhere and still be worthless to label.

``novelty_mean`` / ``novelty_p90``
    Cosine distance from each pixel's AlphaEarth vector to its nearest of the
    6,492 *training* plots, in the concatenated 2018+2024 space. This is the
    "have we seen anything like this before" axis, and it is computed against
    the label set rather than against the rest of the sample on purpose: a patch
    that is unremarkable globally but unlike anything labelled is precisely the
    patch worth labelling. AlphaEarth embeddings are unit-norm per year, so
    cosine is the metric the representation is built for.

Entropy and novelty answer different questions and are kept apart rather than
blended into a score. Entropy is "the model is unsure here", which finds
boundaries and mixed pixels; novelty is "the model has no basis to be sure
here", which finds unrepresented biomes and land-use systems. Ranking is
`plan_patch_sampling.py`'s job, and it needs both columns to do it.

Run
---
    python src/infer_patches.py                     # all patches in the manifest
    python src/infer_patches.py --limit 4           # smoke test
    python src/infer_patches.py --rasters coarse3   # nine-transition map only
    python src/infer_patches.py --model siam_s2off_state_pre \
        --rasters coarse3_gated                     # the rare-class read
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from global_patches import RESOLUTION, YEARS, patch_geometry
from infer_cities import CLASS_COLORS, load_year_embeddings
from infer_twotower import MERGED_COLORS, NODATA, write_class_raster
from model_zoo import DEFAULT_INPUT
from project_paths import project_data_dir

#: The deployed recipe, and the default. One model per run, no comparison --
#: see CLAUDE.md. `--model` exists because the patch sample is a *labelling*
#: instrument rather than a product: the two dead coarse3 classes are dead on
#: the deployed map (`Artificial -> Cropland` takes 1 pixel in 21 million), so a
#: round aimed at them has to be navigated by a recipe that surfaces them --
#: `siam_s2off_state_pre` read through its cost gate. That is a choice about
#: where to send labellers, not a change of deployed model.
MODEL = "s2off_centre_m3s3_bf"
#: Pixels subsampled per patch for the novelty distance. The nearest-neighbour
#: distance is an average over a 250k-pixel patch; 4k draws pin its mean to
#: about +/-0.003 and cost one 4k x 6.5k matmul.
NOVELTY_SAMPLE = 4_000
#: Raster products a patch can emit, selectable with ``--rasters``. All four
#: come off the same forward pass, so choosing a subset saves disk and nothing
#: else -- the statistics columns are computed either way. ``coarse3_gated`` is
#: available only for a recipe that ships a cost vector (`c3_costs`).
RASTERS = ("merged2", "coarse3", "coarse3_gated", "topchange")


# --------------------------------------------------------------------------
# Novelty reference: the labelled plots, in AlphaEarth space
# --------------------------------------------------------------------------
def novelty_reference(input_path: Path = DEFAULT_INPUT):
    """``(n_plots, 128)`` unit-norm 2018+2024 AlphaEarth matrix for the labels.

    Rows with any missing band are dropped rather than imputed: this matrix is
    the definition of "already represented", and an imputed row would claim
    representation the label set does not have.
    """
    frame = pd.read_parquet(input_path)
    cols = (sorted(c for c in frame.columns
                   if c.startswith("A") and c.endswith("_2018"))
            + sorted(c for c in frame.columns
                     if c.startswith("A") and c.endswith("_2024")))
    ref = frame[cols].to_numpy("float32")
    ref = ref[np.isfinite(ref).all(1)]
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    return np.ascontiguousarray(ref.T), cols  # (128, n) for a right-multiply


def novelty(emb18, emb24, valid, ref_t, rng, n_sample=NOVELTY_SAMPLE):
    """Mean and 90th-percentile nearest-label cosine distance over the patch.

    Both are reported because they say different things about a patch: the mean
    is how unfamiliar the patch is on average, the p90 is whether it contains a
    *pocket* of something unseen. A patch of ordinary farmland around one
    unrepresented wetland scores low on the first and high on the second, and it
    is the second that is worth a labeller's afternoon.
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return float("nan"), float("nan")
    if idx.size > n_sample:
        idx = rng.choice(idx, n_sample, replace=False)
    q = np.concatenate([emb18.reshape(emb18.shape[0], -1)[:, idx],
                        emb24.reshape(emb24.shape[0], -1)[:, idx]]).T
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # Nearest neighbour by maximum cosine similarity; distance = 1 - that.
    sim = (q @ ref_t).max(1)
    dist = 1.0 - sim
    return float(dist.mean()), float(np.percentile(dist, 90))


def normalised_entropy(probs):
    """Shannon entropy per pixel, divided by ``log(n_classes)``.

    Normalising makes the merged2 (4-class) and coarse3 (9-class) numbers
    comparable to each other and to 1 = "no information at all", which is what a
    ranking threshold has to be expressed against.
    """
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1) / np.log(probs.shape[1])


# --------------------------------------------------------------------------
async def run_patch(row, index, reader, entry, aef_cols, ref_t, out_dir,
                    batch, seed, rng, rasters=RASTERS, costs=None):
    from infer_s2 import predict, stack_aef_bands

    pid = row.patch_id
    _, wgs_bbox, geobox = patch_geometry(row.lon, row.lat, int(row.epsg))
    height, width = geobox.shape.y, geobox.shape.x
    n_px = height * width

    t0 = time.time()
    emb = {}
    for year in YEARS:
        emb[year] = await load_year_embeddings(index, reader, wgs_bbox, year,
                                               geobox)
    t_fetch = time.time() - t0

    aef_bands = stack_aef_bands(emb[YEARS[0]], emb[YEARS[1]], aef_cols)
    aef_valid = np.isfinite(aef_bands).all(0)
    n_valid = int(aef_valid.sum())

    stats = {
        "patch_id": pid, "lon": float(row.lon), "lat": float(row.lat),
        "epsg": int(row.epsg), "n_px": n_px, "valid_px": n_valid,
        "valid_frac": n_valid / n_px, "fetch_s": round(t_fetch, 1),
    }
    if n_valid == 0:
        # AEF served the tile but every pixel is nodata -- open water inside a
        # coastal tile's rectangular footprint. Kept in the frame with zero
        # counts rather than dropped, because dropping it would inflate every
        # class proportion by shrinking the denominator.
        stats.update({"predict_s": 0.0, "entropy_mean": float("nan"),
                      "entropy_change": float("nan"),
                      "novelty_mean": float("nan"),
                      "novelty_p90": float("nan"), "change_px": 0})
        print(f"[{pid}] no valid AlphaEarth pixels (water) "
              f"[fetch {t_fetch:.0f}s]", flush=True)
        return stats, None, None

    t1 = time.time()
    views, classes, fine_classes = predict(
        entry, aef_bands, np.zeros((n_px, 0), "float32"),
        np.zeros(n_px, bool), batch, seed)
    probs, fine_probs = views["on"]
    t_pred = time.time() - t1

    merged_idx = probs.argmax(1)
    fine_idx = fine_probs.argmax(1)
    # O3's coarse3 decision-cost gate, when the recipe ships one. It re-reads
    # the coarse3 arg-max through a per-class multiplier and touches nothing
    # else -- merged2, the entropy and the novelty columns are all identical
    # with it and without it. Counted separately (`pxg_`) rather than replacing
    # `px_`, because the whole point of the gate here is that the two reads
    # disagree about the rare classes and the sampling design needs to see by
    # how much.
    gated_idx = (fine_probs * costs).argmax(1) if costs is not None else fine_idx
    ent = normalised_entropy(fine_probs)

    # The change-restricted arg-max. Two coarse3 classes are dead on the map --
    # `Artificial -> Cropland` takes 1 pixel in 25 million and `Cropland ->
    # Nature` about 100 -- so a labelling round that navigates by arg-max can
    # never be sent to look at them, and they are exactly the two classes whose
    # OOF F1 is ~0. Restricting the arg-max to the six change classes drops the
    # three stable classes that swamp them and asks a different, answerable
    # question: *given* that something changed here, which change is it most
    # like? That is the retrieval channel for a class the unrestricted map
    # cannot surface, and it costs one extra arg-max over a 9-column posterior.
    #
    # It is a candidate generator, not a map. Most of its pixels are stable
    # ground and the labeller will say so; its value is that its errors are at
    # least errors of the right kind.
    change_cols = np.array(
        [i for i, c in enumerate(fine_classes)
         if str(c).split(" -> ")[0] != str(c).split(" -> ")[-1]])
    top_change = change_cols[fine_probs[:, change_cols].argmax(1)]

    is_change = np.array(
        ["->" in str(c) and str(c).split(" -> ")[0] != str(c).split(" -> ")[-1]
         for c in classes])
    change_mask = is_change[merged_idx] & aef_valid

    nov_mean, nov_p90 = novelty(emb[YEARS[0]], emb[YEARS[1]], aef_valid,
                                ref_t, rng)
    counts = np.bincount(fine_idx[aef_valid], minlength=len(fine_classes))
    stats.update({
        "predict_s": round(t_pred, 1),
        "change_px": int(change_mask.sum()),
        "change_frac": float(change_mask.sum() / n_valid),
        "entropy_mean": float(ent[aef_valid].mean()),
        "entropy_change": (float(ent[change_mask].mean())
                           if change_mask.any() else float("nan")),
        "novelty_mean": nov_mean, "novelty_p90": nov_p90,
    })
    tc_counts = np.bincount(top_change[aef_valid], minlength=len(fine_classes))
    g_counts = np.bincount(gated_idx[aef_valid], minlength=len(fine_classes))
    if costs is not None:
        stats["gate_moved_px"] = int((gated_idx != fine_idx)[aef_valid].sum())
    for i, cls in enumerate(fine_classes):
        stats[f"px_{cls}"] = int(counts[i])
        if costs is not None:
            stats[f"pxg_{cls}"] = int(g_counts[i])
        # Pixels where `cls` is the best of the change classes, and how high
        # its posterior gets anywhere in the patch. The second is the one that
        # says whether a dead class is *absent* from this patch or merely
        # outvoted: a 99.9th-percentile posterior of 0.30 is a real candidate,
        # 0.001 is not.
        stats[f"pxtc_{cls}"] = int(tc_counts[i])
        stats[f"p999_{cls}"] = float(
            np.percentile(fine_probs[aef_valid, i], 99.9))

    if "merged2" in rasters:
        codes = np.where(aef_valid, merged_idx, NODATA).astype("uint8")
        write_class_raster(out_dir / f"{pid}_merged2.tif",
                           codes.reshape(height, width), geobox,
                           list(classes), MERGED_COLORS)
    if "coarse3" in rasters:
        fine_codes = np.where(aef_valid, fine_idx, NODATA).astype("uint8")
        write_class_raster(out_dir / f"{pid}_coarse3.tif",
                           fine_codes.reshape(height, width), geobox,
                           list(fine_classes), CLASS_COLORS)
    if "coarse3_gated" in rasters and costs is not None:
        # Same class list, same codes and the same .qml as `_coarse3.tif`, so
        # the pair is an exact counterfactual for what the gate did here.
        g_codes = np.where(aef_valid, gated_idx, NODATA).astype("uint8")
        write_class_raster(out_dir / f"{pid}_coarse3_gated.tif",
                           g_codes.reshape(height, width), geobox,
                           list(fine_classes), CLASS_COLORS)
    if "topchange" in rasters:
        # Same class list and therefore the same codes and the same .qml, so
        # the two rasters are directly comparable in QGIS. This one is the
        # navigation layer for the rare classes, not a product.
        tc_codes = np.where(aef_valid, top_change, NODATA).astype("uint8")
        write_class_raster(out_dir / f"{pid}_topchange.tif",
                           tc_codes.reshape(height, width), geobox,
                           list(fine_classes), CLASS_COLORS)

    print(f"[{pid}] {row.lat:+7.2f},{row.lon:+8.2f}  valid {n_valid/n_px:5.1%}  "
          f"change {stats['change_frac']:6.2%}  H {stats['entropy_mean']:.3f}  "
          f"nov {nov_mean:.3f}  [fetch {t_fetch:.0f}s pred {t_pred:.0f}s]",
          flush=True)
    return stats, list(classes), list(fine_classes)


async def main_async(args):
    from aef_loader import AEFIndex, DataSource, VirtualTiffReader
    from infer_s2 import _coarse3_costs, fit_models

    patches = pd.read_parquet(args.patches)
    if args.limit:
        patches = patches.head(args.limit)
    print(f"{len(patches)} patches from {args.patches}", flush=True)

    models, aef_cols, *_ = fit_models(
        args.s2, args.seed, [args.model], args.seeds, args.model_cache_dir,
        not args.no_model_cache)
    entry = models[args.model]
    ref_t, _ = novelty_reference()
    print(f"novelty reference: {ref_t.shape[1]} labelled plots", flush=True)

    rasters = () if args.no_rasters else tuple(args.rasters)
    # The cost vector is resolved once, before any patch runs, so a recipe that
    # has none fails here rather than 90 patches into a 12-minute run.
    costs = _coarse3_costs(entry["spec"], list(entry["models"][0].fine_classes_))
    if "coarse3_gated" in rasters and costs is None:
        raise SystemExit(f"--rasters coarse3_gated, but {args.model} ships no "
                         f"coarse3 cost vector (no `c3_costs` in its recipe)")
    print(f"model: {args.model}  gate: "
          f"{'on' if costs is not None else 'none'}\n"
          f"rasters: {', '.join(rasters) if rasters else 'none'}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"patches_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()

    rows, classes, fine_classes = [], None, None
    t0 = time.time()
    async with VirtualTiffReader(manifest_cache_dir=args.manifest_cache) as reader:
        for row in patches.itertuples():
            try:
                stats, cls, fine = await run_patch(
                    row, index, reader, entry, aef_cols, ref_t, out_dir,
                    args.batch, args.seed, rng, rasters, costs)
            except Exception as exc:                     # noqa: BLE001
                # One unreadable tile must not cost the other 99 patches. The
                # failure is recorded as a row so the sampling design can see
                # its own realised sample size rather than assuming 100.
                print(f"[{row.patch_id}] FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                rows.append({"patch_id": row.patch_id, "lon": float(row.lon),
                             "lat": float(row.lat), "epsg": int(row.epsg),
                             "error": f"{type(exc).__name__}: {exc}"})
                continue
            rows.append(stats)
            classes = classes or cls
            fine_classes = fine_classes or fine

    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "patch_stats.parquet", index=False)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "model": args.model, "seeds": args.seeds, "seed": args.seed,
        "coarse3_gate": entry["spec"].get("c3_costs"),
        "rasters": list(rasters),
        "resolution": RESOLUTION, "years": list(YEARS),
        "patches": str(args.patches),
        "n_patches": int(len(frame)),
        "n_failed": int(frame["error"].notna().sum()
                        if "error" in frame else 0),
        "merged2_classes": classes, "coarse3_classes": fine_classes,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(frame)} patches in {time.time() - t0:.0f}s\n-> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path,
                        default=project_data_dir("patches", "patches.parquet"))
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N patches (smoke test)")
    parser.add_argument("--model", default=MODEL,
                        help=f"recipe from infer_s2.fit_models (default {MODEL})")
    parser.add_argument("--s2", type=Path,
                        default=project_data_dir(
                            "embeddings", "s2_features_habloss_recover_10m.parquet"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5,
                        help="seed-ensemble size; 5 is the deployed setting")
    parser.add_argument("--batch", type=int, default=200_000)
    parser.add_argument("--rasters", nargs="+", choices=RASTERS,
                        default=list(RASTERS),
                        help="which GeoTIFFs to write (default: all three)")
    parser.add_argument("--no-rasters", action="store_true",
                        help="statistics only, no GeoTIFFs")
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("patches"))
    parser.add_argument("--manifest-cache", type=Path,
                        default=project_data_dir("inference", "manifest_cache"))
    parser.add_argument("--model-cache-dir", type=Path,
                        default=project_data_dir("models", "s2_ensembles"))
    parser.add_argument("--no-model-cache", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
