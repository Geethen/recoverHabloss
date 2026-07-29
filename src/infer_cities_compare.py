"""Run the best hierarchical-softmax net over the two cities and compare to LDA.

``infer_cities.py`` mapped the tuned LDA; this maps the leaderboard's best model
(``HierarchicalSoftmaxNN`` arch=wide, loss=focal, 30 epochs -- merged2 change-F1
0.657 vs LDA 0.621) over the *same* Johannesburg / Oslo pixels, so the two are
compared pixel-for-pixel on one AEF fetch. The net trains in ~1 s on all labelled
plots, so it is fit in-process rather than persisted.

For every city it predicts, over the identical valid-pixel set:
* **LDA direct** (reloaded from the persisted artefact used for the existing maps),
* **hier coarse3** (the informative 9-transition read, ``predict``),
* **hier merged2** (the deployed Veg/Artificial read, ``predict_merged``),

writes hier GeoTIFFs alongside the LDA ones, and reports the deploy-relevant
comparison: change fraction per model, pixel agreement, and Cohen's kappa, both at
the coarse3 legend (like-for-like with LDA-direct) and at the merged2 legend
(LDA collapsed via ``to_merged_label`` vs the hier merged read). GPU prediction is
batched so millions of pixels do not blow the device memory.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from aef_loader import AEFIndex, DataSource, VirtualTiffReader, aoi_geobox
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from infer_cities import (
    CITY_AOIS,
    CLASS_COLORS,
    YEARS,
    build_feature_matrix,
    is_change,
    load_year_embeddings,
    write_geotiff,
)
from model_zoo import (
    DEFAULT_INPUT,
    RARE_LABEL,
    HierarchicalSoftmaxNN,
    to_merged_label,
)
from project_paths import project_data_dir

BASE = dict(arch="wide", loss="focal", epochs=30)


def train_best_hier(input_path: Path, min_count: int):
    """Fit the best hierarchical net on every labelled plot (coarse3 target)."""
    frame, columns = build_frame(input_path)
    target = target_for_legend(frame, LEGENDS["coarse3"], min_count)
    model = HierarchicalSoftmaxNN(columns, **BASE)
    model.fit(frame, target.to_numpy())
    print(f"hier fit on {len(frame):,} plots | fine={model.fine_classes_} | "
          f"merged={model.merged_classes_}", flush=True)
    return model, columns


def hier_predict_batched(model, feats: np.ndarray, columns: list[str],
                         change_threshold=None, batch: int = 300_000):
    """Chunked GPU inference -> (coarse3 labels, merged2 labels) over ``feats``.

    ``change_threshold`` sets the merged2 change gate (None = arg-max / implicit
    0.5); see ``HierarchicalSoftmaxNN.merged_labels_from_probs``.
    """
    fine_codes = np.empty(len(feats), dtype=np.int32)
    merged = np.empty(len(feats), dtype=object)
    for start in range(0, len(feats), batch):
        end = min(start + batch, len(feats))
        chunk = pd.DataFrame(feats[start:end], columns=columns)
        p_fine, p_merged = model._probs(chunk)
        fine_codes[start:end] = p_fine.argmax(1)
        merged[start:end] = model.merged_labels_from_probs(p_merged, change_threshold)
    fine = np.array(model.fine_classes_, dtype=object)[fine_codes]
    return fine, merged


def encode_labels(labels, class_code, n_pixels, valid, nodata):
    """Label array over valid pixels -> full-length uint8 code grid."""
    codes = np.full(n_pixels, nodata, dtype="uint8")
    codes[valid] = np.array([class_code.get(p, nodata) for p in labels], dtype="uint8")
    return codes


def agreement(a_change: np.ndarray, b_change: np.ndarray) -> dict:
    """2x2 change agreement + Cohen's kappa between two boolean change masks."""
    n = len(a_change)
    both = int((a_change & b_change).sum())
    neither = int((~a_change & ~b_change).sum())
    a_only = int((a_change & ~b_change).sum())
    b_only = int((~a_change & b_change).sum())
    po = (both + neither) / n
    pa, pb = a_change.mean(), b_change.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    # F1 of b treating a as reference (how much of a's change b recovers).
    prec = both / max(both + b_only, 1)
    rec = both / max(both + a_only, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"both_change": both, "lda_only": a_only, "hier_only": b_only,
            "neither": neither, "overall_agreement": round(po, 4),
            "cohens_kappa": round(kappa, 4), "hier_vs_lda_change_f1": round(f1, 4)}


async def infer_city(name, bbox, index, reader, hier_model, columns, lda_model,
                     lda_classes, resolution, output_dir, change_threshold=None):
    class_code = {c: i for i, c in enumerate(hier_model.fine_classes_)}
    nodata = 255

    probe = await index.query(bbox=bbox, years=YEARS[0])
    epsg = probe[0].crs_epsg
    geobox = aoi_geobox(bbox, crs=f"EPSG:{epsg}", resolution=resolution,
                        bbox_crs="EPSG:4326")
    height, width = geobox.shape.y, geobox.shape.x
    n_pixels = height * width
    print(f"[{name}] grid {height}x{width} in EPSG:{epsg}", flush=True)

    emb_2018 = await load_year_embeddings(index, reader, bbox, YEARS[0], geobox)
    emb_2024 = await load_year_embeddings(index, reader, bbox, YEARS[1], geobox)
    features = build_feature_matrix(emb_2018, emb_2024, columns)
    valid = np.isfinite(features).all(axis=1)
    n_valid = int(valid.sum())
    print(f"[{name}] {n_valid:,}/{n_pixels:,} valid pixels; predicting...", flush=True)

    feats_valid = features[valid]
    lda_fine = lda_model.predict(feats_valid)
    hier_fine, hier_merged = hier_predict_batched(
        hier_model, feats_valid, columns, change_threshold=change_threshold)

    # Change masks (deploy = merged2 sealing; coarse3 = like-for-like with LDA).
    lda_fine_change = np.array([is_change(p, RARE_LABEL) for p in lda_fine])
    hier_fine_change = np.array([is_change(p, RARE_LABEL) for p in hier_fine])
    lda_merged = np.array([to_merged_label(p) for p in lda_fine], dtype=object)
    lda_merged_change = np.array([is_change(p, RARE_LABEL) for p in lda_merged])
    hier_merged_change = np.array([is_change(p, RARE_LABEL) for p in hier_merged])

    def frac(mask):
        return round(100 * float(mask.mean()), 3)

    stats = {
        "city": name, "epsg": int(epsg), "grid": [int(height), int(width)],
        "n_valid_pixels": n_valid, "change_threshold": change_threshold,
        "change_pct": {
            "lda_coarse3": frac(lda_fine_change),
            "hier_coarse3": frac(hier_fine_change),
            "lda_merged2": frac(lda_merged_change),
            "hier_merged2": frac(hier_merged_change),
        },
        "coarse3_label_agreement": round(float((lda_fine == hier_fine).mean()), 4),
        "coarse3_change_agreement": agreement(lda_fine_change, hier_fine_change),
        "merged2_change_agreement": agreement(lda_merged_change, hier_merged_change),
        "hier_coarse3_class_counts": {
            c: int((hier_fine == c).sum()) for c in hier_model.fine_classes_
            if (hier_fine == c).any()
        },
        "hier_merged2_class_counts": {
            c: int((hier_merged == c).sum()) for c in hier_model.merged_classes_
            if (hier_merged == c).any()
        },
    }

    # Write hier GeoTIFFs alongside the LDA ones.
    output_dir.mkdir(parents=True, exist_ok=True)
    class_colormap = {class_code[c]: (*CLASS_COLORS.get(c, (120, 120, 120)), 255)
                      for c in hier_model.fine_classes_}
    class_colormap[nodata] = (0, 0, 0, 0)
    change_colormap = {0: (200, 200, 200, 255), 1: (214, 39, 40, 255),
                       nodata: (0, 0, 0, 0)}

    fine_code = encode_labels(hier_fine, class_code, n_pixels, valid, nodata).reshape(height, width)
    merged_change_full = np.full(n_pixels, nodata, dtype="uint8")
    merged_change_full[valid] = hier_merged_change.astype("uint8")
    lda_merged_change_full = np.full(n_pixels, nodata, dtype="uint8")
    lda_merged_change_full[valid] = lda_merged_change.astype("uint8")

    write_geotiff(output_dir / f"{name}_transition_hier.tif",
                  [("hier coarse3 transition", fine_code)], geobox, nodata, class_colormap)
    write_geotiff(output_dir / f"{name}_change_compare.tif",
                  [("hier merged2 change", merged_change_full.reshape(height, width)),
                   ("lda merged2 change", lda_merged_change_full.reshape(height, width))],
                  geobox, nodata, change_colormap)

    print(f"[{name}] change%%: LDA-merged2={stats['change_pct']['lda_merged2']} "
          f"hier-merged2={stats['change_pct']['hier_merged2']} | "
          f"kappa={stats['merged2_change_agreement']['cohens_kappa']}", flush=True)
    return stats


async def main_async(args):
    hier_model, columns = train_best_hier(args.input, args.min_class_count)
    lda_model = joblib.load(args.lda_path)
    lda_sidecar = json.loads(Path(args.lda_sidecar).read_text())
    lda_classes = lda_sidecar["classes"]
    if list(columns) != list(lda_sidecar["feature_columns"]):
        raise SystemExit("hier and LDA feature-column order differ; cannot align pixels")

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()

    results = []
    async with VirtualTiffReader(manifest_cache_dir=args.manifest_cache) as reader:
        for name in args.cities:
            results.append(await infer_city(
                name, CITY_AOIS[name], index, reader, hier_model, columns,
                lda_model, lda_classes, args.resolution, args.output_dir,
                change_threshold=args.change_threshold,
            ))

    out = args.output_dir / "compare_hier_vs_lda.json"
    out.write_text(json.dumps({
        "hier_model": BASE, "hier_merged2_cv_change_f1": 0.657,
        "hier_change_threshold": args.change_threshold,
        "lda_direct_path": str(args.lda_path), "lda_merged2_cv_change_f1": 0.621,
        "years": list(YEARS), "cities": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


def main():
    models = project_data_dir("models")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lda-path", type=Path,
                        default=models / "transition_lda_direct.joblib")
    parser.add_argument("--lda-sidecar", type=Path,
                        default=models / "transition_lda_direct.json")
    parser.add_argument("--cities", nargs="+", default=list(CITY_AOIS))
    parser.add_argument("--change-threshold", type=float, default=None,
                        help="Merged2 change gate P(change)>=t (default: arg-max / "
                             "implicit 0.5). Tune with experiment_hier_change_recall.py "
                             "-- t~0.45 for change-F1, t~0.30 for recall/efficiency.")
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=project_data_dir("inference"))
    parser.add_argument("--manifest-cache", type=Path,
                        default=project_data_dir("inference", "manifest_cache"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
