"""Choose a reportable Sentinel-2 feature set on map evidence, not on taste.

The deployed detail tower reads 204 engineered Sentinel-2 columns. On plot
metrics that block is worth almost nothing -- `optimise_s2off.py` showed every
subset of it ties the whole within +/-0.0005 change-F1 -- but the 15-column
`s2off_slim` map visibly loses structure (edge density 0.0902 -> 0.0788 at an
unchanged 6 px median segment). So plot metrics cannot make this choice: they are
blind to the thing that differs.

Two label-free readings can, and this computes both for every named subset in
``twotower_lab.S2_SUBSETS``:

``detail``  `map_detail_metrics`. Edge density says how much structure survives;
            boundary alignment (Sentinel-2 gradient on class boundaries, over the
            scene mean) says whether that structure falls on real image edges.
            Read as a pair -- a subset that raises the first and drops the second
            is manufacturing noise, which is the failure mode U1 and the
            deterministic gate both produced.

``IoU``     per class against the **full-204 map**, which is the incumbent and
            therefore the thing a reviewer will ask about. Not accuracy: no
            labelled plot falls inside the AOI (G4). It is agreement, and it is
            the honest way to say "this smaller feature set reproduces the map we
            already validated" -- with the change class called out separately,
            because it is 0.5% of the surface and a mean IoU would hide it
            entirely.

The selection rule, fixed before the numbers were looked at
----------------------------------------------------------
Take the **smallest** subset that (a) holds change-class IoU against the full
block, (b) does not lose edge density, and (c) does not lose boundary alignment.
Size is the tie-break because size is the reporting cost -- the whole reason this
runs. A subset that wins on detail while disagreeing with the incumbent on the
change class is not a candidate at any size.

Usage::

    python infer_s2.py --aois oslo --models s2off_full s2off_centre_s3_bf ... --seeds 5
    python compare_s2_subsets.py data/inference/s2_<stamp>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import map_detail_metrics as MDM
from project_paths import project_data_dir
from twotower_lab import S2_SUBSET_DESC, S2_SUBSETS


def _read(path: Path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1), src.nodata


def class_labels(path: Path) -> list[str]:
    """Class names in code order, read back from the sidecar QML.

    The writer sorts classes before assigning codes, so the palette's insertion
    order is not the code order -- reading the QML is the only safe way to know
    what code 2 means, and getting it wrong would silently compare two different
    classes.
    """
    qml = path.with_suffix(".qml")
    return re.findall(r'label="([^"]+)"', qml.read_text()) if qml.exists() else []


def iou_against(ref: np.ndarray, other: np.ndarray, valid: np.ndarray,
                labels: list[str]) -> dict:
    """Per-class IoU plus the two aggregates worth quoting."""
    out = {}
    ious = []
    for code, name in enumerate(labels):
        a, b = (ref == code) & valid, (other == code) & valid
        union = int((a | b).sum())
        iou = float((a & b).sum() / union) if union else float("nan")
        out[f"iou_{name}"] = iou
        if union:
            ious.append(iou)
    out["iou_mean"] = float(np.mean(ious)) if ious else float("nan")
    out["agreement"] = float((ref[valid] == other[valid]).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="an infer_s2.py output directory")
    ap.add_argument("--aoi", default="oslo")
    ap.add_argument("--reference", default="s2off_full",
                    help="the map every other one is scored against")
    ap.add_argument("--nir", type=Path, default=None,
                    help="Sentinel-2 NIR GeoTIFF on the same geobox, for "
                         "boundary_align; without it that column is NaN and the "
                         "'is the structure real' half of the read is missing")
    ap.add_argument("--out", type=Path,
                    default=project_data_dir("analysis_results"))
    args = ap.parse_args()

    maps = sorted(args.run_dir.glob(f"{args.aoi}_*_merged2.tif"))
    models = [p.name[len(args.aoi) + 1:-len("_merged2.tif")] for p in maps]
    if args.reference not in models:
        raise SystemExit(f"reference {args.reference} not in {models}")
    print(f"{len(models)} maps in {args.run_dir.name}, "
          f"reference {args.reference}\n", flush=True)

    ref_path = args.run_dir / f"{args.aoi}_{args.reference}_merged2.tif"
    ref, nodata = _read(ref_path)
    labels = class_labels(ref_path)
    change_labels = [c for c in labels
                     if "->" in c and c.split(" -> ")[0] != c.split(" -> ")[-1]]

    rows = []
    for model in models:
        path = args.run_dir / f"{args.aoi}_{model}_merged2.tif"
        arr, nd = _read(path)
        if class_labels(path) != labels:
            raise SystemExit(f"{model} has a different class order than the "
                             f"reference; codes are not comparable")
        valid = (arr != nd) & (ref != nodata)

        row = {"model": model,
               "n_s2_cols": np.nan,
               "detail_columns": S2_SUBSET_DESC.get(
                   model.replace("s2off_", ""), "")}
        row.update(iou_against(ref, arr, valid, labels))
        # The change class is the reason this project exists and is 0.5% of the
        # surface, so it gets its own number rather than being averaged away.
        a = np.isin(ref, [labels.index(c) for c in change_labels]) & valid
        b = np.isin(arr, [labels.index(c) for c in change_labels]) & valid
        union = int((a | b).sum())
        row["iou_change"] = float((a & b).sum() / union) if union else float("nan")
        row["change_px"] = int(b.sum())

        row.update({k: v for k, v in
                    MDM.metrics_for(path, args.nir).items()
                    if k in ("edge_density", "segments_per_mp",
                             "median_segment_px", "hf_power_ratio",
                             "boundary_align")})
        rows.append(row)
        print(f"  {model:26s} scored", flush=True)

    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "s2_subset_map_comparison.csv"
    frame.to_csv(dest, index=False)

    show = ["model", "iou_change", "iou_mean", "agreement", "change_px",
            "edge_density", "segments_per_mp", "median_segment_px",
            "boundary_align"]
    print("\n" + frame[show].to_string(index=False,
                                       float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
