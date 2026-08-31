"""Class IoU between two class rasters, and the seed self-IoU floor.

The rule this exists to enforce (CLAUDE.md): **a 5-seed ensemble reproduces
itself at only ~0.84 change-class IoU across disjoint seed draws**, so a
disagreement between two maps means nothing until that floor is computed for
each of them. Doing this wrong in the obvious direction -- comparing model A
against model B across seed blocks without asking what each does against itself
-- produced a false "this subset does not replicate" verdict in S18.

    # the floor: one model, two disjoint seed blocks
    python src/compare_map_iou.py A/oslo_siam_cos_coarse3.tif \\
                                  B/oslo_siam_cos_coarse3.tif

    # the comparison, which is only readable against the floor above
    python src/compare_map_iou.py A/oslo_siam_cos_coarse3.tif \\
                                  A/oslo_s2off_centre_m3s3_bf_coarse3.tif

Class codes follow the *sorted* class list written into the `.qml` sidecar, so
the labels are read from there rather than assumed -- getting that wrong once
counted stable Vegetation as change.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import rasterio

NODATA = 255


def qml_labels(tif: Path) -> dict[int, str]:
    """Code -> label from the raster's `.qml` sidecar, or {} when absent."""
    qml = tif.with_suffix(".qml")
    if not qml.exists():
        return {}
    return {int(v): lab for v, lab in
            re.findall(r'value="(\d+)"[^>]*label="([^"]*)"', qml.read_text())}


def class_iou(a: np.ndarray, b: np.ndarray, code: int) -> tuple[float, int, int]:
    """IoU of one class between two label arrays, plus each side's pixel count."""
    ina, inb = a == code, b == code
    union = int((ina | inb).sum())
    inter = int((ina & inb).sum())
    return (inter / union if union else float("nan"), int(ina.sum()), int(inb.sum()))


def compare(path_a: Path, path_b: Path) -> dict:
    with rasterio.open(path_a) as sa, rasterio.open(path_b) as sb:
        if sa.shape != sb.shape or sa.transform != sb.transform:
            raise SystemExit("rasters are not on the same geobox; refusing to compare")
        a, b = sa.read(1), sb.read(1)
    valid = (a != NODATA) & (b != NODATA)
    a, b = a[valid], b[valid]
    labels = qml_labels(path_a) or qml_labels(path_b)
    codes = sorted(set(np.unique(a)) | set(np.unique(b)))

    print(f"A {path_a.parent.name}/{path_a.name}")
    print(f"B {path_b.parent.name}/{path_b.name}")
    print(f"{valid.sum():,} comparable pixels | overall agreement "
          f"{float((a == b).mean()):.4%}\n")
    print(f"{'class':32s} {'IoU':>7s} {'A px':>10s} {'B px':>10s} {'B/A':>7s}")
    out = {"agreement": float((a == b).mean()), "classes": {}}
    ious = []
    for code in codes:
        iou, na, nb = class_iou(a, b, int(code))
        name = labels.get(int(code), f"code {code}")
        ratio = (nb / na) if na else float("nan")
        print(f"{name:32s} {iou:7.4f} {na:10,d} {nb:10,d} {ratio:7.3f}")
        out["classes"][name] = {"iou": iou, "a_px": na, "b_px": nb}
        if np.isfinite(iou):
            ious.append(iou)
    # Change classes only -- the mean over all nine is dominated by the stable
    # classes that cover ~99% of the AOI and is near 1 for any two maps at all.
    change = [v["iou"] for k, v in out["classes"].items()
              if " -> " in k and k.split(" -> ")[0] != k.split(" -> ")[1]
              and np.isfinite(v["iou"])]
    out["mean_iou"] = float(np.mean(ious)) if ious else float("nan")
    out["mean_change_iou"] = float(np.mean(change)) if change else float("nan")
    print(f"\nmean IoU (all classes)   {out['mean_iou']:.4f}")
    print(f"mean IoU (change only)   {out['mean_change_iou']:.4f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    compare(args.a, args.b)


if __name__ == "__main__":
    main()
