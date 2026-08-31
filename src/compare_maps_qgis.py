"""Assemble two or more transition maps into one folder you can open and judge.

This project settles model choices on the **user's visual read of the map**
(CLAUDE.md), and the quantitative metrics genuinely tie at the point where that
choice gets made -- S18 established that, and S19 re-established it. So the
deliverable that actually decides anything is a folder QGIS opens with the layers
already in the order you would add them, not another table.

What it writes, numbered so the QGIS browser sorts it into that order:

``00_``  the Sentinel-2 backdrop, if one is given -- true colour and NIR, so the
         classes are read against the imagery rather than against a mental model
``10_``  each map, copied with **its own** ``.qml``. Copied, not re-derived: the
         codes follow the sorted class list and only that raster's sidecar knows
         what code 2 means. Getting this wrong once counted stable Vegetation as
         change.
``20_``  a delta raster per non-reference map: change **both** maps found, change
         only this map found (added), change only the reference found (removed),
         and stable everywhere else. Two change masks side by side answer "is it
         bigger"; the delta answers "bigger *where*", which is the question a
         change-pixel count cannot reach.

The delta's palette is deliberately NOT the merged2 palette -- these four
categories are not land-cover classes, and colouring them like classes invites
reading the delta as a map.

Usage::

    G=/home/geethen.singh/.pixi/envs/geo
    PROJ_DATA=$G/share/proj GDAL_DATA=$G/share/gdal $G/bin/python \\
        src/compare_maps_qgis.py \\
            --maps data/inference/<run>/oslo_*_merged2.tif \\
            --reference oslo_s2off_centre_m3s3_bf_merged2.tif \\
            --backdrop data/inference/s2_backdrop/<run>/oslo_s2_rgb_2024.tif \\
            --out data/inference/oslo_compare

**Read the counts against the seed floor, not against zero.** A 5-seed ensemble
reproduces itself at only ~0.84 change-class IoU across disjoint seed draws, so
run `compare_map_iou.py` on two seed blocks of the *same* model before reading
any delta here as a real difference.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_map_iou import NODATA, qml_labels  # noqa: E402
from phoenix_delta_map import write_paletted  # noqa: E402

DELTA_CLASSES = ("stable (both)", "change (both)", "added", "removed")
DELTA_COLORS = {
    "stable (both)": (235, 235, 235),
    "change (both)": (150, 30, 30),
    "added": (255, 140, 0),      # this map finds change the reference does not
    "removed": (60, 120, 220),   # the reference finds change this map does not
}


def change_mask(path: Path):
    """(is-change, is-valid, profile) for a transition raster, via its sidecar.

    A transition map has no "change" code: change is every class whose two
    states differ, and which codes those are is a property of the sorted class
    list in the `.qml`. Deriving it any other way is the palette-order trap.
    """
    labels = qml_labels(path)
    if not labels:
        raise SystemExit(f"{path.name} has no .qml sidecar; codes are unreadable")
    codes = [c for c, lab in labels.items()
             if " -> " in lab and lab.split(" -> ")[0] != lab.split(" -> ")[-1]]
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile
        nodata = src.nodata if src.nodata is not None else NODATA
    return np.isin(arr, codes), arr != nodata, profile


def copy_with_sidecar(src: Path, dest: Path) -> None:
    shutil.copyfile(src, dest)
    qml = src.with_suffix(".qml")
    if qml.exists():
        shutil.copyfile(qml, dest.with_suffix(".qml"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, nargs="+", required=True)
    ap.add_argument("--reference", default=None,
                    help="filename of the map every delta is taken against "
                         "(default: the first --maps entry)")
    ap.add_argument("--backdrop", type=Path, nargs="*", default=[],
                    help="imagery on the same geobox, copied in as layer 00")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    maps = list(args.maps)
    ref = next((p for p in maps if p.name == args.reference), maps[0])
    others = [p for p in maps if p != ref]
    args.out.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(args.backdrop):
        copy_with_sidecar(img, args.out / f"00_{i}_{img.name}")

    ref_change, ref_valid, profile = change_mask(ref)
    for i, path in enumerate(maps):
        copy_with_sidecar(path, args.out / f"10_{i}_{path.name}")

    print(f"reference: {ref.name}\n")
    print(f'{"map":44s} {"change px":>10s} {"both":>9s} {"added":>8s} {"removed":>8s} {"IoU":>7s}')
    print(f'{ref.name:44s} {int((ref_change & ref_valid).sum()):10,d}')
    for path in others:
        mask, valid, prof = change_mask(path)
        if prof["transform"] != profile["transform"] or mask.shape != ref_change.shape:
            raise SystemExit(f"{path.name} is not on the reference geobox")
        both_valid = valid & ref_valid
        a, b = ref_change & both_valid, mask & both_valid
        delta = np.zeros(mask.shape, "uint8")
        delta[a & b] = DELTA_CLASSES.index("change (both)")
        delta[b & ~a] = DELTA_CLASSES.index("added")
        delta[a & ~b] = DELTA_CLASSES.index("removed")
        delta[~both_valid] = NODATA
        stem = path.name.replace("_merged2.tif", "").replace("_coarse3.tif", "")
        out = args.out / f"20_delta_{stem}_vs_{ref.stem}.tif"
        write_paletted(out, delta, prof, list(DELTA_CLASSES), DELTA_COLORS)
        inter, union = int((a & b).sum()), int((a | b).sum())
        print(f'{path.name:44s} {int(b.sum()):10,d} {inter:9,d} '
              f'{int((b & ~a).sum()):8,d} {int((a & ~b).sum()):8,d} '
              f'{inter / union if union else float("nan"):7.4f}')
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
