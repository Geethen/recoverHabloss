"""Turn a Phoenix run into layers you can actually look at.

`phoenix_refine_map.py` writes a refined change mask and the numbers; this writes
the two products that make the result explorable rather than merely scored.

``*_delta.tif`` — a four-category raster: the change both maps agree on, what
Phoenix **added**, what it **removed**, and stable everywhere else. This is the
layer to load first. Two binary masks side by side answer "is it bigger"; the
delta answers "bigger *where*", which on this AOI is the whole question — U5's
finding was that 57% of the added pixels come from components under 10 px, and
that is visible in one glance here and invisible in a change-count table.

``*_rgb.tif`` — the stretched 8-bit S2 composite that Phoenix actually saw, as a
3-band GeoTIFF, so the delta can be read against the imagery rather than against
a mental model of it. Note this is the *stretched* image, not the raw composite:
it is what the encoder was shown, which is the honest backdrop for judging what
the encoder did.

Both are paletted/tiled with overviews and a `.qml`, matching every other raster
this project writes so QGIS renders them without configuration.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P src/phoenix_delta_map.py \\
        --map data/inference/s2_20260731_100710/oslo_s2off_centre_m3s3_bf_merged2.tif \\
        --refined data/inference/phoenix_oslo_crop/*_phoenix_crop_change.tif \\
        --image data/inference/s2_20260727_130926/oslo_s2_rgb_2024.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoenix_refine_map import (NODATA, is_change, load_map, qml_labels,  # noqa: E402
                                rgb_uint8)

# Deliberately not the merged2 palette: these categories are not land-cover
# classes and colouring them like classes invites reading the delta as a map.
DELTA_COLORS = {
    "stable (both)": (235, 235, 235),
    "change (both)": (150, 30, 30),
    "added by Phoenix": (255, 140, 0),
    "removed by Phoenix": (60, 120, 220),
}


def write_paletted(path: Path, codes: np.ndarray, profile, classes, colors):
    code_of = {c: i for i, c in enumerate(classes)}
    colormap = {i: (*colors[c], 255) for c, i in code_of.items()}
    colormap[NODATA] = (0, 0, 0, 0)
    prof = dict(profile)
    prof.update(driver="GTiff", count=1, dtype="uint8", nodata=NODATA,
                compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(codes.astype("uint8"), 1)
        dst.write_colormap(1, colormap)
        dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        dst.update_tags(ns="rio_overview", resampling="nearest")
    entries = "\n".join(
        f'          <paletteEntry value="{i}" '
        f'color="#{colors[c][0]:02x}{colors[c][1]:02x}{colors[c][2]:02x}" '
        f'alpha="255" label="{c}"/>'
        for c, i in code_of.items())
    path.with_suffix(".qml").write_text(
        '<!DOCTYPE qgis PUBLIC \'http://mrcc.com/qgis.dtd\' \'SYSTEM\'>\n'
        '<qgis version="3.34" styleCategories="AllStyleCategories">\n'
        '  <pipe>\n    <rasterrenderer type="paletted" band="1" opacity="1">\n'
        '      <colorPalette>\n' + entries + '\n      </colorPalette>\n'
        '    </rasterrenderer>\n  </pipe>\n</qgis>\n', encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, required=True, help="input merged2 raster")
    ap.add_argument("--refined", type=Path, nargs="+", required=True,
                    help="Phoenix binary change raster(s) from phoenix_refine_map")
    ap.add_argument("--image", type=Path, default=None,
                    help="S2 RGB composite; written back out 8-bit as the backdrop")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    codes, valid, profile, _ = load_map(args.map)
    labels = qml_labels(args.map)
    change_codes = [c for c, lab in labels.items() if is_change(lab)]
    base = np.isin(codes, change_codes) & valid
    out_dir = args.output_dir or args.refined[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        rgb = rgb_uint8(args.image, valid)
        prof = dict(profile)
        prof.update(driver="GTiff", count=3, dtype="uint8", nodata=None,
                    compress="deflate", tiled=True, blockxsize=256,
                    blockysize=256, photometric="RGB")
        rgb_path = out_dir / f"{args.map.stem}_phoenix_input_rgb.tif"
        with rasterio.open(rgb_path, "w", **prof) as dst:
            for b in range(3):
                dst.write(rgb[..., b], b + 1)
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")
        print(f"backdrop -> {rgb_path}")

    classes = list(DELTA_COLORS)
    for ref_path in args.refined:
        with rasterio.open(ref_path) as src:
            new = (src.read(1) == 1) & valid
        delta = np.full(codes.shape, NODATA, "uint8")
        delta[valid & ~base & ~new] = 0
        delta[valid & base & new] = 1
        delta[valid & ~base & new] = 2
        delta[valid & base & ~new] = 3
        # The run directory is part of the name, always. Arms are sibling
        # directories holding identically-named rasters (crop under
        # efficientvit and under ViT-H are both `..._phoenix_crop_change.tif`),
        # so keying on the stem alone lets one arm's delta overwrite another's
        # and both then read as whichever ran last.
        arm = ref_path.parent.name.replace("phoenix_oslo", "").strip("_") or "base"
        out = out_dir / f"{ref_path.stem.replace('_change', '')}_{arm}_delta.tif"
        write_paletted(out, delta, profile, classes, DELTA_COLORS)
        counts = {c: int((delta == i).sum()) for i, c in enumerate(classes)}
        kept = counts["change (both)"]
        print(f"\n{ref_path.name}")
        for c, n in counts.items():
            print(f"  {c:22s} {n:>10,}")
        print(f"  kept {kept:,} of {int(base.sum()):,} input change px "
              f"({kept / max(int(base.sum()), 1):.1%}); "
              f"added {counts['added by Phoenix']:,}")
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
