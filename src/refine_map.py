"""Edge-aware refinement of a classified map, guided by the Sentinel-2 image (U1).

The context model produces class boundaries that are smooth because AlphaEarth is
smooth, not because the ground is. A guided filter (He, Sun & Tang 2013) transfers
the *edge structure* of a guide image onto another signal at O(1) per pixel, which
is the classic fix: keep the model's semantics, borrow the imagery's boundaries.

Method. The class map is expanded to a one-hot stack, each channel is guided-
filtered against the S2 image, and the stack is re-argmaxed. Filtering one-hot
channels rather than the class codes matters -- codes are arbitrary integers and
smoothing them averages "Vegetation" and "Artificial" into whatever lies between.
Because the input is hard 0/1 rather than posteriors, this is a conservative
version of U1: it can move a boundary and remove an isolated pixel, but it cannot
recover a class the arg-max already discarded. A probability-level version would
be strictly stronger and needs `infer_s2.py` to persist its posteriors.

**What this can and cannot show.** Oslo has no validation plots, so nothing here
measures accuracy. It measures *structure*: whether boundaries land on real image
edges (`boundary_align`), whether objects stay coherent (`median_segment_px`), and
whether the change class survives. A refinement that raises alignment while
holding change pixels is doing what it claims; one that raises alignment by
deleting the minority class is not, which is why the change count is reported
beside every row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from map_detail_metrics import metrics_for
from infer_twotower import MERGED_COLORS, NODATA, write_class_raster


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float):
    """He et al. (2013) guided filter, box-filter implementation.

    Returns ``a * guide + b`` where (a, b) are the per-window linear coefficients
    that best explain ``src`` from ``guide``. ``eps`` sets how much guide variance
    counts as an edge rather than noise: small eps follows the image closely,
    large eps degenerates toward a plain box blur.
    """
    from scipy.ndimage import uniform_filter

    size = 2 * radius + 1
    mean_g = uniform_filter(guide, size, mode="nearest")
    mean_s = uniform_filter(src, size, mode="nearest")
    corr_gg = uniform_filter(guide * guide, size, mode="nearest")
    corr_gs = uniform_filter(guide * src, size, mode="nearest")
    var_g = corr_gg - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return (uniform_filter(a, size, mode="nearest") * guide
            + uniform_filter(b, size, mode="nearest"))


def load_classes(path: Path):
    with rasterio.open(path) as src:
        codes = src.read(1)
        nodata = src.nodata
        profile = src.profile
        try:
            colormap = src.colormap(1)
        except ValueError:
            colormap = None
    return codes, nodata, profile, colormap


def refine(codes, valid, guide, radius, eps, prob_stack=None, prob_codes=None):
    """Guided-filter a per-class stack, then arg-max, restricted to valid pixels.

    With ``prob_stack`` the channels are the model's posteriors; without it they
    are one-hot indicators of the hard class map. The difference is the whole
    point of the probability-level test: one-hot throws away how *sure* the model
    was, so a lone confident change pixel is indistinguishable from a lone
    marginal one and both lose the neighbourhood vote.
    """
    if prob_stack is not None:
        labels = list(prob_codes)
        stack = np.stack([guided_filter(guide, np.nan_to_num(prob_stack[i], nan=0.0),
                                        radius, eps)
                          for i in range(prob_stack.shape[0])])
    else:
        labels = sorted(int(c) for c in np.unique(codes[valid]))
        stack = np.zeros((len(labels), *codes.shape), "float32")
        for i, lab in enumerate(labels):
            channel = ((codes == lab) & valid).astype("float32")
            stack[i] = guided_filter(guide, channel, radius, eps)
    out = np.full(codes.shape, NODATA, "uint8")
    winner = np.array(labels, dtype="uint8")[stack.argmax(0)]
    out[valid] = winner[valid]
    return out


def load_probs(path: Path, labels: dict[int, str]):
    """(stack, codes) aligning posterior bands to the class map's codes.

    Band order comes from the raster's own band descriptions, not from position,
    and each is matched back to the code the class map uses. Getting this wrong
    silently permutes the classes -- the same failure the .qml lookup fixed.
    """
    with rasterio.open(path) as src:
        stack = src.read().astype("float32")
        names = [src.descriptions[i] or "" for i in range(src.count)]
    name_to_code = {v: k for k, v in labels.items()}
    missing = [n for n in names if n not in name_to_code]
    if missing:
        raise SystemExit(f"posterior bands {missing} are not classes of the map "
                         f"({sorted(name_to_code)})")
    return stack, [name_to_code[n] for n in names]


def normalised_guide(path: Path, valid):
    """Guide image scaled to ~[0, 1] so a single eps means the same everywhere."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
    finite = np.isfinite(arr) & (arr > 0) & valid
    if not finite.any():
        raise SystemExit(f"guide {path} has no valid pixels overlapping the map")
    lo, hi = np.percentile(arr[finite], [2, 98])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


def labels_from_qml(map_path: Path) -> dict[int, str]:
    """{code: class name} from the .qml sidecar written beside the raster.

    The codes are NOT the insertion order of ``MERGED_COLORS`` -- ``write_class_raster``
    numbers them from the model's *sorted* class list, so assuming the palette's
    order silently mislabels every class (it counts stable Vegetation, the
    dominant class, as change). The sidecar is the authoritative mapping and is
    written next to every raster this project produces.
    """
    import re

    qml = map_path.with_suffix(".qml")
    if not qml.exists():
        raise SystemExit(f"no {qml.name} beside the raster; cannot trust class codes")
    text = qml.read_text(encoding="utf-8")
    return {int(v): lab for v, lab in
            re.findall(r'paletteEntry value="(\d+)"[^>]*label="([^"]*)"', text)}


def change_fraction(codes, valid, labels: dict[int, str]):
    """Share of valid pixels whose class is a transition, for the sanity column."""
    changed = np.zeros(codes.shape, bool)
    for code, name in labels.items():
        if " -> " in name and name.split(" -> ")[0] != name.split(" -> ")[-1]:
            changed |= codes == code
    return float((changed & valid).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("map", type=Path, help="paletted class GeoTIFF")
    parser.add_argument("--guide", type=Path, required=True)
    parser.add_argument("--probs", type=Path, default=None,
                        help="merged2 posterior stack from infer_s2.py --save-probs. "
                             "Refining posteriors instead of a one-hot class map is "
                             "the strictly stronger version of U1: a confident change "
                             "pixel can outvote its neighbours, which a hard 0/1 vote "
                             "can never do.")
    parser.add_argument("--radius", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--classes", nargs="+", default=list(MERGED_COLORS))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    codes, nodata, profile, _ = load_classes(args.map)
    valid = codes != (nodata if nodata is not None else NODATA)
    labels = labels_from_qml(args.map)
    guide = normalised_guide(args.guide, valid)
    out_dir = args.output_dir or args.map.parent
    prob_stack, prob_codes = (load_probs(args.probs, labels)
                              if args.probs else (None, None))
    mode = "posterior" if prob_stack is not None else "one-hot"
    print(f"refining the {mode} stack")

    base = metrics_for(args.map, args.guide)
    base_change = change_fraction(codes, valid, labels)
    print(f"baseline {args.map.name}: edge {base['edge_density']:.4f}  "
          f"medseg {base['median_segment_px']:.0f}  "
          f"align {base['boundary_align']:.4f}  change {base_change:,.0f} px")

    rows = []
    for radius in args.radius:
        for eps in args.eps:
            refined = refine(codes, valid, guide, radius, eps,
                             prob_stack, prob_codes)
            tag = "gfp" if prob_stack is not None else "gf"
            name = f"{args.map.stem}_{tag}_r{radius}_e{eps:g}.tif"
            path = out_dir / name
            # Keep the original class codes so the written raster shares the
            # input's code->label mapping (and its .qml stays meaningful).
            present = sorted(int(c) for c in np.unique(refined[valid]))
            keep = [labels[c] for c in present]
            remap = np.full(256, NODATA, "uint8")
            for new, old in enumerate(present):
                remap[old] = new
            write_class_raster(path, np.where(valid, remap[refined], NODATA),
                               _GeoBoxShim(profile), keep, MERGED_COLORS)
            stats = metrics_for(path, args.guide)
            changed_px = change_fraction(refined, valid, labels)
            moved = float(((refined != codes) & valid).sum())
            rows.append({
                "radius": radius, "eps": eps,
                "edge_density": stats["edge_density"],
                "median_segment_px": stats["median_segment_px"],
                "boundary_align": stats["boundary_align"],
                "change_px": changed_px,
                "change_delta": changed_px / max(base_change, 1) - 1,
                "px_moved_frac": moved / max(valid.sum(), 1),
            })
            r = rows[-1]
            print(f"  r={radius} eps={eps:<7g} edge {r['edge_density']:.4f}  "
                  f"medseg {r['median_segment_px']:.0f}  "
                  f"align {r['boundary_align']:.4f}  "
                  f"change {changed_px:,.0f} ({r['change_delta']:+.1%})  "
                  f"moved {r['px_moved_frac']:.2%}", flush=True)

    import pandas as pd
    frame = pd.DataFrame(rows)
    csv = out_dir / f"{args.map.stem}_guided_sweep.csv"
    frame.to_csv(csv, index=False)
    print(f"\n-> {csv}")


class _GeoBoxShim:
    """Minimal geobox-like object so write_class_raster can reuse a profile."""

    class _CRS:
        def __init__(self, epsg):
            self.epsg = epsg

    def __init__(self, profile):
        self.transform = profile["transform"]
        self.crs = self._CRS(profile["crs"].to_epsg())


if __name__ == "__main__":
    main()
