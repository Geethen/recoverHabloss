"""Quantify how much spatial detail a classified map actually carries.

Detail has so far been judged by looking at the raster in QGIS. That is how the
AlphaEarth-only Oslo map came to be preferred, and it is a real signal -- but it
is not trackable, not comparable across runs, and not something an experiment can
target. This module turns it into numbers.

**Everything here is label-free.** That is the point: memory records that zero
validation plots fall inside either AOI, so accuracy cannot be measured on Oslo
at all. Structure can. These metrics are computable on any map, anywhere, with no
reference data.

The metrics, and why more than one is needed
--------------------------------------------
``edge_density``    fraction of pixels whose class differs from a 4-neighbour.
                    More detail raises it -- but so does salt-and-pepper noise,
                    so it must never be read alone.
``segment_count``   connected components per megapixel, and their median size.
                    A blurred map has few large blobs; a detailed map has many
                    small ones; a noisy map has thousands of single pixels, which
                    ``median_segment_px`` exposes and ``edge_density`` does not.
``hf_power_ratio``  variance surviving a 3x3 mean filter, over total variance.
                    An effective-resolution measure: a map that is nominally 10 m
                    but structurally 50 m has little high-frequency content.
``boundary_align``  **the quality metric.** Mean Sentinel-2 gradient magnitude on
                    class-boundary pixels, divided by the mean over all pixels.
                    >1 means boundaries fall on real image edges; ~1 means they
                    fall in arbitrary places, which is what noise looks like.
                    Needs a reference image, so it is optional.

Read them as a pair: **edge density says how much structure, boundary alignment
says whether the structure is real.** A map can win the first and lose the
second, and that map is worse, not better. Neither number is meaningful alone,
which is exactly why "it looks sharper" was never sufficient.

Usage::

    python map_detail_metrics.py data/inference/**/oslo_*_coarse3.tif
    python map_detail_metrics.py map.tif --reference s2_composite.tif
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: Path, band: int = 1):
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(band)
        nodata = src.nodata
        res = src.res[0]
    mask = np.ones(arr.shape, bool)
    if nodata is not None:
        mask &= arr != nodata
    return arr, mask, res


def edge_map(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pixels differing from their right or lower neighbour, both sides valid."""
    edge = np.zeros(labels.shape, bool)
    diff_x = (labels[:, :-1] != labels[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    diff_y = (labels[:-1] != labels[1:]) & valid[:-1] & valid[1:]
    edge[:, :-1] |= diff_x
    edge[:, 1:] |= diff_x
    edge[:-1] |= diff_y
    edge[1:] |= diff_y
    return edge & valid


def segment_stats(labels: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """(components per megapixel, median component size in px) over valid pixels."""
    try:
        from scipy import ndimage
    except ImportError:
        return float("nan"), float("nan")
    sizes = []
    for cls in np.unique(labels[valid]):
        comp, n = ndimage.label((labels == cls) & valid)
        if n:
            sizes.append(np.bincount(comp.ravel())[1:])
    if not sizes:
        return float("nan"), float("nan")
    sizes = np.concatenate(sizes)
    megapixels = valid.sum() / 1e6
    return len(sizes) / max(megapixels, 1e-9), float(np.median(sizes))


def hf_power_ratio(labels: np.ndarray, valid: np.ndarray) -> float:
    """Share of variance destroyed by a 3x3 mean filter -- effective resolution.

    Computed on a one-hot indicator stack so it is meaningful for a categorical
    map (smoothing a class *code* would be arithmetic on arbitrary integers).
    """
    try:
        from scipy import ndimage
    except ImportError:
        return float("nan")
    total = smooth = 0.0
    for cls in np.unique(labels[valid]):
        ind = ((labels == cls) & valid).astype("float32")
        blur = ndimage.uniform_filter(ind, 3)
        total += float(ind[valid].var())
        smooth += float(blur[valid].var())
    if total <= 0:
        return float("nan")
    return 1.0 - smooth / total


def boundary_alignment(edge: np.ndarray, valid: np.ndarray,
                       reference: Path | None) -> float:
    """Mean reference-image gradient on boundaries / mean over all valid pixels.

    Answers the question edge density cannot: are these boundaries in the same
    places as real edges in the imagery, or just anywhere?
    """
    if reference is None:
        return float("nan")
    ref, ref_mask, _ = _load(reference)
    if ref.shape != edge.shape:
        return float("nan")
    ref = ref.astype("float32")
    gy, gx = np.gradient(ref)
    grad = np.hypot(gx, gy)
    both = valid & ref_mask & np.isfinite(grad)
    on = both & edge
    if on.sum() < 100 or both.sum() < 100:
        return float("nan")
    base = grad[both].mean()
    return float(grad[on].mean() / base) if base > 0 else float("nan")


def metrics_for(path: Path, reference: Path | None = None) -> dict:
    labels, valid, res = _load(path)
    if valid.sum() == 0:
        return {"map": path.name, "valid_px": 0}
    edge = edge_map(labels, valid)
    per_mp, median_px = segment_stats(labels, valid)
    return {
        "map": path.name,
        "valid_px": int(valid.sum()),
        "resolution_m": res,
        "n_classes": int(len(np.unique(labels[valid]))),
        "edge_density": float(edge[valid].mean()),
        "segments_per_mp": per_mp,
        "median_segment_px": median_px,
        "hf_power_ratio": hf_power_ratio(labels, valid),
        "boundary_align": boundary_alignment(edge, valid, reference),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps", nargs="+", help="classified GeoTIFFs (globs ok)")
    parser.add_argument("--reference", type=Path, default=None,
                        help="co-registered image for boundary alignment")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    paths = []
    for pattern in args.maps:
        paths.extend(sorted(Path(p) for p in glob.glob(pattern, recursive=True)))
    if not paths:
        parser.error("no maps matched")

    rows = [metrics_for(p, args.reference) for p in paths]
    frame = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(frame.round(4).to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
