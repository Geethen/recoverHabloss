"""Turn the stored Sentinel-2 VNIR patches into the detail tower's feature table.

The extractor keeps a 64x64 10 m patch per plot-year (``extract_s2_points.py``).
Everything here is a local computation over that array, so a new feature family
is a re-run of this script and never a re-download -- the failure mode that
killed idea D3 in ``TWOTOWER_RESEARCH.md``.

Feature families, each with its own column prefix so ``twotower_lab.py`` ideas
can select subsets without a rebuild:

``S2c``    centre pixel reflectance -- what a point sample would have given
``S2m3/9/25``, ``S2s3/9/25``  neighbourhood mean and standard deviation at 30 m,
           90 m and 250 m. The **std** family is the point of the whole exercise:
           AlphaEarth is a smooth 10 m context embedding and cannot say whether a
           pixel sits in a homogeneous field or a built-up mosaic, which is
           exactly the distinction the stable-Artificial confusion turns on.
``S2lc``   centre minus its 90 m mean -- signed local contrast, the detail that
           survives after the context is subtracted off
``S2g``    Sobel gradient magnitude at the centre -- edge strength, high on
           roads, roofs and field boundaries, low inside vegetation
``S2p``    8x8 mean-pooled patch (80 m cells) -- a coarse image rather than a
           statistic, for a tower that would rather learn its own texture

Channels are the four native-10 m VNIR bands plus three indices computed from
them (NDVI, NDWI, brightness). Every family exists for both endpoint years and
as a ``_diff`` (2024 - 2018), which is the form the change task consumes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARDS = REPO_ROOT / "data" / "embeddings" / "s2_shards"
DEFAULT_OUT = REPO_ROOT / "data" / "embeddings" / "s2_features_habloss_recover.parquet"

BANDS = ("blue", "green", "red", "nir")
CHANNELS = ("blue", "green", "red", "nir", "ndvi", "ndwi", "bright")
SCALES = (3, 9, 25)
POOL = 8  # 64 / 8 -> 8x8 pooled cells of 80 m
# NDVI below this reads as non-vegetated. Calibrated, not assumed: see
# analyse_ndvi_threshold.py and the S2bf note in features_for_year.
NDVI_VEG_CUT = 0.31


def _channels(patch: np.ndarray) -> np.ndarray:
    """(n, 4, h, w) reflectance -> (n, 7, h, w) with the three indices appended.

    Indices are unitless and robust to the residual illumination differences
    between a 2018 and a 2024 composite in a way raw DN is not, so they carry
    most of the cross-year comparability.
    """
    blue, green, red, nir = (patch[:, i] for i in range(4))
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    bright = patch.mean(1)
    return np.stack([blue, green, red, nir, ndvi, ndwi, bright], axis=1)


def _centre_block(cube: np.ndarray, size: int) -> np.ndarray:
    """The central ``size x size`` block of an (n, c, h, w) cube."""
    h = cube.shape[-1]
    lo = (h - size) // 2
    return cube[..., lo:lo + size, lo:lo + size]


def _sobel_centre(cube: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude at the centre pixel of each (n, c, h, w) patch."""
    block = _centre_block(cube, 3)
    gx = ((block[..., 0, 2] + 2 * block[..., 1, 2] + block[..., 2, 2])
          - (block[..., 0, 0] + 2 * block[..., 1, 0] + block[..., 2, 0]))
    gy = ((block[..., 2, 0] + 2 * block[..., 2, 1] + block[..., 2, 2])
          - (block[..., 0, 0] + 2 * block[..., 0, 1] + block[..., 0, 2]))
    return np.sqrt(gx ** 2 + gy ** 2)


def _pool(cube: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool an (n, c, h, w) cube by ``factor`` in both spatial dims."""
    n, c, h, w = cube.shape
    k = h // factor
    view = cube.reshape(n, c, factor, k, factor, k)
    return np.nanmean(view, axis=(3, 5))


def features_for_year(patch: np.ndarray, year: int) -> dict[str, np.ndarray]:
    """Every feature family for one year's (n, 4, 64, 64) patch stack."""
    cube = _channels(patch)
    out: dict[str, np.ndarray] = {}
    centre = _centre_block(cube, 1)[..., 0, 0]
    for ci, name in enumerate(CHANNELS):
        out[f"S2c_{name}_{year}"] = centre[:, ci]

    means = {}
    for size in SCALES:
        block = _centre_block(cube, size)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(block, axis=(-2, -1))
            std = np.nanstd(block, axis=(-2, -1))
        means[size] = mean
        for ci, name in enumerate(CHANNELS):
            out[f"S2m{size}_{name}_{year}"] = mean[:, ci]
            out[f"S2s{size}_{name}_{year}"] = std[:, ci]

    for ci, name in enumerate(CHANNELS):
        out[f"S2lc_{name}_{year}"] = centre[:, ci] - means[9][:, ci]

    grad = _sobel_centre(cube)
    for ci, name in enumerate(CHANNELS):
        out[f"S2g_{name}_{year}"] = grad[:, ci]

    pooled = _pool(cube, POOL)  # (n, c, POOL, POOL)
    for ci, name in enumerate(CHANNELS):
        for i in range(POOL):
            for j in range(POOL):
                out[f"S2p_{name}_{i}{j}_{year}"] = pooled[:, ci, i, j]

    # Built fraction: share of a window below the NDVI vegetation cut. This is
    # the "built-fraction covariate" TWOTOWER_RESEARCH.md F6 named as the lever
    # for stable-Artificial, and `analyse_ndvi_threshold.py` calibrated the cut
    # against the labelled stable plots -- optimum 0.31 by Youden's J and by
    # balanced accuracy (the user's 0.30 estimate scores within 0.001 of it).
    #
    # The radius matters more than the cut. Sweeping windows for stable-
    # Artificial vs stable-Vegetation separability gives AUC 0.669 (1 px),
    # **0.762 (3 px)**, 0.695 (5 px), 0.684 (9 px), down to 0.648 at the full
    # 64 px patch: a plot is ~10 m, so 30 m captures the built context (a roof
    # and its yard) while a 640 m window dilutes it into the surrounding
    # landscape. All radii are carried so the model can choose, but 3 px is the
    # one the evidence backs.
    ndvi = cube[:, CHANNELS.index("ndvi")]
    for w in (3, 5, 9, 25, 64):
        block = _centre_block(ndvi[:, None], w)[:, 0]
        with np.errstate(invalid="ignore"):
            out[f"S2bf{w}_{year}"] = np.nanmean(block < NDVI_VEG_CUT, axis=(-2, -1))
    return out


def load_shards(shard_dir: Path):
    """Concatenate the extractor's shards into (patches, plotid, years)."""
    paths = sorted(shard_dir.glob("shard_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No shards under {shard_dir}")
    patches, ids = [], []
    years = None
    for path in paths:
        data = np.load(path, allow_pickle=True)
        stored = data["patches"].astype(np.float32)
        stored[~data["finite"]] = np.nan
        patches.append(stored)
        ids.append(data["plotid"])
        years = tuple(int(y) for y in data["years"])
    return np.concatenate(patches), np.concatenate(ids), years


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    patches, plotid, years = load_shards(args.shard_dir)
    print(f"{len(plotid):,} plots | patches {patches.shape} | years {years}")

    columns: dict[str, np.ndarray] = {"PLOTID": plotid}
    per_year = {}
    for yi, year in enumerate(years):
        per_year[year] = features_for_year(patches[:, yi], year)
        columns.update(per_year[year])

    # Endpoint difference: the change task reads this, not the two states.
    first, last = years[0], years[-1]
    for key, values in per_year[last].items():
        stem = key[: -len(str(last))]
        other = f"{stem}{first}"
        if other in per_year[first]:
            columns[f"{stem}diff"] = values - per_year[first][other]

    frame = pd.DataFrame(columns)
    centre_cols = [f"S2c_{c}_{y}" for c in BANDS for y in years]
    frame["s2_present"] = frame[centre_cols].notna().all(axis=1).astype("float32")

    frame.to_parquet(args.output, index=False)
    families = sorted({c.split("_")[0] for c in frame.columns if c.startswith("S2")})
    print(f"wrote {frame.shape[0]:,} x {frame.shape[1]:,} -> {args.output}")
    print(f"  families: {families}")
    print(f"  both-year coverage: {frame['s2_present'].mean():.1%}")


if __name__ == "__main__":
    main()
