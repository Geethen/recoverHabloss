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

Channels are the four native-10 m VNIR bands plus indices computed from them.
**Only 10 m indices**: everything here is a function of B2/B3/B4/B8 alone, so no
family costs a 20 m resample and the SWIR bands the ledger ruled out (section Q,
"more S2 bands are not a lever, do not extract SWIR") are never needed. The set
is split in two and the split is load-bearing:

``CHANNELS_BASE``  the deployed seven -- blue, green, red, nir, NDVI, NDWI,
                   brightness. `s2_subset_columns` restricts to these unless a
                   subset asks otherwise, so adding channels below **cannot**
                   silently change the 78 columns the deployed model reads.
``CHANNELS_10M``   four further 10 m indices spanning axes NDVI does not:
                   EVI2 (greenness past NDVI's saturation), GRVI (green-red
                   contrast -- senescence, ploughed soil), BSI (bare/built, the
                   SWIR-free form) and CI (visible coloration -- red roofs and
                   bare soil against grey concrete).

Every family exists for both endpoint years and as a ``_diff`` (2024 - 2018),
which is the form the change task consumes. The ``_diff`` of an *index* is a
difference of two unitless ratios and is what a diff-only detail tower reads;
the ``_diff`` of a *band* is a difference of two DN composites, which is only
comparable because the endpoints carry no relative radiometric offset (checked:
the four band medians agree between 2018 and 2024 to 2-6%, so the L2A
baseline-04 ``BOA_ADD_OFFSET`` is not sitting in the 2024 end of the difference).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARDS = REPO_ROOT / "data" / "embeddings" / "s2_shards"
DEFAULT_OUT = (REPO_ROOT / "data" / "embeddings"
               / "s2_features_habloss_recover_10m.parquet")

BANDS = ("blue", "green", "red", "nir")
#: The deployed channel set. `twotower_lab.s2_subset_columns` defaults to it, so
#: this tuple -- not `CHANNELS` -- is what pins `s2off_centre_m3s3_bf` to 78
#: columns. Do not reorder it and do not append to it; append to `CHANNELS_10M`.
CHANNELS_BASE = ("blue", "green", "red", "nir", "ndvi", "ndwi", "bright")
#: Further indices, all computable from the four native-10 m bands alone.
CHANNELS_10M = ("evi2", "grvi", "bsi", "ci")
CHANNELS = CHANNELS_BASE + CHANNELS_10M
SCALES = (3, 9, 25)
#: L2A stores reflectance x 10000. NDVI/NDWI/GRVI/BSI/CI are ratios and do not
#: care, but EVI2's soil term is an absolute constant and is wrong by four
#: orders of magnitude without this.
REFL_SCALE = 1e-4
POOL = 8  # 64 / 8 -> 8x8 pooled cells of 80 m
# NDVI below this reads as non-vegetated. Calibrated, not assumed: see
# analyse_ndvi_threshold.py and the S2bf note in features_for_year.
NDVI_VEG_CUT = 0.31


def _channels(patch: np.ndarray) -> np.ndarray:
    """(n, 4, h, w) reflectance -> (n, len(CHANNELS), h, w), indices appended.

    Indices are unitless and robust to the residual illumination differences
    between a 2018 and a 2024 composite in a way raw DN is not, so they carry
    most of the cross-year comparability -- which is the whole argument for a
    difference-driven detail tower, since a difference of two DN composites
    inherits both dates' illumination and a difference of two ratios does not.

    The first seven channels are `CHANNELS_BASE` in that exact order; every
    downstream column name embeds the channel name, so appending here is
    additive and never renames or reorders an existing column.

    All four added indices use B2/B3/B4/B8 only:

    ``evi2``  2.5 (N - R) / (N + 2.4 R + 1) -- Jiang et al.'s two-band EVI, the
              blue-free form. It keeps responding where NDVI saturates, so it
              separates a dense canopy from a dense crop; NDVI cannot.
    ``grvi``  (G - R) / (G + R) -- green against red. Goes negative on
              senescent, ploughed and bare surfaces while NDVI is still mildly
              positive, which is the Cropland-vs-Nature axis NDVI blurs.
    ``bsi``   ((R + B) - (N + G)) / ((R + B) + (N + G)) -- the bare-soil index
              in its SWIR-free four-band form. The built/bare direction that
              `S2bf` currently has to infer from an NDVI threshold alone.
    ``ci``    (R - B) / (R + B) -- coloration. Red roofs and iron-rich soil
              against grey concrete and asphalt, a distinction every one of the
              other channels is blind to.
    """
    blue, green, red, nir = (patch[:, i] for i in range(4))
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    bright = patch.mean(1)
    # Scaled copies, for the one index with an absolute term in it.
    b, g, r, n = (x * REFL_SCALE for x in (blue, green, red, nir))
    evi2 = 2.5 * (n - r) / (n + 2.4 * r + 1.0)
    grvi = (green - red) / (green + red + eps)
    bsi = ((red + blue) - (nir + green)) / ((red + blue) + (nir + green) + eps)
    ci = (red - blue) / (red + blue + eps)
    stack = {"blue": blue, "green": green, "red": red, "nir": nir, "ndvi": ndvi,
             "ndwi": ndwi, "bright": bright, "evi2": evi2, "grvi": grvi,
             "bsi": bsi, "ci": ci}
    return np.stack([stack[name] for name in CHANNELS], axis=1)


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

    # The pooled-patch family stays on `CHANNELS_BASE`. It is 64 columns per
    # channel per year -- three quarters of the whole table -- and it exists for
    # a tower that learns its own texture, which is an architecture question the
    # added indices have nothing to say about. Widening it would triple the
    # parquet to serve no experiment on the board.
    pooled = _pool(cube, POOL)  # (n, c, POOL, POOL)
    for name in CHANNELS_BASE:
        ci = CHANNELS.index(name)
        for i in range(POOL):
            for j in range(POOL):
                out[f"S2p_{name}_{i}{j}_{year}"] = pooled[:, ci, i, j]

    # Built fraction: share of a window below the NDVI vegetation cut. This is
    # the "built-fraction covariate" TWOTOWER_RESEARCH.md F6 named as the lever
    # for stable-Artificial, and `analyse_ndvi_threshold.py` calibrated the cut
    # against the labelled stable plots -- optimum 0.31 by Youden's J and by
    # balanced accuracy (the user's 0.30 estimate scores within 0.001 of it).
    #
    # The radius matters, but far less than this comment used to claim. The
    # original sweep read AUC 0.669 (1 px), 0.762 (3 px), 0.695 (5 px), 0.684
    # (9 px), 0.648 (64 px) and concluded 3 px won decisively. Those numbers
    # came from a tie-blind rank in `analyse_ndvi_threshold.auc`; built
    # fraction over a 3x3 window has only ten distinct values, so it was the
    # column the bug flattered most. Recomputed with mid-ranks the sweep is
    # **0.667 / 0.686 / 0.685 / 0.681 / 0.665 / 0.648** for 1/3/5/9/25/64 px.
    #
    # 3 px still comes first, so the choice stands -- but it leads 5 px by
    # 0.001, not 0.067, and the curve is flat out to 9 px before it decays.
    # The reading that survives is the weaker one: a plot is ~10 m, so a tight
    # window captures the built context (a roof and its yard) while a 640 m
    # window dilutes it into the landscape. All radii are carried so the model
    # can choose, which is what makes the near-tie cheap rather than a risk.
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
