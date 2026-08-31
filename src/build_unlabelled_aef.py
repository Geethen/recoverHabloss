"""Sample an unlabelled AlphaEarth 2018/2024 endpoint pool for section N4.

Every idea in `SIAMESE_RESEARCH.md` up to N3 rearranges the same 6,414 labelled
plots. N3 found that Barlow Twins on the stable endpoint pairs matches the
gate-supervised cosine to within seed noise while needing **strictly less
supervision** -- only which plots are stable, never which transition they are.
That is what makes this script worth writing: the objective transfers to pixels
that carry no label at all, and the learning curves (S19, +0.026 change-F1 per
doubling of labels) say added information is the one thing this problem is
actually short of.

What "unlabelled" costs here
----------------------------
The Barlow term is applied to sampled pixels **assumed stable**. That assumption
is wrong on exactly the change fraction of the AOI, which the deployed Oslo map
puts at ~0.5% of pixels. So ~1 sampled pair in 200 is a change pair being pushed
toward year-invariance -- a contamination rate two orders of magnitude below the
signal, and a wrong direction on a redundancy-reduction term rather than on a
classification loss. That is the whole approximation, and it is stated here
rather than buried because it is the reason the idea is cheap.

Sampling is uniform over valid pixels **on purpose**. Sampling toward change
would raise the contamination rate; sampling toward the stable majority is what
the term wants and is what uniform sampling over a ~99.5%-stable scene gives.

Contamination of the evaluation is not a concern in the other direction: **zero
labelled plots fall inside either AOI bbox** (G3/G4), so no pixel drawn here can
be a test plot under any fold.

Usage::

    python build_unlabelled_aef.py --aoi oslo --n-pixels 200000
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd

from aef_loader import AEFIndex, DataSource, VirtualTiffReader, aoi_geobox
from infer_cities import CITY_AOIS, load_year_embeddings
from infer_s2 import YEARS, stack_aef_bands
from model_zoo import DEFAULT_INPUT, feature_columns
from project_paths import project_data_dir


async def sample_aoi(name: str, n_pixels: int, resolution: float, seed: int,
                     manifest_cache: Path | None) -> pd.DataFrame:
    frame = pd.read_parquet(DEFAULT_INPUT)
    aef_cols = feature_columns(frame)

    index = AEFIndex(source=DataSource.SOURCE_COOP)
    await index.download()
    index.load()

    bbox = CITY_AOIS[name]
    async with VirtualTiffReader(manifest_cache_dir=manifest_cache) as reader:
        probe = await index.query(bbox=bbox, years=YEARS[0])
        geobox = aoi_geobox(bbox, crs=f"EPSG:{probe[0].crs_epsg}",
                            resolution=resolution, bbox_crs="EPSG:4326")
        print(f"[{name}] geobox {geobox.shape.y}x{geobox.shape.x} @ {resolution} m",
              flush=True)
        emb = {}
        for year in YEARS:
            emb[year] = await load_year_embeddings(index, reader, bbox, year, geobox)

    # Same band stacking as the inference path, so a column named A07_2018 here
    # is the same quantity as A07_2018 in the training frame. Building it any
    # other way would be a train/serve skew with no test to catch it.
    bands = stack_aef_bands(emb[YEARS[0]], emb[YEARS[1]], aef_cols)
    valid = np.isfinite(bands).all(0)
    flat = bands.reshape(bands.shape[0], -1)
    idx = np.flatnonzero(valid.ravel())
    print(f"[{name}] {len(idx):,} valid pixels of {valid.size:,} "
          f"({valid.mean():.1%})", flush=True)

    rng = np.random.default_rng(seed)
    take = rng.choice(idx, size=min(n_pixels, len(idx)), replace=False)
    out = pd.DataFrame(flat[:, take].T.astype("float32"), columns=aef_cols)
    out["aef_present"] = np.float32(1.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi", default="oslo", choices=sorted(CITY_AOIS))
    parser.add_argument("--n-pixels", type=int, default=200_000)
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest-cache", type=Path,
                        default=Path(".aef_manifest_cache"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or project_data_dir(
        "embeddings", f"unlabelled_aef_{args.aoi}.parquet")
    frame = asyncio.run(sample_aoi(args.aoi, args.n_pixels, args.resolution,
                                   args.seed, args.manifest_cache))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    print(f"{len(frame):,} unlabelled endpoint pairs -> {out}")


if __name__ == "__main__":
    main()
