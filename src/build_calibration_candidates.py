"""Calibration points, drawn from plots that already have an agreed answer.

Why this exists before any real batch
-------------------------------------
`ACTIVE_LEARNING.md` AL8 made the campaign safe for two experts, and the
finding it rests on is that **three people labelling to three standards produce
an agreement number that reads high over nothing**. The defence is a calibration
set worked before the first real batch, in two stages, with agreement read per
**expert** and reported *with the confusion pairs* rather than as a percentage.

The two stages do different jobs and neither substitutes for the other:

``teach``     the answer is shown after every call. This is what makes the
              legend stick, and it is also what makes the score meaningless.
``qualify``   the answer is withheld until the end. This is the measurement,
              and it is only interpretable because `teach` came first.

What gets drawn, and why it is not a random sample of the label set
--------------------------------------------------------------------
A calibration set stratified by *frequency* is mostly stable Nature, which
teaches nothing and measures nothing: everyone agrees on closed forest. The draw
here is deliberately weighted toward the places the campaign already knows
readers and models disagree:

* **The Cropland/Nature boundary.** The ledger's standing explanation for the
  change-F1 ceiling is label noise on exactly this pair, and `grass` and `crop`
  carry the highest stable-confusion rates measured in AL-T (0.31 / 0.21). If
  two interpreters split on long fallow, this is where it shows.
* **Bare ground.** AL-T: the model misreads it as built-up 3.1x more often than
  forest and is *more* confident when it does. The coverage batch is 27/75 bare,
  so the interpreters need to have agreed what bare-and-not-built looks like
  before they see it.
* **Seasonal water.** The other half of the map error, at JRC occurrence 5-20%.
* Every change class, so the rare ones are seen at least once before they are
  seen in `rar001`.

The reference answer is the plot's own RECOVER transition. That is what "agreed"
means here and it is worth being honest about the limit: it is one prior
reading, not an adjudicated panel, so a disagreement against it is evidence of a
*difference*, not proof the interpreter is wrong. The confusion pairs are the
output that matters, not the headline rate.

Run
---
    G=/home/geethen.singh/.pixi/envs/geo
    PROJ_DATA=$G/share/proj PROJ_LIB=$G/share/proj GDAL_DATA=$G/share/gdal \\
    /home/geethen.singh/.cache/phoenix-test/venv/bin/python \\
        src/build_calibration_candidates.py --n-teach 25 --n-qualify 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir

RESULTS = project_data_dir("analysis_results")

#: Relative weight per plot, by the stratum the plot sits in. Multiplicative with
#: the class weight below. These are the disagreement-prone grounds named above.
STRATUM_WEIGHT = {"grass": 3.0, "crop": 3.0, "bare": 3.0, "wetland": 3.0,
                  "shrub": 2.0, "tree": 1.0, "built": 1.5, "moss/lichen": 1.0,
                  "snow/ice": 1.0, "mangrove": 1.0}
WORLDCOVER = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built",
              60: "bare", 70: "snow/ice", 80: "water", 90: "wetland",
              95: "mangrove", 100: "moss/lichen"}

#: Every class must appear at least this often in each stage, so a rare
#: transition is seen before it is met in a real batch.
MIN_PER_CLASS = 2

#: Seasonal water, the wetland half of the map error (AL-T: `as_crop` peaks at
#: 5-20% JRC occurrence, 0.345 against a 0.118 base).
SEASONAL_WATER = (5.0, 20.0)
SEASONAL_BONUS = 2.0


def build_pool() -> pd.DataFrame:
    """Labelled plots with their transition, coordinates and terrain stratum."""
    from twotower_lab import load_context
    view = load_context().view("full")
    pool = pd.DataFrame({
        "id": view.frame["PLOTID"].astype(str).to_numpy(),
        "lon": view.frame["lon"].to_numpy(),
        "lat": view.frame["lat"].to_numpy(),
        "transition": view.target.to_numpy(),
    })
    terrain = RESULTS / "terrain_plots.parquet"
    if terrain.exists():
        t = pd.read_parquet(terrain)
        t["PLOTID"] = t["PLOTID"].astype(str)
        pool = pool.merge(
            t[["PLOTID", "worldcover", "water_occurrence", "slope"]]
            .rename(columns={"PLOTID": "id"}).drop_duplicates("id"),
            on="id", how="left")
        pool["wc"] = pool["worldcover"].map(WORLDCOVER)
    else:
        pool["wc"] = None
        pool["water_occurrence"] = 0.0
    return pool


def weights(pool: pd.DataFrame) -> np.ndarray:
    """Per-plot draw weight: rare class x disagreement-prone ground."""
    counts = pool["transition"].value_counts()
    # 1/sqrt(n) rather than 1/n: the latter makes a 46-plot class 90x more likely
    # than a 4,200-plot one and fills the set with a single transition.
    w = pool["transition"].map(lambda c: 1.0 / np.sqrt(counts[c])).to_numpy()
    w = w * pool["wc"].map(STRATUM_WEIGHT).fillna(1.0).to_numpy()
    seasonal = pool["water_occurrence"].between(*SEASONAL_WATER).to_numpy()
    w = np.where(seasonal, w * SEASONAL_BONUS, w)
    return w / w.sum()


def draw(pool: pd.DataFrame, w: np.ndarray, n: int, rng,
         taken: set) -> pd.DataFrame:
    """Weighted draw without replacement, with a floor per transition class."""
    keep = []
    avail = ~pool["id"].isin(taken).to_numpy()
    # Floor first, so the rare transitions cannot be crowded out by the weights.
    for cls in sorted(pool["transition"].unique()):
        idx = np.flatnonzero(avail & (pool["transition"] == cls).to_numpy())
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(MIN_PER_CLASS, len(idx)), replace=False,
                          p=_renorm(w[idx]))
        keep.extend(pick.tolist())
        avail[pick] = False
    remaining = n - len(keep)
    if remaining > 0:
        idx = np.flatnonzero(avail)
        pick = rng.choice(idx, size=min(remaining, len(idx)), replace=False,
                          p=_renorm(w[idx]))
        keep.extend(pick.tolist())
    return pool.iloc[sorted(keep[:n])].copy()


def _renorm(v: np.ndarray) -> np.ndarray:
    total = v.sum()
    return v / total if total > 0 else np.full(len(v), 1.0 / len(v))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-teach", type=int, default=25)
    ap.add_argument("--n-qualify", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=RESULTS)
    args = ap.parse_args()

    pool = build_pool()
    w = weights(pool)
    rng = np.random.default_rng(args.seed)
    print(f"{len(pool):,} labelled plots in the pool")

    taken: set = set()
    for stage, n in (("teach", args.n_teach), ("qualify", args.n_qualify)):
        sub = draw(pool, w, n, rng, taken)
        taken |= set(sub["id"])
        sub = sub.assign(channel="calibration", stage=stage,
                         rank=np.arange(1, len(sub) + 1), cell_km=5.0)
        out = args.outdir / f"calibration_{stage}.csv"
        sub[["id", "lon", "lat", "channel", "rank", "cell_km", "transition",
             "wc", "water_occurrence", "slope"]].to_csv(out, index=False)
        print(f"\n-> {out}  ({len(sub)} points)")
        print(sub["transition"].value_counts().to_string())
        print("  ground: " + ", ".join(
            f"{k} {v}" for k, v in sub["wc"].value_counts().items()))

    # The two stages must not share a plot: a point seen with its answer in
    # `teach` and then scored in `qualify` measures memory, not agreement.
    assert len(taken) == args.n_teach + args.n_qualify, "stages overlap"
    print(f"\n{len(taken)} distinct plots, no overlap between stages")


if __name__ == "__main__":
    main()
