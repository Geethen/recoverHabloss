"""Candidates for the **retrieval** channel: the change classes in deficit.

What this channel is bought on
-----------------------------
Confirmed plots per point, **per class** -- not accuracy. `ACTIVE_LEARNING.md`
AL1 is unambiguous that steering a campaign by rare-class retrieval damages the
model: `proto_sim` bought +203 change plots and cost **-0.046 change-F1**, and
-0.213 F1 on `Artificial -> Nature` alone, because it piles the training set into
one direction of change space. `oracle_change` -- perfect retrieval -- is worth
only +0.005. So this channel is deliberately **separate and small**, and it is
priced on plots returned, never quoted as an accuracy gain.

It is still the channel that matters most for round one, because
`PATCH_SAMPLING.md` section C names the one number the plan is missing: **the
change-restricted channel has no measured confirm rate.** Its `patches_needed`
is a lower bound -- what it would cost if every candidate confirmed, which none
will. A first batch weighted toward the scarcest classes measures that rate as a
side effect of doing the labelling anyway.

Where the points come from, and why two runs
--------------------------------------------
Both runs map the *same* 100 equal-area patches, so they are two readings of one
piece of ground:

``patches_20260728_150523``  the deployed `s2off_centre_m3s3_bf`. Supplies four
                             of the five classes, from its `topchange` layer for
                             `Cropland -> Nature` and its `coarse3` arg-max for
                             the rest. Already drawn as `label_points.csv`.
``patches_20260804_114321``  `siam_s2off_state_pre`. Supplies **`Artificial ->
                             Cropland` only**, from `coarse3_gated`.

That second row needs stating plainly. The deployed model returns **0** patches
holding a labellable hectare of `Artificial -> Cropland` on either of its layers;
the state-pretrained siamese returns **12** on identical ground. That is the
`STATE_PRETRAIN_RESEARCH.md` result ("breaks Art->Crop 0.000 -> 0.19") showing up
as a search tool.

**This does not re-open the deployed model.** A candidate generator is not a
product: nothing here is mapped, scored or shipped, and the plots it finds are
labelled by a human who is free to say the model was wrong -- which for a class
at 46 plots is the expected outcome most of the time. Using the deployed model
here instead would return zero candidates for the class that most needs them.

Allocation
----------
Weighted toward **scarcity**, not toward the sizing target. "Double every change
class" implies new plots proportional to *current* counts, which hands the
largest share to the most abundant change class and the smallest to the one with
46 plots. For a first batch whose job is partly to measure a confirm rate, the
scarce classes are worth more per point.

Run
---
    G=/home/geethen.singh/.pixi/envs/geo
    PROJ_DATA=$G/share/proj PROJ_LIB=$G/share/proj GDAL_DATA=$G/share/gdal \\
    /home/geethen.singh/.cache/phoenix-test/venv/bin/python \\
        src/build_rare_class_candidates.py --n-select 75
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir

PATCHES = project_data_dir("patches")
RESULTS = project_data_dir("analysis_results")
OUT = RESULTS / "rare_class_candidates.csv"

DEPLOYED_RUN = "patches_20260728_150523"
SIAM_RUN = "patches_20260804_114321"
SIAM_LAYER = "coarse3_gated"

#: Plots held per change class in the 6,414-plot frame. The allocation is a
#: function of these, so they are named rather than recomputed from a fit.
HELD = {
    "Artificial -> Cropland": 46,
    "Cropland -> Nature": 114,
    "Artificial -> Nature": 123,
    "Nature -> Cropland": 243,
    "Cropland -> Artificial": 333,
    "Nature -> Artificial": 383,
}

#: Points per class, for a 75-point batch. Written out rather than derived from
#: a formula because the shape of it is a judgement and should be arguable:
#: the two classes that cannot be reached by the deployed map at all get the
#: most, the binding class in the sizing gets a measurable share, and the most
#: abundant change class gets none because the equal-area control arm (`b001`)
#: already returns it at the base rate.
DEFAULT_MIX = {
    "Artificial -> Cropland": 20,
    "Cropland -> Nature": 20,
    "Artificial -> Nature": 15,
    "Cropland -> Artificial": 10,
    "Nature -> Cropland": 10,
    "Nature -> Artificial": 0,
}

#: Minimum pixels of the class in a patch before a point is drawn from it. The
#: `PATCH_SAMPLING.md` section B rule: 100 px of 10 m is 1 ha, and a class
#: occupying 40 scattered pixels cannot be labelled however much it is wanted.
MIN_PX = 100
#: Points a single patch may contribute for one class. Points a few hundred
#: metres apart are close to the same observation.
MAX_PER_PATCH = 3
MIN_SEP_M = 500.0


def class_codes(run_dir: Path, suffix: str) -> dict:
    """Class -> raster code, from the `.qml` sidecar OF THAT LAYER.

    **Never** assume the palette order. Codes follow the *sorted* class list and
    getting this wrong once counted stable Vegetation as change (CLAUDE.md).

    The sidecar has to be the one belonging to ``suffix``, and taking the first
    `*.qml` in the directory is not that: a run writes several layers, and on
    the deployed run the first match is `merged2` -- a FOUR-class legend whose
    codes mean different transitions from `coarse3`'s nine. Reading a coarse3
    raster through it returns `Artificial -> Cropland` as absent and silently
    draws points for whatever class happens to share the code.
    """
    import re
    qml = next(run_dir.glob(f"*_{suffix}.qml"), None)
    if qml is None:
        raise SystemExit(f"no *_{suffix}.qml sidecar in {run_dir}")
    text = qml.read_text()
    pairs = re.findall(r'value="(\d+)"[^>]*label="([^"]+)"', text)
    if not pairs:
        # QGIS writes the two attributes in either order depending on version,
        # so the fallback reads them reversed and swaps them back.
        pairs = [(value, label) for label, value in re.findall(
            r'label="([^"]+)"[^>]*value="(\d+)"', text)]
    return {label: int(value) for value, label in pairs}


def draw_from_rasters(run_dir: Path, suffix: str, wanted: dict,
                      seed: int) -> pd.DataFrame:
    """Stratified points inside blobs of ``wanted`` classes, one raster at a time."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import xy

    codes = class_codes(run_dir, suffix)
    rng = np.random.default_rng(seed)
    rows = []
    for path in sorted(run_dir.glob(f"*_{suffix}.tif")):
        pid = path.name.split("_")[0]
        with rasterio.open(path) as src:
            arr = src.read(1)
            transform, crs = src.transform, src.crs
        cell = max(int(round(MIN_SEP_M / abs(transform.a))), 1)
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        for cls in wanted:
            code = codes.get(cls)
            if code is None:
                continue
            rr, cc = np.nonzero(arr == code)
            if len(rr) == 0:
                continue
            # One point per `cell`-sized block, so a patch never contributes
            # several readings of the same field.
            keys = (rr // cell) * 10_000 + (cc // cell)
            _, first = np.unique(keys, return_index=True)
            pick = rng.permutation(first)[:MAX_PER_PATCH]
            for i in pick:
                x, y = xy(transform, int(rr[i]), int(cc[i]))
                lon, lat = to_wgs.transform(x, y)
                rows.append(dict(patch_id=pid, pred_class=cls, lon=lon, lat=lat,
                                 class_px_in_patch=int(len(rr)),
                                 fragment=bool(len(rr) < MIN_PX),
                                 source=run_dir.name, layer=suffix))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-select", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-fragments", action="store_true",
                    help="keep points from patches holding < 1 ha of the class. "
                         "On by default for `Artificial -> Cropland`, because "
                         "for a class this rare the fragments are the only "
                         "candidates that exist at all -- but they are marked.")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    mix = {k: v for k, v in DEFAULT_MIX.items() if v > 0}
    scale = args.n_select / sum(mix.values())
    quota = {k: int(round(v * scale)) for k, v in mix.items()}

    # --- the four classes the deployed run already drew --------------------
    dep = pd.read_csv(PATCHES / DEPLOYED_RUN / "label_points.csv")
    dep["source"], dep["layer"] = DEPLOYED_RUN, dep["channel"]
    dep = dep[["patch_id", "pred_class", "lon", "lat", "class_px_in_patch",
               "fragment", "source", "layer"]]

    # --- the dead class, from the state-pretrained run ---------------------
    siam = draw_from_rasters(PATCHES / SIAM_RUN, SIAM_LAYER,
                             ["Artificial -> Cropland"], args.seed)
    print(f"{DEPLOYED_RUN}: {len(dep)} drawn points")
    print(f"{SIAM_RUN}/{SIAM_LAYER}: {len(siam)} `Artificial -> Cropland` points "
          f"from {siam['patch_id'].nunique() if len(siam) else 0} patches")

    pool = pd.concat([dep[dep["pred_class"] != "Artificial -> Cropland"], siam],
                     ignore_index=True)

    rng = np.random.default_rng(args.seed)
    chosen = []
    for cls, k in quota.items():
        sub = pool[pool["pred_class"] == cls]
        # Prefer whole-hectare candidates; fall back to fragments only for the
        # class where fragments are all there is.
        solid = sub[~sub["fragment"]]
        if len(solid) >= k or not (args.allow_fragments
                                   or cls == "Artificial -> Cropland"):
            sub = solid
        if sub.empty:
            print(f"  !! {cls}: no candidates, {k} points unfilled")
            continue
        # Biggest blobs first -- a 3 ha patch of a class is a surer read than a
        # 1 ha one -- then randomise within to avoid taking three from one patch.
        sub = sub.sort_values("class_px_in_patch", ascending=False)
        take = sub.iloc[rng.permutation(min(len(sub), max(k * 3, k)))[:k]] \
            if len(sub) > k else sub
        if len(take) < k:
            print(f"  !  {cls}: {len(take)} of {k} available")
        chosen.append(take)

    out = pd.concat(chosen, ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out["id"] = [f"r{i:04d}" for i in range(len(out))]
    out["channel"] = "retrieval"
    out["rank"] = np.arange(1, len(out) + 1)
    out["cell_km"] = 5.0
    keep = ["id", "lon", "lat", "channel", "rank", "cell_km", "pred_class",
            "class_px_in_patch", "fragment", "source", "layer", "patch_id"]
    out[keep].to_csv(args.out, index=False)

    print(f"\n-> {args.out}  ({len(out)} points)")
    print(out.groupby("pred_class").agg(
        n=("id", "size"), fragments=("fragment", "sum"),
        patches=("patch_id", "nunique")).to_string())
    print("\nheld now vs this batch (if every point confirmed -- none will):")
    for cls in sorted(HELD, key=lambda c: HELD[c]):
        n = int((out["pred_class"] == cls).sum())
        print(f"  {cls:<24} {HELD[cls]:>4} +{n:<3} = {HELD[cls] + n:>4}"
              f"   ({n / HELD[cls]:+.0%})")


if __name__ == "__main__":
    main()
