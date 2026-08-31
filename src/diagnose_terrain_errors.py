"""Are the map's two stable-class errors terrain errors? -- and are they acquirable?

The user reports, from inspecting the deployed `s2off_centre_m3s3_bf` map:

    mountains      read as  Artificial -> Artificial
    wetland areas  read as  Cropland  -> Cropland

Neither shows up anywhere in the ledger, and that is not an oversight -- both
sides of both errors are **stable** classes, so `change_f1` cannot see them by
construction and `macro_f1` dilutes them across nine classes. They are only
visible on the map, which is exactly why they were found by looking at one.

This script asks three questions in order, and the third is the one that decides
what to do about them:

1. **Are they real in the held-out data?** Condition the 5-seed OOF prediction
   for true `Nature -> Nature` plots on slope, elevation, water occurrence and
   the ESA WorldCover class. If the misread rate does not rise with terrain, the
   map impression is a visual artefact of where the eye goes and there is
   nothing to fix.
2. **Is the model confident when it is wrong there?** A confident error and an
   uncertain one need opposite acquisitions -- uncertainty sampling reaches the
   second and is blind to the first.
3. **Is that terrain under-represented in the label set?** The acquisition
   question. Label density per terrain stratum against that stratum's share of
   global land: a stratum the campaign under-samples 5x is a coverage gap a
   diversity score can close, and one that is sampled at its land share is not.

Global land shares come from a 30,000-point equal-area draw sampled in Earth
Engine (`--refresh-land`), cached to
``data/analysis_results/terrain_land_share.json`` so the diagnostic runs offline
after the first call. Points, not a `reduceRegion` histogram: the global 1 km
histogram returns an empty dict rather than an error, which fails silently.

Run
---
    G=/home/geethen.singh/.pixi/envs/geo
    PROJ_DATA=$G/share/proj PROJ_LIB=$G/share/proj GDAL_DATA=$G/share/gdal \\
    /home/geethen.singh/.cache/phoenix-test/venv/bin/python \\
        src/diagnose_terrain_errors.py --seeds 5

Writes ``data/analysis_results/oof_terrain.parquet`` (one row per plot: truth,
5-seed OOF prediction, confidence, terrain) and prints the three tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from al_lab import build_context, fit_predict, spatial_folds
from project_paths import project_data_dir

RESULTS = project_data_dir("analysis_results")
OOF_OUT = RESULTS / "oof_terrain.parquet"
TERRAIN = RESULTS / "terrain_plots.parquet"
LAND_SHARE = RESULTS / "terrain_land_share.json"

#: ESA WorldCover v200 codes. 0 is the unmask fill, not a class.
WORLDCOVER = {0: "none", 10: "tree", 20: "shrub", 30: "grass", 40: "crop",
              50: "built", 60: "bare", 70: "snow/ice", 80: "water",
              90: "wetland", 95: "mangrove", 100: "moss/lichen"}

#: Bin edges shared by the plot tables and the global land histogram, so the two
#: are directly divisible. Changing one without the other silently produces a
#: ratio between different strata.
SLOPE_BINS = [-0.01, 1, 3, 6, 12, 90]
SLOPE_LABELS = ["0-1", "1-3", "3-6", "6-12", "12+"]
ELEV_BINS = [-500, 200, 600, 1200, 2500, 9000]
ELEV_LABELS = ["<200", "200-600", "600-1200", "1200-2500", "2500+"]
WATER_BINS = [-0.01, 0.5, 5, 20, 50, 101]
WATER_LABELS = ["0", "0-5", "5-20", "20-50", "50+"]

STABLE_NAT = "Nature -> Nature"
STABLE_ART = "Artificial -> Artificial"
STABLE_CROP = "Cropland -> Cropland"


# ---------------------------------------------------------------------------
def oof_with_terrain(n_seeds: int, n_folds: int, fold_seed: int) -> pd.DataFrame:
    """5-seed ensemble OOF on spatial folds, joined to the terrain covariates."""
    ctx = build_context()
    from twotower_lab import load_context
    pid = load_context().view("full").frame["PLOTID"].astype(str).to_numpy()

    folds = spatial_folds(ctx.frame["lon"].to_numpy(),
                          ctx.frame["lat"].to_numpy(), n_folds, fold_seed)
    oof = np.zeros((len(ctx.frame), len(ctx.fine_classes)))
    for f in range(n_folds):
        te, tr = np.flatnonzero(folds == f), np.flatnonzero(folds != f)
        for s in range(n_seeds):
            pf, _ = fit_predict(ctx, tr, te, s)
            oof[te] += pf / n_seeds
        print(f"  fold {f} done", flush=True)

    pred = np.asarray(ctx.fine_classes, dtype=object)[oof.argmax(1)]
    art_i = ctx.fine_classes.index(STABLE_ART)
    crop_i = ctx.fine_classes.index(STABLE_CROP)
    d = pd.DataFrame({"PLOTID": pid, "truth": ctx.target, "pred": pred,
                      "conf": oof.max(1), "p_art_stable": oof[:, art_i],
                      "p_crop_stable": oof[:, crop_i], "fold": folds})
    t = pd.read_parquet(TERRAIN)
    t["PLOTID"] = t["PLOTID"].astype(str)
    return d.merge(t, on="PLOTID", how="left")


def strata(d: pd.DataFrame) -> pd.DataFrame:
    """Attach the three binned terrain strata plus the WorldCover name."""
    return d.assign(
        slope_bin=pd.cut(d["slope"], SLOPE_BINS, labels=SLOPE_LABELS),
        elev_bin=pd.cut(d["elevation"], ELEV_BINS, labels=ELEV_LABELS),
        water_bin=pd.cut(d["water_occurrence"], WATER_BINS, labels=WATER_LABELS),
        wc=d["worldcover"].map(WORLDCOVER))


def error_table(d: pd.DataFrame, by: str) -> pd.DataFrame:
    """Misread rates for true `Nature -> Nature`, per stratum of ``by``."""
    nat = d[d["truth"] == STABLE_NAT].dropna(subset=[by])
    rows = []
    for k, g in nat.groupby(by, observed=True):
        as_art = g["pred"] == STABLE_ART
        rows.append(dict(
            stratum=str(k), n=len(g),
            correct=float((g["pred"] == STABLE_NAT).mean()),
            as_art=float(as_art.mean()),
            as_crop=float((g["pred"] == STABLE_CROP).mean()),
            # Question 2: when it makes the mountain error, how sure is it? An
            # uncertainty score can only reach the errors the model doubts.
            conf_when_art=float(g.loc[as_art, "conf"].mean()) if as_art.any()
            else np.nan))
    return pd.DataFrame(rows)


def coverage_table(d: pd.DataFrame, by: str, land: dict | None) -> pd.DataFrame:
    """Label share per stratum against that stratum's share of global land."""
    got = d[by].value_counts(normalize=True, dropna=True)
    rows = []
    for k in got.index:
        share_land = (land or {}).get(str(k))
        rows.append(dict(stratum=str(k), n=int((d[by] == k).sum()),
                         label_share=float(got[k]),
                         land_share=share_land,
                         ratio=(float(got[k]) / share_land)
                         if share_land else np.nan))
    return pd.DataFrame(rows).sort_values("stratum")


# ---------------------------------------------------------------------------
def land_shares(n_points: int = 30000, chunk: int = 2000,
                seed: int = 0) -> dict:
    """Global land share per stratum, from an equal-area point draw.

    **Not** a `reduceRegion` histogram. The obvious version -- one global
    `frequencyHistogram` at 1 km -- returns an empty dict here rather than an
    error, so it fails silently and produces a coverage table full of NaN that
    looks like a missing file. Points are also the right unit: they use the same
    frame as `global_patches.py` (uniform in longitude and in **sin(latitude)**,
    equal area on the sphere, cut at -60 deg), so a label share and a land share
    are shares of the same population.

    Non-land is rejected the way the patch sampler rejects it -- on the
    WorldCover footprint, minus permanent water.
    """
    import ee
    from extract_terrain_gee import init_gee, covariate_image
    init_gee()
    img = covariate_image()

    rng = np.random.default_rng(seed)
    s_lo, s_hi = np.sin(np.radians([-60.0, 84.0]))
    lon = rng.uniform(-180.0, 180.0, n_points)
    lat = np.degrees(np.arcsin(rng.uniform(s_lo, s_hi, n_points)))
    pts = pd.DataFrame({"PLOTID": np.arange(n_points).astype(str),
                        "lon": lon, "lat": lat})

    from extract_terrain_gee import sample_chunk
    parts = []
    for i in range(0, len(pts), chunk):
        parts.append(sample_chunk(img, pts.iloc[i:i + chunk], 1000))
        print(f"  land draw {min(i + chunk, len(pts)):6d}/{n_points}", flush=True)
    d = pd.concat(parts, ignore_index=True)
    d["elevation"] = d["elev_srtm"]
    d["slope"] = d["slope_srtm"]
    # `covariate_image` unmasks everything, so ocean comes back as a valid row
    # with worldcover 0. That is the rejection test.
    d = d[(d["worldcover"] > 0) & (d["worldcover"] != 80)]
    d = strata(d)
    print(f"  {len(d):,} of {n_points:,} draws on land ({len(d) / n_points:.1%})")
    return {axis: d[axis].value_counts(normalize=True, dropna=True)
            .rename(index=str).to_dict()
            for axis in ("slope_bin", "elev_bin", "water_bin", "wc")}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--refresh-land", action="store_true",
                    help="recompute the global land histogram from Earth Engine")
    ap.add_argument("--reuse-oof", action="store_true")
    args = ap.parse_args()

    if args.reuse_oof and OOF_OUT.exists():
        d = pd.read_parquet(OOF_OUT)
        print(f"reusing {OOF_OUT}")
    else:
        d = oof_with_terrain(args.seeds, args.folds, args.fold_seed)
        OOF_OUT.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(OOF_OUT, index=False)
        print(f"-> {OOF_OUT}")
    d = strata(d)

    land = None
    if args.refresh_land or not LAND_SHARE.exists():
        try:
            land = land_shares()
            LAND_SHARE.write_text(json.dumps(land, indent=2))
            print(f"-> {LAND_SHARE}")
        except Exception as exc:                              # noqa: BLE001
            print(f"(land shares unavailable: {type(exc).__name__} "
                  f"{str(exc)[:160]})")
    if land is None and LAND_SHARE.exists():
        land = json.loads(LAND_SHARE.read_text())

    nat = d[d["truth"] == STABLE_NAT]
    print(f"\n{len(d):,} plots | {len(nat):,} true {STABLE_NAT} "
          f"| {args.seeds}-seed OOF on {args.folds} spatial folds")
    print(f"  read correctly {(nat['pred'] == STABLE_NAT).mean():.3f} | "
          f"as {STABLE_ART} {(nat['pred'] == STABLE_ART).mean():.3f} | "
          f"as {STABLE_CROP} {(nat['pred'] == STABLE_CROP).mean():.3f}")

    fmt = lambda v: f"{v: .3f}"                               # noqa: E731
    for axis, title in (("slope_bin", "SLOPE (deg) -- the mountain hypothesis"),
                        ("elev_bin", "ELEVATION (m)"),
                        ("water_bin",
                         "WATER OCCURRENCE (% months) -- the wetland hypothesis"),
                        ("wc", "ESA WorldCover class at the plot")):
        print(f"\n=== {title} ===")
        t = error_table(d, axis)
        if axis == "wc":
            t = t[t["n"] >= 15].sort_values("as_art", ascending=False)
        print(t.to_string(index=False, float_format=fmt))
        if land:
            c = coverage_table(d.dropna(subset=[axis]), axis, land.get(axis))
            print("  coverage (label share / global land share):")
            print(c.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
