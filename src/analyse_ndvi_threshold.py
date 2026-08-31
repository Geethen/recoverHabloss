"""Calibrate an NDVI vegetation threshold against the labelled stable plots.

The user's hypothesis: a hard NDVI cut, calibrated on stable-Nature points,
separates vegetated from built ground, and it lands near 0.3.

This is worth testing rather than assuming, because it is the one lever the
Tessera search explicitly named and never built. `TWOTOWER_RESEARCH.md` F6
closed section F with: *"If more accuracy on built-up is genuinely needed, the
lever is a built-fraction covariate or a sub-pixel label, not another fusion."*
A thresholded NDVI patch **is** a built-fraction covariate -- the fraction of the
64x64 patch below the cut -- so this converges on the documented conclusion from
the other direction.

Three questions, in order:

1. **Where is the threshold, empirically?** Fit it on stable-Vegetation vs
   stable-Artificial centre pixels and report the optimum under several criteria
   (Youden's J, balanced accuracy, F1), not just one.
2. **How separable are the classes at all?** A threshold is only worth carrying
   if the distributions actually part. AUC answers that independently of where
   the cut is put.
3. **Does it see what AlphaEarth misses?** F6 found 62% of the misread built-up
   plots sit closer to the Vegetation centroid in AlphaEarth space. If NDVI
   separates *those specific plots*, it is new information; if it fails on them
   too, it is a label problem and no feature will fix it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHARDS = REPO_ROOT / "data" / "embeddings" / "s2_shards"
OUT = REPO_ROOT / "data" / "analysis_results"

STABLE_VEG = "Vegetation -> Vegetation"
STABLE_ART = "Artificial -> Artificial"


def ndvi_from_patches(patches: np.ndarray) -> np.ndarray:
    """(n, 2, 4, H, W) reflectance -> (n, 2, H, W) NDVI."""
    red = patches[:, :, 2]
    nir = patches[:, :, 3]
    return (nir - red) / (nir + red + 1e-6)


def threshold_scan(veg: np.ndarray, art: np.ndarray, grid: np.ndarray):
    """Separation quality of 'NDVI < t means built' over a grid of cuts."""
    rows = []
    for t in grid:
        # Positive class = Artificial (built): predicted built when NDVI < t.
        tp = float((art < t).sum())
        fn = float((art >= t).sum())
        fp = float((veg < t).sum())
        tn = float((veg >= t).sum())
        sens = tp / max(tp + fn, 1)       # built correctly called built
        spec = tn / max(tn + fp, 1)       # vegetation correctly left alone
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        rows.append({"threshold": t, "sensitivity": sens, "specificity": spec,
                     "youden_j": sens + spec - 1, "bal_acc": (sens + spec) / 2,
                     "precision": prec, "f1": f1})
    return pd.DataFrame(rows)


def auc(veg: np.ndarray, art: np.ndarray) -> float:
    """P(a random built plot has lower NDVI than a random vegetated one).

    Ties take **mid-ranks** (`rankdata`), which is the definition -- a tie is
    half a win, not a win. The first version of this ranked with
    `ranks[argsort(values)] = arange(1, n+1)`, giving equal values *distinct*
    ranks in array order; since `values` puts the Artificial class first, every
    tie resolved in its favour and the score came out too high.

    That is not a rounding-level detail on this data. Built fraction over a
    3x3 window takes only ten distinct values, so it is almost all ties, and
    the bug inflated it from a true 0.686 to 0.762 -- which is how the radius
    sweep came to show a decisive 3 px optimum that is really a dead heat with
    5 px (0.685). Columns with few ties were unaffected to within 0.001, which
    is why the error survived: only the winner was wrong. See section U4 of
    `docs/research/S2_DETAIL_RESEARCH.md`.
    """
    from scipy.stats import rankdata

    values = np.concatenate([art, veg])
    labels = np.concatenate([np.ones(len(art)), np.zeros(len(veg))])
    ranks = rankdata(values)
    n1, n0 = labels.sum(), (1 - labels).sum()
    # Built should sit LOW, so invert the usual direction.
    return 1.0 - (ranks[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARDS)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    import build_s2_features as B
    import twotower_lab as L

    patches, plotid, years = B.load_shards(args.shard_dir)
    ndvi = ndvi_from_patches(patches)
    yi = list(years).index(args.year)
    centre = ndvi.shape[-1] // 2

    ctx = L.load_context()
    view = ctx.view("full")
    frame = view.frame.copy()
    frame["_merged"] = view.truth_merged
    lookup = pd.DataFrame({"PLOTID": plotid,
                           "ndvi_centre": ndvi[:, yi, centre, centre]})
    # Built fraction at a grid of cuts is computed later; carry the patch too.
    joined = frame[["PLOTID", "_merged"]].merge(lookup, on="PLOTID", how="inner")
    index = {p: i for i, p in enumerate(plotid)}
    joined["_row"] = [index[p] for p in joined["PLOTID"]]

    veg = joined.loc[joined["_merged"] == STABLE_VEG, "ndvi_centre"].to_numpy()
    art = joined.loc[joined["_merged"] == STABLE_ART, "ndvi_centre"].to_numpy()
    veg = veg[np.isfinite(veg)]
    art = art[np.isfinite(art)]

    print(f"=== Centre-pixel NDVI {args.year} ===")
    print(f"stable Vegetation n={len(veg):,}  stable Artificial n={len(art):,}")
    for name, arr in (("stable Vegetation", veg), ("stable Artificial", art)):
        q = np.percentile(arr, [5, 25, 50, 75, 95])
        print(f"  {name:18s} median {np.median(arr):.3f}  "
              f"IQR {q[1]:.3f}-{q[3]:.3f}  p5-p95 {q[0]:.3f}-{q[4]:.3f}")
    print(f"  AUC (separability, 0.5 = none): {auc(veg, art):.3f}")

    grid = np.round(np.arange(0.05, 0.81, 0.01), 2)
    scan = threshold_scan(veg, art, grid)
    print("\n=== Optimal cut by criterion ===")
    for crit in ("youden_j", "bal_acc", "f1"):
        best = scan.loc[scan[crit].idxmax()]
        print(f"  {crit:9s} -> t={best['threshold']:.2f}  "
              f"sens={best['sensitivity']:.3f} spec={best['specificity']:.3f} "
              f"balacc={best['bal_acc']:.3f}")
    at30 = scan.loc[scan["threshold"] == 0.30].iloc[0]
    print(f"\n  user's t=0.30 -> sens={at30['sensitivity']:.3f} "
          f"spec={at30['specificity']:.3f} balacc={at30['bal_acc']:.3f} "
          f"J={at30['youden_j']:.3f}")

    # -- the decisive question: does NDVI see the plots AlphaEarth misreads? ---
    errors_path = OUT / "stable_artificial_errors.csv"
    if errors_path.exists():
        err = pd.read_csv(errors_path)
        key = "PLOTID" if "PLOTID" in err.columns else err.columns[0]
        misread = set(err[key].astype(str))
        art_rows = joined[joined["_merged"] == STABLE_ART].copy()
        art_rows["is_misread"] = art_rows["PLOTID"].astype(str).isin(misread)
        bad = art_rows.loc[art_rows["is_misread"], "ndvi_centre"].dropna()
        good = art_rows.loc[~art_rows["is_misread"], "ndvi_centre"].dropna()
        if len(bad):
            best_t = scan.loc[scan["youden_j"].idxmax(), "threshold"]
            print(f"\n=== Do the {len(bad)} AlphaEarth-misread built-up plots "
                  f"look built to NDVI? ===")
            print(f"  correctly-read built-up: median NDVI {np.median(good):.3f}, "
                  f"{(good < best_t).mean():.1%} below t={best_t:.2f}")
            print(f"  MISREAD built-up:        median NDVI {np.median(bad):.3f}, "
                  f"{(bad < best_t).mean():.1%} below t={best_t:.2f}")
            print(f"  stable Vegetation:       median NDVI {np.median(veg):.3f}, "
                  f"{(veg < best_t).mean():.1%} below t={best_t:.2f}")
            print(f"  AUC on the misread subset alone: {auc(veg, bad.to_numpy()):.3f}")

    # -- built fraction over the patch, the actual covariate F6 asked for -------
    print("\n=== Built fraction (share of the 64x64 patch below the cut) ===")
    best_t = scan.loc[scan["youden_j"].idxmax(), "threshold"]
    rows = joined["_row"].to_numpy()
    frac = np.nanmean(ndvi[rows, yi] < best_t, axis=(1, 2))
    joined["built_frac"] = frac
    for cls in (STABLE_ART, STABLE_VEG):
        sub = joined.loc[joined["_merged"] == cls, "built_frac"]
        print(f"  {cls:26s} median {sub.median():.3f}  IQR "
              f"{sub.quantile(.25):.3f}-{sub.quantile(.75):.3f}")
    bf_auc = auc(joined.loc[joined["_merged"] == STABLE_VEG, "built_frac"].to_numpy(),
                 joined.loc[joined["_merged"] == STABLE_ART, "built_frac"].to_numpy())
    print(f"  AUC of built fraction: {1 - bf_auc:.3f} "
          f"(vs centre-pixel NDVI {auc(veg, art):.3f})")

    OUT.mkdir(parents=True, exist_ok=True)
    scan.to_csv(OUT / f"ndvi_threshold_scan_{args.year}.csv", index=False)
    print(f"\nscan -> {OUT / f'ndvi_threshold_scan_{args.year}.csv'}")


if __name__ == "__main__":
    main()
