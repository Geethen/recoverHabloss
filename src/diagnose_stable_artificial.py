"""Where do the misread stable-Artificial plots live? (backlog F6)

Every modelling lever aimed at the stable-Artificial / stable-Vegetation
confusion has now come back flat: extra supervision does nothing (F1), and the
calibration correction is exhausted (F3 and F7 turn out to fix the same thing).
So the residual ~20% is neither an objective nor an operating-point problem, and
the next question is not "which architecture" but **what are these plots**.

This script takes the deployed model's seed-ensembled out-of-fold probabilities
and anatomises the errors on the 979 true ``Artificial -> Artificial`` plots:

* **confidence** -- are they narrowly lost or confidently wrong? A narrow loss is
  a model that can be pushed; a confident one is a plot whose features say
  Vegetation.
* **which Vegetation** -- Nature or Cropland underneath the merged label. A
  built-up plot read as Cropland is a different failure from one read as forest.
* **spatial concentration** -- per block, and the Gini of the error over blocks.
  If a handful of the 83 spatial blocks carry the errors, the fix is data.
* **source and Tessera coverage** -- is one interpreter campaign, or the absence
  of the detail modality, carrying the confusion?
* **feature-space position** -- each error's AlphaEarth 2024 vector against the
  class centroids of stable Artificial and stable Vegetation. If the errors sit
  closer to the Vegetation centroid than to their own, the label and the pixel
  disagree and no classifier reading that pixel can win.

Usage::

    python diagnose_stable_artificial.py
    python diagnose_stable_artificial.py --model seed_ensemble_mc
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import twotower_lab as lab
from project_paths import project_data_dir
from twotower_metrics import ART_STABLE, VEG_STABLE

OUT = project_data_dir("analysis_results") / "stable_artificial_errors.csv"


def ensembled(view: lab.View, source: str) -> np.ndarray:
    stack = [c[0] for c in (lab.load_oof(source, view.name, s) for s in range(5))
             if c is not None]
    if not stack:
        raise SystemExit(f"no cached OOF for {source} on {view.name}")
    return np.mean(stack, axis=0)


def gini(counts: np.ndarray) -> float:
    """Concentration of the errors over blocks: 0 = spread evenly, 1 = one block."""
    x = np.sort(np.asarray(counts, dtype="float64"))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def centroid_report(frame: pd.DataFrame, cols: list, truth: np.ndarray,
                    err: np.ndarray) -> dict:
    """Cosine distance of each group to the two stable-class centroids.

    Centroids are built from the *correctly classified* plots of each class, so
    the errors are measured against a clean definition of what each class looks
    like rather than against a mean they themselves polluted.
    """
    X = frame[cols].to_numpy("float64")
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    art_ok = (truth == ART_STABLE) & ~err
    veg = truth == VEG_STABLE
    c_art = X[art_ok].mean(0)
    c_veg = X[veg].mean(0)
    c_art /= np.linalg.norm(c_art)
    c_veg /= np.linalg.norm(c_veg)

    out = {}
    for name, mask in (("art_correct", art_ok), ("art_error", err),
                       ("veg_stable", veg)):
        if not mask.any():
            continue
        d_art = 1 - X[mask] @ c_art
        d_veg = 1 - X[mask] @ c_veg
        out[name] = {
            "n": int(mask.sum()),
            "to_artificial": round(float(d_art.mean()), 4),
            "to_vegetation": round(float(d_veg.mean()), 4),
            # Per-plot, not just on the group mean: a group mean can sit closer
            # to Artificial while most of its members do not.
            "share_closer_to_veg": round(float((d_veg < d_art).mean()), 3),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mc_dropout_scalars")
    parser.add_argument("--read", default="full", choices=["full", "subset"])
    args = parser.parse_args()

    ctx = lab.load_context()
    view = ctx.view(args.read)
    frame = view.frame
    classes = view.merged_classes
    probs = ensembled(view, args.model)
    pred = lab.labels_from_probs(probs, classes)
    truth = view.truth_merged

    art = truth == ART_STABLE
    err = art & (pred == VEG_STABLE)
    other_err = art & (pred != ART_STABLE) & (pred != VEG_STABLE)
    print(f"{args.model} · {args.read} · {len(frame):,} plots")
    print(f"stable Artificial: {int(art.sum())} plots · "
          f"{int((art & (pred == ART_STABLE)).sum())} correct · "
          f"{int(err.sum())} read as stable Vegetation · "
          f"{int(other_err.sum())} read as something else\n")

    # -- 1. confident or narrow? --------------------------------------------
    i_art = classes.index(ART_STABLE)
    i_veg = classes.index(VEG_STABLE)
    margin = probs[:, i_veg] - probs[:, i_art]
    print("CONFIDENCE of the misread plots (P[stable Veg] − P[stable Art])")
    q = np.quantile(margin[err], [0.25, 0.5, 0.75])
    print(f"  quartiles {q[0]:.3f} / {q[1]:.3f} / {q[2]:.3f}")
    for cut in (0.1, 0.2, 0.4):
        share = float((margin[err] < cut).mean())
        print(f"  within {cut:.1f} of flipping: {share:6.1%} "
              f"({int((margin[err] < cut).sum())} plots)")
    print(f"  P(stable Artificial) on the errors, mean {probs[err, i_art].mean():.3f}\n")

    # -- 2. which Vegetation, underneath the merged label? ------------------
    print("WHAT the misread plots were called, on the fine legend")
    fine_pred = pd.Series(np.asarray(view.truth_fine)[err]).value_counts()
    print("  their true coarse3 transition:")
    for k, v in fine_pred.items():
        print(f"    {k:34s} {v:4d}")
    print("  their 2024 interpreted cover:")
    for k, v in frame.loc[err, "lc_2024"].value_counts().items():
        print(f"    {k:34s} {v:4d}")
    print()

    # -- 3. spatial concentration -------------------------------------------
    per_block = pd.DataFrame({
        "block": frame["block_id"], "art": art, "err": err,
    }).groupby("block").sum()
    per_block = per_block[per_block["art"] > 0]
    per_block["rate"] = per_block["err"] / per_block["art"]
    print(f"SPATIAL: {len(per_block)} of {frame['block_id'].nunique()} blocks hold "
          f"stable-Artificial plots")
    print(f"  Gini of the error count over those blocks: "
          f"{gini(per_block['err'].to_numpy()):.3f}")
    top = per_block.sort_values("err", ascending=False).head(6)
    share = top["err"].sum() / max(per_block["err"].sum(), 1)
    print(f"  worst 6 blocks hold {int(top['err'].sum())} of "
          f"{int(per_block['err'].sum())} errors ({share:.0%}):")
    for blk, row in top.iterrows():
        print(f"    block {str(blk):>12s}  {int(row['err']):3d}/{int(row['art']):3d} "
              f"= {row['rate']:.0%}")
    print()

    # -- 4. source campaign and Tessera coverage ----------------------------
    for col in ("source",):
        if col not in frame:
            continue
        tab = pd.DataFrame({"art": art, "err": err, col: frame[col]}).groupby(col).sum()
        tab["rate"] = tab["err"] / tab["art"].clip(lower=1)
        print(f"BY {col.upper()}")
        print(tab[tab["art"] > 0].to_string())
        print()
    has_t = frame["tess_present"].to_numpy() > 0.5
    for tag, mask in (("Tessera present", has_t), ("Tessera absent", ~has_t)):
        m = art & mask
        if m.sum():
            print(f"  {tag:16s} {int((err & mask).sum()):3d}/{int(m.sum()):3d} "
                  f"= {(err & mask).sum() / m.sum():.1%} misread")
    print()

    # -- 5. feature space ----------------------------------------------------
    aef24 = sorted(c for c in ctx.aef_cols if c.endswith("_2024"))
    rep = centroid_report(frame, aef24, truth, err)
    print("FEATURE SPACE · cosine distance to the class centroids (AlphaEarth 2024)")
    print(f"  {'group':14s} {'n':>5s} {'→Artificial':>12s} {'→Vegetation':>12s} "
          f"{'% closer to Veg':>16s}")
    for name, r in rep.items():
        print(f"  {name:14s} {r['n']:5d} {r['to_artificial']:12.4f} "
              f"{r['to_vegetation']:12.4f} {r['share_closer_to_veg']:15.1%}")
    print()

    frame.loc[err, ["PLOTID", "lon", "lat", "block_id", "lc_2018", "lc_2024"]].assign(
        p_artificial=probs[err, i_art], p_vegetation=probs[err, i_veg],
        tess_present=has_t[err],
    ).to_csv(OUT, index=False)
    print(f"{int(err.sum())} misread plots -> {OUT}")


if __name__ == "__main__":
    main()
