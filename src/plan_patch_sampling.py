"""Size the next labelling round from the 100-patch pilot.

Three questions, in order, each answered from the patch statistics that
`infer_patches.py` writes:

**A. How much of each class does the map actually put on the ground?**
    A design-based ratio estimate over the patch sample, with a standard error
    from the between-patch variance. The patch, not the pixel, is the sampling
    unit: 250,000 pixels inside one 5x5 km patch are nowhere near 250,000
    independent observations, and treating them as such would understate the
    standard error by well over an order of magnitude and produce a confidently
    wrong patch count.

**B. How many patches must be drawn to label enough of the classes that need
    it?**
    Not "how many pixels" -- a labeller cannot use a class that occupies 40
    scattered pixels in a patch. A patch is *usable* for class c when it holds
    at least `--min-px` contiguous-ish pixels of it, and it yields at most
    `--max-points` label points, because points in the same patch a few hundred
    metres apart are close to the same observation. The yield per patch is
    therefore

        y_i(c) = min(max_points, floor(px_i(c) / min_px))

    and the patches needed for a target of T_c new plots is T_c / mean_i y_i(c).
    The answer is driven by the rarest target class and by nothing else.

**C. Which patches from an oversample are worth labelling?**
    Rank on novelty and entropy, but only among patches that are *eligible* --
    that hold enough of a class the round is trying to collect. Ranking without
    that filter selects spectacular terrain that contains none of the classes in
    deficit. Novelty and entropy enter as rank-normalised columns and are
    reported alongside the composite so a selection can be argued with.

Which classes are in deficit
----------------------------
Taken from the learning curves (`data/analysis_results/learning_curves.csv`),
not asserted here. A class is a target if it is a *change* class, because every
one of them is either the product's purpose or starved of labels:

    Artificial -> Cropland     46 plots   OOF F1 0.000   dead
    Cropland   -> Nature      114 plots   OOF F1 0.014   dead
    Artificial -> Nature      123 plots   OOF F1 0.407   still climbing
    Nature     -> Cropland    243 plots   OOF F1 0.310   still climbing
    Cropland   -> Artificial  333 plots   OOF F1 0.591
    Nature     -> Artificial  383 plots   OOF F1 0.469   the headline class

The three stable classes hold 5,172 of the 6,414 plots and their curves have
flattened; more of them buys nothing. The default target is to **double** each
change class, which the curves price at about +0.026 change-F1 per doubling.

Run
---
    python src/plan_patch_sampling.py                       # A + B + C
    python src/plan_patch_sampling.py --draw-points         # also emit labels
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import project_data_dir

#: Classes the next round is collecting, with why. Every change class qualifies.
TARGETS = ("Artificial -> Cropland", "Cropland -> Nature",
           "Artificial -> Nature", "Nature -> Cropland",
           "Cropland -> Artificial", "Nature -> Artificial")
#: Pixels of a class a patch must hold before a labeller can use it. 100 px at
#: 10 m is 1 ha -- about the smallest patch of anything a human will confidently
#: call from imagery, and the size the existing plots are drawn at.
MIN_PX = 100
#: Label points to take from one patch for one class. Points 5 km apart at most
#: are correlated; taking twenty of them buys far less than twenty plots.
MAX_POINTS = 3
#: Minimum spacing between label points inside a patch, in metres.
MIN_SEP_M = 500.0


def class_px(stats: pd.DataFrame, cls: str) -> np.ndarray:
    """Pixels of ``cls`` per patch, with absent counts read as zero.

    An all-water patch carries no ``px_*`` columns at all, so the column is
    missing for it rather than zero. Both mean the same thing here -- the patch
    holds none of that class -- and a NaN left in place would silently drop the
    patch from the denominator of every proportion.
    """
    col = stats.get(f"px_{cls}")
    if col is None:
        return np.zeros(len(stats), "float64")
    return col.fillna(0.0).to_numpy("float64")


def retrieval_px(stats: pd.DataFrame, cls: str, channel: str) -> np.ndarray:
    """Per-patch pixel count for one class under one retrieval channel.

    ``argmax`` is the map -- the class the model actually assigns. ``topchange``
    is the arg-max restricted to the six change classes, which exists because
    two of them are never the unrestricted arg-max anywhere and so cannot be
    navigated to at all. See `infer_patches.run_patch`.
    """
    prefix = {"argmax": "px_", "topchange": "pxtc_"}[channel]
    col = stats.get(f"{prefix}{cls}")
    if col is None:
        return np.zeros(len(stats), "float64")
    return col.fillna(0.0).to_numpy("float64")


def latest_run(root: Path) -> Path:
    runs = sorted(p for p in root.glob("patches_*") if (p / "patch_stats.parquet").exists())
    if not runs:
        raise SystemExit(f"no completed patch runs under {root}")
    return runs[-1]


def label_state(curves: Path) -> pd.DataFrame:
    """Per coarse3 class: current plot count and out-of-fold precision.

    ``support`` at ``frac=1.0`` is the full label count (``n_train_cls`` is the
    per-fold training count and so is only ``1 - 1/k`` of it).

    Precision is here because it is what converts a *predicted* pixel into an
    expected *confirmed* plot, and leaving it out is the single easiest way to
    plan a labelling round that comes up short. Stratifying on the map is the
    right way to find rare classes -- there is no other way to find them -- but
    the map's change classes run at precision 0.30-0.51, so a labeller sent to
    three predicted `Nature -> Artificial` points brings back about 1.4 of them.
    The sizing below therefore reports both the predicted-point yield and the
    confirmed-plot yield, and plans against the second.
    """
    curve = pd.read_csv(curves)
    full = curve[(curve.frac == 1.0) & (curve.level == "coarse3")
                 & (curve.split == "oof")]
    out = full.groupby("cls")[["support", "precision"]].mean()
    return out.drop(index=["CHANGE", "MACRO"], errors="ignore")


# --------------------------------------------------------------------------
# A. Class composition of the mapped land surface
# --------------------------------------------------------------------------
def composition(stats: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Ratio estimate of each class's share of land, with a patch-level SE.

    ``p_hat = sum_i y_i / sum_i m_i`` over patches, whose variance is the
    standard ratio-estimator form

        var(p) = n / (n - 1) * sum_i (y_i - p m_i)^2 / (sum_i m_i)^2

    A patch of open water contributes ``m_i = 0`` and drops out of both sums
    without being deleted from the frame, which is the correct handling: it is
    a real draw that happened to hold no mappable land.
    """
    m = stats["valid_px"].to_numpy("float64")
    n = len(stats)
    total = m.sum()
    rows = []
    for cls in classes:
        y = class_px(stats, cls)
        p = y.sum() / total
        resid = y - p * m
        var = n / (n - 1) * (resid ** 2).sum() / (total ** 2)
        se = math.sqrt(max(var, 0.0))
        rows.append({
            "cls": cls,
            "px": int(y.sum()),
            "prop": p,
            "se": se,
            "lo95": max(p - 1.96 * se, 0.0),
            "hi95": p + 1.96 * se,
            "km2_per_patch": p * total / n * 1e-4 * 1e-2,
            "patches_present": int((y > 0).sum()),
            "patches_usable": int((y >= MIN_PX).sum()),
        })
    out = pd.DataFrame(rows).set_index("cls")
    # Effective sample size for the proportion, as a plain binomial would have
    # to have to give this SE. It is the number that says how badly pixels
    # overstate the information: the pilot has 25 million of them.
    out["n_eff"] = (out.prop * (1 - out.prop) / out.se.pow(2)).replace(
        [np.inf, -np.inf], np.nan)
    return out


# --------------------------------------------------------------------------
# B. Patches needed
# --------------------------------------------------------------------------
def patch_yield(stats: pd.DataFrame, cls: str, min_px: int, max_points: int,
                channel: str = "argmax"):
    """Per-patch label-point yield for one class, and its mean and SE."""
    y = retrieval_px(stats, cls, channel)
    per = np.minimum(max_points, np.floor(y / min_px))
    n = len(per)
    return per, per.mean(), per.std(ddof=1) / math.sqrt(n)


def sizing(stats: pd.DataFrame, state: pd.DataFrame, targets, min_px,
           max_points, target_mode="double", target_n=0) -> pd.DataFrame:
    """Patches needed per target class, on whichever channel can reach it.

    A class is sized on the **map** (`argmax`) when the map puts enough of it on
    the ground to navigate to. When it does not -- and for two classes it does
    not, anywhere in 25 million pixels -- the row falls back to the
    change-restricted channel and is marked as such.

    That fallback is deliberately left **unpriced**. The argmax rows are scaled
    by the model's own out-of-fold precision, which is measured; there is no
    measured confirm rate for the change-restricted channel, and inventing one
    would put a fabricated number at the exact place the plan is weakest. Its
    `patches_needed` is therefore a *lower* bound -- what it would cost if every
    candidate confirmed, which none of them will. Measuring the real rate is a
    small job (`change-restricted arg-max over the existing OOF cache`) and
    should happen before that row is used to commit anyone's time.
    """
    rows = []
    for cls in targets:
        have = int(state["support"].get(cls, 0))
        want = have if target_mode == "double" else max(target_n - have, 0)
        per, mean, se = patch_yield(stats, cls, min_px, max_points, "argmax")
        channel, prec = "argmax", float(state["precision"].get(cls, np.nan))
        if mean == 0.0:
            per, mean, se = patch_yield(stats, cls, min_px, max_points,
                                        "topchange")
            channel, prec = "topchange", np.nan
        # Predicted points -> confirmed plots of this class. A point the
        # labeller rejects is not wasted -- it becomes a plot of whatever it
        # really is, which is why this scales the *yield* rather than being
        # charged as an overhead.
        scale = 1.0 if channel == "topchange" else prec
        mean_c, se_c = mean * scale, se * scale
        # Upper bound on the patch count from the lower end of the yield's own
        # confidence interval: the pilot estimates the yield from 100 draws and
        # a rare class's yield is estimated from the handful of patches that
        # held any of it, so the point estimate alone is not a plan.
        #
        # When that lower end reaches zero the pilot simply does not bound the
        # requirement from below, and the honest report is `inf` rather than a
        # number produced by an epsilon floor. A class in that state needs a
        # bigger pilot before it can be planned for at all.
        lo = mean_c - 1.96 * se_c
        rows.append({
            "cls": cls, "have": have, "want": want,
            "channel": channel, "precision": prec,
            "usable_patches": int((per > 0).sum()),
            "usable_rate": float((per > 0).mean()),
            "pts_per_patch": mean,
            "plots_per_patch": mean_c,
            "patches_needed": math.ceil(want / mean_c) if mean_c > 0 else np.inf,
            "patches_needed_hi": math.ceil(want / lo) if lo > 0 else np.inf,
        })
    return pd.DataFrame(rows).set_index("cls")


# --------------------------------------------------------------------------
# C. Ranking an oversample
# --------------------------------------------------------------------------
def rank_patches(stats: pd.DataFrame, targets, min_px, channels=None,
                 w_novelty=0.5, w_entropy=0.5) -> pd.DataFrame:
    """Eligible patches, ranked by rank-normalised novelty and entropy.

    Eligibility comes first and is not a tiebreak: a patch that holds none of
    the classes in deficit cannot supply a label for them however novel it is.
    Among eligible patches the two axes are averaged after being converted to
    percentile ranks, because they are on incomparable scales (a cosine distance
    and a normalised entropy) and a raw sum would let whichever has the wider
    spread decide the whole ordering.

    `novelty_p90` is the novelty axis rather than `novelty_mean`: the pocket of
    unfamiliar land inside an otherwise ordinary patch is the thing worth
    sending someone to look at. `entropy_change` is the entropy axis for the
    matching reason -- entropy averaged over a patch that is 95% confidently
    stable measures the stable class's confidence, not the change class's.
    """
    channels = channels or {}
    work = stats[stats["valid_px"] > 0].copy()
    have = np.zeros(len(work), dtype=float)
    for cls in targets:
        # Each class is counted on the channel it is actually sized on, so a
        # patch that can only supply a dead class through the change-restricted
        # layer still counts as eligible for it.
        px = retrieval_px(work, cls, channels.get(cls, "argmax"))
        have += (px >= min_px).astype(float)
    work["n_target_classes"] = have
    work["eligible"] = have > 0

    # Fall back to the whole-patch entropy where a patch has no change pixels to
    # average over, so an eligible patch is never dropped for a missing column.
    ent = work["entropy_change"].fillna(work["entropy_mean"])
    work["r_novelty"] = work["novelty_p90"].rank(pct=True)
    work["r_entropy"] = ent.rank(pct=True)
    work["score"] = (w_novelty * work["r_novelty"]
                     + w_entropy * work["r_entropy"])
    work.loc[~work["eligible"], "score"] = np.nan
    return work.sort_values("score", ascending=False)


# --------------------------------------------------------------------------
# D. Label points inside the selected patches
# --------------------------------------------------------------------------
def draw_points(run_dir: Path, selected: pd.DataFrame, fine_classes: list[str],
                targets, max_points, min_sep_m, channels=None, min_px=MIN_PX,
                seed=0) -> pd.DataFrame:
    """Stratified label points from each selected patch, per target class.

    Each class is drawn from the layer it is sized on: `*_coarse3.tif` for the
    classes the map carries, `*_topchange.tif` for the ones it does not. Points
    are thinned to one per ``min_sep_m`` cell so a patch never contributes
    several readings of the same field.

    The class written on a point is the *model's* call. It is a stratification
    label and not a truth value -- the labeller's job is to overrule it, and the
    precision adjustment in B is the arithmetic of expecting them to.

    ``fragment`` marks a point drawn from a patch holding less than ``min_px``
    of that class. Those points are below the size the sizing model counts, and
    they are kept rather than dropped because for a dead class the fragments are
    the only candidates that exist at all -- but they are marked, because a
    sub-hectare speck is the case most likely to be a model artefact.
    """
    import rasterio

    channels = channels or {}
    rng = np.random.default_rng(seed)
    code_of = {c: i for i, c in enumerate(fine_classes)}
    stats = selected.set_index("patch_id")
    rows = []
    for pid in selected["patch_id"]:
        # Both layers for this patch, read once. They share a grid and a class
        # list, so one transform serves both.
        layers, transform, crs = {}, None, None
        for suffix in ("coarse3", "topchange"):
            path = run_dir / f"{pid}_{suffix}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as src:
                layers[suffix] = src.read(1)
                transform, crs = src.transform, src.crs
        if transform is None:
            continue
        cell = max(int(round(min_sep_m / abs(transform.a))), 1)
        for cls in targets:
            code = code_of.get(cls)
            if code is None:
                continue
            channel = channels.get(cls, "argmax")
            codes = layers.get("coarse3" if channel == "argmax" else "topchange")
            if codes is None:
                continue
            n_px = float(retrieval_px(stats.loc[[pid]], cls, channel)[0])
            ys, xs = np.nonzero(codes == code)
            if ys.size == 0:
                continue
            # One candidate per min_sep cell, chosen at random within it.
            key = (ys // cell) * (codes.shape[1] // cell + 1) + (xs // cell)
            order = rng.permutation(ys.size)
            ys, xs, key = ys[order], xs[order], key[order]
            _, first = np.unique(key, return_index=True)
            take = first[:max_points]
            for y, x in zip(ys[take], xs[take]):
                px, py = transform * (float(x) + 0.5, float(y) + 0.5)
                rows.append({"patch_id": pid, "pred_class": cls,
                             "channel": channel, "class_px_in_patch": int(n_px),
                             "fragment": bool(n_px < min_px),
                             "row": int(y), "col": int(x),
                             "x": px, "y": py, "epsg": crs.to_epsg()})
    points = pd.DataFrame(rows)
    if points.empty:
        return points
    # One reprojection per zone, so the labelling file carries lon/lat too.
    from rasterio.warp import transform as warp_transform
    lon, lat = [], []
    for epsg, grp in points.groupby("epsg", sort=False):
        gx, gy = warp_transform(f"EPSG:{epsg}", "EPSG:4326",
                                grp.x.tolist(), grp.y.tolist())
        lon.append(pd.Series(gx, index=grp.index))
        lat.append(pd.Series(gy, index=grp.index))
    points["lon"] = pd.concat(lon).sort_index()
    points["lat"] = pd.concat(lat).sort_index()
    return points


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="an infer_patches output dir (default: latest)")
    parser.add_argument("--curves", type=Path,
                        default=project_data_dir("analysis_results",
                                                 "learning_curves.csv"))
    parser.add_argument("--min-px", type=int, default=MIN_PX)
    parser.add_argument("--max-points", type=int, default=MAX_POINTS)
    parser.add_argument("--min-sep-m", type=float, default=MIN_SEP_M)
    parser.add_argument("--target-mode", choices=["double", "absolute"],
                        default="double")
    parser.add_argument("--target-n", type=int, default=500,
                        help="with --target-mode absolute: plots per class")
    parser.add_argument("--oversample", type=float, default=3.0,
                        help="draw this many times the needed patches, then "
                             "keep the top-ranked ones")
    parser.add_argument("--draw-points", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    pd.set_option("display.width", 200)

    run_dir = args.run_dir or latest_run(project_data_dir("patches"))
    stats = pd.read_parquet(run_dir / "patch_stats.parquet")
    meta = json.loads((run_dir / "meta.json").read_text())
    fine_classes = meta["coarse3_classes"]
    failed = stats["error"].notna().sum() if "error" in stats else 0
    stats = stats[stats.get("valid_px", pd.Series(1, index=stats.index)).notna()]
    state = label_state(args.curves)

    print(f"run   {run_dir.name}   model {meta['model']} x{meta['seeds']} seeds")
    print(f"{len(stats)} patches, {failed} failed, "
          f"{int(stats.valid_px.sum()):,} valid px "
          f"({stats.valid_px.sum() * 1e-4:.0f} km2), "
          f"{int((stats.valid_px == 0).sum())} all-water\n")

    # A ---------------------------------------------------------------
    comp = composition(stats, fine_classes)
    show = comp.copy()
    show["prop_%"] = (show.prop * 100).round(4)
    show["95% CI"] = [f"{a * 100:.4f}-{b * 100:.4f}"
                      for a, b in zip(show.lo95, show.hi95)]
    print("A. Composition of the mapped land surface (patch-level ratio "
          "estimate, n=%d patches)" % len(stats))
    print(show[["px", "prop_%", "95% CI", "patches_present", "patches_usable",
                "n_eff"]].sort_values("prop_%", ascending=False)
          .to_string(float_format=lambda v: f"{v:,.1f}"))
    print(f"\n   'patches_usable' = patches holding >= {args.min_px} px "
          f"({args.min_px / 100:.0f} ha) of the class.")
    print("   n_eff is the binomial sample size that would give this SE: the "
          "pilot's 25M pixels\n   are worth that many independent "
          "observations, which is why the patch is the unit.\n")

    # B ---------------------------------------------------------------
    size = sizing(stats, state, TARGETS, args.min_px, args.max_points,
                  args.target_mode, args.target_n)
    print(f"B. Patches needed ({args.target_mode} target, "
          f"<= {args.max_points} points/patch/class, precision-adjusted)")
    print(size.to_string(float_format=lambda v: f"{v:,.3f}"))
    channels = size["channel"].to_dict()
    # The plan is sized on the classes the map can be trusted to price, i.e.
    # the argmax rows. A `topchange` row's patch count is an unpriced lower
    # bound (see `sizing`), so letting it set the binding number would quietly
    # replace a measured requirement with an optimistic one.
    priced = size[size.channel == "argmax"]["patches_needed"].replace(
        np.inf, np.nan)
    need = int(np.nanmax(priced)) if priced.notna().any() else 0
    worst = priced.idxmax() if priced.notna().any() else None
    fallback = size.index[size.channel == "topchange"].tolist()
    unreachable = size.index[size["patches_needed"] == np.inf].tolist()
    print(f"\n   binding priced class: {worst} -> {need:,} patches "
          f"({need * 25:,} km2 to map)")
    if fallback:
        print(f"   no arg-max area anywhere in the pilot, sized on the "
              f"change-restricted channel (unpriced, lower bound):")
        for cls in fallback:
            row = size.loc[cls]
            print(f"     {cls:24s} {row.usable_patches:3.0f}/{len(stats)} "
                  f"patches usable, >= {row.patches_needed:,.0f} patches")
    if unreachable:
        print(f"   NOT REACHABLE on either channel: {', '.join(unreachable)}")
    over = int(math.ceil(need * args.oversample))
    print(f"   oversample x{args.oversample:g} -> draw {over:,} patches, "
          f"rank, keep the top {need:,}\n")

    # C ---------------------------------------------------------------
    ranked = rank_patches(stats, TARGETS, args.min_px, channels)
    n_elig = int(ranked["eligible"].sum())
    print(f"C. Ranking (pilot sample as a worked example): {n_elig}/"
          f"{len(ranked)} patches eligible")
    cols = ["patch_id", "lat", "lon", "n_target_classes", "novelty_p90",
            "entropy_change", "r_novelty", "r_entropy", "score"]
    print(ranked[ranked.eligible][cols].head(15)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ranked.to_parquet(run_dir / "patch_ranking.parquet", index=False)
    comp.to_csv(run_dir / "composition.csv")
    size.to_csv(run_dir / "sizing.csv")
    (run_dir / "plan.json").write_text(json.dumps({
        "n_patches": int(len(stats)), "min_px": args.min_px,
        "max_points": args.max_points, "target_mode": args.target_mode,
        "target_n": args.target_n,
        "patches_needed": need, "binding_class": worst,
        "unreachable": unreachable, "fallback_channel": fallback,
        "channels": channels,
        "oversample": args.oversample, "draw_next": over,
        "eligible_rate": float(ranked["eligible"].mean()),
    }, indent=2, default=str))

    # D ---------------------------------------------------------------
    if args.draw_points:
        keep = ranked[ranked.eligible].head(need if need else n_elig)
        points = draw_points(run_dir, keep, fine_classes, TARGETS,
                             args.max_points, args.min_sep_m, channels,
                             args.min_px, args.seed)
        if points.empty:
            print("\nD. no label points drawn")
        else:
            out = run_dir / "label_points.parquet"
            points.to_parquet(out, index=False)
            points.to_csv(out.with_suffix(".csv"), index=False)
            print(f"\nD. {len(points)} label points from "
                  f"{points.patch_id.nunique()} patches")
            print(points.pred_class.value_counts().to_string())
            print(f"-> {out}")

    print(f"\n-> {run_dir}")


if __name__ == "__main__":
    main()
