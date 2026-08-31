"""Conformal diagnostics for the cached siamese OOF probabilities.

The ledger scores labels, so it cannot see the things a conformal read is
actually for. This computes them from the same cache, for free:

* **coverage** -- does the calibrated set contain the truth as often as claimed,
  marginally and per class? The folds here are spatial blocks, so exchangeability
  across them is violated by construction and the guarantee is nominal only.
  Realised coverage is therefore a measurement, not a formality, and it is the
  first thing to read in the output.
* **set size / singleton fraction** -- the price of that coverage, and the
  selective-prediction read: a singleton set is a plot the model is confident
  about at the stated level, and accuracy on those rows against their share of
  the data is the abstention curve a map uncertainty band would be built on.
* **the two dead coarse3 classes** -- `Artificial -> Cropland` and
  `Cropland -> Nature` are 0.0000 F1 for every model in sections N, O and P. A
  set-valued read asks a weaker question than an arg-max: are they in the set at
  all? That distinguishes "unreachable at the arg-max" from "the model has no
  signal", which P6 could only argue about from an oracle.

Usage::

    python conformal_report.py                       # both levels, 5 seeds
    python conformal_report.py --sources siam_cos --n-seeds 15
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import twotower_lab as lab
from project_paths import project_data_dir

OUT = project_data_dir("analysis_results") / "conformal_siam.csv"

MERGED_SOURCES = ("siam_s2off_cos", "siam_cos", "s2off_centre_m3s3_bf")
FINE_SOURCES = ("base_siam_s2off_cos_fine", "base_siam_cos_fine",
                "base_deployed_fine")


def diagnostics(view, probs, classes, truth, alpha, kind, mondrian, seed):
    """Coverage, set size and the singleton read for one seed at one alpha."""
    scores, q, sets = lab.nested_conformal(view, probs, classes, truth, alpha,
                                           kind=kind, mondrian=mondrian,
                                           seed=seed)
    index = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([index.get(t, -1) for t in truth])
    ok = true_idx >= 0
    covered = np.zeros(len(truth), dtype=bool)
    covered[ok] = sets[np.arange(len(truth))[ok], true_idx[ok]]
    size = sets.sum(1)
    single = size == 1
    argmax = np.array(classes, dtype=object)[probs.argmax(1)]
    row = {
        "alpha": alpha, "score": kind,
        "mode": "mondrian" if mondrian else "marginal", "seed": seed,
        "coverage": float(covered[ok].mean()),
        "set_size": float(size.mean()),
        "singleton_frac": float(single.mean()),
        "empty_frac": float((size == 0).mean()),
        # The abstention curve in two numbers: how much of the data is retained
        # if only singleton sets are trusted, and how accurate the arg-max is
        # there against how accurate it is on the rest.
        "acc_singleton": float((argmax[single] == truth[single]).mean())
        if single.any() else np.nan,
        "acc_rest": float((argmax[~single] == truth[~single]).mean())
        if (~single).any() else np.nan,
    }
    # The full size distribution, not only its mean and its two tails. A mean of
    # 1.46 over four classes is consistent with almost every plot at 1-2 and with
    # a third of them empty against a third at 3 -- and those are different
    # products. Sizes run 0..K, so the columns are fixed by the level.
    for s in range(len(classes) + 1):
        row[f"size{s}"] = float((size == s).mean())
    row["size_p90"] = float(np.quantile(size, 0.90))
    row["size_max"] = int(size.max())
    for k, cls in enumerate(classes):
        rows = truth == cls
        slug = cls.lower().replace(" -> ", "_to_").replace(" ", "")
        row[f"cov_{slug}"] = float(covered[rows].mean()) if rows.any() else np.nan
        row[f"inset_{slug}"] = float(sets[:, k].mean())
        row[f"n_{slug}"] = int(rows.sum())
    return row


def run(ctx, source, level, seeds, alphas):
    view = ctx.view("full")
    rows = []
    for seed in seeds:
        cached = lab.load_oof(source, "full", seed)
        if cached is None:
            continue
        if level == "merged2":
            probs = cached[0]
            classes, truth = view.merged_classes, view.truth_merged
        else:
            fine = lab.load_oof_fine(source, "full", seed)
            if fine is None:
                continue
            probs, classes = fine
            truth = view.truth_fine
        for kind in ("lac", "aps"):
            for mondrian in (True, False):
                for alpha in alphas:
                    row = diagnostics(view, probs, classes, truth, alpha, kind,
                                      mondrian, seed)
                    row.update(source=source, level=level)
                    rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--alphas", default="")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    alphas = ([float(a) for a in args.alphas.split(",") if a]
              or list(lab.CONFORMAL_ALPHAS))
    wanted = {s for s in args.sources.split(",") if s}
    ctx = lab.load_context()
    seeds = range(args.n_seeds)

    rows = []
    for source in MERGED_SOURCES:
        if not wanted or source in wanted:
            rows += run(ctx, source, "merged2", seeds, alphas)
    for source in FINE_SOURCES:
        if not wanted or source in wanted:
            rows += run(ctx, source, "coarse3", seeds, alphas)
    frame = pd.DataFrame(rows)
    if frame.empty:
        print("nothing cached for the requested sources")
        return

    keys = ["source", "level", "score", "mode", "alpha"]
    agg = frame.drop(columns=["seed"]).groupby(keys).mean(numeric_only=True)
    agg["n_seeds"] = frame.groupby(keys)["seed"].nunique()
    agg = agg.reset_index()
    agg.to_csv(args.out, index=False)

    for level in ("merged2", "coarse3"):
        part = agg[agg.level == level]
        if part.empty:
            continue
        print(f"\n=== {level}: coverage and its price "
              f"(nominal coverage = 1 - alpha) ===")
        show = ["source", "score", "mode", "alpha", "coverage", "set_size",
                "singleton_frac", "empty_frac", "acc_singleton", "acc_rest"]
        print(part[show].to_string(index=False,
                                   float_format=lambda v: f"{v:.4f}"))

        size_cols = [c for c in part.columns
                     if c.startswith("size") and c[4:].isdigit()
                     and part[c].notna().any()]
        print(f"\n--- {level}: SET-SIZE DISTRIBUTION "
              f"(share of plots at each size) ---")
        print(part[["source", "score", "mode", "alpha", "set_size"]
                   + size_cols + ["size_p90"]].to_string(
                       index=False, float_format=lambda v: f"{v:.3f}"))

        cov_cols = [c for c in part.columns if c.startswith("cov_")]
        print(f"\n--- {level}: PER-CLASS coverage, Mondrian vs marginal "
              f"(alpha=0.10) ---")
        sub = part[(part.alpha == 0.10)]
        cols = ["source", "score", "mode"] + cov_cols
        print(sub[cols].to_string(index=False,
                                  float_format=lambda v: f"{v:.3f}"))
    print(f"\n{len(agg)} rows -> {args.out}")


if __name__ == "__main__":
    main()
