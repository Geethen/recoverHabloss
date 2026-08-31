"""What the co-teaching selector actually rejected (section T).

A co-teaching arm's ledger row says whether the metric moved. It cannot say
*why*, and on this problem the two candidate mechanisms make the same sign of
movement on the headline number:

* the selector drops **mislabelled** rows, which is the method working; or
* the selector drops **rare** rows, which is the method being a class-imbalance
  filter wearing a noisy-label hat. A small-loss (or high-posterior) criterion
  is biased against every minority class by construction -- a 46-plot transition
  the model has barely learned is exactly a high-loss row -- and this project's
  target is 4,200 stable plots against 46 in its rarest transition.

The second is a real failure mode with a legible signature, so it is measured
rather than argued about: this script re-runs the fold loop of ``s2off_cv``,
keeps each fitted model's ``coteach_keep_counts_`` instead of throwing it away,
and reports the keep rate **per coarse3 class**. A selector that is filtering
noise keeps the rare classes at roughly the rate it keeps the common ones; a
selector that is filtering rarity does not.

It also writes the by-product that is arguably worth more than the arms: a
per-plot table of how often each plot survived selection, pooled over folds and
seeds. Under the noise reading, the plots at the bottom of that table are the
model's nomination of which interpretations to check -- a **relabelling queue**,
ordered, on a project whose learning curves say +0.026 change-F1 per doubling of
labels and whose ceiling is interpreter disagreement.

Read the queue as a nomination, never as a verdict: a plot can sit at the bottom
because its label is wrong, or because it is genuinely hard, or because it is
rare, and this script's per-class table is the only thing that separates the
third from the first two.

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P coteach_diagnostics.py --idea siam_sct --n-seeds 3

Writes ``coteach_keep_by_class__<idea>.csv`` and ``coteach_plot_queue__<idea>.csv``
to ``data/analysis_results/``.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model_zoo import HierarchicalSoftmaxNN
from project_paths import project_data_dir
from twotower_lab import (
    IDEAS,
    S2_MASK,
    SIAM_AUX,
    load_context,
    siam_s2off_kwargs,
)

OUTPUT = project_data_dir("analysis_results")

#: The section-T arms this script knows how to rebuild, as the co-teaching
#: overrides on top of `siam_s2off_cos`. Kept here rather than read back out of
#: `IDEAS` because an idea is a closure over the CV loop -- the overrides are
#: not recoverable from it, and duplicating three lines is better than making
#: every idea in the lab carry a machine-readable parameter dict for one script.
ARMS: dict[str, dict] = {
    "siam_sct": dict(coteach="stochastic", coteach_level="fine"),
    "siam_sct_gate": dict(coteach="stochastic", coteach_level="gate"),
    "siam_sct_merged": dict(coteach="stochastic", coteach_level="merged"),
    "siam_coteach10": dict(coteach="classic", coteach_forget=0.10),
    "siam_coteach20": dict(coteach="classic", coteach_forget=0.20),
    "siam_cotrand10": dict(coteach="random", coteach_forget=0.10),
    "siam_coteach10_strat": dict(coteach="classic", coteach_forget=0.10,
                                 coteach_stratify=True),
    "siam_cotrand10_strat": dict(coteach="random", coteach_forget=0.10,
                                 coteach_stratify=True),
    "siam_coteach30_strat": dict(coteach="classic", coteach_forget=0.30,
                                 coteach_stratify=True),
}


def run(ctx, idea: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every fold under the arm's co-teaching setting, keeping the counts."""
    view = ctx.view("full")
    cols, kwargs = siam_s2off_kwargs(
        ctx, siam_cos_weight=SIAM_AUX, siam_cos_margin=0.3, **ARMS[idea])

    n = len(view.target)
    kept = np.zeros(n, dtype="float64")       # summed keep FRACTION per plot
    offered = np.zeros(n, dtype="float64")    # times the plot was in a training fold
    rates, guards = [], []
    for seed in seeds:
        for tr, _te in view.folds:
            model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
            counts = model.coteach_keep_counts_
            if counts is None:
                raise RuntimeError(f"{idea} did not run a co-teaching selector")
            # counts is per row of the fold's TRAINING frame, in the order the
            # fold was built, so `tr` maps it straight back to plot rows.
            kept[tr] += counts / max(model.coteach_steps_, 1)
            offered[tr] += 1.0
            rates.append(model.coteach_keep_rate_)
            guards.append(model.coteach_guard_rate_)
        print(f"  seed {seed}: keep_rate={np.mean(rates):.3f} "
              f"guard_rate={np.mean(guards):.3f}", flush=True)

    keep_freq = np.where(offered > 0, kept / np.maximum(offered, 1), np.nan)
    plots = pd.DataFrame({
        "PLOTID": view.frame["PLOTID"].to_numpy() if "PLOTID" in view.frame
        else np.arange(n),
        "truth_fine": view.truth_fine,
        "truth_merged": view.truth_merged,
        "keep_freq": keep_freq,
        "folds_offered": offered,
    })
    # Rank WITHIN the plot's own class. The raw keep frequency is dominated by
    # which class a plot belongs to (see the by-class table), so a queue sorted
    # on it is a list of the rare classes and nothing more. The within-class
    # standardised score asks the question the queue is for: is this plot
    # unusual *for a plot of its stated transition*?
    grp = plots.groupby("truth_fine")["keep_freq"]
    plots["keep_freq_z"] = ((plots["keep_freq"] - grp.transform("mean"))
                            / grp.transform("std").replace(0.0, np.nan))

    by_class = (plots.groupby("truth_fine")
                .agg(n=("keep_freq", "size"), keep_freq=("keep_freq", "mean"))
                .reset_index()
                .sort_values("keep_freq"))
    by_class["keep_freq_vs_overall"] = by_class["keep_freq"] - plots["keep_freq"].mean()
    by_class.attrs["keep_rate"] = float(np.mean(rates))
    by_class.attrs["guard_rate"] = float(np.mean(guards))
    return by_class, plots


def reverification_check(plots: pd.DataFrame) -> pd.DataFrame | None:
    """Does the selector reject the plots two interpreters actually disagreed on?

    The only external evidence about label noise this project holds is
    ``analyse_label_noise.py``'s 54 RECOVER reverifications -- one plot, two
    independent interpretations, agreeing or not. If the selector is finding
    mislabels, the plots whose two reads disagree should be kept *less* often
    than the plots whose reads agree. If it is finding difficulty, there is no
    reason for the two groups to differ.

    This is the section's one falsifiable prediction and it is a weak test by
    construction, for reasons that must travel with the number:

    * n = 54, and the reverified subset is **change-enriched** -- it targets
      plots the first read flagged -- so the two groups differ in class
      composition as well as in agreement, and the class composition is what the
      by-class table shows the selector responds to most.
    * the frame keeps one row per plot after deduplication, so "disagreement"
      here labels the plot, not the row the model trained on.

    Reported as a difference in mean within-class keep score with a
    Mann-Whitney U p-value, not as a decision rule.
    """
    from scipy.stats import mannwhitneyu

    path = project_data_dir("analysis_results", "label_noise_pairs.csv")
    if not path.exists():
        print(f"\n(no {path.name}; run analyse_label_noise.py for the "
              f"reverification check)")
        return None
    pairs = pd.read_csv(path)
    pairs = pairs[pairs["source_combo"] == "recover | recover"]
    merged = plots.merge(pairs[["PLOTID", "agree_transition", "agree_change"]],
                         on="PLOTID", how="inner").dropna(subset=["keep_freq_z"])
    if merged.empty:
        print("\n(no reverified plots survive the join to the modelling frame)")
        return None

    rows = []
    for column in ("agree_transition", "agree_change"):
        agree = merged.loc[merged[column], "keep_freq_z"]
        differ = merged.loc[~merged[column], "keep_freq_z"]
        if len(agree) < 3 or len(differ) < 3:
            continue
        stat, p = mannwhitneyu(differ, agree, alternative="less")
        rows.append({
            "criterion": column,
            "n_agree": len(agree), "n_disagree": len(differ),
            "keep_z_agree": agree.mean(), "keep_z_disagree": differ.mean(),
            "delta": differ.mean() - agree.mean(),
            "mannwhitney_p": p,
        })
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default="siam_sct", choices=sorted(ARMS))
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--s2", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if args.idea not in IDEAS:
        raise SystemExit(f"{args.idea} is not registered in twotower_lab")
    ctx = load_context(s2_path=args.s2)
    by_class, plots = run(ctx, args.idea, list(range(args.n_seeds)))

    print(f"\n{args.idea}: mean keep rate {by_class.attrs['keep_rate']:.3f}, "
          f"guard fired on {by_class.attrs['guard_rate']:.1%} of selections\n")
    print(by_class.to_string(index=False))

    check = reverification_check(plots)
    if check is not None:
        print("\nreverified plots (analyse_label_noise.py): does the selector "
              "reject the disagreed ones?")
        print(check.round(4).to_string(index=False))

    args.output.mkdir(parents=True, exist_ok=True)
    by_class.to_csv(args.output / f"coteach_keep_by_class__{args.idea}.csv",
                    index=False)
    if check is not None:
        check.to_csv(args.output / f"coteach_reverification__{args.idea}.csv",
                     index=False)
    plots.sort_values("keep_freq_z").to_csv(
        args.output / f"coteach_plot_queue__{args.idea}.csv", index=False)
    print(f"\n-> {args.output}/coteach_keep_by_class__{args.idea}.csv")
    print(f"-> {args.output}/coteach_plot_queue__{args.idea}.csv")


if __name__ == "__main__":
    main()
