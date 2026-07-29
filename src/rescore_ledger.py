"""Rescore every cached two-tower OOF run under the extended metric set.

``twotower_lab`` caches out-of-fold merged2 probabilities for every idea x read x
seed it has ever run, so the whole back-catalogue can be re-judged on metrics
that did not exist when it was fitted -- per-class F1, the stable-Artificial
confusion, the Tessera-availability split -- without refitting anything. That is
the point of the cache, and it means the answer to "which of the 40 tested ideas
actually helped stable Artificial?" costs seconds rather than a day of GPU.

Writes ``data/analysis_results/twotower_lab_metrics.csv``, one row per idea x
read, with:

* every metric averaged over the seeds (``*_mean`` / ``*_std``) -- comparable to
  the ledger's numbers and to each other;
* the same metrics for the **seed-ensemble** (probabilities averaged across
  seeds, then read once, columns prefixed ``ens_``) -- this is what a deployed
  model would actually be, and it is consistently better than the per-seed mean.

Ideas registered in the ``operating-point`` group re-derive their nested change
gate from the cached probabilities (``nested_gate`` is deterministic given
probabilities and folds), so their rows stay honest rather than silently
reverting to arg-max.

Usage::

    python rescore_ledger.py                 # rescore everything cached
    python rescore_ledger.py --ideas mc_dropout_scalars,baseline_aef
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import twotower_lab as lab
from project_paths import project_data_dir

OUT = project_data_dir("analysis_results") / "twotower_lab_metrics.csv"


def cached_runs() -> dict[tuple[str, str], list[int]]:
    """``{(idea, read): [seeds]}`` for everything in the OOF cache."""
    runs: dict[tuple[str, str], list[int]] = {}
    for path in sorted(lab.OOF_DIR.glob("*.npz")):
        match = re.fullmatch(r"(.+)__(full|subset)__seed(\d+)", path.stem)
        if not match:
            continue
        runs.setdefault((match.group(1), match.group(2)), []).append(int(match.group(3)))
    return {k: sorted(v) for k, v in runs.items()}


def operating_point(idea: str):
    """The labelling an ``operating-point`` idea defines, or None for arg-max.

    These ideas are post-hoc reads of cached probabilities by construction -- a
    nested change threshold, a nested per-class cost -- so re-running the idea's
    own function is both cheap and the only way to reproduce *its* decision rule.
    Re-deriving one fixed rule for the whole group silently scored the
    cost-sensitive ideas at the change-threshold operating point, which is a
    different model.
    """
    entry = lab.IDEAS.get(idea)
    if entry is None or entry.group != "operating-point":
        return None
    return entry.fn


def score_one(view: lab.View, probs: np.ndarray, fine: np.ndarray | None,
              labels: np.ndarray | None) -> dict:
    """Extended metrics for one probability matrix at a given operating point."""
    return lab.score_probs(view, probs, fine, labels)


def rescore(ctx: lab.Context, idea: str, read: str, seeds: list) -> dict | None:
    view = ctx.view(read)
    fn = operating_point(idea)
    per_seed, probs_stack, label_stack = [], [], []
    for seed in seeds:
        cached = lab.load_oof(idea, read, seed)
        if cached is None:
            continue
        probs, fine = cached
        if probs.shape[0] != len(view.truth_merged):
            # A stale cache from a different frame build: skip rather than
            # silently score against the wrong truth.
            return None
        labels = None
        if fn is not None:
            try:
                result = fn(ctx, view, seed)
                labels = result[2] if len(result) > 2 else None
            except Exception:
                # The source idea's cache is gone; fall back to arg-max rather
                # than dropping the row, and let the number speak for itself.
                labels = None
        probs_stack.append(probs)
        label_stack.append(labels)
        per_seed.append(score_one(view, probs, fine, labels))
    if not per_seed:
        return None

    row = {"idea": idea, "read": read, "n_seeds": len(per_seed),
           "n_plots": len(view.truth_merged),
           "group": lab.IDEAS[idea].group if idea in lab.IDEAS else "archived",
           "desc": lab.IDEAS[idea].desc if idea in lab.IDEAS else ""}
    for key in per_seed[0]:
        vals = [s[key] for s in per_seed]
        row[f"{key}_mean"] = round(float(np.nanmean(vals)), 4)
        row[f"{key}_std"] = round(float(np.nanstd(vals)), 4)

    # The seed ensemble: average the probabilities, then read once. Averaging
    # *labels* across seeds is a vote and loses the calibration; averaging
    # probabilities is the model a deployment would actually ship. Skipped for
    # operating-point ideas -- their decision rule is fitted per seed against
    # per-seed probabilities, so pooling the probabilities and keeping one seed's
    # rule would report a model that does not exist.
    if len(probs_stack) > 1 and fn is None:
        ens = score_one(view, np.mean(probs_stack, axis=0), None, None)
        for key, val in ens.items():
            row[f"ens_{key}"] = round(float(val), 4) if np.isfinite(val) else np.nan
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--ideas", default="", help="comma-separated subset")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    ctx = lab.load_context(args.input) if args.input else lab.load_context()
    wanted = {n for n in args.ideas.split(",") if n}
    runs = cached_runs()

    rows, skipped = [], []
    for (idea, read), seeds in runs.items():
        if wanted and idea not in wanted:
            continue
        row = rescore(ctx, idea, read, seeds)
        (rows if row else skipped).append(row or f"{idea}/{read}")

    frame = pd.DataFrame(rows).sort_values(
        ["read", "change_f1_mean"], ascending=[True, False])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    show = ["idea", "n_seeds", "change_f1_mean", "macro_f1_mean",
            "art_stable_recall_mean", "art_stable_as_veg_mean",
            "tess_recall_gap_mean", "ens_change_f1", "ens_macro_f1"]
    for read in ("full", "subset"):
        part = frame[frame["read"] == read]
        if part.empty:
            continue
        print(f"\n=== {read} ({int(part['n_plots'].iloc[0]):,} plots) "
              f"ranked by change-F1 ===")
        print(part[[c for c in show if c in part]].head(20).to_string(index=False))
    if skipped:
        print(f"\nskipped (stale cache): {', '.join(map(str, skipped))}")
    print(f"\n{len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
