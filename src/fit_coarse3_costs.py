"""Fit the ONE coarse3 decision-cost vector that a raster run can ship.

Section O's gate (O3 / N6) is nested for *scoring*: each outer fold gets costs
chosen on the other folds, which is what makes 0.4412 an honest held-out
estimate rather than a number tuned on itself. A raster has no folds, so
deploying it needs a single vector, and this fits that vector the only way that
is defensible -- on the **out-of-fold** coarse3 posteriors cached by
`twotower_lab.py`, never on in-sample training posteriors.

That distinction is the whole correctness argument here. A model's in-sample
coarse3 distribution is far sharper than the one it produces on a pixel it has
never seen, so costs fitted on it would be tuned against a confidence the raster
never exhibits, and would systematically under-correct. The OOF cache is the
same model reading unseen rows, which is exactly the raster's situation.

The nested runs stay the evidence that the *procedure* generalises; this fits
the *instance* that ships, using every labelled plot once.

    python src/fit_coarse3_costs.py --idea base_siam_cos_fine --seeds 5

Writes `data/analysis_results/coarse3_costs__<idea>.json`, which
`infer_s2.py` loads by name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project_paths import project_data_dir
from twotower_lab import (COST_GRID, FOCUS_TRANSITIONS, focus_metrics, load_oof,
                          load_oof_fine, load_context)


def fit_costs(truth_fine: np.ndarray, fine_probs: np.ndarray,
              fine_classes: list, passes: int = 2):
    """Coordinate-ascent the four focus multipliers on ``focus_macro_f1``.

    Only the commissioned transitions get a free multiplier, matching the lab
    gate exactly -- nine multipliers is more freedom than F3 found these folds
    can support, and a deployment vector that differs from the scored one is not
    the thing that was scored.
    """
    arr = np.array(fine_classes, dtype=object)
    targets = [i for i, c in enumerate(fine_classes) if c in FOCUS_TRANSITIONS]
    costs = np.ones(len(fine_classes))

    def score(c):
        return focus_metrics(truth_fine, arr[(fine_probs * c).argmax(1)])["focus_macro_f1"]

    best = score(costs)
    for _ in range(passes):
        for j in targets:
            for m in COST_GRID:
                trial = costs.copy()
                trial[j] = m
                got = score(trial)
                if got > best:
                    best, costs = got, trial
    return costs, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idea", default="base_siam_cos_fine")
    parser.add_argument("--read", default="full")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    args = parser.parse_args()

    ctx = load_context()
    view = ctx.view(args.read)

    # Average the seeds' posteriors before fitting, because the raster is served
    # as a seed ensemble (G1) -- costs fitted against a single seed's sharpness
    # would be fitted against a distribution the map does not have.
    stack, classes = [], None
    for seed in range(args.seeds):
        cached = load_oof_fine(args.idea, args.read, seed)
        if cached is None:
            raise SystemExit(
                f"no cached coarse3 probabilities for {args.idea} seed {seed}; "
                f"run: twotower_lab.py --ideas {args.idea} --n-seeds {args.seeds}")
        probs, cls = cached
        if classes is not None and cls != classes:
            raise SystemExit("seed members disagree on coarse3 class order")
        classes, _ = cls, stack.append(probs)
    fine_probs = np.mean(stack, axis=0)

    ungated = focus_metrics(
        view.truth_fine,
        np.array(classes, dtype=object)[fine_probs.argmax(1)])
    costs, best = fit_costs(view.truth_fine, fine_probs, classes)
    gated = focus_metrics(
        view.truth_fine,
        np.array(classes, dtype=object)[(fine_probs * costs).argmax(1)])

    print(f"{args.idea}: focus_macro_f1 {ungated['focus_macro_f1']:.4f} -> "
          f"{gated['focus_macro_f1']:.4f} (in-sample on OOF posteriors)")
    for cls, cost in zip(classes, costs):
        if cost != 1.0:
            print(f"  {cls:28s} x{cost}")
    for cls in FOCUS_TRANSITIONS:
        slug = cls.lower().replace(" -> ", "_to_").replace(" ", "")
        print(f"  {cls:28s} F1 {ungated.get(f'fine_f1_{slug}', float('nan')):.4f} "
              f"-> {gated.get(f'fine_f1_{slug}', float('nan')):.4f}")

    out = args.out_dir / f"coarse3_costs__{args.idea}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "idea": args.idea, "read": args.read, "n_seeds": args.seeds,
        "fine_classes": list(classes), "costs": [float(c) for c in costs],
        # Recorded so a later reader can tell at a glance that this is the
        # in-sample number and NOT the 0.4412 nested estimate the ledger quotes.
        "focus_macro_f1_insample_ungated": float(ungated["focus_macro_f1"]),
        "focus_macro_f1_insample_gated": float(gated["focus_macro_f1"]),
    }, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
