"""Tune the deployed gate-off recipe for the way it is actually served.

The model is trained with two towers and served with the detail gate at zero.
Every training hyper-parameter it inherited, though, was chosen while the model
was being *scored* with the gate on -- `modality_dropout=0.5` in particular means
the AlphaEarth tower stands alone on only half the both-present rows, which is a
training distribution that does not match a deployment where it stands alone
100% of the time.

That mismatch is the one free lever left. `modality_dropout` controls how often
the detail tower is withheld during training; pushed up, the AlphaEarth tower is
trained closer to the condition it will be served in, and the Sentinel-2 tower
degrades to what it is meant to be here -- privileged information that shapes the
shared head without being needed at inference.

Two knobs, both scored on the **gate-off read only**, because that is the
deployed read and nothing else is being shipped:

    modality_dropout   0.3 / 0.5 (inherited) / 0.7 / 0.9
    dropout_tess       0.4 / 0.7 (inherited)   -- does the detail tower still
                       need heavy regularisation when it is never served?

The reference is `A_aef_flat`: AlphaEarth alone, never trained with a second
tower. If no setting beats it, the privileged-information story does not hold on
plot metrics and the extra training cost buys nothing measurable.

Seeds
-----
Default 15, not the usual 5. This project has now twice had a verdict reverse
between 3 and 5 seeds (S10) and between 5 and 15 (the gate-off vs baseline read),
and the effects being chased here are ~0.005. Anything smaller than the seed
spread is not a result no matter how clean the ordering looks.
"""
from __future__ import annotations

import argparse
import itertools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ablate_s2_architecture import _read, rungs
from model_zoo import HierarchicalSoftmaxNN
from project_paths import project_data_dir
from twotower_lab import AEF_MASK, BASE, S2_MASK, load_context, score_probs


def gate_off_cv(view, cols, kwargs, seeds):
    """Out-of-fold metrics under the gate-OFF read -- the deployed read."""
    classes = view.merged_classes
    rows = []
    for seed in seeds:
        probs = np.zeros((len(view.target), len(classes)))
        fine = np.empty(len(view.target), dtype=object)
        for tr, te in view.folds:
            model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
            p_fine, p_merged = _read(model, view.frame.iloc[te], "off")
            block = np.zeros((len(te), len(classes)))
            block[:, [classes.index(c) for c in model.merged_classes_]] = p_merged
            probs[te] = block
            fine[te] = np.array(model.fine_classes_, dtype=object)[p_fine.argmax(1)]
        row = score_probs(view, probs, fine)
        row["seed"] = seed
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-seeds", type=int, default=15)
    ap.add_argument("--modality-dropout", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 0.9])
    ap.add_argument("--dropout-tess", type=float, nargs="+", default=[0.7, 0.4])
    ap.add_argument("--out", type=Path, default=project_data_dir("analysis_results"))
    args = ap.parse_args()

    ctx = load_context()
    view = ctx.view("full")
    seeds = list(range(args.n_seeds))
    aef, s2 = ctx.aef_cols, ctx.s2_stat_cols
    print(f"{len(view.target):,} plots | aef={len(aef)} s2={len(s2)} | "
          f"{args.n_seeds} seeds | scored on the GATE-OFF read", flush=True)

    rows = []

    # Reference: AlphaEarth alone, no second tower at any point.
    t0 = time.time()
    ref = rungs(ctx)[0]
    got = gate_off_cv(view, ref["cols"], ref["kwargs"], seeds)
    for r in got:
        r.update(md=np.nan, dt=np.nan, config="A_aef_flat (no S2 in training)")
    rows.extend(got)
    d = pd.DataFrame(got)
    print(f"  {'A_aef_flat':32s} change_f1={d.change_f1.mean():.4f}"
          f"±{d.change_f1.std():.4f}  asVeg={d.art_stable_as_veg.mean():.4f}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    for md, dt in itertools.product(args.modality_dropout, args.dropout_tess):
        t0 = time.time()
        kwargs = dict(arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"],
                      tower_dim=256, aef_columns=aef, tess_columns=s2,
                      mask_column=S2_MASK, aef_mask_column=AEF_MASK,
                      fusion="gated_mean", modality_dropout=md, dropout_tess=dt)
        got = gate_off_cv(view, aef + s2, kwargs, seeds)
        tag = f"md={md} dropout_tess={dt}"
        for r in got:
            r.update(md=md, dt=dt, config=tag)
        rows.extend(got)
        d = pd.DataFrame(got)
        print(f"  {tag:32s} change_f1={d.change_f1.mean():.4f}"
              f"±{d.change_f1.std():.4f}  asVeg={d.art_stable_as_veg.mean():.4f}  "
              f"macro={d.macro_f1.mean():.4f}  coarse3={d.fine_change_f1.mean():.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(args.out / "s2off_training_sweep.csv", index=False)

    print(f"\n-> {args.out / 's2off_training_sweep.csv'}")


if __name__ == "__main__":
    main()
