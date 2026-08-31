"""`learning_curves.py`, run for `siam_s2off_cos` instead of the deployed recipe.

Same curve machinery, same sizes, same gate-off read -- only the recipe differs,
so the two CSVs are directly comparable and the train/OOF gap and the
per-doubling slope can be quoted for the siamese two-tower (N8b) the way S19
quotes them for `s2off_centre_m3s3_bf`.

Evidence behind sections 6.2 and 7.3 of
``docs/land_cover_change_model_report_v2.md``. Writes
``data/analysis_results/learning_curves_siam_s2off_cos.csv``.

Slopes in that report are OLS of OOF F1 on log2(n_train) over the top six sizes
(frac >= 0.3), per seed, then averaged -- the window that reproduces the
published +0.0264 for the deployed model. Read at any other window both models
move together, so the comparison is robust to the choice and the level is not.

Run
---
    /home/geethen.singh/.pixi/envs/geo/bin/python src/learning_curves_siam.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from learning_curves import curve_rows  # noqa: E402
from project_paths import project_data_dir  # noqa: E402
from twotower_lab import (AEF_MASK, S2_MASK, load_context,  # noqa: E402
                          s2_subset_columns)

#: N8b (docs/research/SIAMESE_RESEARCH.md). The deployed two-tower recipe with
#: its AlphaEarth tower replaced by a shared endpoint encoder, plus the
#: gate-supervised cosine objective. Kept as one dict for the same reason
#: `learning_curves.DEPLOYED` is: so the two cannot drift.
SIAM = dict(arch="two_tower", loss="focal", epochs=30, tower_dim=256,
            fusion="gated_mean", modality_dropout=0.5, dropout_tess=0.7,
            mask_column=S2_MASK, aef_mask_column=AEF_MASK,
            aef_siam=True, siam_dim=128, siam_combine="conc",
            siam_cos_weight=0.3, siam_cos_margin=0.3)
SUBSET = "centre_m3s3_bf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", type=Path,
                    default=project_data_dir("analysis_results")
                    / "learning_curves_siam_s2off_cos.csv")
    args = ap.parse_args()

    ctx = load_context()
    view = ctx.view("full")
    s2 = s2_subset_columns(ctx.s2_stat_cols, SUBSET)
    kwargs = dict(SIAM, aef_columns=ctx.aef_cols, tess_columns=s2)
    print(f"{len(view.target):,} plots | aef={len(ctx.aef_cols)} s2={len(s2)} "
          f"({SUBSET}) | {args.seeds} seeds | GATE-OFF read", flush=True)
    rows = curve_rows(view, ctx.aef_cols + s2, kwargs, list(range(args.seeds)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
