"""Mixture-of-experts trunks and interleaved noise injection for the hier net.

Two independent additions to ``HierarchicalSoftmaxNN``, each scored against the
current best config (``arch=wide, loss=focal``) under the same spatially blocked
CV and merged2 deploy metric as experiment_hier_variants.py:

1. **Mixture-of-experts trunk** (``arch='moe'``). n_experts parallel MLP experts
   over the 192-D embedding, combined by a learned softmax gate -- dense, or a
   sparse top-k gate (``moe_k``) -- with an optional importance-balancing penalty
   (``moe_aux``) to stop the gate collapsing onto one expert. The question is
   whether letting experts specialise on different regions of the embedding
   space (plausibly the change vs stable manifolds) beats one monolithic wide
   trunk at the same objective.

2. **Interleaved noise injection** (Wiemann et al. 2026, arXiv:2607.14466).
   Gaussian noise added to the trunk input and/or representation during training,
   scheduled across epochs. The paper's central claim is that *interleaving*
   clean and noisy epochs beats both a monotone noisy->clean curriculum
   ('anneal') and constant noise, at ~zero cost, by letting the optimiser explore
   in noisy epochs without forgetting clean-epoch features -- and that co-scaling
   the noise with the gradient norm ('gradscale') is what balances the induced
   random-walk against the gradient drift. This sweep tests that ordering
   (off < constant/anneal < interleaved) on our label-noise-limited target, where
   the theory (noise injection == Jacobian/Hessian regularisation) predicts a
   robustness gain on the noisy Cropland/Nature boundary specifically.

The best MoE and best noise setting are then combined. Tables are written to
analysis_results/ (moe, noise, combined). As with the sibling scripts, the
honest headline is recorded whatever it is: these attack the optimiser/capacity,
not the interpreter noise that the merged-legend restructuring already removed.

Headline (30 epochs, 5-fold blocked CV, merged2 change-F1):

* **MoE loses.** Every expert count / gating / load-balancing setting trails the
  plain wide trunk (best MoE 0.635 vs 0.657 merged change-F1, and -6 pts
  balanced accuracy). Splitting a fixed budget across experts starves each on a
  ~6.4k-plot set; the gate trades change recall (0.60) for precision (0.67) and
  the collapsed change signal costs balanced accuracy. Combining it with noise
  does not rescue it.
* **Interleaved noise does not reliably help.** At the default seed the paper's
  ordering holds (interleaved 0.663 > constant/anneal 0.659 > baseline 0.657),
  but a 5-seed re-run erases it: mean delta -0.001 merged change-F1, interleaved
  beating baseline in only 2 of 5 seeds. The single-seed lift was seed luck. The
  paper's mechanism (noise == Jacobian/Hessian regularisation, a robustness gain
  on a noisy boundary) has nothing left to buy here because the merged-legend
  restructuring already collapsed the Cropland/Nature interpreter noise into a
  stable Veg->Veg -- the same reason the forward-correction / SSL variants in
  experiment_hier_novel.py came up empty. gradscale, rep-site, and period were
  swept and are all within noise.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_hier_variants import evaluate_variant, score_variant
from experiment_merged_legend import LEGENDS, build_frame, target_for_legend
from model_zoo import DEFAULT_INPUT, is_change_label, make_splitter, scores, to_merged_label
from project_paths import project_data_dir

BASE = dict(arch="wide", loss="focal")


def moe_configs():
    """MoE trunk sweep: dense vs top-k gating, expert count, load balancing."""
    cfgs = [("baseline_wide", dict(**BASE))]
    for e in (2, 4, 6, 8):
        cfgs.append((f"moe_soft_E{e}", dict(loss="focal", arch="moe", n_experts=e)))
    for e in (4, 6, 8):
        cfgs.append((f"moe_top2_E{e}", dict(loss="focal", arch="moe", n_experts=e, moe_k=2)))
    for aux in (0.01, 0.05):
        cfgs.append((f"moe_top2_E6_aux{aux}",
                     dict(loss="focal", arch="moe", n_experts=6, moe_k=2, moe_aux=aux)))
    return cfgs


def noise_configs():
    """Interleaved-noise sweep isolating schedule, magnitude, site, gradscale."""
    cfgs = [("baseline_wide", dict(**BASE))]
    # Schedule comparison at a fixed std (the paper's core ordering claim).
    for sched in ("constant", "anneal", "warmup", "interleaved"):
        cfgs.append((f"noise_{sched}_0.1", dict(**BASE, noise_std=0.1, noise_schedule=sched)))
    # Magnitude sweep on the winning schedule shape (interleaved).
    for std in (0.05, 0.2):
        cfgs.append((f"noise_interleaved_{std}",
                     dict(**BASE, noise_std=std, noise_schedule="interleaved")))
    # Injection site and gradient-norm scaling, at the reference std.
    cfgs.append(("noise_interleaved_0.1_rep",
                 dict(**BASE, noise_std=0.1, noise_schedule="interleaved", noise_sites=("rep",))))
    cfgs.append(("noise_interleaved_0.1_both",
                 dict(**BASE, noise_std=0.1, noise_schedule="interleaved",
                      noise_sites=("input", "rep"))))
    cfgs.append(("noise_interleaved_0.1_gradscale",
                 dict(**BASE, noise_std=0.1, noise_schedule="interleaved",
                      noise_gradscale=True)))
    cfgs.append(("noise_interleaved_0.1_period3",
                 dict(**BASE, noise_std=0.1, noise_schedule="interleaved", noise_period=3)))
    return cfgs


def run_table(configs, frame, columns, target, groups, truth_fine, truth_merged,
              epochs, n_splits):
    rows = []
    for name, kw in configs:
        kw = dict(kw, epochs=epochs)
        of, om, secs = evaluate_variant(name, kw, frame, columns, target, groups, n_splits)
        row = score_variant(name, of, om, truth_fine, truth_merged, secs)
        rows.append(row)
        print(f"  {name:28s} merged_chgF1={row['merged_change_f1']:.3f} "
              f"balAcc={row['merged_bal_acc']:.3f} fineF1={row['fine_change_f1']:.3f} "
              f"({secs}s)", flush=True)
    return pd.DataFrame(rows)


def best_kwargs(table, configs):
    """Kwargs of the top-merged-change-F1 non-baseline row in a scored table."""
    lut = dict(configs)
    ranked = table.sort_values("merged_change_f1", ascending=False)
    for name in ranked["variant"]:
        if name != "baseline_wide":
            return name, lut[name]
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path,
                        default=project_data_dir("analysis_results"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--min-class-count", type=int, default=20)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    frame, columns = build_frame(args.input)
    groups = frame["block_id"]
    target = target_for_legend(frame, LEGENDS["coarse3"], args.min_class_count)
    truth_fine = target.to_numpy()
    truth_merged = np.array([to_merged_label(t) for t in truth_fine])
    print(f"{len(frame):,} plots | {len(columns)} features | base={BASE} | "
          f"epochs={args.epochs} | {args.n_splits}-fold blocked CV")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("\n[MoE] mixture-of-experts trunk:")
        moe_cfgs = moe_configs()
        moe = run_table(moe_cfgs, frame, columns, target, groups, truth_fine,
                        truth_merged, args.epochs, args.n_splits)

        print("\n[Noise] interleaved noise injection:")
        noise_cfgs = noise_configs()
        noise = run_table(noise_cfgs, frame, columns, target, groups, truth_fine,
                          truth_merged, args.epochs, args.n_splits)

        # Combine the best of each: best MoE trunk + best noise schedule.
        best_moe_name, best_moe = best_kwargs(moe, moe_cfgs)
        best_noise_name, best_noise = best_kwargs(noise, noise_cfgs)
        noise_only = {k: v for k, v in best_noise.items() if k.startswith("noise_")}
        combined_cfgs = [
            ("baseline_wide", dict(**BASE)),
            (f"best_moe[{best_moe_name}]", dict(best_moe)),
            (f"best_noise[{best_noise_name}]", dict(best_noise)),
            (f"moe+noise", dict(best_moe, **noise_only)),
        ]
        print(f"\n[Combined] best_moe={best_moe_name} + best_noise={best_noise_name}:")
        combined = run_table(combined_cfgs, frame, columns, target, groups,
                             truth_fine, truth_merged, args.epochs, args.n_splits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    moe.sort_values("merged_change_f1", ascending=False).to_csv(
        args.output_dir / f"hier_moe{tag}.csv", index=False)
    noise.sort_values("merged_change_f1", ascending=False).to_csv(
        args.output_dir / f"hier_noise{tag}.csv", index=False)
    combined.to_csv(args.output_dir / f"hier_moe_noise_combined{tag}.csv", index=False)
    (args.output_dir / f"hier_moe_noise_meta{tag}.json").write_text(
        json.dumps({"input": str(args.input), "base_config": BASE,
                    "epochs": args.epochs, "n_splits": args.n_splits,
                    "best_moe": best_moe_name, "best_noise": best_noise_name},
                   indent=2), encoding="utf-8")
    print("\nMoE:\n" + moe.sort_values("merged_change_f1", ascending=False).to_string(index=False))
    print("\nNoise:\n" + noise.sort_values("merged_change_f1", ascending=False).to_string(index=False))
    print("\nCombined:\n" + combined.to_string(index=False))


if __name__ == "__main__":
    main()
