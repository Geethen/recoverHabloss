"""Ablation of the deployed Sentinel-2 model: one architecture change per rung.

`mc_s2_drop0.7` is not one design, it is seven decisions stacked on an
AlphaEarth-only network, each taken in a different iteration against a different
metric and never priced together. This walks the ladder one change at a time and
reports, for every rung, **what it bought and what it costs to run**:

    A  aef_flat        AlphaEarth only, flat wide MLP                [reference]
    B  aef_s2_flat     + Sentinel-2 features, still flat (no towers)
    C  tt_additive     + two towers (additive, AlphaEarth always on, md=0)
    D  tt_gated_mean   + symmetric gating, gated_mean fusion
    E  tt_md0.5        + modality dropout 0.5
    F  tt_drop0.7      + asymmetric detail-tower dropout (0.7 vs 0.4)
    G  mc_exact        + stochastic gate, marginalised exactly      [deployed]
    -  G_mc16          G with the gate SAMPLED in 16 passes  (cost-only rung)
    -  G_deterministic G with the gate hard on                (the failure mode)
    -  G_noflat        G minus hierarchical supervision (fine head only)
    H  G_seeds5        G averaged over 5 torch seeds

Accuracy is out-of-fold under the same spatially blocked CV, the same legend and
the same seeds for every rung, so the deltas are attributable to the rung and
nothing else. Inference cost is measured on the **deployed code path**
(``infer_s2.predict``) over a fixed block of pixels, so the number is the one
that governs a global run rather than a proxy for it.

Two corrections to how the lab scored these
-------------------------------------------
1. ``twotower_lab._mc_s2`` averages the gate for the merged2 head but reads the
   coarse3 head from a single deterministic pass. The shipped coarse3 map
   averages it. Every rung here scores both heads exactly as the map produces
   them, so ``coarse3_change_f1`` is finally the map's number.
2. The MC gate is marginalised, not sampled (see ``infer_s2.predict``). Sampling
   16 passes is an estimator of what two passes compute exactly; it is kept as
   a cost-only rung because it is what earlier runs did.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import infer_s2
from model_zoo import HierarchicalSoftmaxNN
from project_paths import project_data_dir
from twotower_lab import (AEF_MASK, BASE, S2_MASK, Context, View, load_context,
                          s2_base_columns, score_probs)

KEEP = infer_s2.MC_KEEP
OSLO_PX = 2_954_952   # the AOI every deployed map has been judged on


# ---------------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------------
def rungs(ctx: Context) -> list[dict]:
    """Ordered ladder; each entry is the previous one plus a single change."""
    aef = ctx.aef_cols
    s2 = s2_base_columns(ctx.s2_stat_cols)   # the published seven-channel block

    def tt(**over):
        """Two-tower kwargs; `over` is the one thing this rung changes."""
        kw = dict(arch="two_tower", loss=BASE["loss"], epochs=BASE["epochs"],
                  tower_dim=256, aef_columns=aef, tess_columns=s2,
                  mask_column=S2_MASK, aef_mask_column=None,
                  fusion="additive", modality_dropout=0.0)
        kw.update(over)
        return kw

    sym = dict(fusion="gated_mean", aef_mask_column=AEF_MASK)
    return [
        dict(key="A_aef_flat", change="AlphaEarth only, flat wide MLP",
             cols=aef, kwargs=dict(BASE), gate="none"),
        dict(key="B_aef_s2_flat", change="+ Sentinel-2 features (flat concat)",
             cols=aef + s2, kwargs=dict(BASE), gate="none"),
        dict(key="C_tt_additive", change="+ two towers (additive, md=0)",
             cols=aef + s2, kwargs=tt(), gate="det"),
        dict(key="D_tt_gated_mean", change="+ symmetric gating, gated_mean",
             cols=aef + s2, kwargs=tt(**sym), gate="det"),
        dict(key="E_tt_md0.5", change="+ modality dropout 0.5",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, **sym), gate="det"),
        dict(key="F_tt_drop0.7", change="+ asymmetric detail dropout 0.7",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="det"),
        dict(key="G_mc_exact", change="+ stochastic gate, exact  [DEPLOYED]",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="mc"),
        # -- variants of G, to price the two things that are inference-only ---
        dict(key="G_mc16_sampled", change="G, gate SAMPLED in 16 passes",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="mc16"),
        dict(key="G_deterministic", change="G, gate hard ON (keep=1)",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="det"),
        dict(key="G_aef_only_view", change="G, gate hard OFF (keep=0)",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="off"),
        # -- the supervision structure itself ---------------------------------
        dict(key="G_no_hierarchy", change="G minus hierarchical supervision",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      level_weights=(0.0, 0.0, 1.0), **sym),
             gate="mc"),
        dict(key="H_seeds5", change="+ 5-seed ensemble  [deployed map]",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="mc", ensemble=5),
        # The model that actually ships was never in this cost table. Its
        # accuracy is G_aef_only_view's -- same weights, same read -- but its
        # cost is a different code path: `probs_aef_only_matrix` builds neither
        # the detail block nor a DataFrame, so it is the only rung whose serving
        # cost is not dominated by pandas.
        dict(key="G_s2off_deploy", change="G served gate-off  [DEPLOYED PATH]",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="off", deploy="aef_only"),
        dict(key="G_s2off_deploy5", change="the deployed path, 5 seeds",
             cols=aef + s2, kwargs=tt(modality_dropout=0.5, dropout_tess=0.7,
                                      **sym), gate="off", deploy="aef_only",
             ensemble=5),
    ]


# ---------------------------------------------------------------------------
# accuracy: blocked CV, both heads read the way the map reads them
# ---------------------------------------------------------------------------
def _read(model, frame, gate: str, rng=None):
    """(p_fine, p_merged) under a rung's gate policy, on one held-out block.

    ``mc`` is the exact marginalisation the deployed path uses; ``mc16`` is the
    old sampled estimator of the same quantity, kept so the two can be compared
    on identical folds.
    """
    has_s2 = frame[S2_MASK].to_numpy("float32")
    v = frame.copy()

    def at(mask):
        v[S2_MASK] = mask
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model._probs(v)

    if gate == "none":
        return at(has_s2)
    if gate == "det":
        return at(has_s2)
    if gate == "off":
        return at(np.zeros_like(has_s2))
    if gate == "mc16":
        af = am = 0.0
        for _ in range(infer_s2.MC_PASSES):
            keep = rng.random(len(frame)) < KEEP
            f, m = at(np.where(has_s2 > 0.5, keep.astype("float32"), 0.0))
            af, am = af + f / infer_s2.MC_PASSES, am + m / infer_s2.MC_PASSES
        return af, am
    if gate == "mc":
        f0, m0 = at(np.zeros_like(has_s2))
        f1, m1 = at(has_s2)
        w = (has_s2 > 0.5).astype("float64")[:, None] * KEEP
        return w * f1 + (1 - w) * f0, w * m1 + (1 - w) * m0
    raise ValueError(gate)


def cv_rung(view: View, rung: dict, seeds: list[int]) -> list[dict]:
    """One row of metrics per seed, out-of-fold over the blocked folds."""
    classes = view.merged_classes
    n_members = rung.get("ensemble", 1)
    rows = []
    for seed in seeds:
        probs = np.zeros((len(view.target), len(classes)))
        fine = np.empty(len(view.target), dtype=object)
        rng = np.random.default_rng(seed)
        for tr, te in view.folds:
            te_frame = view.frame.iloc[te]
            acc_m = np.zeros((len(te), len(classes)))
            acc_f = None
            fine_classes = None
            for member in range(n_members):
                model = HierarchicalSoftmaxNN(rung["cols"], seed=seed * 100 + member,
                                              **rung["kwargs"])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(view.frame.iloc[tr], view.target.iloc[tr].to_numpy())
                p_fine, p_merged = _read(model, te_frame, rung["gate"], rng)
                block = np.zeros((len(te), len(classes)))
                block[:, [classes.index(c) for c in model.merged_classes_]] = p_merged
                acc_m += block / n_members
                # Fine class lists are fold-local; a fold missing a rare
                # transition must not shift the columns of the average.
                if acc_f is None:
                    fine_classes = list(model.fine_classes_)
                    acc_f = np.zeros((len(te), len(fine_classes)))
                if list(model.fine_classes_) != fine_classes:
                    raise SystemExit("ensemble members disagree on fine classes")
                acc_f += p_fine / n_members
            probs[te] = acc_m
            fine[te] = np.array(fine_classes, dtype=object)[acc_f.argmax(1)]
        row = score_probs(view, probs, fine)
        row.update(seed=seed, key=rung["key"])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# cost: the deployed inference path, on a fixed block of pixels
# ---------------------------------------------------------------------------
def time_rung(ctx: Context, rung: dict, n_px: int, batch: int) -> dict:
    """Seconds to predict ``n_px`` pixels through ``infer_s2.predict``.

    Fitting is excluded -- it happens once per deployment, while inference is
    what multiplies by area. The pixel block is built by tiling real plot rows,
    so the feature distribution (and therefore the pandas/torch work) is
    realistic; only the count is synthetic.
    """
    aef_cols = [c for c in rung["cols"] if c in set(ctx.aef_cols)]
    s2_cols = [c for c in rung["cols"] if c not in set(ctx.aef_cols)]
    frame = ctx.frame
    reps = int(np.ceil(n_px / len(frame)))
    take = np.tile(np.arange(len(frame)), reps)[:n_px]
    # Band-major, matching `infer_s2.predict`'s contract (one row per column).
    aef_mat = np.ascontiguousarray(frame[aef_cols].to_numpy("float32")[take].T)
    s2_mat = (frame[s2_cols].to_numpy("float32")[take] if s2_cols
              else np.zeros((n_px, 0), "float32"))
    s2_mat = np.nan_to_num(s2_mat, nan=0.0)
    present = (frame[S2_MASK].to_numpy() > 0.5)[take]

    model = HierarchicalSoftmaxNN(rung["cols"], seed=0, **rung["kwargs"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(frame, ctx.view("full").target.to_numpy())
    members = [model] * rung.get("ensemble", 1)
    mc = rung["gate"] in ("mc", "mc16")
    entry = {"models": members, "spec": {"columns": rung["cols"], "mc": mc}}
    if rung.get("deploy"):
        entry["spec"]["deploy"] = rung["deploy"]
        s2_mat = np.zeros((n_px, 0), "float32")

    def run():
        t = time.time()
        infer_s2.predict(entry, aef_mat, s2_mat, present, batch, 0,
                         want_off=False, mc_sampling=(rung["gate"] == "mc16"))
        return time.time() - t

    run()                       # warm caches / lazy CUDA init
    secs = min(run(), run())
    passes = (infer_s2.MC_PASSES if rung["gate"] == "mc16"
              else 2 if mc else 1) * len(members)
    return {"key": rung["key"], "n_px": n_px, "seconds": round(secs, 2),
            "forward_passes": passes,
            "s_per_Mpx": round(secs / n_px * 1e6, 1),
            "oslo_s": round(secs / n_px * OSLO_PX, 1)}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["accuracy", "timing", "both"], default="both")
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="5 is the project's bar for calling a win; 3 has already "
                         "produced one false verdict on this exact model (S10)")
    ap.add_argument("--only", default="", help="comma-separated rung keys")
    ap.add_argument("--n-px", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--out", type=Path,
                    default=project_data_dir("analysis_results"))
    args = ap.parse_args()

    ctx = load_context()
    view = ctx.view("full")
    ladder = rungs(ctx)
    if args.only:
        want = set(args.only.split(","))
        ladder = [r for r in ladder if r["key"] in want]
    print(f"{len(view.target):,} plots | aef={len(ctx.aef_cols)} "
          f"s2={len(ctx.s2_stat_cols)} | {len(ladder)} rungs", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase in ("accuracy", "both"):
        rows = []
        for rung in ladder:
            t0 = time.time()
            got = cv_rung(view, rung, list(range(args.n_seeds)))
            rows.extend(got)
            d = pd.DataFrame(got)
            print(f"  {rung['key']:18s} change_f1={d.change_f1.mean():.4f}"
                  f"±{d.change_f1.std():.4f}  macro={d.macro_f1.mean():.4f}  "
                  f"coarse3={d.fine_change_f1.mean():.4f}  "
                  f"artStab={d.art_stable_recall.mean():.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(args.out / "s2_ablation_accuracy.csv",
                                      index=False)
        print(f"-> {args.out / 's2_ablation_accuracy.csv'}")

    if args.phase in ("timing", "both"):
        times = []
        for rung in ladder:
            t = time_rung(ctx, rung, args.n_px, args.batch)
            times.append(t)
            print(f"  {rung['key']:18s} {t['seconds']:7.2f}s /{args.n_px:,}px  "
                  f"{t['forward_passes']:3d} passes  "
                  f"Oslo {t['oslo_s']:8.1f}s", flush=True)
            pd.DataFrame(times).to_csv(args.out / "s2_ablation_timing.csv",
                                       index=False)
        (args.out / "s2_ablation_timing.json").write_text(json.dumps(times, indent=2))
        print(f"-> {args.out / 's2_ablation_timing.csv'}")


if __name__ == "__main__":
    main()
