"""The deployable conformal option: exact 1-alpha, ONE predictor at inference.

Everything in `conformal_crossconformal.py` that carries a guarantee -- CCP,
ECCP, ECCP-Exch, WECA -- needs K trained models at serving. This asks the only
question left once that is ruled out: **how good can split conformal be made on
this legend, and what does the guarantee cost?**

## Why split conformal is the whole search space

arXiv 2606.03600's P2E calibrator is *set-preserving* by construction --
`F(alpha) = 1/alpha` exactly, so `{p > alpha}` and `{F(p) < 1/alpha}` are the
same set. At K=1 the e-value machinery therefore returns byte-identical sets to
plain split conformal (verified: zero disagreements on the p-grid). Its value is
entirely in *aggregating* folds or models, and aggregation is what costs K
forward passes. So with a one-predictor budget the menu has one item on it, and
the only free parameters are the split ratio and what to do about a class too
rare to calibrate.

## The trade this measures

Split conformal must hold labels back from training to calibrate on them. That is
not free on a model whose learning curve is still climbing at +0.026 change-F1
per doubling of labels, and it is *especially* not free for a Mondrian quantile,
which needs `ceil((m_c+1)(1-alpha)) <= m_c` rows **of class c** -- 9 at
alpha 0.10, 19 at 0.05. `Artificial -> Cropland` has 46 plots in total, so the
calibration fraction decides whether that class has a threshold at all. Too small
a fraction and its quantile is infinite (the class joins every set and efficiency
collapses); too large and the model is trained on too little.

Both sides are reported, because only one of them is usually remembered:

* `set_size` / `coverage`  -- the conformal side.
* `acc`, `change_f1`       -- the model side. A tighter set on a worse model is
                              not a better product.

## The two inference budgets

* `single` -- one network. The cheapest thing that can carry a guarantee.
* `ens5`   -- the five-seed mean, which is **what the deployed raster already
  runs** (`infer_s2.py --seeds 5`, 5 forward passes, 6 s for Oslo). For conformal
  purposes an ensemble is just "the model", so this buys ensemble accuracy at
  exactly today's serving cost and still has an exact guarantee.

Three-way blocked split: outer fold = test (never seen), the rest split into
proper-training and calibration at `--fracs`.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P src/conformal_deployable.py --n-repeats 5
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import twotower_lab as lab
from model_zoo import HierarchicalSoftmaxNN
from project_paths import project_data_dir

OUT = project_data_dir("analysis_results") / "conformal_deployable.csv"
ENS = 5


def blocked_split(idx, y, blocks, cal_frac, seed):
    """Split ``idx`` into (proper, calibration) holding whole blocks together.

    `StratifiedGroupKFold` at ``n_splits = round(1/cal_frac)`` gives a blocked
    split whose calibration side is ~`cal_frac` of the rows, stratified on the
    coarse3 label so a rare class is not concentrated on one side by luck. Taking
    a random subset of rows instead would put neighbouring plots on both sides
    and quietly inflate every number here.

    **Only ``1/n_splits`` fractions are reachable**, so 0.4, 0.5 and 0.6 all
    resolve to ``n_splits=2`` and produce the identical 50/50 split -- their rows
    in the output are one measurement, not three. `n_proper` and `n_cal` are
    reported per row so the achieved fraction is always visible rather than
    inferred from the requested one.
    """
    n_splits = max(2, int(round(1.0 / cal_frac)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=seed)
    proper, cal = next(iter(splitter.split(idx, y[idx], blocks[idx])))
    return idx[proper], idx[cal]


def fit_predict(view, cols, kwargs, seed, pool, train_idx, predict_idx_list):
    fold_pool = None
    if pool is not None:
        blks = set(view.frame.iloc[train_idx]["block_id"].unique())
        fold_pool = pool[pool["block_id"].isin(blks)]
    model = HierarchicalSoftmaxNN(cols, seed=seed, **kwargs)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(view.frame.iloc[train_idx],
                  view.target.iloc[train_idx].to_numpy(), state_frame=fold_pool)
        for idx in predict_idx_list:
            frame = view.frame.iloc[idx].copy()
            frame[lab.S2_MASK] = 0.0           # the deployed gate-off read
            p_fine, _ = model._probs(frame)
            out.append((p_fine, list(model.fine_classes_)))
    return out


def align(p, model_classes, classes):
    out = np.zeros((len(p), len(classes)))
    out[:, [classes.index(c) for c in model_classes]] = p
    return out


def icp_sets(cal_scores, cal_y, test_scores, alpha, n_classes, mondrian):
    """Split-conformal sets, and the count of classes with no usable quantile.

    An infinite quantile is the honest answer for a class whose calibration
    sample cannot support the level -- it puts that class in *every* set. It is
    reported rather than patched: falling back to the marginal quantile there
    would silently drop the class-conditional guarantee for exactly the classes
    the guarantee was wanted for.
    """
    q = np.zeros(n_classes)
    if mondrian:
        for k in range(n_classes):
            q[k] = lab.conformal_quantile(cal_scores[cal_y == k], alpha)
    else:
        q[:] = lab.conformal_quantile(cal_scores, alpha)
    return test_scores <= q[None, :], int(np.isinf(q).sum())


def metrics(sets, y, classes, probs, truth_fine, n_inf):
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    pred = np.array(classes, dtype=object)[probs.argmax(1)]
    out = {"coverage": float(covered.mean()), "set_size": float(size.mean()),
           "singleton_frac": float((size == 1).mean()),
           "empty_frac": float((size == 0).mean()),
           "n_inf_quantile": n_inf, "n": len(y),
           "acc": float((pred == truth_fine).mean())}
    # The model side of the trade, on the same rows: merged2 change-F1 is the
    # ledger's headline and is what a smaller training set actually costs.
    merged_pred = np.array([lab.to_merged_label(p) for p in pred])
    merged_truth = np.array([lab.to_merged_label(t) for t in truth_fine])
    out["change_f1"] = lab.change_metrics(merged_truth, merged_pred)["change_f1"]
    per = []
    for k, cls in enumerate(classes):
        rows = y == k
        slug = str(cls).lower().replace(" -> ", "_to_").replace(" ", "")
        val = float(covered[rows].mean()) if rows.any() else np.nan
        out[f"cov_{slug}"] = val
        out[f"ncal_{slug}"] = 0
        if rows.any():
            per.append(val)
    out["cov_macro"] = float(np.mean(per))
    return out


def run(ctx, view, repeats, fracs, alphas, kinds):
    cols, kwargs = lab.siam_s2off_kwargs(
        ctx, siam_cos_weight=lab.SIAM_AUX, siam_cos_margin=0.3,
        siam_state_weight=0.0, siam_state_source="external",
        siam_state_pretrain=30)
    pool = lab._state_pool()
    classes = sorted(set(view.truth_fine))
    n_classes = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    y_all = np.array([index[t] for t in view.truth_fine])
    blocks = view.frame["block_id"].to_numpy()

    rows = []
    for rep in range(repeats):
        for outer, (pool_idx, test_idx) in enumerate(view.folds):
            for frac in fracs:
                proper, cal = blocked_split(pool_idx, y_all, blocks, frac,
                                            seed=rep * 17 + outer)
                cal_p, test_p = [], []
                for s in range(ENS):
                    (a, ca), (b, cb) = fit_predict(
                        view, cols, kwargs, rep * 100 + s, pool, proper,
                        [cal, test_idx])
                    cal_p.append(align(a, ca, classes))
                    test_p.append(align(b, cb, classes))
                budgets = {"single": (cal_p[0], test_p[0]),
                           f"ens{ENS}": (np.mean(cal_p, 0), np.mean(test_p, 0))}
                for budget, (cp, tp) in budgets.items():
                    for kind in kinds:
                        rng = np.random.default_rng(rep * 1000 + outer)
                        u_c = rng.random((len(cal), 1))
                        u_t = rng.random((len(test_idx), 1))
                        cs = lab.conformal_score_matrix(cp, kind, None) \
                            if kind == "lac" else _aps(cp, u_c)
                        ts = lab.conformal_score_matrix(tp, kind, None) \
                            if kind == "lac" else _aps(tp, u_t)
                        cal_true = cs[np.arange(len(cal)), y_all[cal]]
                        for mondrian in (True, False):
                            for alpha in alphas:
                                sets, n_inf = icp_sets(
                                    cal_true, y_all[cal], ts, alpha, n_classes,
                                    mondrian)
                                row = dict(
                                    repeat=rep, outer=outer, cal_frac=frac,
                                    budget=budget, score=kind, alpha=alpha,
                                    mode="mondrian" if mondrian else "marginal",
                                    n_proper=len(proper), n_cal=len(cal))
                                row.update(metrics(
                                    sets, y_all[test_idx], classes, tp,
                                    view.truth_fine[test_idx], n_inf))
                                # The binding constraint, recorded per row: how
                                # many calibration plots the rarest class got.
                                row["ncal_min"] = int(min(
                                    (y_all[cal] == k).sum()
                                    for k in range(n_classes)))
                                rows.append(row)
        print(f"repeat {rep} done ({len(rows)} rows)")
    return rows


def _aps(probs, u):
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    out = np.empty_like(cum)
    np.put_along_axis(out, order, cum - u * sorted_p, axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--fracs", default="0.25,0.4,0.5,0.6")
    ap.add_argument("--alphas", default="0.05,0.10,0.20")
    ap.add_argument("--scores", default="lac")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    ctx = lab.load_context()
    view = ctx.view("full")
    rows = run(ctx, view, args.n_repeats,
               [float(f) for f in args.fracs.split(",") if f],
               [float(a) for a in args.alphas.split(",") if a],
               [s for s in args.scores.split(",") if s])
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)

    num = ["coverage", "cov_macro", "set_size", "singleton_frac", "empty_frac",
           "acc", "change_f1"]
    keys = ["score", "mode", "alpha", "budget", "cal_frac"]
    w = frame.assign(**{c: frame[c] * frame["n"] for c in num})
    g = w.groupby(keys + ["repeat"])[num + ["n"]].sum()
    g = g[num].div(g["n"], axis=0).groupby(level=keys).mean()
    extra = frame.groupby(keys)[["n_proper", "n_cal", "ncal_min",
                                 "n_inf_quantile"]].mean()
    show = g.join(extra)

    for kind in frame.score.unique():
        for mode in ("mondrian", "marginal"):
            for alpha in sorted(frame.alpha.unique()):
                part = show.loc[(kind, mode, alpha)]
                print(f"\n{'=' * 96}\n{kind.upper()} {mode} alpha={alpha:.2f}  "
                      f"({args.n_repeats} repeats x 5 outer folds)\n{'=' * 96}")
                print(part.round(4).to_string())
    print(f"\n{len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
