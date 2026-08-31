"""Cross-conformal (Vovk 2012) against the protocol `conformal_torchcp.py` used.

`arxiv.org/abs/1208.0806` defines the cross-conformal predictor (CCP): split the
training set into K folds, train K models each leaving one fold out, score each
fold's rows under the model that excluded them, and for a test object pool the
**counts** across folds into one p-value

    p(y) = ( sum_k #{ i in S_k : alpha_i >= alpha_k(x, y) } + 1 ) / (n + 1)

`conformal_torchcp.py` does **not** do that, and this measures what the
difference is worth. Three protocols, one nested design, the same outer rows:

* **`icp`** -- textbook inductive/split conformal. ONE model trained on 4/5 of the
  training pool, ONE calibration set (the other 1/5), one quantile. Valid by
  construction under exchangeability, and wastes 20% of the labels.
* **`ccp`** -- Vovk's cross-conformal. K models, K scores per test object, counts
  pooled into a single p-value. Uses every label for calibration. Only
  approximately valid: the pooled p-value is not a rank statistic of exchangeable
  variables, which is the paper's own caveat and the reason Barber et al. later
  bounded the CV+ family at 2*alpha rather than alpha.
* **`cvq`** -- what `conformal_torchcp.py` and `twotower_lab.nested_conformal`
  actually do. One quantile cut on the *pooled* out-of-fold scores of the whole
  training pool, applied to a single test score. Uses every label like CCP, but
  the pooled calibration scores come from K **different** models while the test
  score comes from one, so calibration and test scores are not identically
  distributed even under exchangeability.

The distinction that matters and is easy to miss: `cvq` gives the test object
ONE score and compares it to a heterogeneous pool; `ccp` gives it K scores and
compares each to its own fold's homogeneous pool. They coincide only if the K
models agree on the test object.

## Why this needs retraining and could not be read off the cache

CCP needs every fold model's opinion of every *test* object. The OOF cache holds
one posterior per row -- from the single model that excluded it -- so the other
K-1 opinions do not exist and cannot be reconstructed. Hence the nested design:
an outer blocked fold is held out, the K inner models are all trained without it,
and all K may legitimately score it.

Cost is ~25 model fits per seed, about a minute. That is affordable here only
because the fits are ~1s; do not assume it generalises to a heavier recipe.

Usage::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P src/conformal_crossconformal.py --n-seeds 5
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.model_selection import StratifiedGroupKFold

import twotower_lab as lab
from model_zoo import HierarchicalSoftmaxNN
from project_paths import project_data_dir

OUT = project_data_dir("analysis_results") / "conformal_crossconformal.csv"
N_INNER = 5


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def score_matrix(probs: np.ndarray, kind: str, u: np.ndarray | float) -> np.ndarray:
    """``(n, K)`` nonconformity scores, with the APS draw supplied rather than drawn.

    `twotower_lab.conformal_score_matrix` draws `u` internally from an rng, which
    is right when one predictor scores one matrix. It is **wrong here**: the
    randomisation belongs to the test *object*, not to the model, so the K fold
    models must see the same `u` for a given row. Drawing per fold instead makes
    CCP average over K independent randomisations, which is a different (and
    more conservative) estimator than the one Vovk defines, and would show up as
    a protocol effect that is really an implementation artefact.

    Passing `u` explicitly also pairs the two protocols: `ccp` and `cvq` then
    differ only in how scores are compared, not in which random numbers they saw.
    """
    if kind == "lac":
        return 1.0 - probs
    if kind != "aps":
        raise ValueError(f"unknown conformal score: {kind}")
    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    scores = np.empty_like(cum)
    np.put_along_axis(scores, order, cum - u * sorted_p, axis=1)
    return scores


def true_scores(scores: np.ndarray, y: np.ndarray) -> np.ndarray:
    return scores[np.arange(len(y)), y]


# ---------------------------------------------------------------------------
# the three protocols
# ---------------------------------------------------------------------------
def sets_icp(cal_scores, cal_y, test_scores, alpha, n_classes, mondrian):
    """Split conformal: one calibration set, one quantile (per class if Mondrian)."""
    q = np.zeros(n_classes)
    if mondrian:
        for k in range(n_classes):
            q[k] = lab.conformal_quantile(cal_scores[cal_y == k], alpha)
    else:
        q[:] = lab.conformal_quantile(cal_scores, alpha)
    return test_scores <= q[None, :]


def sets_cvq(cal_scores, cal_y, test_scores, alpha, n_classes, mondrian):
    """The protocol in use: one quantile from the POOLED out-of-fold scores.

    Identical arithmetic to `sets_icp`; what differs is that `cal_scores` here
    were produced by K different models rather than by the model that scores the
    test rows. Kept as its own function so the comparison is about where the
    scores came from, not about two implementations of a quantile.
    """
    return sets_icp(cal_scores, cal_y, test_scores, alpha, n_classes, mondrian)


def sets_ccp(fold_cal_scores, fold_cal_y, fold_test_scores, alpha, n_classes,
             mondrian):
    """Vovk's cross-conformal p-value, pooled over the K fold models.

    ``fold_cal_scores[k]``  -- true-class scores of fold k's rows under model k.
    ``fold_cal_y[k]``       -- their labels.
    ``fold_test_scores[k]`` -- ``(n_test, n_classes)`` scores of the test rows
                               under model k.

    The count is ``>=`` and the ``+1`` is on both numerator and denominator, per
    the paper: the test object is conceptually added to the calibration sample,
    which is what keeps the p-value from being anti-conservative at small n.
    """
    n_test = fold_test_scores[0].shape[0]
    counts = np.zeros((n_test, n_classes))
    denom = np.zeros(n_classes)
    for k in range(len(fold_cal_scores)):
        cal, cal_y = fold_cal_scores[k], fold_cal_y[k]
        test = fold_test_scores[k]
        if mondrian:
            for c in range(n_classes):
                rows = cal[cal_y == c]
                if len(rows) == 0:
                    continue
                # #{i in S_k, y_i = c : alpha_i >= alpha_k(x, c)}
                counts[:, c] += (rows[None, :] >= test[:, c][:, None]).sum(1)
                denom[c] += len(rows)
        else:
            for c in range(n_classes):
                counts[:, c] += (cal[None, :] >= test[:, c][:, None]).sum(1)
            denom += len(cal)
    p = (counts + 1.0) / (denom[None, :] + 1.0)
    return p > alpha


# ---------------------------------------------------------------------------
# ECCP -- the e-value cross-conformal predictor (arXiv 2606.03600)
# ---------------------------------------------------------------------------
#: `C_{m,alpha}` is a root-find per (fold size, alpha, s) and is reused across
#: every test row and candidate class, so it is cached rather than re-solved.
_P2E_CACHE: dict = {}


def _p2e_log_e(p, C, alpha, s):
    """``log F_{n,alpha}(p)`` -- computed in log space so large C cannot overflow.

    F(p) = (1/alpha) * (1 + exp(C(alpha - s))) / (1 + exp(C(p - s))), and
    `log(1 + exp(x))` is `logaddexp(0, x)`.
    """
    return (-np.log(alpha)
            + np.logaddexp(0.0, C * (alpha - s))
            - np.logaddexp(0.0, C * (np.asarray(p, dtype=float) - s)))


def p2e_constant(m: int, alpha: float, s_frac: float = 0.5):
    """Solve ``Sigma(C) = 1`` for the P2E calibrator; return ``(C, s)`` or None.

    The paper fixes C by requiring the calibrated variable to be an e-variable --
    mean 1 under the null, where a conformal p-value is uniform on the grid
    ``{1/(m+1), ..., (m+1)/(m+1)}``:

        Sigma(C) = 1/(m+1) * sum_{k=1}^{m+1} F_{m,alpha}(k/(m+1)) = 1

    `Sigma` is continuous and decreasing with `Sigma(0+) = 1/alpha > 1` and
    `Sigma(inf) = floor(alpha(m+1))/(alpha(m+1)) < 1`, so the root exists and is
    unique. `s` is a free parameter of the method on
    ``(alpha, ceil(alpha(m+1))/(m+1))``; `s_frac` places it in that interval.

    Returns None when the interval is empty, which is exactly the paper's
    ``alpha(m+1) not in N`` condition -- the calibrator does not exist there.
    """
    key = (m, alpha, s_frac)
    if key in _P2E_CACHE:
        return _P2E_CACHE[key]
    hi = np.ceil(alpha * (m + 1)) / (m + 1)
    if not (hi > alpha) or m < 1:
        _P2E_CACHE[key] = None
        return None
    s = alpha + s_frac * (hi - alpha)
    grid = np.arange(1, m + 2) / (m + 1)

    def sigma(C):
        return float(np.exp(_p2e_log_e(grid, C, alpha, s)).mean()) - 1.0

    lo_c, hi_c = 1e-8, 1e3
    while sigma(hi_c) > 0 and hi_c < 1e12:      # push the bracket out if needed
        hi_c *= 10.0
    if sigma(lo_c) < 0 or sigma(hi_c) > 0:
        _P2E_CACHE[key] = None
        return None
    C = float(brentq(sigma, lo_c, hi_c, xtol=1e-12, rtol=1e-14))
    _P2E_CACHE[key] = (C, s)
    return _P2E_CACHE[key]


def sets_eccp(fold_cal_scores, fold_cal_y, fold_test_scores, alpha, n_classes,
              mondrian, u_thresh, variant: str, s_frac: float = 0.5):
    """ECCP: fold-wise p-values -> e-values -> averaged, per arXiv 2606.03600.

    Why it is not just a re-parameterisation of `ccp`: Vovk's CCP pools the
    *counts* into one p-value and is only guaranteed at `1 - 2*alpha`; ECCP
    converts each fold's p-value to an e-value with a calibrator built so that
    the single-fold sets are unchanged (`F(alpha) = 1/alpha` exactly), then
    averages the e-values -- and an average of e-values is an e-value, which is
    what buys back the full `1 - alpha` guarantee.

    ``variant="rand"``  -- eq. 15, threshold `U/alpha` with `U ~ Unif(0,1)` drawn
                           once per test object and shared across its classes, so
                           the output is still one coherent set.
    ``variant="ex"``    -- eq. 16, `sup_{t<=K} (1/t) sum_{k<=t} E_k < 1/alpha`.
                           Deterministic, and it depends on the fold ORDER, which
                           is a property of the method and not of this code.

    The Mondrian reading -- restrict each fold's calibration rows to class `c`,
    so both the p-value and the calibrator's `m` become class-specific -- is an
    extension of the paper, which does not treat label-conditional coverage. It
    is required here: marginal conformal covers `Cropland -> Nature` 13% of the
    time on this legend.
    """
    n_test = fold_test_scores[0].shape[0]
    K = len(fold_cal_scores)
    e_vals = np.zeros((K, n_test, n_classes))
    for k in range(K):
        cal, cal_y = fold_cal_scores[k], fold_cal_y[k]
        test = fold_test_scores[k]
        for c in range(n_classes):
            rows = cal[cal_y == c] if mondrian else cal
            m = len(rows)
            const = p2e_constant(m, alpha, s_frac)
            if const is None:
                # No calibrator at this fold size. E = 1 is the neutral e-value
                # (no evidence against the class), which enlarges the set rather
                # than shrinking it -- the safe direction.
                e_vals[k, :, c] = 1.0
                continue
            C, s = const
            p = (1.0 + (rows[None, :] >= test[:, c][:, None]).sum(1)) / (m + 1.0)
            e_vals[k, :, c] = np.exp(_p2e_log_e(p, C, alpha, s))

    if variant == "rand":
        return e_vals.mean(0) < (u_thresh[:, None] / alpha)
    if variant == "ex":
        running = np.cumsum(e_vals, axis=0) / np.arange(1, K + 1)[:, None, None]
        return running.max(0) < (1.0 / alpha)
    raise ValueError(f"unknown ECCP variant: {variant}")


# ---------------------------------------------------------------------------
# the nested run
# ---------------------------------------------------------------------------
def fit_fold(view, cols, kwargs, seed, pool, train_idx, predict_idx_list):
    """Fit one model on ``train_idx``; return its posteriors on each index set.

    The state-pretraining pool is cut to the *training* blocks, exactly as
    `cv_probs_state_fine` does. Skipping that would leak the outer fold into the
    pretraining phase and quietly inflate every arm equally, which is the worst
    kind of bug: invisible in a comparison.
    """
    fold_pool = None
    if pool is not None:
        blocks = set(view.frame.iloc[train_idx]["block_id"].unique())
        fold_pool = pool[pool["block_id"].isin(blocks)]
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


def align(p_fine, model_classes, classes):
    """Model column order -> the level's sorted class order."""
    out = np.zeros((len(p_fine), len(classes)))
    out[:, [classes.index(c) for c in model_classes]] = p_fine
    return out


def run_seed(ctx, view, seed, alphas, kinds, s_frac=0.5):
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
    for outer, (train_idx, test_idx) in enumerate(view.folds):
        inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True,
                                     random_state=seed)
        splits = list(inner.split(train_idx, y_all[train_idx],
                                  blocks[train_idx]))
        fold_cal_probs, fold_cal_y, fold_test_probs = [], [], []
        for k, (tr_local, cal_local) in enumerate(splits):
            tr, cal = train_idx[tr_local], train_idx[cal_local]
            (cal_p, cls_a), (test_p, cls_b) = fit_fold(
                view, cols, kwargs, seed, pool, tr, [cal, test_idx])
            fold_cal_probs.append(align(cal_p, cls_a, classes))
            fold_cal_y.append(y_all[cal])
            fold_test_probs.append(align(test_p, cls_b, classes))

        y_test = y_all[test_idx]
        # `cvq` gives the test object ONE score. The mean posterior over the K
        # models is the honest single read: it is what the deployed raster does
        # (a seed ensemble), and using one arbitrary fold model instead would
        # confound the protocol comparison with a model-selection choice.
        mean_test_probs = np.mean(fold_test_probs, axis=0)
        pooled_cal_y = np.concatenate(fold_cal_y)

        for kind in kinds:
            rng = np.random.default_rng(seed * 100 + outer)
            # ONE draw per test row, reused by every fold model and by both
            # protocols; one draw per calibration row, which belongs to exactly
            # one fold so there is no sharing question.
            u_test = rng.random((len(test_idx), 1))
            u_cal = [rng.random((len(yy), 1)) for yy in fold_cal_y]
            # ECCP's own randomisation (eq. 15) is a threshold draw, separate
            # from APS's score draw: one per test object, shared across its
            # candidate classes so the output stays a single coherent set.
            u_eccp = rng.random(len(test_idx))

            f_cal_s = [true_scores(score_matrix(p, kind, u), yy)
                       for p, yy, u in zip(fold_cal_probs, fold_cal_y, u_cal)]
            f_test_s = [score_matrix(p, kind, u_test) for p in fold_test_probs]
            # `cvq` pools the SAME per-fold calibration scores rather than
            # rescoring the concatenation, so the two protocols are compared on
            # identical calibration numbers.
            pooled_cal_s = np.concatenate(f_cal_s)
            mean_test_s = score_matrix(mean_test_probs, kind, u_test)
            # ICP uses inner fold 0 alone: model 0 trained on 4/5 of the pool,
            # calibrated on the 1/5 it held out. A genuine split-conformal
            # predictor, not a subsample of the cross-conformal one.
            icp_test_s = f_test_s[0]

            for mondrian in (True, False):
                for alpha in alphas:
                    arms = {
                        "icp": sets_icp(f_cal_s[0], fold_cal_y[0], icp_test_s,
                                        alpha, n_classes, mondrian),
                        "ccp": sets_ccp(f_cal_s, fold_cal_y, f_test_s, alpha,
                                        n_classes, mondrian),
                        "cvq": sets_cvq(pooled_cal_s, pooled_cal_y, mean_test_s,
                                        alpha, n_classes, mondrian),
                        "eccp": sets_eccp(f_cal_s, fold_cal_y, f_test_s, alpha,
                                          n_classes, mondrian, u_eccp, "rand",
                                          s_frac),
                        "eccp_ex": sets_eccp(f_cal_s, fold_cal_y, f_test_s,
                                             alpha, n_classes, mondrian,
                                             u_eccp, "ex", s_frac),
                    }
                    for name, sets in arms.items():
                        rows.append(dict(
                            seed=seed, outer=outer, score=kind,
                            mode="mondrian" if mondrian else "marginal",
                            alpha=alpha, protocol=name,
                            **row_metrics(sets, y_test, classes)))
    return rows


def row_metrics(sets, y, classes) -> dict:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    out = {"n": len(y), "coverage": float(covered.mean()),
           "set_size": float(size.mean()),
           "singleton_frac": float((size == 1).mean()),
           "empty_frac": float((size == 0).mean())}
    per = []
    for k, cls in enumerate(classes):
        rows = y == k
        slug = str(cls).lower().replace(" -> ", "_to_").replace(" ", "")
        val = float(covered[rows].mean()) if rows.any() else np.nan
        out[f"cov_{slug}"] = val
        out[f"n_{slug}"] = int(rows.sum())
        if rows.any():
            per.append(val)
    out["cov_macro"] = float(np.mean(per))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--alphas", default="0.05,0.10,0.20")
    parser.add_argument("--scores", default="lac,aps")
    parser.add_argument("--s-frac", type=float, default=0.5,
                        help="where ECCP's free parameter s sits in "
                             "(alpha, ceil(alpha(m+1))/(m+1))")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",") if a]
    kinds = [s for s in args.scores.split(",") if s]
    ctx = lab.load_context()
    view = ctx.view("full")

    rows = []
    for seed in range(args.n_seeds):
        rows += run_seed(ctx, view, seed, alphas, kinds, args.s_frac)
        print(f"seed {seed} done ({len(rows)} rows)")
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out, index=False)

    # Outer folds are pooled by weighted mean, not by mean-of-folds: the folds
    # are unequal (1250..1316 rows) and the rare classes are not evenly spread.
    keys = ["score", "mode", "alpha", "protocol"]
    num = [c for c in frame.columns if c.startswith(("cov", "set_", "singleton",
                                                     "empty"))]
    agg = (frame.assign(**{c: frame[c] * frame["n"] for c in num})
           .groupby(keys + ["seed"])[num + ["n"]].sum())
    agg = agg[num].div(agg["n"], axis=0).groupby(level=keys).agg(["mean", "std"])

    for kind in kinds:
        for mode in ("mondrian", "marginal"):
            part = agg.loc[(kind, mode)]
            print(f"\n{'=' * 84}\n{kind.upper()}  {mode}  "
                  f"({args.n_seeds} seeds, pooled over 5 outer folds)\n{'=' * 84}")
            show = part[[("coverage", "mean"), ("coverage", "std"),
                         ("cov_macro", "mean"), ("set_size", "mean"),
                         ("set_size", "std"), ("singleton_frac", "mean"),
                         ("empty_frac", "mean")]]
            show.columns = ["coverage", "cov_sd", "cov_macro", "set_size",
                            "size_sd", "singleton", "empty"]
            print(show.round(4).to_string())
    print(f"\n{len(frame)} rows -> {args.out}")


if __name__ == "__main__":
    main()
