"""
Design-based area estimators for stratified samples + PPI variants.
Estimand: proportion of area in a transition class (then x total area -> km2).
"""
import numpy as np

# ---------------------------------------------------------------- classical
def stratified_prop(y, strat, Nh, alpha=0.05):
    """Standard stratified estimator of a proportion (Olofsson et al. 2014 / Cochran).
    y: 0/1 indicator, strat: stratum id per obs, Nh: dict stratum -> area/size."""
    y = np.asarray(y, float); strat = np.asarray(strat)
    us = np.array([s for s in np.unique(strat) if s in Nh])
    N = sum(Nh[s] for s in us)
    p = 0.0; var = 0.0
    for s in us:
        m = strat == s
        nh = int(m.sum())
        if nh == 0: continue
        W = Nh[s] / N
        ph = y[m].mean()
        p += W * ph
        if nh > 1:
            sh2 = y[m].var(ddof=1)                 # unbiased stratum variance
            fpc = max(0.0, 1.0 - nh / Nh[s])       # finite population correction
            var += W**2 * (sh2 / nh) * fpc
    se = np.sqrt(var)
    z = 1.959963984540054 if abs(alpha-0.05)<1e-9 else _z(alpha)
    return p, se, (p - z*se, p + z*se)

def _z(alpha):
    if abs(alpha - 0.05) < 1e-9:
        return 1.959963984540054
    from scipy.stats import norm
    return norm.ppf(1 - alpha/2)

def hajek_prop(y, w, alpha=0.05):
    """Hajek (weighted ratio) estimator with linearised variance. w = design weights (N_h/n_h)."""
    y = np.asarray(y,float); w = np.asarray(w,float)
    p = np.sum(w*y)/np.sum(w)
    # linearisation: residuals
    n = len(y)
    r = w*(y - p)
    var = np.sum(r**2) / (np.sum(w)**2) * (n/(n-1))
    se = np.sqrt(var)
    z=_z(alpha)
    return p, se, (p-z*se, p+z*se)

# ---------------------------------------------------------------- difference / PPI
def difference_prop(y, yhat, yhat_pop_mean, w=None, lam=1.0, alpha=0.05):
    """Difference estimator == PPI. yhat_pop_mean: mean prediction over the WHOLE map (known).
    p_hat = lam*mean_pop(yhat) + weighted_mean(y - lam*yhat)"""
    y=np.asarray(y,float); yhat=np.asarray(yhat,float)
    n=len(y)
    if w is None: w=np.ones(n)
    w=np.asarray(w,float); wn = w/w.sum()*n          # normalise to mean 1 (Hajek-style, as ppi_py does)
    rect = np.mean(wn*(y - lam*yhat))
    p = lam*yhat_pop_mean + rect
    var = np.var(wn*(y-lam*yhat), ddof=1)/n           # rectifier variance only (pop mean treated known)
    se=np.sqrt(var); z=_z(alpha)
    return p, se, (p-z*se, p+z*se)

def optimal_lam(y, yhat, w=None):
    """PPI++ power tuning: lam* = cov(y,yhat)/var(yhat), clipped to [0,1]."""
    y=np.asarray(y,float); yhat=np.asarray(yhat,float); n=len(y)
    if w is None: w=np.ones(n)
    wn=np.asarray(w,float); wn=wn/wn.sum()*n
    ybar=np.sum(wn*y)/n; yhbar=np.sum(wn*yhat)/n
    cov=np.sum(wn*(y-ybar)*(yhat-yhbar))/n
    v=np.sum(wn*(yhat-yhbar)**2)/n
    if v<=0: return 0.0
    return float(np.clip(cov/v, 0, 1))


def stratified_ppi_prop(
    y,
    yhat,
    strat,
    Nh,
    yhat_pop_mean,
    yhat_pop_var=None,
    yhat_pop_n=None,
    lam=1.0,
    alpha=0.05,
):
    """Stratified PPI/difference estimator for a population proportion.

    ``yhat_pop_mean`` is a mapping from stratum to the mean prediction in a
    large predicted-only probability sample. If that mean comes from a finite
    sample rather than a complete map reduction, pass its sample variance and
    sample size through ``yhat_pop_var`` and ``yhat_pop_n``. The labelled and
    predicted-only samples are treated as independent within strata.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    strat = np.asarray(strat)
    yhat_pop_var = {} if yhat_pop_var is None else yhat_pop_var
    yhat_pop_n = {} if yhat_pop_n is None else yhat_pop_n
    total = float(sum(Nh.values()))
    point = 0.0
    variance = 0.0

    for h, Ah in Nh.items():
        mask = strat == h
        nh = int(mask.sum())
        if nh == 0:
            raise ValueError(f"No labelled observations in stratum {h!r}")
        if h not in yhat_pop_mean:
            raise ValueError(f"No predicted-only observations in stratum {h!r}")

        Wh = float(Ah) / total
        residual = y[mask] - lam * yhat[mask]
        point += Wh * (lam * float(yhat_pop_mean[h]) + residual.mean())
        if nh > 1:
            variance += Wh**2 * residual.var(ddof=1) / nh

        nu = int(yhat_pop_n.get(h, 0))
        if nu > 1:
            variance += (
                Wh**2
                * lam**2
                * float(yhat_pop_var.get(h, 0.0))
                / nu
            )

    se = float(np.sqrt(max(0.0, variance)))
    z = _z(alpha)
    return point, se, (point - z * se, point + z * se)


def optimal_lam_stratified(
    y,
    yhat,
    strat,
    Nh,
    yhat_pop_var=None,
    yhat_pop_n=None,
):
    """Variance-minimising scalar lambda for ``stratified_ppi_prop``.

    This includes uncertainty from a finite predicted-only sample. A proxy
    that is constant within every design stratum has zero denominator and
    therefore returns lambda=0: it adds no information beyond stratification.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    strat = np.asarray(strat)
    yhat_pop_var = {} if yhat_pop_var is None else yhat_pop_var
    yhat_pop_n = {} if yhat_pop_n is None else yhat_pop_n
    total = float(sum(Nh.values()))
    numerator = 0.0
    denominator = 0.0

    for h, Ah in Nh.items():
        mask = strat == h
        nh = int(mask.sum())
        if nh < 2:
            continue
        Wh2 = (float(Ah) / total) ** 2
        cov = np.cov(y[mask], yhat[mask], ddof=1)[0, 1]
        numerator += Wh2 * cov / nh
        denominator += Wh2 * yhat[mask].var(ddof=1) / nh

        nu = int(yhat_pop_n.get(h, 0))
        if nu > 1:
            denominator += (
                Wh2 * float(yhat_pop_var.get(h, 0.0)) / nu
            )

    if denominator <= 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


# ---------------------------------------------------------------- vector-valued
def stratified_multinomial(Y, strat, Nh, alpha=0.05):
    """Vector-valued stratified estimator of a composition (K classes jointly).

    ``Y`` is an ``n x K`` matrix of 0/1 class indicators (one-hot rows, so each
    row sums to 1 when every observation is classified). Returns the estimated
    composition vector ``p`` (length K) and its full ``K x K`` covariance matrix
    ``Sigma``. Each class marginal reproduces :func:`stratified_prop`; the
    off-diagonal terms carry the within-stratum negative covariance between
    shares that scalar per-class estimation discards.
    """
    Y = np.asarray(Y, float)
    if Y.ndim != 2:
        raise ValueError("Y must be an n x K indicator matrix")
    strat = np.asarray(strat)
    K = Y.shape[1]
    us = np.array([s for s in np.unique(strat) if s in Nh])
    N = sum(Nh[s] for s in us)
    p = np.zeros(K)
    Sigma = np.zeros((K, K))
    for s in us:
        m = strat == s
        nh = int(m.sum())
        if nh == 0:
            continue
        W = Nh[s] / N
        Yh = Y[m]
        p += W * Yh.mean(axis=0)
        if nh > 1:
            # unbiased within-stratum class covariance, then Wh^2 * S / nh * fpc
            Sh = np.cov(Yh, rowvar=False, ddof=1)
            Sh = np.atleast_2d(Sh)
            fpc = max(0.0, 1.0 - nh / Nh[s])
            Sigma += W**2 * (Sh / nh) * fpc
    return p, Sigma


def stratified_ppi_multinomial(
    Y,
    Yhat,
    strat,
    Nh,
    Yhat_pop_mean,
    Yhat_pop_cov=None,
    Yhat_pop_n=None,
    lam=1.0,
    alpha=0.05,
):
    """Vector-valued stratified PPI/difference estimator for a K-class composition.

    ``Y`` and ``Yhat`` are ``n x K`` labelled truth and prediction matrices.
    ``Yhat_pop_mean[h]`` is the length-K mean prediction in the predicted-only
    sample for stratum ``h``. If that mean comes from a finite predicted-only
    sample, pass its ``K x K`` sample covariance in ``Yhat_pop_cov[h]`` and its
    size in ``Yhat_pop_n[h]``. ``lam`` may be a scalar or a length-K vector of
    per-class power-tuning weights. Returns the composition vector ``p`` and its
    ``K x K`` covariance ``Sigma``; labelled and predicted-only samples are
    treated as independent within strata.
    """
    Y = np.asarray(Y, float)
    Yhat = np.asarray(Yhat, float)
    if Y.shape != Yhat.shape or Y.ndim != 2:
        raise ValueError("Y and Yhat must be matching n x K matrices")
    strat = np.asarray(strat)
    K = Y.shape[1]
    lam = np.broadcast_to(np.asarray(lam, float), (K,))
    Yhat_pop_cov = {} if Yhat_pop_cov is None else Yhat_pop_cov
    Yhat_pop_n = {} if Yhat_pop_n is None else Yhat_pop_n
    total = float(sum(Nh.values()))
    p = np.zeros(K)
    Sigma = np.zeros((K, K))

    for h, Ah in Nh.items():
        mask = strat == h
        nh = int(mask.sum())
        if nh == 0:
            raise ValueError(f"No labelled observations in stratum {h!r}")
        if h not in Yhat_pop_mean:
            raise ValueError(f"No predicted-only observations in stratum {h!r}")

        Wh = float(Ah) / total
        residual = Y[mask] - lam * Yhat[mask]
        p += Wh * (lam * np.asarray(Yhat_pop_mean[h], float) + residual.mean(axis=0))
        if nh > 1:
            Rh = np.atleast_2d(np.cov(residual, rowvar=False, ddof=1))
            Sigma += Wh**2 * Rh / nh

        nu = int(Yhat_pop_n.get(h, 0))
        if nu > 1 and h in Yhat_pop_cov:
            Ch = np.atleast_2d(np.asarray(Yhat_pop_cov[h], float))
            L = np.outer(lam, lam)
            Sigma += Wh**2 * L * Ch / nu
    return p, Sigma


def optimal_lam_multinomial_diag(
    Y,
    Yhat,
    strat,
    Nh,
    Yhat_pop_cov=None,
    Yhat_pop_n=None,
):
    """Per-class variance-minimising lambda vector for the vector PPI estimator.

    Tunes each class independently (the diagonal of the general matrix problem):
    ``lam_k = cov(Y_k, Yhat_k) / var(Yhat_k)`` accumulated across strata with
    area weights and the finite predicted-only variance, clipped to ``[0, 1]``.
    A class whose proxy is constant within every stratum returns ``lam_k = 0``.
    """
    Y = np.asarray(Y, float)
    Yhat = np.asarray(Yhat, float)
    strat = np.asarray(strat)
    K = Y.shape[1]
    Yhat_pop_cov = {} if Yhat_pop_cov is None else Yhat_pop_cov
    Yhat_pop_n = {} if Yhat_pop_n is None else Yhat_pop_n
    total = float(sum(Nh.values()))
    num = np.zeros(K)
    den = np.zeros(K)

    for h, Ah in Nh.items():
        mask = strat == h
        nh = int(mask.sum())
        if nh < 2:
            continue
        Wh2 = (float(Ah) / total) ** 2
        Yh = Y[mask]
        Ph = Yhat[mask]
        yb = Yh.mean(axis=0)
        pb = Ph.mean(axis=0)
        cov = ((Yh - yb) * (Ph - pb)).sum(axis=0) / (nh - 1)
        var = ((Ph - pb) ** 2).sum(axis=0) / (nh - 1)
        num += Wh2 * cov / nh
        den += Wh2 * var / nh
        nu = int(Yhat_pop_n.get(h, 0))
        if nu > 1 and h in Yhat_pop_cov:
            den += Wh2 * np.diag(np.atleast_2d(Yhat_pop_cov[h])) / nu

    out = np.zeros(K)
    good = den > 0
    out[good] = np.clip(num[good] / den[good], 0.0, 1.0)
    return out


def contrast_ci(p, Sigma, c, alpha=0.05):
    """Confidence interval for a linear contrast ``c^T p`` of a composition.

    Use for differences (``c = e_i - e_j``) or any weighted sum of class shares.
    The variance ``c^T Sigma c`` uses the full covariance, so negatively
    correlated shares give a tighter interval than treating them independently.
    """
    p = np.asarray(p, float)
    c = np.asarray(c, float)
    Sigma = np.asarray(Sigma, float)
    value = float(c @ p)
    var = float(c @ Sigma @ c)
    se = np.sqrt(max(0.0, var))
    z = _z(alpha)
    return value, se, (value - z * se, value + z * se)


def composition_ratio_ci(p, Sigma, num_idx, den_idx, alpha=0.05):
    """Delta-method CI for a share ``sum(p[num_idx]) / sum(p[den_idx])``.

    ``num_idx`` must be a subset of ``den_idx`` for a genuine share. The
    gradient of the ratio is formed and combined with the full covariance, so
    the shared sampling variation between numerator and denominator cancels the
    way it does in :func:`design_analysis.stratified_ratio` for the 2-class case.
    """
    p = np.asarray(p, float)
    Sigma = np.asarray(Sigma, float)
    K = len(p)
    num = np.zeros(K)
    den = np.zeros(K)
    num[list(num_idx)] = 1.0
    den[list(den_idx)] = 1.0
    Y = float(num @ p)
    X = float(den @ p)
    if X <= 0:
        raise ValueError("Ratio denominator is non-positive")
    ratio = Y / X
    grad = num / X - Y / X**2 * den
    var = float(grad @ Sigma @ grad)
    se = np.sqrt(max(0.0, var))
    z = _z(alpha)
    return ratio, se, (ratio - z * se, ratio + z * se)


def stratified_ppi_bootstrap(
    y,
    yhat,
    strat,
    Nh,
    yhat_unlabelled,
    strat_unlabelled,
    *,
    n_boot=2000,
    lam=None,
    seed=20260718,
):
    """Design-aware bootstrap draws for a stratified PPI area proportion.

    The labelled pairs ``(y, yhat)`` and predicted-only observations are
    resampled independently, with replacement, within each design stratum.
    Known population/area weights ``Nh`` remain fixed. If ``lam`` is ``None``,
    the variance-minimising scalar PPI++ lambda is re-estimated in every draw;
    otherwise the supplied fixed lambda is used.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    strat = np.asarray(strat)
    yhat_unlabelled = np.asarray(yhat_unlabelled, float)
    strat_unlabelled = np.asarray(strat_unlabelled)
    if not (len(y) == len(yhat) == len(strat)):
        raise ValueError("Labelled y, yhat, and strat must have equal length")
    if len(yhat_unlabelled) != len(strat_unlabelled):
        raise ValueError("Predicted-only yhat and strat must have equal length")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")

    total = float(sum(Nh.values()))
    strata_data = []
    for h, Ah in Nh.items():
        labelled_idx = np.flatnonzero(strat == h)
        unlabelled_idx = np.flatnonzero(strat_unlabelled == h)
        if len(labelled_idx) == 0:
            raise ValueError(f"No labelled observations in stratum {h!r}")
        if len(unlabelled_idx) == 0:
            raise ValueError(f"No predicted-only observations in stratum {h!r}")
        strata_data.append(
            (float(Ah) / total, y[labelled_idx], yhat[labelled_idx],
             yhat_unlabelled[unlabelled_idx])
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    lambdas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = []
        numerator = 0.0
        denominator = 0.0
        for Wh, yh, ph, pu in strata_data:
            li = rng.integers(0, len(yh), size=len(yh))
            ui = rng.integers(0, len(pu), size=len(pu))
            yb = yh[li]
            pl = ph[li]
            pup = pu[ui]
            sampled.append((Wh, yb, pl, pup))
            if lam is None and len(yb) > 1:
                numerator += Wh**2 * np.cov(yb, pl, ddof=1)[0, 1] / len(yb)
                denominator += Wh**2 * pl.var(ddof=1) / len(yb)
                if len(pup) > 1:
                    denominator += Wh**2 * pup.var(ddof=1) / len(pup)

        lambda_b = (
            float(np.clip(numerator / denominator, 0.0, 1.0))
            if lam is None and denominator > 0
            else (0.0 if lam is None else float(lam))
        )
        lambdas[b] = lambda_b
        draws[b] = sum(
            Wh * (yb.mean() + lambda_b * (pup.mean() - pl.mean()))
            for Wh, yb, pl, pup in sampled
        )
    return draws, lambdas
