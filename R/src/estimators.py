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
