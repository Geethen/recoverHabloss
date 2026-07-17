"""
Worked example: global-ish binary CHANGE map, rare class, known truth.
Compares: classical / stratified / PPI(lam=1) / PPI++ / cross-PPI, on BINARY vs PROBABILITY maps.
"""
import numpy as np
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci, classical_mean_ci
from ppi_py.cross_ppi import crossppi_mean_ci
from scipy import stats

def build(seed=0, N=60_000, prev=0.02, ua=0.70, pa=0.55):
    """Population with TRUE change prevalence `prev`; map has user's acc `ua`, producer's acc `pa`."""
    r = np.random.default_rng(seed)
    Y = (r.random(N) < prev).astype(float)                    # truth
    # map: detects pa of true change; false positives set so user's acc ~= ua
    Yhat = np.zeros(N)
    tp = (r.random(N) < pa) & (Y == 1); Yhat[tp] = 1
    n_tp = tp.sum(); n_fp = int(n_tp * (1 - ua) / ua)
    cand = np.where((Y == 0) & (Yhat == 0))[0]
    Yhat[r.choice(cand, size=min(n_fp, len(cand)), replace=False)] = 1
    # probability version of the same model (soft scores)
    P = np.clip(np.where(Y==1, r.beta(5,3,N), r.beta(2,8,N)), 0, 1)
    return Y, Yhat, P

Y, Yhat, P = build()
TRUE = Y.mean()
N = len(Y)
print('='*88)
print(f'WORKED EXAMPLE: N={N:,}  TRUE change prevalence = {TRUE:.5f}  ({TRUE*N*0.01:,.0f} km2 @100m px)')
ct = np.array([[((Y==1)&(Yhat==1)).sum(), ((Y==1)&(Yhat==0)).sum()],
               [((Y==0)&(Yhat==1)).sum(), ((Y==0)&(Yhat==0)).sum()]])
print(f"map: user's acc={ct[0,0]/(ct[0,0]+ct[1,0]):.3f}  producer's acc={ct[0,0]/(ct[0,0]+ct[0,1]):.3f}  map prevalence={Yhat.mean():.5f}")
print('='*88)

def run(n_lab=1000, reps=120, use_prob=False):
    """Simple random reference sample of size n_lab; measure coverage of each estimator."""
    tag = 'PROBABILITY map' if use_prob else 'BINARY map'
    M = P if use_prob else Yhat
    out = {k: {'e': [], 'c': 0, 'w': []} for k in
           ['classical', 'ppi_lam1', 'ppi_opt', 'cross_ppi']}
    r0 = np.random.default_rng(99)
    for i in range(reps):
        idx = r0.choice(N, size=n_lab, replace=False)
        unl = np.setdiff1d(np.arange(N), idx, assume_unique=False)
        ys, ms, mu = Y[idx], M[idx], M[unl]

        lo, hi = classical_mean_ci(ys, alpha=0.05)
        out['classical']['e'].append(ys.mean()); out['classical']['c'] += (lo <= TRUE <= hi); out['classical']['w'].append(hi-lo)

        p1 = float(np.atleast_1d(ppi_mean_pointestimate(ys, ms, mu, lam=1))[0])
        c1 = ppi_mean_ci(ys, ms, mu, alpha=0.05, lam=1)
        l1, h1 = float(np.atleast_1d(c1[0])[0]), float(np.atleast_1d(c1[1])[0])
        out['ppi_lam1']['e'].append(p1); out['ppi_lam1']['c'] += (l1 <= TRUE <= h1); out['ppi_lam1']['w'].append(h1-l1)

        p2 = float(np.atleast_1d(ppi_mean_pointestimate(ys, ms, mu))[0])
        c2 = ppi_mean_ci(ys, ms, mu, alpha=0.05)
        l2, h2 = float(np.atleast_1d(c2[0])[0]), float(np.atleast_1d(c2[1])[0])
        out['ppi_opt']['e'].append(p2); out['ppi_opt']['c'] += (l2 <= TRUE <= h2); out['ppi_opt']['w'].append(h2-l2)

        K = 5
        mu_k = np.column_stack([mu]*K)
        c3 = crossppi_mean_ci(ys, ms, mu_k, alpha=0.05)
        l3, h3 = float(np.atleast_1d(c3[0])[0]), float(np.atleast_1d(c3[1])[0])
        out['cross_ppi']['e'].append(float(np.atleast_1d(ppi_mean_pointestimate(ys, ms, mu_k.mean(1), lam=1))[0]))
        out['cross_ppi']['c'] += (l3 <= TRUE <= h3); out['cross_ppi']['w'].append(h3-l3)

    print(f'\n--- {tag}, n_labelled={n_lab}, {reps} reps, TRUE={TRUE:.5f} ---')
    print(f"{'estimator':<12}{'mean':>10}{'bias':>11}{'coverage':>10}{'CI width':>11}{'vs classical':>14}")
    print('-'*68)
    base = np.mean(out['classical']['w'])
    for k, v in out.items():
        m = np.mean(v['e']); w = np.mean(v['w'])
        print(f"{k:<12}{m:>10.5f}{m-TRUE:>+11.5f}{v['c']/reps:>9.1%}{w:>11.5f}{w/base:>13.2f}x")
    return out

run(use_prob=False)
run(use_prob=True)
