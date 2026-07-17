"""
MULTICLASS: is a 'proportion estimator' better than one-vs-rest binary means?
Estimand: the full vector of transition-class area proportions (must sum to 1).
"""
import numpy as np
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci, classical_mean_ci

rng = np.random.default_rng(3)
N = 60_000
# 6 transition classes with wildly different prevalence (mimics your matrix)
prev = np.array([0.60, 0.20, 0.10, 0.055, 0.04, 0.005])
K = len(prev)
cls = rng.choice(K, size=N, p=prev)
Y1h = np.eye(K)[cls]                      # one-hot truth: mean of each column = proportion
# map: confusion — correct w.p. acc, else random other class
acc = 0.75
pred = np.where(rng.random(N) < acc, cls, rng.integers(0, K, N))
M1h = np.eye(K)[pred]
TRUE = Y1h.mean(0)

print('='*86)
print('MULTICLASS TRANSITION AREAS — vector estimand, must sum to 1')
print(f'TRUE proportions: {np.round(TRUE,5)}   sum={TRUE.sum():.4f}')
print('='*86)

def run(n_lab=1200, reps=150):
    res = {k: {'e': [], 'c': np.zeros(K), 'w': []} for k in ['classical_ovr', 'ppi_ovr_joint', 'ppi_ovr_percls']}
    r0 = np.random.default_rng(11)
    for i in range(reps):
        idx = r0.choice(N, size=n_lab, replace=False)
        unl = np.setdiff1d(np.arange(N), idx)
        ys, ms, mu = Y1h[idx], M1h[idx], M1h[unl]

        # 1. classical one-vs-rest
        est = ys.mean(0); res['classical_ovr']['e'].append(est)
        wid = []
        for k in range(K):
            lo, hi = classical_mean_ci(ys[:, k], alpha=0.05)
            res['classical_ovr']['c'][k] += (lo <= TRUE[k] <= hi); wid.append(hi-lo)
        res['classical_ovr']['w'].append(wid)

        # 2. PPI vectorised: ONE lam optimised over the whole vector (lam_optim_mode='overall')
        p = ppi_mean_pointestimate(ys, ms, mu, lam_optim_mode='overall')
        lo, hi = ppi_mean_ci(ys, ms, mu, alpha=0.05, lam_optim_mode='overall')
        res['ppi_ovr_joint']['e'].append(np.atleast_1d(p))
        res['ppi_ovr_joint']['c'] += ((np.atleast_1d(lo) <= TRUE) & (TRUE <= np.atleast_1d(hi)))
        res['ppi_ovr_joint']['w'].append(np.atleast_1d(hi) - np.atleast_1d(lo))

        # 3. PPI per-class lam (each class tunes its own lam) -- what I argued you need
        ests = np.zeros(K); wid = []
        for k in range(K):
            pk = float(np.atleast_1d(ppi_mean_pointestimate(ys[:, k], ms[:, k], mu[:, k]))[0])
            ck = ppi_mean_ci(ys[:, k], ms[:, k], mu[:, k], alpha=0.05)
            lk, hk = float(np.atleast_1d(ck[0])[0]), float(np.atleast_1d(ck[1])[0])
            ests[k] = pk; res['ppi_ovr_percls']['c'][k] += (lk <= TRUE[k] <= hk); wid.append(hk-lk)
        res['ppi_ovr_percls']['e'].append(ests); res['ppi_ovr_percls']['w'].append(wid)

    print(f'\nn_labelled={n_lab}, {reps} reps\n')
    for name, v in res.items():
        E = np.array(v['e']); W = np.array(v['w']); C = v['c']/reps
        print(f'--- {name} ---')
        print(f'  {"class":<7}{"TRUE":>9}{"mean":>10}{"bias":>10}{"cover":>8}{"width":>10}')
        for k in range(K):
            print(f'  {k:<7}{TRUE[k]:>9.5f}{E[:,k].mean():>10.5f}{E[:,k].mean()-TRUE[k]:>+10.5f}{C[k]:>7.0%}{W[:,k].mean():>10.5f}')
        print(f'  SUM of point estimates: {E.mean(0).sum():.5f}  <- coherence check (should be 1.0)')
        print(f'  mean width across classes: {W.mean():.5f}')
        print()
run()
