"""Validate my difference_prop against the reference ppi_py implementation."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from estimators import difference_prop, optimal_lam
pytest.importorskip('ppi_py')
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci

rng=np.random.default_rng(7)
n, N = 500, 50_000
Y   = (rng.random(n)<0.2).astype(float)
Yhat= np.clip(0.2+rng.normal(0,0.15,n),0,1)
Yu  = np.clip(0.2+rng.normal(0,0.15,N),0,1)

print('=== unweighted, lam=1: mine vs ppi_py ===')
mine = difference_prop(Y,Yhat,Yu.mean(),lam=1.0)
ref  = ppi_mean_pointestimate(Y,Yhat,Yu,lam=1)
ref_ci = ppi_mean_ci(Y,Yhat,Yu,alpha=0.05,lam=1)
print(f'  mine point = {mine[0]:.8f}')
print(f'  ppi_py     = {float(ref[0]):.8f}')
print(f'  DIFF       = {abs(mine[0]-float(ref[0])):.2e}')
print(f'  mine CI    = ({mine[2][0]:.6f}, {mine[2][1]:.6f})')
print(f'  ppi_py CI  = ({float(ref_ci[0][0]):.6f}, {float(ref_ci[1][0]):.6f})')
print()

print('=== with design weights, lam=1: mine vs ppi_py(w=) ===')
w = rng.gamma(2,50,n)   # wildly unequal weights
mine_w = difference_prop(Y,Yhat,Yu.mean(),w=w,lam=1.0)
ref_w  = ppi_mean_pointestimate(Y,Yhat,Yu,lam=1,w=w,w_unlabeled=np.ones(N))
print(f'  mine point = {mine_w[0]:.8f}')
print(f'  ppi_py     = {float(ref_w[0]):.8f}')
print(f'  DIFF       = {abs(mine_w[0]-float(ref_w[0])):.2e}')
print()

print('=== lam* power tuning: mine vs ppi_py ===')
lam_mine = optimal_lam(Y,Yhat)
p_ppipy_auto = ppi_mean_pointestimate(Y,Yhat,Yu)   # lam=None -> auto
p_mine_auto  = difference_prop(Y,Yhat,Yu.mean(),lam=lam_mine)
print(f'  my lam*        = {lam_mine:.6f}')
print(f'  my auto point  = {p_mine_auto[0]:.8f}')
print(f'  ppi_py auto    = {float(p_ppipy_auto[0]):.8f}')
print()
print('=== sanity: lam=0 must recover the classical sample mean ===')
p0 = difference_prop(Y,Yhat,Yu.mean(),lam=0.0)
print(f'  lam=0 point = {p0[0]:.8f}   sample mean = {Y.mean():.8f}   DIFF={abs(p0[0]-Y.mean()):.2e}')
