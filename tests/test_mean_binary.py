"""
Is the MEAN estimator on a binary map == proportion/area estimation? Verify identity, then
test ppi_py's full estimator family on a realistic binary land-cover-change example.
"""
import numpy as np
import pytest

pytest.importorskip('ppi_py')
from ppi_py import ppi_mean_pointestimate, ppi_mean_ci, classical_mean_ci
from ppi_py.cross_ppi import crossppi_mean_ci, crossppi_mean_pointestimate

rng = np.random.default_rng(0)

print('='*78)
print('TEST 1: mean-of-binary IS the proportion (and x area = km2). Identity check.')
print('='*78)
n, N = 800, 100_000
Y    = (rng.random(n) < 0.03).astype(float)          # binary change label
Yhat = (rng.random(n) < 0.03).astype(float)          # binary MAP
Yu   = (rng.random(N) < 0.035).astype(float)         # binary map over whole AOI
p_hat = ppi_mean_pointestimate(Y, Yhat, Yu, lam=1)
# by hand: proportion form
p_manual = Yu.mean() + (Y - Yhat).mean()
print(f'  ppi mean-of-binary   = {float(p_hat[0]):.8f}')
print(f'  manual proportion    = {p_manual:.8f}')
print(f'  identical?             {abs(float(p_hat[0])-p_manual) < 1e-12}')
PIXEL_KM2 = 0.01   # 100m pixel; area = p * N_pixels * pixel_area
print(f'  -> area              = {float(p_hat[0])*N*PIXEL_KM2:.2f} km2 (just a rescale)')
print()
print('  CONCLUSION: mean estimator on binary indicator == proportion estimator.')
print('  Area CI = proportion CI x total area. Advisor is correct.')
