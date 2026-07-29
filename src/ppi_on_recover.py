"""
Can PPI help on the RECOVER example? The map stratum IS a predictor (built_loss => likely reversion).
Use map class as the 'model prediction' -- this is the auxiliary actually available today.
"""
import geopandas as gpd, pandas as pd, numpy as np
from scipy.stats import norm
from project_paths import default_habloss_root, project_data_dir
Z=norm.ppf(0.975)
lab=gpd.read_file(default_habloss_root() / 'data/samples_habloss_recover_for_geethen.shp')
sd=pd.read_csv(project_data_dir('scratch') / 'recover_design.csv'); ar=pd.read_csv(project_data_dir('scratch') / 'recover_areas.csv')
rec=lab[lab.source=='recover'].merge(sd[['PLOTID','stratum']],on='PLOTID',how='left')
Nh=dict(zip(ar.stratumLab,ar.area_km2)); A=sum(Nh.values())
rec=rec[rec.stratum.isin(Nh)].copy()
h=lambda x:'Nature' if str(x).startswith('Nature') else x
rec['a']=rec.lc_2018.map(h); rec['b']=rec.lc_2024.map(h)
rec['y']=((rec.a=='Artificial')&(rec.b=='Nature')).astype(float)
rec['mapcls']=rec.stratum.str.split('_').str[1:].str.join('_')
nh=rec.stratum.value_counts().to_dict()
rec['w']=[Nh[s]/nh[s] for s in rec.stratum]

# The MAP predicts reversion where mapcls == 'built_loss'. Yhat = 1 there.
rec['yhat']=(rec.mapcls=='built_loss').astype(float)

# population mean of Yhat = area-share of built_loss strata (KNOWN from pixel counts)
ar['mapcls']=ar.stratumLab.str.split('_').str[1:].str.join('_')
yhat_pop = ar.loc[ar.mapcls=='built_loss','area_km2'].sum()/A
print('='*92)
print('PPI ON THE REAL RECOVER DATA — map class "built_loss" as the auxiliary predictor')
print(f'  population mean of Yhat (area share of built_loss strata) = {yhat_pop:.8f}')
print(f'  => pixel-count area of mapped built_loss = {yhat_pop*A/1e6:.4f} Mkm2')
print('='*92)

# accuracy of that map as a predictor of true reversion
tp=((rec.y==1)&(rec.yhat==1)).sum(); fp=((rec.y==0)&(rec.yhat==1)).sum()
fn=((rec.y==1)&(rec.yhat==0)).sum()
print(f'\n  map-as-predictor (unweighted counts): TP={tp} FP={fp} FN={fn}')
print(f"  user's acc = {tp/(tp+fp):.3f}   producer's acc = {tp/(tp+fn):.3f}")

y=rec.y.values; yh=rec.yhat.values; w=rec.w.values; st=rec.stratum.values

def strat_prop(y,strat,Nh,A):
    p=0.;v=0.
    for s,Ns in Nh.items():
        m=strat==s; n=int(m.sum())
        if n==0: continue
        W=Ns/A; ph=y[m].mean(); p+=W*ph
        if n>1: v+=W**2*(ph*(1-ph)/(n-1))
    return p,np.sqrt(v)

def ppi_strat(y,yh,strat,Nh,A,yhat_pop,lam=1.0):
    """Difference/PPI estimator computed stratum-wise (design-consistent)."""
    d=y-lam*yh
    p_d=0.;v=0.
    for s,Ns in Nh.items():
        m=strat==s; n=int(m.sum())
        if n==0: continue
        W=Ns/A; p_d+=W*d[m].mean()
        if n>1: v+=W**2*(d[m].var(ddof=1)/n)
    p=lam*yhat_pop+p_d
    return p,np.sqrt(v)

def opt_lam_strat(y,yh,strat,Nh,A):
    """lam* minimising the stratified variance of (y - lam*yh)."""
    num=0.;den=0.
    for s,Ns in Nh.items():
        m=strat==s; n=int(m.sum())
        if n<2: continue
        W=Ns/A
        num+=W**2*np.cov(y[m],yh[m],ddof=1)[0,1]/n
        den+=W**2*yh[m].var(ddof=1)/n
    return 0.0 if den<=0 else float(np.clip(num/den,0,1))

p0,se0=strat_prop(y,st,Nh,A)
p1,se1=ppi_strat(y,yh,st,Nh,A,yhat_pop,lam=1.0)
lam=opt_lam_strat(y,yh,st,Nh,A)
p2,se2=ppi_strat(y,yh,st,Nh,A,yhat_pop,lam=lam)

print(f'\n{"method":<26}{"prop %":>11}{"se %":>10}{"95% CI %":>24}{"area Mkm2":>12}{"width":>10}')
print('-'*93)
for nm,p,se in [('stratified (Olofsson)',p0,se0),('PPI lam=1',p1,se1),(f'PPI++ lam*={lam:.3f}',p2,se2)]:
    print(f'{nm:<26}{p*100:>11.5f}{se*100:>10.5f}   [{(p-Z*se)*100:>8.5f},{(p+Z*se)*100:>8.5f}]{p*A/1e6:>12.5f}{2*Z*se*A/1e6:>10.5f}')
print(f'\n  lam* = {lam:.4f}')
print(f'  PPI++ CI width vs stratified: {se2/se0:.3f}x')
