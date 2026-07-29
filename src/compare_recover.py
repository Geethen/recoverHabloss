"""
REAL EXAMPLE: RECOVER artificial->nature reversion.
Compare: pixel-count / naive / Olofsson stratified / Hajek / difference(PPI) / ratio estimator.
Mirrors what WP1/R/scripts/area_estimation.R does, plus the new methods.
"""
import geopandas as gpd, pandas as pd, numpy as np
from scipy.stats import norm
from project_paths import default_habloss_root, project_data_dir
Z=norm.ppf(0.975)

lab=gpd.read_file(default_habloss_root() / 'data/samples_habloss_recover_for_geethen.shp')
sd=pd.read_csv(project_data_dir('scratch') / 'recover_design.csv')
ar=pd.read_csv(project_data_dir('scratch') / 'recover_areas.csv')

rec=lab[lab.source=='recover'].merge(sd[['PLOTID','stratum']],on='PLOTID',how='left')
Nh=dict(zip(ar.stratumLab, ar.area_km2))
A=sum(Nh.values())
rec=rec[rec.stratum.isin(Nh)].copy()
rec['m']=rec.stratum                                  # mapped stratum (design)
rec['mapcls']=rec.stratum.str.split('_').str[1:].str.join('_')

# reference class, matching area_estimation.R's classify_reference:
#   artificial -> nature == the target ("built_loss" reverting to nature)
h=lambda x:'Nature' if str(x).startswith('Nature') else x
rec['a']=rec.lc_2018.map(h); rec['b']=rec.lc_2024.map(h)
rec['y_target']=((rec.a=='Artificial')&(rec.b=='Nature')).astype(float)   # artificial -> nature
rec['y_a2c']  =((rec.a=='Artificial')&(rec.b=='Cropland')).astype(float)  # artificial -> cropland

nh=rec.m.value_counts().to_dict()
rec['w']=[Nh[s]/nh[s] for s in rec.m]

print('='*94)
print('REAL EXAMPLE — RECOVER: area of ARTIFICIAL -> NATURE reversion (2018-2024), global')
print(f'  n labelled = {len(rec)}   strata = {len(nh)}   total area A = {A:,.0f} km2')
print(f'  n_h realised: min={min(nh.values())} med={int(np.median(list(nh.values())))} max={max(nh.values())} (designed 100)')
wv=rec.w.values
print(f'  design weights: {wv.min():,.1f} - {wv.max():,.1f} km2/pt  ({wv.max()/wv.min():,.0f}x spread)')
print('='*94)

def olofsson_prop(y, strat, Nh, A):
    """Olofsson SEa formula EXACTLY as in area_estimation.R: Wh^2 * (b*c/d), d=nh-1."""
    p=0.0; var=0.0
    for s,Ns in Nh.items():
        msk=strat==s; n=int(msk.sum())
        if n==0: continue
        W=Ns/A; ph=y[msk].mean(); p+=W*ph
        if n>1: var+=W**2*(ph*(1-ph)/(n-1))
    se=np.sqrt(var); return p,se,(p-Z*se,p+Z*se)

def strat_fpc(y, strat, Nh, A, npix=None):
    """Corrected: s^2/nh with FPC. Areas are continuous -> fpc negligible, keep for completeness."""
    p=0.0; var=0.0
    for s,Ns in Nh.items():
        msk=strat==s; n=int(msk.sum())
        if n==0: continue
        W=Ns/A; ph=y[msk].mean(); p+=W*ph
        if n>1:
            s2=y[msk].var(ddof=1); var+=W**2*(s2/n)
    se=np.sqrt(var); return p,se,(p-Z*se,p+Z*se)

def hajek(y,w):
    p=np.sum(w*y)/np.sum(w); n=len(y)
    r=w*(y-p); var=np.sum(r**2)/(np.sum(w)**2)*(n/(n-1)); se=np.sqrt(var)
    return p,se,(p-Z*se,p+Z*se)

def ratio_strat(y,x,strat,Nh):
    """Ratio estimator as implemented in area_estimation.R (combined ratio, stratified)."""
    Y=0;X=0; terms=[]
    for s,Ns in Nh.items():
        msk=strat==s; n=int(msk.sum())
        if n==0: continue
        Y+=Ns*y[msk].mean(); X+=Ns*x[msk].mean()
    R=Y/X
    var=0
    for s,Ns in Nh.items():
        msk=strat==s; n=int(msk.sum())
        if n<2: continue
        sy2=y[msk].var(ddof=1); sx2=x[msk].var(ddof=1)
        sxy=np.cov(y[msk],x[msk],ddof=1)[0,1]
        var+=Ns**2*(sy2+R**2*sx2-2*R*sxy)/n
    var/=X**2; se=np.sqrt(var)
    return R,se,(R-Z*se,R+Z*se),Y,X

y=rec.y_target.values; st=rec.m.values; w=rec.w.values

print('\n### METHOD COMPARISON: proportion & area of artificial -> nature ###\n')
rows=[]
# 1 naive
p=y.mean(); se=np.sqrt(p*(1-p)/len(y))
rows.append(('naive unweighted',p,se,(p-Z*se,p+Z*se)))
# 2 Olofsson (as in RECOVER script)
rows.append(('Olofsson (script)',)+olofsson_prop(y,st,Nh,A))
# 3 corrected stratified
rows.append(('stratified s2/nh',)+strat_fpc(y,st,Nh,A))
# 4 Hajek
rows.append(('Hajek weighted',)+hajek(y,w))

print(f"{'method':<22}{'prop %':>10}{'se %':>9}{'95% CI %':>22}{'area Mkm2':>12}{'CI width Mkm2':>15}")
print('-'*90)
for nm,p,se,ci in rows:
    print(f"{nm:<22}{p*100:>10.4f}{se*100:>9.4f}   [{ci[0]*100:>7.4f},{ci[1]*100:>7.4f}]{p*A/1e6:>12.4f}{(ci[1]-ci[0])*A/1e6:>15.4f}")

# 5 ratio estimator: art->crop / (art->crop + art->nature)  [as in the script]
print('\n### RATIO ESTIMATOR (as in area_estimation.R): art->cropland / (art->cropland + art->nature) ###')
yy=rec.y_a2c.values; xx=(rec.y_a2c.values+rec.y_target.values)
R,seR,ciR,Y,X=ratio_strat(yy,xx,st,Nh)
print(f'  numerator area   = {Y/1e6:.4f} Mkm2 (artificial->cropland)')
print(f'  denominator area = {X/1e6:.4f} Mkm2 (artificial->cropland + artificial->nature)')
print(f'  ratio            = {R*100:.2f}%  se={seR*100:.2f}  95% CI [{max(0,ciR[0]*100):.2f}, {min(100,ciR[1]*100):.2f}]')
print(f'  -> relative CI width: {(ciR[1]-ciR[0])*100:.2f} pp')
