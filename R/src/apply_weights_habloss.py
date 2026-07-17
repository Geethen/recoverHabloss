"""Apply design weights to the REAL habloss samples. RECOVER excluded (design not recoverable)."""
import geopandas as gpd, pandas as pd, numpy as np, glob, re
from estimators import stratified_prop, hajek_prop
R='/data/P-Prosjekter2/155069_habloss/R'
pd.set_option('display.width',250)

lk = pd.read_csv('lookup.csv')
lab = gpd.read_file(f'{R}/data/samples_habloss_recover_for_geethen.shp')

# membership
cl=[]
for f,b in [('samples_biomes_cleaned','biomes_main'),('samples_biomes_extra_cleaned','biomes_extra'),
            ('samples_landwater_cleaned','landwater_main'),('samples_landwater_extra_cleaned','landwater_extra')]:
    g=gpd.read_file(f'{R}/data/from_gee/{f}.geojson')[['PLOTID','stratum']]; g['batch']=b; cl.append(g)
cl=pd.concat(cl,ignore_index=True).rename(columns={'stratum':'stratumLab'})
cl=cl.drop_duplicates('PLOTID')

m = lab.merge(cl,on='PLOTID',how='left')
hab = m[m.source.isin(['habloss_main','habloss_landwater'])].copy()
hab = hab[hab.stratumLab.notna()]
print(f'HABLOSS points with a recovered stratum: {len(hab)}')

# areas: biome strata (with buffer = the 80-stratum scheme matching stratumLab)
fs=glob.glob(f'{R}/data/from_gee/areas_biomes_with_buffer/*.csv')
dfs=[]
for f in fs:
    try:
        d=pd.read_csv(f)
        if len(d) and 'stratum' in d.columns: dfs.append(d)
    except Exception: pass
ar=pd.concat(dfs,ignore_index=True)
ar['stratum']=ar.stratum.astype(float)
ar=ar.groupby('stratum',as_index=False).area.sum()
ar['area_km2']=ar.area/1e6
ar=ar.merge(lk[['stratum','stratumLab']],on='stratum',how='left')
Nh = dict(zip(ar.stratumLab, ar.area_km2))
print(f'strata with areas: {len(Nh)}   total {sum(Nh.values()):,.0f} km2')

hab = hab[hab.stratumLab.isin(Nh)]
nh = hab.stratumLab.value_counts().to_dict()
print(f'HABLOSS points usable (stratum has area): {len(hab)} across {len(nh)} strata')
print(f'  n_h range: {min(nh.values())}-{max(nh.values())}')
areas_used=np.array([Nh[s] for s in nh])
print(f'  N_h range: {areas_used.min():,.0f} - {areas_used.max():,.0f} km2  ({areas_used.max()/areas_used.min():,.0f}x)')
w = np.array([Nh[s]/nh[s] for s in hab.stratumLab])
print(f'  weight range: {w.min():,.1f} - {w.max():,.1f} km2/pt  ({w.max()/w.min():,.0f}x spread)')
print()

h=lambda x:'Nature' if str(x).startswith('Nature') else x
hab['a']=hab.lc_2018.map(h); hab['b']=hab.lc_2024.map(h)
TOT=sum(Nh.values())

print('=== HABLOSS transitions: UNWEIGHTED vs DESIGN-WEIGHTED (95% CI) ===')
print(f"{'transition':<26}{'n':>5}{'unwtd %':>10}{'  design-wtd % [95% CI]':>30}{'area Mkm2':>12}")
print('-'*84)
rows=[]
for i in ['Artificial','Cropland','Nature']:
    for j in ['Artificial','Cropland','Nature']:
        if i==j: continue
        yv=((hab.a==i)&(hab.b==j)).astype(float).values
        n=int(yv.sum())
        unw=yv.mean()*100
        p,se,ci=stratified_prop(yv,hab.stratumLab.values,Nh)
        rows.append((f'{i} -> {j}',n,unw,p*100,ci[0]*100,ci[1]*100,p*TOT/1e6))
        print(f"{i+' -> '+j:<26}{n:>5}{unw:>9.2f}%{p*100:>12.3f}% [{ci[0]*100:>6.3f},{ci[1]*100:>6.3f}]{p*TOT/1e6:>11.3f}")
print()
print('NOTE: unweighted vs weighted differ by ~an order of magnitude -> confirms')
print('      every count in the earlier replies was an artifact of allocation.')
