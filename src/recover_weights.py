"""Reconstruct RECOVER design weights (READ-ONLY on P:). 5 map-strata x biomes, slice_head(100)."""
import geopandas as gpd, pandas as pd, numpy as np, glob, re
from project_paths import default_recover_root, project_data_dir

RC = default_recover_root()
OUTPUT = project_data_dir('scratch')
OUTPUT.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width',250)

# lookup: stratum = stratumMap + (BIOME_NUM2-1)*5
bl=pd.read_csv(RC / 'data/biomeLookup.csv')
bl['abbr']=bl.BIOME_NAME2.apply(lambda s:''.join(w[:2] for w in re.findall(r'[A-Za-z]+',s)))
rl=pd.DataFrame({'stratumMap':[1,2,3,4,5],
                 'stratumMapLab':['stable_stable','crop_buffer','built_buffer','crop_loss','built_loss']})
lk=bl.merge(rl,how='cross')
lk['stratumLab']=lk.abbr+'_'+lk.stratumMapLab
lk['stratum']=lk.stratumMap+(lk.BIOME_NUM2-1)*5
print(f'RECOVER design: {len(lk)} strata ({bl.BIOME_NUM2.nunique()} biomes x 5 map classes)')

# areas
fs=glob.glob(str(RC / 'data' / 'from_gee' / 'areas_recover' / '*.csv'))
dfs=[]
for f in fs:
    try:
        d=pd.read_csv(f)
        if len(d) and 'stratum' in d.columns: dfs.append(d)
    except Exception: pass
ar=pd.concat(dfs,ignore_index=True)
ar['stratum']=ar.stratum.astype(int)
ar=ar.groupby('stratum',as_index=False).area.sum()
ar['area_km2']=ar.area/1e6
ar=ar.merge(lk[['stratum','stratumLab','stratumMapLab','abbr']],on='stratum',how='left')
print(f'areas: {len(ar)} strata, total {ar.area_km2.sum():,.0f} km2')

# the AS-DESIGNED sample (what was sent to labellers)
sd=gpd.read_file(RC / 'data/for_gee/samples_recover_v2.shp')
print(f'\nas-designed sample: {len(sd)} rows, cols={list(sd.columns)}')
print(sd.head(2).to_string())
sd.to_csv(OUTPUT / 'recover_design.csv',index=False)
ar.to_csv(OUTPUT / 'recover_areas.csv',index=False)
lk.to_csv(OUTPUT / 'recover_lookup.csv',index=False)
