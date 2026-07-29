"""Reconstruct design weights for HABLOSS/RECOVER from GEE strata areas + cleaned sample files."""
from pathlib import Path
import geopandas as gpd, pandas as pd, numpy as np, glob, os
from project_paths import default_habloss_root, project_data_dir

R = default_habloss_root()
OUTPUT = project_data_dir('scratch')
OUTPUT.mkdir(parents=True, exist_ok=True)
pd.set_option('display.width', 250)

# ---- 1. stratum lookup: biome x 10 loss classes, stratum = num + (BIOME_NUM2-1)*10
loss = pd.DataFrame({
    'stratLoss': ["stable_nature","stable_built","stable_crop","buffer_forest","buffer_crop",
                  "buffer_built","loss_forest","loss_crop","loss_built_crop","loss_built_nature"],
    'num': range(1, 11)})
bl = pd.read_csv(R / 'data/biomeLookup.csv')
import re
bl['abbr'] = bl.BIOME_NAME2.apply(lambda s: ''.join(w[:2] for w in re.findall(r'[A-Za-z]+', s)))
lk = bl.merge(loss, how='cross')
lk['stratumLab'] = lk.abbr + '_' + lk.stratLoss
lk['stratum'] = lk.num + (lk.BIOME_NUM2 - 1) * 10
lk = lk[['stratumLab','stratum','stratLoss','BIOME_NUM2','abbr']]

# ---- 2. stratum areas N_h from GEE (m^2 -> km^2)
def read_areas(d):
    fs = glob.glob(str(R / 'data' / 'from_gee' / d / '*.csv'))
    if not fs: return None
    dfs=[]
    for f in fs:
        try:
            df=pd.read_csv(f)
            if len(df) and 'stratum' in df.columns: dfs.append(df)
        except Exception: pass
    if not dfs: return None
    a = pd.concat(dfs, ignore_index=True)
    a['stratum'] = a.stratum.astype(float)
    return a.groupby('stratum', as_index=False).area.sum().assign(area_km2=lambda x: x.area/1e6)

areas_nb   = read_areas('areas_biomes')
areas_wb   = read_areas('areas_biomes_with_buffer')
areas_lw   = read_areas('areas_landwater_with_buffer_500')
print('=== strata area sources ===')
for nm, a in [('areas_biomes',areas_nb),('areas_biomes_with_buffer',areas_wb),('areas_landwater_500',areas_lw)]:
    if a is None: print(f'  {nm}: MISSING'); continue
    print(f'  {nm}: {len(a)} strata, total {a.area_km2.sum():,.0f} km2, stratum ids {a.stratum.min():.0f}-{a.stratum.max():.0f}')

# ---- 3. sample -> stratum membership from cleaned geojsons
cl = []
for f, batch in [('samples_biomes_cleaned.geojson','biomes_main'),
                 ('samples_biomes_extra_cleaned.geojson','biomes_extra'),
                 ('samples_landwater_cleaned.geojson','landwater_main'),
                 ('samples_landwater_extra_cleaned.geojson','landwater_extra')]:
    g = gpd.read_file(R / 'data' / 'from_gee' / f)[['PLOTID','stratum']]
    g['batch'] = batch
    cl.append(g)
cl = pd.concat(cl, ignore_index=True).rename(columns={'stratum':'stratumLab'})
print(f'\n=== cleaned sample membership: {len(cl)} rows, {cl.PLOTID.nunique()} unique PLOTID ===')
print(cl.batch.value_counts().to_string())

# ---- 4. join to the labelled analysis file
lab = gpd.read_file(R / 'data/samples_habloss_recover_for_geethen.shp')
m = lab.merge(cl, on='PLOTID', how='left')
print(f'\n=== JOIN labelled({len(lab)}) -> strata ===')
print('  matched  :', m.stratumLab.notna().sum())
print('  UNMATCHED:', m.stratumLab.isna().sum())
print('\n  unmatched by source:')
print(m[m.stratumLab.isna()].source.value_counts().to_string())
print('\n  matched by source x batch:')
print(pd.crosstab(m.source, m.batch.fillna('<none>')).to_string())
m.to_file(OUTPUT / 'joined.gpkg', driver='GPKG')
lk.to_csv(OUTPUT / 'lookup.csv', index=False)
for nm,a in [('nb',areas_nb),('wb',areas_wb),('lw',areas_lw)]:
    if a is not None: a.to_csv(OUTPUT / f'areas_{nm}.csv', index=False)
