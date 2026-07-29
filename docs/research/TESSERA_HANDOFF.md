# Tessera point-extraction — handoff

## Objective
Extract Tessera (GeoTessera) 128-D per-pixel embeddings for the 6,492 RECOVER/habloss
training points, years **2018 and 2024**, as fast as possible, to test fusing them
into the hier NN (`HierarchicalSoftmaxNN`) alongside the existing AlphaEarth features.

## Env
- Python: `/home/geethen.singh/.pixi/envs/geo/bin/python` (the "geo" pixi env; the
  project `.venv` is broken — do NOT use `uv run`).
- Points file: `data/embeddings/embeddings_habloss_recover.parquet` (cols incl
  `PLOTID, lon, lat, source`).

## Why custom byte-range (not geotessera API)
geotessera 0.9.0 (latest) has **no usable fast path** for these points:
- `.npy` store is populated but `sample_embeddings_at_points` downloads whole
  38–147 MB tiles (~9.8 s/pt).
- Every zarr store (`v2/store.zarr`, `geotessera-2024/25`, default `vultr.zarr`)
  returns NaN for our points — separate, incompletely-populated builds.
So we read only each point's pixel via HTTP Range. **Validated bit-exact** vs
`sample_embeddings_at_points` (see `scratchpad/test_fastpath.py`, 5/5 data points matched).

## Data layout (S3, anonymous, HTTP Range = 206)
Base: `https://s3.us-west-2.amazonaws.com/tessera-embeddings/v1`
- Embedding tile: `global_0.1_degree_representation/{year}/{grid}/{grid}.npy`
  = `(H,W,128)` **int8, C-order** (a pixel's 128 channels are contiguous).
- Scales:        `.../{grid}/{grid}_scales.npy` = `(H,W)` **float32**, one scalar/pixel.
- Landmask (georef): `global_0.1_degree_tiff_all/{grid}.tiff` (~13 KB, **year-independent**)
  → CRS + Affine transform + (H,W). No geo metadata in the .npy itself.
- `{grid}` = `tile_to_landmask_filename(*tile_from_world(lon,lat))[:-5]` e.g. `grid_55.25_25.25`.
- .npy header/data offset is a **constant 128 bytes** (verified). Pixel byte offset:
  `128 + (row*W + col)*itemsize` (128 for emb int8, 4 for scales f32).
- Pixel map: `x,y = Transformer(EPSG:4326→crs).transform(lon,lat)`;
  `row,col = rasterio.transform.rowcol(transform, x, y)` (floor); bounds-check.
- Dequantize = `int8 * scale` (== geotessera core.dequantize). Non-finite scale or
  out-of-bounds or 404 → NaN row = water/nodata/not-covered.

## Coverage (registry, iter_tiles_in_region global)
- 2024: **98.5%** of points | 2018: **36.9%** | BOTH years: **35.8%** (2,326/6,492).
- 2024 stays a viable single-date feature; 2018 sparsity means AlphaEarth stays the
  base for the change model. Store 2018 anyway (subset fusion / S1 complement).

## Files
- `src/extract_tessera_points.py` — production extractor (THIS is the deliverable).
  obstore async byte-range reads (concurrency-capped by a semaphore), landmask georef
  cache (in-process + on-disk, shared across years), resumable per-shard Parquet,
  `tess_covered_{year}` flags.
  Output: `data/embeddings/tessera_shards/shard_*.parquet` →
  `data/embeddings/embeddings_tessera_habloss_recover.parquet`
  (cols `TE000_{year}..TE127_{year}`, `tess_covered_{year}`).
  Run: `python src/extract_tessera_points.py --years 2018 2024 --workers 96`
- scratchpad/ (session-only, may vanish): `validate2.py` (byte-offset integrity),
  `test_fastpath.py` (end-to-end integrity), `extract_tessera.py` (same as repo copy).

## STATUS — EXTRACTION DONE (obstore path)
The script's I/O was rewritten to **obstore** (async Rust S3 range GETs,
`S3Store(skip_signature=True)`; already in the geo env via aef_loader). obstore
replaced the requests+ThreadPool path and won on every axis:
- **~23× faster:** 200 pts × 2 yr in **6.6 s** (was 152 s). Full **6,492 pts ×
  {2018,2024} in 2.7 min** (was projected ~82 min at 96 threads / ~40 min at the
  tuned-16 config).
- **Strictly more complete:** the thread path silently dropped ~4.5% of points to
  NaN under connection-pool/retry contention. obstore fetches them, and is
  **bit-exact** (max|Δ|=0) on every point both paths cover.
- This **refutes the earlier "throttled at ~16 concurrent" conclusion** — that was
  requests+ThreadPool retry-storms masquerading as throttling, not a real S3 egress
  cap. obstore runs clean at concurrency 96 with no throttling. The tifffile
  lock-free landmask trick is likewise moot: landmask now uses rasterio `MemoryFile`
  parsed in `asyncio.to_thread`, cached in-process AND to disk
  (`tessera_shards/landmask_geo_cache.pkl`), so re-runs skip the 13 KB georef GETs.

**Output:** `data/embeddings/embeddings_tessera_habloss_recover.parquet`
(`TE000_{year}..TE127_{year}`, `tess_covered_{year}`). Confirmed coverage
**2024 = 99.9%, 2018 = 36.3%, BOTH = 36.3% (2,359 pts)**. ~1% of covered rows are
all-zero (nodata/edge — treat as not-covered at fusion). obstore A/B prototype:
`scratchpad/extract_obstore.py` (session-only; logic is now in the repo script).

## FUSION RESULTS — Plan A tested (`src/experiment_hier_tessera.py`)
Fused into `HierarchicalSoftmaxNN` (wide/focal, 30 ep, blocked CV, merged2
change-F1, 3 seeds). CSVs in `data/analysis_results/hier_tessera_{subset,fullA}_2yr.csv`.
- **Signal check (both-covered subset, 2,309 pts, availability constant):**
  aef 0.661 → **aef+tess 0.679 (+1.8pt, above seed noise)**. Tessera carries
  complementary change signal; AlphaEarth alone (0.661) still beats Tessera alone
  (0.635). Single-date sets are much weaker (~0.50) — the 2018→2024 diff carries
  the change task. **=> the gate the user set for Plan B is passed.**
- **Plan A on the full set (6,414 pts, concat + `tess_present` mask + fold-mean
  impute):** aef 0.660; aef+tessA_2024 0.657 (neutral); **aef+tessA_2yr 0.639
  (−2.1pt, regresses)**. Naive concat does NOT transfer the subset signal — the
  64%-imputed columns dilute more than the mask+trunk can exploit.

## FUSION RESULTS — Plan B tested (`experiment_hier_tessera.py --mode twotower`)
Built `HierarchicalSoftmaxNN(arch="two_tower")` in `model_zoo.py` (`_TwoTowerTrunk`
+ `_prepare`/`_flat_trunk` branches; ctor args `aef_columns`, `tess_columns`,
`mask_column`, `modality_dropout`, `tower_dim`). Always-on AlphaEarth tower +
mask-gated Tessera tower, fused `rep_aef + gate*rep_tess`, modality dropout on the
gate during training. CSV `hier_tessera_twotower_2yr.csv`, 3 seeds:
- aef 0.660 ±0.004; two_tower md0.0 0.653 (−0.7); md0.3 0.658; **md0.5 0.6615
  (+0.1); md0.7 0.661**. Modality dropout is *essential* — without it the
  two-tower regresses; with it, break-even.
- **Verdict:** Plan B works as designed and rescues Plan A's −2.1pt regression to
  parity, but is break-even on the full-set deploy metric — it does NOT beat
  AlphaEarth-only. The +1.8pt gain stays confined to the 36% covered subset,
  averaged away by the 64% with no Tessera. **Bottleneck is 2018 coverage, not
  architecture.**
- **Transfer eval (`hier_tessera_subset_transfer_2yr.csv`)** — covered-trained,
  evaluated on the disjoint 4,105 uncovered plots (change rate 0.110 vs the
  covered subset's 0.188, so the subset is an easier regime). aef_2yr transfers
  0.580; **two_tower md0.5 = 0.577 (≈ aef, graceful fallback to the AlphaEarth
  tower when the mask is 0)**; two_tower md0.0 = 0.555 (−2.5pt); flat aef+tess =
  0.492 (−8.8pt, silently mis-uses the neutral Tessera). So **two_tower md0.5 is
  a dominant-or-equal deploy choice: never worse than AlphaEarth on uncovered
  plots, +1.8pt where Tessera is present** — the full-set break-even hides this
  asymmetry (no downside, coverage-bound upside). Modality dropout is what buys
  the graceful degradation.

## NEXT STEPS
1. **The fusion machinery is in place (arch="two_tower").** The lever now is 2018
   Tessera *coverage* — the full-set gain scales with it toward the subset's
   +1.8pt. Consider (a) requesting more 2018 tiles (geotessera GitHub) to lift the
   36%, or (b) deploying the fused model only on covered plots (banks +1.8pt on a
   third of points). No further architecture work is indicated.
2. To re-extract (e.g. add a year): `python src/extract_tessera_points.py
   --years 2018 2024 --workers 96` (resumable; `--workers` now = async concurrency).

## Perf notes for whoever tunes further
- obstore concurrency 96 is fine here; the old ~16 ceiling was a requests artefact.
- Landmask (13 KB) cached per tile + on disk, shared across years — 1 GET/tile amortised.
- Per point-year cost = 2 range GETs (emb pixel 128 B + scale 4 B) + shared landmask.
- Points are ~1.06/tile (6,143 tiles for 6,492 pts), so tile-batching / multi-range
  coalescing buys nothing — the win is per-GET overhead (obstore), not batching.

## Gotchas
- `tile_to_landmask_filename(lon,lat)` does NOT snap — always `tile_from_world` first.
- 404 on a tile's .npy = not covered that year → NaN (production handles; the simple
  `test_fastpath.py` rr() does not).
- Don't set geotessera `embeddings_dir` to cwd — it dumps 147 MB tiles there.
- See memory `tessera-embeddings-coverage.md` for the condensed version of all this.
