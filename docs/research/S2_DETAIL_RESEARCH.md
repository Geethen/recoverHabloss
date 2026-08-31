# Sentinel-2 VNIR as the detail modality — research backlog

**Premise.** The AlphaEarth+Tessera search (`TWOTOWER_RESEARCH.md`, ~45 ideas)
ended on one sentence: *the bottleneck is 2018 Tessera coverage, not the
architecture.* Sections A–E were all attempts to spend a +1.8pt gain that only
fires on 35.8% of plots, and every attempt to make the fusion smarter failed.

Raw Sentinel-2 L2A has no such hole. Measured on the first 240 plots extracted:

| modality | both-endpoint coverage | native res | source |
| --- | --- | --- | --- |
| AlphaEarth | 100% | 10 m (smooth, context) | GEE annual embedding |
| Tessera | **35.8%** | 10 m (S1+S2, detail) | `tessera-embeddings` S3 |
| **Sentinel-2 VNIR** | **100.0%** | **10 m (raw, sharp)** | `sentinel-cogs` AWS open data |

S2 is also the cheaper deployment: four bands, no 20 m resampling, no
pan-sharpening, windowed reads straight out of public COGs.

**The user's second observation, which sets the target.** The map that handled
stable built-up best was `twotower_20260724_172904/oslo/oslo_aef_only_coarse3.tif`
— the **AlphaEarth-only** view. Adding Tessera made stable-Artificial worse on
the map and cut Oslo's change pixels by 16% (G3). So the bar for S2 is not just
"beat AlphaEarth-only on change-F1"; it is **add detail without the suppression
Tessera introduced**. `art_stable_recall` and the within-AOI change-pixel
counterfactual are therefore first-class metrics here, not afterthoughts.

## Data path

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python
cd src
$P extract_s2_points.py --workers 64      # ~90 min, resumable, → data/embeddings/s2_shards/
$P build_s2_features.py                   # local, seconds → s2_features_habloss_recover.parquet
$P twotower_lab.py --group s2-detail --n-seeds 3
```

`extract_s2_points.py` stores a **64×64 patch** (640 m) per plot-year, not a
point sample. Element 84's COGs are tiled 1024², so GDAL fetches one internal
block per read and a 64×64 window costs exactly what a 5×5 window costs — the
patch is free. Every texture, neighbourhood and segmentation idea below is
therefore a local computation over stored arrays. **Idea D3 in the Tessera
backlog died on "needs re-extraction"; this is the fix.**

Composite: up to 4 scenes per year, the least-cloudy in each season so 2018 and
2024 are sampled from comparable phenology (a season-mismatched pair manufactures
"change" that is really leaf-on/leaf-off), SCL-masked, per-pixel median.

### I/O, after porting `aef_loader_plus`

Profiling the first version said reads were 75% of runtime and **the COG header
parse was 49% of a read** (1318 ms against 1270 ms for the window itself), with
GDAL barely caching it even on a repeat open of the same URL in-process
(1318 → 1087 ms). That is precisely the cost `aef_loader_plus/aef_loader/cache.py`
was written to remove, so `s2_cog.py` ports the idea: fetch the header once,
cache the bytes on disk, serve later reads as obstore range GETs against the
cached tile-offset table. `AEFIndex`'s "download the index, query it locally"
became a disk cache of STAC scene listings, keyed on a ~1 km cell.

Measured end-to-end on 24 plots, **all three paths byte-identical**:

| path | time | vs rasterio |
| --- | --- | --- |
| rasterio (`--no-cache`) | 45.3 s | — |
| cached header, cold | 36.2 s | 1.25× |
| cached header, warm | 28.3 s | **1.60×** |

Cold is already faster because one range GET beats vsicurl's machinery; warm
removes the header round trip entirely. The guard that makes a byte-offset
reader safe is in `s2_cog.metadata`: the tile table must match the image grid
exactly, and anything unverifiable falls back to rasterio — a truncated header
would otherwise read the wrong bytes *without raising*. Validated bit-exact
against rasterio on 100 real reads spanning both resolutions and granule edges.

The cache is worth most where this work is heading: the AOI inference path (S7)
reads far more COGs than the plot extraction does.

**Async obstore was tested and rejected — the Tessera precedent does not
transfer.** `extract_tessera_points.py` credits async obstore with ~23× over
requests+ThreadPool, so the obvious next move was to make `s2_cog` async. On
identical jobs with byte-identical output it is **1.5× slower**:

| payload | sync+threads(64) | async(128) |
| --- | --- | --- |
| 1 KB slices | 1.08 s | **0.45 s (async 2.43× faster)** |
| full COG tiles (~1 MB) | **5.93 s (39 MB/s)** | 8.68 s (27 MB/s) |

The crossover is payload size: Tessera reads 128 bytes per point and decodes
nothing, so it is pure latency; a COG tile is ~1 MB, and there the single
event-loop thread bottlenecks marshalling bytes from Rust into Python while a
thread pool spreads it across cores. Decode is *not* the cause — it is 1% of
read time and zlib releases the GIL (that was the first hypothesis, and it was
wrong; the 1 KB-vs-1 MB comparison is what settled it).

## Feature families

Written by `build_s2_features.py`, 1,535 columns, each family its own prefix so
an idea is a column subset:

| prefix | what | why |
| --- | --- | --- |
| `S2c` | centre pixel reflectance | what a point sample would have given |
| `S2m3/9/25` | neighbourhood mean at 30/90/250 m | local context |
| `S2s3/9/25` | neighbourhood **std** | **the point of the exercise** — a smooth embedding cannot say whether a pixel sits in a homogeneous field or a built-up mosaic |
| `S2lc` | centre − 90 m mean | signed local contrast: detail after context is subtracted |
| `S2g` | Sobel gradient magnitude | edge strength — high on roads and roofs, low inside vegetation |
| `S2p` | 8×8 mean-pooled image | a coarse picture rather than a statistic, for a tower that learns its own texture |

Channels: **eleven, all native 10 m** — blue, green, red, NIR + NDVI, NDWI,
brightness (`CHANNELS_BASE`, the deployed seven) + EVI2, GRVI, BSI, CI
(`CHANNELS_10M`, added in S19). Each family exists for 2018, 2024 and their
difference. `S2p` is the one exception and stays on the base seven: at 64
columns per channel per year it is three quarters of the table.

A subset is therefore a choice on **three** axes — families, channels, years —
and `s2_subset_columns` filters on all three. **The channel and year filters
default to the deployed seven and to all three year blocks**, which is what
keeps `s2off_centre_m3s3_bf` at exactly 78 columns after the channel expansion.
Anything that consumes the whole stat block instead of a named subset must say
which block it means by calling `s2_base_columns`; `tests/test_s2_subset_axes.py`
is the guard on both. The table is now
`s2_features_habloss_recover_10m.parquet` — see the S19 environment note for
why the old filename could not be reused.

## Metrics

Unchanged from `AUTORESEARCH.md`, so every number is comparable to the existing
ledger. A win clears ±0.005 seed noise on its target and loses ≤0.005 elsewhere.

| metric | deployed (`mc_dropout_scalars`) | AlphaEarth-only |
| --- | --- | --- |
| `change_f1` | 0.6704 | 0.6577 |
| `macro_f1` | 0.6993 | — |
| `art_stable_recall` | 0.639 | — |
| `art_stable_as_veg` | 0.220 | — |

## Iteration log

**Iteration 1 (2026-07-26) — infrastructure, no verdict.** Extraction in flight
(2/33 shards, 100% coverage holding). Harness plumbing verified end-to-end
against the partial shards: `load_context` attaches the S2 table, `s2_present`
lands on exactly the 400 extracted plots (6.24% of 6,414), no NaN leakage among
present rows, all six S2 ideas registered with column blocks of the designed
size (stat 189, texture 105, patch 1,344, scalars 5). **No idea run — a fold
built on 6% coverage would measure the extraction's progress, not the modality.**

**Iteration 2 (2026-07-26) — extraction complete, S0 answered, one real bug found.**
6,416 plots in 88 min, **99.4% both-endpoint coverage** (2018 99.8%, 2024 99.6%).

The first S0 run returned `change_f1=0.0000`, `artStab=1.000` — a degenerate
single-class collapse, **not** a result, and it was purged from the ledger and
the OOF cache rather than recorded. Cause: `HierarchicalSoftmaxNN._prepare`
standardises the flat (`arch="wide"`) path with plain `X.mean(0)`/`X.std(0)`,
which **propagates NaN**. Sentinel-2 reaches 99.3% of plots, not 100%, so 0.34%
NaN poisoned every column statistic and NaN'd the whole design matrix. The
two-tower branch was already NaN-safe (`nanmean`/`nanstd`, absent → 0); the flat
branch now matches it. `nanmean == mean` on NaN-free input, so no existing
result moves — verified: `baseline_aef` seed 0 = 0.6567, against 0.6577 at five
seeds. **Any modality that is not 100% dense would have hit this**; Tessera never
did only because it is imputed fold-wise upstream.

**Iteration 3 (2026-07-26) — S1 negative, and it fails the way Tessera failed.**
The headline fusion loses 0.0064 change-F1 against AlphaEarth-only, driven
entirely by change **recall** (−0.079) against a precision *gain* (+0.058), with
`art_stable_as_veg` worsening 0.203 → 0.247.

That is the important part. The Tessera line was abandoned on the theory that
its coverage hole was the bottleneck. **A modality with 99.4% coverage
reproduces the same change-suppression signature**, which is evidence against
that theory: the problem may be *adding a second sharp-but-noisy tower at all*,
not how often it fires. If S2 (`texture`) and S4 (asymmetric dropout) suppress
the same way, coverage was never the explanation and the two-tower design itself
is what suppresses change.

**Iteration 4 (2026-07-26) — the suppression was the GATE, not the modality.**

Three deterministic S2 fusions (S1, S2t, S4) all failed the same way: change
recall 0.634–0.652 against baseline's 0.713, precision up, `art_stable_as_veg`
worse. The tuned-threshold ceiling ruled out a calibration artefact — every
variant was below baseline even at its own best threshold (0.6517/0.6518/0.6537
vs 0.6594), so it was not an operating-point illusion.

The fix came from reading the ledger rather than trying another architecture:
`mc_dropout_scalars` carries change recall 0.7146 where its own deterministic
parent `tt_symmetric_md0.5` sits at 0.6899. Keeping the modality gate
**stochastic at test time** is the one lever already known to reverse this exact
symptom. It transfers: `mc_s2_drop0.7` restores recall 0.6516 → 0.7094 and turns
a −0.010 loss into a +0.008 gain over AlphaEarth-only.

**What this means for the Tessera conclusion.** Iteration 3 suspected the
two-tower design itself suppresses change regardless of coverage. That was half
right: **the deterministic gate suppresses change, and a stochastic one does
not** — a property of how the gate is read, not of what the modality says or how
often it fires. Tessera's coverage hole was still real, but it was never the only
thing standing between the two-tower and a usable detail modality.

**Where this leaves the deployment question.** `mc_s2_drop0.7` ties the deployed
Tessera model on the headline metrics, beats it on both built-up metrics, and
does so with **99.4% coverage instead of 35.8%**, from a source that is cheaper
and faster to fetch. It does *not* beat AlphaEarth-only on `art_stable_as_veg`
(0.212 vs 0.203) — on the specific error the user flagged in the Oslo map, plain
AlphaEarth is still the cleanest at plot level. **T2 (the within-AOI change
counterfactual) is now the deciding test**, and it needs the S7 inference path.

**Iteration 5 (2026-07-27) — section S is closed. The gate is the whole story.**

Every remaining S idea ran. Laid out by how the detail tower is read, the deploy
change-F1 sorts itself perfectly:

| how the S2 features enter | ideas | change-F1 |
| --- | --- | --- |
| **deterministically gated tower** | S1 0.6513 · S2t 0.6474 · S4 0.6479 · S5 0.6428 · S3 0.5976 | **all below baseline 0.6577** |
| **stochastically gated tower** | S6 0.6656 · S9 0.6666 | **above baseline** |
| **flat covariate, no tower** | N4 0.6590 (and best built-up anywhere) | at baseline, wins elsewhere |

**Five deterministic variants, five losses, no exceptions** — spanning hand-built
statistics, texture-only, asymmetric dropout, change scalars and a learned
pooled-image tower. The one structural change that reverses it is refusing to
commit to the gate at test time. That is now a well-evidenced claim rather than a
single observation.

Two secondary findings: hand-built statistics beat a learned texture badly (S3 is
the worst result on the board — 1,344 pooled values on 6,414 plots cannot be
fitted), and composing the two wins (S9) overlaps rather than adds, which is the
same shape F7 found. **On this problem, levers touching the same class decision
do not stack.**

**Plot-level modelling has converged.** The recommendation table below is stable,
and every remaining open question — T2, U1, U2, `boundary_align` — is about the
*map*, which is blocked on the S7 inference path. That is the next iteration's
work, and it is a build rather than a hypothesis test.

**Iteration 6 (2026-07-27) — S7 shipped, T2 answered: −8.8%, not the hoped-for zero.**

`infer_s2.py` maps the three recommended models over an AOI. Its `--self-check`
earned its place immediately by catching **two real train/serve mismatches before
any map was written**: built fraction's NaN convention (`NaN < cut` is False, so
an absent pixel counts as *not built* and stays in the denominator — excluding it
instead shifted the feature by up to 0.32 on a 0–1 scale) and an even-window
centring offset on `S2bf64`. Both would have produced plausible-looking maps from
features the models were never fitted on. Final agreement 1.2e-04, float32
rounding.

**T2's verdict is genuinely mixed and should not be rounded up.** S2 *does*
suppress change on the map (−8.8%), which was the failure the whole line was
meant to avoid. It suppresses about half as hard as Tessera (−16.0%), it does so
while making the map finer rather than blurrier, and unlike Tessera it costs no
measurable change recall at plot level (0.7094 vs 0.7132). Those three facts
together favour commission-removal over lost detection — but favouring is not
showing, and **G4 remains the only thing that would settle it: labelled plots
inside the AOI.** Fifty interpreted plots stratified on the 3,783 pixels where
the two views disagree would decide it directly.

## Backlog

### S. Does S2 detail help at plot level?

| # | idea | status | result |
| --- | --- | --- | --- |
| S0 | **`s2_only`** — S2 features alone on the wide/focal hier NN. Does raw VNIR carry the transition signal at all, before any fusion question? Calibrates everything below. | **ANSWERED** | **Yes, but it is much the weaker modality: change-F1 0.5231 ±0.004 full / 0.5532 ±0.005 subset, against AlphaEarth-only 0.6577/0.6585 and Tessera-only 0.6361 (subset).** So S2 alone is ~13pt below AlphaEarth and ~8pt below Tessera on the subset. That is the *expected* shape — AlphaEarth is a learned embedding over a full annual time series, S2 is four bands at four dates — and it does not bear on the fusion question, which is whether it carries something AlphaEarth lacks. It does set the prior: expect a fusion gain, not a replacement. `art_stable_as_veg` 0.351 is far worse than the deployed 0.220, so raw VNIR alone is *not* a built-up detector either. |
| S1 | **`tt_s2_stat`** — the headline candidate. AlphaEarth context tower + S2 detail tower over centre/neighbourhood/contrast/gradient, gated on a mask that is ~1 everywhere instead of 0.358. | **NEG** | **change-F1 0.6513 ±0.003 against AlphaEarth-only 0.6577 — a 0.0064 loss, outside the ±0.005 band. macro-F1 0.6761 vs 0.6899.** The mechanism is not a general degradation but a specific one: **change recall 0.635 vs 0.713 (−0.079) while change precision rises 0.611 → 0.669 (+0.058)**. The S2 tower makes the model *more conservative about change* — the same signature as Tessera's −16% Oslo suppression, arrived at from a modality with 99.4% coverage instead of 35.8%. `art_stable_as_veg` also worsens, 0.203 → 0.247, which is the specific error the AEF-only map was preferred for. Subset read behaves the same (0.6602 vs 0.6585, flat). **Caveat before writing the modality off: a recall-for-precision trade of this shape is what an unchanged arg-max does when a class's probability mass shifts, so this may be an operating-point result rather than a feature result** — E1/F3 (nested threshold and cost gate) are the honest test and are cheap on the cached OOF. |
| S2 | **`tt_s2_texture`** — the detail hypothesis in isolation: the tower sees ONLY std, local contrast and gradient, none of which AlphaEarth can represent. A win here is evidence about *detail*; a win only on S1 is evidence about *more features*. | TODO | |
| S3 | **`tt_s2_patch`** — hand the tower the 8×8 pooled image and let it learn its own texture. | **NEG (worst on the board)** | **0.5976 ±0.004 full / 0.6136 subset**, artStab 0.533, art→veg 0.382 — 0.06 below AlphaEarth-only and worse even than `s2_only`. 1,344 raw pooled values on 6,414 plots is squarely the overfitting regime, and it carries the deterministic gate on top. **Hand-built statistics beat a learned texture by a wide margin: there is not enough labelled data to learn filters.** |
| S4 | **`tt_s2_drop0.7`** — transfer C2, the one architectural lever that ever moved this metric. S2 is dense where Tessera was sparse, so the dropout optimum may well move; sweep it if 0.7 is not flat. | TODO | |
| S5 | **`tt_s2_scalars`** — D1+C2 transferred: per-modality change scalars on both towers under asymmetric dropout. | **NEG** | 0.6428 ±0.003 full / 0.6534 subset, art→veg 0.247. The change scalars that helped Tessera do not rescue a deterministically gated S2 tower — it lands in the same 0.643–0.651 band as every other deterministic variant. |
| S9 | **`mc_s2_bf`** — compose the two wins: built fraction flat on the always-on AlphaEarth tower, S2 texture behind the stochastic gate. | **FLAT — they overlap, exactly like F7** | **5 seeds: change-F1 0.6666 ±0.004, macro-F1 0.6971, artStab 0.6498, art→veg 0.2102.** Change-F1 is +0.0010 over `mc_s2_drop0.7` (inside noise), and **both built-up metrics land *between* the two parents** rather than above them (artStab 0.6462 → 0.6498 → 0.6572; art→veg 0.2118 → 0.2102 → 0.1916). This is the third time this shape has appeared on this problem — F7 (seed ensemble + cost gate), and now this. **Levers that move the same class decision overlap; they do not stack.** Choose one by goal instead of composing. |
| S2t | **`tt_s2_texture`** — detail families only. | **NEG** | 0.6474 ±0.002, the worst of the three deterministic variants, and the same signature (recall 0.634, art→veg 0.267). Texture alone is not the missing ingredient. |
| S4 | **`tt_s2_drop0.7`** — asymmetric dropout on the detail tower. | **NEG deterministic → see S6** | 0.6479 ±0.003 with the gate deterministic. The same configuration under a stochastic gate (S6) is the best S2 result, so the dropout was never the problem. |
| S6 | **`mc_s2_drop0.7`** — E4 on the S2 tower: keep the modality gate stochastic at test time, average 16 passes. Aimed directly at the recall collapse S1/S2t/S4 all share. | **WIN (qualified)** | **5 seeds, deploy read: change-F1 0.6656 ±0.005, macro-F1 0.6980, artStab 0.6462, art→veg 0.2118.** Against AlphaEarth-only (0.6577/0.6899/0.6415/0.2035): **+0.0079 change-F1 and +0.0081 macro-F1**, both just outside seed noise. Against the deployed Tessera model (0.6704/0.6993/0.6394/0.2204): change-F1 −0.0048 and macro-F1 −0.0013 — **inside noise, i.e. a tie** — while **beating it on both built-up metrics** (artStab +0.007, art→veg −0.009). `veg_stable_as_art` 0.0329 confirms it did not simply flood Artificial. **Change recall is fully restored: 0.7094 against the deterministic gate's 0.6516 and baseline's 0.7132.** |

### N. NDVI threshold and the built-fraction covariate (user's proposal)

The user proposed calibrating a hard NDVI cut against the stable-Nature points,
estimating ~0.3. `analyse_ndvi_threshold.py` tests it against the labelled plots.

| # | idea | status | result |
| --- | --- | --- | --- |
| N1 | **Where is the cut, empirically?** | **CONFIRMED — 0.31** | Fitting stable-Vegetation (n=4,524) against stable-Artificial (n=979) centre-pixel NDVI 2024: **Youden's J and balanced accuracy both peak at t=0.31** (sens 0.668, spec 0.664). The user's 0.30 scores balanced accuracy 0.665 against the optimum's 0.666 — **within 0.001**. Separability is real but modest: AUC 0.728, medians 0.428 (vegetation) vs 0.225 (built). |
| N2 | **Does the threshold fix the stable-Artificial error?** | **NO — and it is decisive evidence, not a null** | On the 215 AlphaEarth-misread built-up plots the cut fails almost completely: median NDVI **0.383** against stable Vegetation's 0.428, with **37.2% below t=0.31 versus vegetation's 33.6%** — nearly indistinguishable. AUC on that subset alone is **0.563**, barely above chance. **NDVI is a direct physical measure of greenness, and it says these plots are green.** That is independent confirmation of F6 from a completely different instrument: the plots are vegetated in the reflectance and built-up in the interpretation, so it is a label/protocol disagreement and *no spectral feature can win them*. |
| N3 | **Built fraction — the radius matters more than the cut.** | **CALIBRATED — 3 px** | Sweeping the window for stable-Artificial vs stable-Vegetation separability: AUC 0.669 (1 px), **0.762 (3 px / 30 m)**, 0.695 (5 px), 0.684 (9 px), 0.675 (15 px), 0.648 (64 px / 640 m). A plot is ~10 m, so 30 m captures a roof and its yard while 640 m dilutes it into the landscape. **3 px beats the continuous centre-pixel NDVI (0.728) and is the best simple built-up feature found.** |
| N4 | **`aef_builtfrac`** — AlphaEarth + the built-fraction covariate only, flat, no second tower. F6's exact prescription. | **WIN — best built-up result on the board** | **5 seeds, deploy read: artStab 0.6572, `art_stable_as_veg` 0.1916, change-F1 0.6590 ±0.002, macro-F1 0.6930.** Against AlphaEarth-only (0.6415 / 0.2035 / 0.6577 / 0.6899): **artStab +0.016 and art→veg −0.012, at no cost to change-F1** (+0.0013, inside noise) and with change *recall* also up (0.7164 vs 0.7132). `veg_stable_as_art` 0.0364 vs 0.0346 confirms it did not simply flood Artificial. **This is the lowest `art_stable_as_veg` of anything tested — below the deployed model's 0.2204 and below plain AlphaEarth's 0.2035 — from 15 extra columns and no architectural change.** |

**Why N4 works where N2 failed, and why that is not a contradiction.** N2 says a
threshold cannot rescue the 215 *worst* plots, because those are mislabelled
relative to their pixels. N4 says the same threshold, aggregated to a 30 m
built-fraction, still improves the class *overall* — it wins the ordinary
built-up plots more cleanly, which is a different population from the pathological
215. The user's instinct was right about the physics; F6 was right that the
residual is a label problem. Both hold.

### S10-S12. Tuning the model the user picked

The user's visual verdict (2026-07-27): **`mc_s2_drop0.7` is best** -- it fixes
stable built-up *and* catches more detail in places than Tessera did;
`aef_builtfrac` second, `baseline_aef` third. Note this **contradicts the plot
metrics**, which rank `aef_builtfrac` best on built-up (art->veg 0.192 vs 0.212).
Third plot/map disagreement in this project; the map is what ships, so these
target the MC model.

| # | idea | status | result |
| --- | --- | --- | --- |
| S10 | **Detail-tower dropout sweep.** 0.7 was carried over from Tessera and never chosen for S2, which is denser and cleaner. C2's dose-response was the biggest architectural lever ever found here. | **FLAT (and a 3-seed trap)** | At 3 seeds the sweep looked decisive -- 0.4/0.5/**0.6**/0.7/0.8 = 0.6637/0.6653/**0.6701**/0.6656/0.6612, a clean unimodal peak at 0.6 worth +0.0045. **At 5 seeds it collapsed to +0.0014 (0.6670 +/-0.006 vs 0.6656 +/-0.005) -- inside noise.** The inherited 0.7 was already fine. Textbook case for the 5-seed rule. |
| S11 | **MC keep-probability sweep** -- how often the detail tower is trusted per pass. 1.0 is the deterministic gate that suppressed change, 0.0 is AlphaEarth alone; 0.5 was inherited, never chosen. | **FLAT on the headline, REAL on built-up** | change-F1 0.3/0.5/0.7/0.9 = 0.6624/**0.6656**/0.6630/0.6555 -- 0.5 already optimal, and trusting S2 harder is clearly worse. But `art_stable_as_veg` moves **monotonically** with it: 0.198 / 0.212 / 0.216 / 0.229. **Trusting the detail tower *less* buys built-up accuracy**, which is the same direction `aef_builtfrac` points. A usable dial: keep=0.3 costs ~0.003 change-F1 and returns ~0.014 art->veg. |
| S12 | **`seed_ensemble_s2`** -- F2 transferred. Average the cached torch seeds. Free at inference; the deployed Oslo map currently runs a SINGLE seed. | **WIN (modest, and free)** | **change-F1 0.6658 +/-0.0015, macro-F1 0.6992, artStab 0.6529, art->veg 0.2096.** Change-F1 is flat, but **artStab is the best of any MC variant (+0.0067 over `mc_s2_drop0.7`), macro-F1 matches the deployed Tessera model, and the seed spread tightens 3.5x (0.0015 vs 0.0052)**. That last number is the deployment argument: a one-seed map is a draw from a distribution 3.5x wider than it needs to be. |

**Answer to "are there further gains?" -- modelling has converged; deployment
and data have not.** Two of the three knobs tested were flat at 5 seeds. What is
left that is real: the seed ensemble (free, and it makes the map reproducible),
the keep-probability dial if built-up matters more than change-F1, more S2
scenes per year (currently 4 -- a data-side lever never tried), and G4's
labelled plots inside the AOI, which is still the only thing that can settle
whether the -8.8% suppression is a fix or a fault.

### T. Does S2 avoid Tessera's suppression?
The reason the user pointed at `oslo_aef_only_coarse3.tif`. These are the
counter-checks, run on the same OOF cache.

| # | idea | status | result |
| --- | --- | --- | --- |
| T1 | **Stable-Artificial read.** `art_stable_recall` / `art_stable_as_veg` for every S2 idea via `rescore_ledger.py`. F6 concluded the residual confusion is a label-vs-pixel disagreement — but that diagnosis was made in *AlphaEarth* space. Built-up is exactly what 10 m VNIR texture sees and a smooth embedding does not, so F6's conclusion deserves a re-test on features that can actually resolve it. | TODO | |
| T2 | **Within-AOI change counterfactual.** Re-run G3's design with the S2 gate forced off: same model, same Oslo pixels, count change with and without S2. Tessera gave −16.0%. If S2 gives ≈0% while lifting plot metrics, that is the whole objective met. | **ANSWERED — −8.8%, half of Tessera, but not zero** | **`mc_s2_drop0.7`, 2,954,952 identical Oslo pixels: 14,589 change with the S2 gate forced off → 13,300 with it on, −8.8%; 2,536 px removed, 1,247 added.** So **S2 also suppresses change at pixel level** — the hoped-for ≈0% did not happen — but at roughly half Tessera's rate on the same AOI and the same design. Structurally the suppression is *not* a blurring: with the gate on the map gains edges (0.0935 vs 0.0892) and finer segments (median 4 px vs 6 px) at essentially unchanged boundary alignment (1.608 vs 1.611, a 0.2% difference that is noise). So S2 redistributes toward finer structure while netting out fewer change pixels. **The plot evidence differs from Tessera's in the right direction**: S2's change recall is 0.7094 against baseline's 0.7132 — statistically unchanged — whereas Tessera's deterministic read cost 0.06–0.08 of recall. That argues the removed pixels are commission rather than detection, but **it cannot be proven without labelled plots inside the AOI** — the same G4 limit that blocked the Tessera version. |
| T2b | **Counterfactual for a flat model.** | **N/A, and worth recording so it is not re-run** | `aef_builtfrac` returns +0.0% (0 px moved) because `arch="wide"` has no modality gate and never reads the mask column — forcing it to zero is a no-op. Reported naively this looks like proof of "no suppression" and is nothing of the kind. `infer_s2.py` now skips the counterfactual for ungated recipes. The meaningful comparison for a flat model is against `baseline_aef`: **11,610 → 14,255 change px, +22.8%** — it *adds* change. |
| T3 | **Three-modality read.** S2 and Tessera in one detail tower on the 35.8% where both exist — does Tessera add anything S2 does not already carry? Decides whether Tessera stays in the stack at all. | TODO | |

### U. Map-level detail (the HQ-SAM direction)
The user's second thread: sharpening the *map*, not the plot score. Honest
scoping first — **memory records that zero validation plots fall inside either
AOI**, so nothing here can be scored the way sections S/T are.

| # | idea | status | result |
| --- | --- | --- | --- |
| U1 | **Guided / joint-bilateral refinement.** Use the S2 VNIR composite as the guide image for edge-aware filtering of the class-probability raster. Cheap, no training, and the classic fix for "context model, blurry boundaries". | **NEG — it buys alignment by eating the change class** | `refine_map.py`, guided filter (He et al. 2013) over the one-hot class stack, S2 NIR 2024 as guide, on `oslo_mc_s2_drop0.7` (baseline: edge 0.0935, medseg 4 px, align 1.6080, **13,300 change px**). It does exactly what it promises structurally — **boundary alignment rises to 1.677 (r=1) and 1.781 (r=2), and median segment goes 4 → 9–12 px** — but the change class does not survive: **−11.7% at the mildest setting (r=1, eps=1e-3) and −32.8% at r=2**, against only 1.0–2.7% of pixels moved. Change is 0.45% of the map and spatially fragmented, so a neighbourhood arg-max is close to a majority vote and the minority class loses every tie. **This is the same shape as the gate suppression: smoothing of any kind preferentially destroys the class the user cares about.** A probability-level variant would be strictly stronger (a confident change pixel could outvote its neighbours) and needs `infer_s2.py` to persist posteriors — that is the version worth trying, not this one. |
| U2 | **Segment-constrained averaging (SAM / HQ-SAM).** Segment the S2 composite, average probabilities within each segment. HQ-SAM's contribution is boundary fidelity, which is exactly the failure mode. Cost: a ViT over every AOI tile — check it is worth it over U1 before building it. | **NEG — built and measured, Iteration 10** | SAMRefiner (ICLR 2025) tested directly rather than assumed under U1b's argument, since a prompt-based refiner is not an averaging scheme. From a 3×3 seed it is a **no-op** (IoU 1.000 with its own input); from the NDVI built mask it **hurts** (`boundary_align` 1.144 → 1.113, built fraction dilated 0.387 → 0.484). SAM's *segmentation* is real — `boundary_align` 1.277 against a 0.998 random-partition control, 100% of chips — but the mask containing the plot centre is bimodal (median 126 px, **q75 ≈ 4,090 of 4,096**). Motive largely removed by **U4**. |
| U4 | **The radius sweep behind the deployed feature block.** Fell out of U2's baseline failing to reproduce. | **BUG — 3 px won on a tie-blind rank** | `analyse_ndvi_threshold.auc` gave tied values distinct ranks in array order, with the Artificial class concatenated first. Built fraction over 3×3 has ten distinct values and was the column most flattered: **0.762 → 0.686**, against 5 px at 0.685. The deployed block is unchanged and 3 px still leads — by 0.001, not 0.067. Fixed to mid-ranks; verified against sklearn. |
| U0 | **Track detail as a number instead of an impression** (`map_detail_metrics.py`). Label-free, so it runs on Oslo where no plots exist. Four metrics: `edge_density`, `segments_per_mp` + `median_segment_px`, `hf_power_ratio`, and `boundary_align` (mean S2 gradient on class boundaries ÷ mean over all pixels). | **DONE — and it reproduces the user's visual judgement** | Oslo, same 2.95 M pixels: `aef_only` edge density 0.079, **median segment 5 px**, 1,992 segments/Mpx · `both` 0.111, **1 px**, 5,958 · `tess_only` 0.180, **1 px**, 15,991. **Tessera more than doubles edge density while collapsing the median segment to a single pixel — that is speckle, not detail**, and it is why the AlphaEarth-only map read better by eye. **`edge_density` alone is an actively misleading detail metric; it must be read against `median_segment_px`.** `boundary_align` is the quality half and needs an S2 composite over the AOI — blocked on S7. |
| U1b | **Probability-level refinement** — refine the posteriors rather than the hard class map, so a confident change pixel can outvote its neighbours. The strictly stronger version of U1. | **NEG — worse than U1, and the reason inverts the premise** | At matched settings (r=1, eps=1e-3) it buys more alignment (**1.7293** vs one-hot's 1.6770) and costs **nearly double the change class: −22.2% vs −11.7%**; at r=2 it reaches −44.1%. **The premise was backwards.** Change pixels are the model's *least confident* predictions — median top-probability **0.493 against stable's 0.751**, median margin **0.155 against 0.570**, and **34.9% of change pixels sit within 0.1 of flipping versus 7.0% of stable ones**. So one-hot's information loss was accidentally *protective*: it promotes a 0.49-confidence change pixel to a full 1.0 vote and lets it compete with a confident stable neighbour. Handing the filter the true posteriors just tells it which pixels are cheap to erase. |
| U3 | **A scorable proxy for U1/U2.** The stored 64×64 patches make a patch-level read possible: refine over the patch, score the centre pixel against its label. **Blocked** — it needs AlphaEarth at all 4,096 patch pixels, and AEF was only ever extracted at points. Resolving this is a GEE re-extraction, and it is what would turn U1/U2 from "looks sharper" into a number. | BLOCKED | |
| U5 | **Phoenix (ECCV 2026), a *learned* mask refiner** (`phoenix_refine_map.py`). The user's ask. Same slot as U1/U2 and it inherits SAMRefiner's prompt sampler outright, but its claim is training-side — adversarial mask perturbation + contrastive refinement, i.e. a refiner that has seen realistic segmentation errors — so U1b's smoothing argument does not cover it any more than it covered U2. Built and measured. | **NEG for deployment — and it fails in the opposite direction to U1** | Deployed Oslo map, 16,676 change px, seed floor 0.8423. Three regimes: the paper's own **semantic** protocol **+436.8%** change, align **1.6260 → 1.3579**, the map visually destroyed; **binary** (change mask alone, whole 1024 tile) **+270.0%**, medseg 5 → 180 px; **crop** (each component in its own 256 px window, the most favourable framing) **+62.2%**, IoU with input **0.3656**. **Every refiner in section U now fails on the change class, but U1/U1b erred by omission (−11.7%/−22.2%) and Phoenix errs by commission (+62% to +437%)** — a smoothing refiner deletes a marginal class, a generative one invents objects around it. **The one positive is real and controlled**: crop-regime `boundary_align` **1.4254 → 1.4944**, against **1.4478** for uniform growth to the identical 27,056 px and **1.0042** for randomly relocated blobs — so ~2/3 of the gain is content, not area. It is bought at +62% change pixels and an input-agreement less than half the map's own seed floor. **The structural reason is the object size**: the change class is 992 components with **median 6 px, q75 16 px, max 832 px**, and 650 of them are under 10 px; those sub-10 px components take median area ratio **2.33** and supply **57% of all added pixels**, while components over 100 px sit at ratio ~1.0–1.3 and IoU 0.49–0.66. Deterministic — two independent runs agree on **100.0000%** of pixels. |

### V. Deployment
| # | idea | status | result |
| --- | --- | --- | --- |
| S7 | **S2 inference path.** `infer_twotower.py` maps AEF+Tessera over an AOI; the S2 equivalent needs the same feature families computed over a raster rather than at points. `change_scalar_arrays` is already array-level for exactly this reason — keep the same discipline so train/serve skew stays impossible. | **DONE** | `infer_s2.py`. Sliding-window twin of `build_s2_features`, guarded by `--self-check` (raster path vs training table, per plot, per column; aborts above 1e-3). Passing at **4.24e-05** on `S2lc_bright_2018`. Writes merged2, coarse3, binary change, both posterior stacks, the S2 RGB/NIR backdrops and the gate-off counterfactual. |
| S8 | **Cost read.** S2 VNIR is claimed to be faster to fetch than Tessera. Measure it on a real AOI rather than asserting it — it is one of the stated reasons to prefer this path. | **DONE (Oslo, 2.95 M px, one A40)** | Composites + reference exports **179 s**, sliding-window features **46 s**, predict **224 s** at 5 seeds / 10 forward passes (≈58 s at 1 seed). Written to `summary.json["_timings"]` on every run. The predict stage is dominated by assembling the per-batch feature frame, not the network: 192 cols costs half of 396 cols at the same pass count, and a *second* forward pass adds 0.5%. Once the gate is marginalised (S13) the composite fetch is the bottleneck, so further architecture work cannot make a wide-area run cheaper. |
| S13 | **Marginalise the MC gate instead of sampling it.** At inference the modules are in `eval()`, so dropout is off and BatchNorm is frozen — every pixel is independent and the only stochastic element is the **binary** `S2_MASK`. The 16-pass average is therefore an estimator of a *two-valued* expectation. | **WIN — 8x cheaper, deterministic, and slightly more accurate** | Verified as an identity, not an approximation: a single pass is bit-repeatable (`max|diff|=0`), and sampled MC converges onto the exact value at the 1/sqrt(n) rate (mean abs. diff 1.68e-3 / 7.42e-4 / 3.80e-4 at n=100/500/2000). At the deployed n=16 the sampler still carried mean abs. error **0.004** and flipped **0.5% of merged2 / 1.2% of coarse3** plot labels per RNG seed. Exact beats sampled **+0.0031 change-F1 on 5/5 seeds**. At map level the two agree on 99.85% of Oslo pixels (15,532 → 15,517 change px, −0.10%); the sampled run reproduces the recorded 15,532 fingerprint exactly. **The `mask=0` branch is also the T2 counterfactual**, so that diagnostic stopped costing a second full sweep. Deployed 5-seed predict: **160 forward passes → 10, ~54 min → 3.9 min**. `--mc-sampling` keeps the old path for reproduction. |
| S14 | **Ablate the whole ladder** — seven stacked decisions, none ever priced together, with inference cost beside accuracy (`ablate_s2_architecture.py`). | **DONE — and two results are uncomfortable** | Paired over 5 seeds/folds: hierarchical supervision **+0.0199 (5/5)**, stochastic vs deterministic gate **+0.0186 (5/5)**, S2 tower on vs forced off **+0.0100 (5/5)**, whole S2 stack vs AEF-only **+0.0090 (5/5)**, 5-seed ensemble **+0.0004 (3/5)**. (1) **Every intermediate rung B–F is at or below the AlphaEarth-only baseline** — the apparatus only pays at the last step. (2) **The gate is the whole gain**, and the same model with the detail tower forced off scores 0.6566, i.e. the plain baseline — so S2 contributes *averaging over whether to trust it*, a regularisation effect, not information AEF lacks. (3) **Hierarchical supervision is the largest single lever and is free at inference.** (4) The 5-seed ensemble buys no change-F1 at 4x cost; what it buys is spread (±0.0054 → ±0.0012), coarse3-F1 +0.005 and artStab +0.006 — keep it for published maps, drop it for wide-area production. |
| S15 | **coarse3 output on the map** (`COARSE3_HANDOFF.md`). | **DONE** | The fine posterior was already computed and discarded; retaining it costs one accumulator. Aggregating coarse3 → merged2 reproduces the merged2 raster on **99.19%** of valid pixels — *not* the ~100% the handoff predicted, because arg-max does not commute with the group sum. On the change class the two reads overlap on only 13,239 of 15,517 / 14,656 px (~15% disagreement), so **merged2 stays the read for change** and coarse3 answers *what kind*. In Oslo three of the nine transitions are empty and a fourth has 1 px, so the extra detail is mostly latent in this AOI; the informative split is Nature→Artificial 10,145 px vs Cropland→Artificial 496 px. Also note `twotower_lab._mc_s2` scores the coarse3 head from a *deterministic* pass while the map averages the gate — the ledger's `fine_change_f1` for MC ideas was never the map's number; `ablate_s2_architecture.py` scores both heads as the map produces them. |
| S16 | **Tune the gate-off recipe for the way it is served.** Every hyper-parameter `s2off_deploy` inherited was chosen while the model was being *scored* with the gate on; `modality_dropout=0.5` leaves the AlphaEarth tower standing alone on half the both-present rows against 100% at serving (`experiment_s2off_training.py`, 15 seeds, gate-off read). | **DONE — md is worth +0.0025 and free; the S2 tower is worth nothing measurable** | `md` 0.3/0.5/0.7/0.9 = 0.6482/0.6556/0.6582/0.6584 change-F1 — **matching training to serving is monotone and saturates at 0.7**, and `dropout_tess` (0.4 vs 0.7) is flat everywhere. But the reference is the uncomfortable number: **`A_aef_flat`, with no Sentinel-2 at any point, scores 0.6574** — the best two-tower setting beats it by **+0.0013 change-F1 and +0.0019 macro-F1, against a seed spread of 0.005.** Under the deployed read the privileged-information story does not survive contact with 15 seeds. |
| S17 | **Price every decision in the deployed model in time, not accuracy** (`optimise_s2off.py` for training, `infer_s2.predict` for serving). Serving and training are different budgets and, because the detail tower is never run, most decisions are only on one of them. | **DONE — 10x serving, 189 of 204 columns deleted, no number changed** | See Iteration 8. |
| S18 | **Choose a *reportable* detail-tower feature set on map evidence.** "204 engineered Sentinel-2 features" is not a defensible methods sentence, and plot metrics cannot make the choice — every subset ties. Two label-free readings can: `map_detail_metrics` (does the structure survive, and does it fall on real edges) and per-class **IoU against the full-204 map**, which is the incumbent a reviewer will ask about. `compare_s2_subsets.py`, seven named subsets in `twotower_lab.S2_SUBSETS`. | **DONE — 57 columns, and the control is the result** | See Iteration 9. |

**Iteration 9 (2026-07-27) — S18: the map reproducibility floor, and a control I nearly got wrong.**

Seven subsets mapped over Oslo at 5 seeds, scored on detail and on change-class
IoU against the full 204. Plot metrics cannot make this choice — every subset
ties — so the question is entirely one of map agreement and map structure.

**The reproducibility floor is the number everything else is read against.** The
full 204-feature model, refitted on a **disjoint seed block (5–9 instead of
0–4)**, agrees with itself at only **change IoU 0.8402 / 98.61% pixels**
(merged2; 0.8362 on coarse3). A 5-seed ensemble is still one draw.

**The error worth recording**, because it is the opposite of the project's usual
one. The first pass compared each subset against `full` *across* seed blocks and
read `centre_m3s3_bf` (78 cols) dropping 0.9026 → 0.8318 as a replication
failure. It is not: two models that each reproduce themselves at ~0.84 will agree
with each other at ~0.84, and that is exactly what happened. The right control is
**each model's own** self-reproducibility, which puts 78 level with the deployed
block:

| detail tower | cols | self-IoU (0–4 vs 5–9), merged2 / coarse3 | vs full, same block |
| --- | --- | --- | --- |
| `full` (deployed) | 204 | **0.8402 / 0.8362** | — |
| `centre_m3s3_bf` | 78 | **0.8393 / 0.8097** | 0.9026 / 0.8318 |
| `centre_s3_bf` | 57 | 0.8646 / 0.8454 | 0.9109 / 0.9069 |

So **57 and 78 are both inside the noise floor and the gap between them is
smaller than the floor** — neither map metric can separate them. What *can* be
separated is everything below: `centre_bf` (36) at 0.8824, `centre_3px_lc_bf`
(99) at 0.8244, and `bf`/`s2off_slim` (15) at **0.7696**, genuinely below the
floor. That last one retrospectively justifies Iteration 8's caveat on the slim
map: it really is a different map, not a cheaper one.

**More features is not the axis.** 99 columns scores worse than 57 on every
column, and 162 overshoots the change class (18,664 px). Which families, not how
many.

**Selected: `centre_m3s3_bf`, 78 columns**, on the user's visual read of the
coarse3 map (2026-07-27) — the tie-break this project has used since the
AlphaEarth-vs-Tessera call, and legitimate precisely because the quantitative
metrics tie. It is supported rather than merely permitted by the numbers: on the
coarse3 read of the draw judged, it carries **more structure and better boundary
alignment than the full block** (edge density 0.0970 vs 0.0949, alignment 1.6071
vs 1.6013, `centre_s3_bf` 0.0935 / 1.5959), and on plots it matches the full 204
to the fourth decimal on all four headline metrics (change-F1 0.6557 ±0.0034 vs
0.6555, macro 0.6938 vs 0.6935, coarse3-F1 0.5954 vs 0.5953, artStab 0.6421 vs
0.6420) — the closest of any subset.

**Reportable sentence:** *seven Sentinel-2 channels (blue, green, red, NIR,
NDVI, NDWI, brightness) at the plot centre, plus their mean and standard
deviation in a 3×3 window, plus built fraction at five radii, for 2018, 2024 and
their difference* — 7×3 + 5 = 26 per year, ×3 = **78**.

**Iteration 8 (2026-07-27) — S17: the model was never the cost.**

Two independent findings, one per budget.

**Serving: 66.2 s → 6.5 s for Oslo at 5 seeds, bit-identically.** Profiling the
deployed path per 200k-pixel batch found 611 ms building a DataFrame, 373 ms × 5
standardising it in numpy, and **98 ms of actual tower** — 84% plumbing. Three
changes, none of them touching the model: `probs_aef_only_matrix` takes the raw
matrix and standardises on the device, so the batch crosses the bus once instead
of once per ensemble member; `stack_aef_bands` builds the 2.3 GB pixel matrix
**band-major** (1.3 s of sequential stores against 11.7 s of strided scatter);
and a model wanting exactly `aef_cols` gets that array rather than a second copy
of it (11.1 s and 2.3 GB per model). The Oslo run reproduces the previous map's
change count and **all nine coarse3 class counts exactly** — 17,941 change px,
99.5579% agreement — which is the end-to-end version of the bit-equality the
unit tests assert.

| stage | before | after |
| --- | --- | --- |
| AlphaEarth pixel matrix | 12.3 s | **1.8 s** |
| per-model re-stack | 11.1 s | **0.0 s** |
| predict (5 seeds) | 42.8 s | **4.7 s** |
| **total** | **66.2 s** | **6.5 s** |

The deployed model was also missing from `ablate_s2_architecture.py`'s cost
ladder, which priced every rung *except* the one that ships. Added as
`G_s2off_deploy` / `G_s2off_deploy5`, and the gap is the whole argument for the
gate-off deployment: **2.6 s per Oslo at 1 seed and 4.8 s at 5, against 50.7 s
for the same model served with the gate on and 234.5 s for its 5-seed
ensemble** — 49x, on identical weights. Band-major also made the *general*
(DataFrame) path 25–30% cheaper for free, because a per-batch column is now a
contiguous row slice: `A_aef_flat` 22.6 → 14.6 s, `B_aef_s2_flat` 43.7 → 34.2 s.

**Training: the 204-column detail block is carrying at most one family.** Every
subset tried lands within ±0.0005 change-F1 of the full block, against a seed
spread of 0.004 — built fraction alone (15 cols) 0.6553, centre reflectance alone
(21) 0.6550, texture only (105) 0.6549, single-date (68) 0.6553, no-diff (136)
0.6558, full 204 0.6555. **Built fraction is the family to keep**, and not for
its size: it is the only subset that *improves* both built-up metrics
(artStab 0.6420 → **0.6539**, art→veg 0.1921 → **0.1814**) while tying change-F1
and macro-F1 — the `aef_builtfrac` lever arriving through the privileged tower
instead of flat. Registered as `s2off_slim`. The saving is upstream, since
serving already skips Sentinel-2: no windowed mean/std/contrast/gradient
families, and extraction needs red + NIR + SCL rather than all four VNIR bands —
3 windowed COG reads per scene instead of 5, on the stage measured in hours.

**But the slim map is not the deployed map, and the plots cannot say which is
right.** Oslo, 5 seeds: 17,941 change px → **14,824, −17.4%** (3,692 removed,
575 added), merged2 agreeing on 97.93% of pixels. The plot evidence says the two
are indistinguishable — change recall 0.7296 vs 0.7254, change-F1 0.6555 vs
0.6553 — and the built-up metrics favour the slim one. So this is G4/T2 again:
**a visible change-pixel difference with no labelled plot inside the AOI to
adjudicate it**, and the standing instruction is not to suppress change. Treat
`s2off_slim` as the cheap candidate to judge on the map, not as a drop-in for
`s2off_deploy`.

**Three decisions that must not be cut.** The AlphaEarth `diff` block is not
redundant despite being a linear function of the other two thirds: dropping it
costs **−0.048 change-F1** (ten seed-spreads), in the two-tower and flat alike.
Epochs cannot come down — 30 → 20 → 15 → 10 gives 0.6555 / 0.6274 / 0.6110 /
0.6078, and at 10 epochs `art_stable_as_veg` blows out to 0.36. And `tower_dim`
is **not** the serving cost: 256 → 128 → 64 costs 0.6555 → 0.6534 → 0.6525 and
saves 6% of the tower, because the hard-coded 1024/512 hidden widths are the
FLOPs (halving *those* is 2.1x, and is untested). Keep 256.

**Seeds are now the serving cost**, one member per pass over every pixel
(`optimise_s2off.py --phase seeds`, 10 seeds, exact ensemble metrics from cached
OOF posteriors). The knee is 3, on both recipes: `s2off_deploy` spread
0.0051 → 0.0025 (k=3) → 0.0016 (k=5) and `s2off_slim` 0.0044 → 0.0010 → 0.0001,
with means flat to +0.002. 5 remains right for a published map; 3 is defensible
for wide-area production, and 5 → 3 is now a 35% cut of the whole predict stage
rather than of a stage dominated by pandas.

**Iteration 7 (2026-07-27) — U1 negative; smoothing and the change class are in direct conflict.**

Guided filtering delivers its structural promise (alignment +4.3% to +10.7%,
median segment 4 → 9–12 px) and still fails, because it pays for that in the
change class: −11.7% at the gentlest useful setting. Three separate mechanisms
have now shown the same thing — a deterministic modality gate, Tessera's fusion,
and now neighbourhood smoothing — **anything that regularises the map spatially
removes change pixels first, because change is 0.45% of the surface and
fragmented.** The only interventions that have *added* change are the ones acting
on the classifier's inputs (built fraction, +22.8%) or on its uncertainty (MC
gating), never on its output raster.

A bug worth recording: the first sweep scored change against `MERGED_COLORS`
insertion order, but `write_class_raster` numbers codes from the model's *sorted*
class list, so stable Vegetation was counted as change (1.85 M px instead of
13,300). `refine_map.py` now reads the authoritative code→label mapping from the
`.qml` sidecar, and the corrected baseline reconciles exactly with
`infer_s2.py`'s reported 13,300 px.

**Iteration 8 (2026-07-27) — U1b negative; the change class is intrinsically marginal, and that closes the post-processing branch.**

The probability-level refinement was supposed to fix U1 and made it worse
(−22.2% change against −11.7%). Measuring why produced the most useful number of
the iteration: **change pixels carry median top-probability 0.493 and median
margin 0.155, against stable's 0.751 and 0.570, with 34.9% within 0.1 of
flipping.** The change class is not merely small and fragmented — it is *barely
won*. That single fact explains every suppression result in this document at
once: the deterministic gate, Tessera's −16.0%, S2's −8.8%, guided filtering's
−11.7%, and now −22.2%. Anything that lets neighbouring evidence compete against
a change pixel wins, because the change pixel was only just ahead.

**Section U's filtering branch is therefore closed, and U2 (SAM / HQ-SAM
segment-constrained averaging) should not be built.** Segment-constrained
averaging *is* neighbourhood smoothing with better-shaped neighbourhoods; the
mechanism above predicts it removes change for exactly the same reason, and it
costs a ViT over every AOI tile to find out. If map-level refinement is revisited,
the only design consistent with the evidence is one that **protects the change
class explicitly** (refine stable/stable boundaries only, or carry a
class-asymmetric prior into the arg-max) rather than one that hopes better
boundaries come for free.

Two plumbing defects were caught this iteration, both silent: `--save-probs`
parsed and the run succeeded while writing nothing (the flag and the write block
were wired, the *call site* was not), and the posterior loader now matches bands
to classes by **band description** rather than position, since assuming order
would permute classes into a still-plausible-looking map. The change count
reproducing at exactly 13,300 across independent runs is what makes these
detectable.

**Deployment (2026-07-27) — the seed ensemble is mapped.**
`infer_s2.py --seeds 5` fits N torch seeds and averages their posteriors together
with the 16 MC passes (two different variances: which model was fitted, and how
much the detail tower was trusted). Oslo, `mc_s2_drop0.7`:

| read | change px | edge density | median segment | boundary align |
| --- | --- | --- | --- | --- |
| single seed | 13,300 (0.45%) | 0.0935 | 4 px | 1.6080 |
| **5-seed ensemble** | **15,532 (0.53%)** | 0.0914 | **5 px** | 1.6124 |

The two maps agree on **98.90%** of pixels; 32,477 changed class, and the change
class gains 2,690 px while losing 458 — **ensembling recovers change rather than
smoothing it away**, which is the opposite of every spatial post-process tried
(U1/U1b) and consistent with change pixels being marginal
(median margin 0.155): averaging seeds pushes a barely-lost pixel over the line
as often as it pushes a barely-won one under, and there are more of the former.

**Its T2 counterfactual is −13.4%** (17,941 change with the S2 gate forced off →
15,532 with it on; 3,387 px removed, 978 added), against the single seed's −8.8%.
**Both counts rise and the ratio does not improve** — which replicates G1's
finding on Tessera exactly (single −16.0% → 5-seed −16.8%). Better-calibrated
probabilities do not soften the suppression, so it is a property of what the
modality says rather than of how confidently one seed says it. That is now
established independently on both detail modalities.

**Iteration 10 (2026-07-31) — U2 built and closed on measurement; and the radius sweep that justified the deployed feature block was wrong.**

The user asked for [SAMRefiner](https://github.com/linyq2117/SAMRefiner) (ICLR
2025) as a route to map detail. It is worth separating from the U2 row it lands
on: U2 was written as *segment-constrained averaging*, and Iteration 8 closed it
by the smoothing argument. SAMRefiner is not an averaging scheme — it is a
prompt-based refiner (distance-guided points, context-aware elastic boxes,
Gaussian mask prompts, IoU-adaptive fusion), so the closing argument does not
automatically cover it. It was therefore built and measured. SAM ViT-B
(checksum-verified), the 64×64 chips, 6,416 of 6,492 plots covered. Its
`FastGeodis` dependency is only ever called with `lamb=0.0`, which *is* the
Euclidean distance transform, so scipy substitutes for it exactly — no CUDA
extension needed.

**U2a — SAMRefiner prompted from the 3×3 centre box: a no-op.** Median IoU with
its own input **1.000**; 97% of outputs still exactly 9 px. A 3-pixel seed hands
SAM box, point and mask prompts that all assert the same region, and returning it
unchanged is the right answer to an over-determined question. Nothing about the
method is being exercised at this scale.

**U2b — SAM does see 10 m Sentinel-2, and this is the one positive.** Unprompted,
the automatic mask generator scores `boundary_align` **1.277 against 0.998 for a
random-partition control**, positive on 100% of chips. The control is load-bearing:
any partition of a smooth image scores above 1 from gradient autocorrelation
alone, so the bare 1.277 is meaningless without it. Median mask 14 px — fine, but
not the 1 px speckle U0 caught Tessera producing.

**U2c — given a genuine chip-scale coarse mask, SAMRefiner mildly hurts.** On the
NDVI built/vegetation mask: IoU 0.915 with its input, `boundary_align` **down**
1.144 → 1.113, and built fraction dilated 0.387 → 0.484.

**So the refiner is not the lever; the segmentation is.** That reframes it into
something scorable on inputs rather than on the output raster — replace the
detail tower's fixed square windows with the SAM segment containing the plot
centre, on the theory that a square straddles boundaries and an object does not.
**That reframing is what turned up the real result, and it is not about SAM.**

**U4 — the 3 px radius was chosen on a tie-blind AUC.** Testing the segment
window against the recorded yardstick reproduced the baseline at 0.685 where the
record says **0.762**. The gap is not subsampling (0.685 sits dead centre of the
balanced-subsample distribution, mean 0.690, sd 0.015) — it is that
`analyse_ndvi_threshold.auc` ranked with `ranks[argsort(v)] = arange(1, n+1)`,
giving tied values *distinct* ranks in array order. `values` puts the Artificial
class first, so every tie resolved in its favour.

| window | unique values | recorded | tie-aware |
| --- | --- | --- | --- |
| 1 px | 2 | 0.669 | 0.667 |
| **3 px** | **10** | **0.762** | **0.686** |
| 5 px | 26 | 0.695 | 0.685 |
| 9 px | 82 | 0.684 | 0.681 |
| 25 px | 620 | 0.664 | 0.665 |
| 64 px | 2402 | 0.648 | 0.648 |

Built fraction over a 3×3 window has **ten distinct values** — it is almost all
ties, and it was the single column the bug flattered. Columns with few ties are
unaffected to within 0.001, **which is exactly why this survived: only the winner
was wrong.** `auc()` now uses mid-ranks and agrees with sklearn on tie-heavy and
tie-free data alike; `build_s2_features.py`'s justifying comment is corrected.

**The deployed feature block does not change** — 3 px still comes first. But it
leads 5 px by **0.001, not 0.067**, and the curve is flat out to 9 px. Any future
claim resting on "3 px is decisively the right radius" should be re-read against
this table; the honest statement is that the model is given all radii and the
choice is nearly free.

**U2 verdict: NEGATIVE, and now closed on measurement rather than on argument.**
Iteration 8's prediction was right for a reason it did not name — not only does
segment-constrained averaging smooth, but the segment SAM offers at plot scale is
bimodal: the smallest mask containing the centre has median area ~126 px but
**q75 ≈ 4,090 of 4,096**, so a quarter of the time the "object" is the whole-chip
background. **And U4 removes most of the motive.** The corrected sweep spans only
0.686 (best window) to 0.648 (whole chip), so a perfectly object-aligned window
can buy at most ~0.04 on this proxy — against S14's finding that the entire S2
detail tower is worth +0.0013 change-F1 at 15 seeds under the deployed read.
Chasing a fraction of 0.04 through a ViT at serving is not a trade this project
should take. **The unrun step, if anyone revisits it, is the matched-area
control** — SAM segment against a *square box of the same area* — since only that
separates alignment from size, and nothing above does.

**Iteration 11 (2026-08-14) — U5: Phoenix, and the matched-area control Iteration 10 asked for.**

The user asked for [Phoenix](https://phoenix-eccv26.github.io) (ECCV 2026,
`naver-ai/Phoenix`). It lands on the same row as U1 and U2 and it imports
SAMRefiner's prompt sampler verbatim, so the temptation was to close it by
argument for the third time. That would have been wrong twice over: its claim is
**training-side** (adversarial mask perturbation, contrastive refinement — a
refiner that has seen realistic segmentation errors rather than random geometric
ones), and the result below is not the one the standing argument predicts.

`phoenix_refine_map.py`, `phoenix_efficientvit_xl1`, deployed Oslo map
(`s2_20260731_100710`, 16,676 change px), S2 RGB 2024 as the image, S2 NIR 2024
as the `boundary_align` reference. **Seed floor first, per CLAUDE.md**:
`compare_map_iou.py` on the two disjoint seed blocks gives change-class IoU
**0.8423** — that is what any edit has to be read against.

| regime | change px | Δ | IoU w/ input | edge | medseg | align |
| --- | --- | --- | --- | --- | --- | --- |
| input map | 16,676 | — | — | 0.0924 | 6 | 1.6260 |
| **semantic** (paper protocol, 4 classes) | 89,514 | **+436.8%** | 0.1134 | 0.0284 | 13 | **1.3579** |
| input change mask (binary read) | 16,676 | — | — | 0.0079 | 5 | 1.4254 |
| **binary** (change only, 1024 tile) | 61,695 | **+270.0%** | 0.0991 | 0.0086 | **180** | 1.5743 |
| **crop** (256 px per component) | 27,056 | **+62.2%** | **0.3656** | 0.0103 | 13 | **1.4944** |
| crop, **ViT-H** encoder | 41,025 | +146.0% | 0.2876 | 0.0145 | 27 | 1.5376 |

**Every refiner in section U now fails on the change class — but this one fails
in the opposite direction, and that is the finding.** U1 and U1b removed change
(−11.7%, −22.2%); Phoenix adds it (+62% at best, +437% at worst). Iteration 8
explained the removals by the change class being *barely won* (median margin
0.155), and that explanation was right but incomplete: it accounts for anything
that lets neighbours out-vote a change pixel, and says nothing about a refiner
that does not vote at all. **Phoenix does not average, it draws.** Handed a 6 px
blob it returns the object it thinks is there, and at 10 m there is no object —
the change class is 992 components with median **6 px**, q75 **16 px**, max 832
px. The generalisation that covers both halves of section U is therefore not
"smoothing destroys change" but **anything that re-decides the change class at
neighbourhood scale swamps it, because it is 0.56% of the map in pieces smaller
than the process's own support.**

**The size breakdown is the mechanism, and it is not uniform.** Per component,
crop regime:

| input size | n | median IoU | median area ratio | px added |
| --- | --- | --- | --- | --- |
| <10 px | 650 | 0.200 | **2.33** | **+6,365** |
| 10–30 px | 210 | 0.444 | 1.08 | +1,408 |
| 30–100 px | 104 | 0.460 | 0.84 | +281 |
| 100–300 px | 26 | 0.493 | 1.30 | +3,008 |
| >300 px | 2 | 0.657 | 1.05 | +30 |

**57% of every added pixel comes from components under 10 px**, and above 30 px
the method roughly preserves area and moves the boundary instead. So Phoenix is
not indiscriminately dilating; it is doing something reasonable on the few
objects that are objects, and inventing objects around the 650 specks that are
not.

**Capacity makes it worse, which is the cleanest evidence that the scale is the
problem.** The crop arm re-run with the paper's headline **ViT-H** encoder
(`phoenix_vit_h_instseg`, 2.5 GB, 244 s over the same 992 components):
**+146.0%** change against EfficientViT-XL1's +62.2%, IoU with input **0.2876**
against 0.3656, `boundary_align` 1.5376, medseg 5 → 27 px. The size breakdown
says exactly where it went — the sub-10 px components take median area ratio
**8.00** instead of 2.33, while the ≥30 px bins are unchanged to within 0.35
(1.18/1.22/0.84 against 0.84/1.30/1.05) and their IoU is flat (0.448/0.481/0.654
against 0.460/0.493/0.657). **A stronger encoder is more confident about the
object it is hallucinating and no better on the objects that exist.** Anyone
tempted to read the alignment column as a ranking should note ViT-H "wins" it
(1.538 vs 1.494) purely by drawing bigger.

**The one positive is real, and this time the matched-area control Iteration 10
asked for was run.** Crop-regime `boundary_align` goes **1.4254 → 1.4944**.
Against:

- **1.4478** — the input change mask grown uniformly (outside distance transform,
  thresholded) to *exactly* the same 27,056 px. Area alone buys +0.022 of the
  +0.069.
- **1.0042** — the same blobs randomly relocated. The metric is doing its job;
  the input's own 1.4254 is signal, not gradient autocorrelation.

So roughly two thirds of the alignment gain is content. **Phoenix genuinely sees
10 m Sentinel-2**, which strengthens U2b rather than repeating it: SAM's
*segmentation* was already known to be real (1.277 vs a 0.998 control), and now a
refiner's *edit* beats a matched-area null. It is still not a trade to take: the
cost is +62% change pixels and agreement with its own input at **0.3656 against a
map that reproduces itself at 0.8423** — the refinement is more than twice as
large as the map's entire seed uncertainty, on an AOI where nothing can be scored.

**Two protocol notes, both of which would have produced a wrong verdict.**
(1) The paper's semantic composition assumes classes are objects. On a transition
map `merge_regions` collapses each stable class into one whole-tile "object", the
arg-max over class-probability channels then hands most of the AOI to whatever the
small confident components claim, and the map is destroyed — `edge_density` falls
69%, `boundary_align` falls to **1.3579, below every other row in this
document**. That is a protocol mismatch, not a measurement of Phoenix's
refinement, and it is why the binary and crop arms exist. (2) The first crop run
used `merge_regions` too and read **−83.9%**, which looks exactly like a U1-style
suppression result and is an artefact: on a fragmented mask that function fuses
550 blobs into 41 AOI-spanning regions, and a window centred on a sprawling
region covers a fraction of it. `--crop-merge` reproduces it; the default is raw
components.

**The product is `--regime classmap`: a transition map in, a transition map
out.** Written as `<source stem>_phoenix.tif` beside every map it was applied to,
carrying that map's codes, palette and legend so it opens in QGIS next to its
source. Only the change classes are refined, each component in its own crop
window; the stable classes are the ground and come back at IoU 0.968–0.997.
Pixels the refiner drops fall back to the stable transition for the state they
started in (`X -> Y` becomes `X -> X`) — in a transition map "nothing happened
here" is a class, not an absence, and keeping the input class instead would make
the refiner structurally incapable of ever removing change. Applied to five maps:

| map | change px | Δ | IoU w/ input | agreement |
| --- | --- | --- | --- | --- |
| `s2off_centre_m3s3_bf` merged2 | 16,676 → 27,035 | +62.1% | 0.3660 | 99.31% |
| `s2off_centre_m3s3_bf` coarse3 | 16,609 → 29,671 | +78.6% | 0.3502 | 99.24% |
| `siam_s2off_state_pre` merged2 | 10,399 → 19,465 | +87.2% | 0.3446 | 99.51% |
| `siam_s2off_state_pre` coarse3 | 9,935 → 18,463 | +85.8% | 0.3418 | 99.52% |
| `siam_s2off_state_pre` coarse3_gated | 10,674 → 21,055 | +97.3% | 0.3370 | 99.45% |

**The inflation is a property of the target, not of one map**: five maps from two
different models, change fractions 0.35–0.56%, all land in +62% to +97% with IoU
0.34–0.37. On coarse3 the rare classes move most in relative terms —
`Artificial -> Cropland` 266 → 1,102 px, `Nature -> Cropland` 34 → 246 — which is
worth noting *against* section O's result that those classes are coverable:
Phoenix does not find them, it enlarges the handful of pixels already there.

**The rasters, in one place: `data/inference/phoenix_oslo_qgis/`** — numbered so
the QGIS browser sorts them into the order you would add them (`00` backdrop,
`01`/`02` the map being refined, then each arm's change mask beside its delta,
`40` the semantic output). All 1722×1716, EPSG:32632, paletted with overviews and
a `.qml`. **Sidecars are written, never copied**: the binary change masks first
went out carrying the merged2 four-class legend, so QGIS labelled value 1
"Artificial -> Vegetation" when it means *change* — the same code→label trap as
the palette-order bug, caught by checking each raster's present values against
its own legend.

**Layers to look at rather than read.** `phoenix_delta_map.py` writes a
four-category delta raster per arm — change both maps agree on, added, removed,
stable — plus the *stretched* 8-bit composite Phoenix actually saw, under
`data/inference/phoenix_oslo_explore/`. The delta is the layer that makes the
size story visible in one glance: the crop arm keeps **70.2%** of the input
change and adds 15,347 px, the ViT-H arm keeps 77.3% and adds 28,138, the tile
arm keeps only **42.4%** and adds 54,626. `phoenix_export_panels.py` +
`phoenix_compare_template.html` build a swipe-comparison page over the six
windows of highest disagreement (same template-substitution pattern as
`umap_page_template.html`).

Deterministic: two independent full runs agree on **100.0000%** of pixels
(max |Δcode| = 0), so a single pass is the measurement here and the 3-seed rule
in `AUTORESEARCH.md` — which is about stochastic *training* — is satisfied by the
input map's floor rather than by repetition.

**Verdict: NEGATIVE for deployment, positive for the record.** If map-level
refinement is ever revisited, Iteration 8's prescription still stands and now has
a number attached: the design has to **protect the change class explicitly**, and
the place to apply a learned refiner is the ~130 components above 30 px, where it
holds area (ratio 0.84–1.30) and lifts IoU to 0.46–0.66 — not the 650 specks that
supply the damage. That is a size-gated refiner, and nothing in this project
currently needs one.

**Environment, recorded because it blocks everything torch on this machine.** The
GPU is now an **RTX Pro 6000 Blackwell (sm_120)** and the geo env's torch
(2.5.1+cu121) has no kernels for it — `torch.cuda.is_available()` returns True
and the first real op dies with "no kernel image is available for execution on
the device". The ledger's timings (S8: "one A40") predate this. Phoenix runs from
`/home/geethen.singh/.cache/phoenix-test/venv` (torch 2.11.0+cu128, arch list
through sm_120) layered over the geo env with `--system-site-packages`;
`FastGeodis` does not build and is stubbed with scipy, exact for the `lamb=0`
calls the sampler makes. **`/tmp` is 3.7 GB and was 97% full** — checkpoints do
not fit there.

### S19. Differences instead of states, and four more 10 m indices (user's proposal)

**The question.** The deployed detail tower reads every statistic three times —
at 2018, at 2024 and as their difference — so two thirds of it is a *state* read,
and a state read is what AlphaEarth already supplies. The proposal was to drop
the single-date blocks and give the tower only what moved, over the 10 m bands
and 10 m-derivable indices (no SWIR, no red edge, no 20 m resample — which also
keeps section Q's "do not extract SWIR" intact).

**What was built.** Four indices computable from B2/B3/B4/B8 alone, chosen to
span axes NDVI does not: `evi2` (greenness past NDVI's saturation), `grvi`
(green−red, goes negative on senescent and ploughed ground), `bsi` (bare/built,
the SWIR-free four-band form) and `ci` (red−blue coloration: red roofs and
iron-rich soil against grey concrete). GNDVI was considered and dropped — it is
exactly −NDWI. Then the year and channel axes were added to `S2_SUBSETS`, and
four arms cut across them.

**Radiometry, checked before trusting any difference.** A difference of two DN
composites inherits both dates' calibration, and L2A baseline 04.00 carries a
`BOA_ADD_OFFSET` of −1000 that would sit entirely in the 2024 end. It is not
present here: the four band medians agree between 2018 and 2024 to 2–6%
(blue 537/559, green 803/820, red 844/894, NIR 2637/2658). Index differences are
ratio differences and are immune to it regardless; band differences are only
usable because of this check.

**Plot metrics, 15 seeds, gate-off read, all arms in one block** (the historical
rows were fitted under a different torch, so the controls were re-run here rather
than quoted). Δ and the win count are paired seed by seed against the deployed
78:

| rung | cols | change-F1 | Δ | macro | coarse3 | artStab | **art→veg** | Δ | won |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `N_centre_m3s3_bf` (deployed) | 78 | 0.6565 ±0.0038 | — | 0.6951 | 0.5945 | 0.6451 | 0.1925 | — | — |
| `R0_deployed` | 204 | 0.6557 | −0.0008 | 0.6940 | 0.5947 | 0.6404 | 0.1932 | +0.0007 | 5/15 |
| `S_none` (no S2 at all) | 0 | 0.6564 | −0.0001 | 0.6910 | 0.5961 | 0.6433 | 0.1977 | +0.0052 | 6/15 |
| `S_2024` (single date) | 68 | 0.6546 | −0.0019 | 0.6932 | 0.5936 | 0.6416 | 0.1910 | −0.0015 | 5/15 |
| `S_nodiff` (both states) | 136 | 0.6557 | −0.0008 | 0.6935 | 0.5918 | 0.6398 | 0.1970 | +0.0045 | 6/15 |
| `X_full` (11 ch, all years) | 312 | 0.6534 | −0.0030 | 0.6919 | 0.5920 | 0.6404 | 0.1963 | +0.0038 | 4/15 |
| `N_x10_centre_m3s3_bf` | 114 | 0.6561 | −0.0004 | 0.6945 | 0.5940 | 0.6451 | 0.1923 | −0.0002 | 7/15 |
| `S_diff` | 68 | 0.6524 | −0.0041 | 0.6926 | 0.5904 | 0.6571 | **0.1790** | −0.0135 | **14/15** |
| `X_diff` (11 ch) | 104 | 0.6547 | −0.0018 | 0.6950 | 0.5934 | 0.6586 | **0.1790** | −0.0135 | **14/15** |
| `N_diff_centre_m3s3_bf` | 26 | 0.6544 | −0.0021 | 0.6959 | 0.5901 | 0.6609 | **0.1811** | −0.0114 | **14/15** |
| `N_diff10_centre_m3s3_bf` | 38 | 0.6540 | −0.0024 | 0.6938 | 0.5889 | 0.6584 | **0.1811** | −0.0114 | **13/15** |
| `N_diff10_bfstate_centre_m3s3` | 48 | 0.6561 | −0.0003 | 0.6949 | 0.5939 | 0.6573 | **0.1802** | −0.0123 | **14/15** |

**Two clean readings, and they separate the user's two proposals.**

1. **The channel axis is closed.** Adding the four indices while keeping the year
   structure does nothing: `N_x10` (114) is −0.0004 change-F1 and −0.0002 art→veg
   against the deployed 78, and `X_full` (312) is worse than the 204 it extends.
   The one place the extra channels pay is *inside* a diff-only tower —
   `X_diff` beats `S_diff` by +0.0023 change-F1 at the same art→veg — and that is
   inside the seed spread. **Four more 10 m indices are not a lever.** This is the
   same shape as section Q's band ladder: more S2 channels do not move this model.

2. **The year axis is a real lever on exactly one metric.** Every diff-only arm —
   four of them, spanning 26 to 104 columns and 7 to 11 channels — lands at
   art→veg 0.179–0.181 against 0.192–0.198 for every arm that keeps the endpoint
   states, winning 13–14 of 15 paired seeds. Stable built-up recall moves the same
   way (+0.012 to +0.016). No state-carrying arm moves either. Change-F1, macro-F1
   and coarse3-F1 tie throughout, as S18 said they must.

   **It is a trade, not a free win.** Stable *vegetation* returned as built-up
   moves the other way, +0.0023 to +0.0028, and loses 13–15 of 15 seeds. This is
   the same two-sided boundary the class-weighting section found: the year axis
   moves the *ratio* of the two built-up errors, not their sum. Balanced accuracy
   nets out at +0.0025 to +0.0044.

**Map evidence, Oslo, two disjoint 5-seed blocks.** Self-IoU floors first, as
CLAUDE.md requires — change-class IoU of each model against itself across seed
blocks:

| model | cols | self-IoU (change) | vs deployed, same block | edge density | boundary align | change px |
| --- | --- | --- | --- | --- | --- | --- |
| `s2off_centre_m3s3_bf` | 78 | 0.8361 | — | 0.0942 | 1.6250 | 17,225 |
| `s2off_diff10_centre_m3s3_bf` | 38 | 0.8351 | **0.8348** | 0.0883 | 1.6143 | 17,668 |
| `s2off_diff10_bfstate_centre_m3s3` | 48 | 0.7623 | 0.8184 | **0.0769** | 1.5876 | 15,940 |

**The pure diff-only tower reproduces the deployed map to within the floor.**
0.8348 against floors of 0.8361 and 0.8351 — it agrees with the deployed map as
well as the deployed map agrees with itself. That is a **simplification result**:
half the columns, no endpoint states, same map. It is not an improvement, and it
does cost 6% of edge density.

**Keeping built fraction as a state is where the built-up gain shows up, and it
is bought with structure.** `diff10_bfstate` puts 10% more area into stable
Artificial (1.194 M px vs 1.085 M, against a 1.8% seed-block wobble of its own),
which is the map form of the art→veg gain — but edge density falls 18.4% and
boundary alignment 0.037, and change pixels fall 7.5%. That is the smoothing
signature the standing rule names: *spatial smoothing removes change pixels
first, every time*. `s2off_slim`'s 0.0902 → 0.0788 was called visibly worse in
S18 at the same magnitude, so this fails S18's selection rule on its face.

**Verdict.** Negative for deployment; positive and specific for the record.

* Do not re-run the channel expansion as a lever. It is measured flat at 15
  seeds on the plot read and it does not appear on the map.
* `diff10_centre_m3s3_bf` (38 cols) is the arm to keep in mind if the detail
  block ever has to shrink — it is the cheapest thing found that still lands
  inside the deployed map's own reproducibility.
* The built-up finding is real but is a **ratio trade with a structure cost**,
  not the free move the conformal and CRFE routes were. If stable built-up is
  reopened, the standing better answers are still Mondrian conformal (+0.036
  free) and the CRFE gate.
* **The deployed model does not change.** CLAUDE.md settles it on the user's
  visual read, and nothing here clears the bar to reopen that.

**Environment, and a trap worth writing down.** The share is CIFS
(`//netapp3-cifs.nina.no`, krb5, `soft`). Rewriting the 57 MB feature parquet **at
its existing path** left the path permanently unreadable — `open`, `stat` and
even `unlink` all return EIO while `ls` still shows the file at the right size,
and an identical file written under any other name reads fine. It happened twice,
once via `mv` onto the name and once by writing the name directly. Hence the
rename to `s2_features_habloss_recover_10m.parquet`; the seven-channel table is
kept beside it as `..._c7_backup.parquet`, and the poisoned path is still in the
directory and cannot be removed. **Write a new filename; never rewrite a large
parquet in place here.** Torch runs from
`/home/geethen.singh/.cache/phoenix-test/venv` (the GPU note above), which needs
`PROJ_DATA`/`PROJ_LIB`/`GDAL_DATA` pointed at the geo env's `share/` or every
rasterio CRS lookup dies with "The EPSG code is unknown".

**Layers to look at rather than read: `data/inference/oslo_s19_qgis/`**, built by
`compare_maps_qgis.py` — numbered so QGIS sorts them into the order you would add
them (`00` Sentinel-2 backdrop, `10` each model's merged2 and coarse3 beside each
other, `20` a delta raster per arm). The delta is the layer that makes the
verdict visible in one glance: against the deployed map, `diff10` adds 1,792
change px and removes 1,349 (merged2 IoU 0.8348), while `diff10_bfstate` adds
only 1,117 and removes 2,402 (0.8081) — it is not relocating change, it is
deleting it, which is what the 18% edge-density drop is made of. Sidecars are
copied for the maps (each raster's own legend is the only thing that knows what
code 2 means) and written fresh for the deltas, whose four categories are not
land-cover classes and are deliberately not on the merged2 palette.

Ladder: `data/analysis_results/s2_diff10/s2off_cost_ladder.csv`. Map comparison:
`data/analysis_results/s2_diff10/s2_subset_map_comparison.csv`. Rasters:
`data/inference/s2_20260824_163001` (seeds 0–4) and `s2_20260824_163144`
(seeds 5–9); the deployed map re-run alone at `s2_20260824_170440` is
pixel-identical to its copy in the first.

## FINAL MODEL — `s2off_centre_m3s3_bf` (settled 2026-07-27)

**The search is closed.** The user selected this model on visual inspection of
the Oslo coarse3 map, after S18 established that the quantitative metrics cannot
separate the remaining candidates. Everything below this heading is the record of
how it was arrived at, not a menu.

```bash
python infer_s2.py --aois oslo --models s2off_centre_m3s3_bf --seeds 5
```

AlphaEarth two-tower, hierarchical supervision, trained with a **78-column**
Sentinel-2 detail tower as privileged information and **served AlphaEarth-only
with the detail gate off** — no Sentinel-2 is read at inference. Detail tower:
seven channels at the plot centre plus their 3×3 mean and standard deviation,
plus built fraction at five radii, for 2018 / 2024 / difference.

Gate-off plot metrics, 15 seeds — indistinguishable from the full 204-column
block on every one:

| | change-F1 | macro-F1 | coarse3-F1 | artStab | art→veg |
| --- | --- | --- | --- | --- | --- |
| **`centre_m3s3_bf` (78, final)** | **0.6557 ±0.0034** | **0.6938** | **0.5954** | **0.6421** | 0.1938 |
| `full` (204) | 0.6555 ±0.0049 | 0.6935 | 0.5953 | 0.6420 | 0.1921 |
| `baseline_aef` (no S2) | 0.6574 ±0.0049 | 0.6912 | 0.5969 | 0.6441 | 0.1989 |

### Superseded, and why they are still in the code

`mc_s2_drop0.7` (gate on at inference), `aef_builtfrac`, `baseline_aef`,
`mc_dropout_scalars` (Tessera, 35.8% coverage), `s2off_deploy` (204 cols) and
`s2off_slim` (15) remain registered so every published number stays reproducible
and so the ablation ladder still runs. **None of them is the deployed model.** In
particular `aef_builtfrac` still holds the best built-up numbers on plots
(artStab 0.6572, art→veg 0.1916) — the reason the whole two-tower apparatus is
not what fixed stable-Artificial, a calibrated NDVI built fraction was — but it
was not what the map evidence selected.

## Rules

Inherited from `AUTORESEARCH.md` and non-negotiable, because they are what makes
the existing ledger trustworthy:

1. **One hypothesis per iteration**, registered as an idea with a `desc`.
2. **3 seeds minimum before any verdict, 5 before calling a win.** Sub-1pt
   differences at 1 seed are noise.
3. **Reuse the OOF cache** for anything post-hoc.
4. **Negative results are results** — write the number and the reason.
5. **Do not redo the tested-negative list** at the foot of `TWOTOWER_RESEARCH.md`.
6. **Stop and report** when S and T are exhausted, or after three consecutive
   flat iterations — at that point the bottleneck is data, not modelling.
