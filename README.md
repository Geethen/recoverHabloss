# recoverHabloss — design-based area estimation of land-cover change

Global 10 m land-cover **transition** estimation (2018→2024) for the HABLOSS and
RECOVER projects, focused on transitions between **Cropland**, **Artificial**, and
**Nature**. This repository holds the area-estimation methods, their tests, and the
weight-reconstruction code needed to turn the purposive expert-labelled samples into
valid design-based estimates.

The scientific hook: no current global 10 m product (1) maps these transitions at this
resolution, (2) explicitly caters for **artificial-land reversion** and **cropland
abandonment**, or (3) classifies *what is lost* under impervious-surface expansion.

---

## TL;DR of the methodological findings

1. **Area estimation = the mean estimator on a binary indicator.** Verified as an exact
   identity (agreement to `<1e-12`). Area = proportion × total area; the CI rescales with it.
   For a multiclass transition matrix, use the **vectorised proportion estimator with a
   single joint λ** so the cells stay coherent (sum to 1) — per-class λ is narrower but
   breaks row/column reconciliation to the marginals.

2. **The samples are equal-allocation, not simple-random.** Both HABLOSS (80 biome×loss
   strata) and RECOVER (40 biome×map-class strata) draw ~100 points/stratum via
   `slice_head(n=100)`. Inclusion probabilities therefore span **~10⁶×**. Unweighted counts
   are meaningless: on the real RECOVER reversion example, the naive estimate is **2,280×
   too high** (7.0 Mkm² vs 0.0031 Mkm²).

3. **The existing Olofsson implementation is correct.** For binary indicators its
   `p(1−p)/(n−1)` stratum variance is *algebraically identical* to `s²/n` (verified to
   1e-15). No bug.

4. **PPI helps only with a predictor that varies *within* strata.** Using the existing map
   class as the auxiliary gives **λ\* = 0** — the map *is* the stratification, so it carries
   zero within-stratum information and PPI correctly falls back to the classical estimator.
   AlphaEarth embeddings are a genuinely new auxiliary and *can* help — but the case rests
   on predicting reversion *within* `built_loss`-type strata, a narrower claim than "a
   better map."

5. **A predictor that is noisy where change is impossible destroys PPI's power.** Variance
   in the vast "stable" strata (each ~2.5M km²) inflates the denominator of λ\* with no
   compensating covariance and, because those strata dominate the area weights, overwhelms
   real signal in the active strata. **Calibration in the boring 99% of the map matters more
   than accuracy in the interesting 1%.** With a realistic (confident-where-impossible)
   predictor, PPI halved the CI at corr≈0.83 and cut it to ~0.14× at corr≈0.995.

6. **With the existing map, a large unlabelled sample is redundant** — its population mean is
   a *census* value already available from the pixel-count CSVs. With AlphaEarth it becomes
   **essential** (a continuous predictor's mean is not in those CSVs) and **must be
   π-weighted** — an unweighted `.mean()` reproduces the 2,280× error.

---

## Methods implemented and compared

All estimators target a proportion (× total area → km²); `Z = 1.96` for 95% intervals.

| Method | What it is | When to use | Caveats |
|---|---|---|---|
| **naive** | unweighted sample mean | never (diagnostic only) | 2,280× biased under this design |
| **Olofsson stratified** | standard design-based (Olofsson et al. 2014) | **primary estimator** | symmetric normal CI misbehaves for rare classes |
| **Hájek** | weighted ratio-of-means | when you have weights but no error matrix | no UA/PA/OA; slightly wider |
| **Difference / PPI** | `λ·mean_pop(ŷ) + weighted_mean(y−λŷ)` | with a within-stratum-varying auxiliary | needs `ŷ` beyond the stratification |
| **PPI++ (power-tuned λ)** | difference estimator with optimal λ, clipped [0,1] | default when using any model | provably never worse than classical (λ→0) |
| **Cross-prediction PPI** | K-fold ŷ so all labels serve both roles | when labels are scarce (rare cells) | folds must be **stratified**, not random |
| **Ratio estimator** | area-weighted ratio of two transition classes | **the rare-cell headline** (e.g. reversion composition) | answers a composition, not an absolute area |

Equivalence: **PPI with `lam=1` == the difference estimator == Särndal's model-assisted
GREG with an identity working model.** PPI's contribution is PPI++ power-tuning and the
"never worse than classical" guarantee. This is validated against the reference
[`ppi_py`](https://github.com/aangelopoulos/ppi_py) implementation (point estimates agree to
`0.00e+00`, weighted and unweighted; `lam=0` recovers the classical mean).

### Real-example result (RECOVER, artificial → nature reversion, global, n=2,128)

| method | proportion | 95% CI | area (Mkm²) |
|---|---|---|---|
| naive unweighted | 5.4041 % | [4.44, 6.36] | 7.00 |
| Olofsson (production script) | 0.0024 % | [0.0018, 0.0030] | 0.0031 |
| stratified s²/n | 0.0024 % | [0.0018, 0.0030] | 0.0031 |
| Hájek | 0.0024 % | [0.0016, 0.0031] | 0.0031 |
| PPI / PPI++ (existing map) | 0.0024 % | [0.0018, 0.0030] | 0.0031 (λ\*=0, no gain) |

Ratio estimator: artificial→cropland is **22.3 % [13.1, 31.4]** of all artificial reversion.

**Variance concentration:** 95 % of the total variance comes from just 3 strata
(`TrFo_built_loss`, `TeFo_built_buffer`, `TeFo_built_loss`). Adding ~100 labels to
tropical-forest `built_loss` would shrink the reversion CI more than any estimator change —
the cheapest available win is a labelling decision, not a statistical one.

---

## Proposed research directions

Ranked by leverage for *this* dataset.

### 1. Model-assisted / PPI estimation with AlphaEarth embeddings (primary)
Use embedding-based **P(transition)** as the auxiliary in a π-weighted difference/PPI
estimator. Keeps design-unbiasedness (defensible to reviewers) while narrowing intervals.
The honest, pre-registerable test: **does AlphaEarth predict reversion *within*
`built_loss`-type strata?** If yes, PPI pays; if not, λ\* → 0 tells you so for free, before
you commit to a map. **Precondition to enforce:** the predictor must be *confident (≈0)*
where transitions are structurally impossible, or it destroys λ\* (finding #5).

### 2. Estimate the transition matrix jointly, not six independent areas
Rows/columns must reconcile to the 2018 and 2024 marginals. Dirichlet-multinomial or raking
to the marginals borrows strength into the starved cells (e.g. artificial→cropland, n≈46)
from the large diagonal cells that share a row/column sum. Nearly free variance reduction.

### 3. Report asymmetry and composition, not just absolute rare-class areas
- **Loss:reversion ratios** per pair — correlated cells' design effects partly cancel, so
  the ratio CI is far tighter than either area (validated: ±9 pp on a 22 % ratio).
- **Composition of new impervious surface** — what fraction of new Artificial came from
  Nature vs Cropland. Conditions away the hardest variance and is the claim no existing
  impervious product can make. **This is the strongest-supported novelty in the data.**

### 4. Asymmetric legend to exploit both samples fully
Keep the forest/other split on the **2018** label only (what was destroyed → supports "64 %
of nature loss is non-forest, invisible to GFC/RADD"); leave 2024 Nature unsplit (RECOVER's
reversion target needs no forest call). HABLOSS drives the loss direction, RECOVER the gain
direction — directionally complementary, not a lowest-common-denominator compromise.
*Note:* the "64 %" figure and all raw-count claims must be **recomputed on π-weights** before
use — earlier unweighted versions are retracted.

### 5. Cross-prediction PPI for the rare cells
K-fold so every one of the ~6,500 labels serves both model-fitting and rectification, with no
sample-splitting cost. Folds **stratified by stratum**. This is the difference between the
reversion analysis being possible and not, given double-digit cell counts.

### 6. Right-shape intervals for rare classes
The symmetric normal CI undercovers badly for rare events (simulation: skew ≈ −1, 84 %
coverage at nominal 95 %; empirical z-quantiles −3.8/+1.3, not ±1.96). Test the
**Prediction-Powered Bootstrap** (`ptd.py`, supports `w` and `w_unlabeled`) — a bootstrap CI
does not assume symmetry and may fix both the skew and the nonuniform-sampling issue at once.

### 7. Dual-frame handling of the HABLOSS overlap
`habloss_landwater` overlaps `habloss_main` (152 duplicate coordinates). Coastal points have
higher combined inclusion probability; needs a Hartley / Lohr–Rao adjustment or coastal
change is double-counted — exactly where artificial expansion concentrates.

---

## Data provenance and the weighting recipe

`π_h = n_h / N_h`, weight `w_h = N_h / n_h`, Horvitz–Thompson.

- **N_h (stratum areas):** GEE pixel-count CSVs.
  HABLOSS → `areas_biomes_with_buffer/` (80 strata); RECOVER → `areas_recover/` (40 strata).
- **n_h + membership:** `*_cleaned.geojson` (HABLOSS) / `samples_recover_v2.shp` (RECOVER),
  joined to the consolidated label file by `PLOTID`.
- **Stratum ids:** HABLOSS `stratum = loss_num + (BIOME_NUM2−1)*10`;
  RECOVER `stratum = map_num + (BIOME_NUM2−1)*5`.

Join status against `samples_habloss_recover_for_geethen.shp`: **HABLOSS 4,364/4,364**,
**RECOVER 2,128/2,128**. Realised n_h ≈ 35–66 (designed 100) — the sample is roughly half
labelled, unevenly.

> ⚠️ The RECOVER sampling design lives in a **separate, read-only** project
> (`155020_recover/WP1/R/scripts/`). Do not write there. Weights are reconstructed into this
> repo's scratch outputs only.

### Open blockers
- **Two HABLOSS batches** (`samples_biomes_cleaned` vs `_extra_cleaned`) have different
  inclusion mechanisms; `floor(areaWt·1000)` sends small cells to π=0 (under-coverage). Must
  be weighted per-batch, not pooled.
- **Labelling shortfall** (~44 % of designed sample): confirm it was random, not systematic
  (e.g. skipping cloudy/hard plots), or it is nonresponse bias no estimator fixes.
- **Dual-frame overlap** (item 7 above).

---

## Repository layout

```
R/src/
  estimators.py            # core: stratified, Hájek, difference/PPI, optimal λ (validated vs ppi_py)
  build_weights.py         # reconstruct HABLOSS design weights from GEE areas + cleaned samples
  recover_weights.py       # reconstruct RECOVER design weights (reads 155020_recover read-only)
  apply_weights_habloss.py # π-weighted HABLOSS transition matrix
  compare_recover.py       # real-example method comparison (naive/Olofsson/Hájek/ratio)
  ppi_on_recover.py        # PPI on RECOVER with the existing map as auxiliary (shows λ*=0)
tests/
  test_vs_ppipy.py         # equivalence to reference ppi_py (point ests agree to 0.00e+00)
  test_estimators_sim.py   # known-truth simulation: coverage & bias under equal allocation
  test_mean_binary.py      # mean-of-binary == proportion identity
  worked_example.py        # binary vs probability map; classical/PPI/cross-PPI
  multiclass.py            # proportion vs one-vs-rest; coherence (sum-to-1) check
```

### Running
`estimators.py` needs only `numpy`/`scipy`. Validation against `ppi_py` needs the venv:

```bash
python R/src/compare_recover.py                    # real-example comparison
python tests/test_estimators_sim.py                # known-truth coverage
# ppi_py-dependent tests run under the ppi_py venv (numba, statsmodels)
```

## References
- Olofsson et al. (2014) *Good practices for estimating area and assessing accuracy of land change.* RSE.
- Stehman (2013) *Estimating area from an accuracy assessment error matrix.* RSE.
- Särndal, Swensson & Wretman (1992) *Model Assisted Survey Sampling.*
- Breidt & Opsomer (2017) *Model-assisted survey estimation with modern prediction techniques.* Statist. Sci.
- Angelopoulos et al. (2023) *Prediction-Powered Inference*; Zrnic & Candès *Cross-Prediction-Powered Inference* (PNAS 2024). `ppi_py`: https://github.com/aangelopoulos/ppi_py
