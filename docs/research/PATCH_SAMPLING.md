# Patch sampling — sizing the next labelling round

The learning curves ended the modelling line and started this one: **+0.026
change-F1 per doubling of labels**, per-class curves still climbing on every
change transition, and two coarse3 classes flat at F1 ≈ 0. The question stopped
being "which architecture" and became "how many more plots, of what, found how".

This ledger entry is the pilot that answers the "how many" with a number.

## What you get

Everything lands in one run directory, `data/patches/patches_<stamp>/`:

| output | what it is |
| --- | --- |
| `p0042_coarse3.tif` | **the map** — nine-transition class raster, 500×500 px, paletted uint8 + `.qml`. One per patch. The layer to inspect visually. |
| `p0042_merged2.tif` | the four-transition read of the same forward pass. Use it for *whether* something changed, `coarse3` for *what kind*; they disagree on ~15% of change pixels by construction (S15). |
| `p0042_topchange.tif` | **navigation layer, not a product** — arg-max restricted to the six change classes. The only way to reach the classes the map never assigns (section C). Same grid, same codes, same `.qml`. |
| `patch_stats.parquet` | one row per patch: pixel count per class on both channels, entropy, novelty, timings. Everything the sizing is computed from. |
| `composition.csv` | section A — each class's share of land with a patch-level CI. |
| `sizing.csv` | section B — patches needed per class, and on which channel. |
| `patch_ranking.parquet` | section C — eligibility and the novelty/entropy score, per patch. |
| `label_points.csv` / `.parquet` | section D — stratified points for labelling, with lon/lat, the model's guess, and a `fragment` flag. |
| `meta.json`, `plan.json` | run provenance and the plan's headline numbers. |

Class codes follow the **sorted** class list, never the palette order — read the
mapping from the `.qml` sidecar. Getting this wrong once counted stable
Vegetation as change.

The patch draw itself lives one level up in `data/patches/patches.parquet` and is
independent of any run.

## The design

| stage | script | what it does |
| --- | --- | --- |
| draw | `src/global_patches.py` | equal-area random 5×5 km patches over AEF-covered land |
| map | `src/infer_patches.py` | deployed model per patch + per-patch class counts, entropy, novelty |
| size | `src/plan_patch_sampling.py` | composition, patches needed, ranking, label points |

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python; cd src
$P global_patches.py --n 100 --seed 0
$P infer_patches.py
$P plan_patch_sampling.py --draw-points
```

**Sampling frame.** Uniform in longitude and `sin(latitude)` — equal area, not
equal latitude — rejected against AlphaEarth tile coverage in *both* 2018 and
2024, Antarctica cut at −60°. AEF coverage is the operational land definition
rather than a coastline: a patch the index cannot serve for both endpoints
cannot be mapped, so including it would bias the realised sample away from the
drawn one. 1,457 of 4,096 equal-area draws fell on covered land (35.6%, against
~28.5% true land in that band — the excess is coastal tiles overshooting the
shoreline, and 13 of the 100 patches came back all-water).

**Patch geometry.** Built in each patch's own UTM zone, centre snapped to the
10 m lattice: every patch is exactly 500×500 px and exactly 25.00 km². The easy
lon/lat-box-with-`cos(lat)` version drifts in pixel count with latitude, and
pixel counts are what the whole sizing is denominated in.

**The draw is a prefix.** `--n 400 --seed 0` re-draws the same first 100 patches
followed by 300 new ones (verified). Round two extends the pilot rather than
replacing it, so the 100 mapped patches never need re-mapping.

## Cost, measured — and what a global run would take

100 patches, 21.1 M valid pixels, **717 s wall** (7.17 s/patch), one process, one
GPU. The 5-seed ensemble fits once and caches, so it is not in this budget.

| stage | total | per patch | share |
| --- | ---: | ---: | ---: |
| AlphaEarth fetch (both years) | 657 s | 6.57 s | **92%** |
| forward pass, 5 seeds | 35 s | 0.35 s | 4.9% |
| band stack + novelty + 3 raster writes | 25 s | 0.25 s | 3.5% |

The run is **entirely download-bound**. Inference is 5% of it, because the
deployed recipe is served AlphaEarth-only — no Sentinel-2 composite, no sliding
windows. Fetch time barely varies with content (median 6.4 s, range 4.5–9.7 s,
and an all-water patch still costs 5.3 s), which is the signature of per-request
overhead rather than bytes: a 500×500 window is too small to amortise the tile
query and COG open.

**Do not extrapolate the patch rate to a global map.** At patch scale throughput
is 3.0 km²/s; the same code on Oslo's 295 km² AOI fetches both years in 13.3 s =
**22.2 km²/s**, a 7.3× amortisation from asking for larger windows. A global run
would use large tiles and get the second number.

Global ice-free land ≈ 1.35 × 10⁸ km² = 1.35 × 10¹² pixels at 10 m:

| | single process | 32-way parallel |
| --- | ---: | ---: |
| fetch, at Oslo-tile rate (22.2 km²/s) | ~70 days | ~2.2 days |
| inference, at the measured 1.4 µs/px × 5 seeds | ~22 days | ~0.7 days (GPU-bound) |
| **total, stages serial** | **~3 months** | **~3 days** |
| fetch at the *patch* rate, for contrast | ~450 days | ~14 days |

So a global 10 m map is a **few days of wall clock on a modest fan-out**, and the
fan-out is on the download, not the model. Two caveats: this is one machine on
one network stream, and fetch and inference can overlap (IO vs GPU), which the
serial total above does not assume. Neither changes the conclusion — **the
constraint on this project is labelling, not compute.** Round two's ~1,250
patches are ~2.5 h of fetch.

## A. What the map puts on the ground (n = 100 patches, 2,106 km²)

Ratio estimate over patches with a between-patch SE. **The patch is the sampling
unit, not the pixel.** 250,000 pixels inside one 5×5 km patch are nothing like
250,000 independent observations, and the `n_eff` column is how badly: the
pilot's 21 M pixels are worth 159 independent draws on the majority class.
Treating pixels as independent would have understated every SE by ~400× and
produced a confidently wrong patch count.

| coarse3 class | px | share | 95% CI | patches present | patches ≥1 ha |
| --- | ---: | ---: | --- | ---: | ---: |
| Nature → Nature | 16,886,740 | 80.2% | 74.0–86.4% | 87 | 87 |
| Cropland → Cropland | 2,943,050 | 14.0% | 8.6–19.4% | 50 | 46 |
| Artificial → Artificial | 877,263 | 4.2% | 2.3–6.0% | 62 | 52 |
| **Nature → Artificial** | 115,499 | 0.55% | 0.26–0.83% | 56 | 38 |
| **Artificial → Nature** | 96,857 | 0.46% | 0.05–0.87% | 41 | 21 |
| **Cropland → Artificial** | 95,302 | 0.45% | 0.05–0.86% | 35 | 18 |
| **Nature → Cropland** | 49,297 | 0.23% | 0.07–0.40% | 35 | 25 |
| **Cropland → Nature** | 106 | 0.0005% | 0–0.001% | 5 | **0** |
| **Artificial → Cropland** | 1 | 0.000005% | 0–0% | 1 | **0** |

Total change: **1.7% of land**, which is a defensible figure for 2018→2024 and
says the map is not globally over-committing change even though individual
patches reach 19%. The stable split (14.0% cropland, 4.2% artificial) sits close
to independent global estimates, which is the only external check available.

## B. Patches needed

A labeller cannot use a class that occupies 40 scattered pixels. A patch is
*usable* for class `c` when it holds ≥ 1 ha of it, and yields at most 3 points
(points a few hundred metres apart are close to the same observation):

    yield_i(c) = min(3, floor(px_i(c) / 100))

**Predicted points are not confirmed plots.** The map's change classes run at OOF
precision 0.31–0.51, so three predicted `Nature → Artificial` points return about
1.4 confirmed ones. Sizing against the unadjusted yield is the single easiest way
to plan a round that comes up short, so every priced row below is scaled by the
class's own out-of-fold precision.

Target = **double** each change class (what the curves price at +0.026 change-F1).

| class | have | want | channel | precision | usable | plots/patch | **patches needed** | at the CI's low end |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Cropland → Artificial | 333 | 333 | argmax | 0.514 | 18/100 | 0.267 | **1,247** | 2,175 |
| Nature → Cropland | 243 | 243 | argmax | 0.335 | 25/100 | 0.221 | **1,100** | 1,702 |
| Nature → Artificial | 383 | 383 | argmax | 0.461 | 38/100 | 0.461 | **831** | 1,131 |
| Artificial → Nature | 123 | 123 | argmax | 0.308 | 21/100 | 0.176 | **700** | 1,157 |
| Cropland → Nature | 114 | 114 | topchange | *unpriced* | 31/100 | 0.890 | ≥129 | ≥183 |
| Artificial → Cropland | 46 | 46 | topchange | *unpriced* | 0/100 | 0.000 | **∞** | ∞ |

**≈1,250 patches (31,000 km²) to double every reachable change class**, binding on
`Cropland → Artificial`. About 2.5 hours of AlphaEarth fetch at the measured rate,
so the mapping is not the constraint — the labelling is. Oversample ×3 → draw
3,741, rank, keep the top 1,247.

## C. Two classes the arg-max map cannot reach — and one retrieval channel that can

`Cropland → Nature` and `Artificial → Cropland` are the two classes the learning
curves call dead (F1 0.014 and 0.000). They are also **dead on the map**: 106 and
1 pixels out of 21 million, and **zero** patches holding a labellable hectare of
either. This is the finding that would have wrecked the round: a labelling
campaign that navigates by arg-max can never be sent to look at the two classes
that most need looking at, and the pilot would have reported "no patches needed"
rather than "no route exists".

The fix is to ask a different, answerable question. Restricting the arg-max to
the six change classes — *given* that something changed here, which change is it
most like — drops the three stable classes that swamp them:

| class | arg-max px | change-restricted px | patches ≥1 ha | max posterior anywhere |
| --- | ---: | ---: | ---: | ---: |
| Cropland → Nature | 106 | **355,880** | 0 → **31** | 0.174 |
| Artificial → Cropland | 1 | 11 | 0 → **0** | **0.191** |

`Cropland → Nature` is **recovered**: 31 of 100 patches become usable and it stops
being the blocking class. This is written as `*_topchange.tif` next to each
patch's `*_coarse3.tif`, on the same grid and class codes.

`Artificial → Cropland` is **not recovered and should be treated as unreachable
from this model.** Its posterior never exceeds 0.191 anywhere in 21 M pixels and
it is the best change class in 11 of them. No ranking, threshold or oversample
fixes that — with 46 plots the model has not learned a representation of it, so
it cannot be used to find more of it. It needs an external source (targeted
de-urbanisation/abandonment sites) or it should be merged. This is a data
problem, and the same kind as the stable-Artificial one.

**The change-restricted channel is a candidate generator, not a map, and it is
deliberately left unpriced.** The arg-max rows are scaled by measured OOF
precision; there is no measured confirm rate for this channel, and inventing one
would put a fabricated number exactly where the plan is weakest. Its
`patches_needed` is a *lower* bound — what it would cost if every candidate
confirmed, which none will. **Measure it before committing anyone's time**: run
the change-restricted arg-max over the existing OOF cache and read off the rate.

## D. Ranking and the label draw

Eligibility first, ranking second. A patch that holds none of the classes in
deficit cannot supply a label for them however novel it is — ranking without that
filter selects spectacular terrain that contains nothing wanted. 50 of the 87
land patches were eligible.

Among eligible patches, the score is the mean of two percentile ranks:

* **`novelty_p90`** — 90th-percentile cosine distance from each pixel's
  AlphaEarth vector to its nearest of the 6,490 *labelled* plots. Against the
  label set, not against the rest of the sample: a patch that is unremarkable
  globally but unlike anything labelled is precisely the patch worth labelling.
  p90 rather than the mean because the pocket of unfamiliar land inside an
  otherwise ordinary patch is what is worth an afternoon.
* **`entropy_change`** — normalised nine-class entropy over the pixels the map
  calls change. Restricted to change pixels because entropy averaged over a patch
  that is 95% confidently stable measures the stable class's confidence.

Percentile ranks rather than raw values: a cosine distance and a normalised
entropy are on incomparable scales, and a raw sum would let whichever has the
wider spread decide the whole ordering. They are kept as separate columns —
entropy finds boundaries and mixed pixels, novelty finds unrepresented biomes,
and they are not the same patches.

`--draw-points` emits stratified label points (532 from the 50 eligible pilot
patches), thinned to one per 500 m cell, each on the layer its class is sized on.
`pred_class` is the model's call and a stratification label, **not** a truth
value — the precision adjustment in B is the arithmetic of expecting labellers to
overrule it. `fragment=True` marks a point from a patch holding under 1 ha of
that class: below what the sizing counts, kept because for a dead class the
fragments are the only candidates that exist, marked because a sub-hectare speck
is the case most likely to be an artefact.

## Verdicts

1. **≈1,250 patches to double every reachable change class**, binding on
   `Cropland → Artificial`; 31,000 km² and ~2.5 h of fetch. Mapping is not the
   constraint.
2. **Size on the patch, price on precision.** Both corrections are large — pixel-
   level SEs are wrong by ~400×, and precision moves the requirement by 2–3×.
3. **`Cropland → Nature` is reachable only through the change-restricted
   channel**, which moves it from 0 to 31 usable patches out of 100.
4. **`Artificial → Cropland` is unreachable from this model** at 46 plots.
   External source or merge; ranking cannot fix it.
5. **The change-restricted channel's confirm rate is unmeasured.** It is the one
   number this plan is missing, and it is cheap to get from the OOF cache.

## Before round three

Re-run `learning_curves.py` after round two lands. The saturation read is
per-class and the two regimes look different: still-climbing OOF says label more
of that class; OOF flat *with the train score also near it* says the ceiling is
the label/feature pair and the class should be merged rather than fed. Round
three's split between area estimation and further targeted labelling is that
read, not a judgement call.
