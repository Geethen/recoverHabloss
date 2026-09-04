# Active learning — a global acquisition surface

**Status: design above the AL0 heading, measured below it.** The first half is
the system as designed and nothing in it has been run; the numbers it quotes are
measured by *other* sections — `PATCH_SAMPLING.md`, `S2_DETAIL_RESEARCH.md`,
`CONFORMAL_TORCHCP.md` — and are what the design is built on top of. The second
half (**AL0–AL5**, from `## Measured`) is the replay lab and carries verdicts at
5 seeds x 5 spatial folds, quoted against a measured paired floor.

**Read AL0 before reading any other number here.** The paired noise floor on
change-F1 is **0.016**, and most of the differences between acquisition
strategies are smaller than that.

## The question

`PATCH_SAMPLING.md` ended with a number: **≈1,250 patches of 5×5 km to double
every reachable change class**, binding on `Cropland → Artificial`. It got there
by drawing 100 patches at random, mapping them all, and reading the yield.

That works at 100 patches. It does not work at 1,250, and the reason is not
compute — mapping 1,250 patches is 2.5 h of AlphaEarth fetch. It is that a
**random draw spends the labelling budget at the base rate**, and the base rate
for the classes in deficit is 0.45%, 0.23%, 0.0005% and 0.000005% of land. The
pilot's own table says an equal-area draw returns 0.267 usable
`Cropland → Artificial` plots per patch and 0.000 `Artificial → Cropland`.

Active learning is the question "can the draw be steered, and by how much" —
where the honest unit of the answer is **confirmed plots per patch per class**,
not accuracy.

## Two papers, and they solve different problems

| | Zaytar et al. 2024, *Bootstrapping Rare Object Detection* (arXiv:2403.02736) | Nogueira, Zaytar, Ma et al. 2025, *Core-Set Selection for Data-efficient Land Cover Segmentation* (arXiv:2505.01225 / IEEE 11368852) |
| --- | --- | --- |
| starting point | **no labels, no spatial prior** | **all labels exist** |
| goal | find instances of a rare class | prune to a subset that trains as well or better |
| mechanism | a probability surface `P` over an `H × W` grid, sampled without replacement, optionally reweighted by returned labels | six one-shot ranking criteria over the pool |
| loop | **online** — the surface moves as labels arrive | **offline** — everything is ranked before training starts |
| headline | positive rate 2% → 30%; F1 0.51 at a 300-patch budget where uniform sampling fails outright | 25% of DFC2022 beats 100% of it |

The bootstrapping paper is the architecture — it is exactly the user's idea
("initialise a grid, use existing samples to modify the probability distribution")
already written down and measured. The core-set paper is the **metric catalogue**,
and the thing that makes its criteria worth borrowing is that four of the six
need **no model in the loop**: they score a cell from imagery or from a class
histogram. That is what lets them run over a global grid.

Note the overlap: Akram Zaytar is an author on both. They are two halves of one
programme, and reading them together is the intended use.

## The one measured fact that decides the design

> `Artificial → Cropland`'s posterior **never exceeds 0.191 anywhere in 21 M
> pixels**, and it is the best change class in 11 of them. — `PATCH_SAMPLING.md` §C

Therefore **every model-in-the-loop acquisition function is blind to that class by
construction.** Entropy, margin, BALD, ensemble disagreement, conformal set size —
all of them read the posterior, and the posterior does not go there. A campaign
navigated by any of them will never be sent to look at the class that most needs
looking at, and it will report "no patches needed" rather than "no route exists".

This is the failure mode the pilot already caught once. It is why the offline,
model-free half of the system is not a warm-up stage to be skipped once a model
exists — it is the only channel that reaches the tail, and it stays load-bearing
forever.

The same fact ordered the pipeline. **The model is the expensive stage, so it
must be the last filter, not the first.** The pilot maps 100 random patches then
ranks them; this inverts to *rank 5.4 M cells cheaply, map ~3,700, keep 1,250*.

## The system

### Stage 0 — the grid

An equal-area lattice of **5 × 5 km cells over AlphaEarth-covered land**, which is
the geometry `global_patches.py` already builds (each patch in its own UTM zone,
centre snapped to the 10 m lattice, exactly 500 × 500 px and exactly 25.00 km² —
the lon/lat-box-with-`cos(lat)` version drifts in pixel count with latitude, and
pixel counts are what the sizing is denominated in).

Global ice-free land ≈ 1.35 × 10⁸ km² ⇒ **≈ 5.4 × 10⁶ cells**. One float32 surface
over that is 22 MB; a 64-D AlphaEarth mean embedding for every cell in both years
is **1.4 GB**. Both fit in memory on one machine, and that is the fact that makes
a global surface tractable at all. At 1 km cells it is 1.35 × 10⁸ cells and 35 GB
of embeddings — out-of-core, and not worth it: 5 km is already the labelling unit.

Keep the cell id stable and independent of any run, the way
`data/patches/patches.parquet` already is, so a surface computed today can be
joined to a draw made next year.

### Stage 1 — the offline surface (model-free, runs over all 5.4 M cells)

```
P₀  ∝  prior_aux  ×  rarity  ×  deficit_reach
```

* **`prior_aux`** — auxiliary Earth Engine change layers, below. This is where
  the tail classes get their only unbiased signal.
* **`rarity`** — `cluster_inverse_size` on the coarse AlphaEarth embedding
  (Zaytar eq. 1). Cluster the 5.4 M cells, make every *cluster* equally likely,
  spread each cluster's mass over its members. A cell in a thousand-member
  cluster gets a thousandth of the weight of a singleton. It is "rare things look
  different from their surroundings", it needs no labels, and it is already a
  proper distribution — `Σ Cₖ/(K·Cₖ) = 1` exactly, no renormalisation.
  **Use Bisecting KMeans, not DBSCAN** — that is the paper's measured ordering,
  and it is a large gap, not a preference. Pick `K` by `search_k_vendi`, or the
  whole clustering by inverted silhouette (`choose_clustering_by_silhouette`).
* **`deficit_reach`** — the existing 6,490 plots enter here, and this is the part
  of the user's idea that needs one adaptation to be right.

#### How the existing samples move the surface — and why the sign flips

In the paper every positive is wanted, so a hit always *raises* its
neighbourhood. Here it depends on the class, and both directions are correct:

| a confirmed plot of… | held | effect on its neighbourhood | why |
| --- | ---: | --- | --- |
| `Cropland → Artificial` | 333 | **raise** | 333 short, and conversion fronts cluster in space |
| `Artificial → Cropland` | 46 | **raise hard** | the binding scarcity |
| `Nature → Nature` | ~4,200 | **lower** | nothing left to learn there |

So the update weight is **signed, and the sign is the class's deficit sign**. That
single change reconciles the two readings of "use the existing samples": rare-class
plots are seeds (bootstrapping), abundant-class plots are exclusions (coverage).
`SamplingSurface.proximity_update` takes a signed weight for exactly this.

The coverage half is also available in a stronger form: seed `kcenter_greedy` with
the 6,490 labelled plots' embeddings and the score becomes literally *how much of
the world the existing label set fails to represent*. The pilot's `novelty_p90` is
the same family, measured per patch and referenced to the label set rather than
the pool — which is the right reference, and easy to get wrong.

### Stage 2 — the shortlist (model-in-the-loop, ~3,700 cells)

Draw an oversample from `P₀`, map it with the deployed
`s2off_centre_m3s3_bf` 5-seed ensemble at the measured 7.2 s/cell, then re-rank
on what the model says: `bald`, `conformal_set_size` under **Mondrian** LAC, and
`deficit_weighted_yield` over predicted class counts. Keep the top ~1,250.

`bald` is nearly free here — the ensemble already exists — and it is the score
that separates *the model does not know* from *this pixel is genuinely
ambiguous*. On a target whose change-F1 ceiling is set by `Cropland/Nature` label
noise, that separation is the difference between buying unmapped terrain and
buying another argument about an unlabellable boundary.

### Stage 3 — the online update

Labels come back in batches. For each confirmed plot, apply a signed
`proximity_update` (radius from the phenomenon, not from the paper — they used
200 m for cattle enclosures; a conversion front is kilometres) and a signed
`cluster_update`, renormalise, redraw the next batch. Their weight was
**`w = max(P₀)`**, the highest initial weight — start there rather than tuning.

**This stage only exists if the labelling is sequential.** If all ~1,250 patches
go out to interpreters in one batch, the online half is dead weight and Stage 1–2
is the whole system. In the paper the online variants beat their offline twins at
every budget — `Online Bisecting KMeans` 371 positives vs 58 at 950 patches — so
it is worth structuring the campaign in batches of ~100 if the labelling workflow
tolerates it. That is a decision about people, not about code.

## The acquisition metric catalogue

All implemented in [`src/acquisition.py`](../../src/acquisition.py), pure numpy,
32 tests in [`tests/test_acquisition.py`](../../tests/test_acquisition.py). Every
per-cell score points the same way: **higher = more worth labelling**.

### Model-in-the-loop — read the posterior

| metric | in English | code |
| --- | --- | --- |
| **Entropy** | How undecided the model is, 0 = certain, 1 = flat. The classic. Averaged over a patch that is 95% confidently stable it measures the stable class's confidence and nothing else — mask to change pixels first. | `normalised_entropy(probs)` |
| **Least confidence** | How much probability mass is *not* on the answer. Ignores how the remainder is spread, which is what you want when only the top call is acted on. | `least_confidence(probs)` |
| **Margin** | How close the decision was. A knife-edge between the top two classes sits on a boundary and a label there moves the boundary; mass smeared over seven classes does not. Entropy cannot tell these apart — the tests pin a pair that differ in margin at near-identical entropy. | `margin(probs)` |
| **BALD** | "Is this hard *because the model does not know*?" `H(mean_m p_m) − mean_m H(p_m)`: total uncertainty minus the part that survives a settled model. Only the reducible part is worth buying. Free here — the 5 seeds already exist. | `bald(member_probs)` |
| **Vote entropy** | BALD's blunt cousin: throw away confidence, keep which class each member picked. Coarser, but it is the one disagreement measure a badly calibrated member cannot corrupt, because calibration cannot change an arg-max. | `vote_entropy(member_probs)` |
| **Conformal set size** | Not "how spread out is the posterior" but "how many classes can I not rule out at 90%". Calibrated: size 3 means something quantitative, entropy 0.7 does not. | `conformal_set_size(probs, qhat)` |
| **Conformal class membership** | *The retrieval channel for a dead class.* "Show me every cell where `Cropland → Nature` cannot be ruled out at 90%" is answerable where "where does it win the arg-max" returns 106 pixels in 21 M. `CONFORMAL_TORCHCP.md` already has it: that class is in the 90% set 90% of the time. | `class_in_set(probs, qhat, c)` |

> **Pass a per-class `qhat`.** The marginal `SplitPredictor` reads 0.8999 coverage
> while covering `Cropland → Nature` **13%** of the time. A scalar threshold
> reintroduces exactly the blindness the channel exists to fix. Mondrian LAC is
> already the validated choice.

### Model-free — read the embedding

| metric | in English | code |
| --- | --- | --- |
| **Uniform** | Equal weight everywhere. **The baseline every method must beat**, and the pilot has already measured its per-class yield, so the comparison is free. Do not skip it. | `SamplingSurface.uniform(n)` |
| **Cluster inverse size** (Zaytar eq. 1) | Every cluster equally likely, uniform within. Rare-looking terrain gets weight proportional to how rare it looks. Needs no labels — which is why it is what you initialise with on a class the model cannot see. DBSCAN noise points are treated as singletons, not as one cluster. | `cluster_inverse_size(labels)` |
| **Feature activation** (core-set FA) | One number for "is there anything going on here". `γ = −(1−μ)·log σ`, min-max normalised and inverted. A flat, weak embedding is open water or a parking lot — consistently rejected by all six of their methods. **`μ` and `σ` must be scaled to (0,1] first**: once `σ > 1` the log turns positive and the ranking inverts against the method's own premise. Their features are ReLU-nonnegative; **AlphaEarth's are signed**, so treat FA on this modality as untested, not transferred. | `feature_activation(features, scale=True)` |
| **k-center greedy** (Sener & Savarese; the paper's `CoreSet` baseline) | Take the cell furthest from everything already selected, repeat. Pure coverage — it will never pick two lookalikes, and it will happily pick an outlier, which on satellite embeddings is sometimes an unrepresented biome and sometimes a cloud. Seed it with the 6,490 existing plots. | `kcenter_greedy(features, budget, selected)` |
| **Feature diversity** (core-set FD) | Deal the budget across clusters like cards, one each, then go round again. Every kind of place is represented before any kind is represented twice — the opposite failure mode from k-center, which happily spends the budget on outliers. FD trades the coverage guarantee for robustness to a cloud. | `feature_diversity(labels, budget, rng)` |
| **Vendi score** | **The effective number of distinct things in a set.** 500 patches of the same savanna score ~1; 500 mutually unlike patches score ~500. The only metric here that rates a **batch** rather than a cell, so it is the only one that can see redundancy *between* the cells an acquisition function picked — and therefore the natural way to compare two surfaces before either is labelled. | `vendi_score(features, q)` |
| **Within-cluster Vendi** | The criterion FD's clustering is chosen against: round-robin is only diverse if the clusters are internally boring, so all the variety sits *between* them. | `mean_within_cluster_vendi(feats, labels)` |
| **Order → score** | FD and CB return a *ranking*, not a number, and a ranking cannot be blended with entropy. `s_i = 1 − r_i/N`. | `order_to_score(order, n)` |
| **Vendi-guided K search** | How FD picks its cluster count with no labels: walk `K` up from 2, stop when mean within-cluster Vendi stops moving (<0.5% for 3 steps). An **elbow rule, not an optimum** — the criterion falls forever, so what you look for is where more clusters stop buying more homogeneity. | `search_k_vendi(feats, cluster_fn)` |
| **Silhouette (inverted)** | The bootstrapping paper's answer to "how do I set K with no labels", and **it inverts the usual rule**: they Bayesian-search to *minimise* \|silhouette\|, because that is what correlates with finding positives fast (R² = 0.93 on MOSAIKS features). Clean, well-separated clusters describe the *common* terrain; a clustering pulled apart by oddities scores badly and finds rare things faster. | `silhouette_score`, `choose_clustering_by_silhouette` |
| **Novelty to the label set** | "Unlike anything already labelled" — referenced to the labels, **not** the pool. A patch unremarkable globally but unlike every existing plot is precisely the one worth an afternoon. Reduced at p90, because the pocket of unfamiliar land inside an ordinary patch is the thing being bought. | `novelty_to_reference(feats, ref)` |

> **Shannon entropy and Vendi are the same functional.** `normalised_entropy`
> and `label_complexity` take the entropy of a **class histogram**;
> `vendi_score` takes the entropy of a **similarity spectrum** — the eigenvalues
> of the cosine matrix `K/n` — and exponentiates it, so the answer is a *count*
> of distinct things instead of a number of nats. That is why Vendi works on
> unlabelled cells, where a class histogram does not exist. `q` sets the Rényi
> order: `q=1` is Shannon, `q=∞` is `1/max λ` and is dominated by the single
> largest mode.
>
> **It is affordable globally, exactly.** The `n × n` matrix is never formed:
> `X Xᵀ/n` has the same *nonzero* eigenvalues as the `D × D` matrix `Xᵀ X/n`,
> and the remaining `n − D` are zero and contribute zero entropy. With
> AlphaEarth's `D = 64` that is a 64×64 eigendecomposition on one pass over the
> data — `O(n D²)`, seconds in BLAS for all 5.4 M cells, and **exact, not an
> approximation**. A test pins it against the explicit `n × n` route.

### Composition-driven — read what the cell would *add*

| metric | in English | code |
| --- | --- | --- |
| **Label complexity** (core-set LC) | "How many different things are in this patch?" **Pass `ignore_index`** — the paper drops "unknown"/"ignored" classes, and here that is not a formality: 13 of the 100 pilot patches were all-water, and counted as a class nodata reads as *variety*. A patch that is 100% stable Nature teaches one class in one context; a patch with four classes teaches the boundaries too, which is where the model is wrong. The only score here that prefers mixed ground. | `label_complexity(counts)` |
| **Class balance** (core-set CB) | The only score that rates a cell *against what is already picked*: take the cell that leaves the running class histogram flattest. Chases whichever class is currently scarcest with no hand-set quotas, and re-prioritises for free as the binding class changes. Pass the existing 6,490 plots as `prior`. | `class_balance_greedy(counts, budget, prior)` |
| **Deficit-weighted yield** | The pilot's arithmetic as a score. Refuses three confusions, each large: **a hectare is the unit, not a pixel** (40 scattered pixels cannot be labelled); **predicted is not confirmed** (OOF precision 0.31–0.51 moves the requirement 2–3×); **deficit, not abundance** (a cell full of stable Nature has an enormous raw yield and is worth nothing). | `deficit_weighted_yield(counts, deficit, precision)` |

### Combining them

| metric | in English | code |
| --- | --- | --- |
| **Rank mean** | Average the **percentile ranks**, not the scores. A cosine distance and a normalised entropy are on incomparable scales and a raw sum lets whichever has the wider spread decide the whole ordering. What the pilot's `novelty_p90 + entropy_change` already does. Keep the components as separate columns — entropy finds boundaries, novelty finds unrepresented biomes, and they are not the same cells. | `rank_mean(a, b, ...)` |
| **Convex blend** (core-set FA/CB) | `λ·a + (1−λ)·b`, honest when both are already on [0,1]. | `convex_blend(a, b, lam)` |
| **Cutoff hybrid** (core-set LC/FD) | Not a blend but a regime switch, and it encodes a claim: at small budgets you lack *coverage*, so lead with diversity; past some size you lack *hard examples*, so switch to complexity. The cutoff is a guess until measured. | `cutoff_hybrid(order_a, order_b, cutoff)` |
| **Softmax temperature** | The exploit/explore dial on the surface. →0 collapses to greedy top-k, which has no way to discover that the score is wrong. Sampling rather than top-k is deliberate. | `SamplingSurface.from_score(s, temperature)` |

### Online updates

| metric | in English | code |
| --- | --- | --- |
| **Proximity** | "Rare things cluster." A confirmed conversion sits at the edge of an expanding town and the next one is a kilometre away, not on another continent. Radius comes from the phenomenon. | `surface.proximity_update(i, radius_m, weight)` |
| **Cluster** | The non-spatial half: "things that *look* like this are worth looking at too" — the only channel that carries a hit across a continent. Beat proximity at every budget in the paper. | `surface.cluster_update(i, weight)` |

> Both take a **signed** weight. Negative turns the same mechanism into a
> redundancy penalty for a class that is already satisfied.

Not implemented, named so nobody re-derives them: batch-diverse selection via
determinantal point processes or facility location. At 1,250 cells drawn from
5.4 M, `sample()` without replacement plus the existing `--min-sep-m` thinning
already gives the diversity these buy. Revisit only if the draw visibly clumps.

## What the papers actually measured

Borrowing a method without its verdict is how the ledger fills up with ideas that
were already tested. These are theirs, not ours.

### Bootstrapping paper — Table 1, three simulations per method

| finding | number |
| --- | --- |
| **Bisecting KMeans is the winner**, online | 1,008 positives at a 3K budget, F1 **0.78** — best in the table |
| **DBSCAN is the loser** | 10 / 27 / 120 positives — barely above uniform, and *worse* than uniform on downstream F1 at 3K |
| **Online beats offline at every budget** | Online BKMeans 371 vs offline 58 at 950 patches |
| Proximity weighting alone | real but weaker: 42 → 157 → 422 |
| Uniform baseline | 5 / 15 / 56 |

**The single largest effect in their table is not an acquisition function.** It is
the **RCE loss** — cross-entropy on labelled pixels plus entropy *minimisation*
on unlabelled ones (Grandvalet & Bengio), `J(y,ŷ) = ρ·(CE ⊙ Y_L) + (1−ρ)·(H(ŷ) ⊙ Y_L̄)`
with `ρ` the labelled-pixel fraction. At a 300-patch budget it takes Online
BKMeans from F1 **0.01 → 0.51**, and it "consistently boosted performance across
all experiments". Uniform sampling at 300 patches fails outright with plain CE
and reaches 0.15 with RCE.

That is directly usable here: `build_unlabelled_aef.py` already samples unlabelled
AlphaEarth pixels for the Barlow term, so the pool the entropy term needs exists.
It is a **training** change, not an acquisition one, and it belongs in
`SIAMESE_RESEARCH.md` rather than this document — but on their evidence it is
worth more than the choice of acquisition function, and it should be tested first.

Feature representations they tested: **RCF** (random convolutional features,
MOSAIKS/Rolf et al.), **ColorStats**, and pre-trained **ResNet-18**. AlphaEarth
replaces all three here — but note the representation is a *tested axis*, not a
given, and RCF is the cheap one if AlphaEarth ever looks like the bottleneck.

Their online/proximity weight was **`w = max(P₀)`** — the highest initial weight.
That is the default to start from rather than a free parameter.

### Core-set paper — mIoU, 3 training runs, two architectures, three datasets

| finding | detail |
| --- | --- |
| **Label-based methods (LC, CB) were strongest** | best in 3 dataset–model combinations; "highlighting the importance of label diversity" |
| Combined (LC/FD, FA/CB) | best in 2 |
| **Image-based (FD, FA) gave "more moderate gains"** | best in 1 |
| Selection overhead | ≤ the cost of **one training epoch** on the full dataset |
| Method rankings agree | substantial Kendall-τ correlation; they agree on the *worst* examples too |
| What gets rejected | "very homogeneous scenes (large water bodies or parking lots)" |

**The result that transfers least, and matters most.** Core-set pruning helped most
on **DFC2022 — the dataset with the most label noise** (25–50% beat 100%). On the
cleaner datasets "performance continues to gradually improve as the number of
training examples increases".

This project sits on **both** sides of that line and they point opposite ways:
the change-F1 ceiling is set by `Cropland/Nature` label noise (argues for
pruning), but the learning curves have **every surviving class still climbing at
100% of the data**, at +0.026 change-F1 per doubling (argues hard against it).
**Do not prune the 6,490 plots on this paper's authority.** Use its criteria to
choose what to *acquire*, which is the direction its own §II says is the
active-learning use of core-set selection.

### The caveat that applies to the whole design

> "We find the advantage of these sampling techniques diminish as more labels
> become available, which is where other search methods can also take over."
> — bootstrapping paper, §4.3

Their regime is **zero labels**. This project has 6,490, a deployed model, and
measured per-class precision. The published 2% → 30% gain is a cold-start number
and should not be quoted as an expectation here. What survives the change of
regime is the *architecture* — a probability surface, sampled without replacement,
updated by returned labels — and the specific finding that model-free structure
reaches classes the model cannot. The magnitude has to be re-measured, which is
what the head-to-head pilot in the evaluation section is for.

Reference implementation of the core-set methods:
<https://github.com/keillernogueira/data-centric-rs-classification/>

## Auxiliary Earth Engine layers

The point of these is **not** better accuracy. It is that they are the only
change signal in the system that is **independent of the model's errors** —
`single-date-paths-need-independent-errors` (section P) is the standing negative
on adding a path whose errors correlate with the one you have. An auxiliary prior
earns its place by reaching where the posterior cannot.

Already in the repo, and the first thing to try because it is free:

| asset | in `extract_ppi_gee.py` as | what it gives |
| --- | --- | --- |
| `…/HABLOSS/crop_trend_2018_2024_v1` | `crop_trend` | cropland loss trend — the prior for `Cropland → *` |
| `…/HABLOSS/grey_trend_2018_2024_v1` | `grey_trend` | built-up trend |
| `…/HABLOSS/buildings_trend_2018_2023_v1` | `buildings_trend` | built-up trend, second source |
| `RESOLVE/ECOREGIONS/2017` | `biome_id` | 14 → 8 biome stratification, already coded |

That script already composes them into per-class destination scores on a simplex
(`yhat_nature` / `yhat_cropland` / `yhat_artificial`) and a `stratum_id` crossing
map class with biome. **That is a `prior_aux` surface, built and validated, at
whatever resolution it was exported at.** Reusing it is a day of work, not a
project.

Worth adding, ordered by what they reach that nothing else does:

| dataset | GEE id | why |
| --- | --- | --- |
| **GLAD global cropland** | `users/potapov/Global_cropland_*` (2003/2007/2011/2015/2019) | the only global layer that shows cropland **gain and abandonment** as two epochs. This is the candidate generator for `Artificial → Cropland` and `Cropland → Nature` — the two classes the model cannot reach. Start here. |
| **Dynamic World** | `GOOGLE/DYNAMICWORLD/V1` | 10 m per-pixel class probabilities, 2015→now. Between-year disagreement is a change prior at the model's own resolution, from an entirely different model. |
| **GHSL built-up surface** | `JRC/GHSL/P2023A/GHS_BUILT_S` | 1975–2030 in 5-year steps. Built-up **decrease** is the searchable half of `Artificial → *`. |
| **Hansen forest loss** | `UMD/hansen/global_forest_change_*` | loss year, 30 m. Sharpest available prior for `Nature → *`, and it dates the event, which the two-endpoint model cannot. |
| **ESA WorldCover** | `ESA/WorldCover/v100`, `v200` | 2020 and 2021 at 10 m, two epochs of the same legend. |
| **GLC_FCS30D** | `projects/sat-io/open-datasets/GLC-FCS30D/…` | annual 30 m 1985–2022; the long baseline for "has this cell ever been cropland". |
| **WSF Evolution** | `projects/sat-io/open-datasets/WSF/WSF_EVOLUTION` | settlement extent 1985–2015, independent of GHSL. |
| **GLanCE** | `projects/sat-io/open-datasets/GLANCE/…` | already wired in `build_state_labels.py`; validated against the RECOVER legend at the self-floor. |

**AlphaEarth itself is on Earth Engine** — `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`,
already used by `extract_embeddings_gee.py`. So the 5 km mean embedding for all
5.4 M cells in both years can be computed **inside EE and exported once**, rather
than fetched tile by tile. That single export is what makes Stage 1 global.

> **Two Earth Engine gotchas on that reduction, both hit while writing the
> `Artificial → Cropland` script.** First, `reduceResolution` caps at 65,536
> input pixels per output pixel and 10 m → 5 km is 250,000, so it needs two hops
> (10 m → 500 m → 5 km). Second, and the one that actually errors:
> **`mosaic()` and `mean()` drop the default projection**, so every composite is
> projection-less and `reduceResolution` refuses it —
> *"The input to reduceResolution does not have a valid default projection."*
> Reassert it with `setDefaultProjection` **at the point of use**, not at the
> source: the arithmetic in between, `ee.Image.constant` especially, drops it
> again.
>
> The alternative — sampling at `scale=5000` and taking EE's pyramid mean — is
> cheaper but **not equivalent, and worse here**: it pyramids the *embeddings*
> to 5 km and computes the cosine afterwards. Cosine is not linear, so averaging
> embeddings then scoring is not averaging scores, and a 200 m conversion inside
> a 5 km cell washes out entirely. Score at 10 m, aggregate the score.

## The `Artificial → Cropland` retrieval channel, built

[`src/gee/art_to_cropland_similarity.js`](../../src/gee/art_to_cropland_similarity.js)
is Stage 1 for the one class the model cannot reach, running entirely in Earth
Engine. Constants are regenerated by
[`src/gee/make_art_crop_constants.py`](../../src/gee/make_art_crop_constants.py)
— the RECOVER GEE assets carry no transition labels, so the 46 plots are fitted
locally and pasted in.

**Measured on the 6,490-plot labelled frame, leave-one-out:**

| representation | AUC | target plots in top 100 |
| --- | ---: | ---: |
| 2018 embedding only | 0.820 | 9 |
| 2024 embedding only | 0.806 | 11 |
| concat(2018, 2024) | 0.825 | 13 |
| **normalised difference vector** | **0.915** | **18.4 ± 2.6** |

Base rate 46/6490 = 0.71%, so ~26× enrichment — the same order as the
bootstrapping paper's 15×. `k=3` and `k=8` sub-prototypes tie within seed noise;
3 is kept so each mode is backed by ~15 plots.

This does **not** reopen the tested-negative "normalised-difference features".
That verdict is about per-band ND features as *classifier inputs*; this is the
L2-normalised difference *vector* as a *retrieval direction*.

**What the channel actually detects.** The top 100 by similarity:

| | n |
| --- | ---: |
| `Artificial → Nature` | **31** |
| `Artificial → Cropland` | 26 |
| `Artificial → Artificial` | 16 |
| `Nature → Nature` | 14 |
| everything else | 13 |

73 of 100 start from Artificial against a 17.8% base rate. **The AlphaEarth
channel finds de-urbanisation, not the destination.** Two consequences:

1. **Dynamic World supplies the destination** — `crops` vs `trees/grass/shrub` in
   2024 is the Cropland-vs-Nature call the embedding cannot make, and its errors
   are independent of AlphaEarth's, which is what section P says an auxiliary
   path has to be bought on.
2. **Send the candidates as an `Artificial → {Cropland, Nature}` discrimination
   task.** The 31 contaminants are not waste: `Artificial → Nature` holds 123
   plots and is itself in deficit at ~700 patches. One campaign, two classes.

**Rank, do not threshold.** Precision moves only 0.007 → 0.061 across the whole
usable similarity range, so no threshold buys purity at a 0.71% base rate — but a
rank still orders the search. The script exports top-N cells, not a mask.

**Still unmeasured:** this channel's confirm rate, exactly as in
`PATCH_SAMPLING.md` §C. Read it off the first 100 before committing anyone to
1,250.

## Measured — the replay lab (AL0–AL5)

Everything above this line is design. Everything below is a run with a verdict,
under `AUTORESEARCH.md` rules: 3 seeds minimum, 5 to call a win, negatives
written down with their numbers.

The harness is [`src/al_lab.py`](../../src/al_lab.py) — hide most of the 6,414
labelled plots, let a strategy ask for them back a batch at a time, refit the
deployed recipe each round, score on a held-out **spatial** region.
[`src/al_report.py`](../../src/al_report.py) reads the ledger
(`data/analysis_results/al_lab_ledger.csv`) **paired**; 32 protocol invariants in
[`tests/test_al_lab.py`](../../tests/test_al_lab.py).

**What the replay can and cannot answer.** The labelled frame is 19.4% change
against 1.7% of land — an **11.4× enrichment** — so a realised "plots per
acquisition" here does not transfer to a global draw and is not quoted as if it
did. What transfers is the **ordering** of strategies and the **shape** of the
tradeoffs between them. Absolute yields still need the head-to-head pilot in the
evaluation section.

Three ways the simulation could have been rigged, and what stops each:

| the cheat | what stops it |
| --- | --- |
| test rows near the acquired ones | folds are k-means clusters on the unit sphere, not random rows. The same `random` campaign scores 0.42 on one fold and 0.64 on another, which is how large the difference is. |
| a strategy reading a label it has not bought | scores take features and the current model only. The two `oracle_*` arms are the exception, are named as such, and a test asserts the oracle group has exactly two members. |
| fold-local class blocks placed positionally | every block goes through a name lookup (`_place`), because a rare class drops out of a small early draw and a positional write would permute its neighbours exactly where the curve is most delicate. |

### AL0 — the paired floor, before any verdict

`random` at a fixed (seed, fold) is deterministic and cannot be run against
itself, so `random_b` and `random_c` are the same strategy on a different RNG
stream. Their paired delta against `random` is the smallest gap this harness can
resolve. 5 seeds × 5 folds, 400 → 2,400 labels:

| metric | floor | null arms' sign count |
| --- | ---: | --- |
| `change_f1` | **0.0160** | 13/25, 15/25 |
| `macro_f1` | 0.0095 | 12/25, 13/25 |
| `change_macro_f1` | 0.0099 | 11/25, 14/25 |
| `natStab_as_art` | 0.0106 | 9/25, 12/25 |
| `acq_change_n` | 3.0 plots | 10/25, 13/25 |
| `vendi_state` | 0.25 places | — |

Coin flips, as a null should be. **Every number below is quoted against this
row**, the same way a map comparison is quoted against the ~0.84 self-IoU floor.
The `change_f1` floor is 0.016 and most published-looking AL gains are smaller
than that.

### AL1 — the tradeoff, and it is a real one

21 arms, 5 seeds × 5 folds × 8 rounds of 250 from a stratified 400. Paired deltas
against `random` at round 8 (2,400 labels):

| arm | change-F1 | change plots bought | Vendi (distinct places) | `natStab_as_art` |
| --- | ---: | ---: | ---: | ---: |
| `entropy` | **+0.020** | +60 | **−3.6** | −0.011 |
| `conformal_size` | +0.019 | −12 | −3.2 | −0.012 |
| `least_conf` | +0.016 | +103 | −3.2 | −0.013 |
| `margin` | +0.014 | +119 | −2.6 | −0.014 |
| `bald` | +0.005 | +110 | +1.5 | −0.017 |
| `novelty` | +0.002 | +51 | **+6.0** | **−0.020** |
| `kcenter` | −0.007 | +53 | +5.7 | −0.019 |
| `rarity` | −0.004 | −25 | +4.2 | −0.009 |
| `pred_change` | −0.003 | **+347** | −3.2 | −0.017 |
| `pred_balance` | −0.016 | +292 | −4.2 | −0.029 |
| `proto_sim` | **−0.046** | +203 | −1.6 | −0.003 |
| `oracle_change` | +0.005 | +530 | −0.5 | −0.024 |
| *floor* | *0.016* | *3* | *0.25* | *0.011* |

**No arm is good at more than one thing.** The correlation across arms between
"change plots bought" and "change-F1 gained" is negative, and the two
best-retrieval non-oracle arms are the two worst on accuracy.

Four findings, in decreasing order of how much they should change what you do:

1. **Uncertainty sampling has a cold start, and it is large.** `entropy`'s paired
   delta by budget: **−0.013** at 650 labels, −0.002 at 900, +0.008 at 1,150,
   +0.020 at 1,400, +0.026 at 1,650, **+0.027** at 1,900, +0.024 at 2,150, +0.020
   at 2,400. It is *worse than random* for the first three rounds. Leading a
   campaign with it costs real accuracy.
2. **Diversity is flat on accuracy but is the only thing that moves the map's
   stable-class errors.** `novelty` cuts `Nature → Nature` read as
   `Artificial → Artificial` by **0.020 in 21 of 25 folds** — twice the floor —
   while doing nothing measurable to change-F1. It is however a **ratio trade**:
   `natStab_as_crop` rises +0.016. Same shape as the S19 built-up finding, and it
   should be priced the same way.
3. **BALD loses to plain entropy** (+0.005 vs +0.020) at 3× the cost. Three
   members that differ only in torch init produce almost no epistemic
   disagreement on this model; BALD needs a real ensemble axis — different
   subsets, different feature blocks — before it is worth its price.
4. **`proto_sim` is the cautionary arm and the one to quote at anyone who wants
   to maximise yield.** Pointing the draw at the mean change-direction buys +203
   change plots and costs −0.046 change-F1 and **−0.213 F1 on
   `Artificial → Nature`**, because it concentrates the training set on one
   direction in delta space and starves the others. Retrieval and training-set
   quality are different objectives, and optimising the first can wreck the
   second. `oracle_change` — perfect retrieval, +530 plots — is only +0.005,
   which puts a ceiling on how much accuracy any retrieval channel can buy.

**Tested negative at 5 seeds × 5 folds, inside the floor on every metric:**
`fa` (feature activation — as the catalogue warned, AlphaEarth is signed and the
method's premise is ReLU-nonnegative features), `delta_rarity`, `bald_novelty`,
`delta_mag`, `switch_unc_div`.

### AL-T — the two map errors are one coverage gap, and it is not the one it looks like

The user reported, from inspecting the deployed map: mountains read as
`Artificial → Artificial`, wetlands read as `Cropland → Cropland`. **Neither is
visible in any aggregate in this repo** — both sides of both errors are stable
classes, so `change_f1` cannot see them by construction and `macro_f1` dilutes
them across nine. They were found by looking at a map, which is the only
instrument that has them.

[`src/extract_terrain_gee.py`](../../src/extract_terrain_gee.py) joins SRTM
slope/elevation, JRC surface-water occurrence and ESA WorldCover to all 6,414
plots; [`src/diagnose_terrain_errors.py`](../../src/diagnose_terrain_errors.py)
conditions the 5-seed OOF prediction on them.

**The mountain error is not about steepness.** For true `Nature → Nature`, the
misread-as-Artificial rate *falls* monotonically with slope and with elevation:

| slope | 0–1° | 1–3° | 3–6° | 6–12° | 12°+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| read as `Artificial → Artificial` | 0.096 | 0.087 | 0.088 | 0.069 | **0.022** |
| n | 510 | 769 | 479 | 261 | 321 |

What it rises with is **ESA WorldCover `bare`** — 0.128 against 0.041 on `tree`,
3.1× — and the model is *more* confident when it makes that error (0.508 vs
0.447). It is scree, rock and alpine desert, not gradient. Two consequences: an
uncertainty score cannot reach it, because the model is not in doubt; and **the
same confusion is firing on lowland desert**, which nobody has been looking at.

**The wetland error is seasonal water, not permanent water.** `as_crop` peaks at
JRC occurrence **5–20%** (0.345 against 0.118 at zero occurrence), and WorldCover
`wetland` reads 0.174. n is 29 and 46, so this is directional, not settled.

**Both sit in a genuine coverage gap.** Label share against global land share,
the latter from a 30,000-point equal-area draw (the `global_patches.py` frame):

| class | label share | land share | ratio |
| --- | ---: | ---: | ---: |
| `built` | 0.100 | 0.008 | **12.5×** |
| `mangrove` | 0.011 | 0.001 | 15.4× |
| `crop` | 0.302 | 0.100 | 3.0× |
| `grass` | 0.224 | 0.237 | 0.94× |
| `tree` | 0.244 | 0.346 | 0.71× |
| `wetland` | 0.009 | 0.016 | **0.57×** |
| `shrub` | 0.037 | 0.073 | 0.51× |
| **`bare`** | 0.066 | **0.174** | **0.38×** |
| `moss/lichen` | 0.005 | 0.025 | 0.21× |
| `snow/ice` | 0.000 | 0.019 | 0.008× |

The campaign has sampled where the change is — built-up fringes at 12.5× their
share of land, cropland at 3× — and has left the world's bare and sparse terrain
under-covered by **2.6×**. Slope above 12° is *over*-sampled 2.6×, which is the
direct refutation of the steepness reading.

> **Do not compute the land shares with `reduceRegion` + `frequencyHistogram`.**
> The global 1 km version returns an **empty dict rather than an error**, so it
> fails silently and produces a coverage table of NaN that looks like a missing
> cache file. Equal-area points, sampled the way the patch draw samples them.

**What this does not establish.** The 100-patch pilot has no ground truth, so the
map-side read is confounded: patch-mean slope correlates +0.37 with predicted
`Artificial → Artificial` *area*, but real built-up is also correlated with
terrain and nothing available separates the two. The plot-level OOF read is the
only one with labels behind it, and it is the one quoted above.

### AL3 — the coverage gap is real, closing it is measurable, and it buys no change-F1

AL1's "diversity does nothing for accuracy" had an obvious way to be an artefact:
the replay pool is drawn from the same distribution as the seed set, so there is
**no coverage gap for a diversity score to close** and the negative would be
guaranteed by the design rather than measured. AL3 removes that objection by
building a gap on purpose.

`--seed-mode biased_terrain` seeds the campaign from ordinary low-slope vegetated
ground only and leaves the whole awkward stratum — slope > 8°, or WorldCover
bare / snow / wetland / mangrove / moss, **1,202 of 6,414 plots (18.7%)** — in the
pool. That is exactly the terrain §AL-T identified. 5 seeds × 5 folds × 8 rounds
of 250 from a biased 800.

| arm | withheld plots recovered | `natStab_as_art` | `change_f1` | Vendi |
| --- | ---: | ---: | ---: | ---: |
| `novelty` | **+233** (25/25) | **−0.0143** | −0.012 | +5.3 |
| `kcenter` | +213 (25/25) | **−0.0145** | −0.008 | +5.0 |
| `rarity` | +142 (25/25) | −0.0083 | −0.010 | +4.1 |
| `bald` | +56 (24/25) | −0.0122 | −0.009 | +1.4 |
| `entropy` | **−67** (0/25) | −0.0070 | +0.011 | −2.9 |
| `fa` | −70 (0/25) | −0.0034 | +0.001 | −2.7 |
| *floor* | *5.4* | *0.0043* | *0.0126* | *0.15* |

Three verdicts, and the third is the one that matters:

1. **Diversity closes the gap it is aimed at, decisively.** `novelty` recovers 233
   more withheld plots than random in **25 of 25** folds, against a floor of 5.
2. **`entropy` actively avoids it** — 67 *fewer*, 0 of 25 folds. Uncertainty
   sampling does not merely fail to fix a coverage hole, it walks away from one.
   Consistent with §AL-T: the model is *confident* on bare ground when it is
   wrong there (0.508 vs 0.447), so an uncertainty score cannot see the error.
3. **Closing the gap does not buy change-F1.** Every diversity arm is negative on
   `change_f1` and inside the floor. Recovering 233 plots of the exact terrain the
   label set is missing moves the change objective by nothing measurable — and
   moves `natStab_as_art` by 0.014, three times its floor.

**So the AL1 negative is real, not an artefact.** Coverage and change-F1 are
decoupled on this target. A campaign has to decide which one it is buying,
because no acquisition function buys both.

The ratio trade from AL1 reappears and is larger here: `rarity` +0.032,
`novelty` +0.024, `kcenter` +0.021 on `natStab_as_crop`, all well outside the
0.0085 floor. **Buying bare ground costs the wetland read.** Anyone acting on the
diversity recommendation should price both errors, not the one they went looking
for.

### AL4 — batching is what makes uncertainty sampling work at all

Same total budget (2,000 acquisitions from a 400 seed set), split four ways.
Paired delta vs `random` at each design's final round:

| arm | 1 × 2000 | 4 × 500 | 8 × 250 | 20 × 100 |
| --- | ---: | ---: | ---: | ---: |
| `entropy` | **−0.003** | +0.022 | +0.020 | **+0.031** |
| `bald` | −0.010 | +0.004 | +0.005 | +0.007 |
| `novelty` | −0.011 | +0.001 | +0.002 | +0.001 |
| `proto_sim` | −0.036 | −0.040 | −0.046 | −0.036 |
| *floor* | *0.005* | *0.001* | *0.001* | *0.007* |

**One shot is worth nothing.** `entropy` in a single batch of 2,000 is −0.003 and
inside its floor; the same budget in twenty batches of 100 is **+0.031**. The
mechanism is not mysterious — a one-shot draw scores all 2,000 candidates with
the round-0 model, which was trained on 400 plots, and AL1 already showed that
that model's uncertainty is worse than random. Every re-fit buys the next batch a
better scorer.

**Model-free arms do not care.** `novelty` is flat across all four designs,
because it never reads the model. `proto_sim` is flat and bad in all four.

Control, and it is the reason the table is readable: **`random`'s own final level
is flat across the four designs** — change-F1 0.5765 / 0.5709 / 0.5731 / 0.5688,
Vendi 35.36 / 35.43 / 35.32 / 35.43. The design axis moves nothing by itself, so
the entropy column is the schedule and not the shape of the experiment.

One cost of batching, small and in the other direction: `novelty`'s Vendi falls
6.36 → 5.77 from 1 × 2000 to 20 × 100. Greedy diversity is myopic, and a single
large batch is marginally more diverse than twenty sequential ones.

> **This answers decision 1 in "Decisions this design could not make".** If the
> labelling is one-shot, do not use a model-in-the-loop acquisition function at
> all — it is worth nothing at that design and costs a mapping run. Batching is
> not a refinement of the online stage, it is the precondition for it.

### AL2 — the advantage decays as the label set grows, and we are past the peak

Fixed acquisition budget (1,000 = 4 × 250), varying the *starting* label set.
Paired delta vs `random` in `change_f1`:

| arm | 200 | 400 | 1,000 | 2,000 | 3,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `entropy` | +0.012 | **+0.020** | +0.013 | +0.008 | +0.004 |
| `bald` | +0.015 | +0.003 | −0.008 | +0.009 | +0.008 |
| `kcenter` | −0.019 | +0.003 | −0.007 | +0.002 | −0.001 |
| `novelty` | −0.016 | −0.003 | −0.007 | −0.001 | +0.001 |
| `proto_sim` | −0.046 | −0.054 | −0.051 | −0.015 | −0.015 |
| *floor* | *0.004* | *0.010* | *0.009* | *0.007* | *0.010* |

**Internal consistency check:** AL2 at seed 400 and AL1 at round 4 are the same
design point (400 + 4 × 250), and both read **+0.0202** for `entropy`. Two
independently launched runs agreeing to four decimals is the evidence that the
harness is doing what it says.

`entropy`'s advantage peaks around a 400-label start and decays monotonically
past 1,000; **at a 3,000-plot start it is +0.004 against a floor of 0.010** —
nothing. `natStab_as_art` decays the same way for every arm (`novelty` −0.019 at
200 → −0.001 at 3,000).

**The confound, stated because it changes the conclusion.** As the seed set grows
the pool shrinks — 5,130 minus the seed — so at a 3,000 start there are only
~2,100 candidates to choose from and *any* selector has less room to beat random.
The decay is therefore a mixture of "there is less left to learn" and "there is
less left to choose", and this frame cannot separate them. The real campaign has
a 5.4 M-cell pool, so only the first half of that mixture transfers.

What survives the confound: **the replay gives no evidence that any acquisition
function beats random on change-F1 from a starting set the size of ours.** That
is a weaker claim than "it will not", and the way to settle it is the head-to-head
pilot, not another replay.

### AL5 — the schedule is not the lever; the batch size is

AL1 and AL4 together imply a recipe — delay uncertainty until the model is worth
listening to — so AL5 builds it and tries to beat plain `entropy` with it.
`switch_rand_unc` is random for the first half of the campaign and `entropy` for
the second; `entropy_novelty` is `rank_mean` of the two winners on the two
different axes. 20 rounds x 100 from a stratified 400, 5 seeds x 5 folds.

| arm | `change_f1` | `macro_f1` | `natStab_as_art` | Vendi |
| --- | ---: | ---: | ---: | ---: |
| `switch_rand_unc` | **+0.0334** (23/25) | +0.0102 | −0.0046 | −1.6 |
| `entropy` | +0.0308 (22/25) | **+0.0132** (18/25) | −0.0040 | −3.7 |
| `entropy_novelty` | +0.0130 (14/25) | +0.0067 | **−0.0119** (20/25 down) | +2.1 |
| `novelty` | +0.0005 | +0.0040 | −0.0100 (17/25 down) | **+5.8** |
| *floor* | *0.0130* | *0.0107* | *0.0088* | *0.29* |

**The schedule does not beat its own best half.** `switch_rand_unc` is +0.0026
above `entropy` — a fifth of the floor. Both clear the floor comfortably against
random and neither clears it against the other; on `macro_f1` plain `entropy` is
ahead. **Verdict: negative, and it is the useful kind of negative** — the cold-start
harm AL1 measured is a property of **batch size**, not of schedule. At 20 x 100 no
single batch is large enough to do damage and there is nothing to delay; at
1 x 2000 there is only one batch and nothing to delay it *to*. Delaying only ever
mattered in between, and the cheaper fix is to make the batches smaller. Do not
build a schedule; build a smaller batch.

**The one arm that gets partial credit on both axes is the blend.**
`entropy_novelty` (`rank_mean` of the percentile ranks, not the scores) takes
**−0.0119 on `natStab_as_art` — better than `novelty` alone**, 20 of 25 folds —
and +2.1 Vendi, while holding `change_f1` at +0.0130. That last number sits
exactly on its floor, so read it as "does not lose change-F1", not as a gain.
That is the honest shape of the compromise: **about 40% of `entropy`'s change
gain surrendered to keep the map-error and coverage gains nearly whole.** If the
campaign has to serve both objectives with one draw, this is the arm; if it can
run two channels, running `entropy` and `novelty` separately is strictly better
than blending them.

Reproducibility note: AL5's `entropy` row and AL4's `20 x 100` `entropy` column
are the same design at the same seeds and read identically (+0.0308), which is
what a correctly seeded harness should do and is worth having checked once.

### AL6 — start size, not batch granularity; and the point where the replay runs out

AL2 varied the start size at 4 x 250 and AL5 varied the batching at a 400-plot
start, so the two axes were confounded and the cell that matters — fine batches
from a *large* start — was unmeasured. AL6 runs it: the same 1,000 acquisitions
in **10 batches of 100** at four start sizes, against AL2's 4 x 250.

`entropy`, paired delta in `change_f1`, floor beneath each cell:

| start | 400 | 1,000 | 2,000 | 3,000 |
| --- | ---: | ---: | ---: | ---: |
| **10 x 100** | **+0.0334** | +0.0076 | +0.0037 | +0.0095 |
| *floor* | *0.0130* | *0.0144* | *0.0042* | *0.0067* |
| **4 x 250** | +0.0202 | +0.0127 | +0.0082 | +0.0044 |
| *floor* | *0.0101* | *0.0092* | *0.0066* | *0.0098* |

**Fine batching does not rescue a large start.** At 400 it nearly doubles the
gain (+0.033 vs +0.020); from 1,000 upward both batchings sit at or inside their
floors. The decay AL2 found is driven by **start size**, and AL5's headline
+0.031 was a small-start number that does not generalise. The earlier reading was
right and the AL5 correction to it was wrong.

The same decay hits the coverage arms. `natStab_as_art` at 10 x 100:

| start | 400 | 1,000 | 2,000 | 3,000 |
| --- | ---: | ---: | ---: | ---: |
| `entropy_novelty` | −0.0189 | −0.0108 | −0.0058 | −0.0038 |
| `novelty` | −0.0172 | −0.0116 | −0.0099 | −0.0010 |
| *floor* | *0.0088* | *0.0041* | *0.0042* | *0.0045* |

#### Where the replay stops being able to answer

The pool is the frame minus the test fold minus the seed set, so **the pool
shrinks exactly as the start grows**, and the fraction of it the campaign
acquires goes up with it:

| start | pool | 1,000 acquisitions is… |
| --- | ---: | ---: |
| 400 | ~5,130 | 19% of the pool |
| 1,000 | ~4,530 | 22% |
| 2,000 | ~3,530 | 28% |
| 3,000 | ~2,130 | **47%** |

Taking 47% of a pool is taking the top half of any ranking, and ranking quality
barely matters when you take half. So the decay is a mixture of *there is less
left to learn* and *there is less left to choose*, and **this frame cannot
separate them at all** — the two move together by construction, and no arm can be
added that fixes it. That is a limit of the replay, not a finding about
acquisition.

**What this means for the two recommendations, and they come apart here.**

* For **change-F1**, both readings point the same way — the acquisition function
  either genuinely decays, or the replay cannot show it working, and neither
  licenses building one. The recommendation stands on either reading.
* For **coverage**, the decay is an artefact and should be ignored. A 2,130-plot
  pool drawn from the same enriched frame as the labels **does not contain the
  coverage gap the real world has** — §AL-T measures bare ground at 17.4% of
  global land against 6.6% of the label set, and no subset of the labelled frame
  can reproduce that deficit. The load-bearing evidence for the coverage
  recommendation is §AL-T's direct land-share measurement and AL3's constructed
  gap, **not** AL2/AL6's decay curves.

### What AL0–AL6 add up to

**How the acquisition function compares to simply labelling more** — the
learning curves price a doubling of labels at **+0.026 change-F1**, and that is
the number every acquisition result here has to be read against:

| | `entropy`, paired vs random | floor | |
| --- | ---: | ---: | --- |
| 400-plot start, 20 x 100 (AL5) | **+0.031** | 0.013 | real |
| 400-plot start, 10 x 100 (AL6) | +0.033 | 0.013 | real |
| 400-plot start, 8 x 250 (AL1) | +0.020 | 0.016 | real |
| 1,000-plot start, 10 x 100 (AL6) | +0.008 | 0.014 | inside floor |
| 2,000-plot start, 10 x 100 (AL6) | +0.004 | 0.004 | inside floor |
| 3,000-plot start, 10 x 100 (AL6) | +0.010 | 0.007 | marginal |
| 400-plot start, 1 x 2000 (AL4) | −0.003 | 0.005 | nothing |

AL6 separated the two axes that AL2 and AL5 confounded, and the answer is **start
size, not batch granularity**. From a 400-plot start the acquisition function is
worth about as much as a doubling of the label set. From 1,000 upward it is at or
inside the floor at *both* batchings. **The campaign starts from 6,414.**

So the original reading holds, with the confound now removed rather than assumed
away:

> On the change objective, getting the *number* of new labels right dominates
> getting the *choice* of them right at our operating point. A month spent
> tuning the surface and a month spent labelling are not close.

That is the negative result, and it is the most useful thing in this section
because it is the one that stops a plausible piece of work. It is also robust to
the one thing the replay cannot measure: AL6 shows the pool shrinks as the start
grows (47% of it acquired at a 3,000-plot start), so the decay is partly an
artefact — but *both* readings, real decay and artefact, say the same thing about
whether to build a posterior-driven surface, because an effect the replay cannot
demonstrate is not one to spend a month on.

**But change-F1 is not the only objective, and on the other one the ordering
reverses.** The two errors the user can see on the map are invisible to
`change_f1` by construction, and the only arms that move them are the
model-free coverage arms — `novelty` −0.020, `kcenter` −0.019 on
`natStab_as_art`, both about twice their floor, in 21 of 25 folds. Uncertainty
moves that error half as much (−0.011) and under a real coverage gap **walks away
from it** (AL3: −67 withheld plots, 0 of 25 folds).

So the two halves of the campaign are not two stages of one thing. They are two
objectives, and each has exactly one instrument:

| you want | use | do not use | price it on |
| --- | --- | --- | --- |
| change-F1 | more labels; `entropy` in **many small batches** | one-shot uncertainty (worth 0), a hand-built schedule (AL5, negative), retrieval scores | `change_f1`, paired |
| the map's stable-class errors | coverage — `novelty` / `kcenter`, terrain-stratified | uncertainty (blind: the model is *confident* when wrong on bare ground) | `natStab_as_art` **and** `natStab_as_crop` |
| both, from one draw | `entropy_novelty` — keeps the map-error and coverage gains nearly whole, surrenders ~40% of the change gain | blending when you could instead run two separate channels, which is strictly better | both, and say which you traded |
| plots of a specific rare class | the `Artificial → Cropland` retrieval channel, kept separate and small | `proto_sim` as the campaign's steer (−0.046) | confirmed plots per patch |

#### The correction to "explore until the curves saturate, then optimise complexity"

The instinct is right and the mechanism is not. Three things the replay changes:

1. **Diversity is not better than random early — it is flat everywhere on
   change-F1**, at every seed-set size from 200 to 3,000 and with or without a
   deliberate coverage gap. What is actually true is that *uncertainty is worse
   than random early*: `entropy` is −0.013 over the first batch from a 400-plot
   model. So there is nothing to lead *with* — but there is also nothing to
   build, because **AL5 built the delay and it tied**. The cold-start harm is a
   property of batch size, and the fix is a smaller batch, not a schedule.
2. **The curves have not saturated and the complexity phase has still already
   passed.** Every class is still climbing at 100% of the data, so by the stated
   rule we should still be exploring — but `entropy`'s advantage peaks at a
   400-plot start and is inside the floor by 3,000. Saturation of the *learning
   curve* and exhaustion of the *uncertainty signal* are different clocks, and on
   this target the second runs faster.
3. **Diversity earns its place on a different axis than the one it was proposed
   on.** Not as a warm-up for complexity, but as the permanent instrument for map
   quality — and permanently, because §AL-T's coverage gap is a property of where
   the campaign has been sampling, not of how many plots it has.

#### Multiple rounds — yes, and it is not optional

AL4 is unambiguous: the same 2,000 acquisitions are worth **−0.003** in one batch
and **+0.031** in twenty. If the labelling workflow cannot return batches, delete
the model-in-the-loop half of the design rather than running it once — it costs a
mapping run and buys nothing. The model-free half is indifferent to batching and
should be used either way.

#### For the global map specifically

Everything that survives is model-free and therefore runs over all 5.4 M cells
inside Earth Engine in one export: the coverage stratification, the rarity
surface, the terrain strata in §AL-T. The expensive stage — mapping an oversample
with the deployed ensemble to compute a posterior — is the half whose payoff AL2
says has already decayed at our label count. **That inverts the build order's
cost profile in our favour**: the part worth doing is the cheap part.

#### What would change these verdicts

* A **real ensemble axis** for BALD. Three members differing only in torch init
  give almost no epistemic disagreement (AL1: BALD +0.005 vs entropy +0.020).
  Members differing in feature block or in training subset might.
* The **pool-shrinkage confound**, quantified in AL6 and structurally
  unremovable in this frame: at a 3,000-plot start the campaign takes 47% of the
  pool, and ranking quality barely matters when you take half. The global pool is
  5.4 M cells and has neither problem. This is the reason the coverage
  recommendation rests on §AL-T's land-share measurement rather than on the
  decay curves.
* The **head-to-head pilot**. Nothing here measures realised confirmed plots per
  patch, which is the number the campaign is actually bought on, and the replay's
  11.4× enrichment means it cannot.

## AL7 — the labelling instrument, built

Everything above is about *which* points to send. This is the thing that sends
them, collects the answers and reads the yield — built 2026-08-25, and the piece
without which steps 7 and 8 of the build order cannot be run at all.

| | |
| --- | --- |
| [`app/label_app.html`](../../app/label_app.html) | the interpreter's app: MapLibre + Esri Wayback, one file, shareable by URL |
| [`app/apps_script/Code.gs`](../../app/apps_script/Code.gs) | the Google Sheet backend |
| [`src/build_label_batches.py`](../../src/build_label_batches.py) | ranked candidates → batches + manifest |
| [`src/label_rounds.py`](../../src/label_rounds.py) | the round back out of the sheet, and what it bought |
| [`app/README.md`](../../app/README.md) | deployment, the batch format, how to use it |
| [`src/build_batch_evidence.py`](../../src/build_batch_evidence.py) | point values + the annual timeline, baked into the batch (AL8) |
| [`app/config.js`](../../app/config.js) | deployment URLs **and the expert roster** (AL8) |
| [`tests/test_label_batches.py`](../../tests/test_label_batches.py) | 29 tests over the round-trip |
| [`tests/test_label_app.py`](../../tests/test_label_app.py) | 35 browser tests over the app |

**The measured findings above are wired into the tool rather than written down
next to it**, which is the only way a verdict survives contact with a campaign:

* **§AL4 is the batch size.** `--batch-size` defaults to 100 and there is
  deliberately no schedule parameter, because §AL5 built a schedule and it tied.
  The app is structured around finishing a batch and syncing it, not around an
  infinite queue.
* **§AL-T is the `meta` block.** Whatever the surface knew about a cell —
  terrain stratum, WorldCover class, biome — is passed through and rendered as
  *"why this point was drawn"*. The coverage gap is a property of where the
  campaign has been sampling, so the interpreter should be able to see which
  stratum they are being sent to.
* **The two channels stay in separate batches.** Not interleaved: a movement in
  `natStab_as_art` cannot be attributed to the coverage points if the batch also
  carried retrieval points. `--channel` stamps every point, and
  `label_rounds.py` reports yield per channel.
* **The posterior is hidden by default.** `prior` and `conformal_set` render
  behind a closed disclosure carrying §AL-T's warning, because the errors this
  campaign exists to fix are ones where the model is *confident and wrong*, and a
  visible posterior would launder them into the label set. A `conformal_set`
  must be built with per-class (Mondrian) thresholds — the marginal
  `SplitPredictor` reads 0.8999 coverage while covering `Cropland -> Nature` 13%
  of the time.
* **`random` is a first-class channel, not a stand-in.** The falsification test
  this document states in advance — ≥ 2× the equal-area rate on the binding
  class — needs an equal-area arm *in the same round*. `label_rounds.py` reports
  the enrichment as **unavailable** rather than as 1.0 when there are no `random`
  rows: a missing control is not a passing control.

**Calibration is part of the instrument, not a nicety.** Every interpreter works
a batch of points with known answers before their first real batch
(`build_label_batches.py --calibration`, 25 points, reference drawn from the
existing labelled set). The app never shows the reference before the call, and
`label_rounds.py` reports agreement per labeller *and the confusion pairs*. The
pair matters more than the rate: one person calling long fallow `Cropland` three
times out of twenty is a ten-minute briefing, and the same headline number made
of scattered singletons is a different problem. Without this, the first sign that
three people labelled to three different standards is the agreement number at the
end of the round — at which point the ceiling this document keeps citing has
already been bought.

**The two-endpoint legend cannot express what interpreters will see, so the app
records the rest separately.** Wayback shows the whole sequence, so a labeller
will find ground cleared in 2020 and regrown by 2024. The honest transition label
is `Nature -> Nature`, and it throws the observation away. Two fields catch it: a
`transient_change` flag, and an approximate **change year** offered on any change
call. The year is nearly free — the interpreter has already stepped through the
dates to make the call — and it is the only thing in the campaign that could ever
support an annual model. Two endpoints is a modelling choice that can be
revisited; an unrecorded observation cannot.

**Two things the tool measures that nothing else in this document could.** Both
are cheap here and unrecoverable later:

1. **Inter-rater agreement**, from points read by more than one person. The
   ledger's standing verdict is that `Cropland`/`Nature` label noise sets the
   change-F1 ceiling; this is the campaign's only measurement of its own noise.
   Rows are keyed by `(campaign, batch_id, point_id, expert_id)`, so a re-label
   by the same person is a correction and a second person's read is kept — a
   dedupe on `(batch, point)` alone would silently delete the measurement, and
   `Code.gs` shipped exactly that fault until AL8. **`expert_id`, not the typed
   name**: see AL8.
   **Double-label ~5% of every batch.** The app enforces the one condition that
   makes the number mean anything: it will tell a second reader *that* someone
   holds a point and *who*, and will not show them the first reading. The Sheet
   backend has two endpoints for this reason — `mine` returns a labeller their
   own rows in full, `labelled` returns ids and names only. Merging them would
   turn an agreement measurement into a confirmation measurement.
2. **Seconds per point**, recorded by the app. This is what prices a round in
   interpreter-days, and the whole justification of the design is a ratio of
   compute hours to interpreter months. A synthetic round at a 76 s median
   prices 1,250 points at ~27 interpreter-hours; the real number is what decides
   whether the 1,250 sizing is affordable.

**The yield denominator is attempts, not successes.** A point returned as *cannot
interpret* consumed interpreter time and returned nothing, so it stays in the
denominator. Dividing by usable rows would flatter every channel by exactly its
own failure rate, and a channel that sends people to 50% cloud would score like
one that does not.

**What it does not do.** It does not rank. Candidates arrive already scored and
the cut preserves that order — batch 1 is ranks 1–100. Steps 3–6 (the global
grid, `prior_aux`, `build_surface.py`) are still unbuilt, and this tool is
indifferent to which of them produces the candidate table.

## AL8 — the two-expert pilot build

Built 2026-08-26, against the work plan in `app/TODO.md`. AL7 built the
instrument for one interpreter; this is what it took to make it safe for two,
plus the evidence and latency work the plan grouped behind it. Recorded here
because three of these were *silent* faults — the app looked correct while
losing the measurement — and a silent fault that has been fixed is exactly the
kind of thing that gets reintroduced.

**The annotation key did not carry the expert, and nothing could see it.**
`Code.gs` upserted on `(campaign, batch_id, point_id)`. Expert B's save
*replaced* expert A's row, so the inter-rater agreement number — the campaign's
only handle on the label noise this document says caps change-F1 — would have
been computed over nothing and reported a clean 100%. The key is now
`(campaign, batch_id, point_id, expert_id)` end to end: sheet, localStorage
namespace, outbox key, POST group, `label_rounds.py` groupby.

The reason it survived review is worth keeping: **`tests/test_label_app.py`
mocked the sheet with the key the docs describe**, not the one `Code.gs`
implemented, so the suite passed while production lost data. A Python double of
a system cannot catch that system disagreeing with its own documentation. The
tests for it now parse `Code.gs` directly, and reverting the fix turns the suite
red.

**A typed name is not an identity.** `label_rounds.py` grouped on the free-text
`labeller` field, where "Ann", "ann", "Ann " and "Anne" are four experts and the
failure is silent until the round report. The header is now a roster dropdown
fed from `config.js`; `expert_id` is stable and goes in the sheet, `labeller` is
a display name and is never keyed on. Off-roster names get a slugged id, with a
transliteration table for the letters NFKD does not decompose — without it
"Hana Ø." and "Hana" are one expert, which is the same collision reintroduced by
the escape hatch built to avoid it.

**Overlap was a checkbox.** Forgetting it in one direction produces duplicate
work; forgetting it in the other produces zero overlap, and this document asks
for ~5%. It is now a property of the batch file: `build_label_batches.py
--experts e1,e2 --double-frac 0.05` writes `primary_expert` and
`required_readers` per point, deterministically, and the app's two queues (**My
points** / **Second readings**) are decidable with no network at all. Two
experts opening a batch offline used to both see every point as available.

**Three smaller things that were quietly wrong about the imagery**, all of which
would have shown up as unexplainable disagreement rather than as errors:

* `snap to 2018 ⇆ 2024` put the *newest* release on the right, which now serves
  2025/2026 imagery. Both sides now snap to their own target, in two stages —
  release date immediately, then per-point capture date — and each swipe label
  carries its gap in years. Past ~1.5 yr the panel offers the
  `imagery date gap` flag rather than leaving the interpreter to notice.
* `imagery_a` / `imagery_b` followed the last map *click*, so an interpreter
  comparing a neighbouring field recorded that field's capture dates as the
  provenance of their call — defeating the exact purpose those columns exist
  for. They are computed at the point now; click-to-inspect is a separate,
  labelled read-out.
* The Hansen layer was pinned to the `2023_v1_11` vintage, whose `lossyear`
  stops at 23, in an app whose label window *ends in 2024*. A 2024 clearance was
  invisible with nothing on screen to say so. Moved to `2025_v1_13`, and the
  rule this establishes is that **every dataset in the UI shows its end year** —
  which also caught GHSL being a 2015→2020 layer answering a 2018→2024 question.

**`cannot interpret` was a one-way door carrying no information.** `K` sits
between `M` and `G`; a mis-key wiped a completed call with no way back. The class
pair is stashed and restored now, and the flag requires a structured
`uninterpretable_reason`. These rows never reach the training set, so a
countable cause is the only thing they can still buy — and each cause points
somewhere different: `cloud` says draw elsewhere, `no imagery at one date` says
the Wayback archive is thin there, `capture dates too far from targets` says the
label *window* is the problem rather than the points.

**Two blinding decisions, applied.** `rank`, `score` and `channel` are hidden
until the point is saved: "rank 1, uncertainty" tells the interpreter the model
finds this point hard before they have looked at it, which is the same anchoring
the collapsed posterior exists to prevent. Descriptive `meta` stays visible —
§AL-T's coverage gap is *why* those points are in the batch. And the live
change/stable tally moved from under the strip to batch completion: a running
total of your own answers in your eyeline is an argument for the next call.

**Evidence is baked, and the growing season is latitude-aware.**
`src/build_batch_evidence.py` writes point values, an annual index timeline
(NDVI/NDMI/NBR plus six raw S2 bands), ESRI annual land cover and terrain/water
into the batch JSON at build time. Hosting is static, so the whole labelling loop
has to work with Earth Engine never signed in; only the Sentinel-2 chips are
live. The one thing here that would have been wrong in a way that looks like
data: `c2c_ts_server.py`, which the S2 recipe comes from, hardcodes Jun–Sep
because it is a Europe-only tool. These points are drawn globally, and a
southern-hemisphere point composited over Jun–Sep is its **dry** season — a
dry-season NDVI series read as a growing-season one says "vegetation loss" about
a place where nothing happened. The window flips by hemisphere (and the southern
one starts in the previous calendar year), the tropics get the full year, and the
app's live chip builder mirrors the function exactly, because a divergence makes
every disagreement between the filmstrip and the chart an artefact.

**Earth Engine on the campaign's account, because the sign-in was a step
labellers could fail at.** The panel now defaults to a **service account held by
the deployment** (`eeAuthMode: 'auto'`): the Apps Script that already backs the
Sheet mints a one-hour token, the page applies it, and nobody signs in to
anything. The sign-in button survives as the fallback, and a deployment with no
service account — or one still serving an older `Code.gs` — is left exactly
where it was, silently, because a broken token broker is not something the
person reading the panel can act on.

The shape of this was forced, not chosen. **A private key cannot go in the
page**: `ee.data.authenticateViaPrivateKey` opens with `if ("window" in t) throw
Error("Use of private key authentication in the browser is insecure…")`, so the
SDK refuses the obvious implementation outright. The key therefore lives in
Script Properties and the browser only ever holds a token that expires within
the hour. Three more facts came off the same build and each one is a bug if you
get it wrong: `setAuthToken`'s **seventh** argument, `updateAuthLibrary`, must
be `false` — that is what skips `Aj`, the Identity Services loader, and lets the
whole path run without Google's sign-in library on the page at all; refresh is
the **SDK's** job once you hand it `setAuthTokenRefresher`, since applying a
token schedules `setTimeout(Cj, expires_in*1000*.81)` itself, so the hour
boundary needs no timer of ours; and `Cj` **refuses to refresh while the client
id is null** (`wj&&xj`), so `setAuthToken`'s first argument gets a placeholder
that is never sent anywhere. Pressing the button to use your own account after
that has to *clear* the refresher first, because `Aj` installs the GIS one only
when there is none (`wj||=`) — otherwise the new session quietly keeps renewing
itself off the broker.

The cost is a scope question, not a convenience one, and it is stated in
`app/README.md` where whoever deploys this will read it: an "Anyone" web app
plus a token that ships in `config.js` means anyone you gave the app to can mint
Earth Engine tokens for the project. `roles/earthengine.writer` — viewer cannot
mint map tiles, see below — and a set `SUBMIT_TOKEN` are what bound it, and
writer is wider than the app uses: a custom role of
`earthengine.computations.create` + `earthengine.maps.create` +
`serviceusage.services.use` is the tight version.

**Why the evidence sits apart from the model hint, in one line on screen.**
Section P of `TWOTOWER_RESEARCH.md`: an auxiliary path is bought on **error
independence**, not accuracy. Dynamic World, WorldCover, Hansen and the S2
series fail in ways unrelated to how the deployed two-tower fails. The posterior
does not — it is the thing being corrected.

**Latency.** Points are 5 km cells drawn globally, so consecutive points share no
tiles and every Enter was a cold fetch of everything. The worst single case was
`snap`: up to ~50 sequential HTTP round trips behind one button press, from a
metadata cache keyed on the rounded map zoom (so any wheel nudge invalidated it),
a 13-hop sequential fallback walk, A-then-B awaited in series, and the whole
thing fired twice because `applyRelease` and `applyReleaseB` each triggered it.
Now: cache keyed on (release, point), concurrent band probes, `Promise.all`,
coalesced on a microtask. Elsewhere — `Save & Next` no longer touches
localStorage synchronously except for the outbox (the durability guarantee), the
strip is patched rather than rebuilt, the next one to two outstanding points are
warmed at idle, MapLibre is self-hosted, and the sync debounce went from 1.2 s to
10 s because every write takes one global Apps Script lock and a 100-point batch
was ~100 lock-taking POSTs. `?debug=1` prints p50/p95 point-ready and
save-acknowledgement times, the evidence cache-hit rate and dropped stale
requests.

**What running the evidence builder against Earth Engine for the first time
found.** The whole of the above was written and tested against a synthetic
fixture. Pointing it at the real thing found four faults, three of them of the
kind this section keeps being about -- they returned *no data* rather than an
error, so a batch would have shipped looking complete:

1. ``median()`` of an EMPTY collection returns an image with **no bands**, so
   the next ``.select('B2')`` throws. Empty is routine, not exceptional: the
   southern Dec-Mar window for 2017 starts before Sentinel-2 coverage.
2. ``reduceRegions`` labels a **single-band** output by the REDUCER name
   (``first``), not by the band name -- it only uses band names from two bands
   up. Six of the sixteen point-value rows (both WorldCover vintages, the bare
   flag, Hansen, both GHSL epochs) therefore came back under a key nothing read
   and rendered as "no data".
3. Hansen ``lossyear`` and JRC surface water return **nothing at all** through
   ``reduceRegions`` unless unmasked, at any point, even where ``reduceRegion``
   on the same coordinate reports the pixel valid.
4. ``bare_flag`` answered "no" where the answer was unknown -- inverted for
   exactly the §AL-T error it exists to expose.

**And the one that is worth remembering beyond this file: Earth Engine evaluates
in TILES.** A request is costed by the tiles it must materialise, so *spatially
clustered* points are nearly free to batch and *widely-spread* points must be
mapped over independently. These points are a global equal-area draw --
consecutive points are on different continents by construction -- so batching
them is the worst case there is. Measured, on the S2 cloud-probability join:

| granularity | result |
| --- | --- |
| 100 scattered points, one request | `User memory limit exceeded` |
| 20 scattered points | `User memory limit exceeded` |
| 1 point x 9 years x 10 bands | `Computation timed out` (13 min) |
| **1 point x 1 year** | **2.4 s** |

Lowering the chunk size does not fix a problem whose cost is the spread, which
is why the first two attempts failed the same way. The timeline is therefore one
request per **(point, year)**; the points-batched path survives only for the
point-value datasets, which are plain per-pixel lookups with no collection to
reduce and are cheap to sample anywhere.

**Budget it at ~20 s per point, not 2.4.** A measured 100-point bake took **36
minutes**, against the ~2 that 900 requests x 2.4 s / 16 workers predicts: Earth
Engine rate-limits interactive ``getInfo``, so the pool buys far less than its
width. This is a build step run once per round and nobody waits on it -- but do
not size a round on the optimistic arithmetic.

**One presentation finding, from looking at it rather than testing it.** Written
flat, the point-value table is 23 rows and pushed the annual-index chart -- the
more useful of the two instruments -- off the bottom of the panel. The schema
now marks rows that collapse to one line (``pair`` renders ``a -> b``, which is
also the shape of the 2018 -> 2024 question; ``seq`` renders the whole annual
sequence on one line, because in an annual series the sequence *is* the
reading). 23 rows became 11 and the chart is above the fold.

**Still deferred, deliberately** (phase 4 of the plan): adjudication — revealing
both calls after both experts submit and storing a *third* consensus record
linked to the originals. That is the thing that actually fixes `Cropland`/
`Nature` drift, and it runs on real disagreements rather than exercises, so it
is scheduled for as soon as the pilot produces some. Also deferred: a SQLite
store with a real `UNIQUE` constraint (only if the campaign goes past two experts
or Apps Script latency is *measured* as a problem), RADD/LandTrendr/CCDC
evidence, and typography.

## AL9 — making the chips fast, and the three ways that did not work

Built 2026-08-27. The complaint was concrete: the filmstrip is slow next to the
DIST-ALERT inspector this app's evidence panel was ported from.

**The first version of this section got the reason wrong, and the correction is
the most useful thing in it.** It said the inspector is fast because
`c2c_ts_server.py` caches chips server-side — implying a technique a static
deployment cannot copy, and closing the question. The server's own docstring
says otherwise:

> Everything is file-cached, so a second visit to a pixel is instant and (for
> /ts) offline. **Nothing is computed before a click: the cache fills lazily,
> one pixel at a time.**

So the cache is not what makes it fast. **A cold pixel in the inspector costs
exactly what a cold point costs here** — it is the same Earth Engine, the same
composite, the same wait. What makes it *feel* fast is that by the time a human
looks, somebody has already paid: either they clicked that pixel before, or
`scripts/warm_ts_cache.py` walked the labelled pixels and the accuracy sample
**through the running server** ahead of the review session. That script's own
docstring is explicit that this is not free — it exists precisely because
editing a segmenter invalidates the cache and "the next click on each pixel pays
the full Earth Engine round trip again".

Two things follow, and they point in opposite directions from the original
framing:

1. **The target is right but the reason was wrong.** It is still "do not make
   the interpreter wait" rather than "be as fast as the inspector" — not because
   their technique is unavailable, but because *nobody's* first look is fast,
   theirs included. The only move anyone has is to stop it being a first look.
2. **The technique IS available, and it is the one that worked.** Paying the
   Earth Engine cost before the human looks is exactly what the prefetch below
   does, and `warm_ts_cache.py` is its precedent — the inspector even does the
   same thing one level in, `_prefetch_thumbs` warming the nine grid chips in a
   16-worker pool after a `/ts` miss. The stronger version, **baking the
   filmstrip at batch build time**, is left open at the foot of this section
   rather than dismissed.

Measured with `getThumbURL` from the Python client against `app/batches/b001.json`
points, nine chips per point, nine-way thread parallelism — which is *kinder*
than the browser, where six connections is the cap.

**Three negatives first.**

| tried | result |
| --- | --- |
| **One request for the whole filmstrip.** Nine yearly composites `translate`d side by side into one mosaic, one `getThumbURL`, sliced client-side as a CSS sprite. It works — the image is correct — and the argument was that one request beats nine against a rate-limited API. | **Negative, 4 of 5 points.** 3.1/39.4/67.1/33.5/45.2 s against 3.1/37.2/37.2/40.0/7.8 s per-year. EE parallelises nine separate requests across its own backend; a single thumbnail does not get nine times the compute. |
| **A cheaper cloud mask.** `MSK_CLDPRB` (the same s2cloudless product, already a band of the SR scene, so no join at all) and Cloud Score+ `cs_cdf` via `linkCollection`, both against the `ee.Join.saveFirst` recipe the baked timeline uses. | **Inside the noise**: 36.8 / 35.3 / 12.4 s and 36.2 / 34.5 / — on two points, in an order that rotated. The join is not the cost. |
| **Assuming request shape or recipe is the cost at all.** | Both arms of every pairing pile up at **~34-35 s**. That is a plateau, not a compute time: nine concurrent thumbnail requests from one user hit a **concurrency throttle**, and under it everything costs the same. |

**The one thing that did win: cap the scene count.** `sort('CLOUDY_PIXEL_PERCENTAGE').limit(12)`
inside the *same* season, run **first** in every pairing so it could not benefit
from the raw-tile cache an earlier arm had warmed:

| | median | wins | total, 10 points |
| --- | --- | --- | --- |
| cap 12 | **30.1 s** | **9 / 10** | **216 s** |
| uncapped | 34.3 s | 1 / 10 | 323 s |

The medians are close only because both arms hit the throttle plateau; where the
request is *not* throttled the cap is 4.1-5.5 s against 13.8-34.8 s. It is
never meaningfully worse and it is sometimes eight times better.

**What the cap costs**, because a cap is a cap and the least-cloudy *scenes* are
not the least-cloudy scenes *over the point*: mean reflectance in the 100 m plot,
capped against uncapped, over both label years and all six bands —
**median relative difference 0.002, maximum 0.018**. The picture does not change.
The **season is untouched**, which is the property §AL8 says must not drift, and
the baked chart stays an uncapped median because it is a measurement rather than
a picture. The lightbox drops the cap: an interpreter who has deliberately
enlarged a year is waiting on purpose.

**And the fix that was not about Earth Engine at all.** Four things in the app
were adding waiting to a wait that was already the ceiling:

1. **The prefetch bought the wrong half — and it is the same move as
   `warm_ts_cache.py`.** It minted the thumbnail URL for two years of the next
   two points and stopped. Minting is the 2-6 s round trip; the **GET of that
   URL** is what makes Earth Engine compute the composite, and that is the
   5-40 s. It now fetches the bytes, for **all nine years** — which the cap made
   affordable — so a point an interpreter reaches after thirty seconds of
   reading the previous one paints from browser cache. This is the whole of what
   the inspector does, at a smaller radius: pay Earth Engine before the human
   looks.
2. **The other seven years were *awaited* behind 2018 and 2024.** Prioritising
   those two is right; making the rest wait for them to fully arrive is a
   guaranteed second wave. All nine are now issued at once, those two first.
3. **Changing the vis scheme cleared the chip cache** — which is keyed by scheme
   and width already, so it threw away every image the interpreter had waited
   for and made stepping back to a scheme cost the full wait again.
4. **The strip asked for 220 px for an 88 px cell.**

**The part that needed no Earth Engine at all.** The batch already bakes six
bands of reflectance per year, for the spectral profile. Mixing them through the
current vis scheme gives every year's colour **client-side, offline, instantly**
— which is now the colour of each chart dot and the background of each chip cell
before its image lands. "Which year is different" is answered the moment the
point opens; the images are the confirmation. The consequence for the offline
path is the sharper one: *not connected* no longer replaces the filmstrip with a
sentence, because the app was throwing away a real reading of the same nine years
in order to report a missing connection.

### What else changed with it

* **The vis scheme is one setting**, and NDVI/NDMI/NBR are defined once, in
  `CHIP_INDEX`, driving the chip pixels, the dot colours and the plotted series.
  Two band lists for one word called NDVI is a disagreement waiting to be read
  as change — the same hazard as `growingSeason()`, one level down.
* **A dense series**, on demand, one Earth Engine call: every clear-sky
  observation at the point, all seasons. It is here for the Cropland / Nature
  boundary specifically — a cropped field and rough grazing have the same
  annual-composite NDVI and completely different shapes *inside* the year, which
  one composite a year cannot see by construction. It deliberately does **not**
  use the s2cloudless join: a first attempt with it took 103 s for 269 scenes,
  because the join has to materialise a pair per scene.
* **`.body` was two different things.** The map+panel layout row and the inner
  text of `details.legend` were both spelled `class="body"`, so the layout's
  `display:flex` reached into the legend and laid its three class definitions
  out as three squashed columns. "What counts as what" had been unreadable since
  the flex layout landed. The layout row is `#layout` now; a word that generic
  does not get to own a rule.
* **The class names were set in the swatch colours.** `--cropland` is `#f0a30a`,
  chosen to be told apart at 9 px on a map; as 11.5 px text on the panel's
  `#f8fafc` it is a 2:1 contrast ratio. Every place a class *name* is written now
  uses an `-ink` pair.
* **The call no longer scrolls away.** Reading evidence means scrolling, which
  took the two date pickers and the answer they produce off the top of the panel.
  Head and foot are fixed flex children with the evidence scrolling between them.
* **Datasets.** ESRI annual land cover runs to **2025**, not 2023 — the only
  annual 10 m product that answers both ends of the question, and the end-year
  rule paying off in the direction nobody watches. Added: **JRC forest cover
  2020 V3** (10 m, EUDR definition, *excludes agricultural plantations* — the
  Hansen-says-tree-cover / this-legend-says-Cropland case) as both an overlay
  and a baked value, and **Open Buildings Temporal** presence 2018 → 2023 at 4 m
  as an overlay only, because it is regional and a global draw would bake mostly
  nulls. The terrain overlay was still on the **deprecated** `COPERNICUS/DEM/GLO30`
  while the builder was already on `GLO30_2024_1` — the overlay and the baked
  `slope_deg` were reading two different assets. `EVIDENCE_VERSION` is `ev2`.
### Built: bake the filmstrip, the way `warm_ts_cache.py` does

The prefetch warms two points ahead, inside the session, and needs the
interpreter to already be moving. `warm_ts_cache.py` warms **the whole known
pixel list, once, offline, before anybody sits down** — and for this app the
known pixel list is the batch. The exact analogue is baking the filmstrip
alongside the evidence bake that already runs.

Dismissed once on size, which was not measured; the arithmetic is not
prohibitive. A 9-year strip at 176 px measured 141 KB as PNG at 220 px, so
~30-50 KB per point as WebP — **~4 MB per 100-point batch per vis scheme**, as
sidecar files rather than inside the batch JSON, which has to stay near 1 MB.
Build cost at the capped recipe is 5 s/point unthrottled and ~35 s at the
throttle plateau, so 10-60 min for 100 points: the same order as the ~36 min the
evidence bake already takes, and nobody waits on it either.

The shape it would take: bake the **default scheme only** and leave the other
six live and prefetched, because seven schemes is 30 MB and the default is what
almost every point is read in. That keeps the live path — and therefore the
"losing the chips is acceptable" property — exactly as it is, and makes the
first look at a point free rather than merely early.

**Built and run** — `src/build_batch_chips.py`, against b001. The estimates
above were pessimistic on both axes:

| | estimated | **actual, 100 points** |
| --- | --- | --- |
| size | ~40 KB/point, ~4 MB | **24 KB median, 2.24 MB** (54 KB worst) |
| time | 10-60 min | **~4 min**, 26 points/min at 8 workers |

Slicing is exact, measured on all 900 cells by finding the painted ring: median
**-0.5 px** off centre, p95 **1.5 px**, 97.4% within ±3 px. The tail is not
misalignment — it cannot be, since every cell of a sprite is offset by the same
`half` — it is bare soil matching the ring's red in SWIR false colour, confirmed
by eye on the worst point. After a bake, opening a point costs one static file
and **no Earth Engine at all**, verified in the browser with nobody signed in.

**What running it surfaced: an empty year is a BLACK chip.** 3.3% of year-cells
(30 of 900, and one point in all nine years) have no cloud-free growing-season
composite, and Earth Engine renders an empty collection as black — which reads
as a broken image, not as an absence. The same silence the "outside coverage"
rule exists to prevent, one level down. The baked timeline already knows which
years those are (it has no reflectance for them), so the cell is now hatched and
labelled **"no clear"**, the sprite skips it, and the live path does not spend a
request on it. Guarded on the timeline being present at all: with no baked
evidence the app knows nothing and must not guess.

**And here the mosaic idea is right.** One sprite per point, sliced with
`background-position`, is the same construction the live experiment above
rejected — and the rejection does not carry: it was about Earth Engine's
scheduler preferring nine parallel requests to one big one, and a static file
has no scheduler. One request is simply one request. The idea was sound and only
the setting was wrong.

Everything degrades to the live path: no bake, a scheme that was not baked, a
different chip width, an unknown `version`, a batch dropped in as a file with no
URL, a sprite that 404s. Three tests hold that (`test_a_baked_filmstrip_needs_no_earth_engine`,
`..._falls_back_to_the_live_path`, `..._for_another_scheme_or_width_is_not_used`),
because a bake that silently serves the wrong scheme is worse than no bake.

Two consequences worth keeping:

* **The default scheme only.** Seven schemes is 30 MB for six almost nobody
  switches to. The others stay live and prefetched.
* **Prefetch goes six points deep instead of two** when a bake is present. A
  static file is cheap enough to warm ahead in a way nine Earth Engine
  thumbnails never were.
* **`parseBatch` is an allow-list**, and `chips` had to be added to it. Worth
  knowing before adding any other batch-level key: an unlisted one is dropped
  silently and the feature reads as "not configured".

* **Not added, and why**: GHSL's 2025 and 2030 epochs are **extrapolated, not
  observed**. GLAD/OPERA DIST-ALERT is in Earth Engine as a 60-tile collection
  with a single unnamed band, and its baseline starts in 2022 — it cannot see
  2018-2021, and this app's own rule is that no layer goes on screen without a
  legend that has been checked against the asset. GLC_FCS30D ends in 2022.

## Scalability, costed

Measured, from `PATCH_SAMPLING.md`: the patch run is **entirely download-bound** —
92% AlphaEarth fetch, 4.9% forward pass, at 6.57 s/patch. Inference is 5% because
the deployed recipe is served AlphaEarth-only. And throughput amortises 7.3× when
you ask for bigger windows: 3.0 km²/s at patch scale, 22.2 km²/s at Oslo-AOI scale.

| stage | cells touched | unit cost | total |
| --- | ---: | --- | ---: |
| 1. auxiliary prior (EE) | 5.4 × 10⁶ | EE-side reduction, one export | hours, once |
| 1. coarse AEF embedding (EE) | 5.4 × 10⁶ × 2 yr | one export, ~1.4 GB | hours, once |
| 1. cluster + score | 5.4 × 10⁶ | numpy, in memory | minutes |
| 2. map the oversample | ~3,700 | 7.2 s/cell measured | **~7.5 h**, one process |
| 3. re-rank + draw | ~3,700 | numpy | seconds |
| — | — | — | — |
| the labelling | ~1,250 patches | **the actual constraint** | months of people |

Two structural points:

* **Everything global is model-free and runs once.** The 5.4 M-cell stage never
  touches torch and never fetches a 10 m tile. The parts that do are capped at
  the oversample, which is set by the budget, not by the planet.
* **The whole exercise is worth ~7 h of compute to steer months of interpreter
  time.** That ratio is the entire justification, and it is why it is worth
  getting the acquisition function right rather than shipping the first one.

## Evaluation — and what would falsify this

This is the weakest part of the plan and it is stated as such.

**You cannot simulate on the globe.** Zaytar et al. could report a 2% → 30%
positive rate because they held polygon labels for every boma in three scenes.
There is no equivalent here; that is the whole reason for the campaign.

**One thing can be measured with no labels at all: the Vendi score of the
drawn batch.** It is the effective number of distinct places a surface selected,
it needs only the embeddings, and it is computable for every candidate surface
before a single interpreter is asked for anything. It does not say the batch is
*useful* — a surface could maximise it by drawing 1,250 unrelated deserts — so it
is a necessary condition, not a sufficient one, and it is the cheapest way to
kill a surface that has silently collapsed onto one kind of terrain.

Three things that *can* be measured against labels, in increasing cost:

1. **Replay on the 6,490 labelled plots.** Hold out the labels, rank the plots by
   each acquisition score, and read how many rare-class plots the first *b* draws
   recover. Measures **ranking quality** and nothing else — the labelled set is
   not a random sample of the globe, so the base rate is inflated and the
   realised yield will not transfer. Cheap, and it will separate the useless
   scores from the plausible ones. Must respect **LLTO**, not `block`:
   `STATE_PRETRAIN_RESEARCH.md` has 20° blocks reading 0.024 high and *reversing*
   an ordering.
2. **The change-restricted channel's confirm rate.** Already flagged in
   `PATCH_SAMPLING.md` §C as the one number the plan is missing, and it is cheap —
   run the change-restricted arg-max over the existing OOF cache and read it off.
   Until it exists, `Cropland → Nature`'s "≥129 patches" is an unpriced lower
   bound. **Do this before anything else in this document.**
3. **A head-to-head pilot.** Draw 100 cells from `P₀` and 100 equal-area, send
   both to interpreters blind, compare confirmed plots per patch per class. This
   is the only measurement that answers the actual question, and it costs 200
   patches of labelling. It is also the only way to price the online update,
   which needs the batches to come back in sequence.

**Falsification, stated in advance:** if the acquisition surface's realised
confirmed-plot rate is not **≥ 2× the equal-area baseline** on the binding class,
it is not worth the complexity and the campaign should go back to random draws
with the pilot's sizing. The paper's own gain was ~15× on a class at a 2% base
rate; 2× on a 0.45% class is a low bar, and failing it means the auxiliary priors
are not carrying information about *this* legend.

Standing rules that apply and have burned this project before:

* **3 seeds minimum, 5 to call a win.** Verdicts here have reversed between 3 and
  5, between 5 and 15, and between seed blocks.
* **Compute the self-comparison floor first.** A 5-seed ensemble reproduces
  itself at only ~0.84 change-class IoU. Any "surface A finds different cells
  from surface B" reading is meaningless until each is compared against itself.
* **Change-pixel counts move ±5% between seed blocks.** They are a run
  fingerprint, not a result.

## Build order

| # | what | depends on | output |
| --- | --- | --- | --- |
| 0 | change-restricted confirm rate from the OOF cache | nothing | one number; unblocks §C sizing |
| 1 | `acquisition.py` + tests | — | **done** — 25 metrics, 32 tests |
| 2 | replay harness on the 6,414 plots, spatial folds | 1 | **done** — `al_lab.py`, 25 arms, AL0–AL5 below |
| 2b | terrain covariates + the two map errors | — | **done** — `extract_terrain_gee.py`, `diagnose_terrain_errors.py`, §AL-T |
| 2c | the labelling instrument — app, batches, read-back | 1 | **done** — `app/`, `build_label_batches.py`, `label_rounds.py`, §AL7 |
| 3 | global 5 km grid + EE export of coarse AEF | — | `data/grid/cells.parquet` (~1.4 GB) |
| 4 | `prior_aux` from the existing HABLOSS trend assets | 3 | one surface layer |
| 5 | `build_surface.py` — compose `P₀`, write the ranked grid | 2, 3, 4 | `P₀` over 5.4 M cells |
| 6 | oversample draw → `infer_patches.py` → re-rank | 5 | ~1,250 cells, ~7.5 h |
| 7 | head-to-head pilot, 100 vs 100 | 6, 2c | the only real verdict |
| 8 | online loop, if labelling is batched | 7, 2c | signed updates between batches |

Steps 0 and 2 are cheap and answer whether the rest is worth building. Do not
build 3–6 before 2 comes back.

**Step 2 has come back and it changes steps 4–6.** AL1 says the acquisition
function is worth ~0.02 change-F1 at best, which is barely above the 0.016 floor,
while §AL-T says the label set is under-covering bare ground 2.6× and that this
is where the map errors the user can actually see are coming from. So `prior_aux`
should carry a **terrain/land-cover stratification** — the coverage axis — rather
than only the change-trend layers it was specified with, and the surface should
be read as buying *map* quality, not *change-F1*.

## AL10 — the chips were a single colour, and the ramp was the reason

Built 2026-08-28. The complaint was again concrete, and again not what it looked
like: *why do the chips appear as a single colour?* The obvious reading is the
tint placeholder — §AL9's answer to "the chips are slow" paints each cell the
colour of that year's baked reflectance before the image lands, and with Earth
Engine not connected and no sprite deployed, that is all anyone ever sees. That
case is real and is a deployment question (`S.batchUrl` is unset for a
**dragged-in** batch, so the sprite path cannot be built; the width slider must
be exactly 640; only baked schemes have sprites).

But it is not the answer. **A quarter of the baked filmstrip really was one
flat colour**, and it was the display ramp.

### The measurement

All 900 year-cells of the `chip1` bake of b001, decoded and classified:

| | cells | reads as |
| --- | --- | --- |
| normal | 664 (74%) | fine |
| clipped **bright**, >50% of px ≥ 250 | 55 | flat cream / orange |
| clipped **dark**, >50% of px ≤ 5 | 38 | flat near-black |
| flat midtone, sd < 6 | 101 | one colour |
| fully black | 42 | 37 correctly hatched "no clear"; 5 not |

**The median cell's 98th percentile is 252 of 255.** Half of every chip in the
batch was pressed against the top of the ramp.

The cause is `min: 0, max: 3500` — one linear ramp, hard-coded in both
`COMBOS` and `CHIP_RGB`, applied to a **global** point draw. 3500 DN is 0.35
reflectance; SWIR1 over bare and arid ground and NIR over dense canopy both run
0.35–0.5, so two of three channels peg while Green (~0.08) does not, which is
exactly the cream. Water floors both at 0. Confirmed by eye: `p0005` and `p0022`
were uniform cream for nine years, `p0029`/`p0039`/`p0041` uniform black for
nine.

The five black-but-not-hatched cells are all `p0095`, whose baked B11 is 6–17 DN
— open water. The app decides "no clear" from *missing* reflectance, and
near-zero is not missing.

### What the re-bake did

Same audit, same 900 cells, after:

| | chip1, fixed 0–3500 | chip2, per-point |
| --- | --- | --- |
| normal | 664 | **809** |
| clipped bright | 55 | **3** |
| clipped dark | 38 | **21** |
| flat midtone | 101 | **29** |
| black | 42 | 38 |
| median 98th percentile | 252 / 255 | **240 / 255** |
| **reads as one flat colour** | **236 (26%)** | **91 (10%)** |

Of the 38 that are still black, **37 are the cells the app hatches as "no
clear"** — they have no cloud-free growing-season composite and the sprite skips
them. `p0095` is now legible water. The one that is left, `p0069` in 2021, is
the failure mode the scene cap was always going to have and now demonstrably
has at **1 cell in 900**: the chart's *uncapped* median found a clear pixel and
the *12-scene capped* composite did not, because the twelve least-cloudy scenes
by whole-tile cloud percentage were all cloudy over this point.

### What the ramp is now, and the two things that were tried first

Per point, measured from the point's own nine years (p2–p98 at 20 m, one extra
Earth Engine request, cached in the batch so a second scheme re-uses it), stored
in `chips.stretch`, and read by **both** sides — Python renders the sprite
through it, JavaScript paints the tint and mints the live request through it.
`tests/test_chip_ramp.py` runs the app's own `comboBounds()` in node against the
baker's `combo_bounds()` on 300 random tables, which is the §AL8 rule applied
one level down: a Python double cannot police a contract the JavaScript has not
signed.

**Tested negative: per-band bounds.** The textbook stretch, and the first thing
built. It is a decorrelation stretch, and two things were wrong with it. It
moves **hue**, and hue here is a *convention* — the tips and the legend teach
"vegetation is green in SWIR/NIR/GREEN" — so a chip that renders the convention
differently per point teaches the interpreter nothing; on the trial bake
`p0000`'s green fields came back magenta. And a narrow per-band ramp turns the
few hundred DN of ordinary atmospheric drift between two years into a full-scale
colour swing: `p0022`, an unchanging desert, cycled yellow / brown / white /
black / teal / orange / blue across nine years. **A filmstrip that manufactures
change is worse than one that is flat**, and this is the same hazard as a
per-year auto-stretch, which is why neither the bands nor the years get their
own ramp.

What ships is one affine transform applied identically to all three channels —
`lo` the darkest of the three bands' floors, `hi` the brightest of their
ceilings — which is an exposure and contrast adjustment rather than a
recolouring. On the A/B: `p0005` gains visible dark patches and vegetation
specks, `p0041` and `p0010` become legible water with current structure,
`p0022` becomes a uniform orange that is *honest* about a uniform desert, and
`p0000` is unchanged. `--stretch fixed` bakes the old ramp; a point the reduce
could not measure keeps it.

The guard is `STRETCH_MIN_SPAN = 120` DN (0.012 reflectance), grown about the
**centre** so a floored span loses contrast it never had rather than shifting
brightness. It exists because mapping four DN onto 0–255 shows sensor noise as
though it were ground texture.

### The three latency things built with it

1. **All four three-band schemes are baked**, not the default only. That call
   was made on an estimated ~40 KB per point and "30 MB for six schemes almost
   nobody switches to"; measured is 33–43 KB, so four is **15 MB per 100
   points**, ~30 min at eight workers including the ramp pass. (Sprites got
   *bigger* than `chip1`'s 24 KB, because a chip with real texture in it
   compresses worse than a flat one — the fix pays for itself in bytes.) What
   the estimate traded away is sharp: switching scheme on an unbaked one drops
   to live Earth Engine at ~30 s a point, or to **no image at all** for anyone
   not signed in. Index schemes stay live — one normalised difference through a
   ramp, they do not clip, they are cheap.
2. **The whole batch is warmed, not a window.** Warming N points ahead is a
   live-Earth-Engine compromise: each point is 5–40 s of someone's quota, so you
   buy only what the interpreter is about to reach. A baked sprite is one static
   GET of ~30 KB, and the window is then the wrong *shape* — it never covers
   stepping backwards, a point revisited after a `?` flag, or the second expert
   opening the batch cold. Four lanes at low priority, behind whatever is on
   screen. The live path keeps its window, because there it is quota rather than
   bandwidth.
3. **The dense series is baked**, `src/build_batch_dense.py`, and the note
   saying it could not be is now wrong. "Nine years of Sentinel-2 at a point is
   a few hundred rows, and a hundred points of that is a batch file nobody can
   download" was true of the batch JSON — which must stay near a megabyte — and
   stopped being the question the moment `build_batch_chips.py` established the
   **sidecar**. Measured: 298 observations, **10.5 KB**, one request, 38 s. So
   it is one static file per point, ~1 MB per batch, it works with Earth Engine
   never signed in, and it is **on by default** — a series the interpreter has
   to ask for is one they ask for *after* they have already made the call, which
   is the wrong way round for the only instrument in the panel that separates
   the two classes it exists for.

### Two things the first build got wrong, found by re-reading "still one colour"

**A re-bake is invisible.** The sprites are written to the *same paths*, so a
browser or CDN that already fetched the `chip1` pixels keeps serving them — and
from a fresh profile everything looks correct, which is why the smoke test
passed and the report came back unchanged. `chips.built` is now stamped onto the
URL as `?v=`, the way `config.js` already was. This applies to every future
re-bake, not just this one.

**And the fallback was silent.** Six conditions drop the strip to live Earth
Engine — no bake, stale `version`, no `S.batchUrl` (a dragged-in file), an
unbaked scheme, a mismatched width, a 404 — and *all six produce the same strip
of flat tints*, which is also what no bake at all produces. That is a design
value of this app inverted: the panel is built on saying what is missing rather
than looking broken, and the one part with six silent failure modes was the part
somebody had to ask about twice. `chipBakeMiss()` names the condition in the
note, and where the condition is a **stale `chipVis.w`** — remembered per
browser, so a 1280 from a previous session disables every sprite in the batch
for one person and survives a reload — it offers the width back.

The button did not work at first, and the reason is worth keeping: the note
wired its listener and then appended the Earth Engine line with `innerHTML +=`,
which **re-parses the whole subtree and discards the listeners already attached
inside it**. It rendered perfectly and did nothing. The note is now built as one
string, written once, and wired afterwards.

### "It still shows one colour" — the third pass, 2026-08-31

Reported again after the re-bake and after `chipBakeMiss`, with the observation
that it started *after stepping between points*. The whole strip path was walked
in a browser against the deployed `b001`: slow stepping over 15 points, 35 ms
skipping over 16, clicking the point dots, repeated **Defer**, and three scheme
switches fired inside a sprite's own load. **All of them end 9/9 painted** — the
generation guard and `cancelChips` are correct and there is no race here. Three
other things are.

**1. A remembered width outlives the batch it was set in.** `chipVis.w` is
persisted and the sprite path requires it to equal `chips.width_m` *exactly*, so
one nudge of the width slider disables every baked chip in every batch, for that
browser, across reloads. Measured: nudge to 720, step to the next point, reload —
still flat. §AL10 already named this as "the one that actually bites" and
answered it with a note and a *Use 640 m* button, which is the right answer for
the person reading the note and no answer at all for the person who moved that
slider four points ago. Width is a **view preference**; the bake is the
instrument. `adoptBatch` now snaps to the baked width, and moving the slider
mid-batch still drops to live Earth Engine for as long as the interpreter wants
it — what it no longer does is follow them into tomorrow.

**2. An index scheme is not a bake that failed.** NDVI / NDMI / NBR are never
baked, on purpose (above: one normalised difference through a ramp does not clip
and is cheap live). Picking one made every point ~10 s of flat tints — measured
on `p0000`: 6 of 9 images at 4 s, 9 of 9 at 10 s — under a note reading *baked
chips are not being used*. A design decision reported down the fault channel,
and the interpreter's next move is to go looking for the broken thing rather
than to wait. It now says it is always live, and that the colours under the wait
are already this point's baked index — the same number the chart plots.

**3. Some chips are flat because the ground is, and nothing said so.** This is
the residual the re-bake left: 91 of 900 cells (10%) still read as one colour,
and the per-point version of that number is what people hit. Across b001's 99
measurable points, the widest of the three bands uses this much of the shared
ramp:

| | p0022 | p0040 | p0065 | p0097 | p0045 | p0025 | median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| widest band / ramp | **3.5%** | 5.3% | 5.7% | 13.3% | 18.0% | 18.3% | 53.1% |
| measured cell sd | 2.7 | 3.5 | 3.7 | 7.3 | 9.2 | 7.9 | 21.6 |

The mechanism is the shared ramp doing exactly what it was designed to do. At
bright bare ground the three bands sit far apart — SWIR1 6591–6711, NIR
4811–4951, GREEN 2738–2874 at `p0022` — so `lo` = 2738 and `hi` = 6711 gives a
3973 DN ramp while each band's own p2–p98 spread is 140 DN. The widest band gets
9 of 255 levels and nine years bake as nine identical orange squares. That is
the correct picture of a uniform desert, and on screen it is indistinguishable
from the bake having failed.

**It is not fixed by touching the ramp, and this is the second time that has to
be said.** Per-band bounds move hue; any narrow ramp turns inter-year
atmospheric drift into full-scale colour swings; both are tested-negative above,
and a shared ramp narrowed to the *band spread* rather than the band *union*
floors two of three channels to black, which is a hue change by another route.
The information is already in the batch, so the app says it: `chipFlatNote()`
computes the ratio from `chips.stretch` at render time and prints the two
numbers. **6 of 100 points on b001 carry it.**

The three notes are now visually distinct — grey is a fault, a blue edge is
something working as designed — because a flat strip looks the same whichever it
is, which is the whole of this section three times over.

### The same failure again, one panel over: the Earth Engine overlays

Reported straight after the chips were fixed — *select an auxiliary layer and it
does not appear*. The app-side plumbing was verified correct in a browser
(`ee-active` is inserted directly below `labels`, above the basemaps and above
the Wayback raster, at 0.7 opacity, no exception), so the failure is downstream
of the mint, and it was **silent for exactly the same reason**.

`getMapId` validates the *request*. Most of what can go wrong in an Earth Engine
expression goes wrong when a **tile** is rendered — a band name that does not
match, a filter that leaves the collection empty, a memory limit, a token that
has since rotated. Those come back as an HTTP error per tile, MapLibre reports
them on `map.on('error')`, and this app's handler was:

```js
map.on('error', e => console.warn('map', e && e.error));
```

So the mint succeeded, `eeLayerNote('')` cleared the note, and the map stayed
exactly as it was. A `console.warn` is not a report.

`eeTileError()` now writes it into the layer note, and `eeTileReason()` **GETs
one tile directly** to recover Earth Engine's own message — MapLibre loads
raster tiles as images, so `AJAXError.body` is empty and the status is all it
can offer, while the tile response carries the actual sentence. 401/403 and 429
get their own wording because they are the two the interpreter can act on. The
message ends by saying the empty map is **not** an absence of change, which is
the same rule as the "outside coverage" wording one step earlier.

### And what it turned out to be: `earthengine.maps.create`

**Settled empirically 2026-08-28** by `src/check_ee_service.py` against the live
deployment, which walks the labeller's own path from a terminal:

```
✓ it mints a service-account token — habloss-labelling@ee-gsingh.iam.gserviceaccount.com
✓ the token may compute — earthengine.computations.create
✗ the token may mint map tiles — HTTP 403: Permission 'earthengine.maps.create'
                                 denied on resource 'projects/ee-gsingh'
```

That asymmetry is the finding, and it is reusable: the account **can** compute
and **cannot** draw, with `roles/earthengine.viewer` and both Service Usage
roles already granted. So `serviceusage.serviceUsageConsumer` is fine (compute
needs it and compute passes), and **`roles/earthengine.viewer` grants
`earthengine.computations.create` but NOT `earthengine.maps.create`**. The fix
is `roles/earthengine.writer`. Every piece of guidance in this repository that
said "grant viewer" -- `Code.gs`, this section's first draft -- was insufficient
for the only thing the overlays do.

With the report finally on screen:

> Permission `earthengine.maps.create` denied on resource `projects/ee-gsingh`

Not an app fault, and confirmed from Python that `ee-gsingh` mints map tiles
perfectly well under a *human* account. The refused identity is the **campaign
service account**, and a brand-new service account has **no IAM roles at all**,
so this is the ordinary state of a fresh deployment rather than a corruption.
The fix is `roles/earthengine.writer` (with
`roles/serviceusage.serviceUsageConsumer`, which `Code.gs` already documented)
on the project — **not** viewer, which is what was already granted and is the
whole finding; see the measured paragraph below.

Two things in the codebase were wrong, and both are the interesting part.

**The deployment self-test could not see it.** `eeTokenSelfTest()` probes
`value:compute`, and its comment argued the probe was sufficient because "every
single thing it does -- getMapId, getThumbId, reduceRegion -- is a
COMPUTATION". That is false in the one place it matters:
**`earthengine.maps.create` is a separate IAM permission from
`earthengine.computations.create`.** An account can pass the self-test and be
refused every map tile — which is precisely what happened, so the deployment
reported itself healthy and a labeller found it instead. `eeMapsSelfTest()` is
the other half; the probe is `ee.Image(1)` serialised exactly as the client
serialises it, because `maps.create` will not take a constant.

Both halves still live *inside the Apps Script editor*, so they are run by hand,
by whoever remembers, and never before a campaign.
`src/check_ee_service.py --url <exec>` is the same walk from a terminal — ping,
mint, compute, `maps.create`, **one tile**, then a real panel overlay (`dw24`
over a batch point) — and exits non-zero, so it can gate a round. The last two
steps are the ones neither editor probe makes: tiles are fetched by MapLibre as
plain images with **no `Authorization` header** (the map id carries its own
credential — verified against a live mint), which is why a rotated token, or an
expression that only fails at render time, passes both self-tests and still
draws nothing.

**And the maps probe itself was wrong, in the shape of the thing it tests.**
`maps.create` requires `fileFormat`; without it the answer is HTTP 400
*"Missing or unrecognized image file format: IMAGE_FILE_FORMAT_UNSPECIFIED"*
however complete the account's IAM is. So `eeMapsSelfTest()` could never have
gone green — a deployment that had just been fixed would still report that it
could not draw, which is the *inverse* of the compute probe's failure and would
have sent the next person chasing roles that were already granted. Both probes
now send `AUTO_JPEG_PNG` (what `convert_to_image_file_format(None)` yields in
the client), and both classify a 400 as *the request*, never as the account.
Confirmed after the grant: all six steps green, `dw24` mints and its tile at the
point is 32,687 bytes — byte-identical to the same tile under a human account.

**`roles/earthengine.viewer` is not enough and `roles/earthengine.writer` is**,
measured on this deployment: viewer + both Service Usage roles gave
`earthengine.computations.create` and a denial on `earthengine.maps.create`.
Note what writer widens — the token this app hands to *every browser that opens
it* can now create and delete assets in the project, so the campaign account
belongs in a project that holds nothing, or on a custom role of
`earthengine.computations.create` + `earthengine.maps.create` +
`serviceusage.services.use`.

**Apps Script `/exec` is intermittently a 404.** Measured 1 request in 4 against
a healthy deployment, the other three fine, answering with Google's HTML error
page rather than JSON. Single-shot, that costs a labeller every overlay for the
whole session — `'auto'` falls back to a sign-in button nobody can complete,
`'service'` shows a deployment fault — for a fault that has already gone away.
`eeServiceToken()` now retries three times with a short backoff; a `null`
(*not configured*) is an answer and still returns on the first attempt.

**And the app told the interpreter to read it as coverage.** The mint-failure
branch ended "Treat this as **outside coverage**, not as an absence of change"
— for *any* failure, permission denials included. Those need opposite
reactions: "outside coverage" is a statement about the ground, a 403 is a
statement about the deployment, and it is every point for every labeller.
Handing someone a configuration fault dressed as a coverage boundary is the
exact failure the `cover` field and the "never blank" rule exist to prevent,
arriving through the other door. `eeFailureAdvice()` classifies it, and takes
the fallback wording from its **caller**: an empty mint may honestly be
coverage, a tile that returned HTTP 400 never is.

Worth keeping separate from that: several overlays are **masked differences**
(`dwbuilt` masks to |Δ| > 0.15, `obtemporal` likewise), so a fully transparent
result is a *correct* answer at a stable point and is indistinguishable by eye
from a broken layer. `dw18` / `dw24` are unmasked class maps and are the right
thing to select when checking whether the overlays work at all.

### Three things found by one failing test, after config.js became a secret

`config.js` is now a **deployment artefact**: blank in the repository, injected
from the `LABEL_APP_CONFIG_JS` Actions secret at Pages build time. That turned
`test_saving_without_an_identity_is_refused` red, and chasing it turned up four
faults rather than one.

**The suite was reading the deployment's config.** `stub_config`'s docstring
already said why that is wrong -- "a test that says 'unconfigured' was really
saying 'whatever the deployment happens to hold'" -- but the lesson had been
applied to five call sites out of forty-four. With `experts: []` the identity
gate falls back to a free-text box instead of a roster dropdown, so the `a`/`z`
keystrokes the test presses to prove labelling is REFUSED were typed into that
box and Enter created an expert called "az". `open_app` now defaults to
`FIXTURE_CONFIG`, so the suite states what it assumes.

**The `server` fixture leaked its Sheet.** Module-scoped, so rows accumulated
across the whole file and `pullSheetState` answered a later test with an earlier
test's rows -- points marked as held, `advanceIfSettled()` moving off the point
under the cursor. It failed after four predecessors and not after a fifth, which
is the shape of a leak rather than of a bug in the test. Reset per test, server
kept.

**And the identity gate focuses on a timer that outlived it.** `setTimeout(...
.focus(), 50)` -- a pick made inside that window left the timer pending, so it
fired after the gate closed and put the cursor back into a now-hidden text box,
where `inField()` correctly swallows every key. The app is keyboard-first and
the very next thing anyone does is press a class key, so this reads as the app
being dead. The handle was also closure-local, so a re-entrant gate (a class key
pressed before the first gate takes focus reaches the document, `pick()` calls
`requireIdentity()`, and that shows the gate again) armed a second timer the
first gate's close could not cancel. Now: one module-level handle, cancelled and
both controls blurred on close, and `showIdentityGate` refuses to build a second
gate over the first.

**What is NOT closed**: a key pressed in the exact instant focus is handed back
is still dropped, roughly one press in six in a headless browser. No page error,
`CLASSES` intact, focus already off the field by the time the key is sent. A
person presses again and thinks nothing of it, so this is a nuisance rather than
a fault, and `press_until` in the tests presses the way a person does. That the
app hands the keyboard back **at all** is asserted separately and without any
retry, in `test_the_gate_lets_go_of_the_keyboard_when_it_closes`.

### And the panel order, which is a labelling decision

The evidence panel led with twenty-three rows of Dynamic World, WorldCover,
ESRI, GLanCE and Hansen, and put the spectral profile behind a fold at the
bottom. What is being collected here is training data for a **10 m model over
AlphaEarth**, so the reading that decides the call has to be the one made at
10 m from the sensor. Other people's classifications at the top of a panel are
an anchor: the interpreter spends their attention agreeing with Dynamic World
instead of looking at the point — the same hazard §AL7 built the collapsed model
hint to prevent, from a source nobody had thought to collapse.

So: **spectral profile** (open), then **index over time** with the dense series
behind it, then **the existing maps, collapsed, last**. They stay — §P is
explicit that their value is that they fail *differently* — one click away, with
their end years, to be read after the call rather than before it. Pinned in
`test_the_evidence_renders_with_earth_engine_never_signed_in`, which now asserts
the DOM order and that the fold is shut.

## AL11 — the pre-pilot pass: what a label means, and four silent losses

Two reviews (an external source-and-asset read, and the user's own testing)
landed together on 2026-08-31, days before the pilot. What follows is what
survived checking against the code — several of the review's claims were
correct in diagnosis and wrong in remedy, and those are recorded here because
the remedy is the part that would have cost something.

### AL11.1 — the labelling unit was never drawn, and the brief named the wrong one

**This is the one that changes what a label means, and everything else in AL11
is housekeeping beside it.**

The original RECOVER sampling called each **10 m cell by majority cover**. The
app's first-run brief said *"Judge the point, not the whole square"* — the
opposite rule — and nothing on the map drew a 10 m footprint at all: the marker
is a 2.6 px dot inside a 13 px halo in **screen** pixels, so it is the same size
at every zoom and names no ground area. An interpreter cannot apply a
majority-cover rule to a footprint they cannot see, and two interpreters each
inventing one arrive in the data as a disagreement about the *legend* — on the
Cropland / Nature boundary this ledger already names as the change-F1 ceiling.

Three things now state the same rule and have to move together: the `cell` map
layer, the brief, and the **dense series footprint**. That last one was the
quiet part — the dense series read a **30 m radius circle at 20 m**, ~28× the
area of the cell being called, so the chart that justifies a call described a
60 m neighbourhood. A hedge, a track or a field margin *outside* the cell moved
the line. Now `CELL_M = 10` with `buffer(5).bounds()`, mirrored in
`denseFetchLive` and guarded by `tests/test_chip_ramp.py`; bake version
`dense1 → dense2`, so old sidecars fall back to live rather than being served
against the new brief.

### AL11.2 — confidence is required

Optional meant absent. Without it an unresolved two-expert disagreement cannot
be told apart from an ambiguous legend, which is the input the adjudication work
in T4.1 will need first. Affordable *only* because `1`/`2`/`3` were already
bound and merely untaught — a mandatory field that needed the mouse would be a
different and much worse change. Not asked on the `uninterpretable` path.

### AL11.3 — four silent losses, and one review remedy worth rejecting

| | what it did | fixed by |
| --- | --- | --- |
| **Dropping an export back in** | `adoptBatch` replaced the batch's 100 points with the 52 labelled ones. The strip lost every outstanding point, and this is the file you reach for *when sync has already failed* | branch on content; a labels CSV restores, grouped by batch, filed under the expert who made the rows |
| **The outbox ack race** | `sendGroup` snapshotted rows, awaited, then dropped by key — deleting a correction made during the flight and marking it `_synced` | identity check: `add()` replaces the object, so `held[key] !== rec` **is** the revision test |
| **Progress denominator** | `done / S.points.length`, so an expert who owed 52 of 100 read `52 %` at completion | `myWorkload()` counts assigned readings |
| **"Second readings"** | claimed another expert had already called the point; nothing knows that, and on a fresh batch they have not | renamed *Independent overlap* |

**The rejected remedy.** The review asked for revision IDs on the wire —
`{key, revision}`, acknowledged and matched server-side. Unnecessary: `Outbox`
already replaces the object at a key rather than mutating it, so object identity
is exactly the revision check, for three lines and no change to the `Code.gs`
contract. It also asked for **server-side rejection of unassigned points**. That
one cannot work as described — the Apps Script has no copy of the batch file and
never sees `required_readers`, so enforcing it means shipping assignment into
the Sheet and keeping it in step with every batch cut; and a hard reject turns a
legitimately reassigned point into a row the outbox retries into `rejected` and
then abandons.

### AL11.4 — the map's ⓘ did nothing, for a CSS reason worth generalising

`details > summary { display: flex }` — an **element** selector, on a page that
also contains somebody else's widget. MapLibre's compact attribution control is
a `<details><summary>`, so the app's one-disclosure-idiom rule turned a 24×24
absolutely-positioned icon into a flex row with a `▸` hung off the end. Scoped
off `.maplibregl-ctrl-attrib-button` rather than out-specified: the next
third-party `<details>` gets excluded there too.

### AL11.5 — two things that make the evidence readable

* **The chart dot was switched to the filmstrip's colour, and then switched
  back.** See §AL11.8 — that one is settled by measurement, not by argument.
* **EOX Sentinel-2 cloudless 2018 and 2024 are basemaps.** Every other imagery
  source in the app is a different sensor at a different resolution on a date
  somebody else chose; these two are Sentinel-2 at 10 m — the model's own sensor
  and pixel — at exactly the two years being called. Carto Positron removed (did
  not draw; note that its tile URL answers 200 outside a browser, so re-adding it
  unchanged will not fix it).

### AL11.6 — the Wayback fan-out

One arrow press issued **~40 concurrent metadata requests**: two targets × four
candidate releases × five resolution-band probes. The cache was written only on
resolution, so the overlapping asks (refine, read-out, prefetch all want the
current release) each missed and re-issued; and nothing was cancelled on point
change — `wbSnapRefine` discarded the *result* via `gen` while the requests went
on competing with the new point's tiles and sprites. Now: the **promise** is
cached, concurrency is capped at 6, the candidate walk goes outward two at a
time and stops when the nearer pair answers, and leaving a point aborts
everything in flight for it.

### AL11.6b — three things only running the app could find

Everything above came out of reading source. These came out of launching it and
looking, and none of them would have failed a test:

* **The chart dots were still blue.** AL11.5 set the colour as a `fill=`
  *presentation attribute*, and in SVG a CSS **property** always wins — so
  `.ev-plot .ev-dot { fill: var(--accent) }` repainted all nine, every time, over
  a correct attribute. The test asserted `getAttribute('fill')` and passed
  throughout. Fixed by emitting an inline `style="fill:…"` (which does beat the
  stylesheet) and by asserting `getComputedStyle`, which is the claim being made.
  **Read the computed value whenever a test is standing in for "somebody can see
  it".**
* **The ⓘ had a second, independent cause.** AL11.4 fixed how it *renders*; it
  was still dead to the mouse, because `#ev-strip` re-enables `pointer-events`
  across the full window width to be scrollable, and the empty space to the right
  of nine chips sat over the map's bottom-right corner. `width: fit-content`
  (plus `max-width: 100%`, so a narrow window still scrolls) and a z-index bump
  on MapLibre's bottom controls. Two causes, one symptom — the first fix looked
  like it had worked because the button now *looked* right.
* **§AL9's `.body` bug, reintroduced within the hour.** `.card label` is
  `display: flex`, so a new checkbox written as a text node plus a `<b>` became
  two flex items with a 7 px gap and independent wrapping — three squashed
  columns again. **Any text put inside a flex container needs its own single
  span.**

### AL11.8 — the dot colour, settled by measurement

Three positions in one day; the third has numbers behind it.

| | |
| --- | --- |
| **AL10** | dot = the plotted index's ramp. Colour and height say the same number twice; NDVI's 0.31 cut becomes a colour boundary. |
| **AL11.5** | dot = the chip's colour, so a dot can be matched to its filmstrip cell. Argument: saying one number twice is not worth a channel. |
| **AL11.8** | **back to the index ramp.** |

The reversal is not a change of mind about the argument. It is that a
**false-colour mix is not the index**, and at some points it runs the other way.
Measured over the 99 stretched points of b001 under the default
`SWIR1/NIR/GREEN`, NDVI against perceived greenness:

* **84 points positive** (median r = **+0.708**) — the common case, which is why
  it looked right on the first point anyone opened;
* **15 points negative**, worst p0005 (**r = −0.853**), p0056 (−0.842),
  p0010 (−0.814), p0013 (−0.706).

Red is SWIR1 in that scheme, so dry-but-vegetated ground renders brown at a high
NDVI. Four of the first fourteen points do it. **A colour scale that reverses at
15% of points is not a scale** — it cannot be read without first checking which
scheme is loaded, which is the opposite of what a colour channel is for.
Re-checked after the revert: across all 100 points there is now **no** year where
a higher NDVI gives a lower ramp position. The cross-link AL11.5 wanted is
genuinely lost, and that is the price.

**Two traps this left in the tests, both worth more than the verdict:**

1. The dot test asserted `getAttribute('fill')` and passed for weeks while every
   dot rendered accent blue (§AL11.6b). Read `getComputedStyle` whenever a test
   stands in for *somebody can see it*.
2. The first monotonicity assertion measured "greener" as `G − R`, and would
   have **failed on a correct ramp**: a bare point whose nine years all sit near
   NDVI 0.1 is legitimately brown throughout, and inside that first ramp segment
   red rises faster than green. p0005 and p0010 are exactly that shape. The test
   now asserts two composable facts — each dot is `rampColor` at *its own* value,
   and the ramp constant runs brown → green — instead of a perception heuristic
   that has to be right about colour.

Also fixed here: `EVIDENCE_BATCH` in the suite declared an NDVI series running
0.71 → 0.26 while setting **every band to the same values**, so the real NDVI was
0 in all nine years and every dot came out one brown. A fixture whose declared
series and declared bands disagree cannot test a colour computed from the bands;
`B8` is now derived from the declared NDVI.

### AL11.7 — still open

* **Index filmstrips are not baked.** `CHIP_INDEX` is offered in the scheme
  picker but `build_batch_chips.COMBOS` holds only the four RGB schemes, so
  picking NDVI/NDMI/NBR drops all nine years to live Earth Engine — AL9/AL10's
  failure mode, still live for three of seven options. The indices derive from
  bands the existing per-point request already fetches, so this is near-free to
  add *during* a re-bake and a full second bake to add afterwards.
* **The deploy workflow checks bake directories and now bake VERSIONS**
  (§AL12: it compares each batch's declared `chips`/`dense` version against the
  builders' constants, and the builders' against the app's — a directory that
  exists is not a bake that is served). Still unchecked: `evidence_version`,
  point ids, sprite combos and sidecar counts. b001 shipped as `ev1` with ESRI
  stopping at 2023 against an `ev2` builder reaching 2025, and nothing said so.
* **b001 carries a pre-AL12 bake.** Its `dense` sidecars are `dense2` (a
  square centred on the point) and its sprites paint the old ring, so the app
  falls back to live Earth Engine for the series and shows the wrong footprint
  on the pictures until `build_batch_dense.py` and `build_batch_chips.py` are
  re-run. Budget ~36 min per 100 points for the chips.
* **Adjudication** (T4.1) is unchanged and still waits for a real disagreement.
  One thing did move: `agreement()` counted only rows with a transition, so a
  *usable vs cannot-interpret* pair was neither an agreement nor a disagreement
  nor a doubled point — it left the denominator entirely, and it is the most
  informative pair in the set.

## AL12 — the cell is a pixel, and the app is addressable (2026-08-31)

Prompted by reading `eo-timeseries-explorer.js`, the Earth Engine app this
panel's chip strip is descended from, for anything worth porting. Almost
nothing was — it has no cache, no prefetch, no scene cap and no bake, and the
techniques it does have (`paint` + `blend` marker, `crs: 'EPSG:3857'`, async
`evaluate` behind a placeholder, chart dots coloured from area-mean
reflectance) are all already here in stronger form. Three things came out of
it, and one of them changed what a label means for the second time in a day.

### AL12.1 — the labelling unit is the SENTINEL-2 PIXEL, not a square at the point

AL11.1 drew the cell for the first time and got the geometry wrong in a way
that only shows up when you ask what the square *is*. A 10 m square **centred
on the point** is not a pixel of anything: it straddles four Sentinel-2 pixels
and covers no one of them. So the interpreter judged one footprint, the dense
series read a second (the mean of up to four pixels), and the model predicts a
third — three answers about "this cell", none of them the same ground.

There is exactly one square that removes the ambiguity and it is not a choice.
Sentinel-2 granules sit on the UTM grid of their MGRS tile with 10 m pixel
edges on multiples of 10 m, **and so does the deployed map**:
`oslo_s2off_centre_m3s3_bf_merged2.tif` is EPSG:32632, 10 m, origin
589230/6652940 — both exact multiples of 10. Snapping the point's UTM
coordinates down to a multiple of 10 therefore names the same square in the
imagery, in the evidence and in the model's own output raster.

`src/label_cell.py` is that definition and `s2Cell()` in `label_app.html` is
the same definition in JavaScript, because the app must draw the cell for a
batch it built (baked into the batch as `cell`, and drawn in preference to
anything computed) *and* for a file dropped on the window. They are checked
against each other in node in `tests/test_label_cell.py`: same zone, same
pixel, corners within 5 cm, on 400 global points including 32V and the Svalbard
row — **the MGRS zone exceptions are not academic here**, the study area is
Norway and Bergen is in one of them.

Four things moved together, the way AL11.1 says they must:

* the `cell` map layer draws the snapped quad — the four corners transformed
  back **one at a time**, not a lon/lat bounding box: grid convergence rotates
  the square by up to ~3° at a zone edge, which is 0.45 m over a 10 m cell;
* the **dense series** reads `reduceRegion` over the **point** at scale 10,
  which is the value of the pixel containing it in the granule's own grid —
  the pixel exactly, with no geometry to get wrong. Bake `dense2 → dense3`;
* the **chip filmstrip** paints the cell into every cell of the sprite. What
  was there was a red ring at `max(width_m * 0.02, 6)` m — a 12.8 m radius at
  the default 640 m width, against a 5 m cell, and it *changed size with the
  width slider*. It is now a white **locator** ring (a fixed fraction of the
  width, so ~7 px in the strip at any width, and naming no ground area) with
  the red cell square inside it. Rendered and looked at: at the strip's 176 px
  the cell is ~3 px and reads as a dot inside the ring, at the lightbox's
  512 px it is ~8 px and is plainly a square. Both are correct and neither is
  the surface the call is made on — that is the map, where the cell is tens of
  pixels across at working zoom;
* **the marker dot is gone.** It was 2.6 screen pixels at every zoom and it was
  the only thing on the map naming the call, so it said "judge this point" in
  the only language a map has. What is left is the halo, faded out by zoom
  between z15 and z17 — over exactly the zooms where the cell becomes big
  enough to be its own marker — so at working zoom nothing but the pixel is on
  screen. The Wayback **compare** map never drew the cell at all, which is the
  one view where a change call is actually made.

The point survives as the *address* of a pixel. Nothing should read a buffer
around it again, and the `reduceRegion`-over-a-buffer form is now asserted
against in `tests/test_chip_ramp.py`.

Two consequences to be honest about. Sentinel-2 tiles overlap and a point in an
overlap sits in two granules whose zones can differ — there the two candidate
grids are rotated relative to each other and no square is "the" pixel; the MGRS
rule picks the granule the composite is dominated by almost everywhere and is
not worth more than that. And **b001 must be re-baked** (`chips` and `dense`),
because its sidecars are `dense2` and its sprites carry the old ring.

### AL12.2 — the scene cap was counting granules, not dates

`CHIP_SCENE_CAP = 12` (AL9's largest single latency win) sorted by
`CLOUDY_PIXEL_PERCENTAGE` and took twelve **images**. MGRS tiles overlap, so a
point whose chip box sits in an overlap sees the same overpass as two — or, at
a tile corner, four — granules, and spends its slots on them.
`distinct('DATATAKE_IDENTIFIER')` between the sort and the limit — jdbcode's
`col.distinct('date')`, which is what put the question — spends the cap on
twelve **dates**.

**Measured on b001 itself** (a global random draw, 2024, the 640 m chip box):
**26 of 100** points span more than one MGRS tile — 2 or 4 of them — and
**49 of 100** carry duplicate datatakes. Across those 49, 9,944 granules are
5,472 acquisitions: they were seeing each date 1.8× over on average, and the
worst point 1,610 → 404, a clean **4×**. A twelve-granule cap at that point was
a median over **three dates**. This is much larger than the tile-overlap
geometry suggests on paper, and it had been the recipe since AL9.

Two things checked against the live service rather than assumed. `distinct`
**preserves the sort**, so the granule that survives each date is still that
date's clearest and no re-sort is needed. And the cap is not free: at p0004 the
mean scene cloud of the chosen twelve rises **27.1% → 39.4%**, because twelve
dates reach further down the list than six do. That is the right trade — the
mask is per pixel and `CLOUDY_PIXEL_PERCENTAGE` is a whole-granule statistic
only loosely related to a 640 m box, so more dates is more chances of a clear
observation *here* — but it is a trade and not a free win. Mint and fetch of a
two-year sprite at that point: 1.8 s and 2.0 s, inside AL9's noise.

In both the app and the builder, which is the standing rule for that recipe.

### AL12.3 — the view is addressable

The explorer rebuilds itself entirely from `ui.url` (`lon`, `lat`, `rgb`,
`index`, `chipwidth`). This app took `?batch=` and the config overrides and
nothing that named a *point*. A two-expert campaign settles a disagreement by
one person looking at what the other looked at, and the most that could be
handed over was a batch — so `?point=p0042&scheme=NIR/SWIR1/RED&w=640` now
opens the point, on the same imagery, under the same stretch, and `goTo` and
`applyChipVis` rewrite it so the link to send is always the one in the address
bar. The rest of the query string is preserved (a rewrite that dropped `batch`
would break the link it was pasted into), a named point that is not in the
batch resumes rather than showing nothing, and `expert` is never *added*: the
annotation key is `(campaign, batch_id, point_id, expert_id)` and this link is
meant to be sent to the other reader.

## Decisions this design could not make

1. ~~**Is the labelling sequential or one-shot?**~~ **Settled 2026-08-25:
   sequential.** §AL7's app is batch-structured — an interpreter takes a batch,
   finishes it, syncs it, takes the next — so Stage 3 and step 8 stay in. This
   was the decision §AL4 said was worth ±0.034 change-F1, and it was a decision
   about people rather than about code, which is why building the batched
   workflow is what made it.
2. **Does `Artificial → Cropland` get an external source, or get merged?** At 46
   plots the model has not learned a representation of it, so it cannot be used
   to find more of it; the auxiliary GLAD cropland layer is the only proposed
   route. If nobody will source it externally, merging it is the honest call and
   the nine-class map becomes an eight-class map on the page as well as in fact.
3. **Is "double every reachable change class" still the target?** Everything in
   §B and in `deficit_weighted_yield` is denominated in it.
4. **5 km cells, or finer?** 5 km matches the existing patch geometry and the
   labelling unit. Finer is 25× the memory for a prior that is not that sharp.
