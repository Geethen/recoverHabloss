# Mapping the loss and recovery of vegetation, 2018–2024

**A working report on the current model, its accuracy, and what we are unsure
about.** Version of 28 July 2026.

## What this document is, and what we are asking for

We have built a model that predicts, for every 10 m pixel on the ground, how the
land cover changed between 2018 and 2024 — specifically, whether it moved between
**Nature**, **Cropland** and **Artificial** (built-up, sealed) land. This report
describes how it works, how accurate it is, and where it fails.

It is written for ecologists and remote-sensing colleagues who understand models
in outline but have not worked on this project. No prior knowledge of our code is
assumed, and no jargon internal to the project is used.

**We would particularly value your reaction to four things:**

1. Whether the accuracies in §6 are good enough for the ecological questions you
   would want to ask of a map like this — and if not, what the threshold would be.
2. Our reading of the evidence in §7.3, that the limiting factor is the number of
   interpreted plots rather than anything about the model.
3. The caveats in §8, especially the sampling caveat in §2.2, which affects how
   every accuracy number here should be interpreted.
4. Anything in §9, the open questions we cannot currently answer.

Section 10 lists what we have already tried and abandoned, with the reasons, so
that a suggestion already tested does not cost anyone time. If your idea is on
that list and you think our test of it was wrong, we would genuinely like to hear
that — several entries rest on a single experiment.

---

## 1. Background and aim

The wider project estimates how much land moved between Nature, Cropland and
Artificial cover globally, at 10 m resolution, over 2018–2024. Three things make
it different from existing products: no current global 10 m product maps these
*transitions* at this resolution; none explicitly handles built-up land reverting
to vegetation, or cropland being abandoned; and none says *what was lost* when
impervious surface expands.

Two distinct products come out of the work, and it is worth separating them at
the outset:

- **Area estimates** — how many square kilometres moved from one class to
  another. These come from statistical estimation on the interpreted plot
  sample, not from the map, for reasons given in §8.2.
- **Maps** — where the change is. That is what this report is about.

The response variable is not land cover at one date but the *pair* of classes at
the two dates. With three classes there are nine possible transitions, and all
nine occur in our reference plots (Table 1). We call this the **nine-class**
reading.

Pooling Nature and Cropland into a single **Vegetation** class gives a coarser,
four-class reading. The two answer different questions and are not
interchangeable:

- **Four-class**: *did this pixel change between vegetated and built-up?* This is
  the habitat-loss question, and it is the reading whose accuracy we can defend.
- **Nine-class**: *what kind of change?* A field going under tarmac is
  ecologically different from a forest doing the same, and that distinction
  disappears in the four-class reading.

Both are produced for every map, from a single model.

## 2. The reference data

### 2.1 What the plots are

The model is trained on **6,414 plots**, each visually interpreted by a trained
analyst at both dates and assigned a class for 2018 and for 2024.

**Table 1. Training plots by transition.**

| nine-class transition | plots | four-class reading | plots |
| --- | ---: | --- | ---: |
| Nature → Nature | 2,532 | Vegetation → Vegetation | 4,550 |
| Cropland → Cropland | 1,661 | | |
| Nature → Cropland | 243 | | |
| Cropland → Nature | 114 | | |
| Artificial → Artificial | 979 | Artificial → Artificial | 979 |
| Nature → Artificial | 383 | Vegetation → Artificial | 716 |
| Cropland → Artificial | 333 | | |
| Artificial → Nature | 123 | Artificial → Vegetation | 169 |
| Artificial → Cropland | 46 | | |

The imbalance is the central difficulty. Loss of vegetation to built-up cover —
the thing the work exists to measure — is 11% of the plots and, once mapped,
**under 1% of the pixels**. Any method that treats the classes even-handedly will
predict "no change" everywhere and score well for doing so.

### 2.2 An important caveat about what these plots represent

The plots were **not** drawn at random across the land surface. They come from two
stratified sampling programmes that divided the world into strata by biome and by
what an existing map said had happened there, and then interpreted roughly the
same number of plots in every stratum. Places where change was expected are
therefore represented far more heavily than their share of the land, by several
orders of magnitude.

This has a consequence that should be carried through every number in this
report: **the accuracies in §6 describe performance on this sample, not on the
land surface as a whole.** Because rare transitions are heavily over-represented
relative to reality, we think these figures are more likely to flatter the model
than to understate it — a change class that is 11% of our plots is well under 1%
of the world, and rare classes are harder in proportion to their rarity.

We have not yet produced sample-weighted versions of these accuracies. Whether
that is worth doing, and how you would prefer to see it presented, is one of the
open questions in §9.

## 3. What the model looks at

Two data sources are used, chosen because they fail in opposite ways.

**Annual satellite embeddings — 192 variables, the *context* source.** Google's
AlphaEarth model compresses a full year of multi-sensor satellite observation
into 64 numbers for each 10 m pixel — a learned summary rather than a reflectance
measurement. We use the 2018 vector, the 2024 vector and their difference
(64 × 3 = 192). Because each number summarises a whole year and a neighbourhood,
these variables are stable and quiet, but spatially smooth: they blur the edges of
small features. They are available for 100% of plots and 100% of the mapped
areas.

**Sentinel-2 reflectance and texture — 78 variables, the *detail* source.** For
each plot and year we build a cloud-free composite from up to four Sentinel-2
scenes, one per season, so that 2018 and 2024 are seen at comparable points in
the growing cycle — a leaf-on / leaf-off mismatch would otherwise manufacture
"change" that is only phenology. Seven channels are derived (blue, green, red,
near-infrared, NDVI, NDWI, brightness), and from them we compute, for 2018, 2024
and their difference:

| variables | what it measures | why it might help |
| ---: | --- | --- |
| 21 | the seven channels at the plot centre | what a simple point sample would have given |
| 21 | their average over a 30 m window | the pixel's immediate surroundings |
| 21 | their **standard deviation** over the same window | *heterogeneity* — whether a pixel sits inside a uniform field or a built-up mosaic. A smoothed annual summary cannot express this |
| 15 | built fraction at five window sizes | the share of nearby pixels below an NDVI threshold of 0.31, calibrated against our stable plots: a direct, readable index of sealed surface |

These are sharp and consequently noisier — cloud, shadow, haze and viewing
geometry all enter. 99.3% of plots have usable Sentinel-2 at both dates.

## 4. The model, in plain terms

### 4.1 Two branches, and one of them is only a training aid

Each data source passes through its own small neural network, and the two
resulting summaries are averaged. Each branch has an on/off switch, so a plot with
no usable Sentinel-2 still gets a prediction from the embeddings alone: the model
degrades rather than refusing. During training, one branch is switched off at
random half the time, which stops the network becoming dependent on Sentinel-2.

**When maps are made, the Sentinel-2 branch is switched off permanently.** The
model is trained with the detail source and used without it. This is sometimes
called learning from *privileged information*: an extra source available while
learning but not in operation, whose value lies in shaping what the network
learns rather than in what it contributes at prediction time.

The practical consequence is large. Mapping an area needs no Sentinel-2 download
or processing at all. The prediction step for our test area (Oslo, 2.95 million
pixels) takes **under 5 seconds** on one GPU, against roughly nine minutes for
the identical model used with the detail branch active.

Whether that branch is earning its place at all is discussed honestly in §7.2.

### 4.2 One prediction, read at three levels

The network predicts the nine transitions. The coarser readings are then obtained
**by addition, not by a second model**: the probability of "Vegetation →
Artificial" is the sum of the probabilities of "Nature → Artificial" and
"Cropland → Artificial", and the probability of "changed" is the sum over all six
transitions that involve a change.

Training penalises errors at all three levels at once — nine-class, four-class,
and changed / unchanged — using a loss function that gives less weight to
examples the model already gets right, a standard remedy for the imbalance in
Table 1.

This turns out to be the single most valuable design decision in the model
(§7.1). Because the coarse readings are sums of the fine one, the maps are
consistent with each other by construction, and the plentiful changed/unchanged
signal helps steady the nine-way problem instead of leaving it to be learned from
46 plots of Artificial → Cropland.

Training is short and deliberately so: 30 passes over the data. Beyond roughly 30
to 50 passes, accuracy on the change classes declines, because the network starts
fitting inconsistencies in the interpretation of the Cropland/Nature boundary
rather than the signal.

### 4.3 Five networks, not one

A neural network fitted once is a single draw; refit it from a different random
starting point and you get a slightly different model. For a map this matters more
than for a table, because pixels visibly move between runs. We therefore fit
**five networks differing only in their random starting point** and average their
predictions.

This does not improve accuracy measurably. What it buys is stability, tightening
the run-to-run spread about fourfold. Even so, the map does not fully reproduce
itself — see §7.5, which is the number to read before comparing any two maps.

## 5. How we tested it

All accuracies are out-of-sample under **spatially blocked cross-validation**:
whole spatial blocks of plots are held out together, so plots from the same
landscape cannot appear in both the training and the test set. Ordinary random
splitting would inflate every number reported here, because the plots are
strongly clustered in space.

Three measures are reported, and each exists because the others hide something:

- **change-F1** — the balance between precision and recall for the changed /
  unchanged reading, where *recall* is the share of truly changed plots that were
  found and *precision* is the share of predicted changes that are real. F1 is
  their harmonic mean; 1.0 is perfect. This is our headline figure.
- **macro-F1** — the unweighted average of F1 over the four coarse transitions,
  so an apparent gain cannot come from the stable class that holds 71% of plots.
- **stable built-up recall** — how often existing built-up land is recognised as
  such. Added after visual inspection revealed an error that change-F1 is
  structurally blind to: a model can score well on change while systematically
  misreading existing towns as vegetation.

Every experiment is repeated from five random starting points, and differences
smaller than ±0.005 are treated as noise rather than as results. This rule was
learned the hard way: several apparently clear findings in this project reversed
when repeated.

## 6. Accuracy

### 6.1 Headline

**Table 2. Out-of-sample accuracy, five repeats, full training set.**

| reading | measure | value |
| --- | --- | ---: |
| four-class (the map we recommend using) | change-F1 | **0.657 ± 0.004** |
| four-class | macro-F1 | **0.694 ± 0.003** |
| nine-class | change-F1 | 0.596 ± 0.003 |
| nine-class | macro-F1 | 0.439 ± 0.002 |

### 6.2 Per class

"Training F1" is measured on the same plots the model was fitted on. The gap
between the two columns indicates how much of the model's apparent skill does not
transfer to new places.

| four-class transition | plots | out-of-sample F1 | training F1 |
| --- | ---: | ---: | ---: |
| Vegetation → Vegetation | 4,550 | 0.91 | 0.94 |
| Vegetation → Artificial | 716 | 0.69 | 0.79 |
| Artificial → Artificial | 979 | 0.68 | 0.78 |
| Artificial → Vegetation | 169 | 0.49 | 0.66 |

| nine-class transition | plots | out-of-sample F1 | training F1 |
| --- | ---: | ---: | ---: |
| Nature → Nature | 2,532 | 0.75 | 0.81 |
| Cropland → Cropland | 1,661 | 0.72 | 0.78 |
| Artificial → Artificial | 979 | 0.68 | 0.78 |
| Cropland → Artificial | 333 | 0.59 | 0.70 |
| Nature → Artificial | 383 | 0.47 | 0.63 |
| Artificial → Nature | 123 | 0.41 | 0.54 |
| Nature → Cropland | 243 | 0.31 | 0.43 |
| **Cropland → Nature** | 114 | **0.01** | 0.05 |
| **Artificial → Cropland** | 46 | **0.00** | 0.06 |

The ordering is almost exactly the ordering of the plot counts. Conversion of
vegetation to built-up land, the target of the project, is recovered at F1 = 0.69
in the four-class reading. Reversion — built-up land returning to vegetation — is
recovered at 0.49, and it is the rarest class in the sample.

The gaps between out-of-sample and training accuracy are moderate (0.02–0.06 on
the two large classes, 0.10–0.16 on the change classes), and the training
accuracy itself is nowhere near perfect. The network is not memorising the
training plots; it is short of information about them. That reading is supported
by the fact that stronger regularisation, smaller networks and added noise have
all been tried and all made things worse.

### 6.3 Two of the nine classes do not work at all

Artificial → Cropland (46 plots) scores **exactly zero** out of sample at every
training-set size and in every repeat, and Cropland → Nature (114 plots) scores
0.01. Their training accuracy is near zero too: the network does not even
memorise them, it absorbs them into neighbouring classes.

The map confirms it. Over our test area, across two independent runs, both classes
are painted on **zero** of 2.95 million pixels, and Nature → Cropland on 16.
**The nine-class map is in practice a six-class map.** The plot-level evidence
predicted exactly which classes would vanish, before the map was consulted.

We read this as a limit of the reference data rather than a fault in the model.
These are the classes on the Cropland/Nature boundary that our own reverification
shows interpreters disagree about, and they have the smallest sample sizes in the
set.

## 7. What we have learned, including the uncomfortable parts

### 7.1 What each design decision is worth

Every component was priced by removing it and refitting on the same folds from the
same five random starting points, so the comparisons are like-for-like.

| removing this component costs | change-F1 |
| --- | ---: |
| the three-level training objective (§4.2) | **−0.020** |
| averaging over whether to trust the detail branch | **−0.019** |
| the entire Sentinel-2 apparatus, versus embeddings alone | −0.009 |
| the difference between the 2018 and 2024 embedding vectors | −0.048 |
| ten of the thirty training passes | −0.028 |

Two of these deserve comment. The **three-level objective is the largest single
lever**, larger than the entire second data source, and it costs nothing at
prediction time. And the difference block is not redundant even though it is
arithmetically just the 2024 vector minus the 2018 one: the network cannot form
that subtraction well enough for itself, and removing it is the most damaging
single change we have found.

### 7.2 An honest note on Sentinel-2

Measured on plots, in the configuration we actually use, **Sentinel-2 buys
nothing**. The model scores change-F1 0.6557 ± 0.0034; the same architecture
trained with no Sentinel-2 at any point scores 0.6574 ± 0.0049. The difference is
smaller than the noise and, if anything, favours the simpler model.

The detail branch was kept on the basis of the map: on visual inspection it
carried more structure, and its boundaries fell more often on real edges in the
imagery. We consider that a legitimate way to break a tie the numbers cannot
break — no interpreted plot falls inside our mapped area (§8.1), so the map itself
cannot be scored — but it should be reported for what it is. **This report claims
no measured accuracy gain from Sentinel-2.**

This is one of the places we would most welcome an outside view: whether to keep a
component that is justified by appearance rather than by measurement.

### 7.3 The clearest result: the limit is the number of labelled plots

The most informative experiment we ran was also the simplest — refit the model on
5%, 10%, …, 100% of the training plots and watch the accuracy curve. Every class
that works at all is still improving at 100% of the data, with no sign of
levelling off.

| reading | gain per doubling of labelled plots |
| --- | ---: |
| four-class change-F1 | **+0.026** |
| four-class macro-F1 | +0.020 |
| nine-class change-F1 | +0.029 |

The steepest classes are the change classes — Artificial → Vegetation (+0.039),
Nature → Artificial (+0.035), Cropland → Artificial (+0.034) — that is, the ones
the project exists to map.

For scale: every modelling idea we tested moved change-F1 by less than 0.005, and
most by less than the run-to-run noise. **Doubling the interpreted sample is worth
roughly five times everything the entire model-development effort produced.** We
believe this is the main finding to act on, and we would like it challenged if you
disagree.

### 7.4 The two map readings disagree about change

Over our test area the four-class map calls 16,676 pixels changed and the
nine-class map 16,609 — but only **14,522 are changed in both**. About 13% of each
map's change pixels are absent from the other.

The cause is arithmetic rather than a fault: taking the most likely of nine
classes and then pooling is not the same as pooling first and taking the most
likely of four. Three fine classes at 30/30/35% elect the single class at the fine
level and the pair of classes after pooling.

Practical guidance: **use the four-class map for anything about *whether* change
happened**, since that is the reading whose accuracy we measured; use the
nine-class map to ask *what kind*, on pixels the four-class map has already called
changed.

### 7.5 How much of a map is real

The same model, refitted from five *different* random starting points and mapped
over the same pixels, agrees with itself on only **84% of change pixels**. The
total number of change pixels moves by 5% between the two runs (16,676 against
15,841) for an identical recipe.

This is the noise floor of the product. Two maps differing by less than this are
not different in any way the data supports. We have made this mistake ourselves:
we once read an 83% agreement between two candidate models as a replication
failure, when in fact each model reproduced itself at 84%, and two maps that each
reproduce themselves at 84% will agree with each other at about 84%. **Any
comparison of two maps should first establish what each map's agreement with
itself is.**

If you take these maps and compare them against another product, this is the
number we would ask you to bear in mind.

## 8. Limitations

These bound what the maps can be used for. We would rather over-state them here
than have someone discover them later.

1. **No interpreted plots fall inside the mapped area.** Every accuracy in this
   report comes from plots elsewhere in the world. Nothing here measures accuracy
   *in the place we mapped*; local quality has been judged visually and by
   internal consistency only. Acquiring interpreted plots inside a mapped area is
   the single most valuable outstanding action, and it would settle several open
   questions at once.

2. **Pixel counts are not area estimates.** Counting the pixels of a classified
   map is a biased way to estimate area, and the bias is worst for exactly the
   rare classes of interest here. Defensible areas come from design-based
   estimation on the plot sample, which is a separate strand of the project.
   **These maps are for locating change, not for quantifying it.**

3. **Accuracies describe our sample, not the land surface** (§2.2). The plots
   over-represent places where change was expected.

4. **The Cropland/Nature boundary carries interpreter noise.** Reverification
   shows the two classes are confused with one another in the reference data
   itself. This places a ceiling on nine-class accuracy that no model change can
   lift, and it is why the four-class reading — which pools them — remains the
   defensible one. Within the nine-class map, trust the vegetated / built-up
   distinction more than the Nature / Cropland one.

5. **Do not spatially smooth the output.** Edge-aware filtering makes boundaries
   visibly crisper and deletes 12–22% of the change class in doing so. Change
   pixels are the model's least confident predictions — 35% of them are within
   0.1 probability of flipping, against 7% of stable pixels — so any procedure
   that lets neighbouring pixels vote removes them first. Everything that has
   *added* change acted on the model's inputs or on its handling of uncertainty;
   nothing that acted on the finished map ever has.

6. **The nine-class legend over-promises.** Two of its entries are never painted
   (§6.3). This should be stated whenever the nine-class map is shown.

## 9. Open questions we would like help with

1. **Is a change-F1 of 0.66 useful?** For which ecological questions, and at what
   spatial aggregation? A per-pixel accuracy of this level may be perfectly usable
   for landscape summaries and unusable for site-level inference — we would like
   to know where you would draw that line.
2. **Is change suppression a fix or a fault?** In the earlier configuration that
   left the detail branch switched on while mapping, it removed 9–13% of the
   change pixels relative to switching it off — which is part of why we now switch
   it off. The plot evidence weakly suggests those were false alarms being cleaned
   up rather than real change being lost, but we cannot demonstrate it without
   interpreted plots inside the mapped area. This is the same gap as limitation 1, and it is the most consequential
   thing we do not know.
3. **How should accuracies be weighted?** (§2.2.) Sample-weighted accuracies
   would be more honest about performance across the land surface, and much less
   precise for the rare classes. Which do you want to see?
4. **Where should new interpretation effort go?** Our view is: the two failing
   classes, the Cropland/Nature boundary, and plots inside a mapped area. That
   trades breadth for the specific weaknesses we have identified, and it is worth
   challenging.
5. **Is the three-class legend the right one?** It was chosen so that two sampling
   programmes could be combined. A Forest / Other Nature split on the 2018 date is
   available in part of the reference data and would support statements about what
   kind of nature is being lost — but only in one direction.
6. **Would you use the nine-class map?** Given §6.3 and §7.4, it may be more of a
   liability than an asset outside farmland-rich regions.

## 10. What we have already tried that did not work

Listed so that no one spends time on a path we have already closed. Each was
tested on the same plots under the same validation scheme. If you think a test was
badly designed, please say so — several of these rest on a single experiment.

- **Smoothing or sharpening the finished map** (edge-aware filtering, applied both
  to the class map and to the underlying probabilities). Delivers crisper
  boundaries and destroys 12–22% of the change class. See limitation 5 for the
  mechanism.
- **A second, sharper embedding product** as the detail source instead of
  Sentinel-2. It performed well where it existed, but only 36% of plots had it at
  both dates, and the gap could not be worked around.
- **Letting the network learn its own texture** from small image patches, instead
  of hand-computed statistics such as standard deviation and built fraction.
  Clearly worse: there are not enough labelled plots to learn image filters.
- **A hard NDVI threshold** to separate built-up from vegetation. The threshold
  itself is well calibrated (0.31), and as a *proportion* over a 30 m window it is
  our single most useful built-up variable — but as a rule it cannot fix the
  built-up plots the model misreads, because those plots are genuinely green in
  the reflectance and built-up in the interpretation. That is a disagreement
  between the label and the pixel, and no spectral index can win it.
- **Class-balanced sampling** during training. Underperforms handling the
  imbalance in the loss function instead.
- **Mixture-of-experts architectures, added training noise, knowledge
  distillation, and supervising the two dates separately.** All within noise or
  worse.
- **More capacity, or longer training.** Both worse; see §4.2.
- **Additional derived variables** (per-band products, normalised differences).
  Tie with or underperform the difference variables we already use.

One recurring pattern is worth flagging, because it may be the most transferable
thing we have learned: **interventions that make the model more cautious remove
real change before they remove false alarms**, because change pixels are a small
fraction of the surface and the model is barely confident about them. Anything
that regularises, smooths or votes will look like an improvement on every
aggregate measure while quietly deleting the class of interest.

## 11. Terms used

| term | meaning here |
| --- | --- |
| **transition** | the pair of land-cover classes at 2018 and 2024, e.g. Nature → Artificial |
| **Artificial** | built-up / sealed surface |
| **out-of-sample** | measured on plots the model was not fitted on |
| **spatially blocked validation** | holding out whole spatial blocks, so nearby plots cannot leak between training and testing |
| **precision / recall** | of predicted changes, the share that are real / of real changes, the share that were found |
| **F1** | the harmonic mean of precision and recall; 1.0 is perfect |
| **macro-F1** | F1 averaged over classes without weighting by class size |
| **embedding** | a learned numerical summary of a year of satellite observation, one vector per pixel |
| **NDVI / NDWI** | standard vegetation and water indices computed from reflectance |

## 12. Availability

The code, the maps described here, and the full research record — including every
negative result and the experiment that produced it — are held in the project
repository and can be shared on request. Learning-curve figures for §7.3 and the
maps for §6.3 and §7.4 are available as separate files.

Comments, objections and suggestions are all welcome, including on the points this
report presents as settled.
