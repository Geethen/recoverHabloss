# Mapping the loss and recovery of vegetation, 2018–2024

**A working report on the current model, its accuracy, and what we are unsure
about.** Version of 29 July 2026. Supersedes the version of 28 July 2026.

## What this document is, and what we are asking for

We have built a model that predicts, for every 10 m pixel on the ground, how the
land cover changed between 2018 and 2024 — specifically, whether it moved between
**Nature**, **Cropland** and **Artificial** (built-up, sealed) land. This report
describes how it works, how accurate it is, and where it fails.

It is written for ecologists and remote-sensing colleagues who understand models
in outline but have not worked on this project. No prior knowledge of our code is
assumed, and no jargon internal to the project is used.

**We would particularly value your reaction to five things:**

1. Whether the accuracies in §6 are good enough for the ecological questions you
   would want to ask of a map like this — and if not, what the threshold would be.
2. **Which of the two built-up error modes in §6.4 you would rather have.** This
   is a genuine choice the evidence does not settle, and it is the main reason
   this version of the model is presented alongside the previous one rather than
   simply replacing it.
3. Our reading of the evidence in §7.3, that the limiting factor is the number of
   interpreted plots rather than anything about the model.
4. The caveats in §8, especially the sampling caveat in §2.2, which affects how
   every accuracy number here should be interpreted, and the change-count
   question in §7.6.
5. Anything in §9, the open questions we cannot currently answer.

Section 10 lists what we have already tried and abandoned, with the reasons, so
that a suggestion already tested does not cost anyone time. If your idea is on
that list and you think our test of it was wrong, we would genuinely like to hear
that — several entries rest on a single experiment.

### What changed since the previous version

The model has one structural change: **the part of the network that reads the
satellite record now reads each of the two dates with the same set of weights,
rather than reading a single stacked 2018-and-2024 vector with one wide layer**
(§4.2). Added to it is a small training objective that says, in effect, *a plot
that did not change should look the same in 2018 and in 2024* (§4.3). Everything
else — the data, the plots, the validation scheme, the three-level output, the
five-network average, the fact that Sentinel-2 is never read when a map is made —
is unchanged.

The consequences, in one place:

| | previous version | this version |
| --- | ---: | ---: |
| four-class change-F1 | 0.657 | **0.664** |
| four-class macro-F1 | 0.694 | **0.707** |
| Nature → Artificial F1 (nine-class) | 0.469 | **0.487** |
| Artificial → Nature F1 (nine-class) | 0.407 | **0.442** |
| stable built-up called stable **Vegetation** | **0.196** | 0.225 |
| stable built-up called a **change** | 0.167 | **0.129** |
| gap between training and out-of-sample change-F1 | 0.114 | **0.048** |
| agreement between the four- and nine-class maps | 99.54% | **99.91%** |
| change pixels on our test area | 16,676 | 9,911 |

The first four rows are improvements outside the noise band. Rows five and six
are the trade described in §6.4 and are **not** a strict improvement. The last
row is a 41% reduction in mapped change with no reference data anywhere near the
mapped area to say which count is closer to the truth (§7.6). We think the
evidence favours this version, and we are not certain of it.

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
pixels) takes **5.9 seconds** on one GPU, against roughly nine minutes for the
same model used with the detail branch active. The previous version of the model
took 4.6 seconds; the shared encoder is fractionally more expensive to run and the
difference is of no practical consequence.

Whether that branch is earning its place at all is discussed honestly in §7.2.

### 4.2 The change in this version: one reader for both dates

The embedding branch used to receive the 2018 vector, the 2024 vector and their
difference as one stacked list of 192 numbers, and had to work out for itself that
the first two blocks were the same 64 measurements six years apart. It was never
told.

It is now told. A single small network — one set of weights, which we will call
the *encoder* — is applied **separately to the 2018 vector and to the 2024
vector**, producing a 128-number summary of each date. The rest of the model then
reads five things:

| what the classifier reads | what it is for |
| --- | --- |
| the 2018 summary | what was there at the start |
| the 2024 summary | what is there at the end |
| their difference | how much moved, and in which direction |
| the size of that difference | how much moved, regardless of direction |
| the **angle** between the two summaries | a single number for "do these two dates look like the same place?" |

plus the raw difference of the original embeddings, which is kept because it
still earns its place (§7.1).

Two reasons this is the right shape for this problem. Sharing one encoder across
the two dates roughly halves the number of parameters in the first layer without
discarding any information, and the evidence in §7.3 says this model is short of
data rather than short of capacity. And it makes the *pair* of dates something the
training objective can address directly, which is what §4.3 does.

The description of the transition classes as `from → to` is preserved throughout:
the classifier still sees both endpoint summaries, not only their difference. An
earlier variant that showed it only the difference was clearly worse, for the
obvious reason — a model that cannot tell stable Nature from stable Artificial
cannot tell Nature → Artificial either.

### 4.3 A second, small training objective: dates should agree where nothing happened

Alongside the usual classification objective, training now asks that the two date
summaries of a **stable** plot point in the same direction, and that those of a
**changed** plot point at least some distance apart. That is one scalar per plot,
supervised by a label we already have.

The point of it is that this signal is carried by the *stable majority* of the
plots — 81% of them — so it costs nothing on the rare transitions that are the
target of the work. It pushes the encoder toward describing the ground rather
than the acquisition: differences in sun angle, atmosphere and phenology between
two years are exactly what it penalises.

It is deliberately weak. Its weight was set by measuring the size of the two
objectives before running anything, so that it contributes about a tenth of the
total — a nudge, not a second goal. Running it at three times that strength gives
no further gain and a wider spread between repeats.

A formally different version of the same idea — asking that the two dates'
summaries be *statistically* interchangeable across a batch of stable plots,
rather than aligned plot by plot — performs identically to within noise. We read
that as evidence that the lever is the general pressure toward year-invariance and
not either specific formula. Combining the two adds nothing and roughly doubles
the run-to-run spread, so we use one.

### 4.4 One prediction, read at three levels

The network predicts the nine transitions. The coarser readings are then obtained
**by addition, not by a second model**: the probability of "Vegetation →
Artificial" is the sum of the probabilities of "Nature → Artificial" and
"Cropland → Artificial", and the probability of "changed" is the sum over all six
transitions that involve a change.

Training penalises errors at all three levels at once — nine-class, four-class,
and changed / unchanged — using a loss function that gives less weight to
examples the model already gets right, a standard remedy for the imbalance in
Table 1.

This remains the single most valuable design decision in the model (§7.1).
Because the coarse readings are sums of the fine one, the maps are consistent
with each other by construction, and the plentiful changed/unchanged signal helps
steady the nine-way problem instead of leaving it to be learned from 46 plots of
Artificial → Cropland.

Training is short and deliberately so: 30 passes over the data. Beyond roughly 30
to 50 passes, accuracy on the change classes declines, because the network starts
fitting inconsistencies in the interpretation of the Cropland/Nature boundary
rather than the signal.

### 4.5 Five networks, not one

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

Five measures are reported, and each exists because the others hide something:

- **change-F1** — the balance between precision and recall for the changed /
  unchanged reading, where *recall* is the share of truly changed plots that were
  found and *precision* is the share of predicted changes that are real. F1 is
  their harmonic mean; 1.0 is perfect. This is our headline figure.
- **macro-F1** — the unweighted average of F1 over the four coarse transitions,
  so an apparent gain cannot come from the stable class that holds 71% of plots.
- **per-transition F1 on the four commissioned transitions** — Nature →
  Artificial, Cropland → Artificial, Artificial → Nature, Artificial → Cropland,
  averaged without weighting. Added because change-F1 is dominated by the 4,550
  stable Vegetation plots and can be flat while the classes the work exists to
  find move substantially.
- **stable built-up recall** — how often existing built-up land is recognised as
  such. Added after visual inspection revealed an error that change-F1 is
  structurally blind to: a model can score well on change while systematically
  misreading existing towns as vegetation.
- **the two ways stable built-up goes wrong**, reported separately: called stable
  *Vegetation*, or called a *transition*. §6.4 explains why these must never be
  summed. Reporting only the first hid a real difference between the two model
  versions for several iterations.

Every experiment is repeated from five random starting points, and differences
smaller than ±0.005 are treated as noise rather than as results. This rule was
learned the hard way: several apparently clear findings in this project reversed
when repeated, including one that reversed between three repeats and five.

## 6. Accuracy

### 6.1 Headline

**Table 2. Out-of-sample accuracy, five repeats, full training set.** The
previous version of the model is shown for comparison, on identical folds, plots
and repeats.

| reading | measure | this version | previous |
| --- | --- | ---: | ---: |
| four-class (the map we recommend using) | change-F1 | **0.664 ± 0.002** | 0.657 ± 0.003 |
| four-class | macro-F1 | **0.707 ± 0.004** | 0.694 ± 0.002 |
| nine-class | change-F1 | 0.597 ± 0.004 | 0.596 ± 0.003 |
| nine-class | macro-F1 | 0.440 ± 0.004 | 0.439 ± 0.002 |

The four-class gains clear the ±0.005 noise band; the nine-class aggregates are
ties. The change in the nine-class reading is a redistribution between classes
rather than a net gain — see §6.2.

The two models also sit at different operating points, which matters more for the
map than the F1 values do:

| | change precision | change recall | plots called changed |
| --- | ---: | ---: | ---: |
| this version | **0.651** | 0.679 | ~923 |
| previous | 0.597 | **0.730** | ~1,081 |

against **885 genuinely changed plots**. The previous version over-calls change by
22%, this one by 4%. §7.6 is about what that does to a map.

### 6.2 Per class

"Training F1" is measured on the same plots the model was fitted on. The gap
between the two columns indicates how much of the model's apparent skill does not
transfer to new places.

| four-class transition | plots | out-of-sample F1 | training F1 | gap | previous version, out-of-sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vegetation → Vegetation | 4,550 | 0.92 | 0.93 | 0.01 | 0.91 |
| Vegetation → Artificial | 716 | 0.70 | 0.74 | 0.04 | 0.69 |
| Artificial → Artificial | 979 | 0.70 | 0.73 | 0.04 | 0.68 |
| Artificial → Vegetation | 169 | 0.52 | 0.58 | 0.07 | 0.49 |

| nine-class transition | plots | out-of-sample F1 | training F1 | gap | previous version, out-of-sample |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nature → Nature | 2,532 | 0.76 | 0.79 | 0.03 | 0.75 |
| Cropland → Cropland | 1,661 | 0.72 | 0.76 | 0.04 | 0.72 |
| Artificial → Artificial | 979 | 0.70 | 0.74 | 0.04 | 0.68 |
| Cropland → Artificial | 333 | 0.61 | 0.65 | 0.05 | 0.59 |
| Nature → Artificial | 383 | 0.49 | 0.57 | 0.08 | 0.47 |
| Artificial → Nature | 123 | 0.44 | 0.49 | 0.05 | 0.41 |
| Nature → Cropland | 243 | **0.24** | 0.29 | 0.05 | **0.31** |
| **Cropland → Nature** | 114 | **0.00** | 0.00 | 0.00 | 0.01 |
| **Artificial → Cropland** | 46 | **0.00** | 0.00 | 0.00 | 0.00 |

Every class improves except **Nature → Cropland**, which falls from 0.31 to 0.24.
That transition and Cropland → Nature are the two directions across the boundary
our own reverification shows interpreters disagree about (§8.4); the model has
become less willing to call either of them, which improves the classes on both
sides and costs this one. It is the clearest single cost of this version, and we
report it as a cost rather than folding it into the aggregate.

The ordering is otherwise almost exactly the ordering of the plot counts.
Conversion of vegetation to built-up land, the target of the project, is recovered
at F1 = 0.70 in the four-class reading. Reversion — built-up land returning to
vegetation — is recovered at 0.52, and it is the rarest class in the sample.

**The gap between training and out-of-sample accuracy has more than halved**
against the previous version — 0.04 on average across the four-class transitions
here against 0.10, and 0.07 at worst against 0.16. Fitting two dates with one shared reader is a substantial
reduction in the number of free parameters, and the effect is what a reduction in
over-fitting looks like: the out-of-sample numbers rose while the training
numbers fell. The network is not memorising the training plots; it is short of
information about them. §7.3 is about what that implies.

### 6.3 Two of the nine classes do not work at all

Artificial → Cropland (46 plots) scores **exactly zero** out of sample at every
training-set size and in every repeat, and Cropland → Nature (114 plots) also now
scores exactly zero. Their *training* accuracy is zero too: the network does not
even memorise them, it absorbs them into neighbouring classes.

The map confirms it. Over our test area, across two independent runs, both classes
are painted on **zero** of 2.95 million pixels, and Nature → Cropland on 25 and 20
pixels respectively. **The nine-class map is in practice a six-class map.** The
plot-level evidence predicted exactly which classes would vanish, before the map
was consulted.

We read this as a limit of the reference data rather than a fault in the model.
These are the classes on the Cropland/Nature boundary that our own reverification
shows interpreters disagree about, and they have the smallest sample sizes in the
set. §6.5 describes a decision rule that partly recovers one of them without
retraining anything.

### 6.4 The one place the previous model is better, and it is a genuine choice

Of the 979 plots that were built-up at both dates, here is where the errors go:

| | recognised correctly | called stable **Vegetation** | called a **change** |
| --- | ---: | ---: | ---: |
| this version | **0.646** | 0.225 | **0.129** |
| previous version | 0.637 | **0.196** | 0.167 |

This version recognises slightly more stable built-up land, and **fabricates 23%
fewer habitat-loss events on it** — but when it does err, it errs more often
toward Vegetation. Four separate attempts to close that second column without
giving up the rest failed, including adding the built-fraction variables that fix
it on the previous architecture; it appears to be a property of reading both dates
with one shared encoder rather than a missing input.

**These are not equivalent failures, and which one is worse is a decision about
the product rather than about the model.** A stable built-up plot returned as
`Vegetation → Artificial` is a fabricated habitat-loss event: it enters the map as
the very thing being counted. One returned as stable Vegetation is a
misclassification that invents nothing — it puts a town in the wrong stable class,
where nobody counting habitat loss will look at it.

Our view is that the second error is the cheaper one for this product, which is
why this version is presented as the recommended one. That view is the main thing
we would like challenged, because judgement of the previous model's maps has been
made largely on exactly the first column — how much of a city is read as
vegetation is highly visible on screen, and fabricated change is not.

A separate, free adjustment (a per-class decision cost, fitted on held-out folds)
recovers part of the gap on both versions but does not close it: the previous
model moves further under the same treatment than this one does.

### 6.5 An optional re-read of the nine-class map

The nine-class prediction is a full probability distribution, and everything above
reports only its most likely entry. Re-reading it with per-class decision costs —
fitted on the other folds and applied to the held-out one, so the estimate stays
honest — costs nothing, requires no retraining, and changes only the nine-class
map:

| | standard read | cost-adjusted read |
| --- | ---: | ---: |
| average F1 over the four commissioned transitions | 0.385 | **0.432** |
| Artificial → Cropland | 0.000 | **0.21** |
| Nature → Artificial | 0.487 | 0.493 |
| Cropland → Artificial | 0.609 | 0.591 |
| Artificial → Nature | 0.442 | 0.432 |
| four-class reading | unchanged | unchanged |

The dead class is not, in fact, unreachable — it was merely never the most likely
answer anywhere. There is enough signal in the distribution to recover it at
F1 = 0.21, paid for with about 0.02 on each of two of the other transitions.

Three honest caveats. The costs are tuned on the same measure they are then scored
on, and although the held-out fitting makes that a fair estimate rather than a
circular one, it is a shift of operating point and not new information. The
run-to-run spread on that measure roughly quadruples, so a single-repeat reading
of this adjustment is worthless. And **we have not yet produced a map with it**
for this version of the model, so the claim is a plot-level one.

We would use it only if the nine-class map is wanted for its own sake. The
four-class map is bit-identical either way.

## 7. What we have learned, including the uncomfortable parts

### 7.1 What each design decision is worth

Every component was priced by removing it and refitting on the same folds from the
same five random starting points, so the comparisons are like-for-like. The last
three rows were measured on the previous architecture and have not been re-run on
this one; we have no reason to think they moved, and we flag them rather than
imply otherwise.

| removing this component costs | change-F1 | measured on |
| --- | ---: | --- |
| the shared date encoder **and** the objective that depends on it (§4.2–4.3), i.e. reverting to the previous version entirely | **−0.008** | this version |
| the "dates should agree" objective (§4.3) | −0.001 change-F1, but **−0.004 macro-F1 and +0.017 on the built-up error** | this version |
| the entire Sentinel-2 apparatus, versus embeddings alone | −0.001 change-F1, −0.006 macro-F1 | this version |
| the difference between the 2018 and 2024 embedding vectors | **−0.004** | this version, without Sentinel-2 |
| the three-level training objective (§4.4) | −0.020 | previous version |
| averaging over whether to trust the detail branch | −0.019 | previous version |
| ten of the thirty training passes | −0.028 | previous version |

Three of these deserve comment.

**The "dates should agree" objective is nearly invisible on the headline figure
and is not a null result.** It is worth +0.001 change-F1, which is inside the
noise band — but +0.004 on macro-F1, +0.008 on stable built-up recall, and
−0.017 on the built-up-as-vegetation error. Reported on change-F1 alone it would
have looked like a wasted component. This is the clearest case in the project of
a single headline metric being unable to see a real effect, and it is why §5 now
lists five measures rather than three.

**The difference block is now nearly redundant, where before it was essential.**
Removing it from the previous architecture cost 0.048 change-F1, the single most
damaging change we ever found — that model could not form the subtraction well
enough for itself. Removing it here costs 0.004, an order of magnitude less: the
shared encoder has largely internalised it. It is still worth keeping, and the old
finding should no longer be quoted as a general one.

**The three-level objective remains the largest single lever we know of**, larger
than the entire second data source, and it costs nothing at prediction time.

### 7.2 An honest note on Sentinel-2

In the previous version of the model, Sentinel-2 measurably bought nothing: 0.6557
with it against 0.6574 without, which is a tie favouring the simpler model. The
detail branch was kept on the basis of the map alone.

That has changed, though not by much. Measured on plots, against the same model
trained with no Sentinel-2 at any point:

| | with Sentinel-2 | embeddings only |
| --- | ---: | ---: |
| change-F1 | 0.664 | 0.663 |
| macro-F1 | **0.707** | 0.701 |
| Artificial → Nature F1 | **0.442** | 0.418 |
| Artificial → Cropland F1 | 0.000 | 0.000 |
| stable built-up called Vegetation | **0.225** | 0.234 |
| **Nature → Artificial F1** | 0.487 | **0.502** |

change-F1 is still a tie. macro-F1 now clears the noise band, as does the gain on
the recovery class. But the cost lands on **Nature → Artificial** — the first of
the transitions this work was commissioned to find — where the embeddings-only
model is better by 0.015. The two models are genuinely different rather than one
dominating the other, and if Nature → Artificial were the only thing that mattered
we would ship the simpler one.

**So this report claims a small measured gain from Sentinel-2 on the aggregate
readings, no gain on change-F1, and a measured loss on one commissioned
transition.** The branch also carries the appearance argument that justified it
before — on visual inspection it produces boundaries that fall more often on real
edges in the imagery. No interpreted plot falls inside our mapped area (§8.1), so
the map itself cannot be scored, and that argument cannot be made numerical.

This is one of the places we would most welcome an outside view.

### 7.3 The clearest result: the limit is the number of labelled plots

The most informative experiment we ran was also the simplest — refit the model on
5%, 10%, …, 100% of the training plots and watch the accuracy curve. Every class
that works at all is still improving at 100% of the data, with no sign of
levelling off.

| reading | gain per doubling of labelled plots | previous version |
| --- | ---: | ---: |
| four-class change-F1 | **+0.021** | +0.026 |
| four-class macro-F1 | +0.017 | +0.020 |
| nine-class change-F1 | +0.022 | +0.029 |

The steepest classes are again the change classes — Artificial → Vegetation
(+0.043), Artificial → Nature (+0.044), Nature → Artificial (+0.031) — that is,
the ones the project exists to map.

The slopes are slightly *shallower* than the previous version's, which is what
should happen: this model starts from a better place at every training-set size,
so there is less headroom below the same ceiling. The conclusion is unchanged.
**Every modelling idea we have tested, this one included, moved change-F1 by less
than 0.01; doubling the interpreted sample is worth roughly three to five times
everything the entire model-development effort has produced.**

Two further pieces of evidence now point the same way, and both are new:

- **The over-fitting explanation is now largely excluded.** The gap between
  training and out-of-sample accuracy has more than halved (§6.2) while
  out-of-sample accuracy rose. The model is not memorising 6,414 plots and
  failing to generalise; it is short of information about them.
- **Unlabelled data does not substitute for labelled data.** We added 200,000
  unlabelled pixel-pairs from the mapped area to the "dates should agree"
  objective, which needs no class label. It bought nothing: +0.001 change-F1 at
  low weight — inside noise, with every commissioned class flat — and at high
  weight it *traded the commissioned classes away*, with Nature → Artificial
  falling from 0.50 to 0.49 and stable built-up recall declining steadily as the
  weight rose. It behaves as a mild regulariser and nothing more. What this
  problem is short of is **interpretations**, not satellite pixels.

We also tested whether labelled land-cover data from *outside* the project could
substitute. A 13,000-point sample from a global reference product, harmonised to
our legend, passes a stringent agreement check against our own interpreters — but
adding it to training moved nothing that a control using only our own plots did
not move at least as well, and at strength it traded away the commissioned
classes in the same pattern as the unlabelled experiment. The gain that looked
like it came from the external data came from the extra piece of model structure
that consumed it.

We believe this is the main finding to act on, and we would like it challenged if
you disagree.

### 7.4 The two map readings now agree almost exactly

Over our test area the four-class map calls 9,911 pixels changed and the
nine-class map 10,132, with **9,519 changed in both** — about 4% and 6% of each
map's change pixels are absent from the other. For the previous version the
figure was 13% on both sides (16,676 and 16,609 change pixels, 14,522 in common).
Taken over all 2.95 million pixels the two readings now agree on **99.91%** of
them — a disagreement of 2,676 pixels, against 13,529 for the previous version at
99.54%.

The residual cause is arithmetic rather than a fault: taking the most likely of
nine classes and then pooling is not the same as pooling first and taking the most
likely of four. Three fine classes at 30/30/35% elect the single class at the fine
level and the pair of classes after pooling. The previous version left three
classes in near-ties far more often; the shared encoder makes the nine-way
distribution sharper, and the disagreement mostly disappears with it.

Practical guidance is unchanged in principle and much less consequential in
practice: **use the four-class map for anything about *whether* change happened**,
since that is the reading whose accuracy we measured; use the nine-class map to
ask *what kind*, on pixels the four-class map has already called changed. With
this version the two will almost always say the same thing.

### 7.5 How much of a map is real

The same model, refitted from five *different* random starting points and mapped
over the same pixels, agrees with itself on **85% of change pixels** (0.852,
against 0.842 for the previous version). The total number of change pixels moves
by about 9% between the two runs (9,911 against 9,018) for an identical recipe.

This is the noise floor of the product. Two maps differing by less than this are
not different in any way the data supports. We have made this mistake ourselves:
we once read an 83% agreement between two candidate models as a replication
failure, when in fact each model reproduced itself at 84%, and two maps that each
reproduce themselves at 84% will agree with each other at about 84%. **Any
comparison of two maps should first establish what each map's agreement with
itself is.**

If you take these maps and compare them against another product, this is the
number we would ask you to bear in mind.

### 7.6 This version maps 41% less change, and we cannot say who is right

The two model versions produce genuinely different maps. Over 2.95 million pixels,
replicated on two independent sets of five random starting points:

| | change pixels, run A | run B | share of area |
| --- | ---: | ---: | ---: |
| this version | 9,911 | 9,018 | 0.34% / 0.31% |
| previous version | 16,676 | 15,841 | 0.56% / 0.54% |

The two maps agree with each other on only **58%** of change pixels, far below
either model's ~85% agreement with itself, replicated across both runs. **This
difference is real and is not a random draw.**

**Two things say it is calibration rather than erasure.** The plot metrics
predicted it, through the operating point rather than through F1: this version
calls change on 4% more plots than are truly changed, the previous one on 22%
more (§6.1). On the labelled data, the lower count is the *closer* of the two to
truth. And the maps are not smoother — edge density (0.090 against 0.092), median
patch size (6 pixels for both) and high-frequency content are unchanged; there
are simply fewer change patches (1,831 against 2,239 per million pixels). Every
previous intervention that reduced change on this project reduced it by
smoothing, with no compensating gain in precision, and looks nothing like this.

**Two things say we should not be confident.** The plot sample has a 13.8% base
rate of change and the mapped area has about 0.5%; a precision advantage measured
at one base rate transfers in direction but not in magnitude to the other. And no
interpreted plot falls inside the mapped area, so the 6,765-pixel difference is
**unadjudicable** — exactly as an earlier 16% reduction from a different data
source was, and remains.

If more change pixels are wanted, the honest lever is the decision threshold,
which is already implemented and tunable; this version's higher precision is
precisely what buys room to lower it without flooding the map. We have not chosen
a value, because there is nothing to choose it against.

## 8. Limitations

These bound what the maps can be used for. We would rather over-state them here
than have someone discover them later.

1. **No interpreted plots fall inside the mapped area.** Every accuracy in this
   report comes from plots elsewhere in the world. Nothing here measures accuracy
   *in the place we mapped*; local quality has been judged visually and by
   internal consistency only. Acquiring interpreted plots inside a mapped area is
   the single most valuable outstanding action, and it would settle several open
   questions at once — including §6.4 and §7.6, which are currently the two
   largest open decisions.

2. **Pixel counts are not area estimates.** Counting the pixels of a classified
   map is a biased way to estimate area, and the bias is worst for exactly the
   rare classes of interest here. Defensible areas come from design-based
   estimation on the plot sample, which is a separate strand of the project.
   **These maps are for locating change, not for quantifying it.** This applies
   with particular force to §7.6: the fact that one map has fewer change pixels
   than another is not an area estimate and should not be read as one.

3. **Accuracies describe our sample, not the land surface** (§2.2). The plots
   over-represent places where change was expected.

4. **The Cropland/Nature boundary carries interpreter noise.** Reverification
   shows the two classes are confused with one another in the reference data
   itself. This places a ceiling on nine-class accuracy that no model change can
   lift, and it is why the four-class reading — which pools them — remains the
   defensible one. Within the nine-class map, trust the vegetated / built-up
   distinction more than the Nature / Cropland one. This version is measurably
   *more* cautious about that boundary than the previous one (§6.2), which we
   read as appropriate and which does cost one transition.

5. **Do not spatially smooth the output.** Edge-aware filtering makes boundaries
   visibly crisper and deletes 12–22% of the change class in doing so. Change
   pixels are the model's least confident predictions — 35% of them are within
   0.1 probability of flipping, against 7% of stable pixels — so any procedure
   that lets neighbouring pixels vote removes them first. Everything that has
   *added* change acted on the model's inputs or on its handling of uncertainty;
   nothing that acted on the finished map ever has.

6. **The nine-class legend over-promises.** Two of its entries are never painted
   (§6.3), and a third is painted on about 20 pixels in 3 million. This should be
   stated whenever the nine-class map is shown. §6.5 partly recovers one of them
   at the plot level; that has not yet been carried onto a map.

7. **Stable built-up land is the weakest part of this version** (§6.4), and it is
   the part of a map that is most visible to the eye.

## 9. Open questions we would like help with

1. **Which built-up error would you rather have?** (§6.4.) Fewer fabricated
   habitat-loss events on existing towns, at the price of more towns falling into
   the stable-Vegetation class — or the reverse. We have made a choice; the
   evidence does not compel it, and it is the single most consequential
   preference we would like recorded from outside the project.
2. **Is a change-F1 of 0.66 useful?** For which ecological questions, and at what
   spatial aggregation? A per-pixel accuracy of this level may be perfectly usable
   for landscape summaries and unusable for site-level inference — we would like
   to know where you would draw that line.
3. **Is the 41% reduction in mapped change a fix or a fault?** (§7.6.) The plot
   evidence says the newer model's change count is better calibrated, and says so
   at a base rate 25× higher than the map's. We cannot demonstrate it on the map
   without interpreted plots inside the mapped area. This is the same gap as
   limitation 1, and it is the most consequential thing we do not know.
4. **How should accuracies be weighted?** (§2.2.) Sample-weighted accuracies
   would be more honest about performance across the land surface, and much less
   precise for the rare classes. Which do you want to see?
5. **Where should new interpretation effort go?** Our view is: the two failing
   classes, the Cropland/Nature boundary, and plots inside a mapped area. That
   trades breadth for the specific weaknesses we have identified, and it is worth
   challenging. §7.3 is now unusually firm that this, and not modelling, is where
   the remaining accuracy is.
6. **Is the three-class legend the right one?** It was chosen so that two sampling
   programmes could be combined. A Forest / Other Nature split on the 2018 date is
   available in part of the reference data and would support statements about what
   kind of nature is being lost — but only in one direction.
7. **Would you use the nine-class map?** Given §6.3 it may still be more of a
   liability than an asset outside farmland-rich regions, although the internal
   inconsistency that was the other half of the objection is now largely gone
   (§7.4).

## 10. What we have already tried that did not work

Listed so that no one spends time on a path we have already closed. Each was
tested on the same plots under the same validation scheme, most at five random
starting points. If you think a test was badly designed, please say so — several
of these rest on a single experiment.

**On the data:**

- **Unlabelled satellite data at scale.** 200,000 unlabelled pixel-pairs added to
  the year-agreement objective bought nothing, and at strength cost the
  commissioned classes (§7.3).
- **Labelled land-cover data from outside the project.** A global reference
  product, harmonised to our legend, passes a stringent agreement check and then
  adds nothing a control using only our own plots does not add (§7.3). A second
  candidate source turned out to cover 8 of our 83 spatial blocks and could not
  be cleared as a global pool at all. A structural obstacle worth recording: both
  end before 2024, so they can only ever supervise the earlier date of a
  `from → to` target.
- **A second, sharper embedding product** as the detail source instead of
  Sentinel-2. It performed well where it existed, but only 36% of plots had it at
  both dates, and the gap could not be worked around.

**On the architecture:**

- **Letting the network learn its own texture** from small image patches. Two
  distinct attempts. Flattening the pixels of a patch into a long list of
  variables is catastrophic. A proper small image encoder over the same patches is
  merely *no better* than our hand-computed statistics, at ten times the training
  cost, and eight free rotations and reflections of every patch move the result
  by literally nothing.
- **Separate calibration of the two dates** inside the shared encoder, to recover
  the built-up error in §6.4. Refuted the explanation it was built to test.
- **Collapsing the three stable classes into one**, on the intuition that a
  smaller label space is an easier problem. Strongly negative, and instructively
  so: the stable classes are the state-recognition signal that the change classes
  borrow from. A model that cannot tell stable Nature from stable Artificial
  cannot tell Nature → Artificial either. Fewer classes, less signal.
- **Mixture-of-experts architectures, added training noise, knowledge
  distillation, supervising the two dates separately, and recurrent models over
  the annual series.** All within noise or worse.
- **More capacity, or longer training.** Both worse; see §4.4.
- **Additional derived variables** (per-band products, normalised differences).
  Tie with or underperform the difference variables we already use.

**On the decision rule and the output:**

- **Smoothing or sharpening the finished map** (edge-aware filtering, applied both
  to the class map and to the underlying probabilities). Delivers crisper
  boundaries and destroys 12–22% of the change class. See limitation 5.
- **Class-balanced sampling** during training. Underperforms handling the
  imbalance in the loss function instead.
- **Re-training only the final classifier layer on balanced classes**, a standard
  remedy for long-tailed problems. It costs 0.018 change-F1 and nearly doubles
  the rate of fabricated habitat loss on existing built-up land. It does not reach
  the dead classes at all.
- **A hard NDVI threshold** to separate built-up from vegetation. The threshold
  itself is well calibrated (0.31), and as a *proportion* over a 30 m window it is
  our single most useful built-up variable — but as a rule it cannot fix the
  built-up plots the model misreads, because those plots are genuinely green in
  the reflectance and built-up in the interpretation. That is a disagreement
  between the label and the pixel, and no spectral index can win it.
- **Combining the two decision-rule adjustments** of §6.4 and §6.5. They correct
  the same thing and land between their parts — the third time on this project
  that two mechanisms aimed at one problem have composed to less than either.

One recurring pattern is worth flagging, because it may be the most transferable
thing we have learned: **interventions that make the model more cautious remove
real change before they remove false alarms**, because change pixels are a small
fraction of the surface and the model is barely confident about them. Anything
that regularises, smooths or votes will look like an improvement on every
aggregate measure while quietly deleting the class of interest. §7.6 is the one
case on this project where a reduction in mapped change came with a measured gain
in precision instead — which is exactly why it is reported separately and at
length rather than as an improvement.

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
| **shared encoder** | one set of network weights applied separately to each of the two dates, so that position *i* means the same measurement in both (§4.2) |
| **year-agreement objective** | the additional training term asking that an unchanged plot look the same at both dates (§4.3) |
| **privileged information** | a data source available during training but deliberately not used when predictions are made (§4.1) |
| **NDVI / NDWI** | standard vegetation and water indices computed from reflectance |

## 12. Availability

The code, the maps described here, and the full research record — including every
negative result and the experiment that produced it — are held in the project
repository and can be shared on request. Learning-curve figures for §7.3, and both
versions' maps for §6.3, §7.4 and §7.6, are available as separate files. The
previous version of the model remains reproducible and its maps are retained, so
any comparison in this report can be repeated.

Comments, objections and suggestions are all welcome, including on the points this
report presents as settled.
