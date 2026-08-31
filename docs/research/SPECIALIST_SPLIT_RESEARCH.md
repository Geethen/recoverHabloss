# Frequency-split specialists — section V

Two models instead of one: a head model over the classes that already reach an
acceptable accuracy, and a tail model over the ones that do not. The split is
not a judgement call — the base model's own per-class table has a clean gap:

| head (F1 ≥ 0.51) | n | F1 | | tail (F1 ≤ 0.44) | n | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Nature → Nature | 2532 | 0.766 | | Artificial → Nature | 123 | 0.440 |
| Cropland → Cropland | 1661 | 0.729 | | Nature → Cropland | 243 | 0.272 |
| Artificial → Artificial | 979 | 0.700 | | Cropland → Nature | 114 | 0.000 |
| Cropland → Artificial | 333 | 0.611 | | Artificial → Cropland | 46 | 0.000 |
| Nature → Artificial | 383 | 0.509 | | | | |

The cut is taken at that gap and never moved.

**The mechanism is N0's own diagnosis, taken at its word.** N0 states the rare
transitions die because "a class that small cannot support a decision boundary
against 4,200 stable plots under focal loss", and concludes that
`Artificial -> Cropland` is *"a labelling ask"*. Every rare-class idea in ~50
since has kept all nine classes in one softmax and attacked the loss (focal,
cb_focal, Dice), the sampler (G-H), the parameterisation (proto, tau-norm, cRT)
or the decision rule (O3's cost gate, R5's conformal cuts). **None of them
removed the condition N0 named.** A specialist trained on the 526 tail plots
alone never draws that boundary: within the tail, `Artificial -> Cropland` faces
480 rivals rather than 6,368.

**The composition is exact and introduces no new decision rule.** For a
partition of the coarse3 classes into blocks `B`,

```
P(k) = P_base(B) · P_spec(k | B)          k ∈ B
```

— the chain rule. It sums to one, it leaves every block *mass* untouched (so the
merged2 and gate reads barely move), and **it collapses to the base model exactly
when the specialist reproduces the base's own within-block conditional**. There
is therefore no composition-rule confound to control for: the base *is* that
control, and any difference is the specialist's conditional alone.

## Standing table (5 seeds, `full` read, identical folds)

| | change-F1 | macro-F1 | **focus macro** | artStab | as_veg | vegStab→art |
| --- | --- | --- | --- | --- | --- | --- |
| `siam_s2off_cos` (base) | **0.6644** ±0.0024 | **0.7067** | 0.3847 ±0.005 | 0.6458 | 0.2249 | 0.0292 |
| `c3gate_siam_s2off_cos` (O3b, free) | 0.6644 | 0.7067 | 0.4318 ±0.0231 | 0.6458 | 0.2249 | 0.0292 |
| `spec_sharp_tail4_max` (**V0, control**) | 0.6661 ±0.0030 | 0.7102 | 0.3810 ±0.0044 | 0.6468 | 0.2198 | 0.0294 |
| **`spec_tail4` (V1)** | 0.6642 ±0.0024 | 0.7048 | **0.4581** ±0.0105 | 0.6427 | 0.2327 | 0.0284 |
| `spec_tail6` (V2) | 0.6568 ±0.0052 | 0.7026 | 0.4344 ±0.0039 | 0.6400 | 0.2427 | 0.0283 |
| `spec_split45` (V3, two models) | 0.6601 ±0.0037 | 0.7027 | 0.4555 ±0.0079 | 0.6441 | 0.2411 | 0.0293 |
| `c3gate_spec_tail4` (V5) | 0.6642 | 0.7048 | 0.4396 | 0.6427 | 0.2327 | 0.0284 |
| `conf_c3_spec_tail4` (V6) | 0.6642 | 0.7048 | 0.4488 | 0.6427 | 0.2327 | 0.0284 |

**V1 is a win by the standing rule** — +0.073 on its target, and nothing else
moves more than 0.005: change-F1 −0.0002, macro-F1 −0.0019, `art_stable_recall`
−0.0031, `veg_stable_as_art` −0.0008.

## V0 — the control, and it is why this section is readable

A specialist over four classes is *sharper* than a nine-way softmax on the same
rows for reasons that have nothing to do with what it learned — fewer rivals, a
flatter prior, a balanced training set. Under a composition that fixes the block
mass, concentration alone raises the block's arg-max and can win the class
outright. That is a decision-rule effect and it is free.

`spec_sharp_tail4_max` puts **all** of the base's tail mass on the base's own
within-tail arg-max — the sharpest any composition over this block can be, with
nothing trained. **focus macro 0.3810 against the base's 0.3847: flat.** Power-4
sharpening (V0b) is 0.3844, also flat. So concentration explains none of V1, and
the gain is the specialist's conditional.

## Where the gain is: one class, and it is the one N0 called unreachable

Per-class coarse3 F1, 5 seeds, read from the stored labels:

| | n | base | O3b gate | R5f conformal | **V1** |
| --- | --- | --- | --- | --- | --- |
| Nature → Nature | 2532 | 0.763 | 0.762 | 0.684 | **0.763** |
| Cropland → Cropland | 1661 | 0.725 | 0.724 | 0.644 | **0.725** |
| Artificial → Artificial | 979 | 0.696 | 0.694 | 0.662 | 0.695 |
| Nature → Artificial | 383 | 0.487 | 0.493 | 0.506 | 0.487 |
| Cropland → Artificial | 333 | 0.609 | 0.591 | 0.580 | 0.609 |
| Nature → Cropland | 243 | 0.239 | 0.238 | **0.354** | 0.235 |
| Artificial → Nature | 123 | 0.442 | 0.432 | 0.437 | 0.422 |
| Cropland → Nature | 114 | 0.000 | 0.000 | 0.062 | 0.000 |
| **Artificial → Cropland** | **46** | **0.000** | 0.211 | 0.171 | **0.314** |
| focus macro | | 0.3847 | 0.4318 | 0.4235 | **0.4581** |
| all-9 macro | | 0.4402 | 0.4607 | 0.4556 | **0.4722** |

**`Artificial -> Cropland` — 46 plots, returned at 0.000 by every model in the
ledger and named "a labelling ask" by N0 — reaches F1 0.314.** It is reachable
from the modelling side after all, and what unlocked it is exactly the condition
N0 identified: not competing against 4,200 stable plots.

**And it is not bought by calling more plots `Artificial -> Cropland`:**

| | precision | recall | plots predicted (46 true) |
| --- | --- | --- | --- |
| O3b cost gate | 0.265 | 0.209 | 36.2 |
| **V1** | **0.524** | 0.226 | **20.0** |

Twice the precision at the same recall, on half as many calls. The gate reaches
the class by lowering its bar; the specialist reaches it by ranking it better.

**The cost is `Artificial -> Nature`**, recall 0.592 → 0.478, and the mechanism
is legible: both classes share `from = Artificial`, so inside the tail block the
specialist's redistribution moves mass between exactly those two. Net across the
four commissioned transitions it is strongly positive, but this is a trade and
`Artificial -> Nature` is the recovery class the project is named for.

**Stability, which on a 46-plot class is the first thing to doubt:**

| | Art→Crop F1, per seed | mean ± sd |
| --- | --- | --- |
| O3b cost gate | 0.228, 0.074, 0.358, 0.156, 0.241 | 0.211 ±0.095 |
| **V1** | 0.344, 0.262, 0.348, 0.299, 0.319 | **0.314 ±0.032** |

**A third of the variance, and every seed above the gate's mean.** A tuned cost
multiplier on 46 plots is an unstable instrument; a model of those 46 plots is
not. Focus macro likewise: ±0.0105 against the gate's ±0.0231.

## V2/V3/V4 — the cut, and whether the *head* model is worth anything

* **V2 `spec_tail6`** (every transition in the tail, only the three stable
  classes left to the base): **0.4344, below V1's 0.4581**, and it costs
  change-F1 (0.6568) because `Nature -> Artificial` falls 0.487 → 0.429. The
  finer cut wins: putting a class the base already handles at F1 0.51 into the
  specialist's block costs more than it buys.
* **V3 `spec_split45`** — the user's construction in full, a head model *and* a
  tail model with the base reduced to a router. **0.4555 against V1's 0.4581 at
  change-F1 0.6601 against 0.6642.** The head specialist buys nothing and costs
  −0.004 change-F1.
* **V4 `spec_split36`**: 0.4322, the worst of the four.

**So the answer to "two models" is: one model, and only for the tail.** The head
classes have enough data that a 9-way softmax already serves them; training a
head specialist only removes the tail plots from its training set. The
asymmetry is the finding — the split is worth making on one side of the cut.

## V5/V6 — the compositions, one prediction confirmed and one refuted

Both registered with their predictions before being read.

* **V5 (O3's cost gate on top of V1): 0.4396 — lands between its parts**
  (V1 0.4581, O3b 0.4318), the **fourth** replication of the F7 / N3b / O4
  precedent. Two mechanisms that break the same class compose to somewhere
  between them. Art→Crop precision falls 0.524 → 0.340 as the gate widens what
  V1 had narrowed.
* **V6 (R5f's Mondrian conformal cuts on top of V1): 0.4488 — also below V1, and
  this one was predicted to ADD.** R5's finding was that search and calibration
  break *different* classes, and V1 owns Art→Crop while moving Nat→Crop not at
  all, so the two looked disjoint. They are not: conformal does lift
  `Nature -> Cropland` 0.235 → 0.342 as advertised, but it costs Art→Crop
  precision 0.524 → 0.281 to do it. **Prediction refuted, recorded as such.**

**V1 stands alone.** Nothing composed with it improves on it.

## V7 — it replicates on the deployed model

The O3c pattern, and not optional: a construction measured on one base is an
observation about that base until a second one carries it.

| (5 seeds) | base | O3c gate | **V7 `spec_tail4_deployed`** |
| --- | --- | --- | --- |
| focus macro | 0.3667 | 0.4076 | **0.4367** |
| Artificial → Cropland | 0.000 | 0.156 | **0.278** |
| Cropland → Nature | 0.014 | 0.000 | **0.085** |
| change-F1 | 0.6568 | 0.6568 | 0.6546 |
| artStab | 0.6374 | 0.6374 | 0.6288 |

**+0.070 over its own base and +0.029 over its own gate** — the siamese base gave
+0.073 and +0.026. The construction transfers, and on this base it also lifts
`Cropland -> Nature` off zero. The one thing to note honestly: `art_stable_recall`
−0.0086 here, just outside the ±0.005 band, where on the siamese base it was
−0.0031.

## Status and what is *not* established

**Section V is the first idea in this ledger to move `Artificial -> Cropland`
off zero at high precision, and it is a win at 5 seeds on two bases with a free
control that is flat.** It is also narrow, and the following are the honest
limits:

1. **The entire gain is one class of 46 plots.** Focus macro is an unweighted
   mean of four, so a single class moving 0.000 → 0.314 is +0.079 of it by
   arithmetic. Everything else is flat or slightly down. Anyone reading
   `focus_macro_f1` alone will overstate what this does.
2. **It costs `Artificial -> Nature` recall**, 0.592 → 0.478. Whether buying the
   dead class with the recovery class is worth it is a product decision, not a
   modelling one — the same shape of trade O1 hit, and O1's version was refused.
3. **Serving cost doubles.** A second AlphaEarth-only network runs at inference.
   In absolute terms that is Oslo's ~6.5 s becoming ~13 s, so it is not a
   blocker, but the deployed path in `infer_s2.py` composes one model today and
   would need to compose two.
4. **No map has been made.** Per CLAUDE.md, the ~0.84 self-IoU floor must be
   computed before any two rasters are compared, and `Artificial -> Cropland`
   is one of the two classes that returns **0 pixels** on the deployed Oslo map.
   Whether F1 0.314 on 46 plots turns into visible pixels is untested, and it is
   the obvious next step.
5. **N0's conclusion needs amending, not deleting.** The class was reachable by
   modelling — but at recall 0.226, so N0's "this is a labelling ask" is still
   the right answer to *how do you actually map this class*. What is refuted is
   the stronger claim that no architecture reaches it.

**Tested-negative, do not redo:** the tail6 cut (V2/V4) · a head specialist in
any form (V3/V4 — it buys nothing and costs change-F1) · sharpening the base's
within-block conditional, at power 4 or in the limit (V0/V0b/V0c) · composing V1
with the O3 cost gate (V5) or with Mondrian conformal (V6).

## Reproduce

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python   # the .venv is broken; uv run fails
cd src
$P twotower_lab.py --ideas base_siam_s2off_cos_fine --read full --n-seeds 5  # the cache V1 reads
$P twotower_lab.py --group section-v --read full --n-seeds 5
```

Implementation: `specialist_cv` / `specialist_idea` and the `FREQ_BLOCKS` cut in
`twotower_lab.py`; the control is `sharpen_cv`. The specialists are the same
recipe as their base, restricted to their block's plots — no new architecture,
no new loss, no new hyperparameter.
