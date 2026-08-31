# Siamese endpoint architectures — section N

A different question from every ledger entry before it. `TWOTOWER_RESEARCH.md`
(A–G) and `S2_DETAIL_RESEARCH.md` (S, T, U) both ask **how to fuse two
modalities**. This section asks **how to structure the two dates**.

Every model in the ledger flattens `[2018 | 2024 | diff]` into one 192-column
vector and lets a wide trunk work out for itself that the first two blocks are
the same 64 measurements six years apart. A siamese encoder is *told* instead:
one shared encoder `f`, applied to both endpoints, `z18 = f(x18)`,
`z24 = f(x24)`, and a head over `[z18, z24, z24−z18, |z24−z18|, cos(z18, z24)]`.

Two reasons that is worth the ledger space here specifically:

1. **Parameter sharing is the lever this problem is short of.** The learning
   curves (S19) put the model at **+0.026 change-F1 per doubling of labels** —
   squarely data-limited. Sharing the encoder across dates halves the
   first-layer parameters at the same input width, which is the one capacity
   reduction that costs no information.
2. **It puts the endpoint pair where an objective can reach it.** `cos(z18, z24)`
   is a single scalar with an unambiguous supervision signal — a stable plot is
   one piece of ground measured twice and its two embeddings should agree; a
   change plot's should not. That supervision is carried by the **stable
   majority**, so it costs nothing on the rare transitions that are the target.
   The Barlow Twins variant goes further and needs no class label at all.

## Objective

Change-F1 alone was already shown to be blind to the failures that matter
(`twotower-three-objectives`); on this section it is blind twice over, because
**Nature → Artificial (383 plots) and Artificial → Cropland (46) contribute
almost nothing to a metric dominated by 4.2k stable plots.** Section N is scored
on the commissioned transitions directly. `focus_metrics` in `twotower_lab.py`
writes these into the ledger for every idea, old and new.

### Standing table (5 seeds, `full` read, identical folds)

| | `baseline_aef` | **deployed** `s2off_centre_m3s3_bf` | `siam_cos` | **`siam_s2off_cos`** |
| --- | --- | --- | --- | --- |
| change-F1 | 0.6577 ±0.0043 | 0.6568 ±0.0032 | 0.6630 ±0.0035 | **0.6644 ±0.0024** |
| macro-F1 | 0.6899 | 0.6943 | 0.7011 | **0.7067** |
| `focus_macro_f1` | 0.3739 | 0.3667 | 0.3815 | **0.3847** |
| Nature → Artificial F1 | 0.4775 | 0.4693 | **0.5021** | 0.4871 |
| Cropland → Artificial F1 | 0.5897 | 0.5906 | 0.6062 | **0.6093** |
| Artificial → Nature F1 | 0.4037 | 0.4066 | 0.4176 | **0.4423** |
| Artificial → Cropland F1 | 0.0248 | 0.0000 | 0.0000 | 0.0000 |
| artStab recall | 0.6415 | 0.6374 | 0.6423 | **0.6458** |
| art → veg error | 0.2035 | **0.1957** | 0.2337 | 0.2249 |
| **art → *change* error** | 0.155 | 0.167 | **0.124** | 0.129 |
| veg → art error | 0.0346 | 0.0345 | **0.0291** | 0.0292 |

`siam_s2off_cos` keeps the deployment property that matters: Sentinel-2 is
privileged information at training time and **is never read at inference**, so it
serves at the same cost as the deployed model.

The deployed model is the incumbent, not a candidate; it is in the table so the
siamese line has an honest comparison rather than only a baseline it was tuned
against.

**Read the two built-up error rows together, never one alone.** They trade
against each other, and N11 found the siamese looking like a 4-point regression
on the first while being a 4-point *improvement* on the second — the one that
fabricates habitat-loss events. `art_stable_as_change` exists because that was
invisible for four iterations.

| metric | `baseline_aef`, 5 seeds | why it is here |
| --- | --- | --- |
| `change_f1` | 0.6577 | the historical headline; must not regress |
| `macro_f1` | 0.6899 | four-class merged2 read |
| `fine_f1_nature_to_artificial` | 0.4775 | **the user's first target class** |
| `fine_f1_cropland_to_artificial` | 0.5897 | the other habitat-loss transition |
| `fine_f1_artificial_to_nature` | 0.4037 | recovery — what the project is named for |
| `fine_f1_artificial_to_cropland` | 0.0248 | **the user's second target class; see N0** |
| `focus_macro_f1` | 0.3739 | unweighted mean of the four, so 46 plots weigh as much as 383 |

Counter-checks that must not blow up: `veg_stable_as_art` (did it just flood
Artificial?) and `art_stable_recall`.

## N0 — the finding that constrains the whole section

**`Artificial -> Cropland` has 46 labelled plots and is already dead.**
`baseline_aef` returns it at **1.5% recall**; `siam_shared` at **0%**. This is
not something an architecture fixes, and it should be said before any effort is
spent aiming at it:

* 46 plots across 5 spatially blocked folds is ~9 per test fold. A class that
  small cannot support a decision boundary against 4,200 stable plots under
  focal loss, and the two ideas below that *raise* precision push it to exactly
  zero rather than rescuing it.
* It agrees with the map evidence already in the ledger — two coarse3 classes
  return **0 pixels** on the deployed Oslo raster, and this is one of them.
* `PATCH_SAMPLING.md` already sizes the fix: the change-restricted sampling
  channel rescues one dead class and **not this one**.

`Nature -> Artificial` (383 plots) is a real target and moves with modelling.
`Artificial -> Cropland` is a labelling ask. Everything below is aimed at the
first; the second is reported every run so the claim stays honest, not because
any idea here is expected to move it.

## Ledger

| # | idea | status | result |
| --- | --- | --- | --- |
| N1 | **`siam_shared`** — shared encoder over the two AlphaEarth endpoint blocks, head on `[z18, z24, z24−z18, \|z24−z18\|, cos]`, plus the raw `diff` block so it is information-matched to `baseline_aef`. The gate question: if a shared encoder cannot match a flat trunk on the same inputs, nothing else in this section has anywhere to live. | **PASS (5 seeds)** | **change-F1 0.6593 ±0.0031 vs 0.6577 ±0.0043 — a tie on the headline, at 70% of the seed variance.** Everything else moved the right way: **macro-F1 0.7001 vs 0.6899 (+0.0102, clears the ±0.005 band)**, `Nature -> Artificial` F1 **0.4988 vs 0.4775 (+0.021)**, `Cropland -> Artificial` **0.6057 vs 0.5897**, `Artificial -> Nature` **0.4146 vs 0.4037**, `focus_macro_f1` 0.3798 vs 0.3739. `veg_stable_as_art` *fell* (0.0293 vs 0.0346), so the gain is not Artificial flooding, and `art_stable_recall` held (0.644 vs 0.642). Fits in 7.0 s against the flat trunk's 8.4 s. **The one regression is `fine_change_f1` (0.5858 vs 0.5975, −0.012)** — the coarse3 arg-max gets worse while every individual focus class gets better, which is the arg-max-vs-group-sum non-commutation of S15 showing up on plots instead of pixels. The pattern across the focus classes is **precision up, recall flat or slightly down**: `Nature -> Artificial` recall 0.4804 vs 0.4836 for +0.021 F1. Section N is open. |
| N2 | **`siam_cos`** — add the gate-supervised cosine loss: pull `z18`/`z24` together on stable plots, push them apart past a margin on change plots, the two group terms weighted equally so the 4:1 stable majority does not turn it into a plain similarity regulariser. This is the user's actual hypothesis, and N1 only built the place to put it. | **WIN (5 seeds)** | **change-F1 0.6630 ±0.0035**, the highest AlphaEarth-only number in the ledger: +0.0053 over `baseline_aef` and **+0.0062 over the deployed model**, both clearing the ±0.005 band. `Nature -> Artificial` F1 **0.5021** (+0.025 over baseline, **+0.033 over deployed**) and, unlike N1, **its recall rose too** (0.4919 vs 0.4836 / 0.4783) — the cosine term partly buys back the recall N1 traded for precision. `focus_macro_f1` 0.3815. Weight 0.3 / margin 0.3, preregistered by measuring the loss magnitudes first (the three-level loss runs 2.49 -> 0.71 over 30 epochs, the cosine term 0.36 -> 0.18), so 0.3 makes the auxiliary ~10% of the objective: a regulariser, not a co-objective. **N2b (weight 1.0) is therefore untested and still open.** |
| — | **Comparison against the deployed model**, `s2off_centre_m3s3_bf`, run on these folds/seeds/metrics via `s2off_cv` (the gate-off read, `S2_MASK` zeroed at predict time — scoring it gate-on would score a model that is never served). The registration reproduces the published 15-seed table to within seed noise (change-F1 0.6568 vs 0.6557, macro-F1 0.6943 vs 0.6938, artStab 0.637 vs 0.642), which is what makes the row trustworthy. | see N9 | **The siamese line beats it on every commissioned transition and loses on the built-up error mode.** change-F1 +0.0062, macro-F1 +0.0068, `focus_macro_f1` +0.0148, `Nature -> Artificial` +0.033, `Cropland -> Artificial` +0.016, `Artificial -> Nature` +0.011, `veg_stable_as_art` *down* 0.0291 vs 0.0345. **But `art_stable_as_veg` is 0.234 against the deployed 0.196** — 23.4% of stable built-up returned as stable Vegetation, ~4 points worse on the exact failure the user has judged every map on, and `art_stable_recall` is unchanged (0.642 vs 0.637), so the errors have *concentrated* into Vegetation rather than grown. Also `fine_change_f1` −0.004. No map claim follows from any of this: the ~0.84 self-IoU floor has not been computed for a siamese raster and none of these plots are anywhere near an AOI. |
| N9 | **`siam_builtfrac`** — the diagnostic the comparison forces: is that art->veg regression a missing covariate? Built fraction is the known lever for it (S8: `aef_builtfrac` holds the board's best art->veg at 0.1916), so add it to N2 as flat extras. Reads Sentinel-2 at inference and so gives up the s2off property — a diagnostic, never a candidate. | **NEGATIVE (3 seeds)** | **It does not fix it: art->veg 0.248, if anything worse than N2's 0.234**, with change-F1 0.6608 and macro-F1 0.6991 both flat against N2. The lever that works on a flat trunk does nothing here, so **the regression is structural to the shared encoder rather than a feature gap** — and note it is already present in N1 (0.2398) *before* any cosine loss, so the auxiliary objective is not the cause either. The mechanism worth testing: the flat trunk keeps separate first-layer weights for the 2018 and 2024 blocks and can calibrate an absolute state reading per year, while one shared map cannot. That is precisely the Siamese-vs-pseudo-Siamese distinction in Daudt et al., and it is the next thing to try (N10). |
| N10 | **`siam_pseudo`** — per-year diagonal affine on the encoder input, identity-initialised: Daudt et al.'s pseudo-siamese at the smallest size that tests N9's mechanism. If per-year calibration is what stable built-up needs, this recovers art->veg without giving back the parameter sharing that produced the focus-class gains. | **FLAT (5 seeds)** | **Refutes the mechanism.** art->veg 0.230 vs N2's 0.234 — inside noise, and nowhere near the deployed 0.196. Everything else is a tie too: change-F1 0.6629 vs 0.6630, macro-F1 0.7014 vs 0.7011, `focus_macro_f1` 0.3826 vs 0.3815, `Nature -> Artificial` 0.5049 vs 0.5021. Per-year calibration is not what the shared encoder was missing, so N9's explanation is wrong and the two negatives together say the art->veg number is **not** an encoder defect at all — which is what N11 then established. Worth recording that at **3 seeds this idea showed `Nature -> Artificial` at 0.5101** and looked like the section's best; at 5 it fell to 0.5049. Textbook case for the 5-seed rule. |
| N11 | **Decompose the art->veg "regression"** instead of guessing at a third architecture. Two consecutive negatives (N9, N10) on a single summary number is a signal to check what the number is made of. | **REFRAMES IT** | **`art_stable_as_veg` was hiding the error that matters.** Over the 979 stable built-up plots, where the errors actually go: <br><br>`s2off_centre_m3s3_bf` (deployed): 0.637 correct · 0.196 stable-Veg · **0.167 spurious change**<br>`baseline_aef`: 0.641 · 0.203 · **0.155**<br>`siam_cos`: 0.642 · 0.234 · **0.124**<br>`siam_pseudo`: 0.645 · 0.229 · **0.126**<br><br>The siamese has **equal-or-better recall and 25–30% less false change on built-up** than the deployed model. Its residual errors land in stable Vegetation rather than being called a transition. For a map commissioned to count habitat loss those are not equivalent failures: a stable built-up plot returned as `Vegetation -> Artificial` is a **fabricated habitat-loss event**, while one returned as stable Vegetation is a misclassification that invents nothing. **This is a trade, not a regression, and which side is better is the user's call — but it was invisible while only `art_stable_as_veg` was tracked.** `art_stable_as_change` is now computed in `twotower_metrics.per_class` for every idea, old and new. |
| N12 | **`siam_cos` under the nested cost gate** (F3 / `nested_cost_gate`, already built, free, no retrain), **with N12b running the same gate on the deployed model** so a gated challenger is not compared against an ungated incumbent — which is how a free decision-rule gain gets misattributed to an architecture. | **PARTIAL (5 seeds)** | **The gate is a real free lever on both, and it does not close the gap.** Gated-vs-gated: `art_stable_recall` deployed **0.700** vs siamese 0.662, `art_stable_as_veg` deployed **0.165** vs siamese 0.221. So N11's boundary reading is only half right — the trade is *partly* recoverable post-hoc (siamese art->veg 0.234 -> 0.221, artStab 0.642 -> 0.662) but the deployed model moves further under the same treatment. Meanwhile the siamese keeps its own wins under the gate: `Nature -> Artificial` **0.5021 vs 0.4693**, `art_stable_as_change` **0.1175 vs 0.1348**, `veg_stable_as_art` **0.0334 vs 0.0470**, change-F1 0.6578 vs 0.6559 and macro-F1 0.6994 vs 0.6971 both ties. **Instrument limitation, and it matters:** the gate reweights *merged2* classes, while `focus_macro_f1` and the per-transition metrics are read off the coarse3 arg-max — so the gate leaves every commissioned transition **exactly unchanged** (0.3815 / 0.5021 gated and ungated, to four decimals). It cannot be used to buy the classes this section is scored on. A cost gate at the coarse3 level would be a different instrument and is not built. |
| N3 | **`siam_barlow`** — Barlow Twins redundancy reduction between `z18` and `z24` **on stable pairs only**. Zbontar et al. need two augmented views and a hand-designed augmentation policy; a stable plot supplies two *genuine* views of one unchanged patch for free. Driving their cross-correlation to the identity asks for features invariant to acquisition/phenology and mutually decorrelated, so what survives in `z24 − z18` is change rather than nuisance. | **WIN, and a tie with N2 (5 seeds)** | **change-F1 0.6624 ±0.0030 against N2's 0.6630 ±0.0035 — indistinguishable**, and both clearly above N1's 0.6593. `Nature -> Artificial` 0.5014 vs 0.5021, `focus_macro_f1` 0.3796 vs 0.3815, `Cropland -> Artificial` 0.6070 (the section's best). It also has the **lowest false-change-on-built-up of any model measured, 0.1154** against the deployed 0.1669. **The tie is the finding.** Two formally different objectives — one on the *angle* of a single pair, needing the change label; one on the *cross-correlation across the batch*, needing only stable/change — produce the same gain to within noise. So the lever is neither formulation specifically: it is *any* pressure toward year-invariance on the stable pairs. **That matters because Barlow gets there from strictly less supervision, which is what makes N4 possible.** |
| N3b | **`siam_cos_barlow`** — do the two auxiliaries compose? Preregistered with the prediction that they might not, since they act on the same pair. | **FLAT (5 seeds)** | change-F1 0.6643 ±0.0060 — nominally the highest mean in the section, but **+0.0013 over N2 is well inside noise and the seed variance nearly doubled** (0.0060 against 0.0035 / 0.0030). `focus_macro_f1` 0.3801, between its parts. Replicates F7 exactly: two mechanisms that correct the same thing compose to somewhere between them, and the wider spread is the tell that nothing new is being added. **Do not stack them.** N3 confirms they are one lever wearing two formulations. |
| N4 | **`siam_barlow_ssl`** — the same Barlow term on an **unlabelled** AlphaEarth pool. N3 is limited to the ~4.2k labelled stable plots; N3 also established the term reaches the same accuracy from strictly less supervision, which is exactly what lets it transfer to pixels carrying no label at all. "Assume a random pixel did not change" is right ~99.5% of the time at the deployed map's base rate, so ~1 sampled pair in 200 is a change pair pushed the wrong way on a redundancy term — two orders of magnitude below the signal, and the whole approximation. **The only idea in the section that adds information rather than rearranging it.** No evaluation contamination is possible: zero labelled plots fall inside either AOI bbox (G3/G4). | **NEGATIVE (5 seeds)** | **200k unlabelled Oslo endpoint pairs buy nothing.** At weight 0.3: change-F1 0.6636 against labelled-only Barlow's 0.6624 — +0.0012, inside noise, with every focus class flat. At weight 1.0 (N4b): change-F1 **0.6667**, the section's highest, but bought by trading the commissioned classes away — `Nature -> Artificial` **0.4884, down from 0.5014**, `Cropland -> Artificial` 0.5986 down from 0.6070, and `art_stable_recall` declining monotonically with the unlabelled weight (0.6384 -> 0.6327 -> 0.6268). `focus_macro_f1` is flat at 0.3812 across all three. The seed variance does tighten (sd 0.0021 vs 0.0027), which is what a regulariser looks like — and a regulariser is all this is. **The straightforward attack on the data bottleneck fails, and it fails informatively: what this problem is short of is _labels_, not embeddings.** More unlabelled AlphaEarth of the same kind adds no information the 6,414 labelled plots did not already carry. That is a direct argument for the `PATCH_SAMPLING.md` route over any SSL route. |
| — | **Plumbing defect found and fixed while running N4** (recorded because it was silent and would have been read as the idea failing). The first N4 run came back at change-F1 **0.6095** with `art_stable_recall` collapsed to 0.443 — a 5-point drop far too large to be a property of the objective. Cause: the extra unlabelled forward pass ran in train mode and folded the pool's distribution into the encoder's **BatchNorm running statistics**, which `eval()` then used to normalise labelled test plots. The pool is one city's pixels; the labelled plots are spread across the sample. Measured drift: running \|mean\| 0.058 -> 0.189, running var 0.368 -> 0.270. Fixed with `_frozen_bn_stats()` (momentum 0 for that pass — the layer still normalises by batch statistics, so the gradient signal is unchanged and only the eval-time state is protected); BN stats afterwards are identical to labelled-only to 4 decimal places. **The lesson generalises to any auxiliary loss that adds a forward pass over out-of-distribution data.** |
| N5 | **`siam_diffonly`** — drop the raw `diff` block and let `z24 − z18` carry it. | **NEGATIVE but informative (5 seeds)** | change-F1 0.6591 vs N2's 0.6630, `focus_macro_f1` 0.3763 vs 0.3815. So the raw block still earns its place — but **it is worth −0.004 here against −0.048 on the flat trunk**, an order of magnitude less. The shared encoder has largely internalised the difference; it just has not fully replaced it. Keep `diff`, and note the settled "the diff block is not redundant" finding in `CLAUDE.md` is a statement about the *flat* trunk specifically. |
| N2b | **`siam_cos_strong`** — the cosine at weight 1.0, a co-objective rather than a regulariser. | **FLAT (5 seeds)** | change-F1 0.6616 ±0.0050 against 0.6630 ±0.0035 at weight 0.3 — no gain and a wider spread. The preregistration paid off: N2's weight was picked by measuring the loss magnitudes before running anything, and the stronger setting confirms it rather than being reached for after the fact. |
| N6 | **`siam_focus`** — a cost gate at the **coarse3** level. N12 found the existing gate reweights merged2 classes only, so it leaves every commissioned transition exactly unchanged and cannot buy them. A coarse3-level gate is a different instrument and is not built. | TODO | The one modelling item left open. |

## N13 — the Oslo map (2026-07-29)

`siam_s2off_cos` registered in `infer_s2.fit_models` as a **new** recipe beside
the deployed one, not as a replacement. Two disjoint 5-seed blocks of both models
on the same geobox (2,954,952 px): `data/inference/s2_20260729_142547` (seeds
0–4) and `s2_20260729_142746` (seeds 5–9).

**The self-IoU floor first, as the rules require.** It reproduces the published
figure, which is what makes the rest of the row trustworthy:

| | merged2 change IoU | coarse3 Nat→Art | coarse3 mean |
| --- | --- | --- | --- |
| **self**, deployed, seeds 0–4 vs 5–9 | 0.8423 | 0.8356 | 0.8241 |
| **self**, siamese, seeds 0–4 vs 5–9 | **0.8524** | 0.8207 | **0.8739** |
| cross, deployed vs siamese (seeds 0–4) | 0.5813 | 0.5726 | 0.6056 |
| cross, deployed vs siamese (seeds 5–9) | 0.5733 | 0.5154 | 0.5787 |

Cross-model agreement (~0.58) sits **far** below either model's own floor
(~0.84–0.85), replicated on two independent seed blocks, so the difference
between these maps is real and is not a seed draw. The siamese is **not** the
less stable of the two — its own reproducibility is equal or better.

**Change-pixel counts.** Deployed 16,676 / 15,841 px (0.56% / 0.54%); siamese
9,911 / 9,018 (0.34% / 0.31%). **~41% fewer change pixels, replicated.** Larger
than the Tessera (−16%) and S2 (−13.4%) suppressions and far outside the ±5%
count noise.

**But this one is explained, and the plot metrics did predict it — through the
operating point, not through F1.** At equal change-F1 the two models sit at
different points:

| | change-F1 | change precision | change recall | plots called change |
| --- | --- | --- | --- | --- |
| deployed | 0.6568 | 0.5974 | 0.7295 | ~1,081 |
| siamese | 0.6644 | **0.6507** | 0.6789 | ~923 |

against **885 true change plots**. The deployed model over-calls change by 22%;
the siamese by 4%. So the siamese trades recall for precision, and on the labelled
data its change count is the *closer* of the two to truth. **That is the opposite
of the gate/smoothing suppressions in U1 and T2**, which removed change with no
compensating precision gain — the mechanism here is calibration, not erasure.

Two things that follow and must not be skipped. The plot base rate (13.8%) is
nothing like Oslo's (~0.5%), and no plot is near the AOI, so this argument
transfers in direction only, not in magnitude. And **nothing here says which map
is correct** — Oslo still has zero labelled plots (G3/G4), so the 6,765-pixel
difference remains unadjudicable exactly as it was for Tessera.

The lever, if more change pixels are wanted: `merged_labels_from_probs(
change_threshold=...)` at t≈0.30–0.35, the tuned gate that already exists. The
siamese's higher precision is what buys room to lower the gate without flooding.

One clear improvement: the siamese's two reads are far more self-consistent —
coarse3→merged2 agreement **99.91% vs 99.54%**, cutting the S15 arg-max /
group-sum disagreement from 13,529 px to 2,676. Structure is unchanged (edge
density 0.0898 vs 0.0924, median segment 6 px both, hf-power ratio equal), so it
is not smoothing; it produces fewer change *segments* (1,831 vs 2,239 per Mpx).

**Still requires the user's visual read**, which is what settled the incumbent.

## N14 — external single-date state labels: does the legend agree? (2026-07-29)

The section verdict below says the bottleneck is labels and that N4 already
failed the unlabelled route. The remaining option is **labelled** data from
outside the project, and the siamese is the first architecture that can take it:
the shared encoder `f` is a function of **one** date, so a single-date state
label is a valid input to it. The flat `wide` trunk has no such entry point —
its first layer eats a 192-column `[2018 | 2024 | diff]` vector.

Two candidate sources, both built to `coarse3` at 2018 by `build_state_labels.py`
and both extracted on the same AlphaEarth 2018 block as the plots:

| | `lucas` | `glance_strict` |
| --- | --- | --- |
| pool after harmonisation | 62,030 | 37,645 |
| sampled (5,000/state, seed 0) | 12,360 | 13,118 |
| **20° blocks covered** | **8** | **83** |
| median distance to nearest RECOVER plot | 5,375 km | **59 km** |

**This is a gate, not a model result.** An external pool whose Nature/Cropland
boundary differs from the RECOVER interpreters' does not add noise, it adds a
*systematic offset* — on the boundary that already caps change-F1
(`cropland-nature-label-noise`). Averaging over more labels entrenches that.
`diagnose_state_pools.py` runs three reads; the third is the one that makes the
other two readable.

**Self-floor first, as the rules require.** RECOVER predicting its own `lc_2018`
state under blocked CV, multinomial logistic on the 64 AlphaEarth 2018 columns:
**accuracy 0.751, macro-F1 0.740** (artificial 0.712 / cropland 0.733 / nature
0.776). Every number below is against that, on the same plots.

| read | acc | macro-F1 | crop→nature | art→nature |
| --- | --- | --- | --- | --- |
| **self-floor** (RECOVER→RECOVER, blocked CV) | 0.751 | 0.740 | 0.240 | 0.243 |
| `glance_strict` → RECOVER | **0.742** | **0.733** | **0.153** | **0.160** |
| `lucas` → RECOVER (all 6,414 plots) | 0.598 | 0.502 | 0.716 | 0.691 |
| self-floor, LUCAS's 8 blocks only (1,160 plots) | 0.748 | 0.746 | 0.145 | 0.179 |
| `lucas` → RECOVER, same 1,160 plots | 0.707 | 0.701 | 0.285 | 0.295 |

**GLanCE passes, and by more than "does not disagree".** −0.009 accuracy and
−0.007 macro-F1 against a floor that has seen RECOVER labels, from a pool that
has seen none. Centroid displacement is 0.17–0.36 of RECOVER's own within-class
spread with nothing misplaced. On the two error modes the project is judged on it
is **better than RECOVER's own out-of-fold model** — crop→nature 0.153 vs 0.240,
art→nature 0.160 vs 0.243 — trading against nature→cropland (0.215 vs 0.148).
Its `Agriculture` class reads the same boundary the interpreters drew.

**LUCAS's global failure is mostly geography, and the in-block row is what says
so.** Read alone, `lucas` → RECOVER at macro-F1 0.502 looks like a legend that
disagrees. But LUCAS covers 8 of 83 blocks, and on the 1,160 plots inside those
blocks the same model reaches 0.701 against a 0.746 floor. Most of the collapse
was a European model asked about the tropics. **A −0.045 macro-F1 residual
survives, and it is on the predicted boundary**: crop→nature 0.285 against the
floor's 0.145 — LUCAS calls twice as much of RECOVER's Cropland "Nature", which
is what an ELC10 legend that routes permanent grassland to `Grassland`→Nature
would do. Real, and smaller than the raw number claimed.

**No contamination.** Zero pool points within 100 m of a RECOVER plot; 9 of 6,414
within 1 km for either pool. The transfer numbers are not co-location.

**Two traps found in the assets, both silent.**

* **`Segment_Type` is null on 95.7% of GLanCE units covering 2018**, and so is
  `LC_Confidence`. They are populated only for STEP, CLUSTERING and
  Training_augment. Filtering on `Segment_Type == 0` therefore *silently*
  restricts to the in-house interpreted subset. That is the pool you want, but it
  must be chosen, not inherited from a null — hence `--glance-quality`.
* **`broad` is 23× larger and worse.** `Change == false` gives 810,249 usable
  units, of which **92% are Nature and 75.8% are South America**, because
  MapBiomas is 785,005 of them. `strict` is 37,645 and geographically matches the
  plots. Size was the wrong axis.

**GLanCE level 1 has no cropland at all** (Water, Ice/snow, Developed, Barren,
Trees, Shrub, Herbaceous). Agriculture is level **2** only, and Herbaceous with
`Glance_Class_ID_level2 == 0` — 88k units — cannot be resolved to Grassland or
Agriculture and is dropped rather than guessed, since that guess *is* the
boundary under test.

**Verdict: `glance_strict` is cleared for use as encoder state supervision;
`lucas` is not cleared as a global pool.** The next step is the state head
itself, and N4's failure signature is what to instrument for — `Nature ->
Artificial` and `art_stable_recall` declining monotonically with the auxiliary
weight. Caveat on these numbers: one seed-0 stratified draw and a linear probe;
the draw is not replicated and a different sample would move them somewhat. That
is adequate for a gate and is **not** a 5-seed verdict on anything.

## N14b–d — the state head: does the external pool actually buy anything? (2026-07-29)

`siam_state_weight` in `HierarchicalSoftmaxNN` adds a linear head
`g(f(x)) -> {Nature, Cropland, Artificial}` on the shared encoder, discarded at
predict time so serving cost is unchanged. Three runs over N2, 5 seeds, identical
folds. **The control is the point of the design**: `siam_state_source=
'endogenous'` feeds the head from the *plots' own* endpoints — a `From -> To`
label is two free state labels — so it adds no data and isolates the mechanism.
Pool rows are cut to each fold's training blocks (`cv_probs_state`), so no
held-out block's feature distribution reaches the encoder.

| | `siam_cos` (N2) | **endo** (control) | **state** w=0.3 | **state** w=1.0 |
| --- | --- | --- | --- | --- |
| change-F1 | 0.6630 ±0.0035 | 0.6612 ±0.0056 | 0.6652 ±0.0068 | 0.6652 ±0.0043 |
| macro-F1 | 0.7011 | 0.7020 | 0.7038 | **0.7052** |
| `focus_macro_f1` | 0.3815 | **0.3868** | 0.3848 | 0.3834 |
| `fine_change_f1` | 0.5921 | 0.5979 | **0.6020** | 0.6010 |
| Nature → Artificial | 0.5021 | 0.5009 | **0.5078** | 0.5030 |
| Cropland → Artificial | 0.6062 | **0.6126** | 0.6045 | 0.5966 |
| Artificial → Nature | 0.4176 | 0.4336 | 0.4270 | **0.4340** |
| Artificial → Cropland | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| artStab recall | 0.6423 | **0.6451** | 0.6439 | 0.6396 |
| art → veg | 0.2337 | 0.2300 | 0.2204 | **0.2182** |
| art → *change* | **0.1240** | 0.1248 | 0.1356 | 0.1422 |

**Verdict: FLAT on the hypothesis as posed. The external labels are not what
helps.** change-F1 +0.0022 at w=0.3 is inside the ±0.005 band **and the seed
variance nearly doubled** (0.0068 against 0.0035) — N3b's exact signature for a
change that adds nothing.

**The control is what makes that readable, and it inverts the obvious reading.**
On `focus_macro_f1`, the classes this section is scored on, the endogenous head
(0.3868) beats the external pool (0.3848) and both beat N2 (0.3815). The
commissioned-class gain is attributable to the **head**, not to the 13,118 GLanCE
labels. Reported without the control it would have read as "+0.003 from external
data", and that would have been wrong. Caveat in the other direction: the head is
also new parameters, so "the head helps" is not yet distinguished from "a small
auxiliary regulariser helps".

**That the endogenous head moves at all is the one genuinely new finding.** F1
(`TWOTOWER_RESEARCH`) supervised the *same* state marginals as group-sums of the
output softmax and came back flat at every weight, concluding "the model is not
short of Artificial supervision". It is flat at the **softmax** and not flat at
the **representation** — `focus_macro_f1` +0.0053, `Artificial -> Nature` 0.4176
→ 0.4336. The shared encoder is what changed, which is consistent with the rest
of section N.

**The external pool's one distinct effect is a trade, and it runs the wrong way
for this product.** `art_stable_as_veg` falls monotonically with the external
weight — 0.2337 → 0.2204 → 0.2182, closing ~40% of the gap to the deployed
model's 0.1957 — while the control barely moves it (0.2300). But
`art_stable_as_change` rises on the same monotone, 0.1240 → 0.1356 → 0.1422. N11
established these must never be read apart: a stable built-up plot returned as a
transition is a **fabricated habitat-loss event**, while one returned as stable
Vegetation invents nothing. GLanCE buys the benign error down by pushing the
harmful one up.

**N4's failure signature is present, milder.** At w=1.0 the commissioned classes
are traded away exactly as the unlabelled pool traded them: `Cropland ->
Artificial` declines monotonically 0.6062 → 0.6045 → 0.5966, `Nature ->
Artificial` falls back to 0.5030 from 0.5078, `art_stable_recall` 0.6451 → 0.6439
→ 0.6396. More external data at strength costs the classes the map is
commissioned to find.

**A structural limit worth recording before anyone retries this. GLanCE ends in
2020, so it can only ever supervise the 2018 endpoint** — the head never sees a
2024-side state label, on a target whose classes are `from -> to`. Any serious
retry needs a pool at both endpoints, and the only candidate is LUCAS 2018→2022
(`JRC/LUCAS/THLOC/V1/2022`), which N14a did not clear as a global pool and which
reaches 8 of 83 blocks. **`Artificial -> Cropland` stays at 0.000 in every column
above**, as it has for every model in this section.

**Section N remains closed. `siam_s2off_cos` is still the best model measured,
and nothing here displaces it.** The state head was not run over the s2off base:
its AlphaEarth-only gain does not clear the band, so composing it would test a
smaller effect on a noisier baseline.

## Section verdict (2026-07-29)

**The section is closed on architecture.** Every idea from the original brief has
run at 5 seeds, plus the composition with the deployed model.

**What worked.** A shared endpoint encoder plus *any* pressure toward
year-invariance on the stable pairs. The cosine (N2) and Barlow (N3) forms are
interchangeable to within seed noise, and composing them (N3b) adds nothing and
doubles the variance. Dropped into the deployed gate-off two-tower (N8) this is
the best model measured on every aggregate: **change-F1 0.6644 ±0.0024, macro-F1
0.7067, `focus_macro_f1` 0.3847**, at unchanged serving cost because Sentinel-2
remains privileged and unread at inference.

**What did not, and is worth not retrying.** More unlabelled AlphaEarth (N4 — a
regulariser, and at strength it trades the commissioned classes away). Lumping
the stable classes (N7 — *strongly* negative; fewer classes, less signal).
Per-year adapters (N10). Built fraction against the built-up error (N9). The
merged2 cost gate as a way to buy focus classes (N12 — it cannot touch them).

**What is still the deployed model's.** Stable built-up. `art_stable_as_veg` is
0.225 for the best siamese against 0.196 deployed, and 0.217 against 0.165 gated.
Four separate attempts failed to close it. It should be read alongside
`art_stable_as_change`, where the siamese is far ahead (0.129 vs 0.167) — the
siamese fabricates fewer habitat-loss events and misclassifies more built-up as
vegetation. **Which of those two errors is worse is a decision about the product,
not about the model, and it is the user's to make.**

**No map claim is made anywhere in this section.** Everything here is plot
metrics under blocked CV. Before any of it reaches a raster the ~0.84 change-class
self-IoU floor has to be computed for a siamese map, and Oslo still has zero
labelled plots to score against.

**The bottleneck is labels.** N4 tested the alternative directly and it failed:
200k more unlabelled embeddings carry nothing the 6,414 labelled plots did not.
`Artificial -> Cropland` sits at 0.000 for every model in this table including the
deployed one. `PATCH_SAMPLING.md` is the route.
| N7 | **lumped-stable legend** — the user's MVP: collapse the three stable transitions into one `Stable -> Stable` class (5,172 plots) and leave the six change transitions distinct. Tests whether the head is spending capacity separating stable Nature from stable Cropland — the boundary `analyse_label_noise.py` calls the noisiest in the legend — at the expense of the change classes. Third `View`, reusing `full`'s folds verbatim so the legend is the only thing that differs. | **NEGATIVE, and strongly (5 seeds)** | **Lumping stable makes everything worse, and it hurts the siamese most.** <br><br>`baseline_aef`: change-F1 0.6577 -> **0.6363**, `focus_macro_f1` 0.3739 -> 0.3492, `Nature -> Artificial` 0.4775 -> 0.4506.<br>`siam_cos`: change-F1 0.6630 -> **0.6516**, `focus_macro_f1` 0.3815 -> **0.3168**, `Nature -> Artificial` 0.5021 -> **0.3718 (−0.130)**.<br><br>**The mechanism, and it is the useful part.** The stable classes are not wasted capacity — they are the *state-recognition* supervision that the change classes borrow from. Collapsing them removes the Nature/Cropland/Artificial endpoint signal on **81% of the plots**, and the merged2 level collapses with them, so the "clean, well-supported middle level" the whole hierarchy is built around stops carrying state information at all. A model that cannot tell stable Nature from stable Artificial cannot tell `Nature -> Artificial` either. The transition target is `from -> to`; lumping deletes most of the evidence about `from`. <br><br>It costs the siamese roughly twice what it costs the flat trunk, which follows: `combine="conc"` feeds `z18` and `z24` to the head *specifically* so it can read the endpoint states, and this removes the supervision that makes those endpoints mean anything. **The intuition that a smaller label space is an easier problem is wrong here in a way worth stating: fewer classes, less signal.** |
| N8 | **`siam_s2off` / `siam_s2off_cos`** — the composition: the **deployed** recipe with its AlphaEarth tower replaced by the shared endpoint encoder. Same 78-column privileged Sentinel-2 detail tower, same mask gating, same modality dropout, same gate-off serving — **no Sentinel-2 is read at inference**, so this costs nothing at serving time. N12 established that stable built-up is genuinely the deployed model's win and N2 that the focus classes are the siamese's; this is the only construction that could hold both. Run as a pair so architecture and objective are not confounded. | **WIN (5 seeds), the section's best** | **`siam_s2off_cos`: change-F1 0.6644 ±0.0024 (the tightest variance in the section), macro-F1 0.7067 and `focus_macro_f1` 0.3847 — both the highest of any model measured, deployed included.** Against the deployed model: **+0.0076 change-F1, +0.0124 macro-F1, +0.018 focus-macro**, `Cropland -> Artificial` 0.6093 vs 0.5906, **`Artificial -> Nature` 0.4423 vs 0.4066 (+0.036, the section's largest single-class gain)**, `art_stable_recall` **0.6458** — higher than the deployed model's 0.6374 — and false-change-on-built-up 0.1293 vs 0.1669. **The privileged tower also recovers part of the built-up gap** (`art_stable_as_veg` 0.2337 -> 0.2249) but does **not** close it: 0.225 against the deployed 0.196, and gated-vs-gated 0.217 against 0.165. The N8 / N8b pair separates the two effects cleanly — the architecture alone gives 0.6635 and the cosine objective adds 0.0009 change-F1 but +0.004 macro, +0.008 artStab and −0.017 art->veg, so the objective is doing the built-up work, not the tower swap. **One cost, and it is on the user's first target class:** `Nature -> Artificial` falls to 0.4871 from `siam_cos`'s 0.5021 (still well above the deployed 0.4693). Plain `siam_cos` remains the best model for that one transition; `siam_s2off_cos` is the best model for everything else. |
| N5 | **`siam_diffonly`** — drop the raw `diff` block and let `z24 − z18` carry it. Removing `diff` from the *flat* model costs −0.048 change-F1, but that model has no learned difference to fall back on. Tests whether the siamese has actually internalised it. | TODO | |
| N2b | **`siam_cos_strong`** — the cosine objective at weight 1.0 rather than 0.3, i.e. a co-objective rather than a regulariser. | TODO | |

## Rules

Inherited from `AUTORESEARCH.md`, unchanged: one hypothesis per iteration,
3 seeds minimum before a verdict and 5 before calling a win, reuse the OOF cache
for anything post-hoc, negative results get written down with their number, and
do not redo the tested-negative list at the foot of `TWOTOWER_RESEARCH.md`.

**Two entries on that list are close enough to this section to name.** Neither
covers it:

* *Contrastive alignment (B-section, negative)* was **cross-modal** InfoNCE
  between the AlphaEarth and Tessera towers — two sensors, one date. N2/N3 are
  **cross-temporal**, one sensor, two dates, and the supervision comes from the
  change label rather than from row identity.
* *Multi-year GRU trajectory (negative)* also read the dates as a sequence, but
  gave them a recurrent trunk with **no** weight sharing argument and no
  auxiliary objective on the pair; it was a capacity change, and it lost.

## Where this must not go

`s2off_centre_m3s3_bf` is the deployed model and is **not** what this section is
re-opening (see `CLAUDE.md`). Section N runs against `baseline_aef` on the plot
metrics; nothing here touches `infer_s2.py` or the published map until an idea
clears 5 seeds on `focus_macro_f1` *and* holds change-F1, and even then the
map-level comparison needs the ~0.84 self-IoU floor computed first.

# Section O — output structure and decision rule (2026-07-29)

Section N closed on **architecture**. Section O keeps its encoder fixed and
changes what sits on top: how the nine coarse3 logits are **parameterised**, and
how the arg-max over them is **taken**. Both are aimed at what section N could
not move — `Artificial -> Cropland` at 0.000 for every model in that table, and
`focus_macro_f1` generally, which N12 showed the merged2 cost gate cannot touch.

**Mamba and sequence blocks were not run, on purpose.** There are two axes they
could scan and both are degenerate. Over time, T = 2: a selective scan over two
steps collapses to a gated affine mix of `z18` and `z24`, which is what
`[z24-z18, |z24-z18|, cos]` already is with fewer parameters — and the multi-year
GRU over the annual trajectory is already tested-negative. Over the feature
dimension, AlphaEarth's 64 channels are an **unordered basis**, so a scan makes
the answer depend on an arbitrary permutation; the order-invariant version of
that idea is attention over feature tokens, i.e. FT-Transformer, also
tested-negative. Every trunk-capacity change on this board has lost, which is
what 6,414 plots at +0.026 change-F1 per doubling predicts.

All rows below are **5 seeds**, `full` read, the same folds as section N.
`base_*_fine` re-run the three incumbents through the fine-probability CV path
and reproduce the published table to four decimals (0.6630 / 0.7011 / 0.3815,
0.6644 / 0.7067 / 0.3847, 0.6568 / 0.6943 / 0.3667), which is what makes the
comparison rows trustworthy.

| | chg-F1 | macro | **focus** | Nat→Art | Crop→Art | Art→Nat | **Art→Crop** | artStab | as_veg | as_chg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| deployed `s2off_centre_m3s3_bf` | 0.6568 | 0.6943 | 0.3667 | 0.4693 | 0.5906 | 0.4066 | 0.0000 | 0.637 | **0.196** | 0.167 |
| `siam_cos` (N2) | 0.6630 | 0.7011 | 0.3815 | 0.5021 | 0.6062 | 0.4176 | 0.0000 | 0.642 | 0.234 | 0.124 |
| `siam_s2off_cos` (N8b) | **0.6644** | **0.7067** | 0.3847 | 0.4871 | 0.6093 | 0.4423 | 0.0000 | 0.646 | 0.225 | 0.129 |
| **`c3gate_siam_cos`** (O3) | 0.6630 | 0.7011 | **0.4412** | **0.5029** | 0.5885 | 0.4006 | 0.2727 | 0.642 | 0.234 | 0.124 |
| `c3gate_siam_s2off_cos` (O3b) | **0.6644** | **0.7067** | 0.4318 | 0.4929 | 0.5905 | 0.4322 | 0.2115 | 0.646 | 0.225 | 0.129 |
| `c3gate_deployed` (O3c) | 0.6568 | 0.6943 | 0.4076 | 0.4754 | 0.5836 | 0.4151 | 0.1562 | 0.637 | **0.196** | 0.167 |
| `siam_endpoint_pure` (O1) | 0.6144 | 0.6732 | 0.3940 | 0.4424 | 0.5844 | 0.2447 | 0.3044 | 0.688 | **0.143** | 0.169 |
| `siam_s2off_endpoint` (O1c) | 0.6293 | 0.6679 | 0.3907 | 0.3936 | **0.6198** | 0.1859 | **0.3633** | 0.744 | 0.185 | 0.071 |
| `siam_endpoint_state` (O1d) | 0.6143 | 0.6570 | 0.3882 | 0.4032 | 0.6162 | 0.1651 | 0.3682 | **0.759** | 0.179 | **0.062** |
| `c3gate_s2off_endpoint` (O4b) | 0.6293 | 0.6679 | 0.4235 | 0.4477 | 0.5871 | 0.3166 | 0.3426 | 0.744 | 0.185 | 0.071 |
| `siam_cos_crt` (O2) | 0.6450 | 0.6909 | 0.3756 | 0.4902 | 0.5894 | 0.4230 | 0.0000 | 0.597 | 0.175 | 0.228 |
| `siam_cos_proto` (O2c) | 0.6487 | 0.6933 | 0.3756 | 0.4775 | 0.6016 | 0.3889 | 0.0345 | 0.651 | 0.242 | 0.107 |
| `siam_s2off_patch` (O5) | 0.6598 | 0.7014 | 0.3833 | 0.4928 | 0.6105 | 0.4298 | 0.0000 | 0.634 | 0.233 | 0.132 |

## O3 (= N6) — the coarse3 cost gate. **WIN, and it is free.**

The one modelling item section N left open, and the largest `focus_macro_f1`
move in either section: **0.3815 -> 0.4412 on `siam_cos`, +0.060.** Nested
exactly like F3 — the per-class multipliers that maximise `focus_macro_f1` on
the *other* folds are applied to the held-out one — and restricted to the four
commissioned transitions, because nine multipliers on four inner folds is more
freedom than F3 found the folds can support.

**`Artificial -> Cropland` is no longer 0.000.** 0.2727 F1 at 0.244 recall on
`siam_cos`; 0.1562 on the deployed model. N0 said this class was beyond
modelling and that the fix was a labelling ask. **N0 was half wrong**: the class
was not unreachable, it was *unreached at the arg-max*. The distribution
carried enough signal all along and no model in section N ever looked at
anything but its mode.

Every aggregate column is **identical to the source model by construction** —
change-F1, macro-F1, `artStab`, `as_veg`, `veg_stable_as_art` all unchanged,
because the gate re-reads only the coarse3 arg-max and leaves the merged2
probabilities untouched. That is what makes it free: no retrain, no new
parameters, no serving cost, 4 s over the cached distribution.

**What it costs, stated plainly.** `Cropland -> Artificial` −0.018 and
`Artificial -> Nature` −0.017; the +0.060 is net of those. And the seed variance
on `focus_macro_f1` rises from 0.0043 to 0.0162 — expected for a tuned operating
point, but it means a 1-seed read of this gate is worthless.

**The sceptical reading, which should be kept.** The gate is tuned on
`focus_macro_f1` and scored on `focus_macro_f1`. Nesting makes that an honest
held-out estimate, not a circular one, but it is an operating-point move and not
new information: it re-spends the existing distribution toward the rare classes.
It lifts all three models and does not reorder them — `siam_cos` gated (0.4412)
> `siam_s2off_cos` gated (0.4318) > deployed gated (0.4076) — so it is not a
challenger dressed as an architecture win. Note the *plain* siamese overtakes the
two-tower siamese under the gate, reversing N8b.

## O1 — endpoint-tied factorised head. **A trade, and it closes the one gap section N could not.**

`logit(A -> B) = log P(from = A | z18) + log P(to = B | z24) + prior(A -> B)`,
with **one shared state head read at both dates** and a learned 9-scalar prior.
Only expressible on a siamese: `f` is a function of a single date, so `g(z18)`
and `g(z24)` are the same question asked twice. This is **not** the
tested-negative `head='bilinear'`, which reads two *separate* heads off one
fused representation with no date structure tying them together.

**It fixes stable built-up, which four attempts in section N failed to do.**
`art_stable_as_veg` **0.1432** against the deployed model's 0.1957 and the
siamese's 0.2337, with `art_stable_recall` **0.688** against 0.637 — the
challenger beats the incumbent on the incumbent's own strength for the first
time in this ledger. N9 (built fraction), N10 (per-year adapters), N12 (the
merged2 gate) and the N8 tower swap all failed at it. `siam_endpoint_state`
pushes it further (artStab 0.759, `as_chg` 0.062). It also gets the highest
`Artificial -> Cropland` measured anywhere, 0.3633.

**And it costs the class the project is named for.** `Artificial -> Nature`
collapses 0.4176 -> 0.2447 (pure) / 0.1859 (two-tower), and change-F1 falls
0.049. **The mechanism is legible and is the ledger's own label-noise finding
resurfacing.** `Art -> Nat` and `Art -> Crop` share `from = Artificial`, so the
factorisation can only separate them through the 2024 state read — which sits
exactly on the Cropland/Nature boundary that `analyse_label_noise.py` calls the
noisiest in the legend. The head converts an implicit "give it all to Nature"
prior into a genuine but noisy split: it buys the dead class with the recovery
class. That is a product decision, not a modelling one.

## O4 — the composition. **They do not compose, as preregistered.**

The gate over the endpoint head lands at `focus_macro_f1` 0.4120 (pure) and
0.4235 (two-tower), **below the gate over plain `siam_cos` (0.4412)** — and with
change-F1 at 0.6293 instead of 0.6630. The prediction registered before the run
was that O1 spends its `Artificial -> Cropland` budget and leaves the gate
nothing to buy; that is what happened. Third replication of the F7 / N3b
signature: two mechanisms correcting the same thing land between their parts.
**Where two levers are one lever, take the free one.**

## O2 / O2c — cRT and the prototype head. **Both NEGATIVE.**

* **`crt_*` (Kang et al. classifier re-training, frozen trunk, class-balanced
  head):** change-F1 0.6450 (−0.018), `focus_macro_f1` 0.3756 (−0.006),
  `art_stable_recall` 0.597 (−0.045), and `art_stable_as_change` **0.124 ->
  0.228** — it nearly doubles the fabricated habitat-loss rate. All it does is
  shift the operating point toward change (recall 0.672 -> 0.789, precision
  0.654 -> 0.545), which is the direction N13 says this product does not want.
  `Artificial -> Cropland` stays at 0.000, so balancing the classifier does not
  reach it. **The decoupling reading of the G-H negative is wrong: balanced
  sampling hurts here whether it is joint or decoupled.**
* **`proto` (cosine classifier / tau-normalisation):** flat-to-worse on
  everything — change-F1 0.6487, focus 0.3756. It moves `Artificial ->
  Cropland` off zero (0.0345) and no further. Removing the weight-norm channel
  is not what this tail needed.

## O5 — conv detail tower over the stored patches. **FLAT, and informative.**

A small conv encoder (global-average-pooled, ~3 blocks) over the 32×32 centre
crop of the stored Sentinel-2 patches, replacing the 78 hand-built columns,
still privileged and still gate-off served. **change-F1 0.6598 against the
hand-built tower's 0.6644, `focus_macro_f1` 0.3833 against 0.3847** — a tie to
slightly behind, at **10× the training cost** (107 s against 11 s).

**The dihedral augmentations do nothing at all**: 0.6598 with, 0.6592 without;
focus 0.3833 against 0.3835. Eight free orientations of every patch move
literally nothing, which says the tower is not data-starved in the way the
augmentation was meant to fix.

**But S3's verdict needed splitting, and this splits it.** S3 flattened an 8×8
pooled patch into 1,344 columns and scored **0.5976**, the worst on the board.
A spatial encoder over the *full* patch reaches 0.6598 — **+0.062**, nearly the
whole gap. So "learned texture loses" was really two claims: *flattening* pixels
into columns is catastrophic, and a proper spatial encoder is merely
**not better** than hand-built statistics. The second is the one that holds, and
it holds at ten times the cost. The stored patches remain worth having; a CNN
over them is not the way to spend them.

## Section O verdict

**Take O3. It is free, it is the largest focus-class move in either section, and
it breaks the dead class.** `c3gate_siam_cos` is the recommendation on the
commissioned transitions: `focus_macro_f1` **0.4412**, `Nature -> Artificial`
**0.5029** (the board's best), `Artificial -> Cropland` **0.2727** from zero,
with change-F1 and macro-F1 unchanged from `siam_cos` by construction. It is a
different arg-max over probabilities the model already computes.

**O1 is a genuine second option and the choice between them is the user's**, on
the same axis N11 identified: it is the only thing in either section that beats
the deployed model on stable built-up (`as_veg` 0.143 vs 0.196), and it pays for
that with `Artificial -> Nature` and 5 points of change-F1.

**Nothing here has been near a raster.** Every number is plot metrics under
blocked CV. Before any of it reaches a map the ~0.84 change-class self-IoU floor
has to be computed for the candidate, and Oslo still has zero labelled plots
(G3/G4). O3 in particular changes only the coarse3 read, so its map effect is on
`*_coarse3.tif` and not on `*_merged2.tif` — the *whether* map is untouched and
the *what kind* map is where it would show.

**Tested-negative, do not redo:** classifier re-training on a frozen trunk (cRT)
· cosine / prototype classifier (tau-norm) · conv encoder over the S2 patches,
with or without dihedral augmentation · composing the coarse3 gate with the
endpoint head · Mamba or any sequence block over the two dates or the 64
unordered AlphaEarth channels (not run — see the head of this section for why).

# Section P — auxiliary single-date paths, read at predict time (2026-07-30)

The user's question: **can an auxiliary path fed by single-year reference data
verify and improve the transitions that touch that state — a cropland model
checking `Cropland ->` in 2018 and `-> Cropland` in 2024?**

N14 already put external single-date labels into the encoder and came back flat,
but it varied only one thing. Section P varies the two it did not:

* **Where the path lands.** N14's state head was a *side-loss discarded at
  predict time*. O1's endpoint head makes the state read the **output
  parameterisation** — `logit(A -> B) = log P(from=A | z18) + log P(to=B | z24)
  + prior` — so external supervision trains the head that actually decides the
  transition. O1 was only ever run with endogenous supervision (O1d) and N14
  only ever with a flat head. **The cell where the two meet was empty**, and it
  is the one the user's framing points at.
* **Whether it is a loss at all.** P3 does not touch the encoder: it fits the
  external state model separately, applies it to **both** endpoint blocks, and
  hands the transition model nine posterior columns. That is the literal reading
  of "use a cropland model to verify", and it is the only form of the idea that
  does not need a siamese encoder — a win would transfer to the flat trunk.

All rows are **5 seeds**, `full` read, the same folds as sections N and O.

## P0 — the structural limit N14 recorded is not a limit

`diagnose_state_year_transfer.py`. N14 closed with: *"GLanCE ends in 2020, so it
can only ever supervise the 2018 endpoint — the head never sees a 2024-side
state label, on a target whose classes are `from -> to`. Any serious retry needs
a pool at both endpoints."* That assumes a 2018 label cannot reach the 2024
read. On a **shared** encoder that is an empirical question, and the answer is
that it can.

| read (linear probe, coarse3 states) | acc | macro-F1 | crop→nature | nature→crop |
| --- | --- | --- | --- | --- |
| self-floor 2018 (RECOVER→RECOVER, blocked CV) | 0.751 | 0.740 | 0.240 | 0.148 |
| self-floor 2024 (RECOVER→RECOVER, blocked CV) | 0.743 | 0.744 | 0.258 | 0.135 |
| fit 2018 → **read 2024**, RECOVER's own labels | 0.737 | 0.737 | 0.270 | 0.139 |
| `glance_strict` → RECOVER **2018** | 0.742 | 0.733 | 0.153 | 0.214 |
| **`glance_strict` → RECOVER 2024** | **0.734** | **0.735** | 0.169 | 0.213 |

**A 2018-fitted probe reads the 2024 block as well as it reads 2018** — macro-F1
0.735 against 0.733, and only −0.009 accuracy below a 2024 self-floor that was
fitted on 2024 labels. The AlphaEarth space is year-stable enough that the
`-> Cropland in 2024` half is reachable from the pool that is **already built**,
and no 2024 label source has to be acquired. Two costs are separable on the
changed plots (1,242), where a 2024 read is doing real 2024 work rather than
memorising 2018: macro-F1 0.620 (2024 self-floor) → 0.588 (year transfer, own
labels) → 0.553 (year transfer + GLanCE's legend). The year costs −0.032 and the
legend −0.035; neither is the blocker N14 expected.

## The table

| | chg-F1 | macro | **focus** | Nat→Art | Crop→Art | Art→Nat | Art→Crop | artStab | as_veg | as_chg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `siam_cos` (N2) | 0.6630 | 0.7011 | 0.3815 | 0.5021 | 0.6062 | 0.4176 | 0.0000 | 0.642 | 0.234 | 0.124 |
| `siam_s2off_cos` (N8b) | **0.6644** | **0.7067** | 0.3847 | 0.4871 | 0.6093 | 0.4423 | 0.0000 | 0.646 | 0.225 | 0.129 |
| `c3gate_siam_cos` (O3) | 0.6630 | 0.7011 | **0.4412** | 0.5029 | 0.5885 | 0.4006 | **0.2727** | 0.642 | 0.234 | 0.124 |
| `siam_endpoint` (O1b) | 0.6145 | 0.6553 | 0.3841 | 0.4021 | 0.6153 | 0.1507 | 0.3681 | 0.754 | 0.183 | 0.063 |
| `siam_endpoint_state` (O1d, endogenous) | 0.6143 | 0.6570 | 0.3882 | 0.4032 | 0.6162 | 0.1651 | 0.3682 | **0.759** | **0.179** | 0.062 |
| **`siam_endpoint_state_ext`** (P1) | 0.6146 | 0.6568 | 0.3849 | 0.4118 | 0.6156 | 0.1574 | 0.3550 | 0.743 | 0.193 | 0.064 |
| `siam_endpoint_state_both` (P1b) | 0.6141 | 0.6575 | 0.3875 | 0.4119 | 0.6196 | 0.1630 | 0.3556 | 0.751 | 0.186 | 0.063 |
| `siam_endpoint_state_ext_strong` (P1c, w=1.0) | 0.6164 | 0.6619 | 0.3830 | 0.4061 | 0.6054 | 0.1553 | 0.3652 | 0.730 | 0.205 | 0.066 |
| `siam_s2off_endpoint` (O1c) | 0.6293 | 0.6679 | 0.3907 | 0.3936 | **0.6198** | 0.1859 | 0.3633 | 0.744 | 0.185 | 0.071 |
| `siam_s2off_endpoint_state` (P2) | 0.6280 | 0.6687 | 0.3856 | 0.3969 | 0.6160 | 0.1694 | 0.3600 | 0.740 | 0.191 | 0.070 |
| **`siam_cos_prior`** (P3) | 0.6587 | 0.6992 | 0.3747 | 0.4830 | 0.6047 | 0.4112 | 0.0000 | 0.651 | 0.234 | 0.115 |
| `siam_cos_prior_endo` (P3b, **control**) | 0.6565 | 0.6965 | 0.3765 | 0.4890 | 0.6076 | 0.4092 | 0.0000 | 0.650 | 0.235 | 0.115 |
| `siam_s2off_prior` (P3c) | 0.6599 | 0.7042 | 0.3816 | 0.4846 | 0.6106 | 0.4312 | 0.0000 | 0.654 | 0.222 | 0.124 |
| `c3gate_endpoint_state_ext` (P4) | 0.6146 | 0.6568 | 0.4269 | 0.4446 | 0.5990 | 0.3041 | 0.3598 | 0.743 | 0.193 | 0.064 |
| `c3gate_siam_cos_prior` (P4b) | 0.6587 | 0.6992 | 0.4262 | **0.5125** | 0.6069 | 0.4011 | 0.1841 | 0.651 | 0.234 | 0.115 |

## P1 / P2 — external supervision on the head that decides. **FLAT.**

Against O1d, the identical configuration with endogenous supervision, P1 moves
nothing that clears the ±0.005 band: change-F1 0.6146 vs 0.6143, macro 0.6568 vs
0.6570, `focus_macro_f1` 0.3849 vs 0.3882. **N14's verdict survives being moved
from a discarded head to the output parameterisation.** P2 repeats it on the
section's best base (0.6280 vs O1c's 0.6293).

**One thing is monotone in the external weight, and it runs the wrong way.**
`art_stable_recall` 0.759 (endogenous) → 0.751 (both) → 0.743 (external) →
0.730 (external at w=1.0), with `art_stable_as_veg` rising 0.179 → 0.186 →
0.193 → 0.205 on the same monotone. **Third replication of the N4 / N14d
signature**: external data at strength trades away the class the construction
exists to buy — here stable built-up, which is the endpoint head's one win over
every other model in the ledger. `Artificial -> Cropland` goes with it, 0.3682 →
0.3550.

The only column favouring the external pool is `Nature -> Artificial`, 0.4118 /
0.4119 against 0.4032 endogenous — consistent across both external variants, and
**still inside the ±0.015 seed spread on that class**. Not a claim.

## P3 — the external state model as input rather than as loss. **NEGATIVE.**

Nine posterior columns at both endpoints cost `siam_cos` −0.004 change-F1,
−0.007 `focus_macro_f1` and **−0.019 `Nature -> Artificial`** (0.4830 vs 0.5021),
the user's first target class and the one column where the loss clears the band.
P3c repeats it on the best base (0.6599 vs 0.6644).

**The control is again what makes it readable, and again it inverts the obvious
reading.** `siam_cos_prior_endo` — the identical nine columns from a probe
fitted on the *training plots' own* endpoints — lands at 0.6565 / 0.3765,
indistinguishable from the external arm. The external arm's only content over
its control is GLanCE's decision boundary, and it is worth +0.002 change-F1 and
−0.002 focus. **Both arms sit below the model with no prior columns at all**, so
the nine columns cost more in dilution than any boundary they carry is worth.

What the columns do buy, in *both* arms equally and therefore not from GLanCE:
`art_stable_as_change` 0.124 → 0.115 and `art_stable_recall` 0.642 → 0.651 —
fewer fabricated habitat-loss events on built-up, at no change to `as_veg`.

## P4 / P4b — the free gate over both. Neither reaches O3.

`focus_macro_f1` 0.4269 over P1 and 0.4262 over P3, against **`c3gate_siam_cos`
at 0.4412**. Section O's recommendation is unmoved. Worth recording that P4b
returns `Nature -> Artificial` **0.5125**, nominally the highest figure anywhere
in this ledger for that class (0.5029 gated incumbent) — but ±0.014 across
seeds, and it is bought by collapsing `Artificial -> Cropland` from 0.2727 to
0.1841 ±0.100, so the focus macro is worse. Not a candidate; recorded so it is
not rediscovered as one.

## P5 — the question underneath: how good would a source have to be?

P1–P3 say the pool that exists does not help. They do **not** say whether a
better one would, and that is the decision actually on the table — whether to go
and acquire single-date reference data. `simulated_state_prior` replaces P3's
nine columns with a synthetic reader that is correct a set fraction of the time
at each date independently. **A ceiling, never a candidate**: it is built from
the held-out plot's own label, and 1.00 is degenerate by construction because two
exact endpoint states *are* the coarse3 label.

| reader accuracy | chg-F1 | macro | focus | Nat→Art | Crop→Art | Art→Nat | **Art→Crop** | artStab | as_veg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| — (`siam_cos`) | 0.6630 | 0.7011 | 0.3815 | 0.5021 | 0.6062 | 0.4176 | 0.0000 | 0.642 | 0.234 |
| 0.74 | 0.7134 | 0.7614 | 0.4367 | 0.6079 | 0.6813 | 0.4577 | 0.0000 | 0.808 | 0.111 |
| 0.75 | 0.7166 | 0.7644 | 0.4428 | 0.6197 | 0.6852 | 0.4664 | 0.0000 | 0.815 | 0.105 |
| 0.85 | 0.7725 | 0.8123 | 0.4976 | 0.7133 | 0.7506 | 0.5263 | 0.0000 | 0.879 | 0.055 |
| 0.95 | 0.8748 | 0.8982 | 0.6008 | 0.8437 | 0.8644 | 0.6951 | 0.0000 | 0.951 | 0.013 |
| 1.00 (degenerate) | 0.9839 | 0.9783 | 0.7020 | 0.9885 | 0.9835 | 0.8360 | 0.0000 | 0.997 | 0.000 |

**The 0.74 row is the finding of the section, and it is a matched pair.** P0
measures *both* real state readers — GLanCE's probe and one fitted on the plots'
own labels — at ~0.74 state accuracy on these embeddings. At that same accuracy:

| a 74%-accurate single-date state reader whose errors are… | chg-F1 | focus | artStab | as_veg |
| --- | --- | --- | --- | --- |
| **correlated** with AlphaEarth (P3, the real probe) | 0.6587 | 0.3747 | 0.651 | 0.234 |
| **independent** of AlphaEarth (P5, the oracle) | **0.7134** | **0.4367** | **0.808** | **0.111** |

**+0.055 change-F1, +0.062 focus and half the stable-built-up error, from
nothing but error independence at identical accuracy.** A model trained on
satellite imagery is wrong exactly where the embedding is ambiguous — precisely
the plots the transition model is also unsure about — so its posterior is close
to a deterministic function of inputs the encoder already reads. That is why
every arm of P1–P3 is flat, and it is a stronger statement than "GLanCE did not
help": **accuracy is not the axis. Error independence is.**

Two consequences worth stating plainly.

* **Do not buy another optical land-cover product.** Dynamic World, ESRI's
  annual layers, WorldCereal, GLanCE's own map products — all are models over the
  same spectral evidence AlphaEarth compresses, so all land on the correlated
  row whatever their headline accuracy. This is a concrete prediction, and it is
  the cheap thing to check before any extraction is run.
* **In-situ data is a different proposition.** LUCAS is field-surveyed and its
  errors are not AlphaEarth's, which is a genuine argument for revisiting it —
  but N14a did not clear it as a *global* pool (8 of 83 blocks, macro-F1 0.502
  globally, 0.701 in-block against a 0.746 floor), so the argument is for a
  regional model, not for this one.

**And the sweep confirms N0 from the other end.** `Artificial -> Cropland` stays
at **0.000 even with both endpoint states known exactly**. The 46-plot class is
not unreachable for want of evidence — a perfect state read does not recover it
at the arg-max, because a 9-way softmax still never selects it. Only O3's
coarse3 gate breaks it (0.2727). N0 said this was a labelling ask; O3 said it
was an arg-max problem; P5 settles it as the arg-max problem, decisively.

## Section P verdict

**Negative. Nothing here displaces `siam_s2off_cos` on the aggregates or
`c3gate_siam_cos` on the commissioned transitions**, and the recommendation from
section O is unchanged.

**What was learned that is worth keeping.**

1. **The 2018-only limit was never real.** A 2018 pool reads the 2024 endpoint at
   the same quality (P0). Any future single-date source only has to exist at
   *one* date.
2. **Where the path lands does not matter** — discarded side-loss (N14), output
   parameterisation (P1), or input feature (P3) all give the same flat result
   from the same labels. **AMENDED 2026-07-31 by section P7: *when* it lands
   does.** All three of those trained the state objective jointly with the
   transition loss; run as a separate pretraining phase the same labels take
   `Artificial -> Cropland` from 0.000 to 0.19, beating both a no-new-data
   control and a shuffled-label control. Do not read this item as "the labels
   are inert" — it was written that way and that part was wrong.
3. **Why does matter, and it is measurable**: a state reader helps in proportion
   to how *independent* its errors are of AlphaEarth, not to how accurate it is.
   The matched pair at 0.74 is +0.055 change-F1 apart.
4. **The endogenous control earned its place a second time.** In both P1 and P3
   it matched or beat the external pool, and in both the naive reading without it
   would have been "+0.002 from GLanCE".

**Tested-negative, do not redo:** external state supervision on the endpoint
head, at any weight or source combination (P1/P1b/P1c/P2) · external state
posteriors as input columns, on either base (P3/P3c) · the coarse3 gate over
either (P4/P4b) · any further *optical map product* as a single-date pool (P5,
by argument — check the correlated/independent axis before extracting anything).

**This list covers joint training only.** Section P7 runs the same pool as a
pretraining phase and is positive; the entries above are not a verdict on it.

# Section Q — the burned-area Swin network's modules, transcribed (2026-07-30)

The user's question: which of the five design choices Zhang et al. (RSE 2025)
report behind their Swin-Transformer burned-area change detector apply to the
siamese line, and what do they do here.

Their network is a two-stream Swin encoder over pre/post-fire image patches.
This one is a plot-level encoder over 64-D AlphaEarth embeddings with a
privileged Sentinel-2 scalar tower. So the translation is not optional — each
idea has to be asked in the form it can take without an image grid, and two of
the five have no such form.

| their design choice | here | why |
| --- | --- | --- |
| **(1) multi-band input** | **testable, Q1** | the detail tower reads 4 bands + 3 derived indices; their set is half SWIR, which this project has never extracted |
| **(2) double-stream feature extraction** | **already the architecture, and its unshared variant is tested** | N1 is the shared two-stream encoder; their streams are *unshared*, which is N10 (`siam_pseudo`), FLAT. Swin itself needs a spatial grid: O5 ran a conv encoder over the stored patches (flat, 10× cost) and FT-Transformer over the 64 unordered AlphaEarth channels is tested-negative |
| **(3) CRFE** | **partly testable, Q2/Q3** | the add/subtract fusion and the channel attention transfer; the **spatial** attention has no form — there are no spatial dims at a plot, and the one place it could live is the patch tower O5 already found flat |
| **(4) pyramid up-sampling decoder** | **the depth half only, Q4** | there is no resolution to recover; what transfers is folding shallow stages back into the deep embedding |
| **(5) deep supervision + hybrid loss** | **testable, Q5/Q6** | both. Note the model already supervises gate/merged2/coarse3 — but that is three reads of one output at full depth, not the same objective at shallower depths |

All rows **5 seeds** unless marked, `full` read, the same folds as sections N/O/P.
`siam_cos` re-run first as a plumbing check: the trunk was refactored to expose
its hidden stages, and it reproduces 0.6630 / 0.7011 / 0.3815 / 0.2337 to four
decimals, so nothing below is a reshuffled initialisation.

| | chgF1 | macro | focus | Nat→Art | Art→Nat | artStab | as_veg | as_chg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `siam_cos` (N2) | 0.6630 | 0.7011 | 0.3815 | 0.5021 | 0.4176 | 0.642 | 0.234 | 0.124 |
| `siam_s2off_cos` (N8b) | 0.6644 | 0.7067 | 0.3847 | 0.4871 | 0.4423 | 0.646 | 0.225 | 0.129 |
| `siam_s2off_b3` (Q1) | 0.6607 | 0.7020 | 0.3810 | 0.4899 | 0.4280 | 0.640 | 0.238 | 0.122 |
| `siam_s2off_b4` (Q1) | 0.6602 | 0.7024 | 0.3844 | 0.4895 | 0.4343 | 0.637 | 0.240 | 0.122 |
| `siam_s2off_b7` (Q1) | 0.6644 | 0.7067 | 0.3847 | 0.4871 | 0.4423 | 0.646 | 0.225 | 0.129 |
| `siam_cos_crfe_sum` (Q2) | 0.6536 | 0.6967 | 0.3756 | 0.4900 | 0.3998 | 0.664 | 0.221 | 0.115 |
| `siam_cos_crfe_attn` (Q3) | 0.6620 | 0.7036 | 0.3787 | 0.4931 | 0.4145 | 0.661 | 0.216 | 0.123 |
| `siam_cos_crfe_full` (Q3b) | 0.6583 | 0.6996 | 0.3788 | 0.4968 | 0.4027 | 0.674 | 0.207 | 0.119 |
| `siam_cos_pyramid` (Q4) | 0.6589 | 0.6979 | 0.3779 | 0.4890 | 0.4131 | 0.651 | 0.223 | 0.126 |
| `siam_cos_deepsup` (Q5) | 0.6642 | 0.7013 | 0.3833 | 0.5018 | 0.4217 | 0.636 | 0.239 | 0.125 |
| `siam_cos_deepsup_strong` (Q5b) | 0.6631 | 0.6999 | 0.3807 | 0.4923 | 0.4203 | 0.627 | 0.248 | 0.125 |
| `siam_cos_dice` (Q6) | 0.6637 | 0.7035 | 0.3823 | 0.4979 | 0.4255 | 0.644 | 0.231 | 0.125 |
| `siam_cos_dice_fine` (Q6b) | 0.6617 | 0.7027 | 0.3842 | 0.5017 | 0.4240 | 0.647 | 0.230 | 0.123 |

## Q1 — more spectral bands. **NEGATIVE, and it answers a spending question.**

`siam_s2off_b7` reproduces `siam_s2off_cos` to four decimals, which is what makes
the ladder trustworthy: the channel filter does select exactly the deployed 78
columns at the top rung. Built fraction is kept in every rung, since it is a
spatial statistic no per-channel column can reconstruct.

**The step that adds a genuine band does nothing, and the step that adds no new
band is the one that moves.** b3 → b4 adds near-infrared: change-F1 0.6607 →
0.6602, `focus_macro_f1` 0.3810 → 0.3844, `art_stable_as_veg` 0.238 → 0.240 —
every column inside noise. b4 → b7 adds only NDVI, NDWI and brightness, all
arithmetic on bands already present: +0.0042 change-F1 and −0.0155 `as_veg`, the
larger of the two moves. **Nonlinear recombination of four bands beats a fifth
band.**

**So do not extract SWIR, and there is a structural reason as well as this
measurement.** AlphaEarth is itself built over Sentinel-2's full spectrum, so
SWIR reflectance is not information the model is missing — it is information it
already has, compressed. What the detail tower adds over AlphaEarth is
*sub-embedding-scale spatial statistics* (the 3 px standard deviation, built
fraction), not spectral coverage, and the ladder is what says so. This is P5's
finding arriving from the other direction: the axis is not how much evidence a
source carries, it is whether its errors are independent of AlphaEarth's.

**The ceiling makes it moot in any case.** The entire 78-column tower is worth
+0.0014 change-F1 over AlphaEarth-only `siam_cos` (0.6630 → 0.6644). A better
band set can only ever compete for a fraction of that. Honest caveat: keeping
built fraction gives the b3 rung an NIR-derived summary through the back door, so
b3 → b4 is a lower bound on a marginal band — but b3 → b7 is +0.0037 in total,
and SWIR would have to beat all three added derived channels combined.

## Q5 / Q6 — deep supervision and the hybrid loss. **Both FLAT.**

* **Deep supervision** (an auxiliary coarse3 head on each hidden encoder stage,
  same three nested levels, discarded at predict time): change-F1 0.6642 against
  0.6630, `focus_macro_f1` 0.3833 against 0.3815 — both inside the ±0.005 band,
  **and the seed variance rises from 0.0035 to 0.0047**, the N3b signature of a
  change that adds nothing. At weight 1.0 (Q5b) it is worse on everything
  including `as_veg` (0.248), so the strength question is closed rather than
  left open. The likely reason is that this model was *already* deeply
  supervised in the sense that matters here: the loss reaches the representation
  through three nested levels, and the gradient-vanishing problem the paper's
  auxiliary heads solve does not exist in a three-layer encoder.
* **Hybrid loss.** Soft-Dice on the change class (Q6) is a differentiable
  change-F1 and lands at 0.6637 / 0.3823, flat. The `fine` variant (Q6b) is the
  more interesting construction — an unweighted mean of the per-class Dice over
  all nine coarse3 classes, i.e. the relaxation of `focus_macro_f1` itself, and
  the only long-tail lever in the section that acts on the *set* rather than
  per-sample (focal) or per-parameter (cRT, tau-norm) — and it too is flat:
  0.3842 against 0.3815, with `Artificial -> Cropland` still at 0.000. **A
  set-level objective does not reach the 46-plot class either.** That is the
  fourth independent confirmation of N0/P5: only O3's coarse3 gate breaks it.

## Q4 — the pyramid decoder as depth fusion. **NEGATIVE.**

change-F1 0.6589 (−0.004), `focus_macro_f1` 0.3779 (−0.004), and the rare-class
prediction fails: `Artificial -> Cropland` stays at 0.000 and `Nature ->
Artificial` does not move. The analogy does not survive the translation, and it
is legible why — their pyramid recovers *spatial* detail destroyed by
down-sampling, and a 192 → 512 → 256 → 128 encoder destroys no spatial detail
because there was none to begin with.

## Q2 / Q3 / Q7 — CRFE. **The one thing in this section that is not flat, and it
moves the frontier sections N–P could not.**

Every aggregate is flat-to-slightly-negative. What moves is **stable built-up** —
which `SIAMESE_RESEARCH.md` records as still the deployed model's after four
failed attempts (N9 built fraction, N10 per-year adapters, N12 the merged2 gate,
the N8 tower swap).

The **2×2, all on the N8b base, one set of folds, 15 seeds** (`siam_s2off_*`):

| appended to the block | no gate | + SE gate |
| --- | --- | --- |
| **nothing** | 0.6604 · artStab 0.644 · as_veg 0.230 | **0.6624 · 0.652 · 0.217** (Q7 `attn`) |
| **`z18 + z24`** (CRFE) | 0.6565 · 0.662 · 0.224 (Q8c) | **0.6594 · 0.669 · 0.208** (Q7b `full`) |
| **fixed random mix** (control) | 0.6603 · 0.646 · 0.227 (Q8) | 0.6596 · 0.663 · 0.210 (Q8b) |

**The gate is the free component.** It improves both built-up numbers in all
three rows at no change-F1 cost. **The sum operator is not width** — its control,
a fixed random linear view of the same pair at the same width, gives 0.646 where
the sum gives 0.662, so appending *that particular* linear map does something a
random one does not. But **once the gate is present the sum's marginal effect is
inside the seed spread** (0.669 vs the control's 0.663, sd 0.008–0.011), so the
mechanism is channel attention over a redundant block, and the sum's job is
mostly to make the block more redundant.

**Read `as_veg` and `as_chg` together, as N11 requires, because it reverses the
ranking of the two arms.** `crfe_attn` looks like the cheap win — change-F1
nominally *up* — but its `art_stable_as_change` **rises** (0.1263 → 0.1311): it
converts benign misclassification into fabricated habitat-loss events.
`crfe_full` moves **both** the right way.

### The matched comparison, 15 seeds, same folds

| | chgF1 | macro | focus | Nat→Art | Art→Nat | artStab | as_veg | as_chg | veg→art |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **deployed** `s2off_centre_m3s3_bf` | 0.6557 | 0.6938 | 0.3674 | 0.4729 | 0.4063 | 0.642 | **0.194** | 0.164 | 0.0354 |
| `siam_s2off_cos` (N8b) | 0.6604 | 0.7035 | 0.3820 | 0.4800 | **0.4378** | 0.644 | 0.230 | 0.126 | 0.0299 |
| **`siam_s2off_crfe_full`** (Q7b) | 0.6594 | 0.7025 | 0.3801 | 0.4843 | 0.4234 | **0.669** | 0.208 | **0.123** | 0.0346 |

**Against N8b, the cost of the whole module is inside noise** — change-F1
−0.0010, macro −0.0010, `focus_macro_f1` −0.0019 — and it buys `art_stable_recall`
+0.025 and `as_veg` −0.021 with `as_chg` also down 0.004. The one real per-class
cost is `Artificial -> Nature`, 0.4378 → 0.4234 (−0.014), the recovery class.

**Against the deployed model it closes ~60% of the gap that section N declared
closed to modelling**, and the standing counter-check does not blow up:
`veg_stable_as_art` 0.0346 against the deployed 0.0354, i.e. the siamese now
calls Artificial at the incumbent's own rate rather than under it. At that
operating point it is **+0.027 on `art_stable_recall`** and **−0.041 on
fabricated change on built-up**, and it keeps every section-N aggregate
(+0.004 change-F1, +0.009 macro, +0.013 focus). Residual gap on `as_veg`: 0.208
against 0.194, down from 0.230.

**What this does not establish.** Nothing here has been near a raster: the
~0.84 change-class self-IoU floor has not been computed for a CRFE map, and Oslo
still has zero labelled plots (G3/G4). The Q7b trade is on the same axis as N11
and O1 and the choice is the user's — but unlike O1 (which bought built-up with
5 points of change-F1 and a collapsed recovery class) this one costs 0.001
change-F1 and 0.014 on one transition.

## Section Q verdict

**Four of the five ideas are flat or negative; the fifth is the first thing in
four sections to move stable built-up at an affordable price.**

* **Do not extract SWIR** (Q1) — the marginal band buys nothing, the whole detail
  tower is worth +0.0014 change-F1, and AlphaEarth already carries the spectrum.
* **`siam_s2off_crfe_full` is a genuine candidate on the built-up frontier** and
  is the row to put in front of the user; `c3gate_siam_cos` (O3) remains the
  recommendation on the commissioned transitions, and the two are compatible in
  principle since O3 only re-reads the coarse3 arg-max — untested together.
* The negative results are informative about *why*: this encoder is too shallow
  to need deep supervision, has no spatial resolution for a pyramid to recover,
  and has a 46-plot class that no objective — per-sample or set-level — reaches
  at the arg-max.

**Tested-negative, do not redo:** more Sentinel-2 reflectance bands, including
SWIR, on the privileged detail tower (Q1, and by the AlphaEarth-already-sees-it
argument) · deep supervision on the encoder stages at either weight (Q5/Q5b) ·
soft-Dice added to focal, on the change class or as a coarse3 macro (Q6/Q6b) ·
pyramid depth fusion into the endpoint embedding (Q4) · Swin, or any windowed
spatial attention, over plot-level features (no form — see the table at the head
of this section).

## Q9 — the Oslo map (2026-07-30)

`siam_s2off_crfe_full` registered in `infer_s2.fit_models` as a **new** recipe
beside the deployed one and beside `siam_s2off_cos`, not as a replacement.
Two disjoint 5-seed blocks on the same geobox (2,954,952 px):
`data/inference/s2_20260730_135016` (seeds 0–4) and `s2_20260730_135106`
(seeds 5–9). Serving cost is unchanged — 6.8 s of predict over 5 forward passes,
no composite fetch, detail tower skipped, and the train/serve self-check passes
at 4.24e-05 worst relative disagreement.

**The self-IoU floor first, as the rules require, and this map is the most
reproducible of the three:**

| | merged2 change IoU | coarse3 Nat→Art | overall agreement |
| --- | --- | --- | --- |
| **self**, deployed, seeds 0–4 vs 5–9 (N13) | 0.8423 | 0.8356 | — |
| **self**, `siam_s2off_cos` (N13) | 0.8524 | 0.8207 | — |
| **self**, `crfe_full`, seeds 0–4 vs 5–9 | **0.8595** | **0.8496** | 99.12% |
| cross, `crfe_full` vs `siam_s2off_cos` (seeds 0–4) | 0.8035 | — | 99.21% |
| cross, `crfe_full` vs deployed (seeds 0–4) | 0.5222 | — | 97.91% |

**And that is what makes the cross rows readable.** Against `siam_s2off_cos` the
change-class IoU is **0.8035 against two floors of 0.85–0.86** — outside seed
noise, but only just: these are nearly the same map. Against the deployed model
it is **0.5222**, less than two thirds of either floor, replicating N13 — the
siamese line and the deployed line are genuinely different maps and the CRFE
module does not change that.

**The map confirms the plot-level built-up finding, in direction.** This was the
prediction to check, since stable built-up is the only thing Q7b moves:

| (seeds 0–4) | change px | frac | **stable Artificial** | Nat→Art | Art→Nat | Crop→Art |
| --- | --- | --- | --- | --- | --- | --- |
| deployed | 16,676 | 0.564% | 1,095,592 | 10,609 | 5,024 | 960 |
| `siam_s2off_cos` | 9,911 | 0.335% | 1,040,339 | 6,512 | 3,487 | 108 |
| **`crfe_full`** | **8,181** | **0.277%** | **1,057,472** | 5,281 | 2,954 | 67 |
| `crfe_full`, seeds 5–9 | 8,957 | 0.303% | 1,074,484 | 6,029 | 3,131 | 44 |

`crfe_full` holds **+1.6% to +3.3% more pixels as stable Artificial** than
`siam_s2off_cos`, moving toward the deployed model's count — the raster signature
of `art_stable_recall` +0.025 and `as_veg` −0.021 — while calling **the least
`Nature -> Artificial` of any map on the board** (5,281 against the siamese's
6,512 and the deployed 10,609), which is the signature of `as_chg` 0.123. The two
readings move in opposite directions, which is exactly what a genuine boundary
improvement looks like rather than a threshold shift.

**The cost, stated plainly: it cuts change pixels a further 17% below the
siamese, to 0.28–0.30% of the AOI against the deployed model's 0.56%.** Read this
the way N13 read the siamese's own suppression. At the plot base rate the model
is *not* suppressing — 15 seeds put it at change precision 0.654 / recall 0.665,
which calls ~900 plots change against 885 true, marginally tighter than the
siamese's ~923 and far tighter than the deployed model's ~1,081. But the plot base
rate is 13.8% and Oslo's is ~0.3%, so that argument transfers in direction only
and **nothing here says which map is right**: Oslo still has zero labelled plots
(G3/G4), so the 1,730-pixel difference from the siamese is as unadjudicable as
every map difference before it.

Two other properties carry over unchanged: coarse3→merged2 self-consistency stays
at the siamese's level (99.87% / 99.84% against the deployed 99.54%), and
`Artificial -> Cropland` and `Cropland -> Nature` remain at **0 px**, as they are
on every map in this project. The recipe carries no `c3_costs`, so O3's coarse3
gate — the thing that breaks the first of those classes on plots — has not been
composed with it here.

**Still requires the user's visual read**, which is what settled the incumbent
(CLAUDE.md) and what N13 also ended on.

## P6 — the cropland transitions specifically (2026-07-30)

Section P was aimed at the cropland classes, so they are reported here in full
rather than through `focus_macro_f1`. Five coarse3 transitions touch Cropland;
only two are in the commissioned four, so three of these rows are not tracked
anywhere else in the ledger. 5 seeds, `full`, off the cached OOF labels.

| | Art→Crop (46) | Crop→Art (333) | Crop→Crop (1,661) | **Crop→Nat (114)** | **Nat→Crop (243)** |
| --- | --- | --- | --- | --- | --- |
| `siam_cos` (N2) | 0.0000 | 0.6062 | 0.7283 | **0.0000** | 0.1902 |
| `siam_s2off_cos` (N8b) | 0.0000 | 0.6093 | 0.7249 | **0.0000** | 0.2389 |
| `c3gate_siam_cos` (O3) | 0.2727 | 0.5885 | 0.7272 | **0.0000** | 0.1903 |
| `siam_endpoint` (O1b) | 0.3681 | 0.6153 | 0.7235 | **0.0000** | 0.0807 |
| `siam_endpoint_state` (O1d, endo) | 0.3682 | 0.6162 | 0.7265 | **0.0000** | 0.1128 |
| `siam_endpoint_state_ext` (P1) | 0.3550 | 0.6156 | 0.7324 | **0.0000** | 0.1276 |
| `siam_endpoint_state_both` (P1b) | 0.3556 | 0.6196 | 0.7321 | **0.0000** | 0.1233 |
| `siam_s2off_endpoint` (O1c) | 0.3633 | 0.6198 | 0.7258 | **0.0000** | 0.0934 |
| `siam_s2off_endpoint_state` (P2) | 0.3600 | 0.6160 | 0.7323 | **0.0000** | 0.1311 |
| `siam_cos_prior` (P3) | 0.0000 | 0.6047 | 0.7295 | **0.0000** | 0.1717 |
| `siam_cos_prior_endo` (P3b, control) | 0.0000 | 0.6076 | 0.7290 | **0.0000** | 0.1680 |
| `siam_s2off_prior` (P3c) | 0.0000 | 0.6106 | 0.7315 | **0.0000** | 0.2384 |
| `prior_oracle_74` (ceiling) | 0.0000 | 0.6813 | 0.8300 | **0.0000** | 0.2930 |
| `prior_oracle_100` (degenerate) | 0.0000 | 0.9835 | 0.9773 | 0.4766 ±0.246 | 0.9624 |

**No cropland transition improved from the external labels.** `Cropland ->
Artificial` is flat everywhere — 0.6156 external against 0.6162 endogenous and
0.6153 with no state path at all, inside a ±0.009 seed spread. `Artificial ->
Cropland` moves only with the endpoint head (O1's architecture), and adding
GLanCE **lowers** it, 0.3682 -> 0.3550.

**The one column where external labels do move a cropland class is `Nature ->
Cropland` under the endpoint head**: 0.0807 (no state path) -> 0.1128
(endogenous) -> 0.1276 (external), and 0.0934 -> 0.1311 on the two-tower. The
+0.015 over the control is ~1.7 seed-sd and is the only external-vs-control gap
in section P that is not visibly zero. **It is still not an improvement.** The
endpoint head *destroys* this class — 0.190 -> 0.081 — and the external labels
repair about a third of the damage. Repairing a wound the architecture inflicted
is not the same as beating the model that never had it.

### Two findings that fall out of this table and are not about GLanCE

**`Cropland -> Nature` is a second dead class, and O3 does not rescue it.**
0.0000 for every model in section N, O and P, gated and ungated. Section O's
verdict says the coarse3 gate "breaks the dead class" — singular, and correct as
written, but this table shows it is not the only one. Even a *perfect* endpoint
oracle only reaches 0.4766 ±0.246 on it, the largest seed spread anywhere in
this ledger, so unlike `Artificial -> Cropland` this class is not merely
unreached at the arg-max. 114 plots on the Cropland/Nature boundary that
`analyse_label_noise.py` already calls the noisiest in the legend — the two
facts are very likely the same fact.

**`Nature -> Cropland` nearly doubles for free, and the only reason it has not
is that nobody pointed the gate at it.** `coarse3_cost_gate` gives free
multipliers to `FOCUS_TRANSITIONS` only, and `Nature -> Cropland` is not one of
the commissioned four. Widening the target set to six and tuning on the matching
six-class macro (nested exactly as O3, 5 seeds):

| | Nat→Crop | Crop→Nat | `focus_macro_f1` (original 4) |
| --- | --- | --- | --- |
| `c3gate_siam_cos` (4-class gate, O3) | 0.1903 | 0.0000 | **0.4412** |
| 6-class gate over `siam_cos` | **0.3623** ±0.011 | 0.0130 | **0.4427** ±0.016 |
| 6-class gate over `siam_s2off_cos` | **0.3658** ±0.005 | 0.0059 | 0.4329 |

**+0.172 on `Nature -> Cropland` at no cost to the commissioned four** (0.4427
against 0.4412, a tie). Same instrument, same nesting, same zero retrain —
purely a wider target set. And it confirms the reading above: the same widening
leaves `Cropland -> Nature` at 0.013, so that class is dead for a different
reason than "unreached".

**This is exploratory and is deliberately not registered as an idea yet**, for
one reason: it changes what the gate is *tuned on*, and whether `Nature ->
Cropland` belongs in the commissioned set is a question about the product, not
about the model. In a project named for habitat loss, nature converted to
agriculture is arguably the most obvious omission from a focus set built around
built-up — but the four were chosen, and widening them is the user's call.

# Section R — conformal prediction (2026-07-30)

Every operating point in this ledger is chosen by **search**: the change gate
(E1) grid-searches a threshold against change-F1, the cost gates (F3, N12, O3)
grid-search per-class multipliers against macro- or focus-F1. Conformal
prediction chooses thresholds by **calibration** instead — each class gets the
cut at which its own held-out nonconformity scores reach a stated coverage level
— and that difference is testable on three things nothing else here can reach:
it needs no target metric and therefore no **focus set** (the question P6 ended
on), its rare-class correction comes from the rare class's own score
distribution rather than from a metric's noise, and it produces **sets**, which
is the first instrument in this project that can say where the model does not
know.

All of it is post-hoc over the cached OOF probabilities: 16 registered rows, ~50
s of numpy in total, no retrain and **no change to serving cost**. Thresholds are
calibrated per outer fold on the *other* folds and applied to the held-out one —
the `nested_gate` discipline, for the same reason. Implementation is section R of
`twotower_lab.py`; the coverage diagnostics are `conformal_report.py` →
`data/analysis_results/conformal_siam.csv`.

**Two instrument checks passed before any row was read.** `conf_siam_cos_marginal`
(R1c) is the control: a single pooled LAC threshold cannot reorder an arg-max, so
it must reproduce the source model *exactly*, and it does, to four decimals on
every column at 15 seeds. And realised coverage tracks nominal to ±0.0005 at
every level tested (0.9502 / 0.8998 / 0.7999 / 0.6998 against 0.95 / 0.90 / 0.80
/ 0.70) — worth stating because the folds here are **spatial blocks**, which
violates exchangeability by construction, so the guarantee was nominal only and
had to be measured rather than assumed. It holds.

## R1 — the merged2 read: calibration beats search on stable built-up

15 seeds, `full`, all over `siam_s2off_cos`'s cached probabilities. The row that
matters is the third, and the row that makes it readable is the fourth.

| | change-F1 | prec | rec | macro-F1 | **artStab** | as_veg |
| --- | --- | --- | --- | --- | --- | --- |
| `siam_s2off_cos` (N8b, base) | 0.6604 | 0.6513 | 0.6699 | 0.7035 | 0.6441 | 0.2296 |
| `conf_siam_cos_marginal` (R1c, control) | 0.6604 | 0.6513 | 0.6699 | 0.7035 | 0.6441 | 0.2296 |
| **`conf_siam_cos_nested` (R1e)** | **0.6609** | 0.6144 | 0.7151 | **0.7058** | **0.6799** | 0.1779 |
| `costgate_siam_s2off` (R1f, matched search) | 0.6600 | 0.6531 | 0.6673 | 0.7039 | 0.6517 | 0.2251 |
| `conf_siam_cos` (R1, alpha pinned 0.10) | 0.6472 | 0.5585 | 0.7696 | 0.6938 | **0.7192** | 0.1182 |
| `conf_deployed` (R1d, matched incumbent) | 0.6161 | 0.5121 | 0.7733 | 0.6633 | 0.7130 | 0.0872 |
| **`conf_crfe` (R1g, over Q7b)** | **0.6626** | 0.6203 | 0.7113 | 0.7050 | **0.7062** | 0.1568 |

**R1f is the cell section N and O never had.** N12 ran the cost gate over
`siam_cos` and N12b over the deployed model, so there was no *search* row on
`siam_s2off_cos` — the model every conformal row re-reads — and without it
calibration-vs-search would have been a comparison across two base models.
Registered before R1e was read, not after.

**With it, the result is clean: +0.036 stable-built-up recall from calibration
against +0.008 from search, on the same model, folds and 15 seeds, both free.**
0.6799 against 0.6517 is 0.028 apart with standard errors of 0.0035 and 0.0053 —
about 5x the section's ±0.005 noise bar, and the largest movement on this metric
in the ledger that does not pay for it in an aggregate. `art_stable_as_veg` falls
0.2296 → 0.1779, the same finding from the other side.

**State the cost precisely, because change-F1 being flat hides a real
movement.** R1e is +0.0005 change-F1 and +0.0023 macro-F1 — both inside noise —
but underneath, precision falls 0.6513 → 0.6144 and recall rises 0.6699 →
0.7151. It is not a free gain in the sense of dominating the base everywhere; it
is a **movement along the precision/recall curve that happens to conserve F1**
while buying a third metric. Whether that is the trade this product wants is the
same question N13 and Q9 ended on, and it is not answerable from plots.

**Alpha is a dial along a Pareto front, and the tuned point is interior.** The
first pass picked the grid's lower edge unanimously across 25 folds, which is not
readable, so `ALPHA_GRID` was extended below the conformal-native range to
0.005 — where every threshold saturates and the margin read collapses to exactly
the arg-max. It does not slide there. The profile (seed 0):

| alpha | 0.001 | 0.005 | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| change-F1 | 0.6638 | 0.6635 | 0.6613 | 0.6569 | 0.6460 | 0.5985 | 0.5918 |
| artStab | 0.655 | 0.673 | 0.679 | 0.689 | 0.708 | 0.728 | 0.728 |
| = arg-max | 98.6% | 97.0% | 95.7% | 94.0% | 91.1% | 87.2% | 86.4% |

So this is better understood as a **knob the user can set** than as a single
model: 0.005–0.05 buys +0.03 to +0.045 artStab for ≤0.004 change-F1, and 0.10
buys +0.075 for 0.013. The pinned-0.10 row is on the board because it was
registered a priori, not because it is the recommendation.

**R1g is the section's best cell and it breaks a standing precedent.** Q7b's CRFE
gate is the only *architectural* thing that moves stable built-up (0.6691), and
the conformal read moves it independently (0.6799 from 0.6441). Composed:
**0.7062, at change-F1 0.6626 — above both parts and above the base — and
macro-F1 0.7050.** The additive expectation from the base was 0.6441 + 0.025 +
0.036 = 0.705 against 0.7062 observed, i.e. **exactly additive**. The
preregistered prediction, from the F7 / N3b / O4 signature that two mechanisms
correcting one error land *between* their parts, is **wrong here** — and the
reason is visible in what each does: CRFE changes the probabilities, the
conformal read changes where they are cut, and those are orthogonal operations
on the same error rather than two attempts at it.

**R1d says the instrument is not model-agnostic.** The same pinned-0.10 read on
the deployed model buys +0.071 artStab for −0.040 change-F1; on the siamese,
+0.075 for −0.013. The siamese is the better base for a calibrated read — a
statement about which model to ship that no aggregate in this ledger makes.

**Tested negative on merged2, and all four are worth the space:**

| | change-F1 | macro-F1 | artStab | verdict |
| --- | --- | --- | --- | --- |
| `conf_siam_cos_ratio` (R1b) | 0.5289 | 0.6079 | 0.6539 | the multiplicative read of the same thresholds — `argmax p_k/(1-q_k)`, literally the cost gate with conformal costs — over-corrects catastrophically. The additive margin read is doing the work; the calibrated *cuts* are not usable as costs. |
| `conf_siam_aps` (R2) | 0.5529 | 0.6285 | 0.6439 | the adaptive score. Better conditional coverage (see R5) but as a point read it is the worst row here, and it does not move artStab at all. With four classes APS's extra freedom is all cost. |
| `conf_siam_setchange` (R3) | 0.4667 | 0.5430 | 0.3978 | the precautionary set read — change if the 90% set still admits a transition. Change recall 0.9214 at precision 0.3126. The exchange rate is bad and the collapse rule is the problem, not the sets. |
| `crc_siam_r90` (R4b) | 0.4619 | 0.5422 | 0.3587 | conformal risk control at a *guaranteed* 0.90 change recall. Delivered (0.9325) at precision 0.3071. |

## R4 — conformal risk control: the threshold question, priced

A different kind of row from the rest of the ledger. It does not try to win
change-F1; it converts *"what threshold should the map use"* — which no amount of
Oslo inspection can settle, since the AOI has zero labelled plots (G3/G4) — into
*"state the recall you require, and this is what it costs"*. The gate is the
largest threshold whose Clopper-Pearson upper bound on the calibration
miss-rate clears alpha, chosen per fold.

| | required rec | realised rec | prec | change-F1 |
| --- | --- | --- | --- | --- |
| `siam_s2off_cos` arg-max | — | 0.6699 | 0.6513 | 0.6604 |
| `crc_siam_r75` | 0.75 | 0.7851 | 0.5530 | 0.6489 |
| `crc_siam_r90` | 0.90 | 0.9325 | 0.3071 | 0.4619 |
| `s2off_centre_m3s3_bf` arg-max | — | 0.7264 | 0.5977 | 0.6557 |
| `crc_deployed_r75` | 0.75 | 0.7910 | 0.5173 | 0.6255 |

Both realised bounds clear their requirement with room, which is the guarantee
being conservative as designed. **The exchange rate is the finding: the first
0.115 of extra recall costs 0.098 of precision, the next 0.147 costs 0.246.**
And the matched pair says the siamese is the cheaper base for a recall
guarantee — a required 0.75 costs it 0.0115 change-F1 against the deployed
model's 0.0302.

## R5 — the coarse3 level: calibration loses to search, and finds a different class

The direct competitor to O3's cost gate, on the same cached probabilities and
folds. 5 seeds, which is all the coarse3 cache holds.

| | focus macro-F1 |
| --- | --- |
| `c3gate_siam_cos` (O3, search) | **0.4412** ±0.016 |
| `c3gate_siam_s2off_cos` (O3b, search) | **0.4318** ±0.023 |
| `conf_c3_siam_nested` (R5f, calibration + tuned alpha) | 0.4235 ±0.021 |
| `conf_focus_gate_siam` (R6, composition) | 0.4061 ±0.034 |
| `c3gate_deployed` (O3c, search) | 0.4076 ±0.010 |
| `conf_c3_siam_aps` (R5e) | 0.3842 ±0.005 |
| `base_siam_s2off_cos_fine` (arg-max) | 0.3847 ±0.005 |
| `conf_c3_siam_cos` (R5, alpha 0.10) | 0.3823 ±0.003 |
| `conf_c3_siam_cos_ratio` (R5b) | 0.3586 ±0.019 |

**Verdict: negative. Search keeps the field at the coarse3 level** — O3b 0.4318
against R5f's 0.4235, and R5's untuned read is flat against the arg-max. But the
per-class table shows the two instruments are not doing the same thing, and that
is the section's other result:

| (5 seeds) | Art→Crop (46) | Crop→Nat (114) | Nat→Crop (243) | Nat→Art (383) | Nat→Nat (2532) | Crop→Crop (1661) |
| --- | --- | --- | --- | --- | --- | --- |
| arg-max | 0.0000 | 0.0000 | 0.2389 | 0.4871 | 0.7627 | 0.7249 |
| O3b cost gate | **0.2115** | 0.0000 | 0.2382 | 0.4929 | 0.7623 | 0.7242 |
| R5 conformal, alpha 0.10 | 0.0000 | **0.0034** | **0.3391** | 0.5053 | 0.7589 | 0.7247 |
| R5f conformal, tuned alpha | 0.1713 | **0.0620** | **0.3537** | 0.5059 | 0.6844 | 0.6442 |

**Search breaks `Artificial -> Cropland`; calibration breaks `Nature ->
Cropland`.** Neither touches the other's class. And the second of those is the
point P6 left open: the 6-class widened gate reached 0.3623 on `Nature ->
Cropland` but only by **naming it a target**, which P6 correctly refused to do on
its own authority because the focus set is a product decision. R5 reaches 0.3391
of that **without naming any focus set at all** — the coverage level is the only
input. If the four commissioned transitions are the right four, use O3; if the
worry is that the four are incomplete, this is an instrument that does not need
to know.

R5f's cost is visible in the same table and disqualifies it as a shipping
candidate regardless of its focus macro: `Nature -> Nature` 0.7627 → 0.6844 and
`Cropland -> Cropland` 0.7249 → 0.6442. Tuning alpha against a four-class macro
lets it spend the majority classes, which `focus_macro_f1` cannot see. **A
reminder that this metric needs the aggregate columns read beside it**, exactly
as N-series `change_f1` needed the per-class ones.

## R6 — conformal nominates the focus set: negative, and the reason is the finding

The composition R5 suggests: let calibration *choose* which classes get O3's free
multipliers — the classes whose coverage falls short — and let the search tune
them, still scored on the commissioned four so the row cannot win by changing its
own metric. **0.4061 against O3b's 0.4318 and O3c's 0.4076 against R6b's 0.3869.
Negative on both bases.**

The mechanism is worth recording because it inverts an intuition this section
started with. **The first version of R6 was an artefact and was fixed before it
was recorded:** it calibrated each class's cut and measured coverage on the same
rows, where the conformal quantile puts coverage above `1 - alpha` by
construction, so every shortfall came out negative and the nomination ranked the
three majority classes (focus macro 0.3868). Nominating out-of-sample, by
leave-one-fold-out inside the calibration folds, is what the recorded row does.

And out-of-sample the shortfalls are **tiny and the rare classes over-cover**:
`Artificial -> Nature` +0.005, `Cropland -> Cropland` +0.005, `Nature -> Nature`
+0.002, against `Artificial -> Cropland` **−0.017** and `Nature -> Cropland`
−0.012. The two classes that are dead at the arg-max are the two that Mondrian
conformal covers *best*. So coverage shortfall is not a nominator for an F1
failure — it points at the majority classes — and the commissioned set beats it.

## R7 — what the sets say about the dead classes

The one thing in this section that no other instrument in the project can
produce, and it settles an argument P6 could only make from an oracle. Per-class
coverage of the 90% conformal set, `base_siam_s2off_cos_fine`, LAC:

| | Art→Crop (46) | Crop→Nat (114) | Nat→Crop (243) | Nat→Art (383) | Nat→Nat (2532) |
| --- | --- | --- | --- | --- | --- |
| marginal (one pooled cut) | 0.226 | **0.005** | 0.569 | 0.810 | 0.981 |
| **Mondrian (per-class cut)** | **0.917** | **0.902** | 0.901 | 0.903 | 0.899 |

**A pooled 90% guarantee is met by over-covering the majority class and almost
never covering the dead ones** — `Cropland -> Nature` is inside the pooled set
for 0.5% of its 114 plots. Per-class calibration takes both to 0.90 on the nose.

**So `Cropland -> Nature` is not a class the model has no signal on.** For 90% of
those plots the truth is inside a set the model can name; what it cannot do is
win an arg-max there. P6 inferred something like this from an endpoint oracle
(0.4766 ±0.246, the largest seed spread in the ledger) and could not separate
"unreachable at the arg-max" from "no signal". This separates them, at 5 seeds,
for free. It does **not** contradict P6's other reading — that these 114 plots
sit on the boundary `analyse_label_noise.py` calls the noisiest in the legend —
it sharpens it: the signal is there and it is not enough to be the top of nine.

The price, stated: mean set size 3.35 of 9 classes at alpha 0.10 Mondrian, and
only 18.9% of plots get a singleton. At the merged2 level it is much cheaper —
1.46 of 4, 62.3% singletons.

**And the same machinery gives the map an uncertainty band, which is what
CLAUDE.md asks for.** The standing verdict is that spatial smoothing removes
change pixels first, every time, and the lever is inputs or uncertainty. A
singleton conformal set is a defensible per-pixel "the model is confident at the
stated level", and on plots it separates sharply — merged2, `siam_s2off_cos`,
alpha 0.10 marginal: **86.0% of plots are singletons and the arg-max is 0.8994
accurate on them against 0.5446 on the rest**; Mondrian at 0.05, 36.6%
singletons, 0.9779 against 0.7754. Nothing in this section maps that to Oslo —
it is the obvious next step and it is not done here.

## Status and what section R changes

**One row is worth keeping and it is not a new model.** `conf_siam_cos_nested`
(R1e) and its composition with Q7b (`conf_crfe`, R1g) are post-hoc reads of
probabilities that already exist: **stable-built-up recall 0.6441 → 0.6799 →
0.7062 at change-F1 0.6604 → 0.6609 → 0.6626 and macro-F1 0.7035 → 0.7058 →
0.7050**, for zero training and zero serving cost. Stable built-up is the metric
`AUTORESEARCH.md` names the open frontier and records ~45 ideas failing to move;
this moves it by calibration, and beats the matched search instrument on the same
model by +0.028.

**What it does not settle.** The gain is a movement along the precision/recall
curve, alpha is a user-set dial rather than a fitted constant, and — as with
every map decision in this project since G3 — **Oslo has no labelled plots, so
whether a map read at a calibrated operating point is a better map is not
answerable from these numbers.** The deployed model is unchanged and
`s2off_centre_m3s3_bf` remains the settled choice (CLAUDE.md); nothing here is a
proposal to replace it.

**Tested-negative, do not redo:** the multiplicative read of conformal cuts, at
either level (R1b, R5b) · APS as a point read, at either level (R2, R5e) ·
set-membership collapsed to a precautionary change call (R3) · conformal
thresholds as a competitor to the coarse3 cost gate (R5/R5f — search wins) ·
coverage shortfall as a nominator for the focus set (R6, both bases, and note the
in-sample version of it is an artefact that will read as −0.02 focus macro).

### R7b — coverage, set size and the size distribution in full

`conformal_report.py` extended to record the whole size histogram rather than its
mean and two tails: a mean of 1.46 over four classes is consistent with almost
every plot at 1–2 and with a third empty against a third at 3, and those are
different products. 5 seeds, `siam_s2off_cos` (merged2) and
`base_siam_s2off_cos_fine` (coarse3), LAC score.

**Marginal validity is exact and needs no further comment** — 0.950 / 0.900 /
0.800 / 0.700 against nominal, every score, every mode, both levels, worst
deviation 0.001. **Conditional validity is where the modes separate**, and the
worst-covered class is the number to read:

| level | mode | alpha | coverage | worst-class coverage | which class |
| --- | --- | --- | --- | --- | --- |
| merged2 | marginal | 0.10 | 0.900 | **0.625** | Art→Veg (169) |
| merged2 | Mondrian | 0.10 | 0.900 | **0.897** | Veg→Art (716) |
| coarse3 | marginal | 0.10 | 0.900 | **0.005** | Crop→Nat (114) |
| coarse3 | Mondrian | 0.10 | 0.900 | **0.896** | Crop→Nat (114) |

**The price, as a distribution.** Share of plots at each set size:

| level / mode / alpha | mean | 0 | 1 | 2 | 3 | 4 | 5+ | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| merged2 marginal 0.05 | 1.40 | 0.000 | 0.673 | 0.259 | 0.069 | 0.000 | — | 2 |
| merged2 marginal 0.10 | 1.14 | 0.000 | **0.860** | 0.139 | 0.001 | 0.000 | — | 2 |
| merged2 Mondrian 0.05 | 2.10 | 0.000 | 0.366 | 0.295 | 0.216 | 0.124 | — | 4 |
| merged2 Mondrian 0.10 | 1.46 | 0.000 | 0.623 | 0.303 | 0.068 | 0.007 | — | 2 |
| coarse3 marginal 0.10 | 2.06 | 0.000 | 0.272 | 0.454 | 0.228 | 0.038 | 0.008 | 3 |
| coarse3 Mondrian 0.05 | 4.77 | 0.000 | 0.102 | 0.079 | 0.098 | 0.138 | 0.582 | 7.8 |
| coarse3 Mondrian 0.10 | 3.35 | 0.000 | 0.189 | 0.150 | 0.193 | 0.203 | 0.265 | 5.8 |

Three things the mean alone hides:

1. **Per-class coverage over nine classes costs a modal set size of 4.** The
   coarse3 Mondrian read at alpha 0.10 puts only 18.9% of plots in a singleton and
   26.5% in a set of five or more, out of nine. As a screening object that is
   weak; the merged2 read (62.3% singletons, mean 1.46) is the level where
   set-valued output is actually cheap, and it is also the level R1e's point read
   is defined on.
2. **A pooled LAC cut above 0.5 collapses the set into accept/abstain.** At alpha
   0.20 the merged2 marginal distribution is exactly 10.0% empty and 90.0%
   singleton and *nothing else* — mechanically, two of four classes cannot both
   clear a cut above one half. Anything tuned there is a selective classifier, not
   a set predictor, and should be described as one.
3. **Only APS produces empty sets at tight coverage** (4.6% at merged2 alpha 0.10,
   1.1% at coarse3) where LAC produces none. An empty set is the honest
   "conforms to nothing" and is arguably the more useful uncertainty flag of the
   two — which is a point in APS's favour despite R2/R5e both being negative as
   *point* reads.

---

# Section W — loss-side class weighting (2026-07-31)

The gap this section closes is a bookkeeping one. Every siamese recipe inherits
`BASE["loss"] = "focal"`, and `focal` maps to a **flat ones-vector** of class
weights at all three levels (`model_zoo.class_weights`, mode `"none"`). So the
long-tail levers this project has tested — G-H class-balanced sampling, cRT
(O2), tau-normalisation (O2c), set-level Dice (Q6b), the merged2 cost gate
(F3/N12), the coarse3 one (O3) — act on the **sampler**, on a **re-fitted
classifier**, on the **set**, or **post-hoc on the decision rule**. Not one of
them reweights the per-class term of the loss itself.

It reads as tested and is not. `hier_variants_2yr.csv` swept the four loss modes
**once, at one seed, on the 2023-era flat `wide` trunk**, and `focal` won on
change-F1 by 0.0046 — inside the ±0.005 band, below AUTORESEARCH's 3-seed floor.
On the two balanced-accuracy columns `cb_focal` won (merged 0.6705 vs 0.6594,
fine 0.4604 vs 0.4470), which is the shape a rare-class reweighting should have.
`focus_macro_f1` did not exist when that sweep ran. `TWOTOWER_RESEARCH.md` F5
still carries it as an open TODO.

`cb_focal` is Cui et al. (2019) effective-number weights (β 0.999, normalised to
mean 1) multiplying the **same** focal modulation, so W1 is a one-factor change
against N8b — unlike `weighted_ce`, which would drop focal too (T4ref: −0.030
change-F1 on its own). W1b adds `cb_levels="fine"`, new in `model_zoo.py`, which
keeps the weights off the 2-class gate and merged2. N8b re-ran in the same
command and reproduced to four decimals, so the plumbing is behaviour-preserving.

## The table (5 seeds, full read)

| | N8b base | W1 `cb` | W1b `cb_fine` | O3 gate | V1 `spec_tail4` |
|---|---|---|---|---|---|
| change-F1 | **0.6644** | 0.6499 | 0.6584 | **0.6644** | 0.6642 |
| change precision | **0.6507** | 0.5589 | 0.6035 | — | — |
| change recall | 0.6789 | **0.7765** | 0.7243 | — | — |
| macro-F1 | **0.7067** | 0.6962 | 0.7052 | **0.7067** | 0.7048 |
| `focus_macro_f1` | 0.3847 | 0.4444 | 0.4483 | 0.4318 | **0.4581** |
| `Nature -> Artificial` | 0.4871 | 0.4686 | 0.4615 | **0.4929** | 0.4871 |
| `Cropland -> Artificial` | **0.6093** | 0.5695 | 0.5742 | 0.5905 | **0.6093** |
| `Artificial -> Nature` | **0.4423** | 0.4172 | 0.4335 | 0.4322 | 0.4216 |
| `Artificial -> Cropland` | 0.0000 | 0.3223 | **0.3241** | 0.2115 | 0.3142 |
| `art_stable_recall` | 0.646 | 0.662 | **0.665** | 0.646 | 0.643 |
| `art_stable_as_veg` | 0.225 | **0.151** | 0.187 | 0.225 | 0.233 |
| `veg_stable_as_art` | **0.029** | 0.041 | 0.035 | 0.029 | 0.029 |

## W1 / W1b as a focus-macro lever. **NEGATIVE.**

**The whole +0.064 is one class.** `Artificial -> Cropland` goes 0.000 → 0.324,
and *every other focus class falls*: `Cropland -> Artificial` −0.035,
`Nature -> Artificial` −0.026, `Artificial -> Nature` −0.009. `focus_macro_f1`
is an unweighted mean of four, so breaking a zero moves it +0.081 on its own and
the other three give −0.017 back.

That is the same class O3, R5f and V1 already break, and **they break it for
free**: O3 and the conformal cuts leave change-F1 and macro-F1 *exactly*
unchanged, V1 costs 0.0002. W1b pays −0.006 change-F1, −0.047 change precision,
−0.035 on the section's second-best transition — and still lands **below V1's
0.4581**. A retrained instrument that costs aggregates and loses to a post-hoc
one that costs nothing is not a lever. **Do not pursue class weighting as a
route to the focus classes.** That is the fifth independent confirmation of
N0/P5, and the first one where the mechanism *did* fire and still lost.

The preregistered N13 counter-check confirms it: change precision 0.651 → 0.604
against recall 0.679 → 0.724. W1b calls more change less accurately, which is
the tuned threshold in a costume, and this product does not want it.

## The result is somewhere else: **stable built-up**

`art_stable_as_veg` **0.225 → 0.151** and `art_stable_recall` 0.646 → 0.665.
For scale, `SIAMESE_RESEARCH` has recorded this gap as **the deployed model's**
through four sections — deployed 0.196 ungated, 0.165 gated (N12), against the
siamese's 0.225 / 0.217. **W1's 0.151 beats both.** And it is a move no post-hoc
instrument can make: O3, V1 and every conformal row above leave
`art_stable_recall` and `art_stable_as_veg` at exactly 0.646 / 0.225, because
they only touch the coarse3 arg-max, while these are merged2 metrics.

**W1 vs W1b separates the mechanism, and against the preregistration.** W1b was
registered on the argument that the imbalance "lives entirely at the fine
level". It does not: W1 reweights merged2 and the gate and gets `as_veg` 0.151;
W1b leaves them unweighted and gets 0.187. **The merged2/gate reweighting is
what does the built-up work**, and the fine-level weights are what break the
dead class. The two effects are separable and they live at different levels.

Cost, stated plainly: −0.0145 change-F1, −0.0105 macro-F1, −0.092 change
precision, and `veg_stable_as_art` 0.029 → 0.041 (the false-built-up direction,
which is the error the user's map judgement weights most). Seed variance widens
on both arms (change-F1 sd 0.0024 → 0.0039/0.0040).

## Status

* **W1/W1b as focus-macro instruments: closed, negative.** V1 dominates them.
* **W1 as a stable-built-up instrument: open, and the strongest single move on
  that metric in the ledger.** The composition that has to be run before this is
  a verdict is **W1 (which owns built-up, and is a trained model) under O3 or V1
  (which own the dead class, and are free post-hoc)** — the built-up gain should
  survive, since the two act at different levels, and the focus macro should
  then clear V1's 0.4581. The F7 / N3b / O4 "lands between" precedent does *not*
  apply here, because for once the two instruments are not the same mechanism.
* Not run: `weighted_ce` (confounds the focal drop, T4ref), and the `level_weights`
  sweep that F5 also flags. A merged2-only `cb_levels` arm would isolate the
  built-up effect further and is the cheapest next thing after the composition.

## W2 / W3 — the composition. **The frontier stays two points.**

| 5 seeds, full | N8b | O3 | V1 | W1 | **W2** = O3(W1) | **W3** = V1(W1) |
|---|---|---|---|---|---|---|
| change-F1 | **0.6644** | **0.6644** | 0.6642 | 0.6499 | 0.6499 | 0.6492 |
| macro-F1 | **0.7067** | **0.7067** | 0.7048 | 0.6962 | 0.6962 | 0.6935 |
| `focus_macro_f1` | 0.3847 | 0.4318 | **0.4581** | 0.4444 | 0.4438 | 0.4490 |
| `Nature -> Artificial` | 0.4871 | **0.4929** | 0.4871 | 0.4686 | 0.4779 | 0.4697 |
| `Cropland -> Artificial` | **0.6093** | 0.5905 | **0.6093** | 0.5695 | 0.5832 | 0.5666 |
| `Artificial -> Nature` | **0.4423** | 0.4322 | 0.4216 | 0.4172 | 0.4123 | 0.3894 |
| `Artificial -> Cropland` | 0.0000 | 0.2115 | 0.3142 | 0.3223 | 0.3020 | **0.3703** |
| `art_stable_recall` | 0.646 | 0.646 | 0.643 | **0.662** | **0.662** | 0.661 |
| `art_stable_as_veg` | 0.225 | 0.225 | 0.233 | **0.151** | **0.151** | 0.158 |

**W2 — O3 over W1 finds nothing, and that is the finding.** `focus_macro_f1`
0.4438 against W1's 0.4444: **−0.0006**. On the plain-focal base the same gate
was worth **+0.047**. So the coarse3 cost gate's entire gain was headroom that
`cb_focal`'s fine-level weights have already spent — **a loss-side per-class
weight and a post-hoc per-class threshold are the same instrument applied at two
ends of the pipeline**, and they do not stack. O3 is free and W1 is not, so on
this axis O3 wins outright. (The merged2 columns come through bit-identical,
0.6499 / 0.151, which is the plumbing check O3-cannot-touch-merged2 passing.)

**W3 — V1 over W1 lands BETWEEN its parents, against the preregistration.** The
arm was registered predicting the F7 / N3b / O4 precedent would *not* apply,
because W1 and V1 act at different levels on different metrics. Half right:

* On `focus_macro_f1` the precedent **holds** — 0.4490, above W1's 0.4444 and
  below V1's **0.4581**. Two instruments aimed at the same class landed between,
  a fifth time.
* On built-up it **does not** — `art_stable_as_veg` 0.158 comes through nearly
  intact from W1's 0.151, because V1 only rewrites the coarse3 block masses.
* On the dead class alone it **adds**: `Artificial -> Cropland` **0.3703**,
  above both parents (W1 0.3223, V1 0.3142) and the highest in the ledger. It is
  paid for by `Artificial -> Nature` 0.3894 and `Cropland -> Artificial` 0.5666,
  both the worst of any row above, which is what drags the macro back under V1.

## Section W verdict

**Nothing composed here holds both parents' wins.** V1 still owns
`focus_macro_f1` at 0.4581 for free; W1 still owns `art_stable_as_veg` at 0.151
and is still the only instrument in the ledger that moves it. **The frontier is
two points, not one, and section W did not collapse it.**

W3 is the best single joint object — 0.4490 focus and 0.158 `as_veg`, i.e. most
of both — but it costs −0.0152 change-F1, −0.0132 macro-F1 and
`veg_stable_as_art` 0.029 → 0.043, and it clears neither parent on that parent's
own metric. On the ledger's own rule that is not a win.

* **Closed:** class weighting as a rare-class lever (W1/W1b), the O3 composition
  (W2 — same mechanism, does not stack), the V1 composition (W3 — lands between).
* **Open, and the only live thread:** W1 as a *built-up* instrument, unstacked.
  0.151 against a deployed 0.196 and a gated 0.165 is the largest move on that
  metric in the ledger, and it survives every composition tried. What it has not
  had is a price check on the map, where `veg_stable_as_art` 0.029 → 0.041 is
  the error the user's map judgement weights most and the aggregates are −0.015
  change-F1. That is a visual-inspection question, not a metric one.
* Still unrun: a merged2-only `cb_levels` arm to isolate the built-up effect
  from the fine-level weights that W2 just showed are redundant with O3, and the
  `level_weights` sweep F5 also flags.

## W4 — the Oslo map (2026-07-31)

`infer_s2.py --aois oslo --models s2off_centre_m3s3_bf siam_s2off_cos
siam_s2off_cb siam_s2off_cb_fine --seeds 5`, plus a second block at seeds 5–9
for the floor. Train/serve self-check passed (worst relative disagreement
4.24e-05). No Sentinel-2 read at inference for any of the four; W1/W1b serve at
the deployed model's exact cost, 5–6 s per Oslo forward pass.

**The floor first, per model, as CLAUDE.md requires.** Change-class IoU between
seeds 0–4 and seeds 5–9 of the *same* recipe:

| self-IoU (change only) | merged2 | coarse3 | change px A → B |
|---|---|---|---|
| `s2off_centre_m3s3_bf` (deployed) | 0.8423 | 0.7238 | 16,676 → 15,841 (−5.0%) |
| `siam_s2off_cos` (N8b) | **0.8524** | **0.8052** | 9,911 → 9,018 (−9.0%) |
| `siam_s2off_cb` (W1) | **0.7484** | **0.7078** | 11,918 → **15,406 (+29.3%)** |

**W1 is a materially less reproducible map, and this is the run's main finding.**
Its merged2 self-IoU is 0.748 against the deployed model's 0.842 and N8b's
0.852, and its change-pixel count swings **+29% between seed blocks** where
CLAUDE.md's stated normal is ±5% — which the deployed model (−5.0%) and N8b
(−9.0%) both respect. The plot metrics only hinted at this (change-F1 sd 0.0024
→ 0.0039); at the map it is a 9-point drop in self-agreement. A recipe whose map
moves this much between two draws of the same ensemble cannot be adjudicated
against the deployed one on a visual read, because the thing being inspected is
not stable.

**Against the deployed map** (block A, both at 5 seeds). Every number is far
below every floor above, so all three disagreements are real rather than seed
noise:

| vs deployed, change-only IoU | merged2 | coarse3 |
|---|---|---|
| `siam_s2off_cos` | 0.5813 | 0.3673 |
| `siam_s2off_cb` | 0.5748 | 0.2152 |
| `siam_s2off_cb_fine` | 0.5069 | 0.1850 |

**The plot-level "calls more change" does not survive the base-rate shift.** On
plots W1 had change recall 0.777 against 0.679 and precision 0.559 against
0.651. On the map it calls **fewer** change pixels than the deployed model
(11,918 vs 16,676), and the cut is concentrated in one class: `Vegetation ->
Artificial` **13,057 → 7,376 px, −43.5%**, against `Artificial -> Artificial`
+77k px. That is the built-up gain arriving exactly as the plot metric promised
(`art_stable_as_veg` 0.225 → 0.151 means built-up stops being read as
vegetation) — but the same reweighting that stops calling stable built-up
"vegetation" also stops calling new built-up "change". The plots are ~25% change
and the map is 0.5%; a class-balanced prior fitted at the first rate does not
transfer to the second.

**The dead coarse3 classes do appear on the map.** `Artificial -> Cropland` 0 →
**965 px** and `Cropland -> Nature` 0 → **1,028 px** under W1 (996 / 738 under
W1b), and `Nature -> Cropland` 16 → 1,947 px. `dead-coarse3-classes-are-coverable`
said the signal exists but cannot win an arg-max; class weighting is the first
thing to make it win one *at the pixel level*. Whether those pixels are right is
not answerable here — Oslo has zero labelled plots inside the AOI (G3/G4).

W1 also raises the S15 arg-max/group-sum disagreement: coarse3→merged2 agreement
99.38% (W1) and 99.32% (W1b) against the deployed 99.54% and N8b's 99.91%.

**Verdict.** The map does not support W1 as a candidate. It buys stable built-up
by suppressing `Vegetation -> Artificial` — the transition this product is
commissioned on — and it does so with a map that reproduces itself 9 points worse
than the incumbent. The plot-level built-up win is real and remains the ledger's
best on that metric; it does not survive contact with the 0.5% map base rate.
**Section W closes negative on all arms.** Maps are in
`data/inference/s2_20260731_100710` (seeds 0–4) and
`data/inference/s2_oslo_seedblockB` (seeds 5–9) if a visual read is still wanted.

# Section P7 — the state path as a *phase* rather than a term (2026-07-31)

The user's question: was the siamese ever adapted to take sparse single-year
data and then fed GLanCE 2018? It was — `encode_single` is exactly that
adaptation and N14 is exactly that test — but every arm that has used it, in
N14 and in all of section P, trained the state objective **jointly** with the
transition loss. Where the path lands was varied three ways and came back flat
three times. *When* it lands was never varied.

`siam_state_pretrain` runs the same objective as its own phase: n epochs of
`g(f(x)) -> {Nature, Cropland, Artificial}` over the pool alone, updating the
shared encoder and the state head, after which the fit proceeds normally from
those weights **with the auxiliary term off**. No weighting between the two
objectives at any step, and the encoder gets the pool's whole capacity instead
of a 0.3-weighted share. BatchNorm running statistics are frozen for the phase
(N4's defect); only training rows enter (`tr_idx`), so the early-stopping split
stays clean; the pool is cut to each fold's training blocks as everywhere else.

**Section P predicted this would be flat**, on the grounds that GLanCE's errors
are correlated with AlphaEarth's however they are fed. It is not flat, and P5's
reasoning survives anyway — see the reading at the foot.

## The table (5 seeds, `full`, same folds as N/O/P)

| | chg-F1 | macro | **focus** | **Art→Crop** | artStab | as_veg | as_chg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `siam_cos` (N2) | 0.6630 | 0.7011 | 0.3815 | **0.0000** | 0.642 | 0.234 | 0.124 |
| **`siam_cos_state_pre`** (P7a) | 0.6616 | **0.7081** | **0.4307** ±0.009 | **0.1915** ±0.036 | 0.661 | 0.215 | 0.124 |
| `siam_cos_state_pre_endo` (P7b, control) | 0.6515 | 0.6948 | 0.3880 | 0.0439 ±0.036 | 0.658 | 0.231 | 0.111 |
| `siam_cos_state_pre_shuf` (P7g, control) | 0.6609 | 0.7003 | 0.3830 | **0.0000** | 0.639 | 0.242 | 0.119 |
| `siam_cos_state_pre_long` (P7c, 100 ep) | 0.6608 | 0.7063 | 0.4254 | 0.1560 ±0.080 | 0.662 | 0.207 | 0.132 |
| `c3gate_siam_cos` (O3, incumbent) | 0.6630 | 0.7011 | **0.4412** ±0.016 | 0.2727 | 0.642 | 0.234 | 0.124 |
| `c3gate_siam_cos_state_pre` (P7d) | 0.6616 | 0.7081 | 0.4404 ±0.011 | 0.2578 | 0.661 | 0.215 | 0.124 |
| `siam_s2off_cos` (N8b) | 0.6644 | 0.7067 | 0.3847 | **0.0000** | 0.646 | 0.225 | 0.129 |
| **`siam_s2off_state_pre`** (P7e) | 0.6643 | 0.7092 | 0.4157 ±0.020 | 0.1075 ±0.080 | 0.658 | 0.204 | 0.138 |
| `siam_s2off_state_pre_endo` (P7k, control) | 0.6560 ±0.005 | 0.7006 | 0.3908 | 0.0276 | 0.662 | 0.220 | — |
| `siam_s2off_state_pre_both` (P7i) | **0.6450** ±0.003 | 0.6935 | 0.3816 | 0.0078 | **0.664** | 0.215 | — |
| `c3gate_siam_s2off_cos` (O3 on N8b) | 0.6644 | 0.7067 | 0.4318 ±0.023 | 0.2115 | 0.646 | 0.225 | 0.129 |
| `c3gate_siam_s2off_state_pre` (P7f) | 0.6643 | 0.7092 | **0.4383** ±0.008 | 0.2283 | 0.658 | 0.204 | 0.138 |
| `c3gate_siam_s2off_state_pre_both` (P7m) | 0.6450 | 0.6935 | 0.4204 | 0.1896 | 0.664 | 0.215 | — |

## P7i / P7k — the pretrain **pool** at the transition level, and it inverts

`STATE_PRETRAIN_RESEARCH` U1c/U3 measured the union pool — GLanCE **plus** the
plots' own endpoints — at +0.0234 paired over GLanCE alone, 24 of 25 folds, on
an LLTO *state* read, and recommended it. Run on the deployed base as P7i, it is
the worst arm in the section on the headline metric. P7k is the endogenous-only
control on the same base, which P7b never was (it ran on `cos`).

| pretrain pool | rows | chg-F1 | macro | artStab | Art→Crop |
| --- | --- | --- | --- | --- | --- |
| none (N8b) | — | **0.6644** | 0.7067 | 0.646 | 0.0000 |
| GLanCE only (P7e) | 13,118 | **0.6643** | **0.7092** | 0.658 | **0.1075** |
| endpoints only (P7k) | 12,828 | 0.6560 | 0.7006 | 0.662 | 0.0276 |
| **both (P7i)** | 25,946 | **0.6450** | 0.6935 | **0.664** | 0.0078 |

**Two monotone trends in opposite directions, and the endogenous rows drive
both.** change-F1 falls 0.6644 → 0.6643 → 0.6560 → 0.6450 as the endogenous half
enters and then doubles; `art_stable_recall` climbs 0.646 → 0.658 → 0.662 →
0.664 along the same ladder. P7i is **−0.019 change-F1**, four times the ±0.005
band, to buy **+0.018 artStab**. The gate does not change the ordering — P7m's
focus 0.4204 sits below P7f's 0.4383 — because the gate is a decision rule over
probabilities the pool has already made worse.

**It is change *recall* that goes, not precision**: 0.679 → 0.702 (P7e) → 0.667
(P7k) → 0.650 (P7i), at precision 0.651 / 0.631 / 0.646 / 0.640. That is this
project's most familiar failure mode arriving by a new route — the pretraining
phase suppresses change the way gates, fusion and guided filtering do
(`spatial-smoothing-eats-change`), and for a legible reason: the endogenous rows
train the encoder to answer *what state is this*, using the same plots the
transition fit then has to separate by *how they differ*. A representation
organised by land-cover class is not one organised by change.

**Verdict: P7i is negative and the state-level recommendation does not
transfer.** `siam_state_source="external"` (P7e) remains the pretrain option to
use — it is free on change-F1 (0.6643 against 0.6644), takes macro-F1 and artStab
with it, and breaks `Artificial -> Cropland` from 0.000 to 0.108. **Do not set
`siam_state_source="both"`**; CLAUDE.md and `state-pool-should-be-both`
recommended it on the state read and are corrected here.

**What this costs the LLTO protocol's standing: nothing, and that is the point.**
Section U/V's state-level reads are sound — they were re-verified under V0's
fixed fold geometry — and they still failed to predict the transition result.
That is section W's lesson for the third time in this project: **a plot-level
gain on an auxiliary objective is not evidence about the deployed metric**, and
the only remaining use for a state-level number is to *rank candidates for a
transition run*, never to conclude one.

**A code path was dead until this was run.** `_pretrain_state`'s endogenous
branch sliced `Xs[:, :d_end]` using `siam_columns_18`, which is `None` on the
`two_tower` base — so `source="endogenous"` or `"both"` raised `TypeError` on
every s2off recipe and had only ever run on `cos`. Fixed with
`_state_endpoint_slices`, the index-level twin of the `_state_endpoint_columns`
the external path already used. Additive: it is only reachable from a
configuration that previously crashed, so no existing number moves.

## P7a — **the first external-attributable GLanCE result in the project**

`focus_macro_f1` 0.3815 → 0.4307, and **the whole of it is one class**:
`Artificial -> Cropland` goes 0.000 → 0.1915 (precision 0.31, recall 0.14 on 46
plots). Every other commissioned transition is inside its seed spread. change-F1
is flat at −0.0014; macro-F1 is up 0.0070.

**Two controls, and this time the external arm wins both.**

* **Endogenous** (P7b): the identical phase on the training plots' own
  endpoints — no new data — reaches 0.0439 ±0.036. Non-zero, so the phase itself
  does something, but a quarter of the external arm and it *costs* change-F1
  (0.6515, −0.011).
* **Shuffled labels** (P7g): the identical pool, epochs and step count with the
  state labels permuted, so the encoder moves the same distance for the same
  cost and only the label-to-embedding correspondence is destroyed. It lands
  **exactly back on baseline** — focus 0.3830, `Artificial -> Cropland` 0.0000,
  artStab 0.639 against 0.642. That is what makes P7a attributable to the
  **land-cover content** of GLanCE's labels rather than to pretraining as such,
  and it is the control P7b alone left open.

N14b, P1 and P3 all had the *endogenous* control match or beat the external
pool. This is the first arm where it does not, and by 4 seed-sd on the class
that moves.

## The stable built-up movement, which is separate and is not free everywhere

`art_stable_recall` 0.642 → 0.661 with `art_stable_as_veg` 0.234 → 0.215 **and
`art_stable_as_change` unchanged at 0.124** on the AlphaEarth base. N11's rule is
that those last two are never read apart, and on this base they pass: the benign
error falls and the fabricated-habitat-loss error does not rise. N14d and P1 both
bought the same `as_veg` reduction by pushing `as_change` up, and this does not.

**On the deployed recipe's base it is not free.** P7e gives the same `as_veg`
0.225 → 0.204 but `as_chg` 0.129 → 0.138. Milder than N14d's trade, and in the
same wrong direction. A deployment decision has to price that, not the AEF row.

## P7d / P7f — against O3, the actual incumbent

**On the AlphaEarth base the two do not add.** P7a alone (0.4307) does not reach
O3's free gate (0.4412), and the composition is a tie (0.4404 ±0.011). They are
two routes to the same gain: a representation that has seen single-date states
and a decision rule that reweights the softmax both break the same dead class,
and neither leaves anything for the other.

**On the s2off base the composition is nominally the best focus number in the
ledger** — 0.4383 ±0.008 against O3's 0.4318 ±**0.023**. The gap is +0.0065
against a spread three times that, so it is **not** a claim. What it does carry
that O3 cannot is the aggregates: O3 returns the merged2 probabilities untouched
by construction, so macro-F1 0.7092 vs 0.7067 and artStab 0.658 vs 0.646 belong
to the pretraining alone.

## P7c — not an undertrained phase

100 pretrain epochs instead of 30 gives 0.4254 ±0.017 against 0.4307 ±0.009, and
`Artificial -> Cropland` 0.156 ±0.080 against 0.1915 ±0.036 — the same result
with double the variance. The phase is saturated at 30 epochs; more of the pool
is not a lever.

## What this changes in the ledger, and what it does not

**Section P's finding 2 is now wrong as written.** "Where the path lands does not
matter — discarded side-loss, output parameterisation, or input feature all give
the same flat result from the same labels" is still true of all three, but the
implied conclusion that the labels are inert is not. **When** the path lands
matters, and it is the difference between 0.000 and 0.19 on a class six prior
arms left dead.

**P5 is not overturned, and it explains the shape of this result.** A correlated
state reader cannot help the transition model *decide* — which is why P1 and P3,
where the state path feeds the decision, are flat. What a pretrain phase does
instead is organise the representation before the transition loss ever sees it,
and the class it rescues is the one the ledger already knows is not
evidence-limited: P5's oracle sweep put `Artificial -> Cropland` at 0.000 even
with **both endpoint states known exactly**, i.e. an arg-max problem, not a
signal problem (`dead-coarse3-classes-are-coverable`). P7a is a third instrument
on the same arg-max — after O3's gate and section V's tail specialist — and its
tie with O3 is what a shared cause predicts.

**`Cropland -> Nature` stays dead** in every arm above, as it does under O3 and
under the 6-class gate. Consistent with P6: that class is not merely unreached.

## Status

**Positive, bounded, and not yet a deployment recommendation.** It does not beat
O3 on the AlphaEarth base and only ties it within noise on s2off; its distinct
contribution is macro-F1 and stable built-up, which are free on the AEF base and
cost `as_change` on the s2off one. Serving cost is unchanged — the state head is
discarded and the phase is training-time only (+11 s/seed over the 12 s base).

**What would settle it, in order.** (1) The s2off composition at 15 seeds, since
±0.023 on O3 is what currently blocks the claim. (2) An Oslo map for P7e/P7f —
section W's lesson is that a plot-level built-up win need not survive the 0.5%
map base rate, and `Artificial -> Cropland` pixel counts are the direct read on
whether the rescued class is real. (3) Nothing else; the phase is saturated and
both controls have been run.

## P7h — the Oslo map (2026-07-31)

`siam_s2off_state_pre` (P7e), 5 seeds, gate-off, plus a disjoint block (seeds
5–9) for the floor. The recipe is in `infer_s2.py:fit_models`; the pool is
handed over whole at deployment because there is no held-out block for the
per-fold split to protect. Serving is unchanged — 6 s over 5 forward passes, no
Sentinel-2 composite, no GLanCE at inference.

```
data/inference/s2_20260731_120119   seeds 0-4
data/inference/s2_20260731_120223   seeds 5-9 (the floor)
data/inference/s2_20260731_100710   the incumbents, same seed block
```

**The floor first, as the rules require.** merged2 change-class IoU **0.8406**
across the two blocks — the same ~0.84 CLAUDE.md records, so this model
reproduces itself as well as any other and nothing below is a broken run. The
coarse3 floor is much lower, 0.7186, and the rare classes are why:
`Artificial -> Cropland` has a seed-block IoU of **0.546** on counts of 266 and
473 px, a 1.78× swing.

| coarse3 px | deployed | `siam_s2off_cos` | **state_pre (A)** | state_pre (B) | state_pre gated |
| --- | --- | --- | --- | --- | --- |
| Artificial → Artificial | 1,095,592 | 1,040,339 | **984,159** | 981,660 | 984,133 |
| **Artificial → Cropland** | **0** | **0** | **266** | 473 | **1,744** |
| Artificial → Nature | 5,024 | 3,487 | 2,604 | 2,657 | 1,885 |
| Cropland → Artificial | 960 | 108 | 137 | 199 | 93 |
| Cropland → Nature | 0 | 0 | **0** | 0 | 0 |
| Nature → Artificial | 10,609 | 6,512 | 6,894 | 7,860 | 6,918 |
| Nature → Cropland | 16 | 25 | 34 | 24 | 34 |
| Nature → Nature | 1,803,267 | 1,866,889 | **1,920,800** | 1,923,002 | 1,920,641 |
| change total | 16,609 | 10,132 | 9,935 | 11,213 | 10,674 |

**The dead class does break at the pixel level.** 266 px ungated and 1,744 px
gated, against **exactly zero** for both incumbents on this base. Third
instrument to do it after O3 and section V's tail specialist, and the first to
do it from the representation. But 266 px at a seed-block IoU of 0.546 is a
class that exists rather than a class that is located, and Oslo has no plots to
say whether any of those pixels is right (G3/G4).

**`Cropland -> Nature` stays at 0 px**, as everywhere else. P6's reading holds.

**And the built-up read goes the wrong way — which the plot metrics did not
say.** `Artificial -> Artificial` falls 1,040,339 → 984,159 against its own
base, −56k px, and `Nature -> Nature` rises by almost exactly the same 54k. The
model reads 56k px of stable built-up as stable nature that its base did not.
On plots this recipe *improved* the same error (`art_stable_as_veg` 0.225 →
0.204, `art_stable_recall` 0.646 → 0.658). Against the deployed model the gap is
−111k px (−10.2%), of which the siamese line already owned −55k.

**This is not seed noise and the floor is what says so.** state_pre against its
own base is 0.7370 merged2 change IoU and 0.4095 coarse3, against self-floors of
0.8406 and 0.7186. The two maps genuinely differ, and the difference is the
stable-built-up/stable-nature boundary.

## P7h verdict — SUPERSEDED by the user's visual read (2026-07-31)

**The user inspected the maps and judged `siam_s2off_state_pre` the best of
them visually.** That is the authority this project settles map questions on —
the same read that settled the deployed model (CLAUDE.md) — and it overrides the
structural verdict recorded below, which stands unedited as what the metrics
said. Architecture figure regenerated to match:
`data/analysis_results/siam_s2off_state_pre_architecture.{svg,png}`
(`src/plot_siam_arch.py`).

**What that costs is on the record and does not go away**: −56k px of stable
built-up against its own base, with `Nature -> Nature` up the same amount. If a
later read of the Oslo built-up areas looks wrong, this is the first thing to
check, and the paragraph below is the measurement to check it against. The
metric-side reading was that this is section W's failure repeating; the visual
read says the pixels it moves are better ones. **Both are recorded; neither is
deleted.**

**Not yet a change to the deployed model.** `s2off_centre_m3s3_bf` remains what
CLAUDE.md names until the user says otherwise — "best of the maps shown" is not
the same statement as "replaces the deployment", and P7e has one AOI, no plots
inside it (G3/G4), and a 5-seed ensemble behind it.

---

**Verdict (metric-side, superseded above): the map does not support P7e as a
candidate.** It is section W's
lesson a second time, and worth stating in the general form now that two
independent recipes have produced it: **a plot-level stable-built-up gain
measured at the plots' ~25% change base rate has now twice failed to appear on a
map at 0.5%, and twice arrived inverted.** `aef-only-map-is-the-stable-artificial-benchmark`
is the standing user judgement, and 56k px of built-up read as nature is the
wrong side of it.

**What survives P7.** The plot-level result stands exactly as section P7 states
it — first external-attributable GLanCE effect, both controls beaten, dead class
broken at the arg-max. It is a finding about how single-date labels reach this
model, not a deployment candidate, and the deployed model is unchanged. If the
`Artificial -> Cropland` rescue is wanted on a map, O3's gate does it at zero
training cost and without touching the built-up boundary — its merged2 raster is
bit-identical to the ungated model's by construction, which is precisely the
property P7e lacks.

---

# Section Q10 — SNIIF-Net, on the P7e base (2026-08-03)

*"Siamese change detection based on information interaction and fusion
network", Sci Rep 15 (2025), s41598-025-15468-w.* A second bi-temporal siamese
change-detection paper, brought in the way section Q brought Zhang et al.: read
for the modules that have a form on a tabular endpoint encoder, transcribed at
the smallest size that tests each one, run on the base a deployment decision
would be taken on.

**The base is `siam_s2off_state_pre` (P7e)**, not N8b — the user asked for these
on it, and it is the arm the user's visual read selected. Every arm below is
therefore state-pretrained on GLanCE, gate-off, and reads no Sentinel-2 at
inference; serving cost is unchanged throughout.

## What the paper introduces, and which parts were new here

| their module | what it is | disposition |
| --- | --- | --- |
| **FPFM** | dual-branch fusion: a subtraction branch `\|f1 − f2\|` **and** a summation branch `f1 + f2`, residual, then summed | **already tested.** It is section Q's CRFE (Q2 / Q7b / Q8), 15 seeds, where the sum operator's marginal effect was inside the seed spread once the gate was present. Not re-run. |
| **FIIM** | spatial attention letting the two branches interact **before** the difference is taken | **Q10f/g.** No tabular form for the spatial part, but the *placement* has one — and it is the one placement never tried: Q7's gate sits downstream of the subtraction. |
| **MSSM** | contrastive supervision of the feature pair at **every** decoder scale, not only the output | **Q10a–e.** The headline idea and the only one with no precedent here. |

MSSM's loss is also double-hinged — `max(θ − D, 0)²` on changed pairs against
`max(D − ε, 0)²` on unchanged — where this project's cosine term hinges only the
change side and drives every stable pair towards cos exactly 1. That is Q10e,
and the Euclidean form is Q10d.

**One implementation note that decides how Q10d reads.** Their `D` is Euclidean
on raw feature maps. Taken literally across a 512-wide BatchNorm+GELU stage and
a 128-wide linear one, a single pair of margins is a different objective at each
depth. `_pair_margin_term` therefore takes the distance on **L2-normalised**
features, `D = sqrt(2(1 − cos)) ∈ [0, 2]`. That makes cosine-vs-Euclidean a
reparameterisation and leaves the squared hinge and the double margin as Q10d's
actual content — a difference from Q10b is about loss *shape*, not about the
metric.

## The table (15 seeds for the replicated arms, 5 otherwise; `full`, same folds as N/O/P/Q)

| | seeds | chg-F1 | macro | **focus** | **Art→Crop** | artStab | as_veg | as_chg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **`siam_s2off_state_pre`** (P7e, base) | 15 | 0.6657 | 0.7092 | 0.4170 ±0.022 | 0.1147 ±0.094 | 0.657 | 0.206 | 0.136 |
| `…_mssm` (Q10a) | 15 | 0.6654 | **0.7094** | 0.4199 ±0.023 | 0.1279 ±0.096 | 0.656 | 0.207 | 0.138 |
| `…_mssm_all` (Q10b) | 5 | 0.6646 | 0.7095 | 0.4184 | 0.1183 | 0.658 | 0.203 | 0.139 |
| `…_mssm_ctrl` (Q10c, control) | 5 | **0.6657** | 0.7095 | 0.4168 | 0.1159 | 0.660 | 0.203 | 0.137 |
| `…_mssm_euclid` (Q10d) | 5 | 0.6637 | 0.7079 | 0.4141 | 0.1031 | 0.656 | 0.209 | 0.136 |
| `…_dm` (Q10e) | 5 | 0.6640 | 0.7089 | 0.4149 | 0.1024 | 0.657 | 0.207 | 0.135 |
| **`…_fiim`** (Q10f) | 15 | 0.6647 | 0.7080 | **0.4427** ±0.017 | **0.2253** ±0.051 | 0.669 | 0.199 | 0.132 |
| `…_fiim_self` (Q10g, control) | 15 | 0.6639 | 0.7074 | 0.4332 ±0.014 | 0.1983 ±0.049 | **0.672** | **0.197** | **0.131** |
| `c3gate_…_state_pre` (P7f, incumbent) | 15 | 0.6657 | 0.7092 | 0.4417 ±0.011 | 0.2330 ±0.024 | 0.657 | 0.206 | 0.136 |
| `c3gate_…_state_pre_fiim` (Q10h) | 15 | 0.6647 | 0.7080 | **0.4452** ±0.011 | **0.2663** ±0.029 | 0.669 | 0.199 | 0.132 |

## Q10a–d — MSSM. **FLAT, and it closes Q5's question rather than reopening it.**

At 15 seeds the pair term at the hidden stages is +0.0029 focus, −0.0003
change-F1, +0.0002 macro, −0.0014 artStab. Everything inside ±0.005, and
`Artificial -> Cropland` moves 0.1147 → 0.1279 on a seed sd of **0.094** — which
is not a movement, it is the same distribution sampled twice. The literal
four-scale reading (Q10b), the Euclidean double-hinge form (Q10d) and the
double margin alone (Q10e) are all flat-to-slightly-negative at 5 seeds, and
Q10c — the control that spends the same total auxiliary weight at the final
embedding — is flat too, so there is not even an effect for "multi-scale" to be
the explanation of.

**This was preregistered as the test of Q5's explanation and it confirms it.**
Q5 hung a *classification* head off each encoder stage, came back flat, and the
reading was that a three-layer encoder under a three-level nested loss is
already supervised at depth. The obvious objection was that this says nothing
about the pair *geometry*, which no loss in the model touches anywhere but at
`z` — the stages are free to interleave the two dates however they like provided
the last layer can separate them. They are free, and constraining them changes
nothing. **The generalised statement is now: this encoder is insensitive to
auxiliary supervision at depth, whatever the auxiliary objective is.** Two
independent objectives, one with parameters and one without, both flat.

The double margin (Q10e) deserves its own line because the prior was reasonable
and it still failed: stable plots are 4:1 here, so the un-hinged stable term is
most of the auxiliary gradient and it goes on pulling pairs that already agree.
Releasing that capacity at cos ≥ 0.9 buys −0.0003 change-F1 and −0.0008 focus.
The capacity was not the constraint.

## Q10f/g — FIIM. **Real, but the control takes most of it and the gate takes the rest.**

Q10f is the section's one non-flat arm, and on the arg-max read it is large:
**focus +0.0257**, `Artificial -> Cropland` **0.115 → 0.225**, `art_stable_recall`
**+0.0115**, and both N11 counter-checks move the right way (`as_veg` −0.0076,
`as_chg` −0.0039). The seed spread *falls* on all of them — focus sd 0.022 →
0.017, Art→Crop sd 0.094 → 0.051, artStab sd 0.0069 → 0.0063 — which is the
opposite of the N3b noise signature and means the module is making a wobbly
result reproducible, not adding a wobble of its own. Cost: change-F1 −0.0010 and
macro −0.0012, both inside the band, with change *recall* −0.0045 at precision
+0.0019.

**Then the control.** Q10g is the same gate at the same parameter count with the
same init draw, reading each date **twice** instead of reading the pair — so the
only thing removed is the cross-branch information the module exists to add. It
reproduces **63% of the focus move** (0.4332 of 0.4427, against a base of
0.4170), **77% of the Art→Crop move**, and it is **better than Q10f on stable
built-up** (0.672 against 0.669) and on both counter-checks.

**So the mechanism is placement, not interaction.** What helps is a
multiplicative per-date gate applied *upstream* of the subtraction — where it
can change `z24 − z18`, the cosine feature and what the pair losses read — and
Q7's channel attention could never do that, sitting as it does on the assembled
block downstream of all three. Reading the other date on top of that is worth
+0.0095 focus and +0.027 Art→Crop over the control, both about 2 SE at 15 seeds:
suggestive, not established, and it is *negative* on the built-up numbers the
section actually cares about. **The honest one-line version is that SNIIF-Net's
FIIM pointed at the right place in the network and its stated content — the
interaction — is not what pays there.**

## Q10h — against the incumbent, which is not an arg-max

The incumbent on the commissioned transitions is P7f, O3's free coarse3 gate
over this base, at focus 0.4417 — and Q10f's arg-max 0.4427 merely *reaches* it.
Gate-to-gate the module is worth **+0.0035 focus**, inside the band. This is O4's
signature again: a representation change and a decision rule that break the same
failure by different routes do not add.

What survives the gate is the part the gate does not touch. `art_stable_recall`
0.657 → 0.669 and `as_veg` 0.206 → 0.199 are unchanged by O3 in either row, so
they are FIIM's alone, and `Artificial -> Cropland` does add: 0.2330 → 0.2663
(+0.033 at sd 0.024/0.029), the best number that class has recorded. Against
that: `Artificial -> Nature` 0.4195 → 0.4121 and `Nature -> Artificial` 0.5151 →
0.5063 inside the gated read, and `veg_stable_as_art` 0.0328 → 0.0346 — still
under the deployed model's 0.0354, so the standing counter-check does not blow
up, but it is moving in the wrong direction.

## Section Q10 verdict

**Two of three modules are already answered or flat; the third works for a
reason other than the one it is sold on.**

* **MSSM is flat** and, with Q5, closes the auxiliary-depth question for this
  encoder in general form.
* **The double margin is flat.** The stable term's capacity was not a constraint.
* **FPFM was section Q.** Do not re-run it.
* **FIIM is a genuine but modest built-up lever whose arg-max headline mostly
  evaporates gate-to-gate**, and whose control takes most of what is left. It is
  the same trade as Q7b — a point of one recovery transition for built-up — at
  roughly the same price, and it is **not** a decision anything here should be
  taken on: nothing in this section has been near a raster, the ~0.84 change-class
  self-IoU floor has not been computed for a FIIM map, and Oslo still has zero
  labelled plots (G3/G4).
* **The one result worth carrying forward** is the finding underneath Q10f/g:
  **a gate upstream of the subtraction is a different object from a gate
  downstream of it**, and this section only ever tried the downstream one. Q7's
  SE gate and Q10g's per-date gate are compatible in principle and untested
  together. **Answered in section Q11 below, and the answer is negative** — they
  are structurally different and functionally one lever, and the composition
  lands between its parts on all three axes.

**Tested-negative, do not redo:** multi-scale contrastive supervision of the
endpoint pair at the encoder's hidden stages, at any of {cosine, Euclidean
squared-hinge} × {stages, stages+final} (Q10a–d) · a double margin on the stable
side of the cosine term (Q10e) · FPFM's summation branch (= Q2/Q8c, section Q) ·
cross-branch *interaction* as the claimed mechanism of an upstream gate (Q10g —
the self-gate control reproduces it).

---

# Section Q11 — the composition Q10 left open. **NEGATIVE: they are one lever.**

Q10 found that a multiplicative gate *upstream* of the endpoint subtraction is a
structurally different object from Q7's SE gate *downstream* of it — only the
upstream one can change `z24 − z18`, the cosine feature and what the pair losses
read — and left the two untested together. This is that test: a 3×2 on one base
and one set of folds, 15 seeds, {nothing, self-gate, cross-gate} upstream ×
{nothing, SE gate} downstream. **The preregistered null was O4's
between-the-parts result and the null is what happened.**

## The 3×2 (15 seeds, `full`, arg-max read; `focus_macro_f1` · Art→Crop · artStab)

| upstream ↓ / downstream → | **nothing** | **+ SE gate** (`crfe='attn'`) |
| --- | --- | --- |
| **nothing** | 0.4170 · 0.115 · 0.657 (P7e) | 0.4309 · 0.169 · 0.661 (Q11a) |
| **self-gate** (control) | 0.4332 · 0.198 · **0.672** (Q10g) | 0.4310 · 0.171 · 0.666 (Q11d) |
| **cross-gate** (FIIM) | **0.4427** · **0.225** · 0.669 (Q10f) | 0.4361 · 0.198 · 0.665 (Q11c) |

**Every composed cell sits between its parts, and below the better of them.**
Upstream alone is +0.0257 focus, downstream alone is +0.0139, and together they
are +0.0191 — where the sum would have been +0.0396 and the better part alone is
+0.0257. `Artificial -> Cropland` repeats it exactly: 0.225 and 0.169 alone,
0.198 composed. `art_stable_recall` likewise: 0.669 and 0.661 alone, 0.665
composed. Three axes, one shape.

**So the two gates are structurally different and functionally the same lever.**
Q10's placement finding is not withdrawn — it is what it always was, an
observation about what each gate is *able* to modify — but the thing they modify
turns out to be one failure, and correcting it twice corrects it once. This is
now the **fifth** independent instance of the F7 / N3b / O4 signature on this
model, and at that count it should be treated as a prior rather than as a result:
**two mechanisms aimed at the same failure on this target land between their
parts.** Predict it before running the composition next time.

## The one thing the composition does buy, and it is not a win

The composed arms **do not pay FIIM's change-F1 cost**: 0.6669 (Q11c) and 0.6671
(Q11d) against 0.6647 for FIIM alone and 0.6657 for the base, with the seed sd
falling from 0.0048 to 0.0026–0.0037. They keep most of the built-up gain at the
same time (artStab 0.665, `as_veg` 0.199, against the base's 0.657 / 0.206). All
of that is inside ±0.005 and none of it is a win by this project's rule — but the
shape is worth recording, because it means the downstream gate's role in the
composition is to *stabilise* rather than to add.

**The cross-branch half is still not the mechanism.** Q11d (self-gate + SE) ties
Q11c (cross-gate + SE) on every aggregate — focus 0.4310 against 0.4361,
change-F1 0.6671 against 0.6669, artStab 0.666 against 0.665. Q10g's reading
survives the composition unchanged.

## Q11b — `crfe='full'` does not transfer to this base. **Base-dependence, recorded.**

Section Q recommended Q7b (the full CRFE module, sum + gate) as the genuine
candidate on the built-up frontier, measured on N8b. On the state-pretrained base
it is **negative on the rare class**: `focus_macro_f1` 0.4085, *below the base's
own 0.4170*, and `Artificial -> Cropland` 0.087 against 0.115. It does give the
best `art_stable_recall` in the whole family (0.670, `as_veg` 0.199) — the
built-up half of Q7b's finding transfers and the rare-class half inverts.

The mechanism is legible from section Q's own control: the sum operator's job
there was "to make the block more redundant", and P7e's pretraining has already
reorganised that block. **A module validated on one base is validated on one
base** — N18's lesson, and now the second time this section has paid for it.

## Section Q11 verdict

* **The composition is negative.** Upstream and downstream gates do not add; use
  one. On the arg-max read the one to use is the upstream gate alone (Q10f).
* **Gate-to-gate nothing here reaches the incumbent by the section's own rule.**
  P7f 0.4417 · Q11g 0.4420 · Q10h 0.4452 · **Q11f 0.4466**, the last being the
  best `focus_macro_f1` recorded in this document — and +0.0049 over the
  incumbent, which is *inside* ±0.005. It is not a win. It is the ceiling of
  where this family lands.
* **`crfe='full'` should not be carried onto the state-pretrained base.**
* **Nothing in Q10/Q11 changes the deployment.** No raster, no self-IoU floor,
  no labelled plots in the AOI (G3/G4).

**Tested-negative, do not redo:** composing an upstream gate with a downstream
one (Q11c/Q11d — between the parts on all three axes) · `crfe='full'` on the
state-pretrained base (Q11b, negative on the rare class) · the cross-branch half
of FIIM as the mechanism, now under composition as well as alone (Q11d = Q11c).

## Q10i — the Oslo map (2026-08-03). **FIIM suppresses the commissioned class.**

`siam_s2off_state_pre_fiim` registered in `infer_s2.fit_models` with its own
fitted coarse3 cost vector (`fit_coarse3_costs.py --idea
siam_s2off_state_pre_fiim --seeds 5`; `Artificial -> Cropland` ×2.0,
`Nature -> Artificial` ×1.2). Gate-off, AlphaEarth-only, 6 s over five forward
passes — serving cost indistinguishable from P7e's.

**Both seed blocks were run for both models**, because P7e has two Oslo maps on
disk (`s2_20260731_120119` = seeds 0–4, `s2_20260731_120223` = seeds 5–9) and
pairing across them is precisely the S18 error. A re-run of P7e's seed-5 block
reproduced `120223` **bit-identically** through the edited `model_zoo`, which
also re-confirms the Q10/Q11 additions are inert at their defaults.

### The floors, then the comparison (coarse3, change-only mean IoU)

| | change-only IoU |
| --- | --- |
| P7e against itself, block A vs B | 0.7186 |
| **FIIM against itself**, block A vs B | **0.7525** |
| FIIM vs P7e, block A (matched) | 0.7127 |
| FIIM vs P7e, block B (matched) | 0.6561 |

**Both cross-model reads sit below both self-floors, so the two maps genuinely
differ.** And FIIM's own floor is *higher* than P7e's — it is the more
reproducible model across seed draws, which is the raster-side counterpart of
its shrinking seed spread on plots.

### What the difference is, and it is one class

| coarse3 class | FIIM A | P7e A | FIIM B | P7e B | cross-IoU | self-IoU |
| --- | --- | --- | --- | --- | --- | --- |
| **`Nature -> Artificial`** | **4,258** | 6,894 | **4,928** | 7,860 | **0.607 / 0.610** | 0.840 / 0.819 |
| `Cropland -> Cropland` | 45,594 | 40,058 | 44,272 | 39,077 | 0.877 / 0.883 | 0.964 / 0.962 |
| `Artificial -> Cropland` | 375 | 266 | 384 | 473 | 0.661 / 0.488 | 0.546 / **0.757** |
| total change px | 8,170 | 10,399 | 8,518 | 11,697 | −21% / −27% | — |

**`Nature -> Artificial` loses 37–38% of its pixels, in both blocks.** The
class's own seed-block swing is +14% (P7e) and +16% (FIIM), so this is roughly
2.5× outside it, and its cross-model IoU of 0.61 is far under its ~0.83
self-floor. `merged2` says the same thing in the read that decides *whether*
change occurred: `Vegetation -> Artificial` 4,901 against 7,468, IoU 0.642.
The pixels go to `Cropland -> Cropland` (+14%) and `Artificial -> Artificial`.

**So the plot-level built-up win is bought by deleting habitat-loss events.**
On plots FIIM moved all three stable-built-up rows the right way at once —
`art_stable_recall` 0.657 → 0.669, `as_veg` 0.206 → 0.199, `as_chg` 0.136 →
0.132 — and that was the whole reason to fetch a raster. The raster says the
mechanism is suppression: fewer pixels called `Nature -> Artificial` improves
every stable-built-up statistic and costs the one class this product is
commissioned on. The plot-side warning was there and was inside the band —
`fine_f1_nature_to_artificial` 0.5080 → 0.5022, −0.006 — and on the map that
same class loses 37%.

**This is the third instance and it should now be stated as a rule.** Section W,
P7h and Q10i: *a plot-level stable-built-up gain measured at the plots' ~25%
change base rate has now three times failed to survive the map's 0.5%, and three
times arrived as change suppression.* `spatial-smoothing-eats-change` is the
same finding reached from the raster side. **Do not read a stable-built-up
improvement on plots as a map improvement without the `Nature -> Artificial`
pixel count.**

### What is real, and is small

`Artificial -> Cropland` — the dead class Q10 rescued — is genuinely more stable
on the map: self-IoU 0.757 against P7e's 0.546, and 375/384 px across blocks
against P7e's 266/473. The class the plots said FIIM fixed is the class the map
agrees it fixed. It is 0.013% of the raster.

**Verdict: do not adopt.** `siam_s2off_state_pre` remains what the user's visual
read selected and `s2off_centre_m3s3_bf` remains the deployed model. The map is
on disk at `data/inference/s2_20260803_134759` (block A) and
`data/inference/s2_oslo_q10_blockB/s2_20260803_134925` (block B) if the user
wants to inspect it — that read outranks this one, as it did in P7h.
