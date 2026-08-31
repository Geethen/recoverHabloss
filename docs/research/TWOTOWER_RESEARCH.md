# Two-tower (AlphaEarth + Tessera) accuracy research backlog

**Objective.** Raise the merged2 accuracy of the AlphaEarth+Tessera two-tower on
the RECOVER/HABLOSS transition frame — the model deployed as
`data/inference/best_20260725_114640` (`mc_dropout_scalars`). Two reads, both
tracked, both reported:

| read | plots | what it is | incumbent |
| --- | --- | --- | --- |
| `full` | 6,414 | deploy metric; Tessera present for 36% | `baseline_aef` 0.660, `tt_additive_md0.5` 0.6615 |
| `subset` | 2,309 | both-years-covered; the fusion actually fires on every row | `baseline_aef` 0.661, fused 0.679-0.683 |

A win must clear seed noise (~±0.005). Every number is 3+ torch seeds under the
same 5-fold spatially blocked CV. `change_f1_bestt` is tuned on OOF and is an
upper bound, not a claim.

## Three headline metrics, not one (2026-07-26)

Everything above section F was judged on **binary change-F1 alone**, under which
`Artificial -> Artificial` and `Vegetation -> Vegetation` score *identically* —
both are "no change". So the ledger was blind to the failure the deployed maps
actually show. Rescoring the entire OOF cache under per-class metrics
(`rescore_ledger.py`, no refits — the cache made it free) says:

| metric | what it catches | deployed model | spread over all ~45 ideas |
| --- | --- | --- | --- |
| `change_f1` | the historical headline | 0.6704 | 0.632 – 0.6704 |
| `macro_f1` | unweighted mean over the 4 merged2 classes | 0.6993 | 0.660 – 0.6993 |
| `art_stable_recall` | stable built-up found | **0.639** | 0.60 – 0.663 |
| `art_stable_as_veg` | **stable Artificial called stable Vegetation** | **0.220** | **0.195 – 0.253** |

**21.6% of stable-Artificial plots come back as stable Vegetation, and not one of
the ~45 ideas tested moved that number outside a 6-point band.** It is the
largest off-diagonal mass in the confusion matrix (216 of 979 plots) and no
experiment ever aimed at it, because the metric could not see it. That is the
open frontier; change-F1 is close to its label-noise ceiling.

**On the omission report — the two levels disagree, and the disagreement is the
finding.** At *plot* level Tessera does not omit: change recall on the
Tessera-covered rows is 0.763 versus 0.671 where it is absent
(`tess_recall_gap` +0.091, positive for 43 of 45 ideas). At *pixel* level it
plainly does: over 2.95M identical Oslo pixels the deployed model calls 12,791
change with Tessera against 15,229 without, **−16.0%** (G3).

These are not contradictory, because they measure different things. The plot
metric conditions on a label and asks "of the real change, how much was found";
the pixel count is unconditional and cannot distinguish suppressed commission
from suppressed detection. Oslo contains **zero** labelled plots, so the plot
evidence does not transfer there. The honest position: Tessera measurably
shrinks the change class on the map, the labelled evidence available says it
shrinks it in the right direction, and nothing in hand proves that for Oslo.
G4 is what would settle it.

**The suppression is not a calibration artefact (G1, 2026-07-26).** The obvious
hypothesis — better-calibrated probabilities would soften it, since that is
exactly what fixed the stable-Artificial confusion — was **wrong**. Under the
5-seed ensemble Oslo reads 14,753 change without Tessera against 12,282 with:
**−16.8%**, marginally *stronger* than the single seed's −16.0%. Both views
shrink a little, the ratio does not move. So the two phenomena are distinct: the
stable-Artificial confusion was seed-level under-confidence and ensembling fixes
it, while Tessera's change suppression survives the same intervention and is a
property of what the modality says, not of how confidently it says it.

**Working hypothesis to exploit (the user's framing).** AlphaEarth is *context*
(dense, smooth, globally consistent, present everywhere); Tessera is *detail*
(10 m, S1+S2, sharper but noisier and sparse). The fusion should therefore let
context **arbitrate** detail rather than average with it, and should transfer
Tessera's detail into the AlphaEarth tower where Tessera is missing.

## Run it

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python
cd src
$P twotower_lab.py --list
$P twotower_lab.py --ideas <name> --n-seeds 3          # appends to the ledger
$P build_research_artifact.py                          # rebuild the artifact page
```

Ledger: `data/analysis_results/twotower_lab_ledger.csv` (append-only, one row per
idea x read). OOF merged2 probabilities are cached per idea/read/seed in
`data/analysis_results/twotower_lab_oof/` — post-hoc ideas (blending, stacking,
threshold, distillation targets) must reuse those instead of refitting.

## Status legend

`TODO` not started · `RUNNING` in flight · `WIN` beat the incumbent past noise ·
`FLAT` within noise · `NEG` clearly worse · `DROP` abandoned, reason recorded

## Backlog

Ordered by expected value. Update the status and paste the number when done.

### A. Harvest the covered-subset gain onto the full set
The +1.8pt fused gain is real but confined to 36% of plots. These convert it
into deploy-metric value without new Tessera downloads.

| # | idea | status | result |
| --- | --- | --- | --- |
| A1 | **Cross-modal distillation.** Fit the fused two-tower on covered plots, take its soft merged2 probabilities as a teacher, train the AlphaEarth-only student on *all* plots with `KL(student‖teacher)` on covered rows + CE elsewhere. Tessera's detail ends up in the AlphaEarth tower and applies to the 64% with no Tessera. | NEG | Once the soft-target path actually fired (first run silently no-op'd -- teacher columns were passed positionally and reindexed to all-NaN), every variant lost: distil_w1 full 0.6573, distil_w3_T2 0.6448, full-teacher 0.6585, vs AlphaEarth-only 0.6601. The teacher is only ~2pt better than the student on the covered rows and worse elsewhere; there is not enough dark knowledge to pay for the extra loss term. |
| A2 | **Modality hallucination.** Regress Tessera-from-AlphaEarth on covered plots (ridge / small MLP), impute the missing 64%, feed with a `synthetic` flag so the trunk can discount it. (SMIL / missing-modality translation.) | NEG | Ridge-hallucinated Tessera for the uncovered 64% (mask switched on + `tess_synthetic` flag): full 0.6625, subset 0.6841 -- both below plain `tt_tessdrop0.7` (0.6647 / 0.6859). A Tessera vector predicted from AlphaEarth carries no information AlphaEarth did not already have; the tower just re-reads the context tower's input. |
| A3 | **Cross-modal contrastive alignment.** InfoNCE between the two towers' representations on covered rows as an auxiliary loss, so the AlphaEarth tower is pulled toward the Tessera manifold even where Tessera is absent. | NEG | CLIP-style InfoNCE between the towers: weight 0.1 full 0.6630 / subset 0.6874 (flat), weight 0.5 full 0.6459 / subset 0.6675 (clear damage). Aligning the two representations mostly destroys the complementarity that made the fusion worth doing. |
| A4 | **Covered/uncovered specialist + router.** Deploy the fused model on covered plots and the AlphaEarth model elsewhere, i.e. a hard router on the mask; measure the pooled full-set F1 (the "bankable gain" read from memory). | NEG | full 0.6554 (-0.006). The covered-only fused model is a *weaker* predictor even on covered plots than the all-plots model -- routing throws away 64% of the training data. |

### B. Fusion that lets context arbitrate detail
| # | idea | status | result |
| --- | --- | --- | --- |
| B1 | **FiLM conditioning.** AlphaEarth rep produces (γ, β) that modulate the Tessera tower — context tells the model *how* to read the detail. | TODO | |
| B2 | **Learned reliability gate.** Replace the binary mask gate with `g = σ(MLP([rep_aef, rep_tess, mask]))`, a per-plot confidence in Tessera. Noisy Tessera rows get down-weighted instead of trusted equally. | TODO | |
| B3 | **Cross-attention fusion.** One or two cross-attention blocks between the towers (query=aef, key/value=tess and vice versa) before the head. | TODO | |
| B4 | **Uncertainty-weighted logit fusion.** Per-tower heads; fuse logits weighted by inverse predictive entropy (evidential / product-of-experts), instead of fusing representations. | TODO | |
| B5 | **Gradient blending** (Wang et al. 2020, "What makes training multi-modal networks hard"). Weight each tower's auxiliary loss by its overfitting-to-generalisation ratio — directly targets the noisier modality overfitting first. | TODO | |

### C. Treat Tessera as the noisy modality
| # | idea | status | result |
| --- | --- | --- | --- |
| C1 | **Noise-robust loss on the Tessera head only** (GCE / bootstrapped CE / symmetric CE) while the AlphaEarth head keeps focal. | TODO | |
| C2 | **Asymmetric regularisation.** Heavier dropout / weight decay / lower LR on the Tessera tower; sweep the ratio. | **WIN** | **Tessera-tower dropout 0.6 -> subset 0.6914 +/-0.006 (+0.0087 vs tt_symmetric 0.6827); dropout 0.7 -> full 0.6647 +/-0.002 (+0.0053 vs 0.6594, +0.0070 vs AlphaEarth-only).** Clean unimodal dose-response over 0.2/0.4/0.6/0.7/0.8 (subset 0.680/0.683/0.691/0.686/0.680) and the tightest seed variance on the board. The first ARCHITECTURAL lever to move this metric. |
| C3 | **Tessera denoising by projection.** PCA / random projection of the 384 Tessera columns to 32–64 dims before the tower (384 dims on 2.3k covered plots is the overfitting regime). | FLAT | PCA to 32 comps: full 0.6633 +/-0.005, subset 0.6752. Full-set read is up on the baseline but inside noise, and the subset read is down -- projection removes noise and signal together. Costs ~4x the runtime. Keep as a stack member only. |
| C4 | **Per-plot Tessera quality score** from the landmask/scale metadata or neighbourhood variance, fed to the gate in B2. | FLAT | Cross-modal agreement scalars (rank-normalised gap between the two modalities' change magnitudes, carried on the always-present AlphaEarth tower): full 0.6640 +/-0.005, subset 0.6773 -- inside noise of `tt_drop0.7_scalars` (0.6665). The learned gate (B2) already had access to this and could not use it either. |

### D. Features (cheap, prior-supported)
| # | idea | status | result |
| --- | --- | --- | --- |
| D1 | **Cosine-distance scalars for Tessera.** `diff+cos` beat `diff` for AlphaEarth (0.6639 vs 0.6567); Tessera never got the same treatment. Add per-modality cos-dist / L1 / norm-change scalars. | WIN? | **full 0.6638 +/-0.006 (+0.0023, best full-set number so far)**; subset 0.6721 (-0.011). Helps the sparse regime, hurts the dense one -- re-run at 5 seeds to clear noise. |
| D2 | **Tessera 2024-only tower.** 2024 Tessera covers ~99% of plots; a dense single-date detail tower plus the sparse change tower may beat gating everything on the 36% mask. | NEG | tess2024 full 0.6325 / subset 0.6581; 2yr-bands-on-2024-mask full 0.6416 / subset 0.6824. Gating on the dense 2024 mask destroys the change signal -- the 2018->2024 *pair* is the whole contribution. |
| D3 | **Spatial neighbourhood Tessera.** 10 m native means a plot has 3×3 neighbours inside one AlphaEarth pixel — pool the local patch (mean + std) so the detail modality contributes texture, not one pixel. Needs re-extraction. | TODO | |

### E. Operating point and ensembling (post-hoc, near-free)
| # | idea | status | result |
| --- | --- | --- | --- |
| E1 | **Threshold tuning, nested.** Pick the change gate on inner folds only, then report on the outer — an honest version of `change_f1_bestt`. Memory: t≈0.45 was best for F1. | TODO | |
| E2 | **Multi-view TTA.** Average the two-tower's `both` / `aef_only` / `tess_only` OOF probabilities at eval; a free ensemble the mask-gating already supports. | FLAT | full 0.6599, subset 0.6787 -- averaging the mask-gated views is a wash; the single-modality views are strictly worse and drag the mean. |
| E3 | **Stacking.** Meta-learner (LDA / logistic) over cached OOF probabilities of the AlphaEarth model, the Tessera model and the fused model, with the mask as a meta-feature. Extended to a deliberately diverse base set (E3b: + change-scalar, FiLM and learned-gate towers). | WIN | **`gate_stack_wide` full 0.6675 +/-0.007 at 5 seeds = +0.0081 vs the best baseline (tt_symmetric 0.6594) and +0.0098 vs AlphaEarth-only (0.6577) -- clears the noise band.** Narrow 2-base stack + gate 0.6639. Subset 0.6782, still below the plain fused two-tower (0.6827): a deploy-set-only win. |
| E4 | **Modality-dropout ensembling at test time.** Sample several gate maskings per plot and average — MC-dropout over modalities. | **WIN** | **Monte-Carlo modality dropout -- keep the Tessera gate stochastic at test time and average 16 passes. full 0.6673 +/-0.002 alone; 0.6704 +/-0.003 combined with the change scalars (`mc_dropout_scalars`), the best number in the whole search.** +0.0110 vs `tt_symmetric_md0.5` and +0.0127 vs AlphaEarth-only, from ONE model. |

**E3c (2026-07-25).** Extending the stack with the asymmetric-dropout and PCA towers did NOT help: `gate_stack_wide2` 0.6660 vs `gate_stack_wide` 0.6675. Stack diversity was already saturated -- adding *stronger* members is not the same as adding *different* ones, and the strong members correlate with what the stack already had.

**C2b/C2c/D1xC2 (2026-07-25).** Three follow-ups on the asymmetric-dropout win:

- **Capacity reduction is NOT the mechanism.** Narrowing the Tessera tower
  (`tess_width` 0.5 / 0.25, new in `_TwoTowerTrunk`) *hurts*: full 0.6570 / 0.6539
  and subset 0.6793 / 0.6664, versus 0.6647 / 0.6859 for full-width at dropout 0.7.
  The quarter-width control at default dropout is equally bad (0.6570 / 0.6796).
  So it is stochastic unit-dropping -- implicit ensembling / co-adaptation
  breaking -- that pays, not "give the noisy tower less room to memorise".
- **No interaction with modality dropout.** At Tessera-tower dropout 0.7, md
  0.3 / 0.5 / 0.7 gives full 0.6652 / 0.6647 / 0.6657 -- flat. The two dropouts
  regularise different things and do not need joint tuning.
- **The two independent gains compose, mildly.** `tt_drop0.7_scalars` (change
  scalars + asymmetric dropout) = **full 0.6665 +/-0.004 at 5 seeds**, the best
  single model on the deploy read: +0.0071 vs `tt_symmetric_md0.5` (0.6594),
  +0.0088 vs AlphaEarth-only (0.6577), and it effectively ties the five-model
  stack + nested gate (`gate_stack_wide` 0.6675) with one network. The gain over
  plain `tt_tessdrop0.7` (0.6647) is +0.0018, inside noise -- so the scalars add
  a little on the deploy read and cost on the subset (0.6808 vs 0.6859).
- Dropout 0.65 sits between: full 0.6638, subset 0.6888. **0.6 stays the subset
  optimum, 0.7 the deploy optimum.**

### F. The stable-Artificial confusion (the open frontier)
Aimed at `art_stable_recall` / `macro_f1`, not at change-F1. A win here must not
cost more than ~0.005 change-F1, and must not simply flood the Artificial class
— `veg_stable_as_art` is in the ledger as the counter-check.

| # | idea | status | result |
| --- | --- | --- | --- |
| F1 | **State-marginal supervision.** A merged2 label *is* a (state_2018, state_2024) pair, so "Artificial in 2018" is a group-sum of the same softmax — two more 0/1 aggregation matrices, no new head, no new parameters. Pools 1,148 Artificial-in-2018 and 1,695 Artificial-in-2024 plots into one built-up decision each, instead of splitting them across two thin transition classes. (`endpoint_weight` in `HierarchicalSoftmaxNN`.) | FLAT | 3 seeds, deploy read: `tt_endpoint0.3` artStab 0.635, `tt_endpoint1.0` 0.637, `mc_endpoint` 0.638, against the deployed model's 0.639 — no movement on the target at any weight, and `art_stable_as_veg` stays at 0.219-0.229. change-F1 drifts up slightly (`mc_endpoint` 0.6721 vs 0.6704) but inside noise. **The model is not short of Artificial supervision; it cannot separate the classes from these features.** That is the diagnosis F6 should confirm. |
| F2 | **Seed ensembling, properly measured.** Average the deployed model's OOF probabilities across its already-trained torch seeds. Memory says seed-ensembling was a wash — that was the AlphaEarth-only hier NN, not the two-tower. | **WIN** | **5 seeds, deploy read: change-F1 0.6756 ±0.001 (+0.0052), macro-F1 0.7033 (+0.0040), artStab 0.646 (+0.007) — every metric up, none down.** Leave-one-seed-out spread is ±0.001, the tightest on the board. Free at inference: the seeds are already trained, only the forward pass repeats. Subset 0.6728. |
| F3 | **Cost-sensitive nested operating point.** Arg-max is the Bayes rule for accuracy, not for macro-F1 over a 979-plot class and a 4,550-plot one. Per-class multipliers chosen on inner folds, applied to the outer. | **WIN** | **5 seeds, deploy read: artStab 0.674 (+0.035 over the deployed 0.639), `art_stable_as_veg` 0.200 (from 0.220), macro-F1 0.6996 (flat), change-F1 0.6692 (−0.0012, inside noise).** The first thing in ~45 ideas to move stable-Artificial past the 0.60–0.663 band the whole search sat in, and it costs nothing on the headline. Subset even better: artStab 0.674, art→veg 0.187. Note the plain (un-nested) prior correction does *not* do this — τ=0.3 gives artStab 0.668 but drops change-F1 to 0.644; choosing the multiplier on inner folds is what makes it free. |
| F4 | **Dense 2024 Tessera as a state feature, not a change feature.** D2 killed the 2024-only *change* tower, but stable-Artificial is a *state* question and 10 m S1+S2 is exactly what built-up detection wants. Feed the 99%-covered 2024 Tessera on its own gate, supervising only the `to`-state marginal (F1), while the change path stays on the sparse both-years pair. | TODO | |
| F5 | **Per-class focal alpha at the merged2 level.** The loss already supports `cb_focal`; it has never been tried with the level weights re-balanced toward merged2, where the Artificial classes live. Cheap, and the incumbent's `level_weights` have never been swept at all. | **PARTLY ANSWERED — see `SIAMESE_RESEARCH.md` section W** | `cb_focal` run on `siam_s2off_cos` at 5 seeds. **As a rare-class lever it is negative** — the whole `focus_macro_f1` gain is breaking `Artificial -> Cropland`, which O3 and V1 already break for free and better, and it costs −0.006 change-F1 and −0.047 change precision. **The result is on stable built-up instead:** `art_stable_as_veg` 0.225 → 0.151, beating the deployed model's 0.196 and the gated 0.165, which four sections could not. New `cb_levels="fine"` in `model_zoo.py` separates the mechanism: the **merged2/gate** weights do the built-up work, the **fine** weights break the dead class. `level_weights` itself is still unswept. |
| F6 | **Where do the 215 plots live?** Diagnostic, not a model: are the stable-Artificial-read-as-Vegetation plots spatially clustered, low-density built-up, or a specific interpreter? If they are one AOI or one land-cover flavour, the fix is data, not architecture — the same conclusion that closed the Cropland/Nature boundary. (`diagnose_stable_artificial.py`) | **ANSWERED** | **They carry an Artificial label on a Vegetation pixel — 62.3% of the 215 sit closer to the stable-Vegetation centroid than to their own class's, in AlphaEarth 2024 space.** Correctly-classified stable-Artificial plots are at cosine 0.463 from the Artificial centroid and 0.682 from the Vegetation one (10.5% closer to Veg); the errors are at 0.679 / **0.645** — *nearer the Vegetation centroid than stable-Vegetation plots themselves are* (0.663). Supporting evidence, all pointing the same way: the model is **not** borderline (median margin 0.318, only 14.4% within 0.1 of flipping, mean P(stable Artificial) on the errors 0.217); all 215 are unambiguously interpreted `Artificial` in 2024, so it is not a legend artefact; no interpreter campaign is to blame (recover 23.6%, habloss_main 22.4%, habloss_landwater 15.6%); Tessera barely helps (20.4% misread where present vs 23.3% where absent); errors concentrate spatially (Gini 0.570; worst blocks 45–59% against 13–14% elsewhere) and land more often in **low-built-up-density blocks** (27.6% vs 18.9%, Spearman −0.24 — directional, not decisive on its own). Written to `data/analysis_results/stable_artificial_errors.csv` for inspection in QGIS. |
| F7 | **Compose F2 and F3.** The cost gate is chosen on probabilities; the seed ensemble produces better-calibrated ones. Run the nested macro-F1 cost search on the seed-ensembled OOF instead of per-seed, and check whether +0.035 artStab and +0.005 change-F1 add or overlap. | FLAT | **They overlap — the composition lands between its two parts, not above them.** `costgate_ensemble` deploy read: change-F1 0.6732, macro-F1 0.7016, artStab 0.654, against F2 alone (0.6756 / 0.7033 / 0.646) and F3 alone (0.6692 / 0.6996 / **0.674**). Mechanism, checked directly: on single-seed probabilities the nested search picks a 1.2x Artificial multiplier in **all five** folds; on seed-ensembled probabilities it picks **1.0 in four of five**. Pooling the seeds already raises mean P(A→A) on true stable-Artificial rows from 0.491 to 0.503, which is most of what the cost gate was applying. **Both levers are correcting the same under-confidence on the minority stable class.** Pick one by goal, do not stack them. |
| F8 | **Widen the cost search.** F3 tunes one multiplier on one class. Search all four merged2 classes (coordinate ascent on the inner folds) and see whether macro-F1 has more to give than the single-class correction found. Cheap — post-hoc on cached probabilities. | NEG | **Worse on every metric: `costgate_wide` 0.6650 change-F1 / 0.6965 macro / 0.669 artStab against `costgate_macro`'s 0.6692 / 0.6996 / 0.674, and double the seed spread (±0.004 vs ±0.002).** The mechanism is exactly the one the idea risked: measured on the same per-seed probabilities, the wide search reaches inner-fold macro-F1 0.7022 against the narrow search's 0.7006 — it *does* fit the training folds better — but its held-out macro-F1 is 0.6896 against 0.6929. The generalisation gap goes +0.0077 → +0.0127. The chosen cost vector also stops being stable: the narrow search moves only the Artificial slot (spread 0.071 across folds, zero elsewhere), the wide one moves all four (0.128 / 0.109 / 0.100 / 0.061). **Four free parameters fitted on four folds is more than 6,414 plots support.** One class was the right amount of freedom. |

**Section F is closed (2026-07-26).** Six ideas, two wins (F2, F3), one flat
(F7), two negative (F1, F8), one decisive diagnostic (F6). Everything left is
deployment — section G.

**F6 closes section F's modelling half (2026-07-26).** The residual
stable-Artificial confusion is a *label-versus-pixel* disagreement, not a model
deficiency: these plots are built-up by the interpretation protocol and vegetated
by the 10 m reflectance — sparse rural settlement, a road through fields, a plot
where the built fraction is real but sub-dominant. A classifier reading that
pixel cannot win them, which is structurally the same finding that closed the
Cropland/Nature boundary and produced the merged legend. **Stop looking for an
architecture.** What is left is worth doing and is all operating-point or
deployment: F8, then G1–G3. If more accuracy on built-up is genuinely needed, the
lever is a built-fraction covariate or a sub-pixel label, not another fusion.

### G. Deployment-side

**Section G is closed except G4 (2026-07-26).** G1 shipped, G2 done as far as the
data allows, G3 blocked as specified and redesigned into a within-AOI
counterfactual that now runs on every map. **G4 is a data ask and cannot be
executed from here** — it needs interpreted plots, not code. The autoresearch
loop stops at this point: every remaining question is answered by labelling, not
by modelling.

| # | idea | status | result |
| --- | --- | --- | --- |
| G1 | **Ship the seed ensemble.** If F2 holds, the inference path should average the trained seeds rather than run one. | **DONE** | `infer_best.py --seeds N` (default 5) fits N models and averages their posteriors at every read — deterministic, AlphaEarth-only counterfactual, and each of the 16 MC passes. `--seeds 1` is the old behaviour exactly. Oslo under 5 seeds: change 12,282 det / 13,021 MC against the single seed's 12,791 / 13,511. **The two maps agree on 99.93% of pixels, but that is ~2,000 pixels against a change class of ~13,000 — the disagreement is concentrated where it matters (net −490 change px on the MC read).** Cost is forward passes only; the rasters are fetched once regardless. |
| G2 | **Re-map Oslo and Johannesburg under the best model** and recount the change fraction, so the "Tessera omits" report is checked on the artefact the user actually looked at rather than only on plots. | PART | Oslo re-mapped under the deployed recipe with the new counterfactual read (`data/inference/omission_check/best_20260726_151935`). Johannesburg not re-run: its 2018 Tessera coverage is 0%, so `both` and `aef_only` are the same model there by construction and the ablation is empty. |
| G3 | **Pixel-level omission check.** Plot-level says Tessera helps recall; the maps suggest otherwise. Reconcile by scoring the deployed rasters against the plots that fall inside each AOI. | **BLOCKED → REDESIGNED** | **As specified it is impossible: zero validation plots fall inside either AOI bbox** (21 and 29 within ±2°). Replaced with a within-AOI counterfactual — the same model on the same pixels with the Tessera gate forced off (`merged_aef` in `infer_best.py`, one extra forward pass beside the 16 MC passes). **Oslo, 2,954,952 identical pixels: 15,229 change without Tessera → 12,791 with, −16.0%. 5,032 pixels turned off by Tessera, 2,594 turned on.** The user's report is real at pixel level and *larger for the deployed model* than for the 24-July symmetric two-tower (−8.2%: 10,643 → 9,768). MC recovers a little of it (13,511 vs 12,791 deterministic). **But with no labelled plots in Oslo there is no way to say whether those 5,032 suppressed pixels were commission errors correctly removed or real change wrongly omitted** — the plot-level evidence, where labels do exist, says Tessera *raises* change recall (0.763 vs 0.671), which argues for commission removal, but no plot in that sample is anywhere near Oslo. See G4. |
| G4 | **Get labelled plots inside a Tessera-covered AOI.** This is the one thing that would settle whether the −16% is a fix or a fault, and no amount of modelling substitutes for it. Fifty interpreted plots inside the Oslo bbox — stratified on the pixels where the two views disagree (`oslo_aefonly_change.tif` XOR `oslo_det_change.tif`, 7,626 px) — would decide it directly. | TODO | The highest-value open item in the whole backlog, and it is a data ask, not a modelling one. |

## Current recommendation

Two post-hoc reads of the *already deployed* network, neither needing a retrain:

| goal | change on top of `mc_dropout_scalars` | change-F1 | macro-F1 | stable-Artificial recall |
| --- | --- | --- | --- | --- |
| deployed today | — | 0.6704 | 0.6993 | 0.639 |
| **best headline** | **F2 seed ensemble** | **0.6756** | **0.7033** | 0.646 |
| **best on the map** | **F3 nested cost gate** | 0.6692 | 0.6996 | **0.674** |

F2 costs 5 forward passes on an AOI that already maps in minutes. F3 costs
nothing at all — it is a different arg-max over the same probabilities.
**Do not stack them:** F7 showed the two correct the same under-confidence on the
minority stable class, and the composition lands between its parts (0.6732 /
0.7016 / 0.654). Choose by what the output is for — the map, or the headline.

| goal | model | 5-seed change-F1 | vs AlphaEarth-only (0.6577) |
| --- | --- | --- | --- |
| **deploy set** | **`mc_dropout_scalars`** | **0.6704 +/-0.003** | **+0.0127** |
| deploy set, single pass | `tt_drop0.7_scalars` | 0.6665 +/-0.004 | +0.0088 |
| deploy set, ensemble | `gate_stack_wide` | 0.6675 +/-0.007 | +0.0098 |
| **covered plots only** | **`tt_tessdrop0.6`** | **0.6914 +/-0.006** | +0.0087 vs the fused incumbent |

`mc_dropout_scalars` = the symmetric two-tower with (a) Tessera-tower dropout 0.7
against AlphaEarth's 0.4, (b) per-modality change scalars on both towers, and
(c) the modality gate left stochastic at inference, averaged over 16 passes.
Three independent gains, one network, no meta-learner. Note it is *worse* on the
covered subset (0.6702) -- the scalars trade subset accuracy for deploy accuracy,
so keep `tt_tessdrop0.6` if you only ever score covered plots.

## What actually moved the metric, and what did not

**Moved it:** asymmetric regularisation of the noisy modality (C2), keeping the
modality gate stochastic at test time (E4), per-modality change scalars (D1), and
blending diverse variants at an honestly chosen operating point (E1+E3).

**Did not:** every attempt to make the fusion *smarter* -- FiLM conditioning,
learned reliability gates, cross-modal distillation, contrastive alignment,
modality hallucination, cross-modal agreement features, capacity reduction,
routing, dense-2024 gating, TTA over mask views. The pattern across all of them:
the two modalities' complementarity is already captured by "add a gated tower";
what was left on the table was **how hard to regularise the noisy one and how to
average over the uncertainty in trusting it**, not a richer interaction between
them.

## Tested-negative (do not redo — from memory)

Minibatch/G-H sampling · MoE trunk · noise injection · mixup · early stopping ·
seed ensembling · multi-year GRU trajectory · FT-Transformer · bilinear head ·
forward loss correction · FixMatch SSL · confident-learning cleaning ·
per-band dot / normalised-difference features · flat concat of Tessera (Plan A,
−2.1pt) · two-tower without modality dropout (−0.7pt). The label-noise ceiling
on the Cropland/Nature boundary is why capacity and optimiser tricks do nothing;
the merged Veg/Artificial legend already absorbs it.
