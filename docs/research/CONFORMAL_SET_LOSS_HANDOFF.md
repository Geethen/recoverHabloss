# Handoff — the set-restricted loss (section S, proposed)

One hypothesis, one auxiliary loss, no new head, no serving cost. Read
`AUTORESEARCH.md` first: **3 seeds minimum before any verdict, 5 before calling a
win**, sub-1pt differences at 1 seed are noise. Read section R at the foot of
`SIAMESE_RESEARCH.md` — this idea comes out of it and its measurements are the
baselines below.

## The hypothesis

Add a cross-entropy term over the fine softmax **renormalised to the conformal
prediction set**: *given the answer is one of these k classes, be right*. It
reweights **which pairs of logits get pushed apart**, and that axis is untested —
every long-tail lever tried on this model reweights *samples* (focal, cb_focal,
G-H sampling) or *class priors* (cRT, tau-normalisation), and all are negative.

Why it should have somewhere to go, measured (5 seeds, `full`,
`base_siam_s2off_cos_fine`):

| level | arg-max acc | 90% set coverage | **headroom** | P(truth in set \| arg-max wrong) |
| --- | --- | --- | --- | --- |
| merged2 (4 cls) | 0.8497 | 0.8997 | +0.050 | 0.463 |
| coarse3 (9 cls) | 0.6872 | 0.9002 | **+0.213** | 0.682 |

**Apply it at merged2 and coarse3, never the gate.** With two classes the
restricted CE is either plain CE or a constant.

**It has to be a training change.** The decode-time version is already tested and
negative: masking the coarse3 softmax to the classes under the merged2 conformal
set and re-arg-maxing gives accuracy 0.6837 vs 0.6872 and `focus_macro_f1` 0.3827
vs 0.3847. The merged2 set vetoes the coarse3 arg-max on 2.3% of rows and is as
often wrong as right there. The headroom is not reachable by re-ranking existing
scores.

## Where the code goes

* **`model_zoo.py:1835` `HierarchicalSoftmaxNN._levels`** — the single hook. It
  already composes the three level losses plus `endpoint_weight` and
  `dice_weight`; add the term the same way, behind `set_ce_weight > 0.0`.
  `p_merged = p_fine @ self._M` and `p_gate = p_merged @ self._G` are the fixed
  0/1 group-sums (`self._M`, `self._G` built at `model_zoo.py:1961`). **There is
  only one head** — the coarse3 logits — so do not add a merged2 classifier; the
  mutual consistency of the three levels depends on that.
* New `__init__` kwargs beside `dice_weight` (`model_zoo.py:1095`):
  `set_ce_weight: float = 0.0`, `set_ce_level: str = "fine"` (`fine` / `merged` /
  `both`), `set_ce_alpha: float = 0.10`.
* Pass-through: `twotower_lab.siam_s2off_kwargs` (`twotower_lab.py:1696`) already
  forwards `**overrides` to the model, so registering ideas needs no plumbing.

## Implementation spec

Per epoch, per level:

1. **Split the training rows in half** on a seeded permutation, alternating
   halves each epoch (`epoch % 2`). ConfTr's construction — the set has to come
   from thresholds the scored rows did not calibrate, or the term is trivially
   satisfied.
2. On the calibration half, per class `k`, compute the LAC quantile of the true
   class scores: `q_k = ceil((n_k+1)(1-alpha))`-th smallest of `1 - p[i, k]` over
   rows with `y_i = k`. **Mondrian (per-class), not pooled** — a pooled cut gives
   `Cropland -> Nature` 0.005 coverage against Mondrian's 0.902, and the pooled
   version cannot produce a subset worth restricting to.
3. On the scored half, `S_i = {k : 1 - p[i,k] <= min(q_k, 1)}`. **Clamp infinite
   quantiles to 1** — a class too rare to calibrate at this alpha otherwise
   dominates every set (bites `Artificial -> Cropland`, 46 plots, at
   `alpha <= 0.02`).
4. Force the true class into the set (`S_i |= {y_i}`) and skip rows with
   `|S_i| < 2` — a singleton restricted CE is zero and contributes no gradient.
5. Loss: `-log( p[i, y_i] / sum_{k in S_i} p[i, k] )`, mean over kept rows.
   Detach nothing. Add `set_ce_weight *` that to the return of `_levels`.

Guards: the quantile is a hard order statistic, so **compute it under
`torch.no_grad()` and treat the set as a constant mask** — a straight-through
quantile is a different (and much slower) experiment. Weight grid `0.3` first
(`SIAM_AUX`, the section's preregistered regulariser strength) and `1.0`
registered separately, the N2/N2b pattern.

## Ideas to register

In `twotower_lab.py`, using `_siam_s2off_idea` (`twotower_lab.py:1724`) so the
base is `siam_s2off_cos` — the model section R re-reads and the section's best.

| name | kwargs | what it is |
| --- | --- | --- |
| `siam_setce_fine` | `set_ce_weight=0.3, set_ce_level="fine"` | S1: the main arm, coarse3 where the headroom is |
| `siam_setce_fine_strong` | `set_ce_weight=1.0` | S1b: strength, registered up front not reached for |
| `siam_setce_both` | `set_ce_level="both"` | S1c: merged2 + coarse3 |
| `siam_setce_rand` | random size-matched sets | **S2: the control.** Sets of the same sizes drawn at random, true class forced in. If this reproduces S1, the finding is "extra contrastive gradient on the tail", not calibration — the design that inverted N14b, P3 and Q8 |

The control is not optional. Register it before reading S1.

## Run

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python   # the .venv is broken; uv run fails
cd src
$P twotower_lab.py --ideas siam_setce_fine --read full --n-seeds 3
$P twotower_lab.py --ideas siam_setce_rand --read full --n-seeds 3
# only if 3 seeds are not flat:
$P twotower_lab.py --ideas siam_setce_fine,siam_setce_fine_strong,siam_setce_both,siam_setce_rand \
      --read full --n-seeds 5
$P conformal_report.py --sources siam_setce_fine --n-seeds 5   # did the sets shrink?
```

Costs GPU time — roughly one siamese ladder rung per seed. Everything in section
R was free; this is not.

## Baselines to beat (15 seeds, `full`)

| | change-F1 | macro-F1 | artStab | focus macro |
| --- | --- | --- | --- | --- |
| `siam_s2off_cos` (base) | 0.6604 | 0.7035 | 0.6441 | 0.3820 |
| `conf_siam_cos_nested` (R1e, free post-hoc) | 0.6609 | 0.7058 | 0.6799 | — |
| `conf_crfe` (R1g, free, best cell) | 0.6626 | 0.7050 | 0.7062 | — |
| `c3gate_siam_s2off_cos` (O3b, 5 seeds) | 0.6644 | 0.7067 | 0.6458 | **0.4318** |

**A training change must beat the free post-hoc reads, not the raw arg-max.**
R1e/R1g cost nothing and already move stable built-up +0.036/+0.062; an idea that
needs 30 epochs to match them has lost.

## Preregistered counter-checks

1. **The majority classes.** R5f raised `focus_macro_f1` to 0.4235 while
   `Nature -> Nature` fell 0.7627 -> 0.6844 and `Cropland -> Cropland`
   0.7249 -> 0.6442, invisible to that metric. Read the per-class coarse3 table
   (the R5 analysis in `SIAMESE_RESEARCH.md` has the recipe) beside every row.
2. **Precision/recall, not just change-F1.** R1e's flat change-F1 hides
   precision 0.651 -> 0.614 / recall 0.670 -> 0.715.
3. **Did the sets actually shrink?** The mechanism's own signature: mean coarse3
   set size at alpha 0.10 Mondrian is 3.354 on the base. If accuracy moves and
   set size does not, something else is doing the work.

## The standing objection — record the answer either way

The subsets the conformal sets identify as confusable are the boundaries this
project already calls label noise: `{Nature->Cropland, Nature->Nature}` is the
largest size-2 set (302/seed) and `analyse_label_noise.py` calls that boundary
the noisiest in the legend; at merged2 it is `{Art->Art, Veg->Art}`, and
`stable-artificial-is-a-label-problem` records that 62% of misread built-up plots
sit closer to the Vegetation centroid than their own. The learning curves put the
model at +0.026 change-F1 per doubling of labels. **A plausible outcome is that
this loss fits interpreter disagreement**, and that outcome is a result worth
writing down, not a failed run — it would be direct evidence for labelling over
modelling on the two boundaries that matter.

Also: only 5.2% of coarse3 errors are recoverable inside a set of size <= 2 (the
modal error row admits 4-5 of 9), so **do not** reach for a pairwise specialist
head when this comes back flat. That has no target.

## Do not redo

The foot of `TWOTOWER_RESEARCH.md` and the tested-negative lists in sections N-R.
Relevant here: MoE, cRT, prototype head, tau-normalisation, G-H class-balanced
sampling, deep supervision, soft-Dice (gate and coarse3 macro), distillation,
endpoint supervision, noise injection — and from section R, the multiplicative
read of conformal cuts, APS as a point read, and coverage shortfall as a
nominator for the focus set.
