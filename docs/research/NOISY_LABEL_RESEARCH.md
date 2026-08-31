# Learning with noisy labels — section T

Every section before this one takes the labels as given and asks what to fit
them with. This one takes the ledger's own most repeated conclusion seriously —
that the ceiling is interpreter disagreement on the Cropland/Nature boundary
(`analyse_label_noise.py`, N0, P6, R7, `cropland-nature-label-noise`) — and asks
whether a training procedure that *expects some labels to be wrong* beats one
that does not.

The methods split on what each has to be **told** about the noise, and that axis
is the reason the section is shaped the way it is:

| needs | methods | runnable here? |
| --- | --- | --- |
| nothing | stochastic co-teaching, GCE, SCE, bootstrapping, ELR | yes |
| a noise **rate** | classic co-teaching's forget rate | only as a contrast |
| a noise **matrix** | forward loss correction | **already tested-negative** |

This project has **no clean validation set**, and its one direct measurement of
interpreter disagreement — the 54 RECOVER reverifications — is change-enriched by
construction and cannot be read as a population noise rate. So a method that
needs no estimate is the only kind deployable here honestly, which is exactly the
property that makes stochastic co-teaching the arm this section is built around.

Forward loss correction, confident-learning cleaning and mixup are all on the
tested-negative list at the foot of `TWOTOWER_RESEARCH.md` and none of the three
is re-run.

**Base:** `siam_s2off_cos` (N8b) — section R and S's base, the ledger's best
aggregate model, Sentinel-2 privileged at training and never read at inference.
**Only network A is served** under every co-teaching arm, so no row can bank a
two-model ensemble as if it were noise robustness.

## Standing table (5 seeds, `full` read, identical folds)

| | change-F1 | change P / R | change-F1 @ tuned t | macro-F1 | focus macro | artStab | as_veg | vegStab→art |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `siam_s2off_cos` (base) | **0.6644** ±0.0024 | 0.651 / 0.679 | **0.6694** | **0.7067** | **0.3847** | 0.6458 | 0.2249 | 0.0292 |
| `siam_cotrand10` (T3, control) | 0.6657 ±0.0032 | 0.651 / 0.681 | 0.6693 | 0.7058 | 0.3853 | 0.6407 | 0.2327 | 0.0281 |
| `siam_elr_aux` (T5b) | 0.6635 ±0.0039 | 0.653 / 0.674 | 0.6706 | 0.7068 | 0.3838 | 0.6537 | 0.2221 | 0.0301 |
| `siam_cotrand10_strat` (T6b, control) | 0.6627 ±0.0029 | 0.654 / 0.672 | 0.6688 | 0.7054 | 0.3805 | 0.6472 | 0.2280 | 0.0290 |
| `siam_coteach10_strat` (T6) | 0.6544 ±0.0024 | 0.615 / 0.699 | 0.6610 | 0.7025 | 0.3759 | **0.7224** | **0.1444** | **0.0505** |

At 3 seeds, the rest of the field:

| | change-F1 | macro-F1 | focus macro | artStab |
| --- | --- | --- | --- | --- |
| base | 0.6648 | 0.7081 | 0.3887 | 0.650 |
| `siam_sct` (T1) | **0.6055** | 0.6281 | **0.2203** | 0.784 |
| `siam_sct_gate` (T1b) | 0.6347 | 0.6559 | 0.3608 | 0.659 |
| `siam_sct_merged` (T1c) | 0.6254 | 0.6274 | 0.3614 | 0.590 |
| `siam_coteach10` (T2) | 0.6161 | 0.5720 | 0.2721 | 0.661 |
| `siam_coteach20` (T2b) | 0.6239 | 0.5706 | 0.2735 | 0.703 |
| `siam_ce` (T4ref) | 0.6349 | 0.6668 | 0.3643 | 0.696 |
| `siam_gce` (T4) | 0.6167 | 0.5734 | 0.1766 | 0.715 |
| `siam_gce_fine` (T4b) | 0.6240 | 0.6581 | 0.3048 | 0.719 |
| `siam_sce` (T4c) | 0.6184 | 0.6053 | 0.2115 | 0.736 |
| `siam_boot_soft` (T4d) | 0.6354 | 0.6670 | 0.3642 | 0.696 |
| `siam_elr` (T5, λ=3) | 0.6401 | 0.6935 | 0.3634 | 0.725 |
| `siam_coteach30_strat` (T6c) | 0.6463 | 0.6899 | 0.3662 | 0.697 |

**Nothing in the section wins. The control is the only arm that is free.**

## T1 — stochastic co-teaching: the largest single regression in the ledger

Bertels et al. (2023, *Sci Rep* 13:16875). Two networks, independently
initialised; each keeps the rows whose given-label posterior clears a threshold
drawn from `Beta(32, 2)` and trains its partner on them, with the threshold
ramped in from zero over epochs 10–20. A draw rather than a rank, so there is no
forget rate and nothing has to be estimated — the property that makes it the one
selection method runnable on this project at all.

**Verdict: strongly negative. −0.059 change-F1 and −0.168 focus macro** against
the base at 3 seeds, the largest single-idea regression recorded anywhere in this
ledger. Reading the selector at the gate level (T1b) or merged2 level (T1c)
recovers about half of it and is still −0.030 / −0.039.

The guard never fired (0.0% of selections at every level), so the `Beta(32, 2)`
prior is not mismatched to this head's confidence in the way T1b was registered
to check — the ramp keeps the effective threshold low enough for long enough. The
mechanism is elsewhere, and `coteach_diagnostics.py` finds it.

## T2 — and classic co-teaching fails the same way, which is the finding

**Verdict: negative at both forget rates** (−0.049 / −0.041 change-F1, macro-F1
down 0.14). Note that the *larger* forget rate is the better of the two, which is
already a sign that the rate is not the thing that matters.

## The diagnostic — a small-loss criterion ranks rarity, not noise

`coteach_diagnostics.py` keeps each fold's `coteach_keep_counts_` and reports how
often each **class** survived selection. This is the section's central result and
it is not a metric, it is a mechanism:

| keep rate by class | `siam_coteach10` (τ=0.10) | `siam_sct` |
| --- | --- | --- |
| Artificial → Cropland (46) | **0.242** | 0.393 |
| Artificial → Nature (123) | **0.250** | 0.398 |
| Cropland → Nature (114) | 0.432 | **0.369** |
| Nature → Cropland (243) | 0.953 | 0.388 |
| Nature → Artificial (383) | 0.855 | 0.452 |
| Artificial → Artificial (979) | 0.798 | 0.734 |
| Cropland → Cropland (1661) | 0.981 | 0.527 |
| Nature → Nature (2532) | **0.987** | **0.691** |
| *overall* | 0.915 | 0.611 |

**A 10% forget budget is spent almost entirely on the commissioned transitions.**
Classic co-teaching drops three quarters of the 46-plot `Artificial -> Cropland`
steps while keeping 98.7% of `Nature -> Nature`. A small-loss (or
high-posterior) criterion cannot distinguish "this label is wrong" from "the
model has not learned this class yet", and on a target of 4,200 stable plots
against 46 in its rarest transition, the second dominates completely. Every
metric movement in T1 and T2 follows from this one table.

This is the same failure Mondrian conformal fixes in R7: a **pooled** cut
over-covers the majority class and never covers the rare ones (`Cropland ->
Nature` at 0.005 pooled coverage against 0.902 per-class). The instruments are
different; the arithmetic is identical.

## T6 — the fix, and the section's most interesting negative

Apply the same forget rate **within each class**. The keep rate then flattens to
0.915 ±0.002 across all nine classes, so the rarity bias is gone by
construction, and any remaining movement is attributable to the loss ranking
inside a class — which is where mislabels would live if the method worked.

**Verdict: negative, and its own control says so.** 5 seeds: `siam_coteach10_strat`
0.6544 against its matched random control's 0.6627 and the base's 0.6644. The
informed selector is −0.0083 against the identically-subsampled random one, which
is 3σ of its own seed spread — small-loss selection inside a class is **worse
than dropping the same rows at random**. T6c raises the rate to 0.30 and it gets
worse again (0.6463), so this is monotone in how much selection is done, not a
tuning miss.

**The one row that looks like a win is disqualified by the preregistered
counter-check.** `siam_coteach10_strat` moves stable built-up recall 0.646 →
0.722 and `art_stable_as_veg` 0.225 → 0.144 — the largest movement on the metric
`AUTORESEARCH.md` calls the open frontier that any *training* change in this
ledger has produced. But `veg_stable_as_art` goes 0.0292 → 0.0505 and change
precision 0.651 → 0.615. It is not fixing the built-up read; it is moving the
Vegetation/Artificial boundary, which the counter-check exists to catch, and it
pays for a 979-plot class out of a 4,200-plot one. Against the free post-hoc
conformal read (R1g: artStab 0.7062 at change-F1 0.6626, zero training cost) it
is dominated on every axis that matters.

## T4/T5 — bounded losses and ELR: negative, and they all fail in one direction

Read against `siam_ce` (T4ref), not against the focal base: a bounded core
replaces cross-entropy and drops the focal modulation with it, so comparing GCE
to a focal baseline would confound two changes.

* **T4ref is itself a result.** Plain CE is −0.030 change-F1 against focal.
  Focal loss is theoretically the *least* noise-robust choice available —
  `(1-p)^gamma` upweights exactly the rows a mislabel produces — and it is worth
  +0.030 here. Whatever is limiting this model, it is not that the objective
  attends too much to hard examples.
* **GCE** (q=0.7) is −0.018 against CE at all three levels and −0.011 applied to
  the coarse3 level only, where the measured noise is. **SCE** is −0.017.
  **Soft bootstrapping** at β=0.95 is exactly flat (0.6354 vs 0.6349) — it does
  nothing, which is what mixing 5% of the model's own posterior into the target
  should do.
* **ELR** at the paper's λ=3 is −0.025; at the project's auxiliary weight 0.3 it
  is flat (0.6635 ±0.0039 vs 0.6644 ±0.0024 at 5 seeds).

**They all fail in the same direction and it is the ledger's oldest pattern.**
Change precision goes up and change recall goes down, every time: GCE 0.702/0.550
and SCE 0.722/0.541 against the base's 0.651/0.679. Bounding the loss on a rare
positive class is a change-suppressor, and `spatial-smoothing-eats-change`
already records that everything which removes change pixels does it to the same
rows first. Noise robustness is a third instrument with that signature.

**And the whole field collapses at a matched operating point.** Column
`change_f1_bestt` — change-F1 at each arm's own tuned gate threshold — reads
0.6694 (base), 0.6693, 0.6706, 0.6688, 0.6610. Once the change threshold is
retuned, nothing here is distinguishable from the base except T6, which is
worse. **Most of what these methods appear to do is move the change gate**, and
the ledger has a free instrument for that (E1, and the conformal reads in R1e).

## T7 — the falsification test, and it comes back empty

The only external evidence about label noise this project holds: 54 RECOVER
plots interpreted twice, independently (`analyse_label_noise.py`). If a selector
is finding mislabels, the plots whose two reads **disagree** should survive
selection less often than the plots whose reads agree. If it is finding
difficulty, there is no reason for the two groups to differ.

Within-class standardised keep score, disagreed minus agreed (negative = the
selector rejects the disagreed plots, which is the prediction):

| arm | n agree / disagree | Δ keep-z | Mann-Whitney p |
| --- | --- | --- | --- |
| `siam_sct` | 19 / 35 | **−0.225** | 0.227 |
| `siam_coteach10_strat` | 19 / 35 | +0.090 | 0.726 |

**No signal.** The stochastic selector has the right sign and does not reach
significance at n = 54; the stratified selector has the wrong sign. Weak test —
the reverified subset is change-enriched, so the two groups differ in class
composition as well as in agreement, and class composition is what the selector
responds to most — but it is the only external check available and it does not
support the mechanism.

**So the relabelling queue is a by-product to be read sceptically, not a
deliverable.** `coteach_plot_queue__<arm>.csv` ranks every plot by within-class
keep frequency and is written by `coteach_diagnostics.py`; on this evidence its
bottom is a list of hard plots, not of wrong ones, and nothing here justifies
spending interpreter time on it in that order.

## Status

**Section T is closed and it is negative throughout.** Nine mechanisms across
three families, every one of them at or below the base, the two controls flat,
the mechanism diagnosed, and the one external falsification test empty.

What it establishes, which is more than the arms:

1. **Sample selection cannot be used on this target as published.** Not because
   the noise is absent but because rarity and mislabelling are the same signal to
   a loss ranking here, and correcting for that (T6) still loses to dropping rows
   at random. Any future selection idea must show its per-class keep table
   before its metric.
2. **The ledger's focal choice is not costing noise robustness**; it is worth
   +0.030 over plain CE, and every bounded alternative to it suppresses change.
3. **Label noise remains the diagnosis and relabelling remains the treatment.**
   Nothing in this section reaches it from the training side. That is consistent
   with `learning-curves-say-label-more` (+0.026 change-F1 per doubling) and with
   R7's finding that the dead classes have signal the arg-max cannot use — both
   of which point at more labels or a different read, not at a loss.

**Tested-negative, do not redo:** stochastic co-teaching at any of the three
posterior levels (T1/T1b/T1c) · classic co-teaching at forget rate 0.10 / 0.20,
pooled (T2/T2b) or 0.10 / 0.30 stratified per class (T6/T6c) · GCE at all levels
or at coarse3 only (T4/T4b) · SCE (T4c) · soft bootstrapping (T4d) · ELR at λ=3
or λ=0.3 (T5/T5b) · plain CE in place of focal (T4ref). And from earlier
sections, still standing: forward loss correction, confident-learning cleaning,
mixup, early stopping, noise injection.

**Do not** reach for DivideMix, co-teaching+, JoCoR or any other selection-based
method next. All of them are refinements of *how* rows are chosen, and T6
establishes that the choosing itself is what loses here — it is beaten by chance
at the same rate, inside the class, at 5 seeds.

## Reproduce

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python   # the .venv is broken; uv run fails
cd src
$P twotower_lab.py --group section-t --read full --n-seeds 3
$P coteach_diagnostics.py --idea siam_coteach10 --n-seeds 1   # the keep-rate table
$P coteach_diagnostics.py --idea siam_sct --n-seeds 3         # + the reverification check
```

Implementation: `robust_loss` / `elr_weight` / `coteach*` on
`model_zoo.HierarchicalSoftmaxNN`; the selector is `_coteach_keep`, the peer
network is `_make_peer`, and `level_loss(..., reduce=False)` /
`_levels(..., per_sample=True)` are the per-row reads the selection needs. The
`_build_network` extraction that makes a peer possible was verified
**bit-identical** on `siam_s2off_cos` seed 0 against its cached OOF before any
section-T arm was run.
