# State pretraining — the phase's own dataset and architecture

A ledger, not documentation. Same rules as everything in `docs/research/`
(`AUTORESEARCH.md`): **3 seeds minimum before a verdict, 5 before a win**, and a
difference under 0.005 on the headline read is noise. Negative results stay in.

Code: [`src/statepre/`](../../src/statepre/). Ledger:
`data/analysis_results/statepre_ledger.csv` (append-only).

## What this section is for

`SIAMESE_RESEARCH.md` section P7 established that single-date state labels help
this project when they are spent as a **phase** — `siam_state_pretrain` epochs of
`g(f(x)) -> state` over a pool, before the transition fit — rather than as a
joint auxiliary term, and the user's map read preferred the result. Everything
downstream of "it helps" was untested. The phase trains on **13,118 GLanCE units
at one year**, on whatever encoder the transition model happens to carry, and its
own quality has never been measured except through a transition metric 30 epochs
later.

Two axes, and a validation protocol that has to come first.

* **Dataset** — which single-date rows the phase is fed. The lead hypothesis is
  the user's: a plot that is *stable* across 2018–2024 holds its state at every
  intermediate year, so its 2019–2023 AlphaEarth embeddings are five more free
  state labels each, drawn from real inter-annual variation. No new data —
  `embeddings_habloss_recover_annual.parquet` has held all seven years since
  2026-07-22 and nothing in the project has ever read the middle five.
* **Architecture** — what is trained on them, from the project's linear probe up
  to the exact `64 -> 512 -> 256 -> 128` encoder `model_zoo._pretrain_state`
  runs, plus the year-conditioning and year-invariance variants that only become
  askable once a pool has more than one year in it.

## U0 — LLTO, and why no number here is readable without it

> **Amended by V0.** Every number in U0–U3 was cut on folds the arm under test
> was allowed to move, because the k-means ran over the training pool and the
> test pool together. The 5-fold results are barely affected (99.6% of test plots
> keep their fold when GLanCE is added) and U3's one recommendation re-runs
> *stronger* under the fix; the 10-fold ones are the weaker read, and separately
> were being deduplicated rather than pooled. See V0.


Year-augmenting a plot creates seven near-duplicate rows of one location.
Consecutive AlphaEarth years of unchanged ground are very nearly the same vector,
so under any split that does not hold a location out **at every year**, an
augmented arm is scored on rows whose near-duplicates it trained on.

That was the reason this protocol was built, and the measurement below **does not
support the form the worry took**: the leak does not grow with augmentation, and
it was already fully present in the unaugmented baseline. The protocol is still
required — the numbers move by 0.045 and the arm *ordering* reverses — but for a
different reason than "the augmented arms cheat more". Recorded this way round
because the prediction was wrong and the correction is the useful part.

LLTO (leave-location-time-out, the protocol the user specified) cuts folds on
**location** so a held-out plot is held out at all seven years, and *places*
those folds by k-means over the coordinates so the neighbouring ground goes with
it. `statepre/llto.py` enforces the first as a hard assertion inside every fold.
Two implementation notes: the k-means runs on **unit-sphere xyz, not (lon, lat)**
— the plots span lon −167..178 and Euclidean k-means on degrees makes Alaska and
Kamchatka the two furthest points on Earth — and **external pool rows are folded
too**, so a GLanCE unit inside the held-out region is removed with the RECOVER
plots there.

The test set is fixed and separate from the arm under test: RECOVER's own
**observed** 2024 rows, 6,414 plots. That is what lets an arm containing no
RECOVER rows at all (`glance`) be scored against one that is nothing but RECOVER
rows.

**The size of the trap, measured.** Linear probe, 5 seeds, macro-F1 on the 2024
read. `random` splits rows; `loc` cuts folds on location but places them at
random; `block` uses the project's existing 20° blocks; `llto` is the protocol.

| training pool | rows/loc | `random` | `loc` | `block` | **`llto`** | random→llto |
| --- | --- | --- | --- | --- | --- | --- |
| `endpoints` | 2 | 0.7651 | 0.7582 | 0.7441 | **0.7203** | −0.045 |
| `stable_years` | up to 7 | 0.7647 | 0.7579 | 0.7465 | **0.7207** | −0.044 |
| `glance` | 1 | 0.7341 | 0.7345 | 0.7297 | **0.7286** | −0.006 |

**The near-duplicate leak is real but small; the spatial leak is the large one.**
Removing the same-location rows (`random` → `loc`) costs 0.007 on the endogenous
pools. *Placing* the folds geographically (`loc` → `llto`) costs a further 0.038.
And it does not get worse with augmentation — `endpoints` loses 0.045 across the
ladder and `stable_years` 0.044, because at two rows per location the leak is
already saturated: for the 81% of plots that are stable, a plot's 2018 vector is
already a near-duplicate of its 2024 one, and five more copies leak nothing new.

`glance` is the control that identifies what the 0.045 is. Its units share no
location with the test plots, and it barely moves across the whole ladder
(−0.006). So the penalty the endogenous pools pay is not generic split hardness:
it is **RECOVER's plots being spatially clustered with each other**, and every
split looser than LLTO lets a test plot be answered by its own neighbours.

**Consequence, and it is the reason this section exists.** `llto` is the *only*
split under which the external pool beats both endogenous ones — 0.7286 against
0.7203 and 0.7207. Under `random`, `loc` and `block` the ordering is reversed.
Every existing number in this project's ledgers is blocked on `block`, which sits
0.024 above LLTO here.

**Read the seed spread with care — it is not an error bar.** k-means over 6,414
globally-spread plots returns the *same* five continental folds at every seed and
only renumbers them: N.America 853, S.America 777, Africa 809, Europe/W-Asia
1,765, Asia/Oceania 2,210. LLTO seed sd is therefore ±0.0001 on the probe, and it
measures model init, not "would this hold in a region we did not sample". The
between-fold spread is ~16 points and swamps every arm effect in this document.
Arms must be compared **paired**, same seed and same fold — `statepre_folds.csv`
and `statepre.run --paired`. Fold ids are canonicalised west-to-east so a fold id
means one region across seeds (`llto._canonical`); rows written to the fold
ledger before 2026-07-31 carry the raw k-means numbering and cannot be grouped by
fold across seeds.

## U1 — the dataset arms

All LLTO, 5 seeds, macro-F1. `linear` is the project probe (comparable to P0's
numbers); `mlp` is `_pretrain_state`'s own encoder and is the read that decides.

| arm | linear `t1_all` | **mlp `t1_all`** | mlp `t1_changed` | mlp `t1_stable` |
| --- | --- | --- | --- | --- |
| **`glance_endpoints`** | 0.7383 | **0.7519** | 0.6037 | **0.7634** |
| `glance_stable_years` | 0.7324 | 0.7393 | 0.5702 | 0.7541 |
| `endpoints` (baseline) | 0.7203 | 0.7376 | **0.6054** | 0.7455 |
| `glance` (what P7 trains on) | 0.7286 | 0.7349 | 0.5776 | 0.7447 |
| `all_years_pseudo` | 0.7216 | 0.7253 | 0.5714 | 0.7359 |
| `stable_years_jit` (control) | 0.7190 | 0.7223 | 0.5948 | 0.7289 |
| **`stable_years`** (the hypothesis) | 0.7207 | 0.7229 | 0.5673 | 0.7357 |
| `stable_years_dup` (control) | 0.7112 | 0.7181 | 0.5857 | 0.7268 |

### U1a — the year augmentation is negative on the encoder that matters

`stable_years` − `endpoints` on `mlp` is **−0.0148** seed-averaged and **−0.0206
paired, negative in 21 of 25 folds** (V0 corrects this from −0.0239; the fold
count is unchanged). That is three times the ±0.005 noise band
and it is not a fold accident. On `t1_changed` it is −0.038.

The controls say what the augmentation is made of, and the two models disagree in
an informative way.

* On the **probe**, the real years beat duplicated 2018 vectors by +0.0095 but
  beat *jittered* ones by only +0.0017. Gaussian noise at the measured
  inter-annual spread recovers nearly all of the gain, so most of what the extra
  years buy a linear model is input-noise regularisation, not temporal structure.
* On the **encoder**, `stable_years` (0.7229), `_jit` (0.7223) and `_dup`
  (0.7181) are one cluster, all ~0.015 below `endpoints` (0.7376). A 512-unit network with
  dropout 0.4 does not need the extra noise, so nothing is left but the cost.

That cost is the **vote distortion**: seven rows for a stable plot against two for
a changed one shifts the pool toward stability. `stable_years` gains on
`t1_stable` and loses on `t1_changed` — it helps exactly where its rows come from
and hurts where they do not.

### U1b — fixing the vote distortion helps, and is not enough

`--weights per_loc` gives every location the same total weight regardless of how
many years it contributes. It is a verified no-op on `endpoints` (0.7203→0.7206
probe, 0.7376→0.7390 encoder — inside the ±0.002 run-to-run drift), which is the
check that it is measuring what it claims: a two-row location already votes
uniformly, so there is nothing there for the scheme to change.

| pool | linear `none` | linear `per_loc` | mlp `none` | mlp `per_loc` |
| --- | --- | --- | --- | --- |
| `endpoints` | 0.7203 | 0.7206 | 0.7376 | 0.7390 |
| `stable_years` | 0.7207 | **0.7244** | 0.7229 | **0.7296** |

Worth +0.004 on the probe and +0.007 on the encoder — so roughly half the damage
*was* the weighting, and there is a genuine temporal signal underneath it. But
0.7296 is still 0.008 below `endpoints`. **The hypothesis is right about the
mechanism and wrong about the net.**

### U1c — the union of the two pools is the win

`glance_endpoints` — GLanCE's 13,118 units plus RECOVER's own 12,828 endpoint
rows — is the best arm on both models and on the aggregate and stable reads.
Against `glance`, which is what the deployed phase actually trains on today, it
is **+0.0161 seed-averaged and +0.0234 paired, positive in 24 of 25 folds**.
Against `endpoints` it is the weaker half of the claim — see U3, where it turns
out to depend on how fine the holdout is — so the robust statement is **adding
the endpoints to GLanCE**, not adding GLanCE to the endpoints.

Adding the year augmentation on top of it *removes* the gain
(`glance_stable_years` 0.7393, −0.012), which is U1a a second time on a different
base.

### U1d — pseudo-labelling the changed plots' middle years is negative

`all_years_pseudo` (the user's Common Ground bridge: a stage-1 model fitted
inside each fold fills the changed plots' 2019–2023) scores 0.7253 on the
encoder, below `endpoints` at 0.7376 and above `stable_years` at 0.7229. It
recovers part of what `stable_years` lost on `t1_changed` (0.5714 vs 0.5673) by
putting changed plots back into the pool, but it does not reach the baseline. Not
pursued further.

## U2 — the architecture arms, and the form the hypothesis works in

LLTO, 5 seeds, macro-F1 on `t1_all`. The `delta` column is the year augmentation's
effect **on that architecture**, and it is the point of the table.

| arch | `endpoints` | `stable_years` | delta | s/run |
| --- | --- | --- | --- | --- |
| **`mlp_yeardrop`** | 0.7432 | **0.7456** | **+0.0024** | 3.3 |
| `mlp_inv` | 0.7392 | 0.7230 | −0.0162 | 6.5 |
| `mlp` (the deployed encoder) | 0.7376 | 0.7229 | −0.0148 | 4.1 |
| `mlp_year` | 0.7381 | 0.7249 | −0.0132 | 4.8 |
| `linear` | 0.7203 | 0.7207 | +0.0003 | 8.1 |
| `rf` | 0.6489 | 0.6495 | +0.0006 | 18.8 |

### U2a — the hypothesis is right, in the form of a sampler and not of rows

`mlp_yeardrop` draws **one row per location per epoch, year chosen uniformly**.
Same locations per epoch as `endpoints`, same total variety as `stable_years`
across epochs. It is the only architecture on which the year augmentation is
positive, and the pair `stable_years` + `mlp_yeardrop` at **0.7456** beats the
deployed encoder on the baseline pool (`endpoints` + `mlp`, 0.7376) by **+0.0080**
— while running *faster*, 3.3 s against 4.1 s, because an epoch is 6,414 rows
instead of 38,688.

This is U1b taken to its limit and it closes the U1 story. The augmentation's
harm was never the year vectors; it was handing a stable plot seven votes.
`per_loc` weighting fixed half of that and recovered +0.008; sampling fixes all of
it and turns −0.018 into +0.002. **"Train on each year of stable data" is a
sampling instruction, not a concatenation instruction.**

**The paired read is what identifies the sampler as a fix rather than a
regulariser, and it corrects the obvious reading of the table above.** Seed-
averaged, `mlp_yeardrop` also gains +0.0046 on `endpoints` alone, which invites
"part of this is generic single-date regularisation". It is not: that gain is
**12/25 folds**, a coin flip, and it does not survive pairing. What does survive
is `yeardrop` on the *augmented* pool — **+0.0310 over `mlp`, 23 of 25 folds**.

So the sampler does nothing where there is no vote distortion to undo, and almost
everything where there is. That is the mechanism claim, and it is the strongest
paired result in this section.

What it does **not** establish is the headline: `stable_years` + `mlp_yeardrop`
against the plain `endpoints` + `mlp` baseline is +0.0071 at **16/25 folds**. The
recipe climbs out of the hole the augmentation dug and arrives roughly back at
the baseline. Reported as promising and unproven, not as a win.

### U2b — explicit year conditioning and explicit invariance both fail

`mlp_year` (a per-year diagonal affine on the input, `model_zoo`'s
`year_adapter="input"` generalised to seven years) is −0.0009 on `endpoints` and
does not rescue `stable_years` (−0.0132). `mlp_inv` (pull same-location
embeddings together, weight 0.1) is +0.0002 and −0.0162. Neither is worth its
parameters. **Telling the encoder about the year, and telling it to ignore the
year, are both worse than letting the sampler decide which year it sees.**

### U2c — random forests are not usable under a spatial holdout here

`rf` scores 0.6489 against the linear probe's 0.7203 — **7 points below a
64-feature logistic regression**, and it is also the slowest arm. A forest fits
the training regions' local structure and none of it transfers across a
continental fold boundary. Noted because the forest is the natural first choice
and is the model in the sketch this section started from; on this data under LLTO
it is the wrong one, and the gap is large enough that a forest-based read of any
arm above would have ranked them differently.

## U3 — what survives a paired read, and what does not

The seed-averaged tables above are not the verdict; the five continental folds
are, and only paired. Every row here is arm A minus arm B on the **same seed and
the same fold**, pooling the 5-fold and 10-fold runs where both exist.

| comparison | mean | folds positive |
| --- | --- | --- |
| `stable_years`/`yeardrop` − `stable_years`/`mlp` | **+0.0310** | **23/25** |
| `glance_stable_years`/`yeardrop` − `glance`/`mlp` | **+0.0271** | **21/25** |
| `glance_endpoints` − `glance` (both `mlp`) | **+0.0234 / +0.0231** | **24/25 and 24/30** |
| `glance_endpoints` − `endpoints` (both `mlp`) | **+0.0156 / −0.0001** | **22/25 but 12/30** |
| `stable_years`/`yeardrop` − `endpoints`/`mlp` | +0.0071 | 16/25 |
| `glance_stable_years`/`yeardrop` − `glance_endpoints`/`mlp` | +0.0056 | 16/25 |
| `endpoints`/`yeardrop` − `endpoints`/`mlp` | +0.0046 | 12/25 |
| `mlp_year` − `mlp` (on `stable_years`) | +0.0035 | 13/25 |
| `mlp_inv` − `mlp` (on `stable_years`) | +0.0015 | 16/25 |
| `stable_years` − `endpoints` (both `mlp`) | **−0.0206** | 4/25 |

**The two rows carrying a `x / y` pair are the 5-fold and the 10-fold run kept
apart, and they were previously pooled by mistake** — the fold ledger had no
`n_folds` column, so a 10-fold run's folds 0–4 deduplicated onto a 5-fold run's
and the 5-fold rows were dropped rather than added (V0). Separating them changes
nothing about the recommendation and sharpens the second row: `glance_endpoints`
− `endpoints` is not the coin flip the pooled "+0.0033 at 19/40" suggested, it is
**fold-count-dependent** — clearly positive when 1/5 of the world is held out,
exactly zero at 1/10, and +0.0021 (47/100) at 1/20 under V0's geometry. The
endogenous pool's advantage is bought from RECOVER's own spatial clustering and
it shrinks as the holdout gets finer, which is U0's lesson arriving a second
time. `stable_years` − `endpoints` moves −0.0239 → −0.0206 at the same 4/25 and
is unaffected in substance.

The four rows above the fold-count line are the only ones that clear a paired
read. Everything from `stable_years`/`yeardrop` − `endpoints`/`mlp` downward sits
between 12/25 and 16/25 — coin flips, regardless of how the seed-averaged means
rank them.

**One claim is robust and it is not the one this section set out to test.** The
pretraining pool should contain **RECOVER's own endpoint rows alongside GLanCE**:
+0.0234 paired at 24/25 folds and +0.0231 at 24/30, reproducing at both fold
counts and again at 20 folds under V0's fixed geometry (+0.0250, 84/100).

**Its converse is weaker and is bought from the holdout's coarseness.**
`glance_endpoints` − `endpoints` is +0.0156 (22/25) at 5 folds, −0.0001 (12/30)
at 10 and +0.0021 (47/100) at 20. So the defensible sentence is still *"add the
endpoints to GLanCE"*, never *"add GLanCE to the endpoints"* — but the reason is
now specific: at a fine holdout the endogenous rows already answer the test
plots' neighbourhoods and the external pool has nothing left to add.

**The best cell in the sweep is `glance_stable_years` + `mlp_yeardrop` at
0.7542**, +0.019 over what the phase trains on today and +0.0271 paired at 21/25.
But its margin over the simple `glance_endpoints` + `mlp` recipe is +0.0056 at
16/25 folds. Both beat the deployed pool; neither beats the other.
**Take the pool change, which is one keyword; treat the sampler as a mechanism
that is established and a win that is not.**

**A note on run-to-run drift.** Re-running an identical cell moves it by ~±0.002
(`endpoints`/`mlp` came back 0.7390 then 0.7376). That is CUDA non-determinism at
fixed seed, not a bug, and it is inside the ±0.005 band — but it means a
sub-0.005 difference in this document is not reproducible even at the same seed,
which is a second reason the paired fold read is the one to trust.

## V0 — the folds have to be cut by the protocol, not by the arm

**Everything in U0–U3 was scored on folds that each arm was allowed to move.**
`location_folds` ran k-means over the *union* of the training pool and the test
pool, so an arm carrying a geographically concentrated pool pulled the cluster
centres toward it and got a different partition of the world than the arm it was
being paired against. Since the between-fold spread is ~16 points and the arm
effects are ~1, a "paired" difference across two partitions is not a paired
difference at all.

Measured, as the fraction of RECOVER test plots that keep their fold id when a
pool is added:

| comparison | 5 folds | 20 folds |
| --- | --- | --- |
| `glance_endpoints` vs `endpoints` | 0.996 | 0.84 / 0.20 (seed 0 / 1) |
| `glance_lucas_endpoints` vs `glance_endpoints` | **0.603** | **0.29 / 0.11** |

`llto.py` now cuts folds on a **fixed reference cloud** — the RECOVER plots, who
are the test set — and assigns every pool row to the fold it falls in
(`fold_ref="reference"`, the default; `"union"` restores the old rule, and every
ledger row written before 2026-07-31 carries it). The same argument applied to
the other two splits: `block` packs folds on the reference's block sizes, and
`loc` now *hashes* a location to a fold instead of drawing one, because a draw
over the arm's own row order was arm-dependent for the same reason.

**U3's one robust claim survives the fix and strengthens.** `glance_endpoints` −
`glance`, re-run at 5 seeds under the fixed geometry: **+0.0204 at 22/25 folds**
(5-fold) and **+0.0250 at 84/100** (20-fold), against U3's +0.0225 at 34/40. The
recommendation below is unaffected. What does not survive is anything that was
resting on the **10-fold** rows: the fold ledger had no `n_folds` column, so
`paired` deduplicated a 10-fold run's folds 0–4 onto a 5-fold run's and
*discarded* the 5-fold rows rather than pooling them, which is not what U3 says
it did. Both ledgers have been migrated (`--migrate`) and `append` now reconciles
against the header on disk, because a headerless append of a wider row silently
shifts every later value one column left.

## V1 — LUCAS, and why a dense regional pool has to be read regionally

LUCAS is 12,360 in-situ field points from the ELC10 reference set. It is the same
*size* as GLanCE's 13,118 and nothing like it in reach: **all of it is EU-27**, 8
of the project's 83 blocks, against GLanCE's 83.

That geometry dictates the protocol. **At 5 folds LUCAS falls entirely inside
one fold**, so holding that fold out deletes the whole pool and `lucas` alone is
scored on four folds against everyone else's five — the aggregate answers
nothing. At **20 folds** LUCAS spans 3–4 folds per seed, so a European fold can
be held out with LUCAS still supplying its other European rows. Every number
below is 20 folds, `mlp`, 5 seeds, fixed geometry, and is read **twice**: on the
folds LUCAS reaches (`in_lucas`, ~16 of 100) and on the folds it does not
(`elsewhere`) — because those are two different experiments and their aggregate
answers neither.

| arm | rows | `t1_all` | `t1_changed` | `t1_stable` |
| --- | --- | --- | --- | --- |
| **`glance_lucas_endpoints`** | 38,306 | **0.7645** | 0.6159 | **0.7742** |
| `lucas_endpoints` | 25,188 | 0.7613 | **0.6280** | 0.7681 |
| `glance_endpoints` (U3's winner) | 25,946 | 0.7613 | 0.6018 | 0.7725 |
| `glance_eudup_endpoints` (control) | 38,306 | 0.7603 | 0.6099 | 0.7708 |
| `endpoints` | 12,828 | 0.7585 | 0.6159 | 0.7662 |
| `glance_lucas` | 25,478 | 0.7386 | 0.5645 | 0.7559 |
| `glance` | 13,118 | 0.7386 | 0.5513 | 0.7564 |
| `lucas` | 12,360 | **0.5641** | 0.4266 | 0.5826 |

### V1a — LUCAS is negative, and it fails hardest exactly where it lives

| comparison | all folds | `elsewhere` | **`in_lucas`** |
| --- | --- | --- | --- |
| `glance_lucas_endpoints` − `glance_endpoints` | +0.0031, 66/100 | +0.0036, 57/84 | **+0.0004, 9/16** |
| `lucas_endpoints` − `endpoints` | +0.0040, 58/100 | +0.0064, 52/84 | **−0.0085, 6/16** |
| `glance_lucas` − `glance` | −0.0004, 49/100 | +0.0000, 41/84 | −0.0025, 8/16 |

Read the middle column against the right one. The seed-averaged +0.0031 is
inside the ±0.005 band already, but the **regional split is what closes it**: a
dense in-situ pool that is worth adding has to be worth adding *in its own
region*, and in its own region LUCAS is a coin flip on the union
(+0.0004, 9/16) and a **loss** on the endogenous pool (−0.0085, 6/16). Whatever
small aggregate movement there is comes from folds LUCAS is nowhere near, which
is not a mechanism anyone would have predicted and is not one to bank on.

**`glance_lucas` − `glance` is the cleanest sentence in the section: −0.0004 at
49/100 folds.** Twelve thousand field-surveyed points, added to a pool of
thirteen thousand, change nothing at all.

### V1b — the mechanism is the legend, and it is measured

`lucas` alone scores 0.5641 where `glance` alone scores 0.7386. The confusion
says why, and it is not noise — it is a boundary drawn somewhere else. Pooled
out-of-fold over 5 seeds, 32,070 scored rows, row-normalised
(`statepre.run --confusion`):

```
lucas          accuracy=0.6089  macro_f1=0.5642
truth \ pred    artificial    cropland      nature    recall
artificial      4,033  48%    222   3%   4,220  50%   0.476
cropland          556   6%  2,797  29%   6,397  66%   0.287
nature            700   5%    448   3%  12,697  92%   0.917

glance         accuracy=0.7367  macro_f1=0.7386
truth \ pred    artificial    cropland      nature    recall
artificial      6,293  74%  1,206  14%     976  12%   0.743
cropland          383   4%  7,876  81%   1,491  15%   0.808
nature          1,437  10%  2,952  21%   9,456  68%   0.683

endpoints      accuracy=0.7572  macro_f1=0.7585   (the self-floor)
truth \ pred    artificial    cropland      nature    recall
artificial      6,861  81%    465   5%   1,149  14%   0.810
cropland          613   6%  7,223  74%   1,914  20%   0.741
nature          1,648  12%  1,999  14%  10,198  74%   0.737
```

**The whole matrix collapses into one column.** LUCAS puts 92% of Nature, 66% of
Cropland and 50% of Artificial into Nature — it is not confused between two
classes, it is answering "Nature" to most of the world. Its *precision* on
Cropland is 0.807 and on Artificial 0.763, both higher than GLanCE's, so what it
does call Cropland is Cropland; there is simply four times too little of it. That
is the signature of a coarser boundary, not of noisy labels, and it is what
section N13a's asymmetry test predicts a pool drawn to a different legend looks
like.

**LUCAS returns two-thirds of RECOVER's Cropland and half of its Artificial as
Nature.** Section N14a saw the plot-level version of this and could not separate
it from geography; the LLTO read separates them, because `in_lucas` holds
geography fixed and the disagreement is *worse* there. And it propagates: adding
LUCAS moves the union pool's `cropland_as_nature` from 0.1715 to 0.1797 — the
wrong way, on the exact boundary this project's change-F1 ceiling sits on.

### V1c — it is not a volume effect

`glance_eudup_endpoints` resamples GLanCE's own rows inside LUCAS's eight blocks
up to LUCAS's 12,360 — identical row count, identical location count, identical
blocks, no label LUCAS's absence would have denied it. It scores 0.7603, i.e.
**−0.0013 (47/100) against the plain `glance_endpoints` baseline and −0.0074 at
2/16 on LUCAS's own folds**. So the volume by itself is mildly harmful, and
LUCAS's +0.0044 over this control is a gain over something already below
baseline. The confound is ruled out in the direction that does not rescue LUCAS.

Worth keeping for its own sake: **resampling a region's existing labels to
over-represent it hurts that region, 2/16 folds positive.** It is `stable_years`'
vote distortion again, in space rather than in time.

## V2 — the geographic vote distortion is real as a mechanism and empty as a lever

U1's whole lesson was that a pool's *composition* is a lever independent of its
content. `_row_weights` now carries that to the other two axes a pool can be
lopsided on: `per_source` equalises the total weight of `recover`/`glance`/
`lucas`, and `per_cell` equalises it across the 20-degree blocks — the spatial
analogue of `per_loc`, and the one aimed at a pool that puts 12,360 rows into 8
blocks where another puts 13,118 into 83.

| arm | `none` | `per_cell` | `per_source` |
| --- | --- | --- | --- |
| `endpoints` | 0.7585 | 0.7581 | 0.7585 |
| `glance_endpoints` | 0.7613 | 0.7624 | 0.7613 |
| `glance_lucas_endpoints` | 0.7645 | 0.7623 | 0.7639 |

The controls pass — `per_source` is an exact no-op on the single-source
`endpoints`, as it must be, and `per_cell` is within drift there — and then
nothing happens. Re-reading the LUCAS gain under each scheme: `per_cell` shrinks
it to +0.0002 (51/100), `per_source` to +0.0025 (62/100). **Neither rescues the
regional read, which is the confirmation that V1's problem is the legend and not
the geography.** You cannot reweight your way out of a pool that disagrees with
you about what Cropland is.

Both schemes are now normalised to mean weight 1 rather than left as raw
`1/count`. For the MLP this changes nothing (its batch loss is already a weighted
mean); for the linear probe it does, so U1b's `per_loc` *probe* numbers are under
the unnormalised version. It also makes `per_loc` on `endpoints` an exactly
verifiable no-op instead of an approximate one.

## V3 — the class-balanced head trades the aggregate for the frontier, 200/200

`mlp_cw` folds class-balanced weights into the row weight (so it composes with
`per_cell`/`per_source` rather than overriding them). The pools are unbalanced by
construction and `artificial` is the state the project's open frontier is made
of. Paired against `mlp`, over both union pools, 20 folds, 5 seeds:

| metric | `mlp_cw` − `mlp` | folds positive |
| --- | --- | --- |
| `macro_f1` | −0.0052 | 70/200 |
| `f1_artificial` | −0.0041 | 74/200 |
| **`artificial_as_nature`** | **−0.0363** | **0/200** |
| **`cropland_as_nature`** | **−0.0254** | **0/200** |

Negative on the read this section is scored on, and the most consistent effect in
the whole document on two others: it cuts the rate at which built-up is returned
as Nature by **27%**, in every one of 200 folds, and Cropland-as-Nature by 15%,
likewise in every fold.

That matters because `artificial_as_nature` is the state-level form of the
project's stated open frontier — `art_stable_recall`, "22.0% of stable built-up
is returned as stable Vegetation, and no idea in ~45 has moved it"
(`AUTORESEARCH.md`). This is a *state-level* read on a *pretraining* phase and
`SIAMESE_RESEARCH.md` section W's lesson applies at full strength: a plot-level
gain need not survive
the map's 0.5% base rate, and this one is not even a gain on the headline. It is
recorded as the one thing here worth a transition-level run.



`model_zoo` **already implements the winning pool** and no idea in the ledger has
ever used it. `_pretrain_state` branches on `siam_state_source`; P7a/P7e ran
`"external"` (GLanCE alone) and P7b ran `"endogenous"` (the endpoints alone), and
`"both"` — which concatenates exactly the `glance_endpoints` arm — was never
registered. One idea, no new code in `model_zoo`:

```python
state_pretrain_idea(                                                      # P7i
    "siam_s2off_state_pre_both", base="s2off", source="both",
    desc="P7i: the pretrain pool is GLanCE AND the plots' own endpoints, not "
         "either alone. U1c/U3 measure that union at +0.0225 paired over GLanCE "
         "alone on an LLTO state read, 34/40 folds, where P7a's external-only "
         "pool and P7b's endogenous-only control are the two halves it beats.")
```

> ### RUN, AND NEGATIVE — do not act on the recommendation above
>
> P7i was run on the deployed base at 5 seeds on 2026-08-03
> (`SIAMESE_RESEARCH.md`, section P7i/P7k). **The union pool is the worst arm in
> that section on change-F1: 0.6450 against 0.6644 with no pretraining at all**,
> −0.019 and four times the ±0.005 band. The endogenous half is what costs it,
> monotonically — none 0.6644, GLanCE-only 0.6643, endpoints-only 0.6560, both
> 0.6450 — and it is change *recall* that goes (0.679 → 0.650), not precision.
>
> `siam_state_source="external"` is the option to use. It is free on change-F1
> and still takes macro-F1, `art_stable_recall` and `Artificial -> Cropland`
> (0.000 → 0.108) with it.
>
> **The +0.0225 was real and it did not transfer.** Every state-level number in
> sections U and V survives re-checking under V0's fixed geometry and none of
> them predicted this. `SIAMESE_RESEARCH.md` section W's lesson, third
> occurrence: a plot-level gain on
> an auxiliary objective ranks candidates for a transition run; it does not
> conclude one.

## X — hcropland30, a global cropland map as a pool (2026-08-03)

Section V closed the dataset axis on the evidence available then, and named the
mechanism that closed it: LUCAS is negative **because** it returns 66% of
RECOVER's Cropland as Nature, on the Cropland/Nature boundary the project's
change-F1 ceiling sits on. `data/hcropland30` is the obvious follow-up — a
100,000-point sample of a **hybrid 30 m global cropland map at 2020**, pointed
at exactly that class — and it is not LUCAS: it reaches **60 of the 83 blocks**,
so it does not have to be read regionally.

Three things about the source govern how it can be used at all.

* **`type` is binary and only the positives are usable.** `type == 0` means "not
  cropland", which in coarse3 is `{artificial, nature}` *plus* the water, ice and
  barren the legend has no home for and `build_state_labels` already drops. That
  is not a state and not even a set-valued one, so the **67,657 negatives are
  discarded** and only the 32,343 cropland points survive. A single-state pool
  cannot be trained alone, so there is no `hcropland` arm — it is only ever
  concatenated, and every comparison below is against the pool it was added to.
* **`uncertaint` is a six-map vote.** It is the sample SD of six contributing
  maps' binary votes: 0.0 when all six agree, then 0.408 / 0.516 / 0.548 for one,
  two and three dissenters. `strict` keeps the unanimous 11,411; `all` keeps all
  32,343. This is the closest thing a map product offers to an interpreted label
  and it turns out to be the only thing in the section that matters.
* **It is another model's decision boundary.** The same concern this file already
  records against GLanCE's GLC30 and MapBiomas components. Extracted at **2020**,
  the year the map is valid for — the long frame has always carried a `year`
  column and the phase's encoder is single-date, so a 2020 row is as usable as a
  2018 one.

`mlp`, 5 seeds, `llto`, `fold_ref="reference"`, `weight="none"`, read at both
fold counts because they disagree:

| arm | rows | `t1_all` @5 | `t1_all` @20 | `f1_cropland` @20 |
| --- | --- | --- | --- | --- |
| `glance_hcrop_endpoints` | 37,357 | 0.7466 | **0.7625** | 0.7496 |
| `glance_cropdup_endpoints` (control) | 37,357 | 0.7473 | 0.7611 | 0.7486 |
| `glance_endpoints` (U3's winner) | 25,946 | **0.7508** | 0.7613 | 0.7483 |
| `hcrop_endpoints` | 24,239 | 0.7310 | 0.7583 | 0.7455 |
| `endpoints` | 12,828 | 0.7376 | 0.7585 | 0.7432 |
| `glance_hcropall_endpoints` | 58,289 | 0.7336 | 0.7408 | 0.7266 |

### X1 — the pool is its own control, at both geometries

`glance_cropdup_endpoints` resamples **GLanCE's own cropland** up to hcropland's
11,411 — identical row count, identical class balance, no label GLanCE did not
already have. It exists because 11,411 cropland rows take the pool from 38%
cropland to 63% before anything is learned, and V3 established that class balance
alone moves this read.

| comparison | @5 folds | @20 folds |
| --- | --- | --- |
| `glance_hcrop_endpoints` − `glance_endpoints` | −0.0036, 8/25 | +0.0022, 51/100 |
| **`glance_hcrop_endpoints` − `glance_cropdup_endpoints`** | **−0.0002, 14/25** | **+0.0022, 52/100** |
| `glance_cropdup_endpoints` − `glance_endpoints` | −0.0034, 10/25 | −0.0000, 55/100 |
| `hcrop_endpoints` − `endpoints` | −0.0039, 9/25 | +0.0007, 53/100 |

**Read the second row and stop.** At 5 folds the pool is −0.0002 from a control
carrying no new information; at 20 folds it is +0.0022 at 52/100 folds, a coin
flip. Whatever small movement the arm shows against `glance_endpoints` is
reproduced by duplicating rows the pool already had. **Thirty-two thousand
cropland labels from a global map are worth nothing this pool did not have** —
the same sentence V1a had to write about LUCAS's twelve thousand field points,
reached by a different route.

### X2 — the sign is a property of the holdout radius, not of the seed

The two geometries disagree, and not noisily. Per-seed means of
`glance_hcrop_endpoints` − `glance_endpoints`:

```
 5 folds:  -0.0031  -0.0022  -0.0009  -0.0039  -0.0078     all five negative
20 folds:  +0.0015  +0.0033  +0.0006  +0.0019  +0.0035     all five positive
```

Ten runs, ten agreements with their own fold count and none across it. A 5-fold
LLTO holds out a fifth of the globe — one continental cluster — and a 20-fold one
holds out a twentieth; the pool is mildly harmful under the far extrapolation and
inert under the near one. **This is the cleanest example in the ledger of a
verdict that is a property of the protocol.** Neither number is outside the
±0.005 band, so nothing here is a win under either reading; what the pair rules
out is quoting either one alone.

It is also a caution about `--confusion`. The 5-fold matrices show
`hcrop_endpoints` taking cropland recall 0.700 → 0.659 against `endpoints` while
raising precision 0.718 → 0.737 — a tidy "narrower cropland concept" story, and a
false one. **At 20 folds it does not replicate**: recall 0.741 → 0.746, precision
0.746 → 0.745. `cropland_as_nature` flips the same way (+0.0074 paired at 5
folds, −0.0064 at 20). A mechanism read off one fold count is a hypothesis, not a
diagnosis.

### X3 — the unanimity filter is the one result that replicates

`glance_hcropall_endpoints` drops the six-map agreement filter and takes the pool
from 11,411 to 32,343:

| comparison | @5 folds | @20 folds |
| --- | --- | --- |
| `glance_hcropall_endpoints` − `glance_hcrop_endpoints` | **−0.0133, 1/25** | **−0.0228, 13/100** |

Both geometries, same sign, one-sided fold counts, and by far the largest effect
in the section — larger than any arm's difference from any baseline. **The 20,932
non-unanimous cropland points are actively harmful**, and the damage is on
cropland itself (`f1_cropland` −0.0232, 16/100). That is the map-product
signature stated plainly: where the contributing maps disagree, the label is
worse than no label. Any future use of a map product as a pool should carry an
agreement filter and should expect to lose a third to two-thirds of the points to
it.

### Section X verdict

**Negative, and it does not re-open section V.** The dataset axis stays closed.
hcropland30 is not a LUCAS-style failure — it does no visible damage at 20 folds
and its labels are not drawn to a wrong legend — it is simply **redundant**: a
control that adds no information matches it at both fold counts. The 11,411 rows,
the second data dependency and the 2020 extraction buy nothing.

Worth keeping for their own sake, independent of the pool:

* **The 5-vs-20-fold sign flip (X2).** Every arm's sign, and the whole mechanism
  story, is a property of the holdout radius. Ledger rows at one fold count are
  not evidence about another, and `--confusion` inherits the problem.
* **Vote agreement is a real filter (X3).** −0.0133 / −0.0228, both geometries.
* **`hcropland_points` drops 68% of its own source on legend grounds.** A binary
  map's negative class cannot enter a three-state legend; the machinery for a
  set-valued `{artificial, nature}` label does not exist here and building it
  would not fix the water/ice/barren leak that motivates the drop.

## Y — encoder capacity, and the constraint the Oslo map imposes (2026-08-03)

Two things arrive together here, from the user's reading of the Oslo inference:
**cropland has room to grow, but it must grow into Nature — a cropland gain paid
for out of built-up is a regression.** That is a constraint the ledger could not
even express, because `artificial_as_cropland` was not a tracked metric. It is
now (`diagnose_state_pools.metrics`, both ledgers migrated), and it changes how
the section below has to be read: `macro_f1` alone would call several of these
arms neutral when they are quietly moving built-up into cropland.

The first thing the new metric does is convict section X in retrospect. Adding
hcropland30 to either pool raises `artificial -> cropland` **and** hands pixels
back to Nature — the trade backwards on both counts, and the map-level form of
what the user saw:

| arm (20 folds, pooled OOF) | `art -> crop` | `nat -> crop` | cropland recall |
| --- | --- | --- | --- |
| `endpoints` | 0.0549 | 0.1444 | 0.7408 |
| `hcrop_endpoints` | **0.0630** | 0.1410 | 0.7458 |
| `glance_endpoints` | 0.0720 | 0.1592 | 0.7704 |
| `glance_hcrop_endpoints` | **0.0782** | 0.1571 | 0.7739 |

### Y1 — capacity, the one axis of this encoder nobody had moved

`64 -> 512 -> 256 -> 128` was chosen once and inherited by every siamese run in
the project. The arms below change **only** the hidden widths, so each stays a
drop-in: same single-date input, same `siam_dim` output, same linear last layer,
which is what `encode_single`, `_SiameseTrunk`'s mixer width `d_comb` and the
pyramid heads (which read `stage_dims`) all depend on. `model_zoo` now takes
`siam_hidden` and derives `stage_dims` from it, so a winner here is deployable
without a second edit; the default is byte-identical to before.

| arch | `enc` params | `t1_all` @5 | `t1_all` @20 |
| --- | --- | --- | --- |
| `mlp_narrow` 256-128 | 66,816 | **0.7517** | 0.7636 |
| `mlp` 512-256 (incumbent) | 199,040 | 0.7508 | 0.7613 |
| `mlp_deep` 512-512-256 | 462,720 | 0.7495 | **0.7637** |
| `mlp_wide` 1024-512 | 660,096 | 0.7470 | 0.7605 |
| `mlp_xl` 1024-512-256 | 759,168 | 0.7422 | 0.7612 |

Paired against `mlp` on the same pool and folds (`--paired-by arch`):

| arch | `macro_f1` @5 | `macro_f1` @20 | `f1_cropland` @5 | `f1_cropland` @20 |
| --- | --- | --- | --- | --- |
| `mlp_narrow` | +0.0024, 15/25 | +0.0015, 52/100 | +0.0045, 17/25 | +0.0019, 53/100 |
| `mlp_deep` | +0.0000, 15/25 | +0.0023, 62/100 | −0.0003, 9/25 | +0.0000, 46/100 |
| `mlp_wide` | −0.0043, 7/25 | −0.0004, 56/100 | −0.0040, 10/25 | −0.0023, 46/100 |
| `mlp_xl` | **−0.0081, 3/25** | −0.0002, 51/100 | **−0.0094, 5/25** | −0.0053, 37/100 |

**More capacity is not a lever, and at the harder holdout it is a liability.** At
5 folds the aggregate falls monotonically with parameter count — `mlp_xl` is
−0.0081 at 3/25 folds with 3.8× the encoder — and at 20 folds every arm collapses
to the incumbent. The one arm that is not below at either geometry is
`mlp_narrow`, a **third** of the incumbent's parameters. Nothing here clears the
±0.005 bar in either direction, so the recommendation is unchanged; what is
settled is that the answer to "make the network bigger" is no, and that the
current width is if anything already past the peak.

### Y2 — the two cropland errors move together, which is why capacity cannot fix this

The user's constraint asks for a *specific* trade: more cropland, taken from
Nature, not from built-up. Paired against `mlp`, lower is better on both:

| arch | `artificial_as_cropland` @5 | @20 | `nature_as_cropland` @5 | @20 |
| --- | --- | --- | --- | --- |
| `mlp_narrow` | +0.0053 | +0.0015 | +0.0040 | +0.0030 |
| `mlp_deep` | +0.0020 | +0.0015 | +0.0003 | −0.0023 |
| `mlp_wide` | **−0.0020** | **−0.0018** | −0.0061 | −0.0048 |
| `mlp_xl` | +0.0038 | +0.0005 | −0.0038 | −0.0060 |

Read the two blocks together and the mechanism is plain. `mlp_narrow` grows
cropland — `f1_cropland` +0.0045, `nature_as_cropland` +0.0040 — and takes it
from built-up as well (`artificial_as_cropland` +0.0053). `mlp_wide` shrinks
cropland — `f1_cropland` −0.0040, `nature_as_cropland` −0.0061 — and relieves
built-up (−0.0020) by predicting less cropland everywhere.

**Capacity is a single knob that scales how much cropland the model predicts. It
does not separate "take from Nature" from "take from built-up" — the two move in
the same direction in all four arms.** No width will satisfy the constraint,
because the constraint is about the *ratio* of two errors and width only moves
their sum. What can move a ratio is an asymmetric, class-pair cost, which is the
family that has already produced this project's only built-up wins (the nested
per-class cost gate, `cb_focal`'s `art_stable_as_veg` 0.225 → 0.151, Mondrian
conformal's +0.036, the CRFE gate). That is the next run, not a bigger encoder.

Two cautions on the numbers above. The fold-positive counts for
`artificial_as_cropland` are weak evidence at 20 folds — **41 of 100 folds are
exact ties**, because a twentieth of the plots leaves few Artificial rows per
fold and the rate saturates; the non-tied split for `mlp_wide` is 33 negative to
26 positive, which is not a result on its own. And section X2 applies here too:
`mlp_wide` and `mlp_xl` are clearly negative at 5 folds and indistinguishable at
20, so the capacity penalty is itself a property of how much world is held out.

### Y3 — `mlp_cw` already satisfies the constraint, and it is not a capacity change

The prediction at the end of Y2 — that a class-pair cost, not a width, is what
moves the ratio — is testable without writing anything, because `mlp_cw` is
already in the ladder. V3 ran it and recommended it for a *different* reason
(`artificial_as_nature` −0.0363 in 200/200 folds); those rows predate
`artificial_as_cropland` and carry NA there, so it was re-run. Paired against
`mlp`, tie counts shown because the constraint metric saturates:

| metric | @5 folds | @20 folds | wanted |
| --- | --- | --- | --- |
| `f1_cropland` | **+0.0063**, 16 pos / 9 neg | **+0.0039**, 56 pos / 35 neg | up |
| `nature_as_cropland` | **+0.0080**, 21 pos / 2 neg | **+0.0088**, 62 pos / 14 neg | up (this is where cropland should come from) |
| `artificial_as_cropland` | **−0.0038**, 13 neg / 5 pos / 7 tie | **−0.0037**, 34 neg / 7 pos / 59 tie | **down** |
| `artificial_as_nature` | −0.0309, 25 neg / 0 pos | −0.0365, 79 neg / 0 pos | down |
| `macro_f1` | −0.0014, 8 pos / 17 neg | −0.0048, 36 pos / 63 neg | the cost |

**All three parts of the user's constraint, at both geometries, one-sided on
every non-tied fold count.** Cropland grows; the growth comes out of Nature; and
built-up leaks *less* into cropland, not more. `mlp_cw` is the first arm in this
file to move the ratio rather than the sum, and it does it by re-weighting
classes at the head — a fifth of a line — while four capacity arms spending up to
3.8× the parameters could not.

The price is the aggregate: `macro_f1` −0.0014 / −0.0048, and `f1_artificial`
−0.0041 at 20 folds. That is the same trade V3 described and the reason it was
already the one arm here worth a transition run; what section Y adds is that the
trade is **better than V3 could see**, because two of the three things it buys
were not being measured. It should still be judged downstream on
`art_stable_recall` and the map, not on `macro_f1`.

Note the direction this settles: `mlp_wide` also lowered `artificial_as_cropland`
(−0.0020 / −0.0018) but did it by shrinking cropland everywhere, and `mlp_cw`
lowers it *while growing* cropland. Those are not the same result, and only the
second is what the Oslo read asked for.

### Y4 — the transition run and the Oslo map. The gain is real and small; the map is the gate's

Y3's state-level result was spent on the run it was recommended for.
`siam_s2off_state_pre_cw` is `siam_s2off_state_pre` (P7e) with
`siam_state_class_weight="balanced"` and nothing else changed — 5 seeds, blocked
CV, `deploy="aef_only"` so serving cost is unchanged. On the GLanCE pool the
weights are `artificial ×1.4024`, `cropland`/`nature ×0.8745`, mean exactly 1.0,
so the loss scale does not move.

| | cw (Y3) | `state_pre` (P7e) | deployed |
| --- | --- | --- | --- |
| `change_f1` | 0.6645 | 0.6657 | 0.6557 |
| `macro_f1` | 0.7088 | 0.7092 | 0.6938 |
| **`focus_macro_f1`** | **0.4239** | 0.4170 | 0.3674 |
| `art_stable_recall` | 0.6568 | 0.6574 | 0.6421 |
| `art_stable_as_veg` | 0.2055 | 0.2063 | 0.1938 |
| **`Artificial -> Cropland` F1** | **0.1365** | 0.1147 | 0.0026 |

**Free, and it lands where Y3 predicted.** `change_f1` and `art_stable_recall`
are unmoved (−0.0012, −0.0006, both inside noise), `focus_macro_f1` is +0.0069,
and effectively all of it is `Artificial -> Cropland` +0.0218 — the class the
whole cropland/built-up question is about. The state-level trade Y3 priced at
−0.0014/−0.0048 macro-F1 does **not** show up at the transition level.

Then the Oslo maps, both models, 5 seeds
(`data/inference/s2_20260803_150749/`). Here is the trap, and it caught this
analysis first time through:

| read | change-only IoU | self-IoU floor | verdict |
| --- | --- | --- | --- |
| `merged2` | **0.9736** | 0.8402 | far above floor — same map |
| `coarse3` (ungated) | **0.9322** | 0.8362 | above floor — same map |
| `coarse3_gated` (**shipped**) | **0.7479** | 0.8362 | **below floor — genuinely different** |

The gated maps differ and the ungated ones do not, which locates the difference
exactly: **it is the coarse3 cost vector, not the network.** `fit_coarse3_costs`
is refitted per model and chose different multipliers —
`{Artificial -> Cropland ×2.0, Cropland -> Artificial ×0.8}` for the baseline
against `{Artificial -> Cropland ×1.4, Nature -> Artificial ×1.2}` for cw. Read
the same class ungated and gated:

| class | baseline ungated | cw ungated | baseline **gated** | cw **gated** |
| --- | --- | --- | --- | --- |
| `Artificial -> Cropland` | 266 | 292 | 1,744 | **820** |
| `Nature -> Artificial` | 6,894 | 6,790 | 6,918 | **8,810** |
| `Cropland -> Cropland` | 40,058 | 40,233 | 39,504 | 40,010 |

The headline "cropland stops eating built-up, `Artificial -> Cropland` halves"
is **the ×2.0 → ×1.4 multiplier**, not the reweighting: ungated, that class goes
the other way (+26 px). Likewise the +27% `Nature -> Artificial` is the cw gate's
×1.2, a class the baseline's vector does not touch at all.

Two things follow and both are worth more than the arm itself:

* **The gate is the cheap lever on this map, and it needs no retraining.** If the
  goal is fewer `Artificial -> Cropland` pixels on the shipped raster, editing
  the cost vector does it directly and in seconds. The multipliers are currently
  chosen to maximise `focus_macro_f1` on plots, which is not the same objective
  as the user's map constraint and has no term for it.
* **Comparing two models' shipped maps compares two gates as well as two
  networks.** Every previous map A/B in this project refitted costs per arm, so
  this confound is in all of them. Compare the **ungated** rasters to see a
  network change, and hold the cost vector fixed if the gated read is wanted.

`merged2` change pixels are 10,399 → 10,251 (−1.4%), inside the ±5% a seed block
moves on its own, so the "whether change" map is untouched. **Oslo has no
labelled plots, so nothing here can be scored** — the gated map is a real
difference and only the user's visual read can say whether it is an improvement.

## Z — a triplet term, and hcropland30 as a positive bank (2026-08-03)

Section X closed hcropland30 as **redundant**: its own class-density control
matched it fold for fold. But that verdict was reached under one objective, and
the reason the pool could only ever be concatenated there is a clue rather than a
conclusion — **a pool of one class cannot train a softmax head**. Under a metric
loss the same property inverts: 32,343 globally-spread cropland points are not a
degenerate classification pool, they are a **positive bank**, 32,343 chances to
say *cropland in Kenya and cropland in Iowa belong together*. That is exactly the
invariance the Cropland/Nature boundary lacks, and it is a different question
from "which state is this row", so X's redundancy finding does not carry
automatically. It has to be re-run.

`_batch_hard_triplet` (`statepre/models.py`) is batch-hard (Hermans et al. 2017)
on L2-normalised embeddings with cosine distance, matching `_invariance` here and
`model_zoo._siam_cos_loss` in the deployed model. **Batch-hard rather than random
triplets is not a preference**: with three classes and a batch of 2048 almost
every random triplet is already separated and contributes exactly zero gradient,
so the term would be a no-op dressed as an experiment. Anchors whose state is
alone in the batch are dropped — load-bearing for these arms, because a batch
drawn mostly from a cropland-only pool can legitimately contain one state, and a
"hardest positive" taken from another class would invert the objective rather
than skip it (`tests/test_statepre.py`).

### Z1 — the triplet works, mildly, and it does not care which pool it is given

`mlp_trip` − `mlp`, paired, 5 seeds, `macro_f1`:

| pool | @5 folds | @20 folds |
| --- | --- | --- |
| `glance_endpoints` | **+0.0025, 19/25** | +0.0018, 55/100 |
| `glance_cropdup_endpoints` | +0.0020, 17/25 | +0.0011, 56/100 |
| `glance_hcrop_endpoints` | +0.0015, 16/25 | **+0.0001, 44/100** |

A small positive, consistently signed at both fold counts — which is more than
most of this file manages (compare X2, where every sign flipped) — but under the
±0.005 bar, so it is not a win. The important column is the last row: **the
triplet helps *least* on the pool that contains hcropland**, which is the
opposite of the hypothesis.

It also does something specific and worth keeping, on `glance_endpoints`:

| metric | @5 folds | @20 folds |
| --- | --- | --- |
| `f1_cropland` | +0.0017 | +0.0001 |
| `nature_as_cropland` | **−0.0062**, 17 neg / 4 pos | **−0.0078**, 55 neg / 12 pos |
| `artificial_as_cropland` | −0.0004 | **−0.0030**, 29 neg / 12 pos |

Both cropland confusions fall while `f1_cropland` stays flat — the term
**sharpens the cropland boundary** rather than moving it. Note what that means
against section Y's frame: this is another lever that moves the *sum* of the two
cropland errors, not their ratio. It satisfies the constraint half of the Oslo
read (`artificial_as_cropland` down, 29 neg / 12 pos) without the growth half.
`mlp_cw` (Y3) remains the only thing that has moved the ratio.

### Z2 — the positive bank does not work, and this closes hcropland30

Under the triplet, against its own control:

| comparison (arch `mlp_trip`) | @5 folds | @20 folds |
| --- | --- | --- |
| **`glance_hcrop` − `glance_cropdup`** | **−0.0006, 13/25** | **+0.0012, 51/100** |
| `glance_hcrop` − `glance` | −0.0046, 6/25 | +0.0005, 44/100 |
| `glance_cropdup` − `glance` | −0.0039, 10/25 | −0.0007, 46/100 |

Compare the first row to X1's: **−0.0002 / +0.0022 under cross-entropy, −0.0006 /
+0.0012 under the triplet.** The same coin flip, from a completely different
objective. On cropland itself the arms are equally silent — `f1_cropland`
−0.0036 / +0.0041 and `artificial_as_cropland` −0.0019 / +0.0016, both flipping
sign with the fold count.

**Two objectives, one verdict: the pool adds nothing that resampling GLanCE's own
cropland does not.** The positive-bank argument was the strongest remaining case
for hcropland30 and it is now tested and negative, so section X's "redundant"
becomes final rather than provisional.

A plausible reading of *why*, offered as a hypothesis rather than a measurement:
batch-hard mining consumes only the **hardest** positive per anchor, so 11,411
additional cropland rows mostly supply positives the miner never selects — and
where they are selected, they are hard precisely because they are a genuinely
different kind of cropland, so pulling them together asks the encoder to collapse
land that the AlphaEarth embedding has separated for a reason. That would explain
both the null against the control and why the triplet's gain is *smallest* on the
pool containing it. Testing it would need a mined-pair audit, which is not worth
a run given the verdict.

### Section Z verdict

**The triplet term is a keeper-in-principle and hcropland30 is closed.**

* `mlp_trip` on `glance_endpoints` is +0.0025 / +0.0018, consistently signed, with
  `nature_as_cropland` −0.0062 / −0.0078 at 17/4 and 55/12 folds. Under the win
  bar, so it does **not** change the recommendation, but it is the only
  architecture arm in this file besides `mlp_cw` that does the same thing at both
  fold counts. If a transition run is ever spent on an encoder-side change, this
  is the second candidate after `mlp_cw`, and the two are compatible
  (`mlp_trip_cw` is registered and unrun).
* **hcropland30 is negative under both a classification and a metric objective.**
  Do not propose a third. Any future single-class pool faces the same two
  questions: it cannot train the head alone, and its positives are matched by
  resampling the positives already present.

$P -m statepre.run --list                                   # arms and archs
$P -m statepre.run --migrate                                # ledger columns, once
$P -m statepre.run --datasets endpoints stable_years \
                   --archs linear mlp --seeds 5             # appends to the ledger
$P -m statepre.run --report --read t1_all                   # seed-averaged read
$P -m statepre.export --arm stable_years                    # -> a real state pool

# section V: a regional pool, read regionally. 20 folds is not optional here --
# at 5 folds LUCAS is one fold and holding it out deletes the pool.
$P -m statepre.run --datasets glance_lucas_endpoints --archs mlp \
                   --seeds 5 --n-folds 20
$P -m statepre.run --paired glance_lucas_endpoints glance_endpoints \
                   --archs mlp --n-folds 20 --by-region

# section X: hcropland30. Extract once (2020, ~10 min), then read at BOTH fold
# counts -- X2 is that the sign is a property of the holdout radius.
$P build_state_labels.py --sources hcropland --max-block-rows 500 --max-workers 12
$P -m statepre.run --datasets glance_hcrop_endpoints glance_cropdup_endpoints \
                   glance_hcropall_endpoints hcrop_endpoints \
                   --archs mlp --seeds 5 --n-folds 5
$P -m statepre.run --paired glance_hcrop_endpoints glance_cropdup_endpoints \
                   --archs mlp --n-folds 5
```

```bash
# section Y: capacity, and the Oslo constraint. --paired-by arch holds the pool
# fixed and compares two encoders; --metric picks the fold-ledger column.
$P -m statepre.run --datasets glance_endpoints \
                   --archs mlp mlp_narrow mlp_deep mlp_wide mlp_xl \
                   --seeds 5 --n-folds 5
$P -m statepre.run --paired mlp_xl mlp --paired-by arch \
                   --datasets glance_endpoints --n-folds 5 \
                   --metric artificial_as_cropland
```

`--metric` and `--paired-by` were added with section Y; before that `paired`
compared datasets only and always on `macro_f1`. Both ledgers gained
`artificial_as_cropland` in the same change, so **rows written before
2026-08-03 carry NA there** — they never computed it, and a backfilled zero would
have read as "this arm leaked no built-up". Re-run an arm to get the column.

Accuracy and the 3x3 confusion matrix for any arm — re-runs it, does not touch
the ledger, because a 3x3 table per (arm, seed, read) does not fit a flat run
log:

```bash
$P -m statepre.run --confusion --datasets lucas glance endpoints \
                   --archs mlp --seeds 5 --n-folds 20
```

The same comparison as a picture, with the pools projected into the plots' own
2018 embedding space and coloured by **Source × state 2018**:

```bash
$P umap_embedding.py --pools lucas glance_strict
# -> outputs/umap_embedding_pools.html
```

That page writes beside `umap_embedding.html` rather than over it: the 2018
manifold is fitted on whatever it is given, so adding 25,478 pool rows moves
every plot in that view. The two pages are different projections and only the
`2018` view contains the pools at all — the pool rows carry null coordinates in
`2024`, `diff` and `both`, and the page hides them there rather than dropping
them, so a row index means the same point in every view.

`--by-region` splits the paired folds by whether `--region-pool` (default
`lucas`) reaches them. It requires `fold_ref="reference"` and refuses to run
otherwise: under `union` the arm holding the pool and the arm without it cut
different folds, so a fold id is not a shared key and the split would compare
two different regions.

`statepre/export.py` is the bridge to the deployed phase and it needs **no change
to `model_zoo`**. `_prepare_state_pool` asks a pool for a `state` column and the
64 `A00_2018..A63_2018` columns, and `_pretrain_state` encodes that block with
`encode_single(..., "2018")` unconditionally — so a 2021 vector written into the
2018 columns is, to the phase, one more single-date state label. The augmentation
arrives as a bigger pool *file*. `block_id` rides along because
`twotower_lab.cv_probs_state` cuts the pool to each fold's training blocks, which
is what keeps the endogenous arms — they *are* RECOVER plots — out of the
held-out block.

## Status

**The dataset axis is closed. Two recommendations are ready to test downstream.**

Sections U1–U3, V and X have now tried, on this phase's own LLTO read, every pool
the project has access to: year augmentation (negative as rows, neutral as a
sampler), pseudo-labelled middle years (negative), GLanCE (positive, and the
recommendation), LUCAS (negative, with the cause named) and a global cropland map
(redundant — its own class-density control matches it at both fold counts). Four
consecutive iterations — V1, V2, V3, X — came back flat or negative on the
headline read, which is `AUTORESEARCH.md` rule 6's stopping condition: **the
bottleneck on this phase is the Cropland/Nature legend, not the pool and not the
architecture.** X is the sharpest form of that: a pool aimed squarely at the
Cropland class, global in reach and 32,343 labels deep, adds nothing, because the
problem was never a shortage of cropland examples. What remains is downstream —
whether `glance_endpoints` and `mlp_cw` do anything at the transition level.

What is settled:

* **LLTO is required and `block` is not a substitute.** Every existing number in
  this project's ledgers is blocked on 20° blocks, which sits 0.024 above LLTO,
  and the *ordering* of external against endogenous pools reverses between them.
* **The year augmentation as extra rows is negative** on the deployed encoder
  (−0.0239 paired, 4/25 folds). The cause is the vote distortion, established by
  three separate instruments: the `_dup` control, the `per_loc` weighting, and
  the `yeardrop` sampler.
* **The sampler undoes it** — `yeardrop` over `mlp` on the augmented pool is
  +0.0310 paired at 23/25 folds, and does nothing (12/25) where there is no
  distortion to undo. The mechanism is established; the resulting recipe still
  only reaches the baseline (16/25) rather than beating it.
* **Random forests are not usable under a spatial holdout here** — 7 points below
  a linear probe.
* **`glance_endpoints` — the union pool — is the win on the state read**,
  +0.0234 paired over GLanCE alone, re-confirmed under V0's fixed geometry at
  22/25 and 84/100 folds — **and it is negative at the transition level**
  (P7i: change-F1 0.6450 against 0.6644 unpretrained). Both statements are true
  and the second is the one a deployment obeys.
* **The fold geometry is a property of the protocol, not of the arm** (V0). It
  was not, and the error was large enough to matter for any regionally
  concentrated pool.
* **LUCAS is negative and the cause is named** (V1). Not volume, not geography,
  not weighting: it returns 66% of RECOVER's Cropland and 50% of its Artificial
  as Nature, and adding it moves the union pool's `cropland_as_nature` the wrong
  way on the boundary the project's change-F1 ceiling sits on. Do not re-open it
  without a legend harmonisation, and see the note below on what that would take.
* **`per_cell` / `per_source` reweighting is inert** (V2). The geographic vote
  distortion is a real mechanism — the `eudup` control loses 2/16 folds at home —
  but no weighting scheme recovers anything from LUCAS.
* **hcropland30 is redundant under BOTH objectives** (X, Z2). 32,343 cropland
  labels from a global map, 60 of 83 blocks, and its own class-density control
  matches it at −0.0002 (14/25) / +0.0022 (52/100) under cross-entropy and
  −0.0006 (13/25) / +0.0012 (51/100) under a batch-hard triplet. The
  positive-bank argument — that a single-class pool is degenerate for a softmax
  but ideal for a metric loss — was the strongest case left for it and is
  negative. Closed; do not propose a third objective.
* **A verdict here can be a property of the holdout radius** (X2). Every seed
  of section X agrees with its own fold count and none agrees across it, and the
  `--confusion` mechanism story reverses too. Quote the fold count with the
  number, and do not diagnose a mechanism from one.
* **Vote agreement is a real filter on a map product** (X3). Dropping hcropland's
  six-map unanimity requirement costs −0.0133 (1/25) and −0.0228 (13/100) — the
  largest effect in the section, at both geometries.
* **Encoder capacity is not a lever, and 512/256 is past the peak if anything**
  (Y1). `mlp_xl` at 3.8× the parameters is −0.0081 at 3/25 folds under the harder
  holdout; `mlp_narrow` at a third of them is never below. `model_zoo` now takes
  `siam_hidden` so this stays testable, but the answer to "make it bigger" is no.
* **Two errors that move together cannot be separated by width** (Y2). Cropland
  taken from Nature and cropland taken from built-up rise and fall together in
  all four capacity arms; width moves their sum, and the Oslo constraint is about
  their ratio. Only a class-pair cost moved the ratio (Y3).

What is not answered and would need a run:

1. ~~**P7i at the transition level.**~~ **Done, 2026-08-03, and negative** — see
   the box above. It is the third time a plot-level gain inverted downstream, and
   the reason the item below is written as a question rather than a plan.
2. **`mlp_cw` at the transition level** (V3, strengthened by Y3). −0.0052
   macro-F1 but `artificial_as_nature` −0.0363 in **200/200 folds**, which is the
   state-level form of `art_stable_recall` — the frontier metric ~45 ideas have
   failed to move. **Y3 adds the other half of the case**: on the metric added
   from the user's Oslo read it grows cropland (`f1_cropland` +0.0063 / +0.0039)
   out of Nature (`nature_as_cropland` +0.0080 / +0.0088) while *reducing*
   built-up leakage into cropland (`artificial_as_cropland` −0.0038 / −0.0037),
   at both fold counts. It is now the clear top priority for a transition run,
   and still a trade rather than a win: judge it on `art_stable_recall` and the
   map against `change_f1`, not on `macro_f1`, which it costs.
3. **Whether `yeardrop` survives on the union pool.** +0.0056 at 16/25 folds is
   the current state; more folds or an AOI read would settle it. Note that every
   `yeardrop` row in the ledger is `fold_ref="union"` and predates V0.
4. **The intermediate years for GLanCE.** Its stable segments carry
   `Start_Year`/`End_Year`, so the same augmentation is available for the
   external pool — but it needs a new GEE extraction, and U2a says to spend that
   only if the sampler result holds up first.

Explicitly **not** worth another run, on this evidence:

* **LUCAS in any concatenated form.** Three independent instruments agree — the
  regional paired read, the density control, and both reweighting schemes — and
  the diagnosis is a legend offset, which none of them can address. The only
  version that has not been tested is a *legend-harmonised* LUCAS, and that is a
  labelling task on the Cropland/Nature boundary, not a modelling one; it is the
  same ask `NOISY_LABEL_RESEARCH.md` section T and the learning curves already
  make, and it should be spent on RECOVER's own plots first, where a doubling of
  labels is worth +0.026 change-F1.
* **hcropland30's non-cropland points, as a set-valued `{artificial, nature}`
  label** (X). Two reasons, and the second is the fatal one: the phase's head is
  a 3-way softmax with no partial-label loss, and — more to the point — "not
  cropland" also covers the water, ice and barren that coarse3 has no home for
  and that GLanCE explicitly drops, so the set is wrong as well as unsupported.
  A set loss would import that leak, not fix it.
* **Any further global cropland product.** X's control result is not about
  *which* map, and Z2 shows it is not about which *objective* either: a pool that
  only adds cropland rows is matched by resampling the cropland already present,
  whether the loss asks "which class" or "which rows are alike". The next
  cropland pool would have to add something other than cropland volume — a
  different legend boundary, or the Nature side of the boundary — to be a
  different experiment.

Explicitly **not** tested, because the user's own validation sketch settles the
reading: the "each year as a new *feature column*" interpretation — a 7×64 input
stack — which would break the single-date contract the phase depends on.
