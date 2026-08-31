# TorchCP conformal methods on `siam_s2off_state_pre` (2026-08-04)

Asked: compare conformal methods from [TorchCP](https://github.com/ml-stat-Sustech/TorchCP)
on ECE, Brier, CRPS, coverage and efficiency, for the model behind
`data/inference/s2_20260731_120223/oslo_siam_s2off_state_pre_coarse3_gated.tif`.

Code: `src/conformal_torchcp.py`. Results:
`data/analysis_results/conformal_torchcp.csv` (2,400 rows, plots) and
`conformal_torchcp_map.csv` (216 rows, Oslo pixels).

Everything below is `full`, the blocked folds, alpha 0.10, 5 seeds unless
stated. TorchCP 1.2.1, installed to `~/.local` (`--user`), pixi env untouched.

---

## Read this before the tables

**Three of the five requested metrics cannot distinguish conformal methods.**
ECE, Brier and CRPS score the probability vector; a conformal predictor consumes
that vector and emits a set without altering it. Measured rather than asserted:
across all 96 score x predictor x alpha cells within one (level, calibrator,
seed), the standard deviation of ECE, Brier, CRPS, NLL and accuracy is **exactly
0.0**. Asking which conformal method has the best ECE is a category error.

So the grid is two axes and each owns some metrics:

| axis | arms | owns |
| --- | --- | --- |
| **calibrator** (probability transform) | `raw`, `temp` (Guo temperature, fitted per fold), `costgate` (the shipped coarse3 cost vector) | ECE, Brier, CRPS |
| **conformal method** (score x predictor) | 6 scores x 4 predictors | coverage, efficiency |

## Cross-check: TorchCP reproduces `twotower_lab.nested_conformal` exactly

Same protocol, two independent implementations, 5 seeds:

| level | mode | coverage (lab / TorchCP) | set size (lab / TorchCP) |
| --- | --- | --- | --- |
| merged2 | Mondrian | 0.8990 / 0.8990 | 1.4537 / 1.4537 |
| merged2 | marginal | 0.8995 / 0.8995 | 1.1500 / 1.1500 |
| coarse3 | Mondrian | 0.8988 / 0.8988 | 3.2565 / 3.2565 |
| coarse3 | marginal | 0.8999 / 0.8999 | 2.0248 / 2.0248 |

Identical to four decimals. The existing `nested_conformal` is correct and
TorchCP adds methods, not a correction.

---

## 1. The calibrator axis — ECE, Brier, CRPS

| level | calibrator | ECE | classwise ECE | Brier | CRPS (ord) | NLL | acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| merged2 | raw | 0.0368 | 0.0236 | 0.2219 | 0.0343 | 0.4220 | 0.8474 |
| merged2 | **temp** (T=0.84) | **0.0150** | **0.0142** | **0.2187** | **0.0342** | **0.4159** | 0.8474 |
| coarse3 | raw | 0.0689 | 0.0162 | 0.4333 | 0.0417 | 0.9039 | 0.6935 |
| coarse3 | **temp** (T=0.85) | **0.0286** | **0.0106** | **0.4268** | **0.0415** | **0.8939** | 0.6935 |
| coarse3 | costgate | 0.0719 | 0.0172 | 0.4351 | 0.0419 | 0.9096 | 0.6932 |

* **Temperature scaling halves ECE and is free.** 0.0689 -> 0.0286 at coarse3,
  0.0368 -> 0.0150 at merged2, at unchanged accuracy (one parameter cannot
  reorder a row). Brier and CRPS barely move, which is the expected signature:
  those are dominated by the refinement term, ECE by the reliability term.
* **T = 0.84 < 1, so the model is UNDER-confident.** It is being sharpened, not
  softened — the opposite of the usual deep-network finding, and consistent with
  the focal loss the recipe trains under.
* **`costgate` is a small calibration regression** (ECE 0.0689 -> 0.0719, Brier
  +0.0018, accuracy -0.0003). Expected and not an argument against it: the gate
  buys `focus_macro_f1` by deliberately biasing the posterior, and this is the
  price on the calibration side. It is fitted on these same rows, so if anything
  it is flattered here.

### CRPS needs an ordering stated, or it is half the Brier score

For nominal categories, CRPS under the discrete metric equals **exactly**
`Brier / 2` (`crps_nominal` in the CSV is that identity, printed to make it
visible). It carries information only against an ordering, so one is declared:

    naturalness: Artificial = 0, Cropland = 1, Nature = 2  (Vegetation = 1 at merged2)
    severity delta = naturalness(end) - naturalness(start)

giving five ordered rungs at coarse3 (-2 = `Nature -> Artificial`, the habitat
loss this project maps, through +2 = `Artificial -> Nature`). `crps_ord` is the
normalised RPS of that collapsed forecast. It separates errors Brier cannot:
predicting one rung / two rungs / four rungs from the truth scores 0.25 / 0.50 /
1.00, where Brier is a flat 2.0 for all three.

**This ordering is an assumption, not a project convention.** Change
`NATURALNESS` in the script if a different severity axis is wanted; nothing else
depends on it.

---

## 2. The conformal axis — coverage and efficiency

### coarse3 (K=9), raw posterior, alpha 0.10, 5 seeds

| score | predictor | coverage | macro cov | CovGap (pp) | set size | singleton | empty |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lac | **classwise** | 0.8988 | 0.9034 | **0.57** | **3.257** | 0.176 | 0.000 |
| aps | classwise | 0.9010 | 0.9026 | 0.77 | 3.477 | 0.177 | 0.018 |
| saps | classwise | 0.9007 | 0.9049 | 0.85 | 3.561 | 0.113 | 0.012 |
| raps | classwise | 0.8999 | 0.9054 | 0.80 | 3.682 | 0.009 | 0.000 |
| margin | classwise | 0.9007 | 0.9060 | 0.76 | 4.363 | **0.329** | 0.000 |
| margin | rc3p | 0.9086 | 0.9083 | 1.17 | 3.663 | 0.084 | 0.000 |
| lac | rc3p | 0.9230 | 0.9174 | 1.84 | 3.797 | 0.050 | 0.000 |
| lac | cluster | 0.9355 | 0.9298 | 4.71 | 6.067 | 0.000 | 0.000 |
| lac | split | 0.8999 | **0.7110** | **21.60** | 2.025 | 0.259 | 0.000 |
| aps | split | 0.9006 | 0.7628 | 16.34 | 2.354 | 0.256 | 0.013 |
| margin | split | 0.9000 | 0.7665 | 15.73 | 2.739 | 0.503 | 0.000 |

**The marginal predictor is the trap, and it fails exactly where this project
already has a problem.** `split` hits marginal coverage 0.8999 — nominally
perfect — while per-class coverage reads:

| class | n | split+lac | classwise+lac |
| --- | --- | --- | --- |
| `Cropland -> Nature` | 114 | **0.132** | 0.907 |
| `Artificial -> Cropland` | 46 | **0.426** | 0.917 |
| `Nature -> Cropland` | 243 | 0.596 | 0.905 |
| `Nature -> Nature` | 2532 | 0.972 | 0.899 |

Those are the two classes the ledger calls dead (0.0000 F1 in sections N, O, P)
and the third is the noisy boundary. A marginal set at 90% covers
`Cropland -> Nature` **13% of the time**. Any uncertainty product built on
`SplitPredictor` would be silently worthless on precisely the transitions it
exists to flag.

It gets worse as the level loosens, which is counter-intuitive: `split` CovGap
runs 14.1 / 21.6 / 25.8 / 26.9 pp at alpha 0.05 / 0.10 / 0.20 / 0.30, while
`classwise` holds 0.53 / 0.57 / 0.55 / 0.73.

### merged2 (K=4), raw posterior, alpha 0.10, 5 seeds

| score | predictor | coverage | macro cov | CovGap | set size | singleton |
| --- | --- | --- | --- | --- | --- | --- |
| lac | **classwise** | 0.8990 | 0.8985 | 0.34 | **1.454** | 0.632 |
| margin | classwise | 0.8998 | 0.9006 | **0.29** | 1.652 | **0.654** |
| aps | classwise | 0.9001 | 0.9004 | 0.59 | 1.638 | 0.469 |
| lac | rc3p | 0.9120 | 0.9038 | 0.87 | 1.567 | 0.543 |
| lac | split | 0.8995 | 0.7928 | 13.40 | 1.150 | 0.852 |
| lac | cluster | 0.9549 | 0.9586 | 5.86 | 3.024 | 0.163 |

### What each method is worth here

* **`LAC` + `ClassConditionalPredictor` wins both levels** — the smallest set at
  honest conditional coverage, no empty sets, and it is already what the repo
  does. TorchCP's contribution is the measured alternatives, not a replacement.
* **`Margin` is the one real find.** At merged2 it takes CovGap to the lowest in
  the grid (0.29) and the singleton fraction to the highest (0.654) for +0.20 set
  size. At coarse3 it nearly doubles the singleton fraction (0.176 -> 0.329) for
  +1.11 set size. If the product read is "how much of the map is called
  unambiguously", Margin is the score to look at, not LAC.
* **`ClusteredPredictor` is negative at this scale.** It over-covers (0.936-0.955)
  and returns 6.07/9 and 3.02/4 classes — no singletons at all at coarse3.
  Clustering classes needs many classes and many calibration rows per class; 9
  classes over 5 blocked folds has neither.
* **`RC3P` is negative here despite being the long-tail method.** It over-covers
  by 2-3 pp and costs +0.54 set size against `classwise` at coarse3. Its rank
  constraint is the binding one and this problem's rarity is not rank-shaped.
* **`RAPS`/`TOPK` destroy the singleton fraction** (0.009 / 0.0002 at coarse3) —
  they are built for 1000-class benchmarks where a size-10 set is a win.
* **`APS`/`SAPS` produce empty sets** (1.2-1.8% on plots, 6.9% on the map). An
  empty set is "no admissible transition", which is not a shippable pixel value.

**Calibration does not move the conformal read.** Temperature scaling changes
set size by -0.038 to +0.19 (mostly -0.01 to -0.03) against a 2.02-to-6.07 spread
across methods. Halving ECE buys nothing conformal. The two axes really are
separate.

---

## 3. The map read (2.95M Oslo pixels, alpha 0.10, 3 seeds)

Coverage is unmeasurable — Oslo has zero labelled plots — but efficiency is not,
and it is a *different number* because the map is ~65% stable Nature. Threshold
fitted once on the 5-seed-mean OOF (the raster is a 5-seed ensemble, so a
single-seed threshold would be cut on the wrong distribution), then applied to
the pixels.

| score | predictor | plot size | **map size** | map singleton | <=2 classes | empty |
| --- | --- | --- | --- | --- | --- | --- |
| margin | split | 2.739 | **1.275** | 0.787 | 0.980 | 0.000 |
| aps | split | 2.354 | 1.280 | 0.656 | 0.977 | 0.050 |
| aps | classwise | 3.477 | 1.319 | 0.676 | 0.940 | **0.069** |
| lac | split | 2.025 | 1.462 | 0.557 | 0.982 | 0.000 |
| **lac** | **classwise** | 3.257 | **1.511** | 0.627 | 0.912 | **0.000** |
| margin | classwise | 4.363 | 2.098 | 0.733 | 0.761 | 0.000 |
| lac | rc3p | 3.797 | 2.460 | 0.391 | 0.480 | 0.000 |
| lac | cluster | 6.067 | 3.562 | 0.000 | 0.202 | 0.000 |

* **Every method's sets shrink on the map** (LAC/classwise 3.26 -> 1.51), and the
  *ranking changes*: APS/classwise is the smaller set on the map and the larger
  one on the plots. Plot-pool efficiency is not deployment efficiency, and
  quoting one for the other would be wrong in both directions.
* **The deployed choice reads well.** LAC/classwise gives 63% of Oslo a single
  transition and 91% two or fewer, at zero empty pixels.
* **APS's 6.9% empty pixels is the disqualifier**, and it is only visible on the
  map — on the plots it is 1.8%.

---

## Verdicts

1. **Keep `LAC` + Mondrian.** Nothing in TorchCP beats it on efficiency at honest
   conditional coverage, and the incumbent implementation is exactly right.
2. **Never ship `SplitPredictor`.** Marginal coverage is met and
   `Cropland -> Nature` is covered 13% of the time. This is the concrete version
   of a warning section R only made qualitatively.
3. **Add temperature scaling if a probability is ever published** (a confidence
   band, a soft-refinement input, an area estimator's weights). ECE halves for
   one parameter at zero accuracy cost, and it is *not* worth doing for the
   conformal sets, which do not notice.
4. **`Margin` is worth one real run** as the singleton-fraction lever
   (0.176 -> 0.329 at coarse3). Registered as an open question, not a finding —
   it has not been read against a map.
5. **`ClusteredPredictor` and `RC3P` are tested-negative at this scale.** Both
   over-cover and both cost set size. Do not re-open without more classes or more
   calibration rows per class.

## Tested negative — do not redo

`ClusteredPredictor` (K=9 and K=4, 5 blocked folds), `RC3PPredictor`,
`RAPS`/`TOPK` as map scores, temperature scaling as a route to smaller conformal
sets, and `costgate` as a calibration improvement.

---

## 4. Was this a cross-conformal setup? No — and here is what that cost

`conformal_torchcp.py` uses a **pooled-quantile CV protocol** (`cvq`), the one
`twotower_lab.nested_conformal` implements: for each blocked fold, cut one
quantile on the out-of-fold scores of the other folds and apply it. That is *not*
the cross-conformal predictor of Vovk 2012 (`arxiv.org/abs/1208.0806`), which
pools **counts** into a p-value

    p(y) = ( sum_k #{ i in S_k : alpha_i >= alpha_k(x, y) } + 1 ) / (n + 1)

The structural difference: CCP gives the test object **K scores**, one per fold
model, each compared against its own fold's homogeneous calibration scores; `cvq`
gives it **one score** and compares it against a pool of scores produced by K
*different* models. They coincide only when the fold models agree.

CCP cannot be computed from the OOF cache — it needs every fold model's opinion
of every test object, and the cache holds only the one model that excluded each
row. `src/conformal_crossconformal.py` therefore runs a nested design: an outer
blocked fold is held out, K=5 inner models are trained without it, and all five
score it. ~25 fits/seed, ~1 min. Three protocols on identical outer rows:

| | calibration data | test score | validity |
| --- | --- | --- | --- |
| `icp` | 1/5 of the pool, one model | that model | exact under exchangeability |
| `ccp` | all of it, counts pooled per fold model | K, one per model | approximate (paper's own caveat) |
| `cvq` | all of it, scores pooled | 1 (mean posterior) | approximate, and the pool is heterogeneous |

### LAC, Mondrian, 5 seeds (nominal coverage in the alpha column)

| alpha | protocol | coverage | macro cov | set size | singleton |
| --- | --- | --- | --- | --- | --- |
| 0.10 | ccp | 0.9073 ±0.0005 | 0.9077 | **3.311** ±0.054 | 0.175 |
| 0.10 | **cvq** (used) | 0.9089 ±0.0011 | 0.9093 | **3.347** ±0.072 | 0.170 |
| 0.10 | icp | 0.9002 ±0.0020 | 0.9217 | 4.564 ±0.061 | 0.015 |
| 0.05 | ccp | 0.9548 | 0.9560 | 4.659 | 0.086 |
| 0.05 | cvq | 0.9556 | 0.9583 | 4.722 | 0.083 |
| 0.05 | icp | 0.9531 | 0.9680 | 5.928 | 0.000 |
| 0.20 | ccp | 0.8073 | 0.8067 | 2.124 | 0.332 |
| 0.20 | cvq | 0.8093 | 0.8124 | 2.158 | 0.323 |
| 0.20 | icp | 0.8002 | 0.8287 | 2.666 | 0.218 |

**1. For LAC the two are indistinguishable.** Δ set size 0.036, Δ coverage 0.0016
at alpha 0.10 — both inside a seed sd. Marginal LAC is closer still (Δ set size
0.004). LAC is the deployed score, so **the method ranking in sections 1–3 stands
under either protocol.** The reason they agree is that `cvq` already delivers
CCP's actual selling point — all n rows calibrate, none are held back.

**2. APS agrees too — after a bug fix. CORRECTION.** The first version of this
section reported CCP over-covering APS by 4.2 pp at alpha 0.10 and removing its
empty sets, and concluded that APS verdicts were protocol-dependent. **That was
an artefact of `conformal_crossconformal.py`, not a property of CCP.** The APS
randomisation `u` belongs to the *test object*, but the code advanced one shared
rng per call, so each fold model scored the same row under a different draw and
CCP was silently averaging over K independent randomisations — a more
conservative estimator than the one Vovk defines. Fixed: one `u` per test row,
shared across the K models and across both protocols. Corrected:

| alpha | protocol | coverage | set size | empty |
| --- | --- | --- | --- | --- |
| 0.10 | ccp | 0.9185 | 3.617 | 0.0179 |
| 0.10 | cvq | 0.9117 | 3.623 | 0.0181 |
| 0.20 | ccp | 0.8233 | 2.502 | 0.0496 |
| 0.20 | cvq | 0.8145 | 2.523 | 0.0488 |

CCP still runs slightly conservative on the randomised score — +0.7 pp at alpha
0.10, +0.9 pp at 0.20, just outside the seed sd — but set size is a tie and the
empty-set rate is identical. **Section 2's "APS emits empty sets" stands as an
APS property.** LAC is unaffected by the fix (it has no randomisation), so
finding 1 never depended on it.

**3. The result that matters is about `icp`, not `ccp`.** Textbook split
conformal — the "correct" setup — is **the worst of the three under Mondrian**:
+1.25 set size at alpha 0.10 and a singleton fraction of 0.015 against 0.175.
Its per-class coverage is not calibrated at all:

| | Art→Art | Art→Crop | Art→Nat | Crop→Nat | Nat→Nat |
| --- | --- | --- | --- | --- | --- |
| ccp | 0.904 | 0.929 | 0.913 | 0.898 | 0.907 |
| cvq | 0.905 | 0.926 | 0.914 | 0.896 | 0.906 |
| icp | **0.864** | **0.980** | **0.970** | 0.936 | 0.904 |

Cause: `icp` calibrates the 46-plot `Artificial -> Cropland` class on **~7 rows**
(one inner fold of the training pool) against ~37 for the other two. A Mondrian
quantile needs n >= 9 at alpha 0.10 and n >= 19 at alpha 0.05, so that class
falls off the bottom — at alpha 0.05 its quantile is infinite and `icp` returns
zero singletons across the whole dataset. **Using the exactly-valid protocol
would have made this comparison worse, not better**, and only because the legend
is this long-tailed.

---

## 5. ECCP — e-value cross-conformal (arXiv 2606.03600). **Negative here.**

"Set-Preserving Calibration from Conformal P-Values to E-Values" (June 2026)
targets exactly the weakness in section 4: CCP is only guaranteed at
`1 - 2*alpha`. It defines a p-to-e calibrator

    F_{n,a}(p) = (1/a) * (1 + exp(C(a - s))) / (1 + exp(C(p - s)))

with `s` free on `(a, ceil(a(n+1))/(n+1))` and `C > 0` the unique root of
`(1/(n+1)) sum_k F(k/(n+1)) = 1` (the e-variable condition). Because
`F(a) = 1/a` exactly, `{p > a}` and `{F(p) < 1/a}` are the same set — that is the
"set-preserving" claim, and it is what lets fold-wise evidence be *averaged*
(an average of e-values is an e-value) to recover the full `1 - alpha`
guarantee. ECCP is then `{y : (1/K) sum_k F(P_k(y)) < U/a}`, `U ~ Unif(0,1)`;
eq. 16 gives a deterministic variant `{sup_{t<=K} (1/t) sum_{k<=t} E_k < 1/a}`.

Implementation verified before use: `E[e] = 1.0000000000` on the null grid,
`F(a)*a = 1`, and zero set disagreements against `{p > a}` at K=1, for every
`(m, alpha)` in play. The Mondrian reading (per-class `m` and per-class
calibrator) is an extension — the paper does not treat label-conditional
coverage, and this legend requires it.

### LAC, Mondrian, 5 seeds

| alpha | protocol | coverage | set size | singleton | empty | max per-class dev |
| --- | --- | --- | --- | --- | --- | --- |
| 0.10 | ccp | 0.9073 | **3.311** | 0.175 | 0.000 | 2.9 pp |
| 0.10 | cvq | 0.9089 | 3.347 | 0.170 | 0.000 | **2.6 pp** |
| 0.10 | **eccp** | **0.8982** | 3.897 | 0.143 | 0.002 | 4.0 pp |
| 0.10 | eccp_ex | 0.9048 | 4.898 | 0.017 | 0.000 | 8.0 pp |
| 0.10 | icp | 0.9002 | 4.564 | 0.015 | 0.000 | 8.0 pp |
| 0.20 | ccp | 0.8073 | **2.124** | 0.332 | 0.000 | — |
| 0.20 | eccp | 0.7956 | 3.183 | 0.214 | 0.036 | — |
| 0.20 | eccp_ex | 0.8443 | 3.729 | 0.057 | 0.000 | — |

**ECCP does what it claims and it is not worth it here.** Its marginal coverage
lands nearest nominal of any protocol (0.8982 vs 0.9073/0.9089 — CCP and the
pooled quantile both run ~0.8 pp rich), so the tighter guarantee is real and
visible. The price is **+0.59 set size at alpha 0.10 (+18%) and +1.06 at alpha
0.20 (+50%)**, plus 3.6% empty sets at 0.20 that neither CCP nor the pooled
quantile produce. On a legend where the whole product question is how much of the
map can be called unambiguously, an 18% wider set to remove an over-coverage of
0.8 pp is the wrong trade.

Not a parameter-tuning failure: sweeping `s` across its whole admissible interval
(`s_frac` 0.02 / 0.50 / 0.98) moves ECCP's set size only 3.81 / 3.88 / 4.00. The
gap to CCP's 3.30 survives the best setting.

**The exchangeable variant (eq. 16) collapses onto ICP.** At marginal LAC it
reproduces `icp` to four decimals (2.0936 vs 2.0936 at alpha 0.10; 1.3824 vs
1.3824 at 0.20), because `sup_{t<=K}` is dominated by its `t=1` term whenever the
first fold's e-value is large — and the `t=1` term *is* a single-fold split
conformal predictor. It inherits ICP's broken per-class coverage (Art→Art 0.864,
Art→Crop 0.980) for free. Do not use it.

### Verdict

`cvq` was the right choice. CCP would not have changed a conclusion drawn on LAC;
ECCP would have changed them for the worse. Ranked on this problem:

    cvq ~= ccp  >  eccp  >  icp ~= eccp_ex

`icp`, `eccp` and `eccp_ex` are tested-negative for this legend at K=9. Read
ECCP again only if a *guarantee* rather than a measurement becomes the
requirement — a regulator, or a level tight enough (alpha <= 0.02) that the rare
classes stop calibrating at all.

---

## 6. The recommendation: a guarantee at today's serving cost

Requirement: an exact coverage guarantee **and** cheap inference.

**That reduces the menu to one item.** Every construction in arXiv 2606.03600
that carries `1 - alpha` other than plain split conformal — ECCP, ECCP-Exch,
UR-ECCP-Exch, WECA, UR-WECA — needs **K trained models at inference**. And the
P2E calibrator is *set-preserving* by construction (`F(alpha) = 1/alpha`), so at
K=1 it returns byte-identical sets to plain split conformal. Verified here: zero
disagreements against `{p > alpha}` on the p-grid for every `(m, alpha)` in play.
The e-value machinery is an *aggregation* device; with a one-predictor budget
there is nothing for it to aggregate and it is not worth implementing.

So the only free parameters are the split ratio and the score. Swept
(`src/conformal_deployable.py`, three-way blocked split, 5 repeats x 5 outer
folds, LAC, Mondrian, alpha 0.10):

| calibration | n train | n cal | rarest class cal rows | classes with no quantile | coverage | set size | singleton | acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20% | 4098 | 1034 | 7.2 | **0.8** | 0.9003 | 4.555 | 0.014 | 0.6922 |
| **33%** | 3398 | 1733 | 12.8 | 0.0 | 0.9076 | **3.702** | 0.152 | 0.6889 |
| 50% | 2580 | 2552 | 18.8 | 0.0 | 0.9085 | 3.658 | 0.163 | 0.6846 |

**The split ratio is decided entirely by one class.** A Mondrian quantile needs
`ceil((m_c+1)(1-alpha)) <= m_c` rows *of that class* — 9 at alpha 0.10, 19 at
0.05. `Artificial -> Cropland` has 46 plots in the whole dataset, so at 20%
calibration it gets ~7, its quantile is infinite, it enters **every** set, and
mean set size jumps to 4.56 with a singleton fraction of 0.014. Lifting
calibration to 33% takes it to ~13 and set size falls to 3.70. Going on to 50%
buys another 0.04 of set size and costs 0.004 accuracy — not worth it.

**alpha 0.05 is unreachable.** Even the 50/50 split leaves 0.4 classes without a
quantile (18.8 rows against the 19 needed). No split ratio fixes that; it needs
more `Artificial -> Cropland` plots, which is the labelling ask the ledger
already records.

### What the guarantee costs against the incumbent

| | protocol | guarantee | models at inference | set size | acc |
| --- | --- | --- | --- | --- | --- |
| incumbent | `cvq`, all 6,414 plots calibrate | none (nominal only) | 5 | **3.347** | **0.6935** |
| **recommended** | Mondrian split conformal, 33% cal | **exact 1-alpha, marginal and per-class** | 5 | 3.702 | 0.6889 |
| | CCP / ECCP | 1-2a / 1-a | 5 per fold = 25 | 3.311 / 3.897 | — |

**+0.355 set size (+10.6%) and -0.005 accuracy, for zero extra serving cost.**
The 5-seed mean is what `infer_s2.py --seeds 5` already runs — for conformal
purposes an ensemble is one predictor — so the guarantee is bought entirely with
labels held out of training, not with forward passes. It beats the single-network
budget on both axes (set size 3.702 vs 3.742, accuracy 0.6889 vs 0.6868), so
there is no reason to drop to one network.

One oddity, recorded rather than explained: `change_f1` *rises* as training data
falls (0.6494 / 0.6530 / 0.6580 across the three ratios) while accuracy falls as
the learning curve predicts. The smaller training set produces a less confident
model, which trades precision for recall on the change class — consistent with
the ledger's note that change-F1 sits near its label-noise ceiling and moves on
the operating point. Do not read it as "less data is better".

### Verdict

Ship **Mondrian LAC split conformal at a 1/3 calibration split on the existing
5-seed ensemble** if a guarantee is required. Keep `cvq` if it is not — it is
10% tighter and uses every plot for training. Do not implement the P2E
calibrator for this: at K=1 it is provably the same set.

## Notes on the environment

* `pip install --user torchcp torchsort`. `torchcp/__init__.py` eagerly imports
  its graph/ and llm/ subpackages, needing torch_geometric, torchvision and
  transformers; `conformal_torchcp.py` installs a meta-path finder that stubs
  those roots rather than adding three trees to the pixi env.
* `RC3PPredictor.calculate_threshold` is broken when driven from precomputed
  logits — it reads `self.num_classes`, which only `calibrate()` (the dataloader
  path) ever assigns. Worked around in `_prepare()`.

## The raster this was asked about is not the instance that was scored

`data/inference/s2_20260731_120223/` was written on 2026-07-31 from checkpoint
`siam_s2off_state_pre_seed0_n5_4323d95142575966.pt`. The recipe hash has since
moved: today's rerun loaded `...ebaf5599d80896e4.pt` (2026-08-03), and the OOF
cache used here is also 2026-08-03. The two maps agree on **99.11%** of pixels
with a change-class IoU of **0.8384** — right at the documented ~0.84 self-IoU
floor, so they are the same model to within reproducibility, but they are not the
same instance. The map read above uses the 2026-08-04 rerun
(`data/inference/s2_20260804_115134/`), whose posteriors and calibration cache
come from one instance. Comparing the July raster against an August calibration
would have been the error CLAUDE.md warns about.
