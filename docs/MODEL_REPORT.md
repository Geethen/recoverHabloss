# Model report — `s2off_centre_m3s3_bf`

**Status:** deployed, settled 2026-07-27. **Read this with `CLAUDE.md`**, which
holds the operating rules; this document is the description and the evidence.

Everything below is measured on 6,414 labelled plots under spatially blocked
5-fold CV at the **gate-off read** — the read the shipped map actually uses —
over 5 torch seeds. New evidence in this report: `data/analysis_results/learning_curves.csv`,
`learning_curves_coarse3.png`, `learning_curves_merged2.png`, produced by
[learning_curves.py](src/learning_curves.py).

---

## 1. What the model is

A hierarchical two-tower network over AlphaEarth 2018/2024 embeddings, trained
with a Sentinel-2 detail tower as **privileged information** and served
**AlphaEarth-only with the detail gate forced off**. Sentinel-2 is never read at
inference.

| | |
| --- | --- |
| Recipe | `s2off_centre_m3s3_bf`, registered in [infer_s2.py](src/infer_s2.py) `fit_models` |
| Inputs, served | 192 AlphaEarth columns (64 channels × 2018 / 2024 / diff) |
| Inputs, training only | 78 Sentinel-2 columns — `S2_SUBSETS["centre_m3s3_bf"]` in [twotower_lab.py:202](src/twotower_lab.py#L202) |
| Trunk | `_TwoTowerTrunk`, two 1024→512→256 towers, `fusion="gated_mean"`, `modality_dropout=0.5`, detail-tower dropout 0.7 vs AlphaEarth's 0.4 |
| Head | one fine head; merged2 and the change gate are exact 0/1 group-sums of the coarse3 softmax |
| Training | focal loss on all three levels, 30 epochs **full-batch** (30 optimiser steps), AdamW + OneCycle |
| Serving | 5-seed posterior average through `probs_aef_only_matrix`; no pandas, no detail tower |
| Labels | 9 coarse3 transitions (Nature / Cropland / Artificial × two endpoints), `MIN_COUNT=20` |

Detail tower = 7 Sentinel-2 channels at the plot centre + their 3×3 mean and
standard deviation + built fraction at 5 radii, for 2018, 2024 and their
difference: 7×3 + 5 = 26 per year, ×3 = 78.

**Two output reads, not interchangeable.** `*_merged2.tif` answers *whether*
change occurred; `*_coarse3.tif` answers *what kind*. They disagree on ~15% of
change pixels because arg-max does not commute with the group sum. Class codes
follow the **sorted** class list — read them from the `.qml` sidecar.

### Reproduce

```bash
/home/geethen.singh/.pixi/envs/geo/bin/python src/infer_s2.py \
    --aois oslo --models s2off_centre_m3s3_bf --seeds 5
```

Deployed outputs: `data/inference/s2_20260727_152853/` (seeds 0–4);
`s2_20260727_153203/` is the seeds 5–9 replication.

---

## 2. Headline accuracy

Gate-off read, blocked 5-fold CV, 5 seeds, full training set.

| level | metric | value |
| --- | --- | --- |
| merged2 (deployed) | change-F1 | **0.6568 ± 0.0035** |
| merged2 | macro-F1 | **0.6943 ± 0.0025** |
| coarse3 (reported) | change-F1 | 0.5964 ± 0.0032 |
| coarse3 | macro-F1 | 0.4388 ± 0.0017 |

Per class, at the full training set:

| class (merged2) | plots | OOF F1 | train F1 | gap |
| --- | ---: | ---: | ---: | ---: |
| Vegetation → Vegetation | 4,550 | 0.9125 | 0.9368 | 0.024 |
| Vegetation → Artificial | 716 | 0.6883 | 0.7915 | 0.103 |
| Artificial → Artificial | 979 | 0.6818 | 0.7785 | 0.097 |
| Artificial → Vegetation | 169 | 0.4945 | 0.6586 | 0.164 |

| class (coarse3) | plots | OOF F1 | train F1 | gap |
| --- | ---: | ---: | ---: | ---: |
| Nature → Nature | 2,532 | 0.7491 | 0.8106 | 0.062 |
| Cropland → Cropland | 1,661 | 0.7244 | 0.7808 | 0.056 |
| Artificial → Artificial | 979 | 0.6847 | 0.7782 | 0.094 |
| Cropland → Artificial | 333 | 0.5906 | 0.6954 | 0.105 |
| Nature → Artificial | 383 | 0.4693 | 0.6288 | 0.160 |
| Artificial → Nature | 123 | 0.4066 | 0.5382 | 0.132 |
| Nature → Cropland | 243 | 0.3104 | 0.4253 | 0.115 |
| **Cropland → Nature** | 114 | **0.0137** | 0.0458 | 0.032 |
| **Artificial → Cropland** | 46 | **0.0000** | 0.0563 | 0.056 |

---

## 3. Learning curves — the new evidence

![coarse3 learning curves](../data/analysis_results/learning_curves_coarse3.png)

![merged2 learning curves](../data/analysis_results/learning_curves_merged2.png)

Training-set size is varied from 5% to 100% of each training fold, **nested**
(the 10% draw is a subset of the 20% draw) and **stratified on the fine
transition**. The test side never shrinks: at every size the model is scored
out-of-fold over all 6,414 plots. The train curve is scored on the subsample the
model was fitted on, so the gap between the two series is readable directly.

The x-axis is each class's **own** training count, because class prevalence spans
46 to 2,532 plots and a shared fraction axis would compress every rare class into
the first tick.

### 3.1 Two coarse3 classes do not exist at inference

`Artificial → Cropland` (46 plots) scores **exactly 0.000 out-of-fold at every
training size and every seed**. `Cropland → Nature` (114 plots) scores 0.014.
Their *train* F1 is also near zero at full size — the network does not even
memorise them; it collapses them into their neighbours.

This is confirmed on the artefact. In the deployed Oslo `coarse3` map
(`summary.json`, both seed blocks, and every subset variant):

| coarse3 class | Oslo pixels |
| --- | ---: |
| Artificial → Cropland | **0** |
| Cropland → Nature | **0** |
| Nature → Cropland | 16 |

**The nine-class map is in practice a six-class map.** Two legend entries are
never painted and a third is painted 16 times in 2.95 M pixels. The curves
predicted exactly which ones, from plots alone, before the raster was consulted.

This is a labelling limitation, not a bug and not an architecture problem: these
are the classes on the Cropland/Nature boundary that
[analyse_label_noise.py](src/analyse_label_noise.py) already identified as
interpreter-noisy, at the smallest supports in the sample.

### 3.2 Every surviving class is still climbing at 100% of the data

Gain in OOF F1 per **doubling** of the training set, fitted over the top half of
each curve:

| read | per doubling | seed spread | ratio |
| --- | ---: | ---: | ---: |
| merged2 change-F1 | **+0.0264** | 0.0035 | 7.5× |
| merged2 macro-F1 | +0.0197 | 0.0025 | 7.9× |
| coarse3 change-F1 | +0.0288 | 0.0032 | 9.0× |
| coarse3 macro-F1 | +0.0179 | 0.0017 | 10.5× |

Steepest classes: `Artificial → Vegetation` +0.039, `Nature → Artificial` +0.035,
`Cropland → Artificial` +0.034 per doubling — the change classes, i.e. the ones
the project exists to map.

For scale: **every modelling idea in the ledger moved change-F1 by less than
0.005**, and most by less than the seed spread. Doubling the labelled sample is
worth roughly **five times** the entire eleven-iteration architecture search.
This quantifies what `AUTORESEARCH.md` rule 6 and `TWOTOWER_RESEARCH.md` G4 both
concluded qualitatively: the bottleneck is data.

### 3.3 The gap is real but is not the binding constraint

Train-minus-OOF F1 is 0.10–0.16 on the change classes and 0.02–0.06 on the two
big stable classes. That is a moderate variance gap, and it is *not* the shape of
a model memorising noise — train F1 at full size is 0.66–0.79 on the change
classes, nowhere near 1.0. The network is under-resourced rather than
over-fitted, which is consistent with heavier regularisation, capacity cuts and
noise injection all having tested negative.

---

## 4. Code audit — what I checked and what I found

Scope: the deployed path end to end (`infer_s2.py` → `model_zoo.HierarchicalSoftmaxNN`
→ `twotower_lab`), plus the scoring harnesses the published numbers come from.
The full test suite passes: **82 passed, 2 skipped**.

> **Caveat on `infer_s2.py`.** That file was being edited by another session
> while this audit ran (it gained a `torch.save`-based model cache — `_cache_metadata`
> / `_load_cached_entry` — and grew ~3 KB mid-review). The findings below cover
> the recipe, `predict`, `run_aoi` and the raster writers, which were stable
> throughout; **the new caching layer has not been reviewed against a settled
> version of the file.** Re-check it once that work lands.

**No bugs found in the deployed path.** Specifically verified sound:

- **Class-code alignment.** `write_class_raster` codes by the position of the
  list it is handed, and `run_aoi` hands it the already-sorted `merged_classes_` /
  `fine_classes_`. Palettes cover all 9 coarse3 and all 4 merged2 labels, so no
  class falls back to grey. The `.qml` sidecar and the raster cannot disagree.
- **Seed-ensemble safety.** `predict` refuses to average members that disagree on
  either class order, at both levels.
- **Fast-path exactness.** `probs_aef_only_matrix` is asserted bit-identical to
  `_probs` with the gate zeroed, including the absent-value → column-mean
  convention (`tests/test_s2off_fastpath.py`).
- **Train/serve agreement.** The scored configuration (`optimise_s2off.gate_off_cv`,
  `S2_MASK = 0` on the test frame) is the served configuration. `aef_present` is
  set on both paths.
- **Self-check gating.** The raster/training feature self-check is correctly
  skipped for gate-off recipes — it would otherwise print a guarantee about code
  the deployed map never runs.
- **`_nan_window`** reproduces `nanmean`/`nanstd` (population, ddof=0) including
  the even-window `origin=-1` tie-break and the `NaN < CUT → False` built-fraction
  convention.
- **Fold class coverage.** All 9 coarse3 classes are present in all 5 training
  folds, so no fold-local class list is short.

**One latent fault hardened.** [optimise_s2off.py:246](src/optimise_s2off.py#L246)
placed the fine-probability block into the OOF cache **positionally** while
placing the merged block by name. `fine_classes_` is `sorted(set(y))` and so is
fold-local; a fold missing a rare transition would have shifted every column to
its right and averaged different classes together — silently. It does not fire
today (§4, fold coverage), but it is exactly the failure the small end of a
learning curve provokes, which is why `learning_curves.py` places every block by
name. Added the same guard `ablate_s2_architecture.py` already carried.

### Optimisation

No further optimisation is worth taking. The serving path was already reduced
66.2 s → 6.5 s at Oslo scale (band-major AlphaEarth, no per-model re-stack, no
DataFrame, on-device standardisation) and the MC gate is marginalised exactly in
2 passes instead of sampled in 16. Training is 1.6 s per fit on one GPU. The
remaining named lever — halving the hard-coded 1024/512 tower widths, ~2.1× — is
a *training*-time saving on a stage that costs seconds, and it would change the
model. Not worth it.

---

## 5. Known limits

- **Oslo has zero labelled plots inside the AOI.** Nothing about that map can be
  scored. Structure metrics, IoU and the within-AOI counterfactual are all that
  is available.
- **Map self-reproducibility is ~0.84 change-class IoU** across disjoint seed
  draws. Compute that floor before reading any two-map disagreement as real.
  Change-pixel counts move ±5% between seed blocks — the deployed model gave
  16,676 px on seeds 0–4 and 15,841 px on seeds 5–9, a 5.0% swing on an
  identical recipe.
- **The coarse3 legend over-promises.** See §3.1. Report the merged2 read for
  *whether*, and state the two dead classes when showing the nine-class map.
- **Sentinel-2 buys nothing measurable on plots.** `baseline_aef` (no S2 at all)
  scores change-F1 0.6574 against this model's 0.6557 at 15 seeds — inside the
  spread. The 78-column tower was selected on the user's visual read of the map,
  which is a legitimate arbiter given the plot metrics tie, but the report should
  not claim a quantitative gain from Sentinel-2.

## 6. What to do next

1. **Label more plots.** §3.2 prices it: +0.026 change-F1 per doubling, ~7× the
   seed spread, ~5× the total yield of the architecture search. This is the only
   lever left with a measurable return.
2. **Stratify that labelling on the dead classes and inside Oslo.** 46 and 114
   plots is what "F1 = 0" costs; and plots inside the AOI would simultaneously
   settle `TWOTOWER_RESEARCH.md` G4 — whether the gate's change suppression is a
   fix or a fault — which is still open and still unanswerable from here.
3. **Do not reopen the model choice.** See `CLAUDE.md`.
