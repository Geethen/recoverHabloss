# Handoff: add the coarse3 (9-transition) output to `infer_s2.py`

> **STATUS: DONE (2026-07-27).** Implemented, verified and mapped. Methods
> write-up: `../land_cover_change_model_report.md`. Ledger entries: S13–S15 in
> `S2_DETAIL_RESEARCH.md`.
>
> Two of this document's predictions were wrong and are corrected there:
> 1. The coarse3→merged2 aggregation agrees on **99.19%** of valid pixels, not
>    "~100%". Arg-max does not commute with the group sum, so exact agreement
>    was never implied. On the change class the disagreement is ~15%.
> 2. The single-seed fingerprint could not be reproduced because the MC gate is
>    no longer sampled — it is marginalised exactly (2 passes, not 16), which is
>    faster, deterministic and slightly more accurate. The 5-seed **15,532**
>    fingerprint *was* reproduced bit-exactly under `--mc-sampling` before the
>    change; the exact path gives 15,517 (−0.10%).

**Task.** `src/infer_s2.py` maps Sentinel-2 models over an AOI but writes only
the **merged2** read (4 classes: Vegetation/Artificial × 2 endpoints) plus a
binary change mask. The user wants the **coarse3** read as well — the informative
9-transition legend (Nature/Cropland/Artificial × 2 endpoints). Add it.

## The key fact: coarse3 is already computed and thrown away

The model is hierarchical. `HierarchicalSoftmaxNN._probs(frame)` returns
**`(p_fine, p_merged)`** (`model_zoo.py:1853`), where `p_fine` *is* the coarse3
posterior and `p_merged = p_fine @ self._M` is the aggregation to merged2.

`infer_s2.predict()` calls `_probs` and keeps only `merged` — `p_fine` is
discarded on every forward pass. **So this is not new inference, it is retaining a
value already being produced.** Do not re-run the model twice.

- Class list: `model.fine_classes_` — 9 entries, verified present.
- Palette: `CLASS_COLORS` (from `infer_cities`), **already imported** in
  `infer_s2.py:55`, and it covers all 9 coarse3 classes (checked — none fall
  back to grey).
- Writer: `write_class_raster(path, codes2d, geobox, classes, colors)` from
  `infer_twotower` — paletted uint8 + overviews + `.qml` sidecar. Already imported.
- Precedent to copy: `infer_twotower.py:326-333` writes exactly this raster.

## What to change

In `infer_s2.py`:

1. **`predict()`** — accumulate a second array for the fine posteriors beside
   `acc`, using `p_fine` instead of discarding it. Divide by
   `passes * len(members)` the same way. Return `(probs, classes, fine_probs,
   fine_classes)`, where `fine_classes = members[0].fine_classes_`.
   The existing guard that all seed members agree on `merged_classes_` must be
   extended to `fine_classes_` too — averaging across differently-ordered class
   vectors silently permutes classes.
2. **`run_aoi()`** — after the merged2 raster, arg-max the fine posteriors, map
   to codes, and `write_class_raster(out_dir / f"{name}_{model_name}{suffix}_coarse3.tif", ...,
   fine_classes, CLASS_COLORS)`. Apply the same `aef_valid` masking and `NODATA`
   convention as merged2.
3. Optionally extend `--save-probs` to write the fine stack too, with band
   descriptions set from `fine_classes` (see point 3 under Gotchas).

## Gotchas that have already bitten this codebase

1. **Class codes are NOT the palette's insertion order.** `write_class_raster`
   numbers codes from the *sorted class list you pass it*. Anything reading the
   raster back must take the code→label mapping from the `.qml` sidecar
   (`refine_map.labels_from_qml`), not from `CLASS_COLORS` order. Getting this
   wrong once counted stable Vegetation as change (1.85 M px instead of 13,300).
2. **Thread new flags all the way through.** A `--save-probs` flag was added with
   its argparse entry and its write block correct, but the *call site* was not
   updated, so it silently defaulted to `False`, the run succeeded, and nothing
   was written. Grep the flag name and confirm every hop.
3. **Match posterior bands to classes by band description, not position.**
4. **Keep `--self-check` on.** It compares the raster feature path against the
   training feature table per plot and has caught two real train/serve skews
   (built-fraction NaN convention; even-window centring). It must pass (~1e-4)
   before any map is trusted.

## How to run

> **Superseded (2026-07-27).** The deployed model is now
> **`s2off_centre_m3s3_bf`**, served AlphaEarth-only with the Sentinel-2 gate
> off — see the FINAL MODEL section of `S2_DETAIL_RESEARCH.md` and `CLAUDE.md`.
> The command and the runtimes below describe `mc_s2_drop0.7`, kept because the
> numbers quoted in this handoff were produced with it.

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python
cd src
$P infer_s2.py --aois oslo --models s2off_centre_m3s3_bf --seeds 5   # current
$P infer_s2.py --aois oslo --models mc_s2_drop0.7 --seeds 5 --save-probs  # historical
```

The current model takes **~1 min** for the 5-seed Oslo ensemble: it reads no
Sentinel-2 at inference, so the composite fetch is skipped entirely, and the
serving path was rewritten in S17 (66 s → 6.5 s of compute). The historical
figures for `mc_s2_drop0.7` were ~45 min at 5 seeds with composites dominating.
Output lands in `data/inference/s2_<timestamp>/`.

## Verification before reporting

- Self-check line reads `passed` with disagreement ~1e-4.
- The coarse3 raster's **merged2 aggregation must reproduce the merged2 raster**:
  collapse Nature+Cropland → Vegetation on the coarse3 codes and compare against
  `*_merged2.tif`. They should agree on ~100% of valid pixels. This is the
  strongest available check that the fine head was read correctly, and it is
  cheap — do it.
- Change-pixel count is a determinism fingerprint *within* a fixed model and seed
  set: `mc_s2_drop0.7` single-seed Oslo is **13,300**, 5-seed **15,532**;
  `s2off_centre_m3s3_bf` 5-seed (seeds 0–4) is **16,676**. Reproducing one of
  those exactly confirms the pipeline is deterministic. It is **not** a stability
  measure across models or seed draws — the count moves ±5% between seed blocks
  (S18), so never read a difference between two models' counts as a result
  without the self-IoU floor beside it.

## Context worth knowing

- **Superseded:** `mc_s2_drop0.7` was the model the user preferred on the map at
  the time this handoff was written. The coarse3 detail it asked for is what
  eventually settled the choice — the final selection (2026-07-27) was made on
  the *coarse3* raster of `s2off_centre_m3s3_bf`.
- Existing outputs: `data/inference/s2_20260727_024652` (single seed) and
  `s2_20260727_093450` (5-seed ensemble). Both carry S2 RGB/NIR backdrops.
- Full research record and every verdict: `src/S2_DETAIL_RESEARCH.md`.
- **Do not spatially post-process the map.** Guided filtering, at class or
  posterior level, removes 11.7–22.2% of the change class; change pixels sit a
  median 0.155 from flipping, so any neighbourhood vote deletes them. Details in
  `S2_DETAIL_RESEARCH.md` section U.
