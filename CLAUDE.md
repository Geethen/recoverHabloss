# recoverHabloss — agent notes

## THE FINAL MODEL (settled 2026-07-27, by the user, on visual inspection)

**`s2off_centre_m3s3_bf`** in `src/infer_s2.py:fit_models`. Do not re-open this
choice, do not substitute a different recipe in a comparison "for convenience",
and do not quote an older model as the deployed one. The earlier candidates
(`mc_s2_drop0.7`, `aef_builtfrac`, `baseline_aef`, `mc_dropout_scalars`,
`s2off_deploy`, `s2off_slim`) are **superseded** — they survive in the code and
the ledger as the record of how this was arrived at, not as options.

Reproduce the deployed map:

```bash
# geo pixi env; the project .venv is broken and `uv run` does not work
/home/geethen.singh/.pixi/envs/geo/bin/python src/infer_s2.py \
    --aois oslo --models s2off_centre_m3s3_bf --seeds 5
```

**What it is.** A hierarchical two-tower network over AlphaEarth 2018/2024
embeddings, trained with a Sentinel-2 detail tower as *privileged information*
and **served AlphaEarth-only with the detail gate forced off**. Sentinel-2 is
never read at inference — the composite fetch and the sliding-window features are
skipped entirely, and `probs_aef_only_matrix` never builds or runs the detail
tower.

**Detail tower = 78 columns.** Seven Sentinel-2 channels (blue, green, red, NIR,
NDVI, NDWI, brightness) at the plot centre, plus their mean and standard
deviation in a 3×3 window, plus built fraction at five radii — for 2018, 2024 and
their difference. 7×3 + 5 = 26 per year, ×3 = 78. Defined in
`twotower_lab.S2_SUBSETS["centre_m3s3_bf"]`; that dict is the single definition,
shared by the training ladder and the inference recipes so they cannot drift.

**Two output reads, and they are not interchangeable.** `*_merged2.tif` is the
four-transition map and is the read for **whether** change occurred;
`*_coarse3.tif` is the nine-transition map and answers **what kind**. They
disagree on ~15% of change pixels because arg-max does not commute with the group
sum — this is expected, not a bug (S15). Class codes follow the *sorted* class
list, never the palette order: read the code→label mapping from the `.qml`
sidecar. Getting this wrong once counted stable Vegetation as change.

## Before you compare two maps

**A 5-seed ensemble reproduces itself at only ~0.84 change-class IoU** across
disjoint seed draws (0.8402 merged2 / 0.8362 coarse3 for the full-feature model
on Oslo). Compute that floor before reading any disagreement between two maps as
a real difference. Doing this wrong in the obvious direction — comparing model A
against model B across seed blocks without asking what each does against itself —
produced a false "this subset does not replicate" verdict in S18. Compare each
model's **self**-IoU first.

Change-pixel counts are a usable run fingerprint but are themselves a draw: they
move ±5% between seed blocks. Quote paper numbers from a replicated run.

## Environment

- Use `/home/geethen.singh/.pixi/envs/geo/bin/python` for anything with torch.
  The checkout's `.venv/` is empty, so `uv run` fails even though the README is
  written that way.
- pandas is 3.x: parquet round-trips floats as nullable extension dtypes that
  reach sklearn as object arrays. Cast with `.astype("float64")` before fitting.

## Where the reasoning lives

`docs/research/` is a ledger, not documentation — every entry is an experiment
with its verdict, including the negative ones. Read the foot of
`TWOTOWER_RESEARCH.md` before proposing an idea; the tested-negative list is long
and specific (MoE, noise injection, G-H sampling, distillation, endpoint
supervision, guided filtering, dot/normalised-difference features).

- `S2_DETAIL_RESEARCH.md` — the Sentinel-2 line and the final model (S16–S18).
- `../land_cover_change_model_report.md` — the shareable, collaborator-facing
  write-up of the deployed model (was `COARSE3_METHODS.md`). Rewritten
  2026-07-28 against `s2off_centre_m3s3_bf`; it is current, it is sent outside
  the project, and it must stay free of repo-internal names. Keep it in step if
  a number moves.
- `AUTORESEARCH.md` — the rules the ledger is kept under. **3 seeds minimum
  before any verdict, 5 before calling a win.** Sub-1pt differences at 1 seed are
  noise, and verdicts on this model have reversed between 3 and 5 seeds, between
  5 and 15, and between seed blocks.

## Things that are settled and cost time to relitigate

- The AlphaEarth `diff` block is **not** redundant despite being a linear
  function of the 2018/2024 blocks — removing it costs −0.048 change-F1.
- 30 epochs is a floor, not a budget: 20/15/10 give −0.028/−0.044/−0.048.
- `tower_dim` is not the serving cost; the hard-coded 1024/512 hidden widths in
  `_TwoTowerTrunk.tower` are.
- Spatial smoothing of the output raster — gates, fusion, guided filtering —
  removes change pixels first, every time. Fix inputs or uncertainty instead.
- Oslo has **zero** labelled plots inside the AOI. Nothing about that map can be
  scored; structure metrics and IoU are all that is available there.
