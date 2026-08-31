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

The feature table now carries **eleven** 10 m channels (S19 added EVI2, GRVI,
BSI, CI) and a subset is a choice on three axes — families, channels, years.
`s2_subset_columns` defaults the channel filter to `CHANNELS_BASE` and the year
filter to all three blocks, which is the *only* thing holding this recipe at 78
rather than 114. Anything reading the whole stat block instead of a named subset
must call `s2_base_columns`. `tests/test_s2_subset_axes.py` guards both.

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
- **The GPU no longer works from that env (2026-08-14).** The card is an RTX Pro
  6000 Blackwell (**sm_120**); geo's torch 2.5.1+cu121 covers sm_50–sm_90 only.
  `torch.cuda.is_available()` still returns True, so `infer_s2.py` selects `cuda`
  and dies on the first op with *"no kernel image is available"* — **the reproduce
  command above fails as written**. Set `CUDA_VISIBLE_DEVICES=""` for a slow CPU
  run, or use the cu128 torch at `/home/geethen.singh/.cache/phoenix-test/venv`
  (2.11.0+cu128, built over geo with `--system-site-packages`). That env needs
  PROJ pointed at geo's data or every rasterio CRS lookup dies with *"The EPSG
  code is unknown"*; with it, a 5-seed Oslo run takes ~90 s:

  ```bash
  G=/home/geethen.singh/.pixi/envs/geo
  PROJ_DATA=$G/share/proj PROJ_LIB=$G/share/proj GDAL_DATA=$G/share/gdal \
  /home/geethen.singh/.cache/phoenix-test/venv/bin/python src/infer_s2.py \
      --aois oslo --models s2off_centre_m3s3_bf --seeds 5
  ```

  Fixing geo itself is still the real repair.
- **Never rewrite a large parquet at its existing path.** `/data/P-Prosjekter2`
  is CIFS. Rewriting `s2_features_habloss_recover.parquet` in place left the path
  permanently unreadable — `open`/`stat`/`unlink` all return EIO while `ls` still
  shows it — twice, and an identical file under a new name was fine. The S2 table
  is therefore `s2_features_habloss_recover_10m.parquet` (S19) with the
  seven-channel one kept as `..._c7_backup.parquet`. Write a new name.
- pandas is 3.x: parquet round-trips floats as nullable extension dtypes that
  reach sklearn as object arrays. Cast with `.astype("float64")` before fitting.

## The labelling deployment — blank in the repo is CORRECT

`app/apps_script/Code.gs` and `app/config.js` are **source, not the running
deployment**, and reading them to decide whether the campaign is configured is
the mistake this section exists to stop. The script that answers `/exec` is a
*copy* pasted into the Sheet's Apps Script editor, and the live values are not
in this repository and must not be:

- `var SUBMIT_TOKEN = ''` in `Code.gs` is the correct committed state; the
  deployed copy carries the real string. The repo is public.
- the Earth Engine **service-account key** lives in that script's *Project
  Settings ▸ Script Properties* as `EE_SERVICE_ACCOUNT_KEY`. It is never in git
  and cannot be in the page — the SDK refuses a browser-side private key.
- the `/exec` URL, submit token and expert roster reach the app from
  `app/config.js` locally and from the `LABEL_APP_CONFIG_JS` Actions secret on
  Pages, which `.github/workflows/pages.yml` injects at build time.

**So a blank token or a missing key in these files says nothing about the
deployment. Ask the deployment, which is the only thing that can answer:**

```bash
curl '<exec-url>?action=ping'     # -> "token_required":…, "ee_service_account":…
/home/geethen.singh/.pixi/envs/geo/bin/python src/check_ee_service.py \
    --url '<exec-url>'            # 6 steps: mint, compute, maps, tile, dw24
```

`check_ee_service.py` exits non-zero unless every step passes; the `/exec` URL
and the submit token are in `app/config.js`. **Verified green 2026-08-28**:
`habloss-labelling@ee-gsingh.iam.gserviceaccount.com` mints, computes, creates
maps and renders `dw24` tiles — but only after `roles/earthengine.writer`,
because `roles/earthengine.viewer` grants `earthengine.computations.create` and
**not** `earthengine.maps.create`. Note what writer widens: the token this app
hands to every browser can then write assets in `ee-gsingh`, which holds the
RECOVER, nyvest and EWT folders. A custom role of `computations.create` +
`maps.create` restores the bound `Code.gs` assumed — every panel layer is a
public dataset, so the account needs no asset access at all.

## Where the reasoning lives

`docs/research/` is a ledger, not documentation — every entry is an experiment
with its verdict, including the negative ones. Read the foot of
`TWOTOWER_RESEARCH.md` before proposing an idea; the tested-negative list is long
and specific (MoE, noise injection, G-H sampling, distillation, endpoint
supervision, guided filtering, dot/normalised-difference features).

- `S2_DETAIL_RESEARCH.md` — the Sentinel-2 line and the final model (S16–S18).
  **S19 closes the channel axis**: four more 10 m indices (EVI2, GRVI, BSI, CI)
  are flat at 15 seeds and invisible on the map. What *does* move is the year
  axis — a detail tower given only the 2024−2018 differences drops stable
  built-up-read-as-vegetation by 0.012 in 14/15 paired seeds, in all four
  diff-only arms and no state-carrying arm — but it is a **ratio trade** (stable
  vegetation read as built-up rises) and, when built fraction is kept as a state
  to get the most of it, costs 18% of map edge density. `diff10_centre_m3s3_bf`
  (38 cols) reproduces the deployed map inside its own seed floor and is the
  arm to reach for if the detail block ever has to shrink.
- `CONFORMAL_TORCHCP.md` — the TorchCP method comparison. Two facts before
  touching conformal anything: **ECE/Brier/CRPS cannot distinguish conformal
  methods** (measured sd exactly 0.0 across 96 score×predictor×alpha cells — they
  score the posterior, which a set constructor does not touch), and **the
  marginal `SplitPredictor` covers `Cropland -> Nature` 13% of the time at a
  nominal 90%** while its marginal coverage reads a perfect 0.8999. LAC+Mondrian
  is confirmed as the right choice and reproduces `nested_conformal` to four
  decimals; Clustered and RC3P are tested-negative at K=9.
- `SPECIALIST_SPLIT_RESEARCH.md` — section V, and the one place N0's "that class
  is a labelling ask" is amended: a specialist trained on the four rare classes
  alone, composed with the base by the chain rule, takes `Artificial -> Cropland`
  from 0.000 to F1 0.314 at precision 0.52, on two bases, with a flat free
  control. Costs `Artificial -> Nature` recall and a second network at serving.
- `NOISY_LABEL_RESEARCH.md` — section T, closed and negative throughout. Before
  proposing any sample-selection method (co-teaching in any form, DivideMix,
  JoCoR): a small-loss criterion ranks **rarity**, not noise, on this target —
  a 10% forget budget drops 76% of the 46-plot transition and 1.3% of the
  4,200-plot one — and correcting that still loses to dropping rows at random.
- `../land_cover_change_model_report.md` — the shareable, collaborator-facing
  write-up of the deployed model (was `COARSE3_METHODS.md`). Rewritten
  2026-07-28 against `s2off_centre_m3s3_bf`; it is current, it is sent outside
  the project, and it must stay free of repo-internal names. Keep it in step if
  a number moves.
- `STATE_PRETRAIN_RESEARCH.md` — the pretraining *phase* on its own, validated
  LLTO (`src/statepre/`). Three things to know before touching it: **`block` is
  not a substitute for a location+spatial holdout here** — it reads 0.024 high
  and reverses the external-vs-endogenous ordering; **`siam_state_source="both"`
  wins the state read by +0.023 and LOSES the transition read by −0.019
  change-F1** (P7i, 5 seeds), so the pretrain pool stays `"external"` and this
  section is the standing example of a plot-level gain that does not transfer;
  and **the dataset axis is closed** (section V) —
  LUCAS is negative because it returns 66% of RECOVER's Cropland and 50% of its
  Artificial as Nature, and no reweighting, density control or fold count
  recovers it. Fold geometry is now cut on a fixed reference cloud
  (`fold_ref="reference"`); rows written before 2026-07-31 are `"union"` and are
  not paired against the newer ones.
- `PATCH_SAMPLING.md` → `ACTIVE_LEARNING.md` — the labelling campaign.
  `PATCH_SAMPLING` sizes round two at ≈1,250 patches; `ACTIVE_LEARNING` is the
  design **plus sections AL0–AL5**, the replay lab (`src/al_lab.py`, 25 arms,
  5 seeds × 5 spatial folds, read paired by `src/al_report.py`).
  **The paired floor on change-F1 is 0.016 — measure it before believing any
  acquisition result, most of them are smaller than that.** Four settled
  verdicts: the acquisition function is worth ≤ +0.02 change-F1 and decays to
  nothing by a 3,000-plot start, so **it is not the lever — more labels is**
  (+0.026 per doubling); **coverage and change-F1 are decoupled** (AL3 recovers
  1,202 withheld steep/bare/wet plots and moves change-F1 by nothing);
  **one-shot uncertainty sampling is worth exactly zero** and needs ≥4 batches
  (AL4); and chasing rare-class *retrieval* wrecks the model (`proto_sim`
  −0.046). Also: `Artificial -> Cropland`'s posterior never exceeds 0.191 in
  21 M pixels, so every model-in-the-loop score is blind to it by construction.
  Metrics in `src/acquisition.py` (numpy only, no torch). **§AL7 is the
  labelling instrument** — `app/label_app.html` (MapLibre + Esri Wayback, one
  file, Google Sheet backend), `src/build_label_batches.py`,
  `src/build_batch_evidence.py`, `src/label_rounds.py`, `app/README.md`. Batch
  size 100 and no schedule parameter are AL4/AL5, not defaults; the posterior
  renders collapsed because the errors being fixed are confident ones; `random`
  is a first-class channel because the 2x falsification test needs an equal-area
  arm in the same round.
  **Calibration batches come before any real labelling**, in two stages
  (`--stage teach` then `--stage qualify`) — reference answers never shown
  before the call, agreement read per **expert** *with the confusion pairs*,
  which is the only defence against three people labelling to three standards.
  **§AL8 made it safe for two experts**, and four of its findings are load-
  bearing. The annotation key is `(campaign, batch_id, point_id, expert_id)` end
  to end, and `expert_id` comes from the roster in `config.js`: a typed name in
  that key makes "Ann"/"ann"/"Ann " three experts and the agreement number reads
  a clean 100% over nothing. `tests/test_label_app.py` mocked the *documented*
  key while `Code.gs` implemented a shorter one, so the suite was green while
  production overwrote every second reading — the tests for this parse `Code.gs`
  itself, and a Python double cannot police a contract the JavaScript disagrees
  with. Overlap is a property of the batch file (`--experts e1,e2`), never a
  checkbox. Evidence is **baked at build time** because hosting is static and the
  loop must work with Earth Engine never signed in; only the S2 chips are live,
  and `growingSeason()` in the app must keep mirroring `growing_season()` in the
  builder or the filmstrip and the chart are two different seasons. **Every
  dataset in the UI shows its end year** — that rule is what caught Hansen
  `v1_11` (`lossyear` stops at 23) answering a question about 2024.
  Two Earth Engine facts the builder cost three failed runs to learn:
  **EE evaluates in tiles**, so spatially clustered points batch and
  widely-spread ones must be mapped over independently — this draw is global, so
  batching 100 *or 20* of its points into one S2 request is `User memory limit
  exceeded` either way, and lowering the chunk does not fix a cost that is the
  spread. The unit is one request per (point, year); budget a 100-point bake at
  **~36 min**, not the ~2 the per-request timing predicts (EE rate-limits
  `getInfo`, so the thread pool buys little). And **`reduceRegions` names a
  single-band output `first`**, not by band name — that silently returned "no
  data" for six of the sixteen point-value rows. The Earth
  Engine panel is `earthengine` scope only and passes the
  project as `ee.initialize`'s **sixth** argument: `ee.data.setProject` does not
  exist in the SDK, and `cloud-platform` drags the app into restricted-scope
  verification for capabilities it never uses. It asks whoever signs in for the
  **project only**; the OAuth client id is a per-deployment constant with a box
  in the panel and a home in `config.js`, and **it cannot be borrowed** — not
  from GeoLibre's published default, not from anywhere. A Web client validates
  the JavaScript origin exactly, and an unregistered origin comes back as
  *silence*: the SDK sets no `error_callback`, so Google prints
  `origin_mismatch` inside the popup and it reads exactly like a blocked one.
  **§AL9 is the chip-latency and evidence work**, and two of its findings stop a
  re-litigation. The DIST-ALERT inspector this panel was ported from is **not
  fast on a cold pixel** — `c2c_ts_server.py`'s own docstring says "nothing is
  computed before a click; the cache fills lazily". It feels fast because
  somebody already paid: a previous click, or `scripts/warm_ts_cache.py` walking
  the pixel list **through the running server** before a review session. So the
  target is "do not make it a first look", and the technique — pay Earth Engine
  before the human does — is available to a static app and is what the prefetch
  now does. Measured: one mosaicked filmstrip request is **negative** (loses
  on 4 of 5 points — EE parallelises nine separate requests better than one big
  one), a cheaper cloud mask is inside the noise, and both arms of everything
  pile up at a ~35 s **concurrency throttle**. What works is a 12-scene cap
  (wins 9/10 points, 216 s vs 323 s, for a median 0.002 relative reflectance
  difference at the plot), prefetching the image **bytes** rather than only
  minting the URL, and colouring the chart dots and chip cells from the
  reflectance the batch **already bakes** — which needs no Earth Engine at all.
  Also: `.body` was both the flex layout row and the legend's inner text, so
  "What counts as what" had been rendering as three squashed columns since the
  layout landed; and class *names* use the `-ink` colours, never the 9 px swatch
  ones. ESRI annual land cover now runs to **2025** and `EVIDENCE_VERSION` is
  `ev2`, so batches baked before 2026-08-27 carry the older row set.
  **`src/build_batch_chips.py` is the `warm_ts_cache.py` analogue**: it bakes
  each point's nine years into one sprite (~3 MB per 100-point batch per scheme)
  so a point costs one static file and no Earth Engine. Note the inversion — one
  mosaicked request LOST live, because EE parallelises nine requests better than
  one, and is exactly right baked, because a static file has no scheduler.
  Everything degrades to the live path (no bake, unbaked scheme, different
  width, unknown `version`, 404), and `parseBatch` is an **allow-list**, so any
  new batch-level key must be added to it or it is dropped silently.
  **§AL10 is the display ramp**, and it is the answer to "why is the chip one
  colour". The fixed `min:0,max:3500` is one ramp for a **global** draw and it
  saturates: 55 of 900 cells more than half-saturated, 38 more than half-floored,
  101 more at sd<6, median 98th percentile **252 of 255**. The ramp is now
  measured per point (`chips.stretch`, p2–p98) and read by **both** sides —
  `combo_bounds()` in Python renders the sprite, `comboBounds()` in JS paints the
  tint and mints the live request; `tests/test_chip_ramp.py` runs the JS in node
  against the Python. Two tested-negative: **per-band bounds** (a decorrelation
  stretch — moves hue, which the legend teaches as a convention, and turned
  green fields magenta) and any **narrow** ramp (turns inter-year atmospheric
  drift into full-scale colour swings — an unchanging desert cycled through six
  colours; a filmstrip that manufactures change is worse than a flat one). One
  affine transform on all three channels, shared by the nine years. Also §AL10:
  all four three-band schemes are baked, the **whole batch** is warmed rather
  than a window (a window is a *quota* compromise and the wrong shape for a
  static file — it never covers stepping backwards), and the **dense series is
  baked too** (`src/build_batch_dense.py`, ~10 KB/point sidecar) and on by
  default. The panel now leads with the spectral profile and the index series,
  and the other products are **collapsed at the foot**: the labels are training
  data for a 10 m model, so twenty-three rows of somebody else's confident
  classification at the top of the panel is an anchor.
  **The same silence had two more instances, both now closed.** A re-bake writes
  new pixels to the SAME paths, so `chips.built`/`dense.built` is stamped on the
  URL as `?v=` — without it a warm browser or CDN serves the old sprites and it
  looks exactly like the re-bake failing. And six conditions drop the chip strip
  to live EE, all producing an identical strip of flat tints: `chipBakeMiss()`
  names which. For the **Earth Engine overlays**, `getMapId` validates the
  *request* — a bad band name, an empty collection, a memory limit or a rotated
  token all mint fine and fail per **tile**, which MapLibre reports on
  `map.on('error')` and the app sent to `console.warn` and nowhere else.
  `eeTileError()` reports it and `eeTileReason()` GETs one tile to recover EE's
  own sentence (MapLibre loads rasters as images, so `AJAXError.body` is empty).
  Note when debugging that `dwbuilt`/`obtemporal` are **masked differences** — a
  transparent result is the correct answer at a stable point; use `dw18`/`dw24`
  to test whether overlays work at all.
- The two stable-class map errors — mountains read as `Artificial -> Artificial`,
  wetlands as `Cropland -> Cropland` — are diagnosed in `ACTIVE_LEARNING.md`
  §AL-T. **The mountain one is a bare-ground error, not a slope error**: the
  misread rate *falls* with slope (0.096 at 0–1° → 0.022 above 12°) and rises
  3.1× on WorldCover `bare`, where the model is more confident when wrong. Bare
  land is 17.4% of the globe and 6.6% of the label set. `change_f1` cannot see
  either error — both sides of both are stable classes.
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
