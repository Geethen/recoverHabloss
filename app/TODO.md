# Labelling app — work plan

**Phases 0–3 of this plan were implemented on 2026-08-26**, and the evidence /
chip-latency work in **§AL9 on 2026-08-27**. What each of them
turned out to be, and which of the faults were silent, is recorded in
[`ACTIVE_LEARNING.md` §AL8](../docs/research/ACTIVE_LEARNING.md) — the ledger,
not here. This file is now the *remaining* plan plus the decisions that still
bind whoever picks it up.

Hand any task to an agent cold. Every task names the files it touches and what
"done" means.

## Decisions already made — do not reopen

| | |
| --- | --- |
| **Hosting** | Static only (Cloudflare R2 is fine). No always-on service. Nothing in this plan may require one. |
| **Label store** | Google Apps Script + Sheet, fixed in place. Not SQLite. |
| **Evidence delivery** | Point values and the annual index timeline are **baked into the batch JSON at build time** (`src/build_batch_evidence.py`); the **dense series** is a per-point sidecar (`src/build_batch_dense.py`, §AL10) and on by default. Sentinel-2 chips are **baked sprites** with live browser Earth Engine as the fallback. The panel leads with the spectral profile and the index series; the other products are **collapsed at the foot**, because the labels are training data for a 10 m model and somebody else's classification at the top of a panel is an anchor. |
| **Identity** | The annotation key is `(campaign, batch_id, point_id, expert_id)`, end to end. `expert_id` comes from the roster in `config.js`; `labeller` is a display name and is never keyed on. |
| **Overlap** | A property of the batch file (`build_label_batches.py --experts`), never a checkbox. Adjudication is still deferred — see below. |
| **Blinding** | `rank` / `score` / `channel` hidden until the point is saved. Descriptive `meta` visible — §AL-T's coverage gap is *why* those points are in the batch and the interpreter should see it. |
| **Dataset vintages** | **Every dataset in the UI shows its end year.** That rule is what caught Hansen `v1_11` (`lossyear` stops at 23) answering a question about 2024, and GHSL's 2015→2020 epochs answering a 2018→2024 one. It cuts both ways: it also caught ESRI annual land cover having moved *forward* to 2025 while the builder still asked for 2023 (§AL9). |
| **Chip speed** | **A cold chip costs what it costs, in the inspector too.** `c2c_ts_server.py` computes nothing before a click; it feels instant because a previous click or `scripts/warm_ts_cache.py` already paid Earth Engine for that pixel. So the target is *stop it being a first look* — §AL9 measured the alternatives (one mosaicked request: **negative**; cheaper cloud mask: noise; both arms plateau at a ~35 s concurrency throttle) and the levers that work are a scene cap, prefetching the **bytes**, and colouring what is already baked. Do not re-open this by proposing a different request shape. The `warm_ts_cache.py` analogue is **built**: `src/build_batch_chips.py` bakes the default scheme's filmstrip to one sprite per point (~4 MB / 100 points), after which a point costs one static file and no Earth Engine. Everything still degrades to the live path, and three tests hold that. **§AL10** then found the remaining "the chips are one colour" was the **display ramp**, not the wait: the fixed `0-3500` saturated a quarter of a global draw, so the ramp is now measured per point and shared by the three channels *and* the nine years — per-band and narrow ramps are tested-negative, they move hue and manufacture change. All four three-band schemes are baked and the **whole batch** is warmed, not a window. |
| **The vis scheme** | One setting, and NDVI/NDMI/NBR are defined **once**, in `CHIP_INDEX`, driving the chip pixels, the chart dot colours and the plotted series. Two band lists for one word called NDVI is the `growingSeason()` hazard one level down. |

**Consequence to hold onto:** because chips stay live-EE, Earth Engine sign-in is
on the critical path for imagery evidence *only*. The baked timeline, point
values and the whole labelling loop must render and work with Earth Engine
never signed in. Losing chips is acceptable; losing the app is not.

**Second consequence, from §AL8:** `growingSeason()` in `label_app.html` and
`growing_season()` in `build_batch_evidence.py` must stay identical. The chart is
baked with one and the chips are rendered live with the other; if they diverge,
every disagreement between the filmstrip and the chart is an artefact and looks
exactly like real change.

---

# Phase 4 — what is left

## T4.1 — Adjudication

**The thing that actually fixes `Cropland`/`Nature` drift.** After both experts
submit, reveal both calls with provenance and store a *third* consensus record
linked to the originals. Corrections are **revisions, never destructive
replacements** — the two original readings are the agreement measurement and
deleting either of them deletes it.

It runs on real disagreements rather than exercises, so **schedule it as soon as
the pilot produces some**. The parts already in place for it: the sheet keys on
the expert so both readings survive; `label_rounds.py` already reports which
pairs disagree; the batch file already names `required_readers`.

Do not build it before there is a disagreement to look at. §AL5's lesson applies
— a hand-built workflow that ties the simple one is a workflow that was not
needed.

## T4.2 — SQLite/WAL store

A real `UNIQUE(campaign, batch, point, expert)` constraint, with Sheets kept as
an async export mirror. **Only if** the campaign goes past two experts, or if
Apps Script latency is *measured* as a problem after the T3.8 work (the read
caches, the incremental key index, the 10 s sync debounce). `?debug=1` prints
save-acknowledgement p50/p95; read it before deciding.

## T4.3 — Evidence that is Europe-only or too heavy for the main loop

* **RADD alerts / forest baseline / pixel query.** Europe-only. Must show
  **"outside coverage"** elsewhere — the rule from T2.4, for the same reason: a
  layer with no data must not look like "no disturbance".
* **LandTrendr / VeRDET break dates** as optional change-*date* evidence, shown
  only after the expert has recorded an initial visual draft, and never as a
  suggested label. Five model opinions beside the buttons is the anchoring §AL7
  built the collapsed disclosure to prevent; this is the same hazard on a timer.
* **CCDC / COLD / S-CCD and full spectral diagnostics.** Adjudication only —
  too slow and too heavy for the main loop.

## T4.4 — Typography and density

**Partly done in §AL9**, on the block that was worst: "What counts as what" was
laid out as three squashed columns (a `.body` class collision with the flex
layout row) with the class names set in the 9 px *swatch* colours, which put
"Cropland" at a 2:1 contrast ratio. That block is now 12.5 px with `-ink` name
colours, and the evidence values are 11.5 px in four labelled groups.

**What is left:** base explanatory text elsewhere is still 10.5 px, and there is
still no comfortable-density option. These are multi-hour expert sessions and
the current sizing was chosen against a 1366×768 laptop, not against a working
day.

## T4.5 — Keyboard gaps

`W` toggles Wayback, `[` / `]` step the release, `/` focuses notes, `F` fits the
cell. Point navigation is still on `←` / `→`; the chip lightbox already takes
those keys while it is open, so the remaining question is whether to move point
navigation off them permanently.

## T4.6 — Nice-to-have reads

Campaign-level progress in the header; strip dots coloured by transition so a
run of identical calls is visible; median seconds/point on screen; an
end-of-batch review table for a final scan pass.

---

## If you are changing the app, read this first

Three things in §AL8 were **silent** faults — the app looked correct while
losing the measurement — and each has a guard that is easy to remove by
accident:

1. **`tests/test_label_app.py` parses `Code.gs` directly** for the four-field
   key and for the `labelled` endpoint's contents. It does that because the
   Python mock of the sheet implements the key the *docs* describe, so it stayed
   green for as long as `Code.gs` implemented a shorter one. Do not "simplify"
   those tests into assertions against the mock.
2. **`action=labelled` returns `{point_id, expert_id}` and nothing else.**
   Not the transition, not the notes, not even the display name. Showing a
   second reader the first reading turns an agreement measurement into a
   confirmation measurement.
3. **The identity is a gate, not a nag.** It is asked before the batch opens,
   because everything per-expert — the localStorage namespace, the queues, the
   sheet pull — is wrong until it is answered.
