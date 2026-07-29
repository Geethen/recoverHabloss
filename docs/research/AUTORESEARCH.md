# Autoresearch loop — two-tower accuracy

The standing instructions for a `/loop` session working on
`data/inference/best_20260725_114640` (`mc_dropout_scalars`: the symmetric
AlphaEarth+Tessera two-tower, Tessera-tower dropout 0.7, per-modality change
scalars, Monte-Carlo modality dropout over 16 passes).

## Objective

Three metrics, all in the ledger, all on the page. A change is a **win** only if
it clears ±0.005 seed noise on its target and does not lose more than 0.005 on
the others.

| metric | now (deploy read) | why it is here |
| --- | --- | --- |
| `change_f1` | 0.6704 | the historical headline; near its label-noise ceiling |
| `macro_f1` | 0.6993 | four-class read; cannot be won on the majority class |
| `art_stable_recall` | 0.639 | **the open frontier** — 22.0% of stable built-up is returned as stable Vegetation, and no idea in ~45 has moved it |

Counter-checks that must not blow up: `veg_stable_as_art` (did the fix just flood
Artificial?) and `tess_recall_gap` (does the fusion omit where Tessera fires?).

## One iteration

```bash
P=/home/geethen.singh/.pixi/envs/geo/bin/python
cd src

$P twotower_lab.py --list                       # what exists
$P twotower_lab.py --ideas <name> --n-seeds 3   # run; appends to the ledger
$P rescore_ledger.py                            # extended metrics from cached OOF
$P build_research_artifact.py                   # rebuild the page
```

Then update the idea's row in `TWOTOWER_RESEARCH.md` — status and the actual
numbers — and republish the artifact to its existing URL.

## Rules

1. **One hypothesis per iteration.** Register it as an idea in `twotower_lab.py`
   with a `desc` that says what it tests and why, run it on both reads, record
   the number. Do not run a batch of six and report the max — that is how seed
   noise gets promoted to a finding.
2. **3 seeds minimum before any verdict**, 5 before anything is called a win.
   Sub-1pt differences at 1 seed are noise; the ledger has the evidence.
3. **Reuse the OOF cache.** Anything post-hoc — blending, stacking, thresholds,
   cost-sensitive reads, seed ensembles — must load cached probabilities rather
   than refit. `rescore_ledger.py` shows how.
4. **Negative results are results.** Write them into the backlog table with the
   number and the reason. Sections A–E are mostly negative and that is the most
   useful part of the document.
5. **Do not redo the tested-negative list** at the foot of
   `TWOTOWER_RESEARCH.md`.
6. **Stop and report** when the F and G sections are exhausted, or when three
   consecutive iterations come back flat on all three metrics — at that point the
   bottleneck is data (label noise, Tessera 2018 coverage), not modelling, and
   the loop should say so rather than keep sampling noise.

## State

Everything is on disk and survives a cold start:

* `data/analysis_results/twotower_lab_ledger.csv` — append-only run log
* `data/analysis_results/twotower_lab_metrics.csv` — extended metrics, rebuilt
* `data/analysis_results/twotower_lab_oof/` — cached OOF probabilities per
  idea × read × seed
* `src/TWOTOWER_RESEARCH.md` — the backlog and every verdict
* `scratchpad/twotower_research.html` — the published page
