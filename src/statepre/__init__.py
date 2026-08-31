"""A lab for the *pretraining* stage alone (docs/research/STATE_PRETRAIN_RESEARCH.md).

Section P7 established that GLanCE single-date state labels help this project
when they are spent as a **phase** -- ``siam_state_pretrain`` epochs of
``g(f(x)) -> state`` over a pool, before the transition fit -- rather than as a
joint auxiliary term, and that the user's map read prefers the result. Everything
about that phase downstream of "it helps" is untested: the pool is 13,118 GLanCE
units at **one year**, the encoder is whatever the transition model happens to
use, and the phase's own quality has never been measured except through the
transition metric 30 epochs later.

This package measures the phase directly, on two axes:

* **dataset choices** (``data.py``) -- which single-date rows the phase is fed.
  The lead hypothesis is the user's: a plot that is *stable* across 2018..2024
  carries its state at **every** intermediate year, so its 2019..2023 AlphaEarth
  embeddings are five more free state labels each, drawn from real inter-annual
  variation. That is a temporal-robustness prior with no new data at all --
  ``embeddings_habloss_recover_annual.parquet`` already holds all seven years.
* **architecture choices** (``models.py``) -- what is trained on them, from the
  project's linear probe up to the exact 64->512->256->128 encoder
  ``model_zoo._pretrain_state`` runs, plus the year-conditioning and
  year-invariance variants a multi-year pool makes askable for the first time.

Validation is **LLTO** (``llto.py``), leave-location-*and*-time-out, and it is
not optional here. Year-augmenting a plot creates six near-duplicate rows of one
location; under any split that does not hold a location out at *every* year, a
model scores itself on rows it has all but memorised, and the augmented arms
would win by construction. LLTO folds space (k-means on the plots' coordinates)
and then removes the held-out fold from the training set **at all years**, so a
2024 read is answered by a model that has never seen that ground.

Run it from ``src/``::

    P=/home/geethen.singh/.pixi/envs/geo/bin/python
    $P -m statepre.run --list
    $P -m statepre.run --datasets endpoints stable_years --archs linear --seeds 3
"""
from __future__ import annotations

import sys
from pathlib import Path

# The rest of the project is a flat module directory on sys.path (`cd src` and
# run the script). Importing this package from anywhere else -- `python -m
# statepre.run` from src/, pytest from the repo root -- has to put that directory
# back before `project_paths`, `experiment_merged_legend` and
# `diagnose_state_pools` resolve.
_SRC = str(Path(__file__).resolve().parents[1])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

__all__ = ["data", "llto", "models", "run"]
