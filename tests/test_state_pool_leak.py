"""The state pool must not carry a held-out plot's own label into pretraining.

``cv_probs_state`` cuts an external state pool to each fold's *training blocks*:

    train_blocks = set(view.frame.iloc[tr]["block_id"].unique())
    fold_pool    = pool[pool["block_id"].isin(train_blocks)]

For the GLanCE pool that is a distribution argument -- no pool point sits within
100 m of a RECOVER plot, so the worst case is the encoder seeing the held-out
block's feature distribution. For an **endogenous** pool it is much stronger than
that, and load-bearing: ``statepre.export`` writes pools (``endpoints``,
``glance_endpoints``, ``stable_years``) whose rows *are* RECOVER plots carrying
``lc_2018``/``lc_2024``, which are the two halves of the very transition label
the model is scored on. If one of those rows survives the filter for the fold
its plot is held out in, the encoder is pretrained on the answer.

**The filter is only equivalent to fold-level exclusion because the folds never
split a block** -- ``make_splitter("blocked", ...)`` is a ``StratifiedGroupKFold``
grouped on ``block_id``. That is the whole argument, it is one keyword deep, and
nothing downstream would report its loss: a leaked pretraining label does not
change a shape, raise a warning, or fail an assertion. It makes the number
better.

So both halves are pinned here: blocked CV excludes every held-out plot, and
random CV -- the mode a future edit could plausibly switch to -- does **not**,
which is why the pairing of "blocked folds" with "block-filtered pool" is not an
implementation detail to be optimised away.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from model_zoo import make_splitter  # noqa: E402

N_SPLITS = 5


def _frame(n_blocks: int = 40, per_block: int = 25, seed: int = 0) -> pd.DataFrame:
    """Plots in spatial blocks, with a transition label. No real data."""
    rng = np.random.default_rng(seed)
    n = n_blocks * per_block
    return pd.DataFrame({
        "PLOTID": [f"p{i:05d}" for i in range(n)],
        "block_id": np.repeat([f"b{b:02d}" for b in range(n_blocks)], per_block),
        "target": rng.choice(["Nature -> Nature", "Nature -> Artificial",
                              "Cropland -> Nature"], n, p=[0.8, 0.1, 0.1]),
        "x": rng.normal(size=n),
    })


def _state_pool(frame: pd.DataFrame) -> pd.DataFrame:
    """What ``statepre.export`` writes for an endogenous arm: one row per plot.

    ``block_id`` rides along because it is the only key the fold filter has.
    """
    return pd.DataFrame({
        "sid": "recover:" + frame["PLOTID"],
        "PLOTID": frame["PLOTID"],
        "block_id": frame["block_id"],
        "state": "nature",
    })


def _leaked(frame: pd.DataFrame, pool: pd.DataFrame, cv: str) -> int:
    """Held-out plots whose own pool row survives ``cv_probs_state``'s filter."""
    splitter = make_splitter(cv, N_SPLITS)
    total = 0
    for tr, te in splitter.split(frame[["x"]], frame["target"], frame["block_id"]):
        train_blocks = set(frame.iloc[tr]["block_id"].unique())     # cv_probs_state
        fold_pool = pool[pool["block_id"].isin(train_blocks)]
        total += len(set(fold_pool["PLOTID"]) & set(frame.iloc[te]["PLOTID"]))
    return total


def test_blocked_folds_never_split_a_block():
    """The premise. Everything below is a consequence of this one property."""
    frame = _frame()
    splitter = make_splitter("blocked", N_SPLITS)
    for tr, te in splitter.split(frame[["x"]], frame["target"], frame["block_id"]):
        shared = (set(frame.iloc[tr]["block_id"])
                  & set(frame.iloc[te]["block_id"]))
        assert not shared, f"block(s) {sorted(shared)} are in both halves"


def test_block_filter_removes_every_held_out_plot_under_blocked_cv():
    """The invariant: an endogenous pool cannot reach the fold it is scored on."""
    frame = _frame()
    assert _leaked(frame, _state_pool(frame), "blocked") == 0


def test_block_filter_does_not_protect_random_cv():
    """The counter-case, and the reason the invariant above is not free.

    Under random CV a block is split across folds, so nearly every block is a
    "training block" and the filter keeps the test plots' own rows. Asserted so
    that switching the CV mode fails here rather than quietly improving a score.
    """
    frame = _frame()
    leaked = _leaked(frame, _state_pool(frame), "random")
    assert leaked > 0.9 * len(frame), (
        f"only {leaked} of {len(frame)} leaked -- if random CV has become "
        "block-aware, this test's warning is stale and should be rewritten")


@pytest.mark.parametrize("cv", ["blocked", "random"])
def test_filter_keeps_the_pool_usable(cv):
    """A guard that does not also delete the pool. A filter that removed
    everything would pass the leak test and silently disable the phase."""
    frame = _frame()
    pool = _state_pool(frame)
    splitter = make_splitter(cv, N_SPLITS)
    for tr, _ in splitter.split(frame[["x"]], frame["target"], frame["block_id"]):
        train_blocks = set(frame.iloc[tr]["block_id"].unique())
        assert len(pool[pool["block_id"].isin(train_blocks)]) > 0.5 * len(tr)
