"""Invariants of the co-teaching selector and the bounded losses (section T).

Section T's verdict rests on the *keep table* -- which classes the selector
rejected -- far more than on any metric it moved, so what has to be tested is
that the selector selects what it claims to. Each of these fails silently
otherwise: the run finishes, the loss goes down, and the section's central table
means something different from what it says.

1. **The pooled selector is rank-based on the loss.** If `classic` did not keep
   exactly the smallest-loss rows, the rarity finding would be an artefact of the
   implementation rather than of the criterion.
2. **Stratification spends the budget at the same rate in every class.** That is
   the entire content of T6, and it is the difference between "selection fails"
   and "selection was never given a chance to filter noise rather than rarity".
3. **`coteach='off'` changes nothing.** The peer, the extra optimiser and the
   per-sample loss path are all additions to a settled model; the deployed recipe
   must be untouched by their presence.
4. **A bounded loss is bounded, and it drops the focal modulation.** GCE that
   still carried `(1-p)^gamma` would be testing the opposite of what it claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

torch = pytest.importorskip("torch")

from model_zoo import HierarchicalSoftmaxNN, level_loss  # noqa: E402


def _selector(mode="classic", forget=0.10, stratify=False, ramp=1):
    """A model carrying only the attributes `_coteach_keep` reads."""
    model = HierarchicalSoftmaxNN.__new__(HierarchicalSoftmaxNN)
    model.coteach = mode
    model.coteach_forget = forget
    model.coteach_stratify = stratify
    model.coteach_ramp = ramp
    model.coteach_warmup = 0
    model.coteach_beta_a = 32.0
    model.coteach_beta_b = 2.0
    model.coteach_min_keep = 0.10
    model.coteach_thresh_per = "batch"
    return model


def test_classic_keeps_exactly_the_smallest_loss_rows():
    loss = torch.tensor([5.0, 0.1, 3.0, 0.2, 4.0, 0.3, 2.0, 1.0, 0.4, 0.5])
    p_true = torch.full((10,), 0.5)
    keep, guard = _selector(forget=0.30).\
        _coteach_keep(p_true, loss, epoch=9, rng=np.random.default_rng(0))
    assert not guard
    assert int(keep.sum()) == 7                      # 1 - 0.30, rounded
    assert loss[keep].max() < loss[~keep].min()


def test_stratify_spends_the_budget_evenly_across_classes():
    """The T6 correction: the rare class loses the same *fraction*, not the same rows.

    Constructed so the pooled selector cannot help but be a rarity filter -- the
    rare class's losses are all above the common class's -- which is the
    situation `coteach_diagnostics.py` measures on the real target.
    """
    groups = torch.tensor([0] * 90 + [1] * 10)
    loss = torch.cat([torch.linspace(0.0, 1.0, 90), torch.linspace(2.0, 3.0, 10)])
    p_true = torch.full((100,), 0.5)
    rng = np.random.default_rng(0)

    pooled, _ = _selector(forget=0.10).\
        _coteach_keep(p_true, loss, epoch=9, rng=rng, groups=groups)
    strat, _ = _selector(forget=0.10, stratify=True).\
        _coteach_keep(p_true, loss, epoch=9, rng=rng, groups=groups)

    # Pooled: the whole 10% budget lands on the rare class.
    assert int(pooled[groups == 1].sum()) == 0
    assert int(pooled[groups == 0].sum()) == 90
    # Stratified: both classes keep 90% of themselves.
    assert int(strat[groups == 1].sum()) == 9
    assert int(strat[groups == 0].sum()) == 81


def test_stochastic_needs_no_forget_rate_and_ramps_in():
    """No `coteach_forget` is read, and the warm-up keeps everything."""
    p_true = torch.linspace(0.01, 0.99, 200)
    loss = torch.zeros(200)
    model = _selector(mode="stochastic", ramp=10)
    model.coteach_warmup = 10
    model.coteach_forget = float("nan")          # must never be consulted

    warm, _ = model._coteach_keep(p_true, loss, epoch=0,
                                  rng=np.random.default_rng(0))
    assert bool(warm.all())                      # eta = 0 -> threshold 0

    late, _ = model._coteach_keep(p_true, loss, epoch=25,
                                  rng=np.random.default_rng(0))
    assert 0 < int(late.sum()) < 200
    # Selection is monotone in the posterior: a kept row never sits below a
    # rejected one, which is what makes the keep table a statement about
    # confidence.
    assert p_true[late].min() > p_true[~late].max()


def test_stochastic_guard_fires_on_an_underconfident_batch():
    """A model whose true-class posterior never clears the draw still trains."""
    p_true = torch.full((100,), 0.02)
    model = _selector(mode="stochastic", ramp=1)
    keep, guard = model._coteach_keep(p_true, torch.zeros(100), epoch=5,
                                      rng=np.random.default_rng(0))
    assert guard
    assert int(keep.sum()) == 10                 # the coteach_min_keep floor


def test_stratify_is_refused_for_the_stochastic_selector():
    model = _selector(mode="stochastic", stratify=True)
    with pytest.raises(ValueError, match="Mondrian"):
        model._coteach_keep(torch.full((10,), 0.5), torch.zeros(10), epoch=5,
                            rng=np.random.default_rng(0),
                            groups=torch.zeros(10, dtype=torch.long))


def test_bounded_losses_are_bounded_and_drop_the_focal_modulation():
    """GCE is capped at 1/q; focal on the same row is not, and is not applied."""
    probs = torch.tensor([[1e-8, 1.0], [0.5, 0.5]]).clamp_min(1e-8)
    target = torch.tensor([0, 0])

    gce = level_loss(probs, target, "focal", robust="gce", robust_q=0.7,
                     reduce=False)
    focal = level_loss(probs, target, "focal", reduce=False)
    ce = level_loss(probs, target, "ce", reduce=False)

    assert float(gce.max()) <= 1.0 / 0.7 + 1e-6      # bounded by construction
    assert float(focal[0]) > 10.0                    # unbounded on a mislabel
    # The confident-wrong row is where focal and GCE disagree most; the
    # near-chance row is where they nearly agree. If GCE still carried
    # (1-p)^gamma the second would be a factor of 4 apart instead.
    assert float(gce[1]) == pytest.approx(float(1 - 0.5 ** 0.7) / 0.7, rel=1e-5)
    assert float(focal[1]) == pytest.approx(0.25 * float(ce[1]), rel=1e-5)


def test_per_sample_levels_mean_matches_the_reduced_loss():
    """`per_sample=True` is the same objective, unreduced -- the selector's read."""
    model = HierarchicalSoftmaxNN.__new__(HierarchicalSoftmaxNN)
    model.loss = "focal"
    model.gamma = 2.0
    model._T = None
    model.robust_loss = "none"
    model.robust_q, model.robust_alpha = 0.7, 0.1
    model.robust_beta, model.robust_a = 1.0, -4.0
    model.robust_levels = "all"
    model.endpoint_weight = model.dice_weight = model.set_ce_weight = 0.0
    model._M = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    model._G = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    torch.manual_seed(0)
    p_fine = torch.softmax(torch.randn(64, 3), dim=1)
    fine_t = torch.randint(0, 3, (64,))
    merged_t = model._M.argmax(1)[fine_t].long()
    gate_t = model._G.argmax(1)[merged_t].long()

    args = (p_fine, fine_t, merged_t, gate_t, None, None, None, 1.0, 1.0, 1.0)
    assert float(model._levels(*args, per_sample=True).mean()) == pytest.approx(
        float(model._levels(*args)), rel=1e-6)


def test_coteach_off_leaves_the_deployed_path_untouched():
    """Two fits of the same seed, one with the section-T defaults spelled out."""
    rng = np.random.default_rng(0)
    n = 120
    frame = pd.DataFrame({
        "A00_2018": rng.normal(size=n), "A01_2018": rng.normal(size=n),
        "A00_2024": rng.normal(size=n), "A01_2024": rng.normal(size=n),
        "A00_diff": rng.normal(size=n),
    })
    y = np.array(["Nature -> Nature"] * 80 + ["Nature -> Artificial"] * 40)
    cols = list(frame.columns)

    plain = HierarchicalSoftmaxNN(cols, arch="wide", loss="focal", epochs=3,
                                  seed=0).fit(frame, y)
    explicit = HierarchicalSoftmaxNN(cols, arch="wide", loss="focal", epochs=3,
                                     seed=0, coteach="off", robust_loss="none",
                                     elr_weight=0.0).fit(frame, y)
    assert np.array_equal(plain._probs(frame)[0], explicit._probs(frame)[0])
    assert plain.coteach_keep_counts_ is None
