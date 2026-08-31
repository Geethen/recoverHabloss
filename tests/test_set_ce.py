"""Invariants of the set-restricted conformal loss (section S).

The mechanism only means what the handoff claims it means if three things hold,
and all three are silent if broken -- the loss still goes down, the run still
finishes, and the verdict is then about something other than conformal sets:

1. the rows that are *scored* are not the rows that *calibrated* the threshold,
   and the two halves swap every epoch (ConfTr's split; otherwise the term is
   trivially satisfied by construction);
2. the quantile is Mondrian, per class -- a pooled cut gives the rare
   transitions near-zero coverage and no subset worth restricting to;
3. the set reaches the loss as a constant 0/1 mask, so nothing differentiates
   through an order statistic and the calibration half gets no gradient.

Plus the S2 control: the random arm must be size-matched to the conformal arm,
or it is not a control for set *size*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

torch = pytest.importorskip("torch")

from model_zoo import HierarchicalSoftmaxNN  # noqa: E402


def _stub(seed=0, alpha=0.10, random=False, epoch=0):
    """A HierarchicalSoftmaxNN carrying only the attributes the term reads.

    The set construction is a pure function of (probs, target, alpha, seed,
    epoch), so it is tested without building a network or a training fold.
    """
    model = HierarchicalSoftmaxNN.__new__(HierarchicalSoftmaxNN)
    model.seed = seed
    model.set_ce_alpha = alpha
    model.set_ce_random = random
    model.set_ce_weight = 0.3
    model.set_ce_level = "fine"
    model._epoch = epoch
    return model


def _probs(n=400, n_classes=4, seed=0, sharpness=None):
    """Softmax rows whose per-class confidence is deliberately uneven.

    ``sharpness[k]`` scales the true-class logit for class ``k``: class 0 is easy
    and class 3 is hard, which is what separates a Mondrian cut from a pooled one.
    """
    rng = np.random.default_rng(seed)
    if sharpness is None:
        sharpness = np.linspace(4.0, 0.5, n_classes)
    y = rng.integers(0, n_classes, n)
    logits = rng.normal(0, 1, (n, n_classes))
    logits[np.arange(n), y] += np.asarray(sharpness)[y]
    p = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1)
    return p, torch.tensor(y, dtype=torch.long)


def test_calibration_and_scoring_halves_are_disjoint_and_alternate():
    """Trap 1. The scored rows must not have set their own threshold."""
    p, y = _probs()
    _, score0, cal0, _ = _stub(epoch=0)._conformal_sets(p, y)
    _, score1, cal1, _ = _stub(epoch=1)._conformal_sets(p, y)

    s0, c0, s1, c1 = (set(t.tolist()) for t in (score0, cal0, score1, cal1))
    assert s0 & c0 == set()
    assert s0 | c0 == set(range(len(y)))
    # Alternating, so every row is scored in one of any two consecutive epochs.
    assert s1 == c0 and c1 == s0


def test_the_loss_does_not_see_the_calibration_half():
    """Disjointness with teeth: perturb a calibration row, get the same number.

    The quantile reads only the *true-class* score of a calibration row, so
    redistributing that row's remaining mass must leave the loss bit-identical --
    and doing the same to a scored row must not.
    """
    p, y = _probs()
    model = _stub()
    _, score_idx, cal_idx, _ = model._conformal_sets(p, y)
    base = model._set_ce_loss(p, y)

    def _reshuffle(rows, i):
        """Keep p[i, y_i]; permute the mass over the other classes."""
        out = rows.clone()
        others = [k for k in range(rows.shape[1]) if k != int(y[i])]
        out[i, others] = out[i, others].flip(0)
        return out

    moved_cal = _reshuffle(p, int(cal_idx[0]))
    assert torch.equal(model._set_ce_loss(moved_cal, y), base)

    moved_score = _reshuffle(p, int(score_idx[0]))
    assert not torch.equal(model._set_ce_loss(moved_score, y), base)


def test_quantiles_are_mondrian_and_a_pooled_cut_would_undercover():
    """Trap 2. One cut per class, and the pooled alternative fails the tail."""
    p, y = _probs(n=2000, n_classes=4)
    model = _stub()
    _, score_idx, cal_idx, q = model._conformal_sets(p, y)

    # Harder classes need a looser cut; a single number cannot serve both.
    assert q[3] > q[0]

    raw = (1.0 - p[score_idx]) <= q[None, :]        # membership before forcing
    cal_scores = 1.0 - p[cal_idx].gather(1, y[cal_idx][:, None]).squeeze(1)
    n_cal = len(cal_idx)
    pooled = cal_scores.sort().values[
        int(np.ceil((n_cal + 1) * (1.0 - model.set_ce_alpha))) - 1]
    raw_pooled = (1.0 - p[score_idx]) <= pooled

    score_y = y[score_idx]
    for cls in range(4):
        rows = score_y == cls
        mondrian = float(raw[rows, cls].to(torch.float64).mean())
        assert mondrian > 0.80, f"class {cls} coverage {mondrian:.3f}"
    hard = score_y == 3
    assert float(raw_pooled[hard, 3].to(torch.float64).mean()) < 0.80


def test_a_class_too_rare_to_calibrate_joins_every_set():
    """Trap 3, and the clamp is cosmetic -- record what it actually does.

    ``rank > count`` is the infinite quantile: fewer than ``(1-a)/a`` calibration
    rows and no order statistic reaches the level. The code clamps it to 1.0,
    but scores are ``1 - p`` in [0, 1], so ``q = 1`` and ``q = inf`` select the
    same thing -- the class is in *every* set either way. The clamp keeps the
    tensor finite; it does not stop a rare class from padding the sets.
    """
    n, n_classes = 400, 4
    rng = np.random.default_rng(0)
    y = rng.integers(0, n_classes - 1, n)           # class 3 almost unobserved
    y[:6] = n_classes - 1
    logits = rng.normal(0, 1, (n, n_classes))
    logits[np.arange(n), y] += 3.0
    p = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1)
    y = torch.tensor(y, dtype=torch.long)

    mask, _, cal_idx, q = _stub()._conformal_sets(p, y)
    assert int((y[cal_idx] == n_classes - 1).sum()) < 9    # ceil((n+1)*0.9) > n
    assert float(q[n_classes - 1]) == 1.0
    assert bool(mask[:, n_classes - 1].all())


def test_singleton_sets_contribute_exactly_zero():
    """A one-class restricted CE is -log(p/p). Those rows are dropped, not averaged in."""
    n, n_classes = 200, 4
    rng = np.random.default_rng(1)
    y = rng.integers(0, n_classes, n)
    logits = np.zeros((n, n_classes))
    logits[np.arange(n), y] = 60.0                  # numerically one-hot
    p = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1)
    y = torch.tensor(y, dtype=torch.long)

    model = _stub()
    mask, _, _, _ = model._conformal_sets(p, y)
    assert int(mask.sum(1).max()) == 1
    assert float(model._set_ce_loss(p, y)) == 0.0


def test_restricted_ce_equals_the_renormalised_cross_entropy():
    """Step 5 of the spec, recomputed in numpy: mean over kept rows only."""
    p, y = _probs()
    model = _stub()
    mask, score_idx, _, _ = model._conformal_sets(p, y)

    pn = p[score_idx].numpy()
    yn = y[score_idx].numpy()
    sets = mask.numpy()
    keep = sets.sum(1) >= 2
    numer = pn[np.arange(len(yn)), yn][keep]
    denom = (pn * sets)[keep].sum(1)
    expected = float(-np.log(numer / denom).mean())

    assert keep.sum() > 0
    assert float(model._set_ce_loss(p, y)) == pytest.approx(expected, rel=1e-9)


def test_no_gradient_reaches_the_quantile_or_the_calibration_rows():
    """Trap 3. The mask is a constant, so d(loss)/dp is zero off the set.

    Two things would be silently wrong otherwise: gradient on calibration rows
    would make the threshold learnable (a different, much slower experiment),
    and gradient on out-of-set classes would make this an ordinary CE.
    """
    p, y = _probs()
    p = p.clone().requires_grad_(True)
    model = _stub()
    mask, score_idx, cal_idx, _ = model._conformal_sets(p.detach(), y)
    model._set_ce_loss(p, y).backward()
    grad = p.grad

    assert torch.count_nonzero(grad[cal_idx]) == 0

    keep = mask.sum(1) >= 2
    scored = grad[score_idx][keep]
    outside = ~mask[keep]
    assert torch.count_nonzero(scored[outside]) == 0
    assert torch.count_nonzero(scored[mask[keep]]) > 0


def test_the_random_control_is_size_matched():
    """S2 is only a control if it holds set size fixed and varies membership."""
    p, y = _probs()
    conformal, score_idx, _, _ = _stub()._conformal_sets(p, y)
    random_sets, _, _, _ = _stub(random=True)._conformal_sets(p, y)

    assert torch.equal(conformal.sum(1), random_sets.sum(1))
    # The truth is forced into both, and the membership actually differs.
    assert bool(random_sets.gather(1, y[score_idx][:, None]).all())
    assert not torch.equal(conformal, random_sets)


# ---------------------------------------------------------------------------
# Wiring: the term has to be reachable from a real fit, not just callable.

BANDS = 6
CLASSES = ["Nature -> Nature", "Nature -> Artificial",
           "Artificial -> Artificial", "Cropland -> Nature"]


def _frame(n=240, seed=0):
    rng = np.random.default_rng(seed)
    cols = {f"A{b:02d}_{yr}": rng.normal(0, 1, n)
            for b in range(BANDS) for yr in (2018, 2024)}
    y = np.array([CLASSES[i % len(CLASSES)] for i in range(n)])
    for i, cls in enumerate(CLASSES):                # a little separable signal
        cols["A00_2024"][y == cls] += 2.0 * i
    return pd.DataFrame(cols), y


def _fit(**kwargs):
    c18 = [f"A{b:02d}_2018" for b in range(BANDS)]
    c24 = [f"A{b:02d}_2024" for b in range(BANDS)]
    base = dict(arch="siamese", epochs=4, siam_columns_18=c18, siam_columns_24=c24,
                siam_extra_columns=[], siam_dim=8, tower_dim=16, seed=0)
    base.update(kwargs)
    frame, y = _frame()
    model = HierarchicalSoftmaxNN(c18 + c24, **base)
    model.fit(frame, y)
    return model


def test_the_epoch_counter_the_split_depends_on_is_actually_advanced():
    """``_epoch`` is read via getattr with a 0 default: if fit never set it, the
    halves would never alternate and nothing would say so."""
    model = _fit(set_ce_weight=0.3)
    assert model._epoch == model.epochs - 1


def test_the_term_is_wired_into_training_and_off_at_weight_zero():
    off = _fit(set_ce_weight=0.0)
    on = _fit(set_ce_weight=0.3)
    baseline = _fit()

    def _flat(m):
        return torch.cat([p.detach().reshape(-1) for p in m.trunk.parameters()])

    assert torch.equal(_flat(off), _flat(baseline))     # default is a no-op
    assert not torch.equal(_flat(off), _flat(on))       # weight 0.3 is not


def test_unknown_level_is_rejected_at_construction():
    with pytest.raises(ValueError, match="set_ce_level"):
        _fit(set_ce_weight=0.3, set_ce_level="gate")
