"""Invariants of the siamese endpoint trunk (docs/research/SIAMESE_RESEARCH.md).

These lock down the three things in section N that were either silently wrong at
some point or would be silently wrong if changed: the pooled standardisation that
makes weight sharing mean anything, the BatchNorm freeze on the unlabelled pass,
and the column ordering the shared encoder depends on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

torch = pytest.importorskip("torch")

from model_zoo import HierarchicalSoftmaxNN, _SiameseTrunk  # noqa: E402


BANDS = 6


def _frame(n=200, seed=0):
    """A small labelled frame with a deliberate between-year offset on band 0."""
    rng = np.random.default_rng(seed)
    cols = {}
    for b in range(BANDS):
        cols[f"A{b:02d}_2018"] = rng.normal(0, 1, n)
        # Band 0 is shifted between the years; every other band is not. A pooled
        # standardisation must PRESERVE that shift -- it is the change signal.
        cols[f"A{b:02d}_2024"] = rng.normal(3.0 if b == 0 else 0.0, 1, n)
    frame = pd.DataFrame(cols)
    y = np.array(["Nature -> Nature"] * (n // 2)
                 + ["Nature -> Artificial"] * (n - n // 2))
    return frame, y


def _model(**kwargs):
    c18 = [f"A{b:02d}_2018" for b in range(BANDS)]
    c24 = [f"A{b:02d}_2024" for b in range(BANDS)]
    base = dict(arch="siamese", epochs=2, siam_columns_18=c18,
                siam_columns_24=c24, siam_extra_columns=[], siam_dim=8,
                tower_dim=16)
    base.update(kwargs)
    return HierarchicalSoftmaxNN(c18 + c24, **base), c18, c24


def test_endpoint_blocks_share_one_standardisation():
    """Both years must be centred by the same mu/sd, or sharing is undone.

    Per-year statistics would re-centre the band-0 offset to zero at both dates
    and the model would read a real between-year shift as no change at all.
    """
    frame, y = _frame()
    model, _, _ = _model()
    Xs = model._prepare(frame, fit=True)
    n_end = BANDS
    x18, x24 = Xs[:, :n_end], Xs[:, n_end:2 * n_end]

    # One statistic per feature, used for both years.
    assert model.mu_end.shape == (n_end,)
    # The band-0 offset survives standardisation; the other bands stay aligned.
    assert x24[:, 0].mean() - x18[:, 0].mean() > 1.0
    assert abs(x24[:, 1].mean() - x18[:, 1].mean()) < 0.5


def test_encoder_sees_one_batchnorm_population():
    """The two dates go through the encoder stacked, not in separate calls.

    Separate calls give each year its own BatchNorm statistics, which would
    re-centre the between-year shift the stacked call preserves.
    """
    trunk = _SiameseTrunk(d_end=BANDS, d_extra=0, out_dim=16, siam_dim=8,
                          dropout=0.0, combine="conc")
    trunk.train()
    x = torch.randn(64, 2 * BANDS)
    x[:, BANDS] += 5.0                      # shift band 0 in 2024 only
    trunk(x)
    # Stacked: the pair differs, so the representations must differ too.
    assert not torch.allclose(trunk.last_z18, trunk.last_z24)


def test_frozen_bn_stats_leaves_running_statistics_untouched():
    """The unlabelled Barlow pass must not move eval-time BatchNorm state.

    This is the defect that read as a 5-point accuracy collapse in N4: the extra
    forward pass over out-of-distribution pixels folded their distribution into
    the running mean/var that eval() then applied to labelled test rows.
    """
    frame, y = _frame()
    model, _, _ = _model()
    model.fit(frame, y)
    bn = next(m for m in model.trunk.enc if isinstance(m, torch.nn.BatchNorm1d))

    before = (bn.running_mean.clone(), bn.running_var.clone())
    # Wildly out of distribution, and on whatever device the fit landed on.
    other = (torch.randn(128, 2 * BANDS, device=model.device) * 4 + 9)
    for module in model._modules_:
        module.train()
    with model._frozen_bn_stats():
        model._encode(other)
    assert torch.allclose(bn.running_mean, before[0])
    assert torch.allclose(bn.running_var, before[1])

    # ...and without the freeze it genuinely moves, so the test is not vacuous.
    model._encode(other)
    assert not torch.allclose(bn.running_mean, before[0])


def test_siam_pair_is_found_inside_a_two_tower():
    """The auxiliary losses must locate the pair whether it is the trunk or a tower."""
    trunk = _SiameseTrunk(d_end=BANDS, d_extra=0, out_dim=16, siam_dim=8,
                          dropout=0.0, combine="conc")
    trunk.train()
    trunk(torch.randn(32, 2 * BANDS))

    model, _, _ = _model()
    model.trunk = trunk
    z18, z24 = model._siam_pair()
    assert z18.shape == z24.shape == (32, 8)

    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.aef_tower = trunk

    model.trunk = _Wrapper()
    assert model._siam_pair()[0].shape == (32, 8)


def test_siam_pair_raises_when_no_encoder_produced_one():
    """A misconfigured auxiliary loss must fail loudly, not add a silent zero."""
    model, _, _ = _model()
    model.trunk = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="no endpoint pair"):
        model._siam_pair()


def test_cosine_loss_separates_stable_from_change():
    """Stable pairs are pulled together and change pairs pushed apart.

    Two identical embeddings on the stable rows and opposed ones on the change
    rows is the loss's own optimum, so it must score lower there than on the
    reverse arrangement.
    """
    model, _, _ = _model(siam_cos_weight=1.0, siam_cos_margin=0.0)
    n = 8
    good = torch.ones(n, 4)
    stable = torch.tensor([True] * (n // 2) + [False] * (n // 2))

    class _Pair:
        pass

    model.trunk = _Pair()
    model.trunk.last_z18 = good.clone()
    model.trunk.last_z24 = torch.cat([good[:n // 2], -good[n // 2:]])
    aligned = float(model._siam_cos_loss(stable))

    model.trunk.last_z24 = torch.cat([-good[:n // 2], good[n // 2:]])
    reversed_ = float(model._siam_cos_loss(stable))
    assert aligned < reversed_


# -- section Q10 (SNIIF-Net): the two properties the arms are only interpretable
# under -- FIIM starts at the identity, and its control differs from it in
# exactly one thing.

def test_fiim_starts_at_the_plain_encoder():
    """Zero-init + tanh means the multiplier is 1, so an untrained FIIM is a no-op.

    Q10f is read as "what the module moves", which is only true if it moves
    nothing before it is trained. A non-identity init would make the arm a
    different random initialisation as well as a different module.
    """
    kw = dict(d_end=6, d_extra=0, out_dim=16, siam_dim=8, dropout=0.0)
    torch.manual_seed(0)
    plain = _SiameseTrunk(fiim="none", **kw)
    torch.manual_seed(0)
    gated = _SiameseTrunk(fiim="cross", **kw)
    plain.eval(), gated.eval()
    x = torch.randn(4, 12)
    assert torch.allclose(plain(x), gated(x), atol=1e-6)


def test_fiim_self_control_matches_cross_in_size_and_not_in_input():
    """The control must differ from FIIM only by the cross-branch information.

    Same parameter count and same init draw, or Q10g stops being a control and
    becomes a second, smaller module.
    """
    kw = dict(d_end=6, d_extra=0, out_dim=16, siam_dim=8, dropout=0.0)
    torch.manual_seed(0)
    cross = _SiameseTrunk(fiim="cross", **kw)
    torch.manual_seed(0)
    self_ = _SiameseTrunk(fiim="self", **kw)
    n_cross = sum(p.numel() for p in cross.parameters())
    n_self = sum(p.numel() for p in self_.parameters())
    assert n_cross == n_self
    # Make the gate live (it is zero-initialised), then feed a pair whose two
    # dates differ: only the cross form can respond to the other date.
    for net in (cross, self_):
        torch.nn.init.normal_(net.fiim_gate.weight, std=0.5)
        net.eval()
    x = torch.randn(4, 12)
    assert not torch.allclose(cross(x), self_(x), atol=1e-5)


def test_mssm_applies_the_same_term_at_every_stage():
    """MSSM must be the final-layer objective repeated, not a second objective.

    If the stage term drifted from ``_siam_cos_loss``, a flat Q10a would be
    about the drift. Pinned by making all three scales identical: the
    multi-scale value must then equal the single-scale one.
    """
    model, _, _ = _model(siam_cos_weight=1.0, siam_cos_margin=0.0,
                         siam_mssm_weight=1.0, siam_mssm_scales="all")
    n = 8
    z18 = torch.ones(n, 4)
    z24 = torch.cat([z18[:n // 2], -z18[n // 2:]])
    stable = torch.tensor([True] * (n // 2) + [False] * (n // 2))

    class _Pair:
        pass

    model.trunk = _Pair()
    model.trunk.last_z18, model.trunk.last_z24 = z18, z24
    model.trunk.last_h = [(z18, z24), (z18, z24)]
    assert float(model._siam_mssm_loss(stable)) == pytest.approx(
        float(model._siam_cos_loss(stable)), abs=1e-6)


def test_stable_margin_releases_pairs_that_already_agree():
    """The double margin (Q10e) must be inert at 0 and slack above it."""
    n = 4
    stable = torch.ones(n, dtype=torch.bool)
    z18 = torch.ones(n, 4)
    # Perturb ONE coordinate: adding a constant to every coordinate leaves the
    # direction unchanged, and this term only sees direction.
    z24 = z18.clone()
    z24[:, 0] += torch.tensor([0.0, 0.2, 0.5, 1.0])

    class _Pair:
        pass

    def _loss(eps):
        model, _, _ = _model(siam_cos_weight=1.0, siam_cos_stable_margin=eps)
        model.trunk = _Pair()
        model.trunk.last_z18, model.trunk.last_z24 = z18, z24
        return float(model._siam_cos_loss(stable))

    assert _loss(0.0) > _loss(0.1) >= 0.0
    assert _loss(1.0) == pytest.approx(0.0)
