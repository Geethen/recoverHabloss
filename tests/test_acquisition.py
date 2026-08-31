"""Invariants of the acquisition metrics (docs/research/ACTIVE_LEARNING.md).

These lock down the properties the design argument leans on, rather than the
numbers: that the cluster surface is a proper distribution without
renormalisation, that BALD separates reducible from irreducible uncertainty,
that the conformal channel reaches a class the arg-max cannot, that the yield
score refuses to be fooled by scattered pixels or by an abundant class, and that
the online update is signed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from acquisition import (  # noqa: E402
    SamplingSurface, bald, choose_clustering_by_silhouette, class_balance_greedy,
    class_in_set,
    cluster_inverse_size, conformal_set_size, deficit_weighted_yield,
    feature_activation, feature_diversity, kcenter_greedy, label_complexity,
    least_confidence, margin, mean_within_cluster_vendi, normalised_entropy,
    novelty_to_reference, order_to_score, rank_mean, search_k_vendi,
    silhouette_score, vendi_score, vote_entropy,
)


def test_normalised_entropy_spans_the_unit_interval():
    one_hot = np.eye(9)[[0, 4]]
    flat = np.full((1, 9), 1 / 9)
    assert normalised_entropy(one_hot) == pytest.approx(0.0, abs=1e-9)
    assert normalised_entropy(flat) == pytest.approx(1.0, abs=1e-9)


def test_margin_separates_contested_from_merely_uncertain():
    """Two rows with near-identical entropy but opposite decision margins.

    The point of keeping both scores: entropy cannot tell a knife-edge boundary
    pixel from mass smeared over the tail.
    """
    contested = np.array([[0.40, 0.39, 0.07, 0.07, 0.07]])
    smeared = np.array([[0.60, 0.10, 0.10, 0.10, 0.10]])
    assert margin(contested)[0] > margin(smeared)[0]
    assert least_confidence(contested)[0] > least_confidence(smeared)[0]


def test_bald_is_zero_when_members_agree_and_positive_when_they_do_not():
    ambiguous = np.tile(np.array([[[0.5, 0.5]]]), (4, 1, 1))   # all members flat
    split = np.array([[[1.0, 0.0]], [[0.0, 1.0]],
                      [[1.0, 0.0]], [[0.0, 1.0]]])             # members disagree
    # Both have the same *total* entropy; only the second is reducible.
    assert normalised_entropy(ambiguous.mean(0))[0] == pytest.approx(
        normalised_entropy(split.mean(0))[0])
    assert bald(ambiguous)[0] == pytest.approx(0.0, abs=1e-9)
    assert bald(split)[0] == pytest.approx(1.0, abs=1e-9)
    assert vote_entropy(split)[0] > vote_entropy(ambiguous)[0] - 1e-9


def test_conformal_set_reaches_a_class_the_argmax_never_wins():
    """The PATCH_SAMPLING section C problem, in miniature.

    Class 2 is never the arg-max anywhere, so an arg-max-navigated campaign
    cannot be sent to it. With a per-class (Mondrian) qhat it is still in the
    90% set, which is what makes set membership a retrieval channel.
    """
    probs = np.array([[0.55, 0.30, 0.15],
                      [0.60, 0.22, 0.18]])
    assert (probs.argmax(axis=1) == 2).sum() == 0
    marginal = conformal_set_size(probs, qhat=0.50)
    mondrian = conformal_set_size(probs, qhat=np.array([0.50, 0.50, 0.90]))
    assert (mondrian > marginal).all()
    assert class_in_set(probs, np.array([0.50, 0.50, 0.90]), 2).all()
    assert not class_in_set(probs, 0.50, 2).any()


def test_cluster_inverse_size_is_already_normalised():
    labels = np.array([0, 0, 0, 0, 1, 2, 2])
    p = cluster_inverse_size(labels)
    assert p.sum() == pytest.approx(1.0)
    # every cluster carries equal total mass, so the singleton beats a member of
    # the big cluster by exactly the size ratio
    assert p[4] / p[0] == pytest.approx(4.0)
    for group in ([0, 1, 2, 3], [4], [5, 6]):
        assert p[group].sum() == pytest.approx(1 / 3)


def test_dbscan_noise_points_are_singletons_not_one_cluster():
    labels = np.array([0, 0, 0, -1, -1])
    p = cluster_inverse_size(labels)
    assert p[3] == pytest.approx(p[4])
    assert p[3] > p[0]


def test_feature_activation_rejects_the_flat_homogeneous_cell():
    rng = np.random.default_rng(0)
    busy = rng.normal(0.8, 0.5, size=(1, 64))
    flat = np.full((1, 64), 0.05) + rng.normal(0, 1e-3, size=(1, 64))
    score = feature_activation(np.concatenate([busy, flat]))
    assert score[0] > score[1]


def test_kcenter_seeded_with_existing_labels_picks_the_unrepresented_cell():
    labelled = np.array([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
    pool = np.concatenate([labelled, np.array([[0.0, 1.0]])])
    picked = kcenter_greedy(pool, budget=1, selected=np.array([0, 1, 2]))
    assert picked.tolist() == [3]


def test_novelty_is_referenced_to_the_labels_not_the_pool():
    reference = np.array([[1.0, 0.0]])
    twin = np.tile(np.array([[1.0, 0.001]]), (50, 1))
    stranger = np.tile(np.array([[0.0, 1.0]]), (50, 1))
    assert novelty_to_reference(twin, reference) < 1e-3
    assert novelty_to_reference(stranger, reference) > 0.9


def test_label_complexity_prefers_the_mixed_patch():
    counts = np.array([[250_000, 0, 0, 0],          # uniform stable patch
                       [100_000, 80_000, 50_000, 20_000]])
    lc = label_complexity(counts)
    assert lc[0] == pytest.approx(0.0, abs=1e-9)
    assert lc[1] > 0.8


def test_class_balance_chases_the_scarcest_class_given_a_prior():
    """With 4,200 stable plots already held, the greedy pick must not be the
    patch full of more of them."""
    counts = np.array([[1000, 0, 0],     # more of the abundant class
                       [0, 10, 0],       # a little of a scarce one
                       [0, 0, 10]])
    prior = np.array([4200.0, 46.0, 114.0])
    order = class_balance_greedy(counts, budget=2, prior=prior)
    assert 0 not in order.tolist()
    assert order[0] == 1          # the scarcest class (46) is taken first


def test_yield_refuses_scattered_pixels_and_abundant_classes():
    # class 0 abundant (no deficit), class 1 wanted
    counts = np.array([[250_000, 40],      # huge but only 40 px of what is wanted
                       [0, 5_000]])        # 50 ha of the wanted class
    score = deficit_weighted_yield(counts,
                                   deficit=np.array([0.0, 333.0]),
                                   precision=np.array([1.0, 0.514]))
    assert score[0] == pytest.approx(0.0)          # 40 px is below one hectare
    assert score[1] == pytest.approx(3 * 0.514)    # capped at max_points


def test_precision_scaling_is_not_optional():
    counts = np.array([[5_000]])
    unadjusted = deficit_weighted_yield(counts, np.array([333.0]), np.array([1.0]))
    adjusted = deficit_weighted_yield(counts, np.array([333.0]), np.array([0.308]))
    assert adjusted[0] / unadjusted[0] == pytest.approx(0.308)


def test_rank_mean_is_immune_to_scale():
    a = np.array([0.001, 0.002, 0.003])
    b = np.array([100.0, 50.0, 0.0])
    assert rank_mean(a, b).tolist() == rank_mean(a, b * 1e6).tolist()
    # a raw sum would let b decide the whole ordering; the rank mean does not
    assert rank_mean(a, b)[1] == pytest.approx(0.5)


def test_surface_samples_without_replacement_and_renormalises():
    surf = SamplingSurface.uniform(50)
    rng = np.random.default_rng(0)
    first = surf.sample(10, rng)
    second = surf.sample(10, rng)
    assert len(set(first.tolist()) & set(second.tolist())) == 0
    assert surf.p.sum() == pytest.approx(1.0)
    assert surf.p[first].sum() == pytest.approx(0.0)


def test_online_update_is_signed():
    """A hit on a wanted class raises its neighbourhood; a hit on a satisfied
    class lowers it. Same mechanism, sign set by the deficit."""
    xy = np.array([[0.0, 0.0], [1_000.0, 0.0], [500_000.0, 0.0]])
    up = SamplingSurface.uniform(3, xy=xy)
    up.proximity_update(0, radius_m=5_000, weight=+1.0)
    assert up.p[1] > up.p[2]

    down = SamplingSurface.uniform(3, xy=xy)
    down.proximity_update(0, radius_m=5_000, weight=-0.3)
    assert down.p[1] < down.p[2]


def test_cluster_update_carries_a_hit_across_the_globe():
    xy = np.array([[0.0, 0.0], [9e6, 0.0], [1_000.0, 0.0]])
    clusters = np.array([7, 7, 3])
    surf = SamplingSurface.uniform(3, xy=xy, clusters=clusters)
    surf.cluster_update(0, weight=1.0)
    assert surf.p[1] > surf.p[2]      # far away but same cluster wins


# --------------------------------------------------------------------------
# Vendi score (added after review: the FD row had no implementation)
# --------------------------------------------------------------------------

def test_vendi_counts_distinct_things_not_rows():
    """1 for duplicates, n for mutually orthogonal, k for k orthogonal groups."""
    same = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (50, 1))
    assert vendi_score(same) == pytest.approx(1.0, abs=1e-8)

    orth = np.eye(4)
    assert vendi_score(orth) == pytest.approx(4.0, abs=1e-8)

    # 50 copies each of 4 orthogonal prototypes: 200 rows, 4 distinct things
    grouped = np.repeat(np.eye(4), 50, axis=0)
    assert vendi_score(grouped) == pytest.approx(4.0, abs=1e-8)


def test_vendi_gram_trick_is_exact_not_approximate():
    """The D x D route must equal the n x n route to machine precision.

    This is the whole scalability claim: with D = 64 and n = 5.4M the n x n
    similarity matrix is never formed, and that has to cost nothing in accuracy.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 12))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)

    fast = vendi_score(x)                       # picks the 12 x 12 Gram matrix
    k = (x @ x.T) / x.shape[0]                  # the explicit 400 x 400 route
    lam = np.linalg.eigvalsh(k)
    lam = lam[lam > 1e-12]
    slow = float(np.exp(-(lam * np.log(lam / lam.sum())).sum() / lam.sum()))
    assert fast == pytest.approx(slow, rel=1e-9)


def test_vendi_is_the_exponential_of_shannon_entropy():
    """Vendi and `normalised_entropy` are the same functional on different
    distributions -- pin that, because it is the reason both are in here."""
    x = np.repeat(np.eye(4), [40, 30, 20, 10], axis=0)
    lam = np.array([40, 30, 20, 10], dtype=float) / 100
    shannon_nats = -(lam * np.log(lam)).sum()
    assert vendi_score(x) == pytest.approx(np.exp(shannon_nats), abs=1e-8)


def test_vendi_renyi_orders_bracket_the_shannon_one():
    x = np.repeat(np.eye(4), [70, 20, 7, 3], axis=0)
    assert vendi_score(x, q=np.inf) == pytest.approx(1 / 0.70, abs=1e-8)
    assert vendi_score(x, q=np.inf) < vendi_score(x, q=1.0) < vendi_score(x, q=0.5)


def test_vendi_normalise_compares_batches_of_different_size():
    rng = np.random.default_rng(1)
    big = rng.normal(size=(200, 8))
    small = rng.normal(size=(20, 8))
    assert vendi_score(big) > vendi_score(small)              # raw favours size
    assert vendi_score(small, normalise=True) > vendi_score(big, normalise=True)


def test_within_cluster_vendi_is_degenerate_raw_and_fixed_normalised():
    """Raw within-cluster Vendi falls monotonically with K, so it cannot pick K.

    The normalised form flips singletons to the worst score, which is the fix.
    """
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(60, 6))
    one_cluster = np.zeros(60, dtype=int)
    singletons = np.arange(60)

    assert mean_within_cluster_vendi(feats, singletons, normalise=False) == \
        pytest.approx(1.0)
    assert mean_within_cluster_vendi(feats, one_cluster, normalise=False) > 1.0
    # normalised: singletons are now the *worst* (1.0), not the best
    assert mean_within_cluster_vendi(feats, singletons, normalise=True) == \
        pytest.approx(1.0)
    assert mean_within_cluster_vendi(feats, one_cluster, normalise=True) < 1.0


def test_feature_diversity_covers_every_cluster_before_repeating():
    labels = np.array([0] * 100 + [1] * 5 + [2] * 2)
    order = feature_diversity(labels, budget=3, rng=np.random.default_rng(0))
    assert sorted(labels[order].tolist()) == [0, 1, 2]


def test_feature_diversity_beats_proportional_sampling_on_batch_vendi():
    """The point of round-robin: a proportional draw is swamped by the big
    cluster, FD is not. Measured with the Vendi score itself."""
    rng = np.random.default_rng(0)
    proto = np.eye(3)
    feats = np.repeat(proto, [500, 20, 5], axis=0)
    labels = np.repeat([0, 1, 2], [500, 20, 5])

    fd = feature_diversity(labels, budget=9, rng=rng)
    proportional = rng.choice(len(labels), size=9, replace=False)
    assert vendi_score(feats[fd]) > vendi_score(feats[proportional])
    assert vendi_score(feats[fd]) == pytest.approx(3.0, abs=1e-8)


def test_order_to_score_is_monotone_and_zero_for_unselected():
    order = np.array([7, 2, 5])
    score = order_to_score(order, n_total=10)
    assert score[7] > score[2] > score[5] > 0
    assert score[[0, 1, 3, 4, 6, 8, 9]].tolist() == [0.0] * 7


# --------------------------------------------------------------------------
# Second audit against the papers: details missed on the first pass
# --------------------------------------------------------------------------

def test_label_complexity_must_be_able_to_drop_nodata():
    """13 of 100 pilot patches were all-water. Counted as a class, nodata reads
    as variety and an empty patch scores as one of the most complex on the map.
    """
    # column 0 = nodata, columns 1..3 = real classes
    counts = np.array([[125_000, 125_000, 0, 0],     # all-water + a sliver
                       [0, 100_000, 80_000, 70_000]])  # genuinely mixed
    naive = label_complexity(counts)
    assert naive[0] > 0.4                       # nodata masquerading as variety
    fixed = label_complexity(counts, ignore_index=0)
    assert fixed[0] == pytest.approx(0.0, abs=1e-9)
    assert fixed[1] > 0.9


def test_feature_activation_scaling_fixes_the_sign_flip():
    """The paper scales mu and sigma to (0, 1] before forming gamma, and its
    features are ReLU-nonnegative. Unscaled, once sigma > 1 the log turns
    positive and the ranking inverts against the paper's own stated assumption
    ("high mean AND high standard deviation carry more information").

    Exact, not random: base has mean 0 and sd 1, so mu and sigma are set
    directly. A and B share a spread; A is the active one.
    """
    base = np.array([1.0, -1.0] * 32)          # mu = 0, sigma = 1
    active = 5.0 + 2.0 * base                  # mu = 5.0,  sigma = 2
    inactive = 0.1 + 2.0 * base                # mu = 0.1,  sigma = 2
    flat = 0.1 + 0.01 * base                   # mu = 0.1,  sigma = 0.01
    feats = np.stack([active, inactive, flat])

    raw = feature_activation(feats, scale=False)
    assert raw[0] < raw[1]        # BUG: the inactive cell outranks the active one

    scaled = feature_activation(feats, scale=True)
    assert scaled[0] >= scaled[1]     # order restored
    assert scaled[2] < scaled[0]      # and the flat cell is still rejected


def test_search_k_vendi_stops_on_a_plateau_not_at_a_minimum():
    """Raw within-cluster Vendi falls forever, so the stopping rule has to be
    the plateau (delta for `patience` steps), not the argmin."""
    rng = np.random.default_rng(0)
    proto = rng.normal(size=(4, 8)) * 5
    feats = np.repeat(proto, 40, axis=0) + rng.normal(0, .05, size=(160, 8))

    def cluster_fn(x, k):
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(x)

    k, labels, trace = search_k_vendi(feats, cluster_fn, k_max=12,
                                      delta=0.005, patience=3)
    assert 4 <= k <= 12                       # finds the four real groups, stops
    assert len(labels) == len(feats)
    assert trace[0] > trace[-1]               # monotone-ish decline
    assert k < 12                             # stopped early, did not run to k_max


def test_silhouette_matches_sklearn():
    rng = np.random.default_rng(0)
    feats = np.concatenate([rng.normal(0, .3, (40, 5)),
                            rng.normal(4, .3, (40, 5))])
    labels = np.repeat([0, 1], 40)
    from sklearn.metrics import silhouette_score as sk_sil
    assert silhouette_score(feats, labels, sample_size=None) == \
        pytest.approx(sk_sil(feats, labels), abs=1e-9)


def test_clustering_is_chosen_by_MINIMUM_silhouette():
    """The bootstrapping paper's Appendix A inverts the usual rule: they run a
    Bayesian search *minimising* |silhouette| because that is what correlates
    with finding positives fast (R^2 = 0.93). Easy to get backwards.
    """
    candidates = {"kmeans_k8": 0.71, "bkmeans_k64": 0.05, "dbscan": -0.40}
    name, score = choose_clustering_by_silhouette(candidates)
    assert name == "bkmeans_k64"        # not the well-separated 0.71
    assert score == 0.05


def test_feature_diversity_start_cluster_is_randomised():
    """Paper: 'the first cluster is randomly selected from the set of K
    clusters'. With a budget below K, a fixed order would always return the
    same cluster."""
    labels = np.repeat(np.arange(6), 10)
    firsts = {int(labels[feature_diversity(labels, budget=1,
                                           rng=np.random.default_rng(s))[0]])
              for s in range(30)}
    assert len(firsts) > 1
