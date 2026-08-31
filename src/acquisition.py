"""Acquisition metrics for the global active-learning surface.

Every function here scores *candidates* — grid cells, patches or pixels — so a
labelling budget can be spent on the ones worth an afternoon. They are pure
numpy over arrays, with no fetch, no torch and no AlphaEarth dependency, so they
can be unit-tested and so the expensive stages (embedding fetch, forward pass)
stay downstream of the ranking rather than upstream of it. That ordering is the
whole scalability argument: see `docs/research/ACTIVE_LEARNING.md`.

The three families, and why the split matters here
--------------------------------------------------
**Model-free** (`cluster_inverse_size`, `feature_activation`, `kcenter_greedy`,
`novelty_to_reference`) score a cell from imagery/embeddings alone. They are the
only family that can run over the whole 5.4 M-cell global grid, and — the point
that decides the design — they are the only family that can reach
``Artificial -> Cropland``. The deployed model's posterior for that class never
exceeds 0.191 anywhere in 21 M pilot pixels (`PATCH_SAMPLING.md` section C), so
*every* model-in-the-loop score is blind to it by construction.

**Model-in-the-loop** (`normalised_entropy`, `least_confidence`, `margin`,
`bald`, `vote_entropy`, `conformal_set_size`) need a forward pass per candidate.
At 7.2 s/patch they are affordable on a shortlist of thousands, not on millions.

**Composition-driven** (`label_complexity`, `class_balance_greedy`,
`deficit_weighted_yield`) score a cell by what it would *add* to the training
set. On an unlabelled pool they run on predicted class counts, which is why
`deficit_weighted_yield` takes a per-class precision: predicted points are not
confirmed plots, and the change classes run at OOF precision 0.31-0.51.

Sources
-------
Zaytar et al. 2024, *Bootstrapping Rare Object Detection in High-Resolution
Satellite Imagery* (arXiv:2403.02736) — the sampling surface, the cluster
inverse-size initialisation (their eq. 1), the online/proximity updates, and the
inverted-silhouette hyperparameter search from their Appendix A. Their verdicts,
worth knowing before choosing a clustering: **Bisecting KMeans wins** (F1 0.78 at
a 3K budget), **DBSCAN loses** (barely above uniform), online beats offline at
every budget, and ``w = max(P_0)`` is their update weight.

Nogueira, Zaytar, Ma et al. 2025, *Core-Set Selection for Data-efficient Land
Cover Segmentation* (arXiv:2505.01225) — Label Complexity (eq. 1), Feature
Diversity, Feature Activation (eq. 2), Class Balance (eq. 3), the hybrids
(eq. 4), and the Vendi-guided choice of K. Reference implementation:
https://github.com/keillernogueira/data-centric-rs-classification/

Not implemented here because it is a *training* change, not an acquisition
score, but it produced the largest single effect in either paper: the **RCE
loss** — cross-entropy on labelled pixels plus entropy minimisation on unlabelled
ones — took F1 from 0.01 to 0.51 at a 300-patch budget. See
`docs/research/ACTIVE_LEARNING.md`.

Note
----
`infer_patches.normalised_entropy` and `infer_patches.novelty` predate this
module and compute the same two quantities. They should be collapsed onto these
implementations when `infer_patches.py` is next touched; until then, changing
one means changing both.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# Model-in-the-loop: uncertainty over a posterior
# --------------------------------------------------------------------------

def normalised_entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row, divided by ``log(n_classes)``.

    *English.* How undecided the model is, on a 0-1 scale where 0 is "certain,
    one class" and 1 is "flat, every class equally likely". The classic
    uncertainty score, and the right one when the classes are comparable.

    *Caveat for this project.* Averaged over a patch that is 95% confidently
    stable, this measures the stable class's confidence and nothing else. Mask
    to the change pixels before reducing — that is what `entropy_change` does.

    Parameters
    ----------
    probs : (..., C) array
        Posterior, rows summing to 1.

    Returns
    -------
    (...) array in [0, 1].
    """
    probs = np.asarray(probs, dtype=np.float64)
    n_classes = probs.shape[-1]
    p = np.clip(probs, EPS, 1.0)
    ent = -(p * np.log(p)).sum(axis=-1)
    return ent / np.log(n_classes)


def least_confidence(probs: np.ndarray) -> np.ndarray:
    """``1 - max(p)``.

    *English.* How much probability mass is *not* on the model's answer. Cheap,
    and unlike entropy it ignores how the remainder is spread — which is what
    you want when only the top call will ever be acted on.
    """
    probs = np.asarray(probs, dtype=np.float64)
    return 1.0 - probs.max(axis=-1)


def margin(probs: np.ndarray) -> np.ndarray:
    """``1 - (p_first - p_second)``.

    *English.* How close the decision was. A cell where the top two classes are
    neck-and-neck sits on a decision boundary, and a label there moves the
    boundary; a cell that is uncertain but not *contested* (mass smeared over
    seven classes) does not. Higher is more informative, so the difference is
    inverted to keep every score in this module pointing the same way.
    """
    probs = np.asarray(probs, dtype=np.float64)
    part = np.partition(probs, -2, axis=-1)
    return 1.0 - (part[..., -1] - part[..., -2])


def bald(member_probs: np.ndarray) -> np.ndarray:
    """Mutual information between the prediction and the model parameters.

    ``BALD = H(mean_m p_m) - mean_m H(p_m)``

    *English.* Entropy asks "is this hard?"; BALD asks "is this hard *because
    the model does not know*, rather than because the pixel is genuinely
    ambiguous?" The first term is total uncertainty, the second is the part that
    survives even with a settled model (aleatoric — label noise, mixed pixels).
    The difference is the reducible part, and only the reducible part is worth
    buying a label for.

    *Why it is nearly free here.* The deployed recipe is already served as a
    5-seed ensemble, so the ``M`` members exist; MC dropout gives more if wanted.
    On a target whose change-F1 ceiling is set by ``Cropland/Nature`` label noise,
    separating the two terms is the difference between spending the budget on
    genuinely unmapped terrain and spending it re-labelling an ambiguous
    boundary that no amount of labels will fix.

    Parameters
    ----------
    member_probs : (M, N, C) array
        Posterior from each of ``M`` ensemble members / dropout passes.

    Returns
    -------
    (N,) array, normalised to [0, 1] by ``log(C)``.
    """
    member_probs = np.asarray(member_probs, dtype=np.float64)
    if member_probs.ndim != 3:
        raise ValueError("member_probs must be (M, N, C)")
    total = normalised_entropy(member_probs.mean(axis=0))
    expected = normalised_entropy(member_probs).mean(axis=0)
    return np.clip(total - expected, 0.0, 1.0)


def vote_entropy(member_probs: np.ndarray) -> np.ndarray:
    """Normalised entropy of the ensemble's hard-vote histogram.

    *English.* BALD's blunt cousin: throw away each member's confidence, keep
    only which class it picked, and measure the spread of votes. It is coarser
    than BALD (5 members give at most 5 distinct vote patterns) but it is the
    one disagreement measure that survives a member being badly calibrated,
    because calibration cannot change an arg-max.
    """
    member_probs = np.asarray(member_probs, dtype=np.float64)
    if member_probs.ndim != 3:
        raise ValueError("member_probs must be (M, N, C)")
    n_members, n_rows, n_classes = member_probs.shape
    votes = member_probs.argmax(axis=-1)                       # (M, N)
    counts = np.zeros((n_rows, n_classes), dtype=np.float64)
    for c in range(n_classes):
        counts[:, c] = (votes == c).sum(axis=0)
    return normalised_entropy(counts / n_members)


def conformal_set_size(probs: np.ndarray, qhat: np.ndarray | float,
                       normalise: bool = False) -> np.ndarray:
    """Cardinality of the LAC prediction set at the calibrated threshold.

    The LAC (least-ambiguous set-valued classifier) set is
    ``{c : 1 - p_c <= qhat_c}``, i.e. ``{c : p_c >= 1 - qhat_c}``.

    *English.* Not "how spread out is the posterior" but "how many classes can I
    not rule out at 90% confidence". The difference is that this number is
    *calibrated* — a set of size 3 means something quantitative about coverage,
    where an entropy of 0.7 does not.

    *Why this one is the project-specific pick.* `CONFORMAL_TORCHCP.md` already
    has Mondrian LAC fitted and validated, and the finding that makes it an
    acquisition function rather than a diagnostic is this: ``Cropland -> Nature``
    is in the 90% set 90% of the time, while the arg-max reaches it on 106 of
    21 M pixels. Set *membership* is a retrieval channel for the classes the map
    cannot paint. Pass the per-class ``qhat`` — the marginal `SplitPredictor`
    reads 0.8999 coverage while covering ``Cropland -> Nature`` 13% of the time,
    so a scalar threshold reintroduces exactly the blindness this is meant to fix.

    Parameters
    ----------
    probs : (N, C) array
    qhat : float or (C,) array
        Conformal quantile. Scalar = marginal, per-class = Mondrian.
    normalise : bool
        Divide by ``C`` to put the score on [0, 1] alongside the others.
    """
    probs = np.asarray(probs, dtype=np.float64)
    qhat = np.asarray(qhat, dtype=np.float64)
    in_set = probs >= (1.0 - qhat)
    size = in_set.sum(axis=-1).astype(np.float64)
    return size / probs.shape[-1] if normalise else size


def class_in_set(probs: np.ndarray, qhat: np.ndarray | float,
                 class_index: int) -> np.ndarray:
    """Whether one specific class survives into the conformal set.

    *English.* The retrieval channel for a dead class, stated as a boolean
    rather than a ranking. "Show me every cell where ``Artificial -> Cropland``
    cannot be ruled out at 90%" is an answerable question even where "show me
    every cell where it wins the arg-max" returns one pixel in 21 million.
    """
    probs = np.asarray(probs, dtype=np.float64)
    qhat = np.asarray(qhat, dtype=np.float64)
    thresh = qhat if np.ndim(qhat) == 0 else qhat[class_index]
    return probs[..., class_index] >= (1.0 - thresh)


# --------------------------------------------------------------------------
# Model-free: structure of the embedding pool
# --------------------------------------------------------------------------

def cluster_inverse_size(labels: np.ndarray, noise_label: int = -1,
                         noise_weight: float = 1.0) -> np.ndarray:
    """Zaytar et al. eq. 1 — ``P_i = 1 / (K * C_i)``.

    *English.* Cluster the cells, then make every *cluster* equally likely and
    spread each cluster's share evenly over its members. A cell in a
    thousand-member cluster (ordinary terrain, seen a thousand times) gets a
    thousandth of the weight of a cell in a singleton cluster. It is the
    "rare things look different from their surroundings" prior written down,
    and it needs no labels at all — which is why it is what you initialise with
    on a class the model cannot see.

    A nice property worth knowing: this is already a proper distribution.
    Summing over cells gives ``sum_k C_k / (K * C_k) = K/K = 1``, exactly, with
    no renormalisation.

    Parameters
    ----------
    labels : (N,) int array
        Cluster assignment per cell. DBSCAN's ``-1`` noise label is handled
        separately: noise points are *not* one cluster, they are N singletons,
        which is the interpretation that keeps the rare-is-interesting reading.
    noise_weight : float
        Multiplier applied to noise-point mass before the final renormalisation.
        Above 1 leans into unclustered oddities, below 1 treats them as junk.

    Returns
    -------
    (N,) array summing to 1.
    """
    labels = np.asarray(labels)
    p = np.zeros(labels.shape[0], dtype=np.float64)
    is_noise = labels == noise_label
    real = labels[~is_noise]
    uniq, counts = np.unique(real, return_counts=True)
    n_clusters = len(uniq) + int(is_noise.sum())      # noise points are singletons
    if n_clusters == 0:
        return np.full(labels.shape[0], 1.0 / labels.shape[0])
    size_of = dict(zip(uniq.tolist(), counts.tolist()))
    for i, lab in enumerate(labels.tolist()):
        if lab == noise_label:
            p[i] = noise_weight / n_clusters
        else:
            p[i] = 1.0 / (n_clusters * size_of[lab])
    return p / p.sum()


def feature_activation(features: np.ndarray, scale: bool = True) -> np.ndarray:
    """Core-set paper's FA score (their eq. 2), imagery only.

    ``gamma_i = -(1 - mu_i) * log(sigma_i)``, then min-max normalised and
    inverted so that high activation *and* high across-dimension variation
    score high.

    *English.* A one-number stand-in for "is there anything going on in this
    embedding". A cell whose feature vector is uniformly low and flat is open
    water or a parking lot — homogeneous, semantically empty, and by the core-set
    paper's own qualitative read the kind of example consistently *rejected* by
    every one of their six methods. A cell whose vector is both strong and uneven
    has several things in it. One pass over embeddings you already fetched.

    Read this before using it on AlphaEarth
    ---------------------------------------
    The paper's ``mu`` and ``sigma`` are **scaled to (0, 1] across the dataset
    before** ``gamma`` is formed, and their ``F_i`` are non-negative by
    construction because a ResNet-18 ReLU produced them. Both matter, and
    ``scale=False`` is only correct if your features already satisfy that:

    * with ``sigma`` in (0, 1], ``log(sigma) <= 0``, so ``gamma >= 0`` and the
      score is monotone in the intended direction;
    * with raw ``sigma > 1`` the sign of ``log(sigma)`` flips and the score
      inverts on part of the range — it silently starts preferring flat cells.

    **AlphaEarth embeddings are signed and roughly unit-norm, not ReLU
    activations.** ``mu`` is near zero and can be negative, so ``(1 - mu)`` no
    longer reads as "how inactive". Keep ``scale=True`` (which restores the
    intended monotonicity by construction), and treat FA on this modality as
    untested rather than transferred — it is a candidate for the replay harness,
    not a settled score.

    Parameters
    ----------
    features : (N, D) array
        Per-cell embedding.
    scale : bool
        Min-max ``mu`` and ``sigma`` into (0, 1] first, as the paper does.
    """
    features = np.asarray(features, dtype=np.float64)
    mu = features.mean(axis=1)
    sigma = features.std(axis=1)

    if scale:
        def _unit(v):
            lo, hi = v.min(), v.max()
            if hi - lo < EPS:
                return np.ones_like(v)
            return EPS + (1.0 - EPS) * (v - lo) / (hi - lo)
        mu, sigma = _unit(mu), _unit(sigma)
    sigma = np.clip(sigma, EPS, None)

    gamma = -(1.0 - mu) * np.log(sigma)
    lo, hi = gamma.min(), gamma.max()
    if hi - lo < EPS:
        return np.zeros_like(gamma)
    return 1.0 - (gamma - lo) / (hi - lo)


def vendi_score(features: np.ndarray, q: float = 1.0,
                normalise: bool = False) -> float:
    """Vendi score — the *effective number of distinct things* in a set.

    Friedman & Dieng 2023. Build the cosine similarity matrix ``K`` over ``n``
    unit-normalised rows, take the eigenvalues of ``K / n``, and exponentiate
    their Shannon entropy::

        VS = exp( -sum_i lam_i * log lam_i ),   lam = eig(K / n)

    *English.* Count how many genuinely different things are in a set, where
    near-duplicates count as one. 500 patches of the same savanna score ~1; 500
    mutually unlike patches score ~500. It is the number that answers "is my
    batch diverse, or is it 1,250 pictures of the same field", which no
    per-cell score can answer — every other model-free metric here rates cells
    one at a time and cannot see redundancy *between* the ones it picked.

    *It is Shannon entropy.* Exactly the same functional as
    `normalised_entropy` and `label_complexity`, applied to a different
    distribution: they take the entropy of a **class histogram**, this takes the
    entropy of a **similarity spectrum**, then exponentiates to get a count
    instead of a number of nats. Which is why it works on unlabelled cells,
    where a class histogram does not exist.

    Why it is affordable on 5.4 M cells
    -----------------------------------
    The n x n matrix is never formed. For a cosine kernel ``K = X X^T``, and
    ``X X^T / n`` has the same **nonzero** eigenvalues as the ``D x D`` matrix
    ``X^T X / n``; the other ``n - D`` eigenvalues are zero and contribute zero
    entropy. So with AlphaEarth's ``D = 64`` this is a 64x64 eigendecomposition
    on top of one pass over the data -- ``O(n D^2)``, seconds in BLAS for the
    whole globe, and **exact, not an approximation.**

    Parameters
    ----------
    features : (n, D) array
    q : float
        Renyi order. ``q = 1`` is the standard Vendi score (Shannon). ``q = inf``
        gives ``1 / max(lam)``, which is dominated by the single largest mode and
        is the pessimistic read of diversity. ``q < 1`` weights rare modes more.
    normalise : bool
        Divide by ``n`` to get "fraction of the maximum diversity possible at
        this sample size", so batches of different size can be compared.

    Returns
    -------
    float in [1, n]  (or [1/n, 1] if ``normalise``).
    """
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must be (n, D)")
    n_rows, n_dim = features.shape
    if n_rows == 0:
        return 0.0
    if n_rows == 1:
        return 1.0 / n_rows if normalise else 1.0

    x = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True),
                           EPS, None)
    # Whichever Gram matrix is smaller; the nonzero spectra are identical.
    gram = (x.T @ x) / n_rows if n_dim <= n_rows else (x @ x.T) / n_rows
    lam = np.linalg.eigvalsh(gram)
    lam = lam[lam > EPS]
    if lam.size == 0:
        return 1.0 / n_rows if normalise else 1.0
    lam = lam / lam.sum()          # sums to 1 in exact arithmetic; guard drift

    if np.isinf(q):
        score = 1.0 / lam.max()
    elif abs(q - 1.0) < 1e-9:
        score = float(np.exp(-(lam * np.log(lam)).sum()))
    else:
        score = float((lam ** q).sum() ** (1.0 / (1.0 - q)))
    return score / n_rows if normalise else score


def mean_within_cluster_vendi(features: np.ndarray, labels: np.ndarray,
                              normalise: bool = False) -> float:
    """The core-set paper's FD criterion: how homogeneous the clusters are.

    *English.* FD selects round-robin across clusters, so it is only diverse if
    the clusters themselves are internally boring — all the variety should sit
    *between* clusters, not inside them. This scores that: low means each cluster
    is one kind of place, so any member of it represents the rest.

    *The monotonicity, and how the paper handles it.* Raw within-cluster Vendi
    falls as ``K`` rises and bottoms out at exactly 1.0 when every cluster is a
    singleton, so **minimising it cannot choose K**. The paper does not minimise
    it — it walks ``K`` upward and stops on a *plateau*: when the change falls
    below ``delta`` for three consecutive steps. That is `search_k_vendi`, and it
    is the procedure to use.

    ``normalise=True`` is not from the paper. It divides each cluster's score by
    its size, which flips singletons from the best value to the worst and makes
    the raw criterion directly minimisable. Offered because it is occasionally
    convenient; the plateau rule is what was measured.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    scores = []
    for lab in np.unique(labels):
        members = features[labels == lab]
        scores.append(vendi_score(members, normalise=normalise))
    return float(np.mean(scores)) if scores else 0.0


def search_k_vendi(features: np.ndarray, cluster_fn, k_min: int = 2,
                   k_max: int = 200, delta: float = 0.005,
                   patience: int = 3) -> tuple[int, np.ndarray, list[float]]:
    """Choose ``K`` the way the core-set paper's FD does — a Vendi plateau.

    *English.* Start at K=2, cluster, measure the mean within-cluster Vendi,
    increment K, repeat; **stop when the score stops meaningfully moving** —
    a relative change under ``delta`` (their 0.5%) for ``patience`` consecutive
    steps. It is an elbow rule, not an optimum: the criterion falls forever, so
    what you are looking for is where buying more clusters stops buying more
    homogeneity.

    Parameters
    ----------
    features : (N, D) array
    cluster_fn : callable
        ``cluster_fn(features, k) -> (N,) labels``. Kept injectable so this
        module stays numpy-only and the caller picks the clustering — which is
        itself a tested axis (see `choose_clustering_by_silhouette`).
    delta : float
        Relative-change threshold. The paper's 0.5%.
    patience : int
        Consecutive steps under ``delta`` before stopping. The paper's 3.

    Returns
    -------
    (best_k, labels_at_best_k, trace) — ``trace`` is the mean Vendi per K, so
    the plateau can be plotted rather than trusted.
    """
    features = np.asarray(features, dtype=np.float64)
    trace: list[float] = []
    prev: float | None = None
    stable = 0
    best_k, best_labels = k_min, None
    for k in range(k_min, k_max + 1):
        labels = np.asarray(cluster_fn(features, k))
        score = mean_within_cluster_vendi(features, labels)
        trace.append(score)
        best_k, best_labels = k, labels
        if prev is not None:
            rel = abs(score - prev) / max(abs(prev), EPS)
            stable = stable + 1 if rel < delta else 0
            if stable >= patience:
                break
        prev = score
    return best_k, best_labels, trace


def silhouette_score(features: np.ndarray, labels: np.ndarray,
                     sample_size: int | None = 10_000,
                     rng: np.random.Generator | None = None) -> float:
    """Rousseeuw's silhouette: ``s_i = (b_i - a_i) / max(a_i, b_i)``, averaged.

    ``a_i`` is the mean distance to the point's own cluster, ``b_i`` the mean
    distance to the nearest *other* cluster. Ranges −1 (everything in the wrong
    cluster) to 1 (cleanly separated).

    *English.* How well-separated the clustering is. The standard use is to
    maximise it. **The bootstrapping paper does the opposite**, and that is the
    part worth knowing — see `choose_clustering_by_silhouette`.

    ``sample_size`` subsamples before the O(m²) distance matrix; at 5.4 M cells
    the full computation is not an option and the subsample is the standard fix.
    """
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    rng = np.random.default_rng() if rng is None else rng
    n_rows = features.shape[0]
    if sample_size is not None and n_rows > sample_size:
        idx = rng.choice(n_rows, size=sample_size, replace=False)
        features, labels = features[idx], labels[idx]
        n_rows = sample_size

    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    dist = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=-1)
    sil = np.zeros(n_rows)
    for i in range(n_rows):
        own = labels == labels[i]
        n_own = own.sum()
        a = (dist[i, own].sum() / (n_own - 1)) if n_own > 1 else 0.0
        b = np.inf
        for lab in uniq:
            if lab == labels[i]:
                continue
            other = labels == lab
            if other.any():
                b = min(b, dist[i, other].mean())
        sil[i] = 0.0 if not np.isfinite(b) else (b - a) / max(a, b, EPS)
    return float(sil.mean())


def choose_clustering_by_silhouette(candidates: dict) -> tuple[str, float]:
    """The bootstrapping paper's Appendix A — pick hyperparameters with no labels.

    *English, and this is the counterintuitive part.* You have no labels, so you
    cannot tune the clustering against the thing you care about (how fast it
    finds rare objects). The paper's answer: the silhouette score is a usable
    stand-in — it correlates strongly with the number of samples needed to find
    100 positives (**R² = 0.93** with MOSAIKS/RCF features) — and the number of
    samples is minimised where the silhouette score is **minimised**, not
    maximised. They run a Bayesian search minimising ``|silhouette|``.

    That inversion is the whole point and it is easy to get backwards. Clean,
    well-separated clusters describe the *common* terrain that dominates the
    pool. A clustering that scores badly by the usual standard is one that has
    been pulled apart by oddities — and oddities are the target. **Do not
    substitute the usual "maximise silhouette to pick K" here.**

    Parameters
    ----------
    candidates : dict
        ``{name: silhouette_score}`` over feature-representation × clustering ×
        hyperparameter combinations.

    Returns
    -------
    ``(name, score)`` of the minimum ``|silhouette|``.
    """
    if not candidates:
        raise ValueError("no candidates")
    name = min(candidates, key=lambda k: abs(candidates[k]))
    return name, candidates[name]


def feature_diversity(labels: np.ndarray, budget: int | None = None,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """The core-set paper's FD selection: one cell per cluster, round-robin.

    *English.* Deal the budget out across clusters like cards, one each, then go
    round again. It guarantees every kind of place is represented before any kind
    is represented twice — which is the opposite failure mode from
    `kcenter_greedy`, whose furthest-point rule happily spends the budget on
    outliers. FD trades k-center's coverage guarantee for robustness to a cloud.

    Order within a cluster is random, so this is seeded. That randomness is why
    the paper notes FD scores "lack a fixed ordering" -- what is guaranteed is
    the round, not the member.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng() if rng is None else rng
    buckets = []
    # The paper randomises the *starting* cluster as well as the member order;
    # a fixed cluster order would systematically favour whichever label sorts
    # first whenever the budget is not a multiple of K.
    for lab in rng.permutation(np.unique(labels)):
        members = np.flatnonzero(labels == lab)
        buckets.append(rng.permutation(members))
    n_total = labels.shape[0]
    budget = n_total if budget is None else min(budget, n_total)

    picked: list[int] = []
    depth = 0
    while len(picked) < budget:
        progressed = False
        for bucket in buckets:
            if depth < len(bucket):
                picked.append(int(bucket[depth]))
                progressed = True
                if len(picked) >= budget:
                    break
        if not progressed:
            break
        depth += 1
    return np.array(picked, dtype=int)


def order_to_score(order: np.ndarray, n_total: int) -> np.ndarray:
    """Turn a selection order into the paper's ``s_i = 1 - r_i / N`` score.

    *English.* Both FD and CB produce a *ranking*, not a number, and a ranking
    cannot be blended with entropy or novelty. This converts one to the other:
    first-selected scores 1, last scores 0, never-selected scores 0. Needed
    before `convex_blend`; `rank_mean` would also accept the raw order.
    """
    score = np.zeros(n_total, dtype=np.float64)
    order = np.asarray(order, dtype=int)
    if order.size == 0:
        return score
    score[order] = 1.0 - np.arange(order.size, dtype=np.float64) / n_total
    return score


def kcenter_greedy(features: np.ndarray, budget: int,
                   selected: np.ndarray | None = None,
                   metric: str = "cosine") -> np.ndarray:
    """Sener & Savarese core-set: repeatedly take the least-covered cell.

    *English.* Build the subset that leaves nothing far from it. Start from what
    you already have, then take the cell whose distance to its nearest selected
    neighbour is largest, add it, repeat. It is pure coverage: it will never pick
    two cells that look alike, and it will happily pick an outlier — which on
    satellite embeddings is sometimes a genuinely unrepresented biome and
    sometimes a cloud.

    *In this project* the natural ``selected`` seed is the 6,490 already-labelled
    plots. Then the score is literally "how much of the world does the existing
    label set fail to represent", and the greedy order is the fix list.

    Parameters
    ----------
    features : (N, D)
    budget : int
        Number of cells to select.
    selected : (M,) int array, optional
        Indices already in the set. If None, starts from the medoid-ish first
        point (index 0 after a deterministic farthest-first warm start).
    metric : {"cosine", "euclidean"}
        The core-set paper's `CoreSet` baseline uses **Euclidean** on ResNet-18
        activations. Cosine is the default here because AlphaEarth embeddings are
        near-unit-norm and the rest of this module (novelty, Vendi) is cosine;
        pass ``"euclidean"`` to reproduce the published baseline.

    Returns
    -------
    (budget,) int array of selected indices, in selection order.

    Notes
    -----
    O(N * budget * D). At N = 5.4 M this is a shortlist tool, not a global one —
    cluster first, run this within the cluster.
    """
    features = np.asarray(features, dtype=np.float64)
    n_rows = features.shape[0]
    if metric == "cosine":
        norms = np.clip(np.linalg.norm(features, axis=1, keepdims=True), EPS, None)
        feats = features / norms
    elif metric == "euclidean":
        feats = features
    else:
        raise ValueError(f"unknown metric {metric!r}")

    def _dist(idx: int) -> np.ndarray:
        if metric == "cosine":
            return 1.0 - feats @ feats[idx]
        return np.linalg.norm(feats - feats[idx], axis=1)

    if selected is None or len(selected) == 0:
        start = int(np.argmax(np.linalg.norm(feats - feats.mean(axis=0), axis=1)))
        min_dist = _dist(start)
        picked = [start]
    else:
        min_dist = np.full(n_rows, np.inf)
        for idx in np.asarray(selected).tolist():
            min_dist = np.minimum(min_dist, _dist(int(idx)))
        picked = []

    while len(picked) < budget:
        nxt = int(np.argmax(min_dist))
        if not np.isfinite(min_dist[nxt]):
            break
        picked.append(nxt)
        min_dist = np.minimum(min_dist, _dist(nxt))
        min_dist[nxt] = -np.inf
    return np.array(picked[:budget], dtype=int)


def novelty_to_reference(features: np.ndarray, reference: np.ndarray,
                         quantile: float = 0.90) -> float:
    """Cosine distance from each cell to its nearest *labelled* plot.

    *English.* "Is this unlike anything anyone has already labelled?" — measured
    against the label set, not against the rest of the pool. The distinction is
    the point: a patch that is unremarkable globally but unlike every existing
    plot is precisely the patch worth an afternoon, and a pool-referenced score
    would rank it as ordinary.

    Reduced at the 90th percentile rather than the mean, because the pocket of
    unfamiliar land inside an otherwise ordinary patch is the thing being bought.

    Parameters
    ----------
    features : (N, D) array
        Pixels (or cells) in the candidate.
    reference : (M, D) array
        The labelled plots' embeddings.
    quantile : float

    Returns
    -------
    float in [0, 2]; 0 means every pixel has a near-identical labelled twin.
    """
    features = np.asarray(features, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    f = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), EPS, None)
    r = reference / np.clip(np.linalg.norm(reference, axis=1, keepdims=True), EPS, None)
    nearest = (f @ r.T).max(axis=1)
    return float(np.quantile(1.0 - nearest, quantile))


# --------------------------------------------------------------------------
# Composition-driven: what a cell would add to the training set
# --------------------------------------------------------------------------

def label_complexity(counts: np.ndarray,
                     ignore_index: int | list[int] | None = None) -> np.ndarray:
    """Core-set paper's LC (their eq. 1) — entropy of the class histogram
    *within* a cell.

    *English.* "How many different things are in this patch?" A 5x5 km patch that
    is 100% stable Nature teaches one class in one context; a patch holding four
    classes and the boundaries between them teaches the boundaries too, which is
    where the model is wrong. It is the only score here that prefers *mixed*
    ground, and it needs no model — on a labelled pool it runs on the mask, on an
    unlabelled pool on predicted counts.

    *Do not skip ``ignore_index``.* The paper drops "unknown"/"ignored" classes
    from the computation, and here that is not a formality: 13 of the 100 pilot
    patches came back all-water, and AlphaEarth-invalid pixels are a large share
    of coastal cells. Counted as a class, nodata reads as *variety* — an all-water
    patch with a sliver of land would score as one of the most complex cells on
    the map.

    Parameters
    ----------
    counts : (N, C) array
        Pixel count (or area) per class per cell. Rows are normalised here.
    ignore_index : int or list of int, optional
        Column(s) to drop before normalising — nodata, cloud, "unknown".
    """
    counts = np.asarray(counts, dtype=np.float64)
    if ignore_index is not None:
        drop = ([ignore_index] if isinstance(ignore_index, (int, np.integer))
                else list(ignore_index))
        keep = [c for c in range(counts.shape[-1]) if c not in drop]
        if not keep:
            raise ValueError("ignore_index drops every class")
        counts = counts[..., keep]
    total = np.clip(counts.sum(axis=-1, keepdims=True), EPS, None)
    return normalised_entropy(counts / total)


def class_balance_greedy(counts: np.ndarray, budget: int,
                         prior: np.ndarray | None = None) -> np.ndarray:
    """Core-set paper's CB — greedily flatten the *cumulative* class histogram.

    *English.* Every other score here rates a cell on its own. This one rates it
    against what has already been picked: at each step take the cell that, added
    to the running total, leaves the class distribution flattest. The result is a
    selection order that automatically chases whichever class is currently
    scarcest, without anyone having to set per-class targets by hand.

    *Why it fits this project.* `PATCH_SAMPLING.md` section B sets the campaign's
    objective as "double every reachable change class", binding on
    ``Cropland -> Artificial``. That is a class-balance objective stated as a
    quota; this is the same objective stated as a rule, and it re-prioritises for
    free as labels come back and the binding class changes.

    Parameters
    ----------
    counts : (N, C) array
        Class counts each candidate would contribute.
    budget : int
    prior : (C,) array, optional
        Counts already held — the existing 6,490 plots' class histogram. Passing
        it is what makes the first pick chase the deficit rather than the mode.

    Returns
    -------
    (budget,) int array of indices, in selection order.

    Notes
    -----
    O(budget * N * C). Intended for a shortlist of <= ~1e5 candidates. Over the
    full global grid, pre-filter with `deficit_weighted_yield` and run this on
    the survivors.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n_rows, n_classes = counts.shape
    cum = (np.zeros(n_classes) if prior is None
           else np.asarray(prior, dtype=np.float64).copy())
    available = np.ones(n_rows, dtype=bool)
    picked: list[int] = []
    for _ in range(min(budget, n_rows)):
        cand = cum[None, :] + counts                        # (N, C)
        score = label_complexity(cand)      # counts, not probabilities
        score[~available] = -np.inf
        nxt = int(np.argmax(score))
        picked.append(nxt)
        cum = cum + counts[nxt]
        available[nxt] = False
    return np.array(picked, dtype=int)


def deficit_weighted_yield(counts: np.ndarray, deficit: np.ndarray,
                           precision: np.ndarray, min_px: int = 100,
                           max_points: int = 3) -> np.ndarray:
    """Expected *confirmed* plots per cell, weighted by how badly each class is
    wanted.

    ``yield_i(c) = min(max_points, floor(counts[i, c] / min_px))``
    ``score_i    = sum_c yield_i(c) * precision[c] * w_c``,
    with ``w_c = deficit[c] / sum(deficit)``.

    *English.* The plan's arithmetic, as a score. Three things it refuses to
    confuse, each of which moves the answer a lot:

    * **A hectare is the unit, not a pixel.** A class occupying 40 scattered
      pixels cannot be labelled however much it is wanted, hence the
      ``min_px`` floor and the cap at ``max_points`` — points a few hundred
      metres apart are close to the same observation.
    * **Predicted is not confirmed.** The change classes run at OOF precision
      0.31-0.51, so three predicted ``Nature -> Artificial`` points return about
      1.4 real ones. Sizing on the unadjusted yield is the easiest way to plan a
      round that comes up short.
    * **Deficit, not abundance.** A cell full of stable Nature has an enormous
      raw yield and is worth nothing.

    Parameters
    ----------
    counts : (N, C)
    deficit : (C,) — plots still wanted per class; zeros for satisfied classes.
    precision : (C,) — out-of-fold precision per class. Set to 1.0 for a
        retrieval channel whose confirm rate has not been measured, and treat
        the result as the lower bound it is.
    """
    counts = np.asarray(counts, dtype=np.float64)
    deficit = np.asarray(deficit, dtype=np.float64)
    precision = np.asarray(precision, dtype=np.float64)
    per_class = np.minimum(max_points, np.floor(counts / min_px))
    total = deficit.sum()
    weight = deficit / total if total > 0 else np.zeros_like(deficit)
    return (per_class * precision[None, :] * weight[None, :]).sum(axis=1)


# --------------------------------------------------------------------------
# Combining scores
# --------------------------------------------------------------------------

def rank_mean(*scores: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Average the *percentile ranks* of several scores, not the scores.

    *English.* A cosine distance and a normalised entropy are on incomparable
    scales, and a raw sum lets whichever happens to have the wider spread decide
    the whole ordering. Converting each to a percentile rank first makes the
    combination mean what it says. This is what the pilot's
    ``novelty_p90 + entropy_change`` ranking already does.

    Keep the components as separate columns as well — entropy finds boundaries
    and mixed pixels, novelty finds unrepresented biomes, and they are not the
    same cells.
    """
    if not scores:
        raise ValueError("rank_mean needs at least one score")
    stack = []
    for s in scores:
        s = np.asarray(s, dtype=np.float64)
        order = s.argsort(kind="stable")
        ranks = np.empty(len(s), dtype=np.float64)
        ranks[order] = np.arange(len(s), dtype=np.float64)
        stack.append(ranks / max(len(s) - 1, 1))
    stack = np.stack(stack, axis=0)
    if weights is None:
        return stack.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    return (stack * (w / w.sum())[:, None]).sum(axis=0)


def convex_blend(score_a: np.ndarray, score_b: np.ndarray,
                 lam: float = 0.5) -> np.ndarray:
    """Core-set paper's FA/CB hybrid — ``lam * a + (1 - lam) * b``.

    *English.* The simplest possible combination, and the honest one when both
    inputs are already on [0, 1]. Use `rank_mean` when they are not.
    """
    return lam * np.asarray(score_a, dtype=np.float64) + \
        (1.0 - lam) * np.asarray(score_b, dtype=np.float64)


def cutoff_hybrid(order_a: np.ndarray, order_b: np.ndarray,
                  cutoff: int) -> np.ndarray:
    """Core-set paper's LC/FD hybrid — take ``cutoff`` from A, then fill from B.

    *English.* Not a blend but a regime switch, and it encodes a real claim: at
    small budgets what you lack is *coverage*, so lead with diversity; past some
    size the pool is covered and what you lack is *hard examples*, so switch to
    complexity. ``cutoff`` is where you think that crossover is, and it is a
    guess until measured.
    """
    order_a = np.asarray(order_a, dtype=int)
    order_b = np.asarray(order_b, dtype=int)
    head = order_a[:cutoff]
    seen = set(head.tolist())
    tail = [i for i in order_b.tolist() if i not in seen]
    return np.concatenate([head, np.array(tail, dtype=int)])


# --------------------------------------------------------------------------
# The sampling surface: offline init, online update
# --------------------------------------------------------------------------

class SamplingSurface:
    """A discrete probability distribution over grid cells, sampled without
    replacement and optionally reweighted by returned labels.

    This is Zaytar et al.'s system. `initialise` is the offline half — any score
    in this module can supply it — and `proximity_update` / `cluster_update` are
    the online half.

    The one adaptation this project needs
    -------------------------------------
    In the paper every positive is wanted, so a hit always *raises* its
    neighbours. Here "hit" is class-conditional and the two directions are both
    right, for different classes: confirming a ``Cropland -> Artificial`` plot
    should raise its neighbourhood (that class is 333 plots short and change
    clusters in space), while confirming yet another stable ``Nature`` plot
    should lower it (4,200 held, nothing to learn). So `weight` is signed, and
    the sign is the class's deficit sign. Pass a negative weight for a satisfied
    class; that turns the same mechanism into a redundancy penalty.
    """

    def __init__(self, p: np.ndarray, xy: np.ndarray | None = None,
                 clusters: np.ndarray | None = None,
                 floor: float = 0.0):
        self.p = np.asarray(p, dtype=np.float64).copy()
        self.xy = None if xy is None else np.asarray(xy, dtype=np.float64)
        self.clusters = None if clusters is None else np.asarray(clusters)
        self.floor = float(floor)
        self.drawn = np.zeros(self.p.shape[0], dtype=bool)
        self._normalise()

    def _normalise(self) -> None:
        p = np.clip(self.p, self.floor, None)
        p[self.drawn] = 0.0
        total = p.sum()
        if total <= 0:
            live = ~self.drawn
            p = live.astype(np.float64)
            total = max(p.sum(), EPS)
        self.p = p / total

    @classmethod
    def uniform(cls, n_cells: int, **kwargs) -> "SamplingSurface":
        """The baseline every method has to beat: equal weight everywhere.

        Do not skip it. `PATCH_SAMPLING.md` reports the equal-area draw's own
        yield per class, and that is the number an acquisition function has to
        improve on to be worth its complexity.
        """
        return cls(np.full(n_cells, 1.0 / n_cells), **kwargs)

    @classmethod
    def from_score(cls, score: np.ndarray, temperature: float = 1.0,
                   **kwargs) -> "SamplingSurface":
        """Softmax a score into a surface.

        ``temperature`` is the exploit/explore dial: ->0 collapses onto the
        arg-max (greedy, and blind wherever the score is), large flattens toward
        uniform. Sampling rather than taking the top-k is deliberate — a
        deterministic top-k on a wrong score has no way to discover it is wrong.
        """
        score = np.asarray(score, dtype=np.float64)
        z = (score - score.max()) / max(temperature, EPS)
        p = np.exp(z)
        return cls(p / p.sum(), **kwargs)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` cells without replacement and mark them drawn."""
        live = np.flatnonzero(~self.drawn)
        if len(live) == 0:
            return np.array([], dtype=int)
        n = min(n, len(live))
        idx = rng.choice(live, size=n, replace=False, p=self.p[live] / self.p[live].sum())
        self.drawn[idx] = True
        self._normalise()
        return idx

    def proximity_update(self, hit_index: int, radius_m: float,
                         weight: float) -> int:
        """Reweight cells within ``radius_m`` of a hit. Returns cells touched.

        *English.* "Rare things cluster." A confirmed cropland-to-built
        conversion sits at the edge of an expanding town, and the next one is
        probably a kilometre away, not on another continent. The radius is the
        scale you believe the process has — the paper used 200 m for cattle
        enclosures; a land-cover conversion front is kilometres, so this is a
        parameter to set from the phenomenon and not to copy across.
        """
        if self.xy is None:
            raise ValueError("proximity_update needs cell coordinates (xy)")
        d = np.linalg.norm(self.xy - self.xy[hit_index], axis=1)
        near = d <= radius_m
        self.p[near] += weight
        self._normalise()
        return int(near.sum())

    def cluster_update(self, hit_index: int, weight: float) -> int:
        """Reweight every cell in the same cluster as a hit.

        *English.* The non-spatial half of the same idea — "things that *look*
        like this are also worth looking at", which is the only channel that can
        carry a hit across a continent. In the paper this beat proximity at every
        budget, and it is the one to lean on for a class whose instances are
        globally scattered rather than locally clustered.

        *Gotcha.* A hit inside a very large cluster raises nearly every cell by
        the same amount, which after renormalisation is close to a no-op. The
        update only carries information when the clustering is fine enough that
        a cluster means something -- check the returned count against the grid
        size, and re-cluster if one cluster holds most of the world.
        """
        if self.clusters is None:
            raise ValueError("cluster_update needs cluster labels")
        same = self.clusters == self.clusters[hit_index]
        self.p[same] += weight
        self._normalise()
        return int(same.sum())
