"""Leave-one-out medoid anomaly score over a plot's annual embedding trajectory.

Following Wherobots' AlphaEarth change-detection write-up: to judge how unusual a
given year's embedding is relative to its temporal context, compare it to the
*medoid* of the other years rather than their mean. The medoid -- the member that
minimises total distance to the rest -- is a robust central reference that a
single anomalous year cannot drag around, which matters with only a handful of
annual observations.

For a plot with unit-norm annual vectors ``e_y`` (AlphaEarth is L2-normalised, so
cosine distance ``1 - <a,b>`` is the natural metric):

    medoid_{-t} = argmin_{j != t} sum_{k != t} d(e_j, e_k)
    loo_score_t = d(e_t, medoid_{-t})

A high score means year ``t`` sits far from the robust centre of the other years
-- the signature of a real change at (or around) that year.
"""
from __future__ import annotations

import numpy as np


def loo_medoid_scores(traj: np.ndarray) -> np.ndarray:
    """Per-year LOO medoid cosine-distance score.

    ``traj`` is ``(n_plots, n_years, dim)`` and assumed row-unit-norm. Returns
    ``(n_plots, n_years)``: for each plot/year, the cosine distance from that
    year's embedding to the medoid of the *other* years.
    """
    if traj.ndim != 3:
        raise ValueError(f"expected (n_plots, n_years, dim), got {traj.shape}")
    n, t, _ = traj.shape
    if t < 3:
        raise ValueError(f"need >=3 years for a leave-one-out medoid, got {t}")
    # Pairwise cosine distance within each plot: d = 1 - e_j . e_k (unit norm).
    dist = 1.0 - np.einsum("ntd,nsd->nts", traj, traj)
    np.clip(dist, 0.0, 2.0, out=dist)
    # Total distance from each year to all years (self term is 0, so harmless).
    rowsum = dist.sum(axis=2)  # (n, t)
    scores = np.empty((n, t), dtype=float)
    rows = np.arange(n)
    for ti in range(ti_end := t):
        # Medoid of the years other than ti: minimise sum of distances to the
        # remaining years, i.e. drop ti's contribution from each candidate's sum.
        cand = rowsum - dist[:, :, ti]  # sum over k != ti
        cand[:, ti] = np.inf            # ti itself is not a candidate medoid
        medoid = cand.argmin(axis=1)    # (n,)
        scores[:, ti] = dist[rows, ti, medoid]
    return scores


def loo_summary(scores: np.ndarray, years: list[int]) -> dict[str, np.ndarray]:
    """Compact per-plot descriptors of the LOO score trajectory."""
    first, last = years.index(min(years)), years.index(max(years))
    return {
        "loo_start": scores[:, first],
        "loo_end": scores[:, last],
        "loo_max": scores.max(axis=1),
        "loo_mean": scores.mean(axis=1),
        "loo_end_minus_start": scores[:, last] - scores[:, first],
    }
