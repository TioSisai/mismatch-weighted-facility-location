"""Median kernel bandwidth estimation and weighted greedy coverage for FL."""

from __future__ import annotations

import numpy as np

from . import backends
from .backends import pairwise_euclidean_distances, pairwise_sq_distances


def _sampled_pair_sq(emb: np.ndarray, max_samples: int, random_state) -> np.ndarray:
    """Sample at most max_samples points without replacement and return the strict upper triangle of the squared distance matrix."""
    emb = np.asarray(emb, dtype=np.float32)
    n = emb.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(random_state)
        emb = emb[rng.choice(n, size=max_samples, replace=False)]
    dist = pairwise_sq_distances(emb, emb)
    return dist[np.triu_indices(emb.shape[0], k=1)]


def median_sq_bandwidth(emb: np.ndarray, *, max_samples: int = 2000, random_state=None) -> float:
    """Estimate the kernel bandwidth σ² using the median of pairwise squared Euclidean distances.

    Use the mean of positive distances when the median is zero, or 1 when there are no positive distances.

    Args:
        emb: Point set, ``[N, D]``.
        max_samples: Maximum number of points sampled without replacement.
        random_state: Sampling seed.

    Returns:
        Positive kernel bandwidth σ².
    """
    pair = _sampled_pair_sq(emb, max_samples, random_state)
    if pair.size == 0:
        return 1.0
    sigma2 = float(np.median(pair))
    if sigma2 > 0:
        return sigma2
    positive = pair[pair > 0]
    return float(positive.mean()) if positive.size else 1.0


def weighted_facility_location_greedy(
    emb: np.ndarray,
    cand_idx: np.ndarray,
    base_idx: np.ndarray,
    weights: np.ndarray,
    n: int,
    *,
    sigma2: float | None = None,
    dist_matrix: np.ndarray | None = None,
    random_state=None,
) -> list[int]:
    """Greedily select n points from the candidates to maximize the weighted facility location objective.

    Candidates also serve as the points to cover, with RBF kernel ``exp(-d² / (2σ²))``.

    Args:
        emb: Segment representations for the full pool, ``[N, D]``.
        cand_idx: Global indices of candidates, ``[M]``.
        base_idx: Global indices of the initial coverage, which may be empty.
        weights: Nonnegative weights aligned with cand_idx, ``[M]``.
        n: Number of samples to select, at most M.
        sigma2: Kernel bandwidth, estimated from the candidates when None.
        dist_matrix: Optional Euclidean distance cache for the full pool, ``[N, N]``.
        random_state: Sampling seed for bandwidth estimation.

    Returns:
        n global indices in selection order.

    Raises:
        ValueError: There are insufficient candidates.
    """
    emb = np.asarray(emb, dtype=np.float32)
    cand_idx = np.asarray(cand_idx, dtype=int)
    base_idx = np.asarray(base_idx, dtype=int)
    m = cand_idx.shape[0]
    if n > m:
        raise ValueError(f"facility location requires n({n}) <= number of candidates({m})")
    if sigma2 is None:
        sigma2 = median_sq_bandwidth(emb[cand_idx], random_state=random_state)

    # Kernel values use float32, and the backend is determined at call time.
    xp = backends.cp if backends.HAS_CUPY else np
    inv = xp.float32(1.0 / (2.0 * sigma2))
    weights = xp.asarray(np.asarray(weights, dtype=np.float32))

    if dist_matrix is not None:
        d_cc = dist_matrix[np.ix_(cand_idx, cand_idx)]
        d_cb = dist_matrix[np.ix_(cand_idx, base_idx)] if base_idx.size > 0 else None
    else:
        cand_emb = emb[cand_idx]
        d_cc = pairwise_euclidean_distances(cand_emb, cand_emb)
        d_cb = pairwise_euclidean_distances(cand_emb, emb[base_idx]) if base_idx.size > 0 else None

    def rbf(dist):
        # Convert to float32 before squaring to maintain consistent precision between the cached and recomputed paths.
        dist = xp.asarray(dist, dtype=xp.float32)
        return xp.exp(-(dist * dist) * inv)

    k_cc = rbf(d_cc)  # [M, M]
    cov = rbf(d_cb).max(axis=1) if d_cb is not None else xp.zeros(m, dtype=xp.float32)

    chosen = xp.zeros(m, dtype=bool)
    selected: list[int] = []
    for _ in range(n):
        gain = (weights[:, None] * xp.maximum(xp.float32(0.0), k_cc - cov[:, None])).sum(axis=0)  # [M]
        gain[chosen] = -xp.inf
        j = int(xp.argmax(gain))
        selected.append(int(cand_idx[j]))
        chosen[j] = True
        cov = xp.maximum(cov, k_cc[:, j])
    return selected
