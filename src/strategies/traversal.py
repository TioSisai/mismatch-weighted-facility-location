"""Optionally weighted farthest-point traversal and cold starts from a random point."""

from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from .backends import pairwise_euclidean_distances


def farthest_first(
    emb: np.ndarray,
    cand_idx: np.ndarray,
    base_idx: np.ndarray,
    n: int,
    *,
    weights: np.ndarray | None = None,
    dist_matrix: np.ndarray | None = None,
) -> list[int]:
    """Use base_idx as the initial coverage and iteratively select ``argmax_c w_c·d_min(c)``.

    d_min is the minimum Euclidean distance to the coverage set, updated without weighting, with weights of 1 when weights is omitted.
    After weighting, distances and squared distances have different rankings. Select the first candidate first when the initial coverage is empty.

    Args:
        emb: Segment representations for the full pool, ``[N, D]``.
        cand_idx: Global indices of candidates, with at least n entries.
        base_idx: Global indices of the initial coverage, which may be empty.
        n: Number of samples to select.
        weights: Optional nonnegative weights aligned with cand_idx.
        dist_matrix: Optional Euclidean distance cache for the full pool, ``[N, N]``.

    Returns:
        n global indices in selection order.

    Raises:
        ValueError: There are insufficient candidates.
    """
    cand = np.asarray(cand_idx, dtype=int)
    if len(cand) < n:
        raise ValueError(f"farthest traversal requires n({n}) <= number of candidates({len(cand)})")
    base_idx = np.asarray(base_idx, dtype=int)
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    if dist_matrix is not None:
        base_dist = (
            dist_matrix[np.ix_(cand, base_idx)].min(axis=1)
            if base_idx.size > 0 else np.full(len(cand), np.inf)
        )

        def column(j: int) -> np.ndarray:
            return dist_matrix[cand, cand[j]]
    else:
        cand_emb = emb[cand]
        base_dist = (
            pairwise_euclidean_distances(cand_emb, emb[base_idx]).min(axis=1)
            if base_idx.size > 0 else np.full(len(cand), np.inf)
        )

        def column(j: int) -> np.ndarray:
            return pairwise_euclidean_distances(cand_emb, cand_emb[j:j + 1]).ravel()

    d_min = base_dist
    chosen = np.zeros(len(cand), dtype=bool)
    selected: list[int] = []
    for _ in range(n):
        score = d_min if w is None else w * d_min
        j = int(np.argmax(np.where(chosen, -np.inf, score)))
        selected.append(int(cand[j]))
        chosen[j] = True
        d_min = np.minimum(d_min, column(j))
    return selected


def random_seed_farthest_first(
    emb: np.ndarray,
    cand_idx: np.ndarray,
    n: int,
    random_state,
    *,
    dist_matrix: np.ndarray | None = None,
) -> list[int]:
    """Select the first point randomly, then use unweighted farthest-point traversal to reach n points.

    Args:
        emb: Segment representations for the full pool, ``[N, D]``.
        cand_idx: Global indices of candidates, with at least n entries.
        n: Number of samples to select.
        random_state: Random seed for the first point.
        dist_matrix: Optional Euclidean distance cache for the full pool, ``[N, N]``.

    Returns:
        n global indices in selection order.

    Raises:
        ValueError: There are insufficient candidates.
    """
    cand = np.asarray(cand_idx, dtype=int)
    if len(cand) < n:
        raise ValueError(f"random seed farthest traversal requires n({n}) <= number of candidates({len(cand)})")
    rng = check_random_state(random_state)
    first_pos = int(rng.randint(len(cand)))
    first = int(cand[first_pos])
    if n == 1:
        return [first]
    rest = np.delete(cand, first_pos)
    tail = farthest_first(emb, rest, np.array([first], dtype=int), n - 1, dist_matrix=dist_matrix)
    return [first] + tail
