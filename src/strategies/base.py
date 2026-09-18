"""Query interface and the selection procedure shared by MF strategies for grouping by mismatch."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


class QueryStrategy:
    """Base class for segment queries, with subclasses implementing _select.

    Args:
        random_state: Seed for cold starts, bandwidth estimation, and random sampling.
    """

    def __init__(self, *, random_state=None):
        self.random_state = random_state

    def query(
        self,
        embeddings: np.ndarray,
        candidates: np.ndarray,
        labeled: np.ndarray,
        *,
        seg_proba: np.ndarray | None = None,
        seg_labels: np.ndarray | None = None,
        batch_size: int,
        dist_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select batch_size segments from the candidate pool.

        Args:
            embeddings: Segment representations for the full pool, ``[N, D]``.
            candidates: Global indices of unlabeled segments, with at least batch_size entries.
            labeled: Global indices of labeled segments, with an empty array triggering a cold start.
            seg_proba: Segment probabilities for the full pool, ``[N, C]``, or None for a cold start.
            seg_labels: Segment multilabel targets for the full pool, ``[N, C]``.
            batch_size: Number of samples to select.
            dist_matrix: Optional Euclidean distance cache for the full pool, ``[N, N]``.

        Returns:
            Integer indices in selection order, ``[batch_size]``.

        Raises:
            ValueError: There are insufficient candidates.
        """
        emb = np.asarray(embeddings, dtype=np.float32)
        cand = np.asarray(candidates, dtype=int)
        labeled = np.asarray(labeled, dtype=int)
        if len(cand) < batch_size:
            raise ValueError(f"number of candidates({len(cand)}) < batch_size({batch_size})")
        selected = self._select(emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix)
        return np.asarray(selected[:batch_size], dtype=int)

    def _select(
        self,
        emb: np.ndarray,
        cand: np.ndarray,
        labeled: np.ndarray,
        seg_proba: np.ndarray | None,
        seg_labels: np.ndarray | None,
        batch_size: int,
        dist_matrix: np.ndarray | None,
    ) -> list[int]:
        """Implement the selection logic for query and return batch_size global indices of candidates."""
        raise NotImplementedError


def mismatch_first_select(
    cand: np.ndarray,
    labeled: np.ndarray,
    mismatch: np.ndarray,
    batch_size: int,
    boundary_resolver: Callable[[np.ndarray, np.ndarray, int], list[int]],
) -> list[int]:
    """Accept whole groups in descending order of mismatch scores, preserving cand order within each group.

    Pass the first group that exceeds the budget to boundary_resolver, using labeled and selected segments as the initial coverage.

    Args:
        cand: Global indices of candidates.
        labeled: Global indices of labeled segments.
        mismatch: Mismatch scores aligned with cand.
        batch_size: Number of samples to select.
        boundary_resolver: ``(group, base, remaining) -> list[int]``.

    Returns:
        batch_size global indices, with entries from accepted whole groups first and selections from the boundary group last.
    """
    order = np.argsort(-mismatch, kind="stable")
    cand_sorted = cand[order]
    mm_sorted = mismatch[order]
    selected: list[int] = []
    base = labeled
    remaining = batch_size
    for val in np.unique(mm_sorted)[::-1]:
        if remaining <= 0:
            break
        group = cand_sorted[mm_sorted == val]
        if len(group) <= remaining:
            selected.extend(group.tolist())
            base = np.concatenate([base, group])
            remaining -= len(group)
        else:
            selected.extend(boundary_resolver(group, base, remaining))
            break
    return selected
