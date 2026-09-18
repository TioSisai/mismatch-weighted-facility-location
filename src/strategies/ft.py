"""FT uses farthest-point traversal, starting from a random point for a cold start."""

from __future__ import annotations

from .base import QueryStrategy
from .traversal import farthest_first, random_seed_farthest_first


class FT(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        if labeled.size == 0:
            return random_seed_farthest_first(emb, cand, batch_size, self.random_state, dist_matrix=dist_matrix)
        return farthest_first(emb, cand_idx=cand, base_idx=labeled, n=batch_size, dist_matrix=dist_matrix)
