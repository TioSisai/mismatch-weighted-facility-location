"""MF-FT (MFFT) first groups by mismatch, then runs FT on the boundary group, with the same cold start as FT."""

from __future__ import annotations

from .base import QueryStrategy, mismatch_first_select
from .mismatch import candidate_mismatch
from .traversal import farthest_first, random_seed_farthest_first


class MFFT(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        if labeled.size == 0:
            return random_seed_farthest_first(emb, cand, batch_size, self.random_state, dist_matrix=dist_matrix)
        mismatch = candidate_mismatch(emb, cand, labeled, seg_proba, seg_labels)

        def resolve_boundary(group, base, remaining):
            return farthest_first(emb, cand_idx=group, base_idx=base, n=remaining, dist_matrix=dist_matrix)

        return mismatch_first_select(cand, labeled, mismatch, batch_size, resolve_boundary)
