"""MF-FL first groups by mismatch, then runs FL on the boundary group, with the same cold start as FL."""

from __future__ import annotations

import numpy as np

from .base import QueryStrategy, mismatch_first_select
from .coverage import median_sq_bandwidth, weighted_facility_location_greedy
from .mismatch import candidate_mismatch


class MFFL(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        if labeled.size == 0:
            weights = np.ones(cand.shape[0], dtype=np.float64)
            return weighted_facility_location_greedy(
                emb, cand_idx=cand, base_idx=labeled, weights=weights, n=batch_size,
                dist_matrix=dist_matrix, random_state=self.random_state,
            )
        mismatch = candidate_mismatch(emb, cand, labeled, seg_proba, seg_labels)
        # As in FL and MW-FL, the bandwidth is estimated over all candidates.
        sigma2 = median_sq_bandwidth(emb[cand], random_state=self.random_state)

        def resolve_boundary(group, base, remaining):
            weights = np.ones(len(group), dtype=np.float64)
            return weighted_facility_location_greedy(
                emb, cand_idx=group, base_idx=base, weights=weights, n=remaining,
                sigma2=sigma2, dist_matrix=dist_matrix, random_state=self.random_state,
            )

        return mismatch_first_select(cand, labeled, mismatch, batch_size, resolve_boundary)
