"""FL uses uniformly weighted facility location with labeled segments as the initial coverage."""

from __future__ import annotations

import numpy as np

from .base import QueryStrategy
from .coverage import weighted_facility_location_greedy


class FL(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        weights = np.ones(cand.shape[0], dtype=np.float64)
        return weighted_facility_location_greedy(
            emb, cand_idx=cand, base_idx=labeled, weights=weights, n=batch_size,
            dist_matrix=dist_matrix, random_state=self.random_state,
        )
