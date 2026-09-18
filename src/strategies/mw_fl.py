"""MW-FL maximizes the facility location objective with smoothed mismatch weights, with the same cold start as FL."""

from __future__ import annotations

import numpy as np

from .base import QueryStrategy
from .coverage import weighted_facility_location_greedy
from .mismatch import candidate_mismatch, mismatch_to_weights


class MWFL(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        if labeled.size == 0:
            weights = np.ones(cand.shape[0], dtype=np.float64)
        else:
            num_classes = np.asarray(seg_labels).shape[1]
            mismatch = candidate_mismatch(emb, cand, labeled, seg_proba, seg_labels)
            weights = mismatch_to_weights(mismatch, num_classes)
        return weighted_facility_location_greedy(
            emb, cand_idx=cand, base_idx=labeled, weights=weights, n=batch_size,
            dist_matrix=dist_matrix, random_state=self.random_state,
        )
