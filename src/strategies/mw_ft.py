"""MW-FT weights the farthest distances with smoothed mismatch weights, with the same cold start as FT."""

from __future__ import annotations

import numpy as np

from .base import QueryStrategy
from .mismatch import candidate_mismatch, mismatch_to_weights
from .traversal import farthest_first, random_seed_farthest_first


class MWFT(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        if labeled.size == 0:
            return random_seed_farthest_first(emb, cand, batch_size, self.random_state, dist_matrix=dist_matrix)
        num_classes = np.asarray(seg_labels).shape[1]
        mismatch = candidate_mismatch(emb, cand, labeled, seg_proba, seg_labels)
        weights = mismatch_to_weights(mismatch, num_classes)
        return farthest_first(
            emb, cand_idx=cand, base_idx=labeled, n=batch_size, weights=weights, dist_matrix=dist_matrix,
        )
