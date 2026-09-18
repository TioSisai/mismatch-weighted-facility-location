"""Random sampling baseline based on skactiveml RandomSampling."""

from __future__ import annotations

import numpy as np
from skactiveml.pool import RandomSampling as SkactivemlRandomSampling

from .base import QueryStrategy


class RandomSampling(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        # The random stream for an integer seed depends on the number of missing labels in y.
        y = np.full(emb.shape[0], np.nan, dtype=float)
        y[labeled] = 0.0
        sampler = SkactivemlRandomSampling(missing_label=np.nan, random_state=self.random_state)
        selected = sampler.query(emb, y, candidates=cand, batch_size=batch_size)
        return [int(i) for i in np.asarray(selected)]
