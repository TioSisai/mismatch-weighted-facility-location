"""Segment-level max pooling over valid frames only."""

from __future__ import annotations

import numpy as np


def segment_max_pool(frame_array: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Pool frame-level probabilities or labels into segment-level arrays.

    Args:
        frame_array: Frame-level array, ``[N, F, C]``.
        valid_lengths: Valid frame counts, ``[N]``, with values in ``1..F``.

    Returns:
        Pooled ``[N, C]`` array.
    """
    arr = np.asarray(frame_array)
    num_frames = arr.shape[1]
    mask = np.arange(num_frames)[None, :] < np.asarray(valid_lengths)[:, None]  # [N, F]
    masked = np.where(mask[:, :, None], arr, -np.inf)
    return masked.max(axis=1)
