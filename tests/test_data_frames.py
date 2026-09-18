"""Test segment-level max pooling over valid frames."""

from __future__ import annotations

import numpy as np

from src.data.frames import segment_max_pool


def test_segment_max_pool_only_valid_frames():
    arr = np.array([
        [[0.1, 0.9], [0.8, 0.2], [9.0, 9.0]],  # Two valid frames, pooled result [0.8, 0.9]
        [[0.3, 0.4], [9.0, 9.0], [9.0, 9.0]],  # One valid frame, pooled result [0.3, 0.4]
    ], dtype=np.float32)
    out = segment_max_pool(arr, np.array([2, 1]))
    assert out.dtype == np.float32
    assert np.allclose(out, [[0.8, 0.9], [0.3, 0.4]])


def test_segment_max_pool_keeps_full_segments():
    arr = np.array([[[0.2, 0.1], [0.5, 0.7], [0.4, 0.3]]], dtype=np.float32)
    assert np.allclose(segment_max_pool(arr, np.array([3])), [[0.5, 0.7]])
