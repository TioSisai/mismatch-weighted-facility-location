"""Multilabel mAP over valid frames and cumulative AULC."""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torchmetrics.functional.classification import multilabel_average_precision


def _flatten_valid(frame_array: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Extract valid frames, flattening ``[N, F, C]`` to ``[M, C]``."""
    num_frames = frame_array.shape[1]
    mask = np.arange(num_frames)[None, :] < np.asarray(valid_lengths)[:, None]  # [N, F]
    return np.asarray(frame_array)[mask]


def frame_wise_map(
    frame_proba: np.ndarray,
    frame_labels: np.ndarray,
    valid_lengths: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute per-class AP and macro-averaged mAP over valid frames.

    Args:
        frame_proba: Frame-level probabilities, ``[N, F, C]``.
        frame_labels: Frame-level 0/1 multilabel targets, ``[N, F, C]``.
        valid_lengths: Valid frame counts, ``[N]``.

    Returns:
        ``(per_class, macro)``, with per-class AP as float64 ``[C]``.
        Classes without positive examples are assigned NaN and excluded from the macro average, which is NaN when all classes are absent.
    """
    proba = _flatten_valid(frame_proba, valid_lengths).astype(np.float32)
    labels = _flatten_valid(frame_labels, valid_lengths).astype(np.int64)
    num_classes = labels.shape[1]
    with warnings.catch_warnings():
        # Classes without positive examples are subsequently set to NaN.
        warnings.filterwarnings("ignore", message="No positive samples found in target")
        per_class = multilabel_average_precision(
            torch.from_numpy(proba),
            torch.from_numpy(labels),
            num_labels=num_classes,
            average=None,
        ).numpy().astype(np.float64)
    absent = labels.sum(axis=0) == 0
    per_class[absent] = np.nan
    macro = float(np.nanmean(per_class)) if not np.all(absent) else float("nan")
    return per_class, macro


def cumulative_aulc(xs, ys) -> float:
    """Compute normalized AULC by dividing the trapezoidal integral by the x-axis span.

    Args:
        xs: Cumulative labeled counts increasing with each round.
        ys: mAP for the corresponding rounds.

    Returns:
        Normalized area, or NaN if there are fewer than two points or the span is nonpositive.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if xs.shape[0] < 2:
        return float("nan")
    span = xs[-1] - xs[0]
    if span <= 0:
        return float("nan")
    return float(np.trapezoid(ys, xs) / span)
