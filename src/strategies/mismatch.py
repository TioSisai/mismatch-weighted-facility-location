"""Mismatch scores and their smoothed weights."""

from __future__ import annotations

import numpy as np

from .backends import knn_multilabel_fit_predict


def segment_mismatch(seg_proba_cand: np.ndarray, knn_pred_cand: np.ndarray) -> np.ndarray:
    """Compute the Hamming distance between predictions binarized at a threshold of 0.5 and 1-NN reference labels.

    Args:
        seg_proba_cand: Segment probabilities, ``[M, C]``.
        knn_pred_cand: Reference multilabel targets, ``[M, C]``.

    Returns:
        Integer mismatch scores, ``[M]``.
    """
    binary = (seg_proba_cand >= 0.5).astype(int)
    return np.abs(binary - knn_pred_cand).sum(axis=1)


def candidate_mismatch(
    emb: np.ndarray,
    cand_idx: np.ndarray,
    labeled_idx: np.ndarray,
    seg_proba: np.ndarray,
    seg_labels: np.ndarray,
) -> np.ndarray:
    """Compute mismatch scores between candidate predictions and the labels of the nearest labeled segments.

    Args:
        emb: Segment representations for the full pool, ``[N, D]``.
        cand_idx: Global indices of candidates.
        labeled_idx: Nonempty global indices of labeled segments.
        seg_proba: Segment probabilities for the full pool, ``[N, C]``, using only candidate rows.
        seg_labels: Segment multilabel targets for the full pool, ``[N, C]``, using only labeled rows.

    Returns:
        Mismatch scores aligned with cand_idx.
    """
    emb = np.asarray(emb, dtype=np.float32)
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    cand_idx = np.asarray(cand_idx, dtype=int)
    seg_labels = np.asarray(seg_labels)
    knn_pred = knn_multilabel_fit_predict(emb[labeled_idx], seg_labels[labeled_idx], emb[cand_idx])
    return segment_mismatch(np.asarray(seg_proba)[cand_idx], knn_pred)


def mismatch_to_weights(mismatch: np.ndarray, num_classes: int) -> np.ndarray:
    """Smooth scores into float64 weights ``w = (m + 1) / (C + 1) ∈ (0, 1]``."""
    return (np.asarray(mismatch, dtype=np.float64) + 1.0) / (num_classes + 1.0)
