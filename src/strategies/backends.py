"""Euclidean distance and 1-NN backends, preferring cuml/cupy with a fallback to sklearn/numpy."""

from __future__ import annotations

import numpy as np

# Read backend flags at call time to facilitate switching to the CPU fallback in tests.
try:
    import cupy as cp

    # cupy can be imported without a GPU, so visible devices must be checked.
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover
    cp = None
    HAS_CUPY = False

try:
    import cuml  # noqa: F401

    HAS_CUML = HAS_CUPY
except Exception:  # pragma: no cover
    HAS_CUML = False


def backend_signature() -> str:
    """Return the current distance and array backend signature to distinguish results from different floating-point rounding paths."""
    return f"{'cuml' if HAS_CUML else 'sklearn'}+{'cupy' if HAS_CUPY else 'numpy'}"


def to_numpy(array) -> np.ndarray:
    """Convert arrays such as cupy and cudf to numpy, returning numpy inputs directly."""
    if isinstance(array, np.ndarray):
        return array
    for attr in ("get", "to_numpy"):
        method = getattr(array, attr, None)
        if callable(method):
            return np.asarray(method())
    values = getattr(array, "values", None)
    if values is not None:
        return to_numpy(values)
    return np.asarray(array)


def pairwise_euclidean_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances in float32 and return a float64 array.

    Input shapes affect the floating-point reduction order, so distance calls in traversal and coverage must retain their original granularity.

    Args:
        A: Point set, ``[n_a, D]``.
        B: Point set, ``[n_b, D]``.

    Returns:
        Euclidean distance matrix, ``[n_a, n_b]``.
    """
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    if HAS_CUML:
        from cuml.metrics import pairwise_distances as cu_pairwise_distances

        dist = to_numpy(cu_pairwise_distances(A, B, metric="euclidean"))
    else:
        from sklearn.metrics import pairwise_distances

        dist = pairwise_distances(A, B, metric="euclidean")
    return dist.astype(np.float64)


def pairwise_sq_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return squared Euclidean distances between A and B, float64 ``[n_a, n_b]``."""
    return np.square(pairwise_euclidean_distances(A, B))


def knn_multilabel_fit_predict(
    emb_labeled: np.ndarray,
    y_labeled: np.ndarray,
    emb_cand: np.ndarray,
) -> np.ndarray:
    """Use the multilabel targets of the nearest labeled segments as reference labels for candidates.

    Use KNeighborsClassifier to preserve tie and rounding behavior.

    Args:
        emb_labeled: Labeled segment representations, ``[n_lab, D]``, with n_lab at least 1.
        y_labeled: Multilabel targets for labeled segments, ``[n_lab, C]``.
        emb_cand: Candidate segment representations, ``[n_cand, D]``.

    Returns:
        int64 reference multilabel targets, ``[n_cand, C]``.
    """
    emb_labeled = np.asarray(emb_labeled, dtype=np.float32)
    y_labeled = np.asarray(y_labeled, dtype=np.int64)
    emb_cand = np.asarray(emb_cand, dtype=np.float32)
    if HAS_CUML:
        from cuml.neighbors import KNeighborsClassifier
    else:
        from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(emb_labeled, y_labeled)
    pred = to_numpy(knn.predict(emb_cand))
    return pred.astype(np.int64).reshape(emb_cand.shape[0], y_labeled.shape[1])
