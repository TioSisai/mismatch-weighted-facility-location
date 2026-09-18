"""Share precomputed arrays through copy-on-write mmap, keeping the disk cache unchanged."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

META_FILENAME = "meta.json"

# npy filenames corresponding to FrameData fields, without extensions.
ARRAY_FILES = {
    "train_embedding": "train_embedding",
    "train_label": "train_label",
    "train_valid_lengths": "train_valid_lengths",
    "seg_embedding": "train_seg_embedding",
    "seg_labels": "train_seg_labels",
    "val_embedding": "val_embedding",
    "val_label": "val_label",
    "val_valid_lengths": "val_valid_lengths",
    "test_embedding": "test_embedding",
    "test_label": "test_label",
    "test_valid_lengths": "test_valid_lengths",
    "pairwise_distances": "pairwise_distances",
}


# Array fields are compared by object identity.
@dataclass(frozen=True, eq=False)
class FrameData:
    """Arrays and class names for a single dataset.

    Attributes:
        train_embedding: Training frame embeddings, float32 ``[N, F, D]``.
        train_label: Training frame multilabel targets, float32 ``[N, F, C]``.
        train_valid_lengths: Valid frame counts for each training segment, int64 ``[N]``.
        seg_embedding: Training segment representations, float32 ``[N, D]``.
        seg_labels: Training segment multilabel targets, int64 ``[N, C]``.
        val_embedding: Validation frame embeddings, ``[M, F, D]``.
        val_label: Validation frame multilabel targets, ``[M, F, C]``.
        val_valid_lengths: Valid frame counts for each validation segment, ``[M]``.
        test_embedding: Test frame embeddings, ``[K, F, D]``.
        test_label: Test frame multilabel targets, ``[K, F, C]``.
        test_valid_lengths: Valid frame counts for each test segment, ``[K]``.
        pairwise_distances: Euclidean distances between training segments, float64 ``[N, N]``.
        class_names: Class names, ordered to match the last label dimension.
    """

    train_embedding: np.ndarray
    train_label: np.ndarray
    train_valid_lengths: np.ndarray
    seg_embedding: np.ndarray
    seg_labels: np.ndarray
    val_embedding: np.ndarray
    val_label: np.ndarray
    val_valid_lengths: np.ndarray
    test_embedding: np.ndarray
    test_label: np.ndarray
    test_valid_lengths: np.ndarray
    pairwise_distances: np.ndarray
    class_names: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_train(self) -> int:
        return self.train_embedding.shape[0]


def _read_meta(cache_dir: Path) -> dict:
    return json.loads((cache_dir / META_FILENAME).read_text(encoding="utf-8"))


def read_array_meta(cache_dir) -> tuple[int, int, str]:
    """Read ``(n_train, num_classes, backend)`` from meta.json without loading arrays."""
    meta = _read_meta(Path(cache_dir))
    return meta["shapes"]["train_embedding"][0], len(meta["class_names"]), meta["backend"]


def load_frame_data(cache_dir) -> FrameData:
    """Load arrays and class names for a single dataset with ``mmap_mode="c"``."""
    cache_dir = Path(cache_dir)
    meta = _read_meta(cache_dir)
    # Copy-on-write mmap shares the page cache, with writes affecting only process-private pages.
    arrays = {
        field: np.load(cache_dir / f"{stem}.npy", mmap_mode="c")
        for field, stem in ARRAY_FILES.items()
    }
    return FrameData(**arrays, class_names=tuple(meta["class_names"]))
