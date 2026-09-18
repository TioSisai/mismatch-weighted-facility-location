"""Construct synthetic data matching the data cache format and shared pytest fixtures."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Limit threads before importing numpy/torch to prevent CPU fallback from crashing on many-core nodes.
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import rootutils  # noqa: E402

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True)

NUM_FRAMES = 31
EMBEDDING_DIM = 2048


def _num_valid_frames(seconds: float) -> int:
    """Compute the number of valid frames from segment duration, capped at NUM_FRAMES."""
    num_samples = int(round(seconds * 32_000))
    return min((num_samples // 320 + 1) // 32, NUM_FRAMES)


def _mean_pool(frame_embeddings: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    """Average valid frames in float64, then convert back to float32."""
    mask = (np.arange(NUM_FRAMES)[None, :] < valid_lengths[:, None]).astype(np.float64)
    summed = (frame_embeddings.astype(np.float64) * mask[:, :, None]).sum(axis=1)
    return (summed / valid_lengths[:, None]).astype(np.float32)


def build_synthetic_frame_cache(
    cache_dir,
    *,
    n_train: int = 40,
    n_val: int = 12,
    n_test: int = 12,
    num_classes: int = 3,
    num_clusters: int = 4,
    short_ratio: float = 0.2,
    seed: int = 0,
) -> Path:
    """Write a synthetic cache containing clusters and short trailing segments, and return cache_dir."""
    # Generate the distance cache with the current backend for comparison with the recomputation path.
    from src.strategies.backends import backend_signature, pairwise_euclidean_distances

    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((num_clusters, EMBEDDING_DIM)).astype(np.float32) * 3.0
    cluster_classes = rng.random((num_clusters, num_classes)) < 0.6

    def make_split(num_segments: int):
        cluster = rng.integers(num_clusters, size=num_segments)
        seg_end = np.where(
            rng.random(num_segments) < short_ratio,
            rng.uniform(2.0, 9.0, size=num_segments),
            10.0,
        ).astype(np.float32)
        valid_lengths = np.array([_num_valid_frames(float(e)) for e in seg_end], dtype=np.int64)
        embedding = (
            centers[cluster][:, None, :]
            + rng.standard_normal((num_segments, NUM_FRAMES, EMBEDDING_DIM)).astype(np.float32)
        )
        label = np.zeros((num_segments, NUM_FRAMES, num_classes), dtype=np.float32)
        for i in range(num_segments):
            active = rng.random((valid_lengths[i], num_classes)) < 0.5
            label[i, :valid_lengths[i]] = active & cluster_classes[cluster[i]]
        for c in range(num_classes):  # Each class contains at least one positive frame so that mAP is defined.
            if label[..., c].sum() == 0:
                label[c % num_segments, 0, c] = 1.0
        return embedding, label, valid_lengths

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, list[int]] = {}

    def save(name: str, array: np.ndarray) -> None:
        np.save(cache_dir / f"{name}.npy", array)
        shapes[name] = list(array.shape)

    for split, num_segments in (("train", n_train), ("val", n_val), ("test", n_test)):
        embedding, label, valid_lengths = make_split(num_segments)
        save(f"{split}_embedding", embedding)
        save(f"{split}_label", label)
        save(f"{split}_valid_lengths", valid_lengths)
        if split == "train":
            seg_embedding = _mean_pool(embedding, valid_lengths)
            mask = np.arange(NUM_FRAMES)[None, :] < valid_lengths[:, None]
            save("train_seg_embedding", seg_embedding)
            save("train_seg_labels", np.where(mask[:, :, None], label, -np.inf).max(axis=1).astype(np.int64))

    save("pairwise_distances", pairwise_euclidean_distances(seg_embedding, seg_embedding))
    (cache_dir / "meta.json").write_text(
        json.dumps({
            "class_names": [f"class_{c}" for c in range(num_classes)],
            "shapes": shapes,
            "backend": backend_signature(),
        }, indent=2),
        encoding="utf-8",
    )
    return cache_dir


@pytest.fixture
def synthetic_cache(tmp_path) -> Path:
    """Build the default synthetic cache at tmp_path/cache/Toy."""
    return build_synthetic_frame_cache(tmp_path / "cache" / "Toy")
