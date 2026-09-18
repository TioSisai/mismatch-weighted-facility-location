"""UMAP, per-round mismatch fields, and grid smoothing for sample selection plots."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRID_RES = 200
GRID_K = 20
MARGIN = 0.6

UMAP_PARAMS = {"n_neighbors": 15, "min_dist": 0.1, "n_components": 2}


@dataclass(frozen=True, eq=False)
class SmoothingGrid:
    """Two-dimensional background grid and its nearest neighbors in the training pool.

    Attributes:
        extent: Image extent ``(xmin, xmax, ymin, ymax)``.
        idx: int32 nearest-neighbor indices, ``[GRID_RES**2, GRID_K]``, arranged in row-major order with y as the row axis.
        dist: float32 nearest-neighbor distances, with the same shape as idx, ascending within each row.
    """

    extent: tuple[float, float, float, float]
    idx: np.ndarray
    dist: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape ``(GRID_RES, GRID_RES)``, with rows corresponding to y."""
        return (GRID_RES, GRID_RES)


def load_or_compute_array(path, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Load an npy cache, calling compute without arguments and writing atomically if the cache is missing."""
    path = Path(path)
    if path.is_file():
        return np.load(path)
    array = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp_path, array)
    os.replace(tmp_path, path)
    return array


def compute_umap_2d(seg_embedding: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Compute two-dimensional UMAP for the training pool, preferring cuml and otherwise using umap-learn.

    Args:
        seg_embedding: Segment representations, ``[N, D]``.
        random_state: UMAP random seed.

    Returns:
        float32 coordinates, ``[N, 2]``.
    """
    from ..strategies.backends import HAS_CUML, to_numpy

    if HAS_CUML:
        from cuml.manifold import UMAP
    else:
        from umap import UMAP
    reducer = UMAP(**UMAP_PARAMS, random_state=random_state)
    coords = reducer.fit_transform(np.asarray(seg_embedding, dtype=np.float32))
    return to_numpy(coords).astype(np.float32)


def compute_mismatch_fields(
    data,
    run_dir,
    cumulative: Sequence[np.ndarray],
    *,
    hidden_features: int,
    batch_size: int,
    infer_batch_size: int,
    device,
) -> np.ndarray:
    """Compute the mismatch field over the entire pool for each round as done at query time.

    Row i uses the weights and cumulative labels from round i, with selections from round i+1 overlaid when plotting.

    Args:
        data: FrameData array cache.
        run_dir: Seed directory containing per-round checkpoints.
        cumulative: Cumulative labeled indices for each round.
        hidden_features: MLP hidden dimension from the configuration.
        batch_size: Training batch size from the configuration, used only to construct the network.
        infer_batch_size: Inference batch size from the configuration.
        device: Inference device matching the one used to train this experiment.

    Returns:
        Integer mismatch fields, ``[len(cumulative), N]``.
    """
    # A cache hit requires no loading of torch or skorch.
    import torch

    from ..data.frames import segment_max_pool
    from ..models.classifier import build_inference_net
    from ..strategies.backends import knn_multilabel_fit_predict
    from ..strategies.mismatch import segment_mismatch

    run_dir = Path(run_dir)
    embedding = np.asarray(data.seg_embedding, dtype=np.float32)
    fields = []
    for iteration, labeled in enumerate(cumulative):
        (ckpt_path,) = run_dir.glob(f"iter={iteration}-labeled=*.pth")
        net = build_inference_net(
            num_classes=data.num_classes,
            hidden_features=hidden_features,
            state_dict=torch.load(ckpt_path, map_location="cpu"),
            batch_size=batch_size,
            infer_batch_size=infer_batch_size,
            device=device,
        )
        seg_proba = segment_max_pool(net.predict_frame_proba(data.train_embedding), data.train_valid_lengths)
        # Pass anchors in ascending order as during training to keep 1-NN tie handling consistent.
        anchors = np.sort(labeled)
        reference = knn_multilabel_fit_predict(embedding[anchors], data.seg_labels[anchors], embedding)
        fields.append(segment_mismatch(seg_proba, reference))
    return np.stack(fields)


def build_grid(umap_2d: np.ndarray) -> SmoothingGrid:
    """Build a GRID_RES×GRID_RES grid extended by MARGIN on each side and its nearest neighbors, preferring cuml and otherwise using sklearn."""
    from ..strategies.backends import HAS_CUML, to_numpy

    if HAS_CUML:
        from cuml.neighbors import NearestNeighbors
    else:
        from sklearn.neighbors import NearestNeighbors

    xmin, ymin = umap_2d.min(0) - MARGIN
    xmax, ymax = umap_2d.max(0) + MARGIN
    xs = np.linspace(xmin, xmax, GRID_RES, dtype=np.float32)
    ys = np.linspace(ymin, ymax, GRID_RES, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # [GRID_RES(y), GRID_RES(x)]
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=GRID_K)
    knn.fit(umap_2d.astype(np.float32))
    dist, idx = knn.kneighbors(grid_points)
    return SmoothingGrid(
        extent=(float(xmin), float(xmax), float(ymin), float(ymax)),
        idx=to_numpy(idx).astype(np.int32),
        dist=to_numpy(dist).astype(np.float32),
    )


def grid_scalar_field(grid: SmoothingGrid, values: np.ndarray) -> np.ndarray:
    """Smooth point values onto the grid using inverse nearest-neighbor distances as weights.

    Args:
        grid: Return value of build_grid.
        values: Scalar value at each training pool point, ``[N]``.

    Returns:
        float32 field, ``[GRID_RES, GRID_RES]``, with rows corresponding to y.
    """
    weights = 1.0 / (grid.dist + 1e-6)  # [P, K]
    neighbor_values = values[grid.idx].astype(np.float32)
    field = (weights * neighbor_values).sum(axis=1) / weights.sum(axis=1)
    return field.reshape(grid.shape)


def density_alpha(grid: SmoothingGrid, d_in_pct: float = 48, d_out_pct: float = 84) -> np.ndarray:
    """Generate a transparency mask from the grid's nearest-neighbor distances.

    Args:
        grid: Return value of build_grid.
        d_in_pct: Distance percentile threshold for full opacity.
        d_out_pct: Distance percentile threshold for full transparency, with a linear transition between the two thresholds.

    Returns:
        ``[GRID_RES, GRID_RES]`` transparency values in ``[0, 1]``.
    """
    nearest = grid.dist[:, 0]
    d_in = np.percentile(nearest, d_in_pct)
    d_out = np.percentile(nearest, d_out_pct)
    alpha = np.clip((d_out - nearest) / max(d_out - d_in, 1e-6), 0.0, 1.0)
    return alpha.reshape(grid.shape)
