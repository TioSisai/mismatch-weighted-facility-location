"""Test distances, coverage, traversal, and selection by mismatch groups on the default and CPU backends."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategies import backends
from src.strategies.backends import (
    backend_signature,
    knn_multilabel_fit_predict,
    pairwise_euclidean_distances,
    pairwise_sq_distances,
    to_numpy,
)
from src.strategies.base import mismatch_first_select
from src.strategies.coverage import median_sq_bandwidth, weighted_facility_location_greedy
from src.strategies.mismatch import candidate_mismatch, mismatch_to_weights, segment_mismatch
from src.strategies.traversal import farthest_first, random_seed_farthest_first

EMPTY = np.array([], dtype=int)


@pytest.fixture(autouse=True, params=["default", "cpu"])
def backend(request, monkeypatch):
    """Use the default backend and forced CPU fallback in separate cases."""
    if request.param == "cpu":
        monkeypatch.setattr(backends, "HAS_CUML", False)
        monkeypatch.setattr(backends, "HAS_CUPY", False)
    return request.param




def test_backend_signature_reflects_flags(backend):
    signature = backend_signature()
    if backend == "cpu":
        assert signature == "sklearn+numpy"
    else:
        assert signature.split("+") == [
            "cuml" if backends.HAS_CUML else "sklearn",
            "cupy" if backends.HAS_CUPY else "numpy",
        ]


def test_to_numpy_passthrough_and_cupy():
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert to_numpy(array) is array
    if backends.cp is not None:
        restored = to_numpy(backends.cp.asarray(array))
        assert isinstance(restored, np.ndarray) and np.array_equal(restored, array)


def test_pairwise_sq_distances_values():
    A = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    B = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    D = pairwise_sq_distances(A, B)
    assert D.shape == (2, 2)
    assert np.allclose(D, [[0.0, 4.0], [1.0, 5.0]], atol=1e-4)


def test_pairwise_euclidean_distances_values():
    A = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    B = np.array([[0.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    D = pairwise_euclidean_distances(A, B)
    assert D.shape == (2, 2) and D.dtype == np.float64
    assert np.allclose(D, [[0.0, 2.0], [1.0, np.sqrt(5.0)]], atol=1e-4)


def test_knn_multilabel_shape_and_values():
    emb_lab = np.array([[0.0], [0.0], [10.0], [10.0]], dtype=np.float32)
    y_lab = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int64)
    emb_cand = np.array([[0.1], [9.9]], dtype=np.float32)
    pred = knn_multilabel_fit_predict(emb_lab, y_lab, emb_cand)
    assert pred.shape == (2, 2) and pred.dtype == np.int64
    assert pred.tolist() == [[1, 0], [0, 1]]




def test_farthest_first_picks_farthest_from_base():
    # The origin is the coverage point. Candidates 1/2/3 have increasing distances, so 3 is selected first.
    emb = np.array([[0.0], [1.0], [2.0], [5.0]], dtype=np.float32)
    assert farthest_first(emb, np.array([1, 2, 3]), np.array([0]), 1) == [3]


def test_farthest_first_weighted_flips_selection():
    # Without weights, distant point 2 is selected. Increasing the nearby point's weight selects 1.
    emb = np.array([[0.0], [1.0], [4.0]], dtype=np.float32)
    base, cand = np.array([0]), np.array([1, 2])
    assert farthest_first(emb, cand, base, 1) == [2]
    assert farthest_first(emb, cand, base, 1, weights=np.array([10.0, 1.0])) == [1]


def test_farthest_first_dist_matrix_matches_recompute():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 4)).astype(np.float32)
    base, cand = np.array([0, 1]), np.arange(2, 20)
    D = pairwise_euclidean_distances(emb, emb)
    for weights in (None, rng.random(len(cand))):
        assert farthest_first(emb, cand, base, 6, weights=weights) == farthest_first(
            emb, cand, base, 6, weights=weights, dist_matrix=D
        )


def test_farthest_first_raises_when_n_too_large():
    emb = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        farthest_first(emb, np.array([1, 2]), np.array([0]), 3)


def test_random_seed_farthest_first_contract_and_reproducible():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 4)).astype(np.float32)
    cand = np.arange(20)
    a = random_seed_farthest_first(emb, cand, 5, 3)
    assert a == random_seed_farthest_first(emb, cand, 5, 3)
    assert len(set(a)) == 5 and set(a) <= set(cand.tolist())


def test_random_seed_farthest_first_dist_matrix_matches_recompute():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((30, 3)).astype(np.float32)
    cand = np.arange(30)
    D = pairwise_euclidean_distances(emb, emb)
    assert random_seed_farthest_first(emb, cand, 8, 5) == random_seed_farthest_first(
        emb, cand, 8, 5, dist_matrix=D
    )




def test_median_sq_bandwidth_known_values():
    # The point set {0, 1, 3} has squared distances {1, 4, 9}, with a median of 4.
    emb = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    assert np.isclose(median_sq_bandwidth(emb), 4.0)


def test_median_sq_bandwidth_degenerate_positive():
    # Six pairs have zero distance and four pairs have distance 25. The median is zero, requiring a fallback to a positive value.
    emb = np.array([[0.0], [0.0], [0.0], [0.0], [5.0]], dtype=np.float32)
    assert median_sq_bandwidth(emb) > 0.0


def test_facility_location_declusters_redundant_high_weight():
    # With uniform weights, two selections cover the four-point cluster and the isolated point, one each.
    emb = np.array([[0.0], [0.03], [0.06], [0.09], [10.0]], dtype=np.float32)
    sel = weighted_facility_location_greedy(emb, np.arange(5), EMPTY, np.ones(5), n=2)
    assert len(set(sel)) == 2 and 4 in sel
    assert len(set(sel) & {0, 1, 2, 3}) == 1


def test_facility_location_base_covers_cluster():
    # Initial coverage is already in the cluster, so additional coverage should select the isolated point.
    emb = np.array([[0.0], [0.03], [0.06], [10.0]], dtype=np.float32)
    sel = weighted_facility_location_greedy(emb, np.array([1, 2, 3]), np.array([0]), np.ones(3), n=1)
    assert sel == [3]


def test_facility_location_weight_steers_selection():
    emb = np.array([[0.0], [10.0]], dtype=np.float32)
    sel = weighted_facility_location_greedy(emb, np.array([0, 1]), EMPTY, np.array([10.0, 1.0]), n=1)
    assert sel == [0]


def test_facility_location_contract_unique_subset():
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((20, 5)).astype(np.float32)
    cand = np.arange(5, 20)
    sel = weighted_facility_location_greedy(emb, cand, np.array([0, 1, 2]), np.ones(15), n=4)
    assert len(set(sel)) == 4 and set(sel) <= set(cand.tolist())


def test_facility_location_raises_when_n_too_large():
    emb = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        weighted_facility_location_greedy(emb, np.array([1, 2]), EMPTY, np.ones(2), n=3)


def test_facility_location_dist_matrix_matches_recompute():
    # A low-dimensional layout and discrete weights produce near-tied scores, checking bitwise consistency between cached and recomputed results.
    rng = np.random.default_rng(2)
    emb = rng.standard_normal((400, 2)).astype(np.float32)
    base, cand = np.arange(5), np.arange(5, 400)
    weights = rng.integers(1, 5, size=len(cand)).astype(np.float32)
    D = pairwise_euclidean_distances(emb, emb)
    assert weighted_facility_location_greedy(emb, cand, base, weights, n=40) == (
        weighted_facility_location_greedy(emb, cand, base, weights, n=40, dist_matrix=D)
    )


def test_facility_location_random_state_makes_bandwidth_reproducible():
    # More than 2000 candidates trigger sampling for bandwidth estimation, which should be reproducible with a fixed seed.
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((2200, 8)).astype(np.float32)
    cand, base = np.arange(100, 2200), np.arange(5)
    weights = (rng.random(len(cand)) + 0.1).astype(np.float32)
    a = weighted_facility_location_greedy(emb, cand, base, weights, n=30, random_state=0)
    b = weighted_facility_location_greedy(emb, cand, base, weights, n=30, random_state=0)
    assert a == b


def test_facility_location_reads_backend_flag_at_call_time(backend, monkeypatch):
    # Clear the cupy handle to check that the backend is determined at call time.
    if backend == "cpu":
        monkeypatch.setattr(backends, "cp", None)
    emb = np.array([[0.0], [1.0], [5.0]], dtype=np.float32)
    assert len(weighted_facility_location_greedy(emb, np.arange(3), EMPTY, np.ones(3), n=2)) == 2




def test_segment_mismatch_hamming():
    proba = np.array([[0.9, 0.1, 0.6], [0.2, 0.8, 0.4]], dtype=np.float32)  # Binarized result [[1,0,1],[0,1,0]]
    ref = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.int64)
    assert segment_mismatch(proba, ref).tolist() == [1, 1]


def test_candidate_mismatch_uses_nn_reference():
    # Candidate 2's nearest labeled point is 0, giving a mismatch of 2 between reference [1,0] and prediction [0,1].
    emb = np.array([[0.0], [10.0], [0.1]], dtype=np.float32)
    seg_labels = np.array([[1, 0], [0, 1], [0, 0]], dtype=np.int64)
    seg_proba = np.array([[0.0, 0.0], [0.0, 0.0], [0.2, 0.9]], dtype=np.float32)
    assert candidate_mismatch(emb, np.array([2]), np.array([0, 1]), seg_proba, seg_labels).tolist() == [2]


def test_mismatch_to_weights_smoothing():
    w = mismatch_to_weights(np.array([0, 2]), num_classes=3)
    assert w.dtype == np.float64 and np.allclose(w, [1 / 4, 3 / 4])




def test_mismatch_first_whole_group_then_boundary():
    # With a budget of 3, accept the entire group {0,1}, then select one from the boundary group {2,3}.
    cand = np.array([0, 1, 2, 3, 4])
    labeled = np.array([9])
    calls = {}

    def resolver(group, base, remaining):
        calls.update(base=base.tolist(), group=group.tolist(), remaining=remaining)
        return [int(group[0])]

    assert mismatch_first_select(cand, labeled, np.array([2, 2, 1, 1, 0]), 3, resolver) == [0, 1, 2]
    assert calls == {"base": [9, 0, 1], "group": [2, 3], "remaining": 1}


def test_mismatch_first_exact_budget_skips_resolver():
    # No boundary group needs to be processed when an entire group exactly exhausts the budget.
    def resolver(group, base, remaining):
        raise AssertionError("the boundary-group rule must not be called")

    cand = np.array([4, 3, 2, 1, 0])
    assert mismatch_first_select(cand, EMPTY, np.array([2, 2, 1, 1, 0]), 4, resolver) == [4, 3, 2, 1]
