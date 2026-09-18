"""Test the query interfaces and selection behavior of seven strategies, and consistency between cached and recomputed results."""

from __future__ import annotations

import numpy as np
import pytest

from src.strategies import STRATEGY_NAMES, QueryStrategy, backends, build_query_strategy
from src.strategies import coverage, mf_fl
from src.strategies.backends import pairwise_euclidean_distances
from src.strategies.fl import FL
from src.strategies.ft import FT
from src.strategies.mf_fl import MFFL
from src.strategies.mf_ft import MFFT
from src.strategies.mw_fl import MWFL
from src.strategies.mw_ft import MWFT
from src.strategies.random_sampling import RandomSampling
from src.strategies.traversal import farthest_first, random_seed_farthest_first

EMPTY = np.array([], dtype=int)


@pytest.fixture(autouse=True, params=["default", "cpu"])
def backend(request, monkeypatch):
    """Use the default backend and forced CPU fallback in separate cases."""
    if request.param == "cpu":
        monkeypatch.setattr(backends, "HAS_CUML", False)
        monkeypatch.setattr(backends, "HAS_CUPY", False)
    return request.param


def _random_problem(n=60, dim=4, num_classes=4, num_labeled=5, seed=0):
    """Construct a random sample-selection problem and return ``(emb, seg_proba, seg_labels, labeled, cand)``."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    seg_proba = rng.random((n, num_classes)).astype(np.float32)
    seg_labels = (rng.random((n, num_classes)) > 0.5).astype(np.int64)
    return emb, seg_proba, seg_labels, np.arange(num_labeled), np.arange(num_labeled, n)




class _TakeFirst(QueryStrategy):
    def _select(self, emb, cand, labeled, seg_proba, seg_labels, batch_size, dist_matrix):
        return [int(i) for i in cand[:batch_size]]


def test_base_query_contract():
    emb = np.zeros((10, 3), dtype=np.float32)
    sel = _TakeFirst().query(emb, np.arange(2, 10), EMPTY, batch_size=3)
    assert sel.tolist() == [2, 3, 4] and sel.dtype == int


def test_base_query_raises_when_candidates_too_few():
    emb = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        _TakeFirst().query(emb, np.array([0, 1]), EMPTY, batch_size=3)


def test_registry_builds_all_names():
    expected = {
        "random": RandomSampling, "ft": FT, "fl": FL, "mf-ft": MFFT,
        "mf-fl": MFFL, "mw-ft": MWFT, "mw-fl": MWFL,
    }
    assert STRATEGY_NAMES == tuple(expected)
    for name, cls in expected.items():
        strategy = build_query_strategy(name, random_state=3)
        assert type(strategy) is cls and strategy.random_state == 3


def test_registry_unknown_raises():
    with pytest.raises(ValueError):
        build_query_strategy("mwfl", random_state=0)


@pytest.mark.parametrize("name", STRATEGY_NAMES)
@pytest.mark.parametrize("warm", [False, True])
def test_strategy_contract_and_reproducible(name, warm):
    emb, seg_proba, seg_labels, labeled, cand = _random_problem(n=40)
    if not warm:
        labeled, cand, seg_proba = EMPTY, np.arange(40), None
    kwargs = dict(seg_proba=seg_proba, seg_labels=seg_labels, batch_size=6)
    a = build_query_strategy(name, random_state=4).query(emb, cand, labeled, **kwargs)
    b = build_query_strategy(name, random_state=4).query(emb, cand, labeled, **kwargs)
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == 6 and set(a.tolist()) <= set(cand.tolist())


@pytest.mark.parametrize("name", STRATEGY_NAMES)
@pytest.mark.parametrize("warm", [False, True])
def test_strategy_dist_matrix_matches_recompute(name, warm):
    # A low-dimensional layout produces near-tied scores, exposing distance or kernel differences between cached and recomputed results.
    emb, seg_proba, seg_labels, labeled, cand = _random_problem(n=300, dim=2, num_labeled=10, seed=1)
    if not warm:
        labeled, cand, seg_proba = EMPTY, np.arange(300), None
    D = pairwise_euclidean_distances(emb, emb)
    kwargs = dict(seg_proba=seg_proba, seg_labels=seg_labels, batch_size=30)
    a = build_query_strategy(name, random_state=5).query(emb, cand, labeled, **kwargs)
    b = build_query_strategy(name, random_state=5).query(emb, cand, labeled, dist_matrix=D, **kwargs)
    assert np.array_equal(a, b)




def test_random_reproducible_and_seed_dependent():
    emb, _, _, labeled, cand = _random_problem(n=30)
    a = RandomSampling(random_state=3).query(emb, cand, labeled, batch_size=7)
    assert np.array_equal(a, RandomSampling(random_state=3).query(emb, cand, labeled, batch_size=7))
    assert not np.array_equal(a, RandomSampling(random_state=4).query(emb, cand, labeled, batch_size=7))




def test_ft_warm_matches_farthest_first_from_labeled():
    emb, _, _, labeled, cand = _random_problem(n=20, num_labeled=3)
    sel = FT().query(emb, cand, labeled, batch_size=5)
    assert sel.tolist() == farthest_first(emb, cand, labeled, 5)


def test_ft_family_cold_start_matches_random_seed_ft():
    emb, *_ = _random_problem(n=20)
    cand = np.arange(20)
    expected = random_seed_farthest_first(emb, cand, 5, 7)
    for cls in (FT, MFFT, MWFT):
        assert cls(random_state=7).query(emb, cand, EMPTY, batch_size=5).tolist() == expected


def test_mf_ft_prefers_high_mismatch():
    # Candidate 4 is close to class 0 but predicts class 1, giving a mismatch of 2, while candidates 5/6 have 0.
    emb = np.array([[0.0], [0.0], [10.0], [10.0], [0.05], [9.9], [10.1]], dtype=np.float32)
    seg_labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=np.int64)
    seg_proba = np.zeros((7, 2), dtype=np.float32)
    seg_proba[:, 1] = 1.0
    sel = MFFT().query(emb, np.array([4, 5, 6]), np.arange(4),
                       seg_proba=seg_proba, seg_labels=seg_labels, batch_size=1)
    assert sel.tolist() == [4]


def test_mf_ft_single_boundary_group_uses_farthest_first():
    # Equal candidate scores form a single boundary group, reducing to FT starting from labeled points.
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((12, 4)).astype(np.float32)
    seg_labels = np.zeros((12, 3), dtype=np.int64)
    seg_labels[:4] = [1, 0, 0]
    seg_proba = np.full((12, 3), 0.6, dtype=np.float32)
    labeled, cand = np.arange(4), np.arange(4, 12)
    sel = MFFT().query(emb, cand, labeled, seg_proba=seg_proba, seg_labels=seg_labels, batch_size=3)
    assert sel.tolist() == farthest_first(emb, cand, labeled, 3)


def test_mw_ft_weight_overrides_distance():
    # Nearby point 1 has a mismatch of 2 and distant point 2 has 0. FT selects 2, and MW-FT selects 1.
    emb = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    seg_labels = np.array([[1, 1], [0, 0], [0, 0]], dtype=np.int64)
    seg_proba = np.array([[0.0, 0.0], [0.0, 0.0], [0.9, 0.9]], dtype=np.float32)
    labeled, cand = np.array([0]), np.array([1, 2])
    assert FT().query(emb, cand, labeled, batch_size=1).tolist() == [2]
    sel = MWFT().query(emb, cand, labeled, seg_proba=seg_proba, seg_labels=seg_labels, batch_size=1)
    assert sel.tolist() == [1]




def _redundant_cluster_problem():
    """Construct candidates far from labeled points, with 1/2 tightly clustered at mismatch 3 and 3/4 isolated at mismatch 2."""
    emb = np.array([[100.0], [0.0], [0.1], [10.0], [20.0]], dtype=np.float32)
    seg_labels = np.zeros((5, 4), dtype=np.int64)
    seg_labels[0] = [1, 1, 1, 0]
    seg_proba = np.full((5, 4), 0.1, dtype=np.float32)
    seg_proba[0] = 0.0
    seg_proba[3:, 0] = 0.9
    return emb, seg_proba, seg_labels, np.array([0]), np.array([1, 2, 3, 4])


def test_mf_ft_gate_takes_redundant_cluster():
    emb, seg_proba, seg_labels, labeled, cand = _redundant_cluster_problem()
    sel = MFFT().query(emb, cand, labeled, seg_proba=seg_proba, seg_labels=seg_labels, batch_size=2)
    assert set(sel.tolist()) == {1, 2}


def test_mw_fl_declusters_redundant_high_mismatch_group():
    emb, seg_proba, seg_labels, labeled, cand = _redundant_cluster_problem()
    sel = set(MWFL().query(emb, cand, labeled, seg_proba=seg_proba, seg_labels=seg_labels, batch_size=2).tolist())
    assert len(sel) == 2 and len(sel & {1, 2}) == 1 and len(sel & {3, 4}) == 1


def test_fl_matches_mw_fl_when_mismatch_vanishes():
    # All-zero scores give equal weights, making MW-FL identical to FL.
    emb = np.array([[0.0], [0.0], [10.0], [10.0], [0.0], [5.0], [10.0]], dtype=np.float32)
    seg_labels = np.zeros((7, 3), dtype=np.int64)
    seg_proba = np.zeros((7, 3), dtype=np.float32)
    labeled, cand = np.arange(4), np.array([4, 5, 6])
    a = FL().query(emb, cand, labeled, batch_size=2)
    b = MWFL().query(emb, cand, labeled, seg_proba=seg_proba, seg_labels=seg_labels, batch_size=2)
    assert a.tolist() == b.tolist()
    assert len(set(b.tolist())) == 2 and set(b.tolist()) <= {4, 5, 6}


def test_fl_family_cold_start_is_uniform_facility_location():
    emb, *_ = _random_problem(n=20)
    cand = np.arange(20)
    expected = FL(random_state=0).query(emb, cand, EMPTY, batch_size=5).tolist()
    for cls in (MFFL, MWFL):
        assert cls(random_state=0).query(emb, cand, EMPTY, batch_size=5).tolist() == expected


def test_mf_fl_prefers_high_mismatch():
    emb = np.array([[0.0], [0.0], [10.0], [10.0], [0.05], [9.9], [10.1]], dtype=np.float32)
    seg_labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0], [0, 0]], dtype=np.int64)
    seg_proba = np.zeros((7, 2), dtype=np.float32)
    seg_proba[:, 1] = 1.0
    sel = MFFL().query(emb, np.array([4, 5, 6]), np.arange(4),
                       seg_proba=seg_proba, seg_labels=seg_labels, batch_size=1)
    assert sel.tolist() == [4]


def test_mf_fl_boundary_group_declusters():
    # The boundary group {1,2,3} has equal scores, with 1/2 clustered. Two selections cover the cluster and isolated point 3.
    emb = np.array([[100.0], [0.0], [0.2], [10.0]], dtype=np.float32)
    seg_labels = np.zeros((4, 4), dtype=np.int64)
    seg_labels[0] = [1, 1, 0, 0]
    seg_proba = np.zeros((4, 4), dtype=np.float32)
    sel = set(MFFL().query(emb, np.array([1, 2, 3]), np.array([0]),
                           seg_proba=seg_proba, seg_labels=seg_labels, batch_size=2).tolist())
    assert len(sel) == 2 and len(sel & {1, 2}) == 1 and 3 in sel


@pytest.mark.parametrize("cls", [FL, MFFL, MWFL])
def test_fl_family_threads_random_state_to_bandwidth(cls, monkeypatch):
    # More than 2000 candidates trigger sampling for bandwidth estimation, requiring random_state to be passed.
    seen = []
    original = coverage.median_sq_bandwidth

    def spy(emb, **kwargs):
        seen.append(kwargs.get("random_state", "MISSING"))
        return original(emb, **kwargs)

    monkeypatch.setattr(coverage, "median_sq_bandwidth", spy)
    monkeypatch.setattr(mf_fl, "median_sq_bandwidth", spy)  # MF-FL estimates bandwidth over all candidates within the module.
    emb, seg_proba, seg_labels, labeled, cand = _random_problem(n=60, num_classes=3)
    cls(random_state=7).query(emb, cand, labeled, seg_proba=seg_proba, seg_labels=seg_labels, batch_size=3)
    assert seen and all(state == 7 for state in seen)
