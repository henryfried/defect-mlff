import numpy as np
import pytest

from defect_mlff.diversification.data_eval import (
    greedy_maxmin_from_distance,
    greedy_min_max,
    kmeans_medoid,
)


@pytest.fixture
def toy_features():
    """
    Small, well-spaced feature set to exercise the selection helpers without fixtures.
    """
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ]
    )


def test_greedy_min_max_is_deterministic_and_unique(toy_features):
    sel1 = greedy_min_max(toy_features, k=3, seed=42)
    sel2 = greedy_min_max(toy_features, k=3, seed=42)
    assert sel1 == sel2
    assert len(sel1) == len(set(sel1)) == 3


def test_greedy_maxmin_from_distance_picks_farthest_pair():
    D = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )
    sel = greedy_maxmin_from_distance(D, k=2, start="pair")
    assert set(sel) == {0, 2}


def test_kmeans_medoid_returns_valid_unique_indices(toy_features):
    sel = kmeans_medoid(toy_features, k=2, seed=0)
    assert len(sel) == len(set(sel)) == 2
    assert all(0 <= idx < len(toy_features) for idx in sel)
