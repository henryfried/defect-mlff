from __future__ import annotations

import logging
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)


def greedy_maxmin_from_distance(D, k, start="pair", rng=None) -> List[int]:
    """
    Greedy farthest-point sampling on a distance matrix D (larger = more distant).
    Returns k selected indices.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)

    if k == 1:
        return [int(np.argmax(D.mean(axis=1)))]

    if start == "random":
        rng = np.random.default_rng(rng)
        i0 = int(rng.integers(n))
        i1 = int(np.argmax(D[:, i0]))
    else:  # start="pair": farthest off-diagonal pair
        M = D.copy()
        np.fill_diagonal(M, -np.inf)
        i0, i1 = np.unravel_index(np.argmax(M), M.shape)

    selected = [i0, i1]

    dmin = np.minimum(D[:, i0], D[:, i1])
    dmin[selected] = -np.inf  # do NOT reselect already-picked indices

    for _ in range(1, k):
        nxt = int(np.argmax(dmin))          # farthest (max of closest distances)
        selected.append(nxt)
        dmin = np.minimum(dmin, D[:, nxt])  # update with one new column
        dmin[selected] = -np.inf

    return selected


def greedy_min_max(features, k, seed, metric="euclidean") -> List[int]:
    """
    Farthest-first traversal selection of k samples.
    """
    n = features.shape[0]
    if k > n:
        logger.warning("Requested k=%d exceeds sample count n=%d; returning all indices.", k, n)
        return [x for x in range(n)]
    np.random.seed(seed)
    first = np.random.randint(n)
    selected = [first]
    D = pairwise_distances(features, metric=metric)
    min_dists = D[first].copy()
    for _ in range(1, k):
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, D[next_idx])

    return selected


def greedy_radius_cover(features, radius, seed=None, metric="euclidean") -> List[int]:
    """
    Select as many samples as possible such that every chosen point is at least
    `radius` away from its nearest previously-selected neighbor.
    """
    if radius <= 0:
        raise ValueError("radius must be positive.")

    n = features.shape[0]
    if n == 0:
        return []

    rng = np.random.default_rng(seed)
    D = pairwise_distances(features, metric=metric)
    D = 0.5 * (D + D.T)  # numerical symmetry

    first = int(rng.integers(n))
    selected = [first]

    min_dists = D[first].copy()
    min_dists[first] = 0.0

    while True:
        next_idx = int(np.argmax(min_dists))
        max_min = float(min_dists[next_idx])
        if max_min < radius:
            break
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, D[next_idx])
        min_dists[next_idx] = 0.0

    return selected


def kmeans_medoid(features, k, seed) -> List[int]:
    """
    Medoid selection based on KMeans clustering.
    """
    km = KMeans(n_clusters=k, random_state=seed).fit(features)
    centers = km.cluster_centers_
    labels = km.labels_
    sel = []
    for ci in range(k):
        members = np.where(labels == ci)[0]
        if members.size == 0:
            continue
        dists = np.linalg.norm(features[members] - centers[ci], axis=1)
        sel.append(int(members[np.argmin(dists)]))
    return sel
