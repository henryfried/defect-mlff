
from typing import Tuple, List, Dict
from pymatgen.core import Structure
import itertools
    
def get_distances_by_bucket(defect_indices, struct):
    """
    Build a dict of distance buckets between defect groups, without any zero distances.

    Parameters
    ----------
    defect_indices : List[List[int]]
        defect_indices[i] is a list of site‐indices of defects of type i.
    struct : pymatgen Structure
        The structure to measure distances in.

    Returns
    -------
    dists : Dict[(int, int), List[float]]
        For each pair (i, j) with i <= j, the sorted list of distances
        between each defect in group i and each defect in group j,
        excluding any self‐comparisons (zero distances).
    """
    lattice = struct.lattice
    dists = {}

    for i, j in itertools.combinations_with_replacement(range(len(defect_indices)), 2):
        idxs_i = defect_indices[i]
        idxs_j = defect_indices[j]

        if i == j:
            pairs = itertools.combinations(idxs_i, 2)
        else:
            pairs = itertools.product(idxs_i, idxs_j)

        bucket = []
        for a, b in pairs:
            if a == b:
                continue

            d = lattice.get_all_distances(
                struct.frac_coords[a],
                struct.frac_coords[b]
            ).ravel()
            d_min = float(d.min())
            bucket.append(round(float(d_min), 3))

        bucket.sort()
        dists[(i, j)] = bucket

    return dists
