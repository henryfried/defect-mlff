from __future__ import annotations

import json
import os
import logging
import numpy as np
from typing import Iterable, Optional, Tuple, Union, List, Dict, Any

from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm
logger = logging.getLogger(__name__)

class DataEquality:
    """
    Build a bucket-distance feature matrix from sampled defect configs.

    Loads bucketized pairwise distances (and defect metadata) from JSON or a list of
    configs, pads them to fixed-length vectors, and L2-normalizes the matrix for
    downstream selection methods such as FPS/greedy Max-min or k-means medoid.
    """

    def __init__(self, json_path=None, configs=None):
        if json_path:
            self.configs, self.defect_pos_cart = self.load_configs(json_path)
        elif configs is not None:
            self.configs = configs
        else:
            raise ValueError("Provide either 'json_path' or 'configs'.")
        self.X = self.build_feature_matrix(self.configs)
        if self.X.size:
            self.X = sk_normalize(self.X, norm="l2")

    @staticmethod
    def load_configs(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        configs = []
        for entry in raw:
            buckets = entry.get("buckets", entry)
            dist = {}
            for k, v in buckets.items():
                i_str, j_str = k.strip("()").split(",")
                i, j = int(i_str), int(j_str)
                dist[(i, j)] = v
            configs.append(dist)
        defect_pos_cart = []
        for entry in raw:
            defect_pos_cart.append(entry.get("defect_pos_cart", entry))
        return configs, defect_pos_cart

    @staticmethod
    def build_feature_matrix(configs):
        keys = sorted({k for d in configs for k in d})
        max_len = {k: max(len(d.get(k, [])) for d in configs) for k in keys}
        X = []
        for d in configs:
            row = []
            for k in keys:
                arr = sorted(d.get(k, []))
                arr += [0.0] * (max_len[k] - len(arr))
                row.extend(arr)
            X.append(row)
        return np.array(X, dtype=float)
    
    def write_json(
        self,
        output_dir: str,
        dict: List[Dict[str, Any]]
    ) -> None:
        """
        Write configurations to JSON, naming file by defect parameters.
        """
        out_path = os.path.join(output_dir, f"defect_pos_cart.json")
        with open(out_path, "w") as fp:
            json.dump(dict, fp, indent=2)
        logger.info(f"Wrote {len(dict)} configs to {out_path}.") 



class DataEqualityDisplacement:
    """
    Compute a SOAP+REMatch similarity matrix for a set of displaced/defected structures.

    Loads structures from JSON (list of Structure dicts) or an iterable of Structures,
    builds per-atom SOAP descriptors, L2-normalizes each environment, and returns an
    NxN REMatch kernel matrix in `self.X`. Species set is unified across all inputs so
    descriptor length is consistent.
    """
    def __init__(
        self,
        json_path: Optional[str] = None,
        configs: Optional[Iterable[Structure]] = None,
        *,
        r_cut: float = 8.0,
        n_max: int = 2,
        l_max: int = 3,
        sigma: float = 0.5,
        periodic: bool = True,
        metric: str = "linear",      # or "rbf"
        alpha: float = 0.1,          # REMatch regularization
        gamma: Optional[float] = None,  # for metric="rbf"
        normalize_kernel: bool = True,
        n_jobs: int = 1,
    ) -> None:
        # Load structures
        if json_path:
            self.structures = self.load_displaced(json_path)
        elif configs is not None:
            self.structures = list(configs)
        else:
            raise ValueError("Provide either 'json_path' or 'configs'.")

        # Species set across all structures (consistent descriptor length)
        species = sorted({sp for s in self.structures for sp in s.symbol_set})

        # SOAP descriptor: per-atom (average=None / 'off'), row-normalized later
        self.soap = SOAP(
            species=species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            average='off',            # IMPORTANT: per-atom environments for REMatch
        )
        self.n_jobs = int(n_jobs)

        # REMatch kernel
        self.re = REMatchKernel(
            metric=metric,
            alpha=alpha,
            gamma=gamma,
            normalize_kernel=normalize_kernel,
            threshold=1e-6,
        )

        # Build similarity (kernel) matrix now
        self.X = self.build_similarity_matrix(self.structures)  # X is K (N×N)

    @staticmethod
    def load_displaced(json_path: str) -> List[Structure]:
        with open(json_path) as f:
            raw = json.load(f)
        return [Structure.from_dict(entry.get("structure", entry)) for entry in raw]

    def calc_soap(self, struct: Structure) -> np.ndarray:
        """Per-atom SOAP, row-normalized: shape (n_atoms, n_feat)."""
        atoms = AseAtomsAdaptor.get_atoms(struct)
        X = self.soap.create(atoms, n_jobs=self.n_jobs)  # (n_atoms, n_feat)
        X = np.asarray(X, dtype=float)
        # L2-normalize each atom environment so REMatch compares on a consistent scale
        X = sk_normalize(X)
        return X

    def build_similarity_matrix(self, structures: Iterable[Structure]) -> np.ndarray:
        """Compute the REMatch kernel across all structures."""
        env_descs = [self.calc_soap(s) for s in tqdm(structures)]
        K = self.re.create(env_descs)         # (N, N)
        K = 0.5 * (K + K.T)                   # symmetrize (numerical hygiene)
        return np.asarray(K, dtype=float)

def greedy_maxmin_from_distance(D, k, start='pair', rng=None):
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

    if start == 'random':
        rng = np.random.default_rng(rng)
        i0 = int(rng.integers(n))
        i1 = int(np.argmax(D[:, i0]))
    else:  # start='pair': farthest off-diagonal pair
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

def greedy_min_max(features, k, seed,metric='euclidean'):
    """
    Farthest-first traversal selection of k samples.
    """
    n = features.shape[0]
    if k > n:
        logger.warning("Requested k=%d exceeds sample count n=%d; returning all indices.", k, n)
        return [x for x in range(n)]
    else:
        np.random.seed(seed)
        first = np.random.randint(n)
        selected = [first]
        D = pairwise_distances(features, metric=metric)
        min_dists = D[first].copy()
        for _ in range(1, k):
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)
            min_dists = np.minimum(min_dists, D[next_idx])

        # exit()
        return selected

def greedy_radius_cover(features, radius, seed=None, metric='euclidean'):
    """
    Select as many samples as possible such that every chosen point is at least
    `radius` away from its nearest previously-selected neighbor.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix (n_samples x n_features).
    radius : float
        Minimum pairwise distance to maintain between selected samples.
    seed : int | None
        Optional random seed for choosing the initial sample.
    metric : str
        Distance metric passed to `pairwise_distances`.

    Returns
    -------
    List[int]
        Indices of selected samples. The list stops automatically once the
        farthest candidate is closer than `radius` to the current set.
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

def kmeans_medoid(features, k, seed,):
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



class DataEqualityDisplacementManual:
    """
    Build Δr and angular histograms for displaced structures relative to a reference,
    using a fixed neighbor list. Captures radial and directional changes per structure.
    """

    def __init__(
        self,
        reference_struct: Union[str, Structure],
        defect_centers: Optional[List[float]] = None,
        json_path: Optional[str] = None,
        configs: Optional[Iterable[Structure]] = None,
        r_cutoff: float = 3.5,
        bins: int = 21,            # prefer odd → centered at 0
        density: bool = True,
        delta_range: Optional[Tuple[float, float]] = None,  # e.g., (-0.06, 0.06)
    ) -> None:
        # Reference structure
        self.ref: Structure = (
            Structure.from_file(reference_struct)
            if isinstance(reference_struct, str)
            else reference_struct
        )
        self.defect_centers = defect_centers
        # Fixed pair list and reference distances r0 + pair unit vectors e0
        self.pairs, self.r0, self._images, self._e0 = self._build_ref_pairs(self.ref, r_cutoff)

        # Load displaced structures
        if json_path:
            self.structures = self.load_displaced(json_path)
        elif configs is not None:
            self.structures = list(configs)
        else:
            raise ValueError("Provide either 'json_path' or 'configs'.")

        # Shared histogram edges
        self.edges = self._delta_edges(self.ref, self.structures, self.pairs, bins, delta_range)
        self.cos_edges = np.linspace(-1.0, 1.0, 13)  # fixed 12 bins, centered at 0

        # Feature matrix X (K × (nbins_dr + nbins_cos))
        self.X = self.build_feature_matrix(self.structures, self.edges, self.cos_edges, density)

    @staticmethod
    def load_displaced(json_path: str) -> list[Structure]:
        with open(json_path) as f:
            raw = json.load(f)
        return [Structure.from_dict(entry.get("structure", entry)) for entry in raw]


    def calc_soap(
        self,
        struct: Structure,
       # defect_centers: List[float],
        *,
        r_cut: float = 8,
        n_max: int = 8,
        l_max: int = 3,
        sigma: float = 0.6,
        periodic: bool = True,
        average: Optional[str] = "off",   # "outer" → per-structure vector; None → per-atom
        normalize: bool = False,
    ):
        """
        Return a (geometry-only) SOAP descriptor for `struct`.

        Parameters
        ----------
        struct : pymatgen.core.Structure
            Structure to describe.
        rcut, nmax, lmax, sigma, periodic : SOAP hyperparameters (DScribe).
            Choose to roughly match your planned MACE cutoff/resolution.
        average : {"outer", None}
            - "outer": per-structure SOAP (atom-averaged) → 1D vector
            - None: per-atom SOAP → (N_atoms, n_features)
        normalize : bool
            L2-normalize the returned vector(s) (recommended before distances/FPS).

        Returns
        -------
        np.ndarray
            If average="outer": shape (n_features,)
            If average=None:   shape (N_atoms, n_features)
        """

        if not hasattr(self, "_soap_species"):
            all_specs = set(struct.symbol_set)
            for s in getattr(self, "structures", []):
                all_specs |= set(s.symbol_set)
            self._soap_species = sorted(all_specs)


        self._soap = SOAP(
            species=self._soap_species,
            periodic=periodic,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            average=average,)

        atoms = AseAtomsAdaptor.get_atoms(struct)
        if self.defect_centers:
            desc = self._soap.create(atoms, centers=self.defect_centers)  # (n_features,) or (N_atoms, n_features)
        else:
            desc = self._soap.create(atoms) 

        if normalize:
            if average == "outer":
                norm = float(np.linalg.norm(desc))
                if norm > 0:
                    desc = desc / norm
            else:
                norms = np.linalg.norm(desc, axis=1, keepdims=True)
                desc = desc / np.where(norms > 0, norms, 1.0)


        return desc.flatten()

    @staticmethod
    def _build_ref_pairs(ref: Structure, r_cutoff: float):
        ii, jj, images, d = ref.get_neighbor_list(r_cutoff)
        m = ii < jj    # dont take the same twice
        ii, jj, images, d = ii[m], jj[m], images[m], d[m]
        pairs = np.stack([ii, jj], axis=1)
        f = ref.frac_coords
        dfrac = f[jj] - f[ii] + images  # fractional pair vectors consistent with neighbor list
        v0 = ref.lattice.get_cartesian_coords(dfrac)  # (M,3)
        nrm = np.linalg.norm(v0, axis=1, keepdims=True)
        e0 = v0 / np.clip(nrm, 1e-12, None)
        return pairs, d, images, e0

    @staticmethod
    def _delta_edges(
        ref: Structure,
        structs: Iterable[Structure],
        pairs: np.ndarray,
        bins: int,
        delta_range: Optional[Tuple[float, float]],
    ) -> np.ndarray:
        if bins % 2 == 0:
            bins += 1
        if delta_range is not None:
            lo, hi = delta_range
            return np.linspace(lo, hi, bins + 1)
        r0 = ref.distance_matrix[pairs[:, 0], pairs[:, 1]]
        dmin, dmax = 0.0, 0.0
        for s in structs:
            dm = s.distance_matrix
            dr = dm[pairs[:, 0], pairs[:, 1]] - r0
            dmin = min(dmin, float(dr.min()))
            dmax = max(dmax, float(dr.max()))
        m = max(abs(dmin), abs(dmax)) + 5e-3
        return np.linspace(-m, m, bins + 1)

    def _pair_distances(self, struct: Structure) -> np.ndarray:
        dm = struct.distance_matrix
        return dm[self.pairs[:, 0], self.pairs[:, 1]]

    def _pair_delta_vectors(self, struct: Structure) -> np.ndarray:
        """Return Δv_ij = v_ij(struct) - v_ij(ref) for each pair, in cartesian coords.
        Uses the same images from the reference neighbor list to maintain consistency.
        """
        f = struct.frac_coords
        dfrac = f[self.pairs[:, 1]] - f[self.pairs[:, 0]] + self._images
        v = struct.lattice.get_cartesian_coords(dfrac)
        # reconstruct reference vectors from e0 and r0 length
        v0 = self._e0 * self.r0[:, None]
        return v - v0
    
    @staticmethod
    def _prob_hist(x: np.ndarray, edges: np.ndarray, weights: Optional[np.ndarray] = None, density: bool = True) -> np.ndarray:
        """Return an L1-normalized histogram (probability vector) if density=True.
        Shapes are enforced to avoid the common "weights shape" error.
        """
        x = np.asarray(x).ravel()
        w = None if weights is None else np.asarray(weights).ravel()
        h, _ = np.histogram(x, bins=edges, weights=w, density=False)
        if density:
            tot = (w.sum() if w is not None else h.sum())
            h = h / tot if tot > 0 else h
        return h
    
    def spherical_fingerprint(self, struct, bins_theta=12, bins_phi=12, min_mag=1e-4, polar_collapse_eps=1e-3):
        dv  = self._pair_delta_vectors(struct)      # (M,3) using your existing method
        mag = np.linalg.norm(dv, axis=1)
        m   = mag > min_mag
        if not np.any(m):
            return np.zeros(bins_theta * bins_phi)
        n = dv[m] / mag[m][:, None]
        cos_theta = np.clip(n[:, 2], -1.0, 1.0)
        phi = np.arctan2(n[:, 1], n[:, 0])
        if polar_collapse_eps is not None:
            pole = np.abs(cos_theta) > (1.0 - polar_collapse_eps)
            phi[pole] = 0.0
        cos_edges = np.linspace(-1.0, 1.0, bins_theta + 1)
        phi_edges = np.linspace(-np.pi, np.pi, bins_phi + 1)
        H, _, _ = np.histogram2d(cos_theta, phi, bins=[cos_edges, phi_edges], density=False)
        H = H / H.sum() if H.sum() > 0 else H
      #  print(H.ravel().shape)
        return H.ravel()
    
    def build_feature_matrix(self, structs: Iterable[Structure], dr_edges: np.ndarray, cos_edges: np.ndarray, density: bool) -> np.ndarray:
        rows = []
        for s in structs:
            # Δr histogram
            dr = self._pair_distances(s) - self.r0
            # h_dr, _ = np.histogram(dr, bins=dr_edges, density=density)

            # Direction-cosine histogram (weighted by |Δv| to suppress noise)
            dv = self._pair_delta_vectors(s)
            mag = np.linalg.norm(dv, axis=1)
            # avoid division by zero
            safe_mag = np.where(mag > 1e-12, mag, 1.0)
            cos = (dv * self._e0).sum(axis=1) / safe_mag
            cos = np.clip(cos, -1.0, 1.0)
          #  weights = mag  # weight by magnitude
            # h_cos, _ = np.histogram(cos, bins=cos_edges, weights=weights, density=False)
            # if density:
            h_dr  = self._prob_hist(dr,  dr_edges)               # L1-normalized Δr histogram
            h_cos = self.spherical_fingerprint(s)
            soap = self.calc_soap(s)
            rows.append(np.concatenate([h_dr, h_cos, soap]))
        
        ncols = (len(dr_edges) - 1) + (len(cos_edges) - 1)
        return np.vstack(rows) if rows else np.empty((0, ncols))
    


    def angle_change_hist(self, struct: Structure, bins: int = 12, density: bool = True, edges: Optional[np.ndarray] = None) -> np.ndarray:
        """Histogram of bond-direction change cosΔθ = ê·ê0 for the current structure.

        ê0: unit bond vectors in the reference (precomputed)
        ê:  unit bond vectors in the given structure (using the SAME images as ref)

        Parameters
        ----------
        struct : Structure
            Displaced structure.
        bins : int
            Number of bins over [-1, 1]. Ignored if `edges` is provided.
        density : bool
            If True, normalize histogram to sum to 1.
        edges : Optional[np.ndarray]
            Optional custom bin edges over [-1, 1]. If None, use linspace.
        """
        f = struct.frac_coords
        dfrac = f[self.pairs[:, 1]] - f[self.pairs[:, 0]] + self._images
        v = struct.lattice.get_cartesian_coords(dfrac)
        nrm = np.linalg.norm(v, axis=1, keepdims=True)
        e = v / np.clip(nrm, 1e-12, None)
        cos_dtheta = (e * self._e0).sum(axis=1)
        cos_dtheta = np.clip(cos_dtheta, -1.0, 1.0)
        if edges is None:
            edges = np.linspace(-1.0, 1.0, bins + 1)
        hist, _ = np.histogram(cos_dtheta, bins=edges, density=density)
        return hist
