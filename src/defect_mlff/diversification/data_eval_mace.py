from __future__ import annotations

import json
import logging
from typing import Iterable, Optional, List, Sequence

import numpy as np
from pymatgen.core import Structure
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize as sk_normalize
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataEqualityDisplacementMACE:
    """
    Compute a pairwise distance matrix from pooled MACE descriptors for structures.
    """
    def __init__(
        self,
        json_path: Optional[str] = None,
        configs: Optional[Iterable[Structure]] = None,
        *,
        invariants_only: bool = False,
        num_layers: int = -1,
        pool: str = "mean",
        normalize: bool = True,
        metric: str = "euclidean",
        defect_pos_cart: Optional[Sequence[Sequence[float]]] = None,
        defect_radius: Optional[float] = None,
        use_rematch: bool = False,
        rematch_metric: str = "rbf",
        rematch_gamma: float = 1.0,
        rematch_alpha: float = 1.0,
        rematch_threshold: float = 1e-6,
        rematch_normalize_kernel: bool = True,
        **kwargs,
    ) -> None:
        if json_path:
            self.structures = self.load_displaced(json_path)
        elif configs is not None:
            self.structures = list(configs)
        else:
            raise ValueError("Provide either 'json_path' or 'configs'.")

        try:
            from mace.calculators import MACECalculator
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise RuntimeError("MACECalculator not available: install mace to use descriptors.") from exc

        mace_kwargs = dict(kwargs)
        if "model_paths" not in mace_kwargs:
            raise ValueError("model_paths must be provided in kwargs for MACE descriptors.")
        calc = MACECalculator(**mace_kwargs)
        adaptor = AseAtomsAdaptor()

        self.defect_pos_cart = self._normalize_defect_pos(defect_pos_cart)
        self.defect_radius = defect_radius

        descs = []
        for struct in tqdm(self.structures):
            ase_atoms = adaptor.get_atoms(struct)
            desc = calc.get_descriptors(
                ase_atoms, invariants_only=invariants_only, num_layers=num_layers
            )
            desc = self._filter_by_defect_radius(desc, struct)
            descs.append(desc)

        if not descs:
            self.features = np.empty((0, 0))
            self.K = np.empty((0, 0))
            self.X = np.empty((0, 0))
            return

        if use_rematch:
            try:
                from dscribe.kernels import REMatchKernel
            except ImportError as exc:
                raise RuntimeError("REMatchKernel not available: install dscribe to use REMatch.") from exc

            self.features = descs
            re = REMatchKernel(
                metric=rematch_metric,
                gamma=rematch_gamma,
                alpha=rematch_alpha,
                threshold=rematch_threshold,
                normalize_kernel=rematch_normalize_kernel,
            )
            K = re.create(descs)
            self.K = 0.5 * (K + K.T)
            if rematch_normalize_kernel:
                self.X = np.sqrt(np.clip(2 - 2 * self.K, 0, None))
            else:
                diag = np.diag(self.K)
                self.X = np.sqrt(
                    np.clip(diag[:, None] + diag[None, :] - 2 * self.K, 0, None)
                )
        else:
            pooled = [self._pool_descriptors(d, pool) for d in descs]
            if pool == "flatten":
                lengths = [p.size for p in pooled]
                max_len = max(lengths)
                if any(l != max_len for l in lengths):
                    logger.warning(
                        "Flattened descriptors have variable lengths (min=%d, max=%d); "
                        "padding with zeros to max length.",
                        min(lengths),
                        max_len,
                    )
                    features = np.zeros((len(pooled), max_len), dtype=float)
                    for i, arr in enumerate(pooled):
                        features[i, : arr.size] = arr
                else:
                    features = np.vstack(pooled)
            else:
                features = np.vstack(pooled)
            if normalize:
                features = sk_normalize(features, norm="l2")
            self.features = features
            self.K = None
            self.X = pairwise_distances(features, metric=metric)

    @staticmethod
    def load_displaced(json_path: str) -> List[Structure]:
        with open(json_path) as f:
            raw = json.load(f)
        return [Structure.from_dict(entry.get("structure", entry)) for entry in raw]

    @staticmethod
    def _normalize_defect_pos(
        defect_pos_cart: Optional[Sequence[Sequence[float]]]
    ) -> Optional[List[np.ndarray]]:
        if defect_pos_cart is None:
            return None
        if len(defect_pos_cart) == 3 and isinstance(defect_pos_cart[0], (int, float)):
            return [np.array(defect_pos_cart, dtype=float)]
        return [np.array(p, dtype=float) for p in defect_pos_cart]

    def _filter_by_defect_radius(self, desc: np.ndarray, struct: Structure) -> np.ndarray:
        if self.defect_pos_cart is None:
            return np.asarray(desc, dtype=float)
        if self.defect_radius is None:
            raise ValueError("defect_radius must be set when defect_pos_cart is provided.")

        arr = np.asarray(desc, dtype=float)
        if arr.ndim == 1:
            return arr

        frac_sites = struct.frac_coords
        frac_defects = np.array(
            [struct.lattice.get_fractional_coords(p) for p in self.defect_pos_cart]
        )
        dists = struct.lattice.get_all_distances(frac_sites, frac_defects)
        min_dists = np.min(dists, axis=1)
        mask = min_dists <= self.defect_radius
        return arr[mask]

    @staticmethod
    def _pool_descriptors(desc: np.ndarray, pool: str) -> np.ndarray:
        arr = np.asarray(desc, dtype=float)
        if arr.ndim == 1:
            return arr
        if pool == "mean":
            return arr.mean(axis=0)
        if pool == "sum":
            return arr.sum(axis=0)
        if pool == "flatten":
            return arr.reshape(-1)
        raise ValueError(f"Unknown pool='{pool}' for MACE descriptors.")
