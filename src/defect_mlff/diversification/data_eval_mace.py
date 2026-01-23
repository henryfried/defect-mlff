from __future__ import annotations

import json
import logging
from typing import Iterable, Optional, List

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
        descriptor_model: str,
        device: str = "cpu",
        invariants_only: bool = False,
        num_layers: int = -1,
        pool: str = "mean",
        normalize: bool = True,
        metric: str = "euclidean",
    ) -> None:
        if json_path:
            self.structures = self.load_displaced(json_path)
        elif configs is not None:
            self.structures = list(configs)
        else:
            raise ValueError("Provide either 'json_path' or 'configs'.")

        if not descriptor_model:
            raise ValueError("descriptor_model must be provided for MACE descriptors.")

        try:
            from mace.calculators import MACECalculator
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as exc:
            raise RuntimeError("MACECalculator not available: install mace to use descriptors.") from exc

        calc = MACECalculator(model_paths=descriptor_model, device=device)
        adaptor = AseAtomsAdaptor()

        descs = []
        for struct in tqdm(self.structures):
            ase_atoms = adaptor.get_atoms(struct)
            desc = calc.get_descriptors(
                ase_atoms, invariants_only=invariants_only, num_layers=num_layers
            )
            vec = self._pool_descriptors(desc, pool)
            descs.append(vec)

        if descs:
            features = np.vstack(descs)
            if normalize:
                features = sk_normalize(features, norm="l2")
            self.features = features
            self.X = pairwise_distances(features, metric=metric)
        else:
            self.features = np.empty((0, 0))
            self.X = np.empty((0, 0))

    @staticmethod
    def load_displaced(json_path: str) -> List[Structure]:
        with open(json_path) as f:
            raw = json.load(f)
        return [Structure.from_dict(entry.get("structure", entry)) for entry in raw]

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
