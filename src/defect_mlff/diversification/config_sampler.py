import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet
from tqdm import tqdm
from pymatgen.core import Structure
import numpy as np

# Local imports
from defect_mlff.generation.distances import get_distances_by_bucket
from defect_mlff.generation.defect_generator import PymatgenPOSCARDefectGenerator

logger = logging.getLogger(__name__)

class DefectConfigSampler:
    """
    Samples unique defect configurations and computes inter-defect distance buckets.
    Optionally computes atomic descriptors using a MACE model.
    """
    def __init__(
        self,
        primitive_path: str,
        supercell: List[int],
        defect_types: List[str],
        defect_species: List[List[str]],
        n_defects: List[int]
    ):
        self.primitive_path = Path(primitive_path)
        self.supercell = tuple(supercell)
        self.defect_types = defect_types
        self.defect_species = defect_species
        self.n_defects = n_defects
        self._base_structure = self._load_structure(self.primitive_path)

    def _load_structure(self, path: Path) -> Structure:
        name = path.name.lower()
        if name == "geometry.in" or path.suffix.lower() == ".in":
            try:
                from pymatgen.io.aims.inputs import AimsGeometryIn
            except ImportError as exc:
                raise RuntimeError("Reading FHI-aims geometries requires pymatgen's aims support.") from exc
            geom = AimsGeometryIn.from_file(str(path))
            return geom.structure
        return Structure.from_file(str(path))

    def _generate_structure(
        self, seed: int, layer: Optional[str] = None,
    ) -> Tuple[Structure, List[List[int]], PymatgenPOSCARDefectGenerator]:
        """
        Generate a defected structure and record defect site indices.
        """
        generator = PymatgenPOSCARDefectGenerator(
            structure=self._base_structure.copy(), supercell=self.supercell, seed=seed
        )
        defect_indices = [
            generator.introduce_defects(
                self.defect_types[i], self.defect_species[i],  self.n_defects[i], layer
            )
            for i in range(len(self.defect_types))
        ]
        struct = self._base_structure.copy()
        if self.supercell != (1, 1, 1):
            struct.make_supercell([
                [self.supercell[0], 0, 0],
                [0, self.supercell[1], 0],
                [0, 0, self.supercell[2]]
            ])
        base = self._base_structure
        nx, ny, nz = self.supercell
        expected_max = base.num_sites * nx * ny * nz
        actual = struct.num_sites
        assert actual <= expected_max, (
            f"Sanity check failed: struct has {actual} sites, but base({base.num_sites}) "
            f"* supercell {nx}x{ny}x{nz} = {expected_max}."
        )
        return struct, defect_indices, generator


    def sample_configs(
        self,
        n_configs: int = 50,
        require_unique: bool = True,
        descriptor_model: Optional[str] = None,
        layer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate defect configurations, distance buckets, and optional descriptors.

        Parameters
        ----------
        n_configs : int
            Number of configurations to generate.
        require_unique : bool
            Skip duplicates based on distance fingerprint.
        descriptor_model : Optional[str]
            Path to MACE model file, or None to skip descriptors.

        Returns
        -------
        List[Dict[str, Any]]
            Generated configuration data.
        """
        if n_configs <= 0:
            logger.warning("n_configs <= 0: returning empty configuration list.")
            return []

        all_configs: List[Dict[str, Any]] = []
        seen: Set[FrozenSet[float]] = set()
        trial = 0
        count = 0

        calc = None
        adaptor = None
        if descriptor_model:
            try:
                from mace.calculators import MACECalculator
                from pymatgen.io.ase import AseAtomsAdaptor
            except ImportError as exc:
                raise RuntimeError("MACECalculator not available: install mace to use descriptors.") from exc
            calc = MACECalculator(model_paths=descriptor_model, device="cpu")
            adaptor = AseAtomsAdaptor()
            logger.info("Descriptor calculation enabled using MACE.")

        while count < n_configs:
            trial += 1
            seed = 42 * trial
            struct, defect_indices, gen = self._generate_structure(seed, layer)
            dists = get_distances_by_bucket(defect_indices, struct)

            key = tuple(tuple(dists[k]) for k in sorted(dists.keys()))
            if require_unique and key in seen:
                if trial >= 20 * n_configs:
                    logger.error(f"No new configs after {trial} trials, aborting.")
                    break
                continue

            seen.add(key)
            buckets = {f"({i},{j})": dists[(i, j)] for (i, j) in sorted(dists.keys())}

            descriptors: Optional[List[float]] = None
            if descriptor_model:
                ase_atoms = adaptor.get_atoms(gen.structure)
                descriptors = calc.get_descriptors(
                    ase_atoms, invariants_only=False, num_layers=-1
                ).tolist()
            config_data = {
                "config": count,
                "structure": gen.structure.as_dict(),
                "defect_indices": defect_indices,
                "defect_pos_cart": struct.cart_coords[defect_indices].squeeze().tolist(),
                "buckets": buckets,
                "descriptors": descriptors
            }
            all_configs.append(config_data)
            count += 1
            #logger.info(f"Generated config {count}.")

        return all_configs

    def write_configs(
        self,
        output_dir: str,
        all_configs: List[Dict[str, Any]]
    ) -> None:
        """
        Write configurations to JSON, naming file by defect parameters.
        """
        os.makedirs(output_dir, exist_ok=True)
        tag = "_".join(
            f"{self.n_defects[i]}_{self.defect_types[i]}_" + "_".join(self.defect_species[i])
            for i in range(len(self.defect_types))
        ).strip("_")
        safe_tag = tag.replace(os.sep, "_")[:100]
        out_path = os.path.join(output_dir, f"{safe_tag}.json")
        with open(out_path, "w") as fp:
            json.dump(all_configs, fp, indent=2)
        logger.info(f"Wrote {len(all_configs)} configs to {out_path}.")





#@typechecked
class DisplacementConfigSampler:
    """
    Puts noise on atomic positions and saves them into a JSON.
    """
    def __init__(
        self,
        struct_path: dict,
        displacement: float,
        seed: int = 42,
    ):
        self.structure = Structure.from_file(struct_path)
        self.displ = displacement
        self.rng = np.random.default_rng(seed)
        
    def random_noise(self, structure):
        for site_index in range(len(structure)):
            displ = self.rng.normal(0.0, self.displ, size=3)
            structure.translate_sites(site_index, displ, frac_coords=False)
        return structure
    
    def sample_displ(self, n_displacements):
        all_displ = []
        s = self.structure.copy()
        for count in tqdm(range(n_displacements)):
            
            struct = self.random_noise(s)
            
            displ_data = {
                "config": count,
                "structure": struct.as_dict(),
                }
            all_displ.append(displ_data)
        return all_displ
    
    def write_displ(
        self,
        output_dir: str,
        all_displ: List[Dict[str, Any]]
    ) -> None:
        """
        Write configurations to JSON, naming file by defect parameters.
        """
        out_path = os.path.join(output_dir, f"displacement.json")
        with open(out_path, "w") as fp:
            json.dump(all_displ, fp, indent=2)
        logger.info(f"Wrote {len(all_displ)} configs to {out_path}.") 
   
