import random
import logging
from typing import List, Optional, Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.util.coord import pbc_diff

log = logging.getLogger(__name__)

class PymatgenPOSCARDefectGenerator:
    """
    A class to generate supercells and introduce point defects (vacancies,
    substitutions, antisites) using pymatgen, with optional layer targeting
    ("top"/"bottom") for any species and support for multi-defect configurations.

    Usage examples:
        gen = PymatgenPOSCARDefectGenerator("POSCAR", supercell=(2,2,1), seed=42)
        # Single vacancy on any Mo site
        idxs = gen.introduce_defects(defect_type="V", defect_species=["Mo"], n=1)
        # Double vacancy on S in bottom layer
        idxs = gen.introduce_defects(defect_type="V", defect_species=["S"], n=2, layer="bottom")
        # Substitute 3 In for Cu host sites
        idxs = gen.introduce_defects(defect_type="Sub", defect_species=["Cu", "In"], n=3)
        gen.write_poscar("out_dir")
    """

    def __init__(
        self,
        poscar_path: Optional[str] = None,
        supercell: Tuple[int, int, int] = (1, 1, 1),
        seed: Optional[int] = None,
        structure: Optional[Structure] = None,
    ):
        """
        Initialize with either a POSCAR path or an in-memory Structure.

        Args:
            poscar_path: Path to a POSCAR/structure file (used if `structure` is None).
            supercell: Supercell replication factors (nx, ny, nz).
            seed: Optional RNG seed for deterministic defect placement.
            structure: Optional pre-loaded pymatgen Structure; bypasses file load.
        """
        if structure is None:
            if poscar_path is None:
                raise ValueError("Provide either poscar_path or structure to PymatgenPOSCARDefectGenerator.")
            structure = Structure.from_file(poscar_path)
        # Load structure and apply supercell
        self.structure = structure.copy()
        if seed is not None:
            random.seed(seed)
        if supercell != (1, 1, 1):
            nx, ny, nz = supercell
            matrix = [[nx, 0, 0], [0, ny, 0], [0, 0, nz]]
            self.structure.make_supercell(matrix)

    def find_defect_candidates(
        self,
        host_symbol: str,
        layer: Optional[str] = None,
        layer_center: float = 0.25,
    ) -> List[int]:
        """
        Find indices of sites matching host_symbol.
        If layer is specified, filter by fractional z coordinate:
          "bottom": z < 0.25, "top": z >= 0.25.
        """
        # log.info(f'layer center set to {layer_center}')
        candidates: List[int] = []
        for i, site in enumerate(self.structure):
            if site.specie.symbol != host_symbol:
                continue
            if layer:
                z = site.frac_coords[2] % 1
                if layer.lower() == "bottom" and z >= layer_center:
                    continue
                if layer.lower() == "top" and z < layer_center:
                    continue
                # if layer.lower() == "top-bot":
                #     self._pick_vertical_pair(host_symbol)
                
            candidates.append(i)
        return candidates
    def introduce_defects(
        self,
        defect_type: str,
        defect_species: List[str],
        n: int = 1,
        layer: Optional[str] = None,
        avoid_vertical_pairs: bool = False, 
    ) -> List[int]:
        """
        Introduce point defects into the structure.

        :param defect_type: 'V' for vacancy, 'Sub' for substitution, 'As' for antisite, '2V' for vertical double vacancy.
        :param defect_species:
            - ['X']       → remove X sites (vacancy)
            - ['X','Y']   → replace X with Y (substitution/antisite)
        :param n: Number of defects to introduce (e.g., n=2 for a double vacancy).
        :param layer: Optional layer filter ("top" or "bottom") for selecting
                    sites by fractional z coordinate.
        :param avoid_vertical_pairs: If True (and layer is None) for single vacancies,
                    ensure no two vacancies are placed on top of each other by picking
                    at most one site per (x,y) column.
        :returns: List of indices of modified sites.
        """
        expected_max = self.structure.num_sites
        # Determine host and new species
        if defect_type == "V" or defect_type == "2V":
            host_symbol = defect_species[0]
        else:
            host_symbol, new_symbol = defect_species

        # Verify host present
        elements = {el.symbol for el in self.structure.composition.elements}
        if host_symbol not in elements:
            raise ValueError(f"Host element '{host_symbol}' not in structure.")

        # Get candidate indices (apply layer filter if given)
        candidates = self.find_defect_candidates(host_symbol, layer)
        if len(candidates) < n:
            layer_info = f" in {layer} layer" if layer else ''
            raise ValueError(
                f"Only {len(candidates)} '{host_symbol}' sites{layer_info}, cannot introduce {n} defects."
            )
        # avoid_vertical_pairs = True
        # Select and apply defects
        if defect_type == "V":
            if avoid_vertical_pairs and layer is None:
                # Group candidate sites by (x,y) key → one group per vertical column
                top_candidates = self.find_defect_candidates(host_symbol, layer="top")
                chosen_tmp = random.sample(top_candidates, n)
                pairs = [self.get_vertical_partner(i)[0] for i in chosen_tmp]

                chosen_pairs = np.vstack([chosen_tmp, pairs]).T  # shape (n, 2)
                chosen = [random.choice(row.tolist()) for row in chosen_pairs]

            else:
                chosen = random.sample(candidates, n)

            self.structure.remove_sites(chosen)

        elif defect_type == "2V":
            chosen = random.sample(candidates, n)
            for defect in chosen:
                pairs = self.get_vertical_partner(defect)
                self.structure.remove_sites(pairs)
            self.structure.remove_sites(chosen)

        else:
            chosen = random.sample(candidates, n)
            for idx in sorted(chosen, reverse=True):
                self.structure.replace(idx, new_symbol)
        drop = expected_max - self.structure.num_sites
        # print(drop)
        expected_drop = n if defect_type == "V" else (2*n if defect_type == "2V" else 0)
        assert drop >= expected_drop, f"Removed {drop}, expected at least {expected_drop}"
        return chosen

    def _frac_xy_dist2(self, i: int, j: int) -> float:
        fi = self.structure[i].frac_coords % 1.0
        fj = self.structure[j].frac_coords % 1.0
        dx = abs(fi[0] - fj[0]) % 1.0
        dy = abs(fi[1] - fj[1]) % 1.0
        if dx > 0.5: dx = 1.0 - dx
        if dy > 0.5: dy = 1.0 - dy
        return dx*dx + dy*dy

    def get_vertical_partner(
        self,
        idx: int,
        zc: float = 0.25,         # use ~0.5 for typical MoS2 monolayers
        frac_tol: float = 5e-3,  # tolerance in fractional coords
    ) -> List[int]:
        """
        Return [partner_idx] from the opposite layer with same (x,y) under PBC.
        """
        elem = self.structure[idx].specie.symbol
        zf = (self.structure[idx].frac_coords[2] % 1.0)
        target_layer = "bottom" if zf >= zc else "top"

        pool = self.find_defect_candidates(elem, layer=target_layer, layer_center=zc)
        if not pool:
            raise ValueError(f"No {target_layer} candidates for {elem}.")

        tol2 = (frac_tol ** 2)
        partner = None
        for j in pool:
            if self._frac_xy_dist2(idx, j) <= tol2:
                if partner is not None:
                    raise ValueError(f"Multiple partners for site {idx}: {partner}, {j}")
                partner = j

        if partner is None:
            raise ValueError(f"No opposite-layer partner found for site {idx}.")
        return [partner]

    
    # def _xy_key(self, idx: int) -> Tuple[float, float]:
    #     x, y = self.structure[idx].frac_coords[:2]
    #     if x <0:
    #         x +=1
    #     if y <0:
    #         y +=1
    # #     # return x, y
    #     print(x,y)
    #     return (round(x % 1, 4), round(y % 1, 4))
    # # def _xy_key(self, idx: int, prec: int = 2) -> Tuple[int, int]:
    # #     """
    # #     Return a periodic (x,y) key as integer bins.
    # #     Using bins avoids 0.9999 vs 0.0001 mismatches and negative coords.
    # #     """
    # #     x, y = self.structure[idx].frac_coords[:2]
    # #     m = 10 ** prec
    # #     ix = int(round((x % 1.0) * m)) % m   # 0.99995*m -> m -> %m -> 0
    # #     iy = int(round((y % 1.0) * m)) % m
    #     return (ix, iy)
    # def get_vertical_partner(
    #     self,
    #     idx: int,
    #     zc: float = 0.25,
    #     frac_tol: float= 5e-2,
    # ) -> int:
    #     """
    #     Given the index of a top-layer site (layer z>=zc) for element X,
    #     return the index of its corresponding bottom-layer partner at the same (x,y).
    #     Raises if no partner found.
    #     """
    #     element = self.structure[idx].specie.symbol
    #     xf, yf, zf = (self.structure[idx].frac_coords % 1.0)
    #     # print(xf, yf, zf)

    #     # target the opposite layer
    #     target_layer = "bottom" if zf >= zc else "top"
    #     pool = self.find_defect_candidates(element, layer=target_layer, layer_center=zc)

    #     partner = None
    #     for j in pool:
    #         xj, yj, _ = (self.structure[j].frac_coords % 1.0)
    #         print(xj,yj,_)
    #         # self.structure.get_distance[j,idx]
    #         # minimum-image delta in fractional coords (ignore z by setting to 0)
    #         dv = pbc_diff([xf, yf, 0.0], [xj, yj, 0.0])  # -> array in [-0.5, 0.5)
    #         # dv = get_all_distances
    #         # print(dv)
    #         # if np.hypot(dv[0], dv[1]) <= frac_tol:
    #         if np.linalg.norm(([xf,yf], [xj, yj])) <= frac_tol:
    #             if partner is not None:
    #                 raise ValueError(f"Multiple opposite-layer partners for site {idx}: {partner}, {j}")
    #             partner = j

    #     if partner is None:
    #         raise ValueError(f"No opposite-layer partner found for site {idx}.")
    #     return [partner]
        # # Determine key and element
        # element = self.structure[idx].specie.symbol
        # key = self._xy_key(idx)
        # z_val = self.structure[idx].frac_coords[2] % 1
        # # Determine partner layer
        # if z_val < zc:
        #     raise ValueError(f"Site {idx} is not in the top layer (z={z_val}).")
        # partner_layer = 'bottom'
        # # Find candidates in bottom layer matching key
        # partners = [i for i in self.find_defect_candidates(element, partner_layer, zc)
        #             if self._xy_key(i) == key]
        
        # if not partners:
        #     raise ValueError(f"No bottom-layer partner found for site {idx}.")
        
        # return partners

    def write_poscar(self, out_path: str):
        """
        Write current structure to a POSCAR file under out_path/POSCAR.
        """
        poscar = Poscar(self.structure, sort_structure=True)
        poscar.write_file(f"{out_path}/POSCAR")
