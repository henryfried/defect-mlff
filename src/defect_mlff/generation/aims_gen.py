from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import logging, os, shutil
import numpy as np
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.aims.inputs import AimsGeometryIn
from pymatgen.io.aims.sets import AimsInputSet
log = logging.getLogger(__name__)
# AimsInputs: helpers to write FHI-aims geometry/control and jobfiles from pymatgen Structures.


class AimsInputs:
    """
    Write FHI-aims inputs (geometry.in, control.in, jobfile) from pymatgen Structures.

    Key args:
    - out_path: destination folder (or base folder for multi-frame writes)
    - defaults_2020 / basis: point to your FHI-aims species_defaults/<basis> directory
    - params: optional control.in overrides; also supports magmom seeding for collinear spin
    """
    def __init__(self, out_path: str | Path):
        self.out_path = Path(out_path)


    def load_structures_from_config(self, config: Dict[str, Any]) -> List[Structure]:
        """
        Accepts config with 'structure' (pymatgen dict or file path).
        Also supports 'structure_path'/'poscar' for CLI convenience.
        """
        spec = config.get("structure") or config.get("structure_path") or config.get("poscar")
        if spec is None:
            raise KeyError(
                "Config entry is missing a 'structure' key. "
                "If you passed a YAML with sampling metadata (home_dir/material/...), "
                "use the sampling script to produce per-config JSONs, or provide "
                "a config dict with a 'structure' path/dict to write inputs."
            )
        if isinstance(spec, dict):
            return [Structure.from_dict(spec)]
        return [Structure.from_file(Path(spec))]

    def load_structures_from_xyz(self, xyz_path: str | Path) -> List[Structure]:
        """Reads single/multi-frame XYZ and returns periodic pymatgen Structures."""
        frames = ase_read(xyz_path, index=":")
        if not isinstance(frames, list):
            frames = [frames]

        adaptor = AseAtomsAdaptor()
        structs: List[Structure] = []
        for i, atoms in enumerate(frames):
            # has_lattice = (atoms.get_cell().volume() > 0) or bool(np.any(atoms.pbc))
            # if not has_lattice:
            #     raise ValueError(
            #         f"XYZ frame {i} has no lattice/PBC; cannot create a periodic Structure."
            #     )
            structs.append(adaptor.get_structure(atoms))
        return structs

    def write_aims_inputs_for_structures(
        self,
        structures: Sequence[Structure],
        *,
        preset: str,
        defaults_2020: str | Path,
        basis: str = "intermediate",
        params: Optional[Dict[str, Any]] = None,
        # optional magmom seeding inputs:
        defect_pos_cart: Optional[Sequence[float] | Sequence[Sequence[float]]] = None,
        mag_cutoff: float = 2.6,
        mag_seed: float = 0.30,
        only_element: Optional[str] = None,
        subdirs_for_multiple: bool = True,
    ) -> List[Path]:
        """
        Writes control.in (+ geometry.in) using pymatgen AimsInputSet.

        If params['spin'] == 'collinear' and defect_pos_cart is provided, seeds 'magmom'
        on sites within mag_cutoff Å of the defect center(s), with moment 'mag_seed'.
        """
        r = preset.lower().strip()
        if r not in {"relax", "relax_cell", "scf"}:
            raise ValueError("preset must be one of: 'relax', 'relax_cell', 'scf'")

        # Defaults for control.in (pymatgen uses FHI-aims tag names)
        defaults_root = Path(defaults_2020)
        species_dir = str(defaults_root / basis)
        # Pymatgen expects AIMS_SPECIES_DIR in its SETTINGS/config; set both env and runtime map.
        env_dir = os.environ.get("AIMS_SPECIES_DIR")
        if env_dir is None or Path(env_dir).resolve() != defaults_root.resolve():
            os.environ["AIMS_SPECIES_DIR"] = str(defaults_root)
        try:
            from pymatgen.io.aims import inputs as _aims_inputs
            _aims_inputs.SETTINGS["AIMS_SPECIES_DIR"] = str(defaults_root)
        except Exception:
            pass

        user_params: Dict[str, Any] = {
            "output_level": "MD_light",
            "xc": "pbe",
            "relativistic": "atomic_zora scalar",
            "charge": 0,
            "spin": "none",
            "k_grid": [1, 1, 1],
            "sc_iter_limit": 200,
            "sc_accuracy_rho": 1e-6,
            "compute_forces": ".true.",
            "species_dir": species_dir,
            "occupation_type":  "gaussian 0.01",
        }
        if r == "relax":
            user_params.update(relax_geometry="bfgs 0.001", relax_unit_cell="none")
        elif r == "relax_cell":
            user_params.update(relax_geometry="bfgs 0.001", relax_unit_cell="full")

        if params:
            user_params.update(params)

        # Optionally seed magmoms (structures only; requires defect center(s))
        spin_mode = str(user_params.get("spin", "none")).lower()
        structs_mut: List[Structure] = list(structures)

        if spin_mode == "collinear":
            if defect_pos_cart is None:
                log.warning(
                    "spin='collinear' but no defect_pos_cart provided → skipping magmom seeding."
                )
            else:
                user_params.setdefault("default_initial_moment", [0, 0, 0])
                log.info(
                    "Seeding magmoms: seed=%.2f µB, cutoff=%.2f Å, only_element=%s",
                    mag_seed, mag_cutoff, only_element,
                )
                structs_mut = [
                    self.magmom(
                        s,
                        defect_pos_cart=defect_pos_cart,
                        cutoff=mag_cutoff,
                        seed=mag_seed,
                        only_element=only_element,
                    )
                    for s in structs_mut
                ]

        # Prepare writer and output dirs
        def _build_input_set(struct: Structure):
            """
            Build an AimsInputSet across pymatgen versions.
            Newer pymatgen expects `parameters` (not `user_params`), so try that first.
            """
            # Preferred signature (current pymatgen): parameters=<dict>, structure=<Structure>
            try:
                return AimsInputSet(parameters=user_params, structure=struct)
            except TypeError as exc_params:
                # Some builds accept the structure first
                try:
                    return AimsInputSet(struct, parameters=user_params)
                except Exception:
                    pass
                # Last resort: raise with a clear hint
                raise RuntimeError(
                    "Failed to construct AimsInputSet with the available pymatgen version. "
                    "Please upgrade pymatgen (>=2024.5 with aims support) or check the API."
                ) from exc_params

        out_dirs: List[Path] = []

        def _rewrite_geometry(struct: Structure, target: Path | str) -> None:
            """Rewrite geometry.in via pymatgen and strip autogenerated headers."""
            target_path = Path(target)
            target_path.mkdir(parents=True, exist_ok=True)

            geom_path = target_path / "geometry.in"
            geom_path.unlink(missing_ok=True)

            geom = AimsGeometryIn.from_structure(struct)
            geom.write_file(directory=target_path)

            atom_lines = 0
            if geom_path.exists():
                with geom_path.open() as fh:
                    body = [line for line in fh if not line.lstrip().startswith("#")]
                geom_path.write_text("".join(body))
                for line in body:
                    if line.strip().startswith(("atom", "atom_frac")):
                        atom_lines += 1
            if atom_lines != len(struct):
                raise RuntimeError(
                    f"geometry.in in {target} has {atom_lines} atoms, "
                    f"but the Structure contains {len(struct)}."
                )

        # Single structure (or force single dir)
        if len(structs_mut) == 1 or not subdirs_for_multiple:
            iset = _build_input_set(structs_mut[0])
            iset.pop("parameters.json", None)
            iset.pop("geometry.in", None)
            iset.write_input(self.out_path)
            _rewrite_geometry(structs_mut[0], self.out_path)
            out_dirs.append(self.out_path)
            return out_dirs

        # Multiple structures → numbered subfolders
        for i, s in enumerate(structs_mut):
            sub = self.out_path / f"frame_{i:03d}"
            sub.mkdir(parents=True, exist_ok=True)
            iset = _build_input_set(s)
            iset.pop("parameters.json", None)
            iset.write_input(sub)
            out_dirs.append(sub)

        return out_dirs

    def write_from_config(
        self,
        config: Dict[str, Any],
        apply_defects: bool = False,
        **writer_kwargs,
    ) -> List[Path]:
        """
        Convenience wrapper: loads structures from a config and writes inputs.

        Parameters
        ----------
        config : dict
            Expected to contain either:
            - a fully *defected* structure under "structure" (preferred), OR
            - a *pristine* structure + defect metadata (e.g. "defect_pos_cart").
        apply_defects : bool, default False
            If True, pass defect metadata through so the writer applies defects
            to a *pristine* structure. If False (default), DO NOT pass defect
            metadata—assume the provided structure(s) are already defected.

        Any additional kwargs are passed to `write_aims_inputs_for_structures`.
        """
        # Load structures (often a single Structure) from the config
        structs = self.load_structures_from_config(config)
        # Guard against double-applying defects:
        # - Default is NOT to apply defects again when a defected structure is given.
        # - Only forward defect metadata if the caller explicitly opts in.
        if apply_defects:
            if "defect_pos_cart" in config and "defect_pos_cart" not in writer_kwargs:
                writer_kwargs["defect_pos_cart"] = config["defect_pos_cart"]
        else:
            # Strip any defect metadata that might cause removals a second time
            for k in ("defect_pos_cart", "defect_indices", "defects"):
                writer_kwargs.pop(k, None)

        return self.write_aims_inputs_for_structures(structs, **writer_kwargs)

    def write_from_xyz(self, xyz_path: str | Path, **writer_kwargs) -> List[Path]:
        """
        Convenience wrapper: loads structures from a (multi-frame) XYZ and writes inputs.
        """
        structs = self.load_structures_from_xyz(xyz_path)
        return self.write_aims_inputs_for_structures(structs, **writer_kwargs)


    def magmom(
        self,
        s: Structure,
        defect_pos_cart: Sequence[float] | Sequence[Sequence[float]],
        cutoff: float = 2.6,
        seed: float = 0.30,
        only_element: Optional[str] = None,  # e.g. "S" or "Mo"; None = all species
    ) -> Structure:
        """
        Add collinear initial magnetic moments to atoms near one or more defects.

        Parameters
        ----------
        s : Structure
            Pymatgen Structure (modified in-place and also returned).
        defect_pos_cart : [x,y,z] or list of such points (Å)
            Defect center(s) in Cartesian Å.
        cutoff : float
            Neighbor cutoff radius (Å).
        seed : float
            Moment (µB) to assign to neighbors.
        only_element : str | None
            If set, only seed atoms with this element symbol.

        Returns
        -------
        Structure
            The same structure with a 'magmom' site property set.
        """
        # Normalize to list of centers
        def _is_point(x):
            return (
                isinstance(x, (list, tuple, np.ndarray))
                and len(x) == 3
                and not isinstance(x[0], (list, tuple, np.ndarray))
            )

        centers = (
            [np.array(defect_pos_cart, dtype=float)]
            if _is_point(defect_pos_cart)
            else [np.array(p, dtype=float) for p in defect_pos_cart]  # type: ignore[arg-type]
        )

        neighbor_idx: set[int] = set()
        for c in centers:
            hits = s.get_sites_in_sphere(c, cutoff)
            # PeriodicNeighbor objects on modern pymatgen; tuples on older versions
            for h in hits:
                idx = getattr(h, "index", h[2])  # fallback for old tuple API
                if only_element is None or s[idx].specie.symbol == only_element:
                    neighbor_idx.add(idx)

        mags = [0.0] * len(s)
        for i in neighbor_idx:
            mags[i] = float(seed)

        if "magmom" in s.site_properties:
            s.remove_site_property("magmom")
        s.add_site_property("magmom", mags)
        return s
    @staticmethod
    def write_jobfile_static(in_path: str, out_path: str, jobfile: str, name: str,
                             ind: int, num_nodes: int = 1, n_tasks: int = 128) -> str:
        src = os.path.join(in_path, jobfile)
        dst = os.path.join(out_path, "jobfile")
        shutil.copy(src, dst)

        with open(dst, "r") as fh:
            lines = fh.readlines()

        with open(dst, "w") as fh:
            for line in lines:
                s = line.strip()
                if s.startswith("#SBATCH -J "):
                    fh.write(f"#SBATCH -J {name}_{ind}\n")
                elif s.startswith("#SBATCH -N "):
                    fh.write(f"#SBATCH -N {num_nodes}\n")
                elif s.startswith("#SBATCH --ntasks-per-node="):
                    fh.write(f"#SBATCH --ntasks-per-node={n_tasks}\n")
                else:
                    fh.write(line)
        return dst
