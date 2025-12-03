from typing import List, Dict, Iterable, Optional
import numpy as np
import os
import shutil
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Incar, Kpoints, Potcar


class VaspInputs:
    """
    Helpers to write VASP inputs (POSCAR/KPOINTS/INCAR/POTCAR/jobfile) from pymatgen
    Structures or JSON configs, with simple relax/scf/train presets and override hooks.

    Args:
        in_path: Template/input directory (expects INCAR, optional jobfiles/POTCAR).
        out_path: Destination directory for generated VASP inputs.
    """
    def __init__(
        self,
        in_path: str,
        out_path: str,
    ):
            self.in_path = in_path
            self.out_path = out_path
        

    def write_kpoints(self, mesh: np.array):
        kpoints = Kpoints.gamma_automatic(mesh,
                                            shift=(0, 0, 0))
        kpoints.write_file(f"{self.out_path}/KPOINTS")
    def write_poscar_from_idx(self):
        pass
    
    def write_poscar_from_json(self, struct):
        print('Update to static version')
        print(self.out_path)
        struct = Structure.from_dict(struct)
        poscar = Poscar(struct, sort_structure=True)
        poscar.write_file(f"{self.out_path}/POSCAR")
        
    @staticmethod
    def write_poscar_from_json_static(struct, out_path):
        struct = Structure.from_dict(struct)
        poscar = Poscar(struct, sort_structure=True)
        poscar.write_file(f"{out_path}/POSCAR")
        
    def write_potcar_from_poscar(self):
        poscar = Poscar.from_file(f"{self.in_path}/POSCAR")
        structure = poscar.structure
        elements = []
        for site in structure:
            sym = site.specie.symbol
            if sym not in elements:
                elements.append(sym)

        potcar = Potcar(symbols=elements, functional="PBE")
        potcar.write_file(f"{self.out_path}/POTCAR")

    def write_potcar_from_elements(self, elements: List[str]):
        potcar = Potcar(symbols=elements, functional="PBE")
        potcar.write_file(f"{self.out_path}/POTCAR")

    def write_jobfile(self, name, count):
        print('rewrite to static')
        shutil.copy(f'{self.in_path}/jobfiles/jobfile', f"{self.out_path}/jobfile")
        if name is not None:
            with open(f"{self.out_path}/jobfile", "r") as file:
                lines = file.readlines()
            with open(f"{self.out_path}/jobfile", "w") as file:
                for line in lines:
                    if line.strip().startswith("#SBATCH -J "):
                        file.write(f"#SBATCH -J {name}_{count}\n")
                        print('replaced job name')
                    else:
                        file.write(line)
    @staticmethod
    def write_jobfile_static(in_path, out_path, jobfile, name, ind, num_nodes=1, n_tasks=1):
        shutil.copy(f'{in_path}/{jobfile}', f"{out_path}/jobfile")
        with open(f"{out_path}/jobfile", "r") as file:
            lines = file.readlines()
        with open(f"{out_path}/jobfile", "w") as file:
            for line in lines:
                if line.strip().startswith("#SBATCH -J "):
                    file.write(f"#SBATCH -J {name}_{ind}\n")
                    print('replaced job name')
                    
                elif line.strip().startswith("#SBATCH -N "):
                    file.write(f"#SBATCH -N {num_nodes}\n")
                elif line.strip().startswith("#SBATCH --ntasks-per-node="):
                    file.write(f"#SBATCH --ntasks-per-node={n_tasks}\n")
                else:
                    file.write(line)

    def write_incar(
        self,
        relax: str,
        system: str,
        temp: Iterable[int] = (100, 400),
        updates: Optional[Dict[str, object]] = None,   # <-- put any "input: value" here
    ) -> str:
        """
        Write an INCAR with presets for `relax`, `scf`, or `train`, and then apply
        arbitrary user-specified overrides via `updates`.

        Parameters
        ----------
        relax : {'relax','scf','train'}
            Which preset to start from.
        system : str
            Value for SYSTEM.
        temp : iterable of two ints
            (TEBEG, TEEND) used only for 'train'.
        updates : dict[str, Any]
            Any extra INCAR key/values to add or change. Keys are case-insensitive.

        Returns
        -------
        str
            Path to the written INCAR.
        """
        t = tuple(temp)
        if len(t) != 2:
            raise ValueError("temp must have exactly two integers: (TEBEG, TEEND)")

        incar = Incar.from_file(os.path.join(self.in_path, "INCAR"))
        incar["SYSTEM"] = system

        r = relax.lower()
        if r == "relax":
            incar["IBRION"] = 2
            incar["ISIF"]   = 2
            incar["NSW"]    = 100
            incar["ISMEAR"] = 0
            incar["SIGMA"]  = 0.05
            incar["ISYM"]   = 0

        elif r == "scf":
            incar["ISMEAR"] = 0
            incar["SIGMA"]  = 0.05
            incar["IBRION"] = -1
            incar["ISIF"]   = 2
            incar["NSW"]    = 0

        elif r == "train":
            incar["LCHARG"]= False
            incar["LORBIT"] = 0
            incar["IBRION"] = 0
            incar["ISIF"]   = 2
            incar["ISMEAR"] = -1
            incar["SIGMA"]  = 0.0258
            incar["NSW"]    = 4000
            incar["POTIM"]  = 1
            incar["TEBEG"]  = t[0]
            incar["TEEND"]  = t[1]

            incar["ML_LMLFF"]   = True      
            incar["ML_MODE"]    = "train"
            incar["RANDOM_SEED"]= '42                0               0'

            incar["ML_ICRITERIA"] = 2
            incar["ML_CX"]        = 0.2 #default 0 (1 :less DFT runs: > 0, more dft runs: < 0; 2: 0.2 approximately every 50 steps a DFT run is performed)
            incar["ML_CTIFOR"]    = 0.002 #initial beysian error estimate to define later CTI_FOR
            incar["ML_WTIFOR"]    = 1.0
            incar["ML_WTOTEN"]    = 1.0
            incar["ML_WTSIF"]     = 1e-10

        else:
            raise ValueError("relax must be one of: 'relax', 'scf', 'train'")

        print(updates)
        # Apply arbitrary user overrides last (case-insensitive keys)
        if updates:
            for k, v in updates.items():
                incar[k.strip().upper()] = v

        out_path = os.path.join(self.out_path, "INCAR")
        incar.write_file(out_path)
        return out_path
        
    def copy_poscar_from_contcar(self):
        shutil.copy(f'{self.in_path}/CONTCAR', f"{self.out_path}/POSCAR")
        
    def copy_poscar_from_contcar_with_displacement(self, displacement_magnitude=0.00):
        contcar_path = os.path.join(self.in_path, 'CONTCAR')

        poscar_path = os.path.join(self.out_path, 'POSCAR')

        structure = Poscar.from_file(contcar_path).structure

        for site_index in range(len(structure)):
            displacement = np.random.normal(scale=displacement_magnitude, size=3)
            structure.translate_sites(site_index, displacement, frac_coords=False)

        poscar = Poscar(structure)
        poscar.write_file(poscar_path)
        
    def copy_potcar(self):
        shutil.copy(f'{self.in_path}/POTCAR', f"{self.out_path}/POTCAR")
        
    def copy_kpoints(self):
        shutil.copy(f'{self.in_path}/KPOINTS', f"{self.out_path}/KPOINTS")
    
    def move_ml_ab(self):
        shutil.move(f'{self.out_path}/ML_ABN', f"{self.out_path}/ML_ABN_T_0")
        
    def clear_ml_files(self):
        os.remove(f'{self.out_path}/ML_LOGFILE')
#         if relax == 'relax':
#             out_path = os.path.join(
#             output_dir,f"{path}/{count}/relax/"
#         )
#         else:
#             out_path = os.path.join(
#             output_dir,f"{path}/{count}/"
#         )
#         print(out_path)
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         gen.write_poscar(out_path)

#         vasp_in = VaspInputs(in_path=vasp_input_path,
#                             out_path=out_path)
#         vasp_in.write_incar(relax)
#         vasp_in.write_kpoints( kpoint_mesh)
#         vasp_in.write_jobfile(path, count)
#        # gen.write_potcar_from_poscar(out_path)
#         vasp_in.write_potcar_from_elements(potcar_title)
#        # gen.wirte_vasp_inputs(out_path, count)
#        # print(type(dists))
#         # with open(f"{out_path}/distance.txt", "a") as f:
#         #     f.write(f"\"{dists}\"")
#         if isinstance(dists, str):
#             dists = ast.literal_eval(dists)
#         assert isinstance(dists, dict)

#         # Turn tuple‐keys into JSON‐safe strings
#         jsonable = { str(k): v for k, v in dists.items() }

#         out = os.path.join(out_path, "distance.json")
#         with open(out, "w") as f:
#             json.dump(jsonable, f, indent=2)
#         print(f"Written config #{count} with distances {sorted(dists)}")
#         if count == n_configs:
#             print('Not all configurations found. Increase n_configs!')
