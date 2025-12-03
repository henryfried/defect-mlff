
import os
from  pymlff import MLAB
from pymatgen.io.aims.outputs import AimsOutput
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
import numpy as np
from typing import List
#from MLAB import MLAB  # Assuming MLAB is a class from the MLAB module

class MLABHandler:
    def __init__(self, input_dir: str):
        self.input_dir = input_dir

    def parse_ml_ab(self, path: str) -> MLAB:
        """Parse a single ML_ABN file."""
        return MLAB.from_file(os.path.join(path, "ML_ABN"))

    def combine_ml_ab(self, ml_ab_list: List[MLAB]) -> MLAB:
        """Merge multiple MLAB instances into one."""
        if not ml_ab_list:
            raise ValueError("ml_ab_list cannot be empty")

        final_ml_ab = ml_ab_list[0]
        for ml_ab in ml_ab_list[1:]:
            final_ml_ab += ml_ab
        return final_ml_ab

    def write_ml_ab(self, ml_ab: MLAB, filename: str) -> None:
        """Write the MLAB object to a file."""
        ml_ab.write_file(filename)

    def generate_xyz(self, ml_ab: MLAB, filename: str) -> None:
        """Generate an .xyz file from MLAB data."""
        ml_ab.write_extxyz(f"{filename}.xyz", "eV/A^3")

    def merge_all_in_directory(self, output_filename: str) -> None:
        """Parse and merge all ML_ABN files in the input directory."""
        ml_ab_list = []
        for entry in os.listdir(self.input_dir):
            full_path = os.path.join(self.input_dir, entry)
            if os.path.isdir(full_path):  # Ensure it's a directory
                try:
                    ml_ab = self.parse_ml_ab(full_path)
                    ml_ab_list.append(ml_ab)
                except Exception as e:
                    print(f"Warning: Failed to parse {full_path}: {e}")

        if not ml_ab_list:
            raise RuntimeError("No valid ML_ABN files found to merge.")

        merged = self.combine_ml_ab(ml_ab_list)
        self.write_ml_ab(merged, output_filename)


class ParseAimstoMACE:
    def __init__(self,
        input_dir
    ):    
        self.ao = AimsOutput.from_outfile(f"{input_dir}/aims.out")
        
        self.structures = self.ao.results if isinstance(self.ao.results, list) else [self.ao.final_structure]
        self.forces_all   = self.ao.all_forces if hasattr(self.ao, "all_forces") else [self.ao.forces]
        self.energies_all = [self.ao.final_energy] if not isinstance(self.ao.results, list) else [
                            AimsOutput.from_outfile("aims.out").get_results_for_image(i).energy
                            for i in range(len(self.structures))]
    def write_data_set(self):
    # 2) Convert each image to ASE Atoms and attach energy/forces
        adaptor = AseAtomsAdaptor()
        images = []
        for i, struct in enumerate(self.structures):
            at = adaptor.get_atoms(struct)
            if self.forces_all and self.forces_all[i] is not None:
                at.arrays["forces"] = np.array(self.forces_all[i])  # eV/Å expected by MACE
            if self.energies_all and self.energies_all[i] is not None:
                at.info["energy"] = float(self.energies_all[i])     # eV
            images.append(at)

        # 3) Write Extended XYZ for MACE
        write("dataset.extxyz", images, format="extxyz")
