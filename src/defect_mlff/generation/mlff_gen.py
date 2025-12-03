
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
import  os
import shutil
from defect_mlff.generation.vasp_gen import VaspInputs

import os
import shutil
import logging

log = logging.getLogger(__name__)


class MLFFJobGenerator:
    """
    Utilities to stage MLFF jobs (relaxations/phonons) by copying templates and
    generating POSCARs from JSON configs, preserving directory structure.

    Args:
        home_dir: Root directory for MLFF runs.
        input_root: Base folder containing template input files.
    """
    def __init__(self, home_dir: str, input_root: str = 'input_files'):
        self.home_dir = home_dir
        self.input_root = input_root

    def generate_relaxation(self, in_path: str, struct: str, output_dir: str, ind: int,
                            name: str, relax_script: str = 'relax.py', jobfile: str = 'jobfile_cpu'):
        """
        Generate unique defect configurations based on all pairwise distances between vacancies.
        """
        out_path = os.path.join(output_dir, f'{ind}')
        print(out_path)
        os.makedirs(out_path, exist_ok=True)
        VaspInputs.write_poscar_from_json_static(struct, out_path)
        shutil.copy(os.path.join(in_path, relax_script), os.path.join(out_path, relax_script))
        VaspInputs.write_jobfile_static(in_path, out_path, jobfile, name, ind)

    def generate_phonon(self, material: str, cp_from: str = 'relax', cp_to: str = 'phonon',
                        python_file: str = 'phonons.py', subpath: str = '', filename: str = None):
        """
        Copies one file from each directory under cp_from/subpath to cp_to/subpath,
        preserving directory structure, plus adds required phonon job input files.
        """
        src_base = os.path.join(self.home_dir, cp_from, subpath) if subpath else os.path.join(self.home_dir, cp_from)
        dest_base = os.path.join(self.home_dir, cp_to, subpath) if subpath else os.path.join(self.home_dir, cp_to)

        if not os.path.isdir(src_base):
            raise ValueError(f"Source path '{src_base}' does not exist or is not a directory")
        if not os.path.isdir(self.input_root):
            raise ValueError(f"Input directory '{self.input_root}' not found")

        for dirpath, _, files in os.walk(src_base):
            if not files:
                continue

            # Choose file to copy
            if filename and filename in files:
                chosen = filename
            elif filename:
                continue
            else:
                chosen = files[0]

            rel_dir = os.path.relpath(dirpath, src_base)
            target_dir = os.path.join(dest_base, rel_dir)
            os.makedirs(target_dir, exist_ok=True)

            shutil.copy2(os.path.join(dirpath, chosen), os.path.join(target_dir, chosen))
            print(f"Copied {chosen} to {target_dir}")

            shutil.copy(os.path.join(self.input_root, python_file), os.path.join(target_dir, 'phonons.py'))
            shutil.copy(os.path.join(self.input_root, 'jobfile_cpu_phonon'), os.path.join(target_dir, 'jobfile_cpu_phonon'))

            last_folder = os.path.basename(os.path.normpath(dirpath))
            ancestor = os.path.dirname(os.path.normpath(dirpath))
            print(ancestor)

            self._update_jobfile_name(target_dir, material, last_folder)

    def _update_jobfile_name(self, target_dir: str, material: str, last_folder: str):
        jobfile_path = os.path.join(target_dir, 'jobfile_cpu_phonon')
        if os.path.isfile(jobfile_path):
            with open(jobfile_path, 'r') as f:
                lines = f.readlines()
            with open(jobfile_path, 'w') as f:
                for line in lines:
                    if line.strip().startswith('#SBATCH -J '):
                        f.write(f"#SBATCH -J {material}_{last_folder}_ph\n")
                        print(f"Replaced job name with {material}_{last_folder}_ph")
                    else:
                        f.write(line)

def generate_relaxation_mlff(
    in_path: str,
    struct: str,
    output_dir: str,
    ind: int,
    name:str,
    relax: str= 'relax.py',
    jobfile: str = 'jobfile_cpu'
    
 ):
    """
    Generate unique defect configurations based on all pairwise distances between vacancies.
    """
    log.warning("Define path to trained MACE model in your relax.py, as it is not yet implemented.")
    out_path = os.path.join(output_dir, f'{ind}')
    os.makedirs(out_path, exist_ok=True)
    VaspInputs.write_poscar_from_json_static(struct, out_path)
    shutil.copy(f'{in_path}/{relax}', f"{out_path}/relax.py")
    VaspInputs.write_jobfile_static(in_path, out_path, jobfile,name, ind)
 

        
def generate_mlff_phonon(
    home_dir,
    name,
    cp_from='relax',
    cp_to='phonon',
    input_root='input_files',
    python_file= 'phonons.py',
    subpath='',
    filename=None
    
):
    """
    Copies one file from each directory under home_dir/cp_from/subpath to home_dir/cp_to/subpath,
    preserving directory structure, plus copies all files listed in input_files
    from home_dir/input_dir_name into each target folder.

    Parameters:
    - home_dir: base directory (e.g., os.path.expanduser('~')).
    - input_files: list of filenames to copy from input_dir_name.
    - cp_from: name of source parent directory under home_dir.
    - cp_to: name of destination parent directory under home_dir.
    - input_dir_name: name of directory under home_dir holding additional input files.
    - subpath: relative path under cp_from and cp_to to process (e.g., 'B' or 'A/B').
    - filename: specific file to copy per folder; if None, use first file found.
    """
    # Build absolute paths for source, destination, and inputs
    src_base = os.path.join(home_dir, cp_from, subpath) if subpath else os.path.join(home_dir, cp_from)
    dest_base = os.path.join(home_dir, cp_to, subpath) if subpath else os.path.join(home_dir, cp_to)
    #input_root = os.path.join(home_dir, input_dir_name)
    
    # Validate directories
    if not os.path.isdir(src_base):
        raise ValueError(f"Source path '{src_base}' does not exist or is not a directory")
    if not os.path.isdir(input_root):
        raise ValueError(f"Input directory '{input_root}' not found")

    # Traverse source tree
    for dirpath, _, files in os.walk(src_base):
        if not files:
            continue

        # Select file to copy from source
        if filename and filename in files:
            chosen = filename
        elif filename:
            continue  # specified filename not present, skip
        else:
            chosen = files[0]

        # Compute relative path from src_base and target directory
        rel_dir = os.path.relpath(dirpath, src_base)
        target_dir = os.path.join(dest_base, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        # Copy the chosen source file
        src_file = os.path.join(dirpath, chosen)
        dst_file = os.path.join(target_dir, chosen)
        shutil.copy2(src_file, dst_file)
        print(f"Copied {src_file} -> {dst_file}")
        print(dirpath)
        shutil.copy(f'{input_root}/{python_file}', f"{target_dir}/phonons.py")
        shutil.copy(f'{input_root}/jobfile_cpu_phonon', f"{target_dir}/jobfile_cpu_phonon")
       # count = 0
        last_folder = os.path.basename(os.path.normpath(dirpath))  
        for _ in range(1):
            ancestor = os.path.dirname(os.path.normpath(dirpath))
        print(ancestor)
        jobfile_path = os.path.join(target_dir, 'jobfile_cpu_phonon')
        if os.path.isfile(jobfile_path):
            with open(jobfile_path, 'r') as f:
                lines = f.readlines()
            with open(jobfile_path, 'w') as f:
                for line in lines:
                    if line.strip().startswith('#SBATCH -J '):
                        f.write(f"#SBATCH -J {name}_{last_folder}_ph\n")
                        print(f"Replaced job name with {last_folder}_ph")
                    else:
                        f.write(line)
        # Copy additional input files into target
        # for infile in input_files:
        #     src_in = os.path.join(input_root, infile)
        #     if not os.path.isfile(src_in):
        #         print(f"Warning: {src_in} does not exist, skipping.")
        #         continue
        #     dst_in = os.path.join(target_dir, infile)
        #     shutil.copy2(src_in, dst_in)
        #     print(f"Copied {src_in} -> {dst_in}")
