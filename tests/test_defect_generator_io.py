import os
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from defect_mlff.generation.defect_generator import PymatgenPOSCARDefectGenerator


def test_write_poscar_errors_when_directory_missing(tmp_path: Path):
    """
    write_poscar should fail if the target directory does not exist (no implicit mkdir).
    """
    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    gen = PymatgenPOSCARDefectGenerator(structure=struct)

    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        gen.write_poscar(str(missing_dir))
