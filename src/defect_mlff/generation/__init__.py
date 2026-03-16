# generation/__init__.py

# --- Distance utilities -------------------------------------------------------
from .distances import get_distances_by_bucket

# --- Defect generators & VASP I/O ---------------------------------------------
from .defect_generator import (
    PymatgenPOSCARDefectGenerator,
)
from .vasp_gen import VaspInputs

# AIMS support is optional; avoid import errors when only VASP/diversification is used.
try:
    from .aims_gen import AimsInputs  # type: ignore
except ModuleNotFoundError:
    AimsInputs = None  # type: ignore

# Phonopy support is optional; avoid import errors when phonopy is not installed.
try:
    from .phonons import PhononCalculator, atoms_to_phonopy, phonopy_to_ase  # type: ignore
except ModuleNotFoundError:
    PhononCalculator = None  # type: ignore
    atoms_to_phonopy = None  # type: ignore
    phonopy_to_ase = None  # type: ignore

from .raman import RamanModeSelector

__all__ = [
    "get_distances_by_bucket",
    "PymatgenPOSCARDefectGenerator",
    "VaspInputs",
    "AimsInputs",
    "PhononCalculator",
    "atoms_to_phonopy",
    "phonopy_to_ase",
    "RamanModeSelector",
]
