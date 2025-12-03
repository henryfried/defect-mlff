# generation/__init__.py

# ─── Distance utilities ───────────────────────────────────────────────────────
from .distances import get_distances_by_bucket

# ─── Defect generators & VASP I/O ─────────────────────────────────────────────
from .defect_generator import (
    PymatgenPOSCARDefectGenerator,
)
from .vasp_gen import VaspInputs

# AIMS support is optional; avoid import errors when only VASP/diversification is used.
try:
    from .aims_gen import AimsInputs  # type: ignore
except ModuleNotFoundError:
    AimsInputs = None  # type: ignore

__all__ = [
    "get_distances_by_bucket",
    "PymatgenPOSCARDefectGenerator",
    "VaspInputs",
    "AimsInputs",
]
