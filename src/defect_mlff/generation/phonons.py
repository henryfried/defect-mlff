"""
Gamma-point phonon calculation for periodic structures.

Uses Phonopy finite-difference force constants with any ASE-compatible
calculator. Produces mass-weighted eigenvectors suitable for mode-targeted
active learning (e.g. Raman displacement generation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms

log = logging.getLogger(__name__)


# ── ASE ↔ Phonopy conversion helpers ──────────────────────────────────────────

def atoms_to_phonopy(atoms: "Atoms"):
    """Convert ASE Atoms to PhonopyAtoms."""
    from phonopy.structure.atoms import PhonopyAtoms
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell(),
        scaled_positions=atoms.get_scaled_positions(),
    )


def phonopy_to_ase(ph_atoms, pbc: bool = True) -> "Atoms":
    """Convert PhonopyAtoms to ASE Atoms."""
    from ase import Atoms
    # Use property-based API — getter methods are absent in some phonopy versions
    return Atoms(
        symbols=list(ph_atoms.symbols),
        cell=np.array(ph_atoms.cell),
        scaled_positions=np.array(ph_atoms.scaled_positions),
        pbc=pbc,
    )


# ── Main class ─────────────────────────────────────────────────────────────────

class PhononCalculator:
    """
    Gamma-point phonon calculation for periodic structures via Phonopy.

    Computes finite-difference force constants and diagonalises the
    dynamical matrix at q=Γ to obtain mass-weighted eigenvectors.
    Eigenvectors are robust to small force errors and are the correct
    input for mode-targeted displacement generation.

    Parameters
    ----------
    calc :
        Any ASE-compatible calculator (e.g. MACECalculator).
    displacement : float
        Finite-difference displacement amplitude in Å (default 0.01).
    n_acoustic : int
        Number of acoustic modes to skip when reporting optical statistics
        (default 3 for a 3D periodic system).
    """

    def __init__(self, calc, displacement: float = 0.01, n_acoustic: int = 3):
        self.calc = calc
        self.displacement = displacement
        self.n_acoustic = n_acoustic

    def compute(self, atoms: "Atoms") -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Gamma-point phonon frequencies and eigenvectors.

        Parameters
        ----------
        atoms : ase.Atoms
            Equilibrium periodic structure (pbc will be set to True).

        Returns
        -------
        freqs : ndarray, shape (n_modes,)
            Phonon frequencies in THz. Negative values indicate imaginary modes.
        eigvecs : ndarray, shape (3*n_atoms, n_modes)
            Real part of mass-weighted eigenvectors; columns are normal modes.
        """
        try:
            from phonopy import Phonopy
        except ImportError as exc:
            raise ImportError(
                "phonopy is required for PhononCalculator. "
                "Install it with: pip install phonopy"
            ) from exc

        atoms = atoms.copy()
        atoms.pbc = True

        ph = Phonopy(
            atoms_to_phonopy(atoms),
            supercell_matrix=np.eye(3, dtype=int),
            primitive_matrix=np.eye(3),
        )
        ph.generate_displacements(distance=self.displacement)

        forces_list = []
        for sc in ph.supercells_with_displacements:
            if sc is None:
                forces_list.append(None)
                continue
            a = phonopy_to_ase(sc)
            a.calc = self.calc
            forces_list.append(a.get_forces())

        ph.forces = forces_list
        ph.produce_force_constants()

        ph.run_qpoints([[0, 0, 0]], with_eigenvectors=True)
        result = ph.get_qpoints_dict()

        freqs: np.ndarray = result["frequencies"][0]    # (n_modes,) in THz
        eigvecs: np.ndarray = result["eigenvectors"][0] # (3*n_atoms, n_modes), complex

        n_imag = int(np.sum(freqs < -0.1))
        if n_imag > 0:
            log.warning(
                "compute(): %d imaginary mode(s) detected (freq < -0.1 THz)",
                n_imag,
            )

        return freqs, eigvecs.real

    def save(
        self,
        output_dir: str | Path,
        config_name: str,
        freqs: np.ndarray,
        eigvecs: np.ndarray,
    ) -> dict:
        """
        Save phonon data to disk and return a summary dict.

        Writes:
          ``{output_dir}/{config_name}_freqs.npy``
          ``{output_dir}/{config_name}_eigvecs.npy``
          ``{output_dir}/{config_name}_summary.json``

        Parameters
        ----------
        output_dir : path-like
            Destination directory (created if it does not exist).
        config_name : str
            Prefix used for all output filenames.
        freqs : ndarray, shape (n_modes,)
            Frequencies in THz as returned by :meth:`compute`.
        eigvecs : ndarray, shape (3*n_atoms, n_modes)
            Eigenvectors as returned by :meth:`compute`.

        Returns
        -------
        dict
            Summary with keys: config, n_modes, n_optical, n_imaginary,
            and frequency range in THz, cm⁻¹, and eV.
        """
        # Unit conversion factors from THz
        THZ_TO_CM1 = 33.3564   # 1 THz = 33.3564 cm⁻¹
        THZ_TO_EV  = 4.13567e-3  # 1 THz = h·1THz in eV

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / f"{config_name}_freqs.npy", freqs)
        np.save(output_dir / f"{config_name}_eigvecs.npy", eigvecs)

        n_optical = len(freqs) - self.n_acoustic
        n_imaginary = int(np.sum(freqs < -0.1))

        freq_min_thz = float(freqs[self.n_acoustic])
        freq_max_thz = float(freqs[-1])

        summary = {
            "config": config_name,
            "n_modes": len(freqs),
            "n_optical": n_optical,
            "n_imaginary": n_imaginary,
            "freq_min_optical_cm1": freq_min_thz * THZ_TO_CM1,
            "freq_max_cm1":         freq_max_thz * THZ_TO_CM1,
            "freq_min_optical_eV":  freq_min_thz * THZ_TO_EV,
            "freq_max_eV":          freq_max_thz * THZ_TO_EV,
        }
        with open(output_dir / f"{config_name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        log.info(
            "%s: %d optical modes, %d imaginary, "
            "freq range [%.1f, %.1f] cm⁻¹  ([%.3f, %.3f] eV)",
            config_name, n_optical, n_imaginary,
            freq_min_thz * THZ_TO_CM1, freq_max_thz * THZ_TO_CM1,
            freq_min_thz * THZ_TO_EV,  freq_max_thz * THZ_TO_EV,
        )
        if n_imaginary > 0:
            log.warning("%s: %d imaginary mode(s)", config_name, n_imaginary)

        return summary
