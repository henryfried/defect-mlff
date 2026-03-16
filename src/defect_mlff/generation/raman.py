"""
Raman activity estimation and mode selection for periodic structures.

Computes finite-difference polarizability derivatives along phonon eigenvectors
and selects Raman-active modes for DFPT submission.
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

THZ_TO_CM1 = 33.3564  # 1 THz = 33.3564 cm^-1


class RamanModeSelector:
    """
    Estimate Raman activity per phonon mode and select modes for DFPT.

    Uses central finite differences of polarizability along each mass-weighted
    eigenvector:  dalpha/dQ ~ (alpha(+delta) - alpha(-delta)) / (2*delta)

    Activity I_i ~ |dalpha/dQ_i|^2

    Parameters
    ----------
    get_polarizability :
        Callable ``(atoms) -> ndarray[6]`` returning the polarizability tensor
        components [xx, xy, xz, yy, yz, zz] for a given structure.
    delta : float
        Finite-difference displacement amplitude in Ang (default 0.01).
    n_acoustic : int
        Number of acoustic modes to skip (default 3).
    """

    def __init__(self, get_polarizability, delta: float = 0.01, n_acoustic: int = 3):
        self.get_polarizability = get_polarizability
        self.delta = delta
        self.n_acoustic = n_acoustic

    def estimate_activities(
        self, atoms: "Atoms", eigvecs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate Raman activity for each optical mode.

        Parameters
        ----------
        atoms : ase.Atoms
            Equilibrium structure.
        eigvecs : ndarray, shape (3*n_atoms, n_modes)
            Mass-weighted eigenvectors from PhononCalculator.

        Returns
        -------
        activities : ndarray, shape (n_optical,)
            Raman activity estimate (sum of |dalpha/dQ|^2) per optical mode.
        optical_indices : ndarray, shape (n_optical,)
            Indices into the full eigvec/freq arrays.
        dalpha : ndarray, shape (n_optical, 6)
            Polarizability derivative per optical mode.
        """
        from tqdm import tqdm

        n_atoms = len(atoms)
        n_modes_total = eigvecs.shape[1]
        optical_indices = np.arange(self.n_acoustic, n_modes_total)

        activities = []
        dalpha_list = []

        for i in tqdm(optical_indices, desc="modes", leave=False):
            mode = eigvecs[:, i].reshape(n_atoms, 3)
            mode_norm = mode / (np.linalg.norm(mode) + 1e-10)

            a_plus = atoms.copy()
            a_plus.positions += self.delta * mode_norm
            alpha_plus = self.get_polarizability(a_plus)

            a_minus = atoms.copy()
            a_minus.positions -= self.delta * mode_norm
            alpha_minus = self.get_polarizability(a_minus)

            dalpha = (alpha_plus - alpha_minus) / (2 * self.delta)
            activities.append(float(np.sum(dalpha**2)))
            dalpha_list.append(dalpha)

        return np.array(activities), optical_indices, np.array(dalpha_list)

    def select(
        self,
        activities: np.ndarray,
        optical_indices: np.ndarray,
        freqs: np.ndarray,
        activity_threshold: float = 0.05,
        min_modes: int = 5,
        max_modes: int = 30,
        freq_min_cm1: float = 50.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Select modes by activity threshold with hard count bounds.

        Includes all modes where activity > activity_threshold * max_activity,
        clipped to [min_modes, max_modes]. Modes below freq_min_cm1 are excluded.

        Parameters
        ----------
        activities : ndarray, shape (n_optical,)
        optical_indices : ndarray, shape (n_optical,)
        freqs : ndarray, shape (n_modes,)
            Full frequency array in THz.
        activity_threshold : float
            Fraction of max activity to use as threshold (default 0.05 = 5%).
        min_modes, max_modes : int
            Hard lower/upper bounds on selected count.
        freq_min_cm1 : float
            Minimum frequency in cm^-1 (default 50).

        Returns
        -------
        selected_optical_local : ndarray
            Indices into the activities/dalpha arrays.
        selected_mode_indices : ndarray
            Indices into the full eigvec/freq arrays.
        """
        freq_min_thz = freq_min_cm1 / THZ_TO_CM1
        optical_freqs = freqs[optical_indices]
        freq_valid = optical_freqs > freq_min_thz

        if not np.any(freq_valid):
            log.warning(
                "No optical modes above freq_min (%.1f cm^-1)", freq_min_cm1
            )
            return np.array([], dtype=int), np.array([], dtype=int)

        valid_local = np.where(freq_valid)[0]
        valid_activities = activities[valid_local]

        max_activity = float(np.max(valid_activities))
        threshold = activity_threshold * max_activity
        n_above = int(np.sum(valid_activities >= threshold))
        n_select = int(np.clip(n_above, min_modes, max_modes))
        n_select = min(n_select, len(valid_local))

        log.debug(
            "activity threshold=%.3e (%.0f%% of max=%.3e), "
            "%d modes above threshold, selecting %d",
            threshold, activity_threshold * 100, max_activity, n_above, n_select,
        )

        ranked = np.argsort(valid_activities)[::-1]
        selected = ranked[:n_select]
        selected_optical_local = valid_local[selected]

        return selected_optical_local, optical_indices[selected_optical_local]

    def save(
        self,
        output_dir: str | Path,
        config_name: str,
        atoms: "Atoms",
        eigvecs: np.ndarray,
        freqs: np.ndarray,
        selected_mode_indices: np.ndarray,
        activities: np.ndarray,
        optical_indices: np.ndarray,
        dalpha: np.ndarray,
    ) -> dict:
        """
        Save selected modes and their metadata to disk.

        Writes:
          ``{output_dir}/{config_name}_selected_eigvecs.npy``
          ``{output_dir}/{config_name}_selected_mode_indices.npy``
          ``{output_dir}/{config_name}_selected_modes.json``

        Returns
        -------
        dict
            Summary with per-mode frequency, activity, and dalpha_norm.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / f"{config_name}_selected_eigvecs.npy",
                eigvecs[:, selected_mode_indices])
        np.save(output_dir / f"{config_name}_selected_mode_indices.npy",
                selected_mode_indices)

        selected_local = [
            int(np.where(optical_indices == i)[0][0])
            for i in selected_mode_indices
        ]

        mode_summary = []
        for rank, (local_idx, mode_idx) in enumerate(
                zip(selected_local, selected_mode_indices)):
            mode_summary.append({
                "rank": rank + 1,
                "mode_index": int(mode_idx),
                "frequency_THz": float(freqs[mode_idx]),
                "frequency_cm1": float(freqs[mode_idx] * THZ_TO_CM1),
                "raman_activity_estimate": float(activities[local_idx]),
                "dalpha_norm": float(np.linalg.norm(dalpha[local_idx])),
            })

        summary = {
            "config": config_name,
            "n_atoms": len(atoms),
            "n_modes_total": len(freqs),
            "n_optical": len(optical_indices),
            "n_selected": len(selected_mode_indices),
            "selected_modes": mode_summary,
        }

        with open(output_dir / f"{config_name}_selected_modes.json", "w") as f:
            json.dump(summary, f, indent=2)

        log.info("%s: selected %d modes:", config_name, len(selected_mode_indices))
        for m in mode_summary:
            log.info(
                "  mode %4d  %7.1f cm^-1  activity=%.3e",
                m["mode_index"], m["frequency_cm1"], m["raman_activity_estimate"],
            )

        return summary
