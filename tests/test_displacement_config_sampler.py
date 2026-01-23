import pytest
from pymatgen.core import Lattice, Structure

from defect_mlff.diversification.config_sampler import DisplacementConfigSampler


def _write_poscar(tmp_path, structure):
    poscar_path = tmp_path / "POSCAR"
    structure.to(fmt="poscar", filename=str(poscar_path))
    return poscar_path


def test_displacement_weights_respect_pbc_and_multiple_defects(tmp_path):
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        ["Si", "Si", "Si", "Si"],
        [
            [0.00, 0.0, 0.0],  # at defect 0
            [0.95, 0.0, 0.0],  # 0.5 A from defect 0 via PBC
            [0.52, 0.0, 0.0],  # 0.2 A from defect 1
            [0.30, 0.0, 0.0],  # outside radius
        ],
    )
    poscar_path = _write_poscar(tmp_path, structure)

    sampler = DisplacementConfigSampler(
        struct_path=str(poscar_path),
        displacement=0.01,
        defect_pos_cart=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        defect_radius=1.0,
        inner_weight=2.0,
        outer_weight=0.1,
        seed=1,
    )

    weights = sampler._compute_site_weights(sampler.structure)
    assert weights.tolist() == [2.0, 2.0, 2.0, 0.1]


def test_displacement_weights_require_radius_when_defects_provided(tmp_path):
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Si"], [[0, 0, 0]])
    poscar_path = _write_poscar(tmp_path, structure)

    sampler = DisplacementConfigSampler(
        struct_path=str(poscar_path),
        displacement=0.01,
        defect_pos_cart=[[0.0, 0.0, 0.0]],
        seed=1,
    )

    with pytest.raises(ValueError, match="defect_radius"):
        sampler._compute_site_weights(sampler.structure)
