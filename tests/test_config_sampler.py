import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.aims.inputs import AimsGeometryIn

from defect_mlff.diversification.config_sampler import DefectConfigSampler


def test_defect_config_sampler_generates_and_writes(tmp_path):
    lattice = Lattice.cubic(3.0)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    poscar_path = tmp_path / "POSCAR"
    structure.to(fmt="poscar", filename=str(poscar_path))

    sampler = DefectConfigSampler(
        primitive_path=str(poscar_path),
        supercell=[1, 1, 1],
        defect_types=["V"],
        defect_species=[["Si"]],
        n_defects=[1],
    )

    configs = sampler.sample_configs(n_configs=2, require_unique=False)
    assert len(configs) == 2
    for cfg in configs:
        assert "structure" in cfg
        assert "buckets" in cfg

    output_dir = tmp_path / "jsons"
    sampler.write_configs(str(output_dir), configs)
    json_files = list(output_dir.glob("*.json"))
    assert len(json_files) == 1
    assert json_files[0].exists()


def test_defect_config_sampler_substitution_and_antisite(tmp_path):
    lattice = Lattice.cubic(3.5)
    structure = Structure(
        lattice,
        ["Si", "Ge", "Si", "Ge"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]],
    )
    poscar_path = tmp_path / "POSCAR_multi"
    structure.to(fmt="poscar", filename=str(poscar_path))

    sampler = DefectConfigSampler(
        primitive_path=str(poscar_path),
        supercell=[1, 1, 1],
        defect_types=["Sub", "As"],
        defect_species=[["Si", "Ge"], ["Ge", "Si"]],
        n_defects=[1, 1],
    )

    configs = sampler.sample_configs(n_configs=1, require_unique=False)
    assert len(configs) == 1
    cfg = configs[0]
    assert "structure" in cfg
    struct = Structure.from_dict(cfg["structure"])
    # After Sub + As, composition should still have 4 sites
    assert struct.num_sites == 4


def test_defect_config_sampler_supercell_expands_lattice(tmp_path):
    lattice = Lattice.cubic(3.0)
    structure = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    poscar_path = tmp_path / "POSCAR_super"
    structure.to(fmt="poscar", filename=str(poscar_path))

    sampler = DefectConfigSampler(
        primitive_path=str(poscar_path),
        supercell=[2, 1, 1],
        defect_types=["V"],
        defect_species=[["Si"]],
        n_defects=[1],
    )

    configs = sampler.sample_configs(n_configs=1, require_unique=False)
    out_struct = Structure.from_dict(configs[0]["structure"])
    assert out_struct.lattice.a == pytest.approx(structure.lattice.a * 2)


def test_defect_config_sampler_reads_aims_geometry(tmp_path):
    lattice = Lattice.cubic(3.2)
    structure = Structure(lattice, ["Ge", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    geom_dir = tmp_path / "aims"
    geom_dir.mkdir()
    AimsGeometryIn.from_structure(structure).write_file(directory=str(geom_dir))
    geom_path = geom_dir / "geometry.in"

    sampler = DefectConfigSampler(
        primitive_path=str(geom_path),
        supercell=[1, 1, 1],
        defect_types=["Sub"],
        defect_species=[["Ge", "Si"]],
        n_defects=[1],
    )

    configs = sampler.sample_configs(n_configs=1, require_unique=False)
    assert len(configs) == 1
    cfg_struct = Structure.from_dict(configs[0]["structure"])
    assert cfg_struct.composition.num_atoms == 2
