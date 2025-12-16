from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar

from defect_mlff.diversification.config_sampler import DefectConfigSampler
from defect_mlff.generation.vasp_gen import VaspInputs

try:
    import pymatgen.io.fleur  # noqa: F401

    HAS_FLEUR = True
except Exception:
    HAS_FLEUR = False


DATA_DIR = Path(__file__).resolve().parent / "data" / "mos2"
FIXTURE_MOS2_AIMS = DATA_DIR / "geometry.in"
FIXTURE_MOS2_POSCAR = DATA_DIR / "POSCAR_mos2"
SUPERCELL = [2, 2, 1]


def _apply_supercell(struct: Structure, supercell):
    if supercell != [1, 1, 1]:
        struct.make_supercell([[supercell[0], 0, 0], [0, supercell[1], 0], [0, 0, supercell[2]]])
    return struct


@pytest.mark.parametrize(
    "structure_path",
    [FIXTURE_MOS2_AIMS, FIXTURE_MOS2_POSCAR],
    ids=["geometry.in", "POSCAR"],
)
def test_mos2_sampler_picks_mo_vacancies(structure_path):
    if not structure_path.exists():
        pytest.skip(f"{structure_path} is missing")
    if structure_path.name == "geometry.in" and not HAS_FLEUR:
        pytest.skip("pymatgen.io.fleur not installed; skipping geometry.in parsing test")

    sampler = DefectConfigSampler(
        primitive_path=str(structure_path),
        supercell=SUPERCELL,
        defect_types=["V"],
        defect_species=[["Mo"]],
        n_defects=[1],
    )

    configs = sampler.sample_configs(n_configs=1, require_unique=False)
    assert configs, "No configurations generated for MoS2 vacancy"

    cfg = configs[0]
    flat_indices = [idx for group in cfg["defect_indices"] for idx in group]

    base = _apply_supercell(Structure.from_file(str(structure_path)), SUPERCELL)
    struct = Structure.from_dict(cfg["structure"])

    assert all(base[i].specie.symbol == "Mo" for i in flat_indices)

    base_counts = base.composition.get_el_amt_dict()
    result_counts = struct.composition.get_el_amt_dict()
    assert base_counts.get("Mo", 0) - result_counts.get("Mo", 0) == len(flat_indices)
    assert base_counts.get("S", 0) == result_counts.get("S", 0)
    assert struct.num_sites == base.num_sites - len(flat_indices)


@pytest.mark.parametrize(
    "structure_path",
    [FIXTURE_MOS2_AIMS, FIXTURE_MOS2_POSCAR],
    ids=["geometry.in", "POSCAR"],
)
def test_mos2_sampler_picks_s_vacancies(structure_path):
    if not structure_path.exists():
        pytest.skip(f"{structure_path} is missing")
    if structure_path.name == "geometry.in" and not HAS_FLEUR:
        pytest.skip("pymatgen.io.fleur not installed; skipping geometry.in parsing test")

    sampler = DefectConfigSampler(
        primitive_path=str(structure_path),
        supercell=SUPERCELL,
        defect_types=["V"],
        defect_species=[["S"]],
        n_defects=[1],
    )
    config = sampler.sample_configs(n_configs=1, require_unique=False)[0]
    struct = Structure.from_dict(config["structure"])

    base = _apply_supercell(Structure.from_file(str(structure_path)), SUPERCELL)
    flat_indices = [idx for group in config["defect_indices"] for idx in group]
    assert all(base[i].specie.symbol == "S" for i in flat_indices)

    base_counts = base.composition.get_el_amt_dict()
    result_counts = struct.composition.get_el_amt_dict()

    assert base_counts.get("S", 0) - result_counts.get("S", 0) == len(flat_indices)
    assert base_counts.get("Mo", 0) == result_counts.get("Mo", 0)
    assert struct.num_sites == base.num_sites - len(flat_indices)


@pytest.mark.parametrize(
    "structure_path",
    [FIXTURE_MOS2_AIMS, FIXTURE_MOS2_POSCAR],
    ids=["geometry.in", "POSCAR"],
)
def test_mos2_substitution_mo_to_s_changes_counts(structure_path):
    if not structure_path.exists():
        pytest.skip(f"{structure_path} is missing")
    if structure_path.name == "geometry.in" and not HAS_FLEUR:
        pytest.skip("pymatgen.io.fleur not installed; skipping geometry.in parsing test")

    sampler = DefectConfigSampler(
        primitive_path=str(structure_path),
        supercell=SUPERCELL,
        defect_types=["Sub"],
        defect_species=[["Mo", "S"]],
        n_defects=[1],
    )
    config = sampler.sample_configs(n_configs=1, require_unique=False)[0]
    struct = Structure.from_dict(config["structure"])

    base = _apply_supercell(Structure.from_file(str(structure_path)), SUPERCELL)
    flat_indices = [idx for group in config["defect_indices"] for idx in group]
    assert all(base[i].specie.symbol == "Mo" for i in flat_indices)

    base_counts = base.composition.get_el_amt_dict()
    result_counts = struct.composition.get_el_amt_dict()

    assert result_counts.get("Mo", 0) == base_counts.get("Mo", 0) - len(flat_indices)
    assert result_counts.get("S", 0) == base_counts.get("S", 0) + len(flat_indices)


@pytest.mark.skip(reason="Interstitial defect generation not implemented in DefectConfigSampler")
def test_interstitials_placeholder():
    pass


@pytest.mark.skipif(not FIXTURE_MOS2_POSCAR.exists(), reason="MoS2 POSCAR fixture is missing")
def test_vasp_inputs_generated_from_config(tmp_path):
    sampler = DefectConfigSampler(
        primitive_path=str(FIXTURE_MOS2_POSCAR),
        supercell=SUPERCELL,
        defect_types=["V"],
        defect_species=[["Mo"]],
        n_defects=[1],
    )
    config = sampler.sample_configs(n_configs=1, require_unique=False)[0]

    template_dir = tmp_path / "template"
    out_dir = tmp_path / "out"
    template_dir.mkdir()
    out_dir.mkdir()
    (template_dir / "INCAR").write_text("ENCUT = 400\nISMEAR = 0\nSIGMA = 0.05\n")

    vasp = VaspInputs(in_path=str(template_dir), out_path=str(out_dir))
    vasp.write_poscar_from_json(config["structure"])
    vasp.write_kpoints(np.array([1, 1, 1]))
    vasp.write_incar(relax="relax", system="MoS2_test")

    poscar_path = out_dir / "POSCAR"
    kpoints_path = out_dir / "KPOINTS"
    incar_path = out_dir / "INCAR"
    assert poscar_path.exists()
    assert kpoints_path.exists()
    assert incar_path.exists()

    incar = Incar.from_file(str(incar_path))
    assert str(incar["SYSTEM"]).lower() == "mos2_test"

    poscar_struct = Structure.from_file(str(poscar_path))
    cfg_struct = Structure.from_dict(config["structure"])
    assert poscar_struct.composition.almost_equals(cfg_struct.composition)
