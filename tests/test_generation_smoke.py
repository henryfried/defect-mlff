from pathlib import Path

from pymatgen.core import Lattice, Structure

from defect_mlff.generation.vasp_gen import VaspInputs


def test_vasp_inputs_smoke(tmp_path: Path):
    """
    Minimal smoke test: write POSCAR/KPOINTS/INCAR from a tiny structure and template.
    Does not touch POTCAR to avoid dependency on installed pseudopotentials.
    """
    template_dir = tmp_path / "template"
    out_dir = tmp_path / "out"
    template_dir.mkdir()
    out_dir.mkdir()
    (template_dir / "INCAR").write_text("ENCUT = 400\nISMEAR = 0\nSIGMA = 0.05\n")

    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])
    struct_dict = struct.as_dict()

    vasp = VaspInputs(in_path=str(template_dir), out_path=str(out_dir))
    vasp.write_poscar_from_json(struct_dict)
    vasp.write_kpoints(mesh=[1, 1, 1])
    vasp.write_incar(relax="relax", system="Si_test")

    assert (out_dir / "POSCAR").exists()
    assert (out_dir / "KPOINTS").exists()
    assert (out_dir / "INCAR").exists()
