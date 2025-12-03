"""
Generation CLI (AIMs/VASP inputs).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from defect_mlff.generation.vasp_gen import VaspInputs
from cli.common import load_mapping, load_config_entry, resolve_path


def cmd_write_aims(args: argparse.Namespace) -> None:
    try:
        from defect_mlff.generation.aims_gen import AimsInputs  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "AIMs support not available: pymatgen build lacks pymatgen.io.aims. "
            "Install a pymatgen wheel with aims support (e.g., conda-forge pymatgen)."
        ) from exc
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    data = load_mapping(config_path)

    if args.indices:
        indices = [int(x) for x in args.indices.split(",") if x.strip()]
    elif args.k is not None:
        indices = list(range(args.k))
    else:
        indices = [args.config_index]

    if isinstance(data, dict):
        data_list: list[dict[str, Any]] = [data]
    else:
        if not isinstance(data, list):
            raise SystemExit("Config must be a list of configs or a single config dict.")
        data_list = data

    for idx in indices:
        try:
            config_entry = data_list[idx]
        except IndexError as exc:
            raise SystemExit(f"config index {idx} out of range (len={len(data_list)})") from exc

        # Resolve a structure from path/json if not already present
        if "structure" in config_entry and isinstance(config_entry["structure"], str):
            config_entry["structure"] = str(resolve_path(config_entry["structure"], base_dir))
        if "structure" not in config_entry:
            # Allow a JSON of sampled configs with per-entry structures
            json_path = config_entry.get("json") or config_entry.get("json_path")
            if not json_path and isinstance(config_entry.get("jsons"), list) and config_entry["jsons"]:
                json_path = config_entry["jsons"][0]
            if json_path:
                json_path = resolve_path(json_path, base_dir)
                if not json_path or not json_path.exists():
                    raise SystemExit(f"Could not find json at {json_path}")
                import json

                payload = json.loads(json_path.read_text())
                idx_json = config_entry.get("json_index", config_entry.get("config_index", 0))
                try:
                    entry = payload[idx_json] if isinstance(payload, list) else payload
                    config_entry["structure"] = entry["structure"]
                except Exception as exc:
                    raise SystemExit(f"Failed to extract structure from {json_path} (index {idx_json})") from exc
            else:
                raise SystemExit("Config entry must contain 'structure' or point to a JSON with structure(s).")

        # Resolve defaults/output_dir from args or config
        defaults_path = args.defaults or config_entry.get("defaults") or config_entry.get("defaults_2020")
        if defaults_path is None:
            raise SystemExit("Provide --defaults or specify 'defaults' in the config.")

        output_dir_cfg = args.output_dir or config_entry.get("output_dir") or config_entry.get("output-dir")
        if output_dir_cfg is None:
            raise SystemExit("Provide --output-dir or specify 'output_dir' in the config.")

        out_base = resolve_path(output_dir_cfg, Path.cwd())
        out_base.mkdir(parents=True, exist_ok=True)

        if len(indices) == 1:
            out_dir = out_base
        else:
            out_dir = out_base / f"config_{idx:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)

        aims = AimsInputs(out_path=out_dir)
        aims.write_from_config(
            config_entry,
            preset=args.preset,
            defaults_2020=str(resolve_path(defaults_path, base_dir) or defaults_path),
            basis=args.basis,
            params=load_mapping(Path(args.params)) if args.params else None,
            apply_defects=args.apply_defects,
            defect_pos_cart=config_entry.get("defect_pos_cart"),
            subdirs_for_multiple=args.subdirs_for_multiple,
        )


def cmd_write_vasp(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    config_entry = load_config_entry(config_path, args.config_index)
    if isinstance(config_entry.get("structure"), str):
        config_entry["structure"] = str(resolve_path(config_entry["structure"], base_dir))

    out_dir = resolve_path(args.output_dir, Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)

    vasp = VaspInputs(in_path=str(resolve_path(args.template_dir, base_dir)), out_path=str(out_dir))
    vasp.write_poscar_from_json(config_entry["structure"])
    vasp.write_kpoints(args.kmesh)
    vasp.write_incar(relax=args.relax, system=args.system)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="defect-mlff generation CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_aims = sub.add_parser("write-aims", help="Write FHI-aims inputs for one or more configuration entries")
    p_aims.add_argument("--config", required=True, help="Path to JSON/YAML config list (from sample-configs) or single config dict")
    p_aims.add_argument("--config-index", type=int, default=0, help="Index of the configuration to use (if list). Ignored if --k or --indices is set.")
    p_aims.add_argument("--k", type=int, help="Write the first k configs from the list")
    p_aims.add_argument("--indices", help="Comma-separated list of config indices to write, e.g. 0,2,5")
    p_aims.add_argument("--output-dir", required=False, help="Directory to write inputs. For multiple configs, subfolders config_<idx> are created. If omitted, falls back to 'output_dir' in the config.")
    p_aims.add_argument("--preset", default="relax", choices=["relax", "relax_cell", "scf"], help="Aims preset")
    p_aims.add_argument("--defaults", required=False, help="Path to FHI-aims defaults directory. If omitted, falls back to 'defaults' in the config.")
    p_aims.add_argument("--basis", default="intermediate", help="Basis set folder under defaults")
    p_aims.add_argument("--params", help="Optional JSON or YAML file with additional control.in parameters")
    p_aims.add_argument("--apply-defects", action="store_true", help="Reapply defect metadata to pristine structures")
    p_aims.add_argument("--subdirs-for-multiple", action="store_true", help="Write multiple structures into numbered subdirectories")
    p_aims.set_defaults(func=cmd_write_aims)

    p_vasp = sub.add_parser("write-vasp", help="Write VASP inputs for one configuration entry")
    p_vasp.add_argument("--config", required=True, help="Path to JSON config list (from sample-configs) or single config dict")
    p_vasp.add_argument("--config-index", type=int, default=0, help="Index of the configuration to use (if list)")
    p_vasp.add_argument("--template-dir", required=True, help="Directory containing INCAR template (and optional jobfile)")
    p_vasp.add_argument("--output-dir", required=True, help="Directory to write inputs")
    p_vasp.add_argument("--kmesh", nargs=3, type=int, default=[1, 1, 1], help="Gamma-centered k-point mesh, e.g. --kmesh 3 3 1")
    p_vasp.add_argument("--relax", default="relax", choices=["relax", "scf", "train"], help="INCAR preset to use")
    p_vasp.add_argument("--system", default="defect", help="SYSTEM tag value")
    p_vasp.set_defaults(func=cmd_write_vasp)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
