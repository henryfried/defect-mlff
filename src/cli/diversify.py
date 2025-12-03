"""
Diversification CLI (sampling defect configurations).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from defect_mlff.diversification.config_sampler import DefectConfigSampler
from cli.common import load_mapping, resolve_path


def cmd_sample_configs(args: argparse.Namespace) -> None:
    config_path = Path(args.config).resolve()
    base_dir = config_path.parent
    cfg = load_mapping(config_path)

    primitive_path = resolve_path(cfg["primitive_path"], base_dir)
    output_dir = args.output_dir or cfg.get("output_dir")
    output_dir = resolve_path(output_dir, base_dir) if output_dir else None

    sampler = DefectConfigSampler(
        primitive_path=str(primitive_path),
        supercell=cfg.get("supercell", [1, 1, 1]),
        defect_types=cfg["defect_types"],
        defect_species=cfg["defect_species"],
        n_defects=cfg["n_defects"],
    )
    all_configs = sampler.sample_configs(
        n_configs=cfg.get("n_configs", 10),
        require_unique=cfg.get("require_unique", True),
        descriptor_model=cfg.get("descriptor_model"),
        layer=cfg.get("layer"),
    )
    if output_dir is None:
        raise SystemExit("Provide output_dir in the config or via --output-dir")
    sampler.write_configs(str(output_dir), all_configs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="defect-mlff diversification CLI")
    parser.add_argument("--configs", dest="configs_alias", help="Path to sampling config (alias for sample-configs)")
    parser.add_argument(
        "--sample-output-dir",
        dest="sample_output_dir",
        help="Override output dir when using --configs alias",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    p_sample = sub.add_parser("sample-configs", help="Sample defect configurations and write JSON")
    p_sample.add_argument("--config", required=True, help="Path to JSON/YAML config describing sampling inputs")
    p_sample.add_argument("--output-dir", help="Override output directory (else taken from config)")
    p_sample.set_defaults(func=cmd_sample_configs)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Alias: defect-mlff-diversify --configs <file> [--sample-output-dir <dir>]
    if getattr(args, "configs_alias", None):
        args.config = args.configs_alias
        args.output_dir = getattr(args, "sample_output_dir", None)
        return cmd_sample_configs(args)

    if args.command is None:
        parser.print_help()
        raise SystemExit(1)

    args.func(args)


if __name__ == "__main__":
    main()
