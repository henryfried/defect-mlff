"""
Shared helpers for CLI modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_mapping(path: Path) -> Any:
    """Load JSON or YAML based on extension."""
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("PyYAML is required to read YAML configs. Install pyyaml.") from exc
        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else base_dir / p


def load_config_entry(config_path: Path, index: int) -> Dict[str, Any]:
    data = load_mapping(config_path)
    if isinstance(data, list):
        try:
            return data[index]
        except IndexError as exc:
            raise SystemExit(f"config index {index} out of range for {config_path}") from exc
    if isinstance(data, dict) and "structure" in data:
        return data
    raise SystemExit(
        "Config must be a JSON/YAML list of configs (as produced by sample-configs) "
        "or a single config dict containing a 'structure' key."
    )
