"""Dependency-free loading for JSON-compatible YAML benchmark files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union


PathLike = Union[str, Path]


def load_config(path: PathLike) -> Dict[str, Any]:
    """Load a JSON-compatible YAML file, with optional PyYAML fallback."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError("benchmark config does not exist: %s" % source)
    text = source.read_text(encoding="utf-8")
    try:
        values = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ImportError as import_error:
            raise ValueError(
                "%s is not JSON-compatible YAML; install PyYAML to parse "
                "extended YAML syntax" % source
            ) from import_error
        values = yaml.safe_load(text)
        if values is None:
            values = {}
    if not isinstance(values, dict):
        raise ValueError("benchmark config root must be a mapping: %s" % source)
    return values


def dump_json_config(values: Dict[str, Any], path: PathLike) -> Path:
    """Write canonical JSON, which is also valid YAML."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


__all__ = ["PathLike", "dump_json_config", "load_config"]
