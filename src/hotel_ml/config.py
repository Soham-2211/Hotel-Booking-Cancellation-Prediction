"""Load config.yaml and resolve paths relative to the project root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: PROJECT_ROOT / v for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg
