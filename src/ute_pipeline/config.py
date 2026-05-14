from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/datasets.json") -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(root: Path, maybe_relative: str | Path) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else root / p

