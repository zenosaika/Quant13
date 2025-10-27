from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

_ENV_LOADED = False


def _load_env_file() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
    _ENV_LOADED = True


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    _load_env_file()
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
