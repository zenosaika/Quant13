from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


@lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Dict[str, Any]:
    """Load system configuration from YAML file."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
