from __future__ import annotations

from typing import Dict

from src.config import load_config


def mock_dl_oracle(ticker: str) -> Dict[str, object]:
    config = load_config()
    data = config.get("mock_data", {}).get("dl_oracle", {}).copy()
    data["ticker"] = ticker
    return data
