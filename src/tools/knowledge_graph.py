from __future__ import annotations

from typing import Dict

from src.config import load_config


def knowledge_graph_alerter(ticker: str) -> Dict[str, str]:
    config = load_config()
    return {
        "ticker": ticker,
        **config.get("mock_data", {}).get("knowledge_graph_alerter", {}),
    }


def knowledge_graph_querier(ticker: str) -> Dict[str, str]:
    config = load_config()
    return {
        "ticker": ticker,
        **config.get("mock_data", {}).get("knowledge_graph_querier", {}),
    }
