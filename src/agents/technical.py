from __future__ import annotations

from typing import Any, Dict

import json

import pandas as pd

from src.agents.base import Agent
from src.models.schemas import TechnicalReport
from src.tools.llm import get_llm_client
from src.utils.indicators import IndicatorConfig, compute_indicator_bundle


class TechnicalAnalyst(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = get_llm_client()

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ohlcv: pd.DataFrame = state["ohlcv"]
        indicator_config = IndicatorConfig(
            sma_periods=self.config.get("sma_periods", [50, 200]),
            ema_period=self.config.get("ema_period", 20),
            macd_short=self.config.get("macd", {}).get("short_window", 12),
            macd_long=self.config.get("macd", {}).get("long_window", 26),
            macd_signal=self.config.get("macd", {}).get("signal_window", 9),
            rsi_window=self.config.get("rsi_window", 14),
            bollinger_window=self.config.get("bollinger", {}).get("window", 20),
            bollinger_std=self.config.get("bollinger", {}).get("std_dev", 2),
            supertrend_period=self.config.get("supertrend", {}).get("period", 7),
            supertrend_multiplier=self.config.get("supertrend", {}).get("multiplier", 3),
            candlestick_lookback=self.config.get("candlestick_lookback", 10),
        )
        indicators = compute_indicator_bundle(ohlcv, indicator_config)

        llm_payload = {
            "ticker": state["ticker"],
            "indicators": indicators,
        }
        messages = [
            {"role": "system", "content": self.config["prompt"]},
            {"role": "user", "content": json.dumps(llm_payload)},
        ]
        llm_response_raw = self.llm.chat(messages, temperature=0.2)
        llm_response = _safe_parse_json(llm_response_raw)

        return {
            "indicators": indicators,
            "llm_report": llm_response,
            "llm_raw": llm_response_raw,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> TechnicalReport:
        return TechnicalReport(
            ticker=state["ticker"],
            indicators=analysis["indicators"],
            llm_report=analysis["llm_report"],
            llm_raw=analysis["llm_raw"],
        )


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {
        "technical_bias": "neutral",
        "summary": raw.strip(),
    }
