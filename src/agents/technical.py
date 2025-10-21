from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from src.agents.base import Agent
from src.models.schemas import TechnicalReport
from src.tools.dl_oracle import mock_dl_oracle
from src.utils.indicators import ema, macd, rsi


class TechnicalAnalyst(Agent):
    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ohlcv: pd.DataFrame = state["ohlcv"]
        prices = ohlcv["close"]
        config = self.config
        moving_averages = {f"ma_{period}": ema(prices, period) for period in config.get("moving_averages", [20, 50, 200])}
        latest_ma = {k: float(v.iloc[-1]) for k, v in moving_averages.items() if not v.dropna().empty}

        rsi_series = rsi(prices, window=config.get("rsi_window", 14))
        macd_df = macd(
            prices,
            short_window=config.get("macd", {}).get("short_window", 12),
            long_window=config.get("macd", {}).get("long_window", 26),
            signal_window=config.get("macd", {}).get("signal_window", 9),
        )

        trend_summary = _build_trend_summary(prices.iloc[-1], latest_ma, rsi_series.iloc[-1], macd_df.iloc[-1])
        key_levels = _extract_key_levels(prices)
        oracle_forecast = mock_dl_oracle(state["ticker"])

        return {
            "classical_summary": trend_summary,
            "key_levels": key_levels,
            "dl_oracle_forecast": oracle_forecast,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> TechnicalReport:
        return TechnicalReport(
            ticker=state["ticker"],
            classical_summary=analysis["classical_summary"],
            key_levels=analysis["key_levels"],
            dl_oracle_forecast=analysis["dl_oracle_forecast"],
        )


def _build_trend_summary(price: float, ma: Dict[str, float], rsi_value: float, macd_row: pd.Series) -> str:
    above_ma = [name for name, value in ma.items() if price > value]
    below_ma = [name for name, value in ma.items() if price <= value]
    parts = []
    if above_ma:
        parts.append(f"Price above {', '.join(above_ma)}")
    if below_ma:
        parts.append(f"Price below {', '.join(below_ma)}")
    if pd.notna(rsi_value):
        if rsi_value > 70:
            parts.append("RSI indicates overbought momentum")
        elif rsi_value < 30:
            parts.append("RSI indicates oversold conditions")
        else:
            parts.append("RSI neutral")
    if pd.notna(macd_row["macd"]) and pd.notna(macd_row["signal"]):
        if macd_row["macd"] > macd_row["signal"]:
            parts.append("MACD bullish crossover persists")
        else:
            parts.append("MACD below signal, momentum cooling")
    return "; ".join(parts)


def _extract_key_levels(prices: pd.Series) -> Dict[str, float]:
    recent = prices.tail(30)
    return {
        "support": float(recent.min()),
        "resistance": float(recent.max()),
        "last_close": float(prices.iloc[-1]),
    }
