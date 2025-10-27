from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.agents.base import Agent
from src.models.schemas import VolatilityReport


class VolatilityModelingAgent(Agent):
    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ohlcv: pd.DataFrame = state["ohlcv"]
        options_chain = state.get("options_chain", [])
        spot_price = float(ohlcv["close"].iloc[-1])

        returns = ohlcv["close"].pct_change().dropna()
        realized_vol = returns.rolling(window=30).std() * math.sqrt(252)
        current_realized_vol = realized_vol.iloc[-1] if not realized_vol.dropna().empty else np.nan

        historical_min = float(realized_vol.min()) if not realized_vol.dropna().empty else current_realized_vol
        historical_max = float(realized_vol.max()) if not realized_vol.dropna().empty else current_realized_vol
        iv_rank = _compute_rank(current_realized_vol, historical_min, historical_max)

        skew_analysis = _compute_skew_analysis(options_chain, spot_price)
        term_structure = _compute_term_structure(options_chain)
        forecast = _summarize_volatility_trend(realized_vol)

        return {
            "iv_rank": iv_rank,
            "volatility_forecast": forecast,
            "skew_analysis": skew_analysis,
            "term_structure": term_structure,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> VolatilityReport:
        return VolatilityReport(
            ticker=state["ticker"],
            iv_rank=analysis["iv_rank"],
            volatility_forecast=analysis["volatility_forecast"],
            skew_analysis=analysis["skew_analysis"],
            term_structure=analysis["term_structure"],
        )


def _compute_rank(current: Optional[float], min_val: Optional[float], max_val: Optional[float]) -> float:
    if current is None or math.isnan(current):
        return 50.0
    if (
        min_val is None
        or max_val is None
        or math.isnan(min_val)
        or math.isnan(max_val)
        or math.isclose(max_val, min_val)
    ):
        return 50.0
    return float(np.clip((current - min_val) / (max_val - min_val), 0.0, 1.0) * 100)


def _compute_skew_analysis(options_chain: Any, spot_price: float) -> str:
    if not options_chain:
        return "Insufficient options data to evaluate skew."
    front_expiry = options_chain[0]
    calls = front_expiry["calls"]
    puts = front_expiry["puts"]
    if calls.empty or puts.empty:
        return "Options chain missing call or put data."
    calls = calls.assign(distance=(calls["strike"] - spot_price).abs())
    puts = puts.assign(distance=(puts["strike"] - spot_price).abs())
    call_row = calls.nsmallest(1, "distance")
    put_row = puts.nsmallest(1, "distance")
    if call_row.empty or put_row.empty:
        return "Unable to locate near-the-money strikes."
    call_iv = float(call_row["impliedVolatility"].iloc[0])
    put_iv = float(put_row["impliedVolatility"].iloc[0])
    skew = put_iv - call_iv
    if abs(skew) < 0.01:
        return "Near-the-money skew is balanced between calls and puts."
    if skew > 0:
        return "Puts imply higher volatility than calls, indicating demand for downside protection."
    return "Calls imply higher volatility than puts, suggesting upside exposure demand."


def _compute_term_structure(options_chain: Any) -> str:
    if len(options_chain) < 2:
        return "Insufficient expirations to evaluate term structure."
    front_ivs = pd.concat(
        [options_chain[0]["calls"]["impliedVolatility"], options_chain[0]["puts"]["impliedVolatility"]]
    ).mean()
    back_ivs = pd.concat(
        [options_chain[1]["calls"]["impliedVolatility"], options_chain[1]["puts"]["impliedVolatility"]]
    ).mean()
    if pd.isna(front_ivs) or pd.isna(back_ivs):
        return "Options IV data incomplete for term structure analysis."
    if front_ivs > back_ivs:
        return "Backwardation: Short-term options imply higher volatility than longer-dated contracts."
    if front_ivs < back_ivs:
        return "Contango: Longer-dated options imply higher volatility than near-term contracts."
    return "Flat term structure between near and far expirations."


def _summarize_volatility_trend(realized_vol: pd.Series) -> str:
    series = realized_vol.dropna()
    if len(series) < 10:
        return "Insufficient data to assess realized volatility trend."
    recent = series.tail(5).mean()
    mid_term = series.tail(20).mean()
    long_term = series.mean()

    if recent > mid_term * 1.1:
        return "Realized volatility is accelerating above its 1-month average."
    if recent < mid_term * 0.9:
        return "Realized volatility is compressing below its 1-month average."
    if recent > long_term * 1.05:
        return "Volatility holding marginally above long-run norms."
    if recent < long_term * 0.95:
        return "Volatility remains subdued versus long-run norms."
    return "Volatility stable relative to recent history."
