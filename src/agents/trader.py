from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from src.models.schemas import TradeLeg, TradeProposal, TradeThesis, VolatilityReport
from src.tools.llm import get_llm_client


class TraderAgent:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt
        self.llm = get_llm_client()

    def propose_trade(
        self,
        thesis: TradeThesis,
        volatility_report: VolatilityReport,
        options_chain: List[Dict[str, Any]],
    ) -> TradeProposal:
        payload = {
            "thesis": thesis.model_dump(),
            "volatility": volatility_report.model_dump(),
            "options_chain": _serialize_options_chain(options_chain),
        }
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]
        raw = self.llm.chat(messages, temperature=0.2)
        data = _safe_json_loads(raw, options_chain)
        return TradeProposal(**data)


def _serialize_options_chain(options_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized = []
    for entry in options_chain:
        serialized.append({
            "expiration": entry.get("expiration"),
            "calls": _serialize_option_side(entry.get("calls")),
            "puts": _serialize_option_side(entry.get("puts")),
        })
    return serialized


def _serialize_option_side(df: Any, limit: int = 10) -> List[Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    allowed_columns = [
        col for col in df.columns if col in {
            "contractSymbol",
            "strike",
            "lastPrice",
            "bid",
            "ask",
            "impliedVolatility",
            "delta",
            "gamma",
            "theta",
            "vega",
            "openInterest",
            "volume",
        }
    ]
    trimmed = df[allowed_columns].head(limit).copy()
    return trimmed.fillna(0.0).to_dict(orient="records")


def _safe_json_loads(raw: str, options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and {"strategy_name", "action", "trade_legs"} <= set(data):
            return data
    except json.JSONDecodeError:
        pass
    fallback_leg = _construct_fallback_leg(options_chain)
    return {
        "strategy_name": "Long Call",
        "action": "BUY_TO_OPEN",
        "quantity": 1,
        "trade_legs": [fallback_leg] if fallback_leg else [],
        "notes": raw.strip(),
    }


def _construct_fallback_leg(options_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
    for entry in options_chain:
        calls = entry.get("calls")
        if isinstance(calls, pd.DataFrame) and not calls.empty:
            row = calls.iloc[0]
            return {
                "contract_symbol": row.get("contractSymbol", ""),
                "type": "CALL",
                "action": "BUY",
                "strike_price": float(row.get("strike", 0.0)),
                "expiration_date": entry.get("expiration", ""),
                "quantity": 1,
                "key_greeks_at_selection": {
                    "delta": float(row.get("delta", 0.0) or 0.0),
                    "gamma": float(row.get("gamma", 0.0) or 0.0),
                    "theta": float(row.get("theta", 0.0) or 0.0),
                    "vega": float(row.get("vega", 0.0) or 0.0),
                    "impliedVolatility": float(row.get("impliedVolatility", 0.0) or 0.0),
                },
            }
    return {}
