from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.models.schemas import TradeLeg, TradeProposal, TradeThesis, VolatilityReport
from src.tools.llm import get_llm_client


class TraderAgent:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.prompt = config.get("prompt", "")
        self.llm = get_llm_client()

    def propose_trade(
        self,
        thesis: TradeThesis,
        volatility_report: VolatilityReport,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
    ) -> TradeProposal:
        payload = {
            "thesis": thesis.model_dump(),
            "volatility": volatility_report.model_dump(),
            "underlying_price": spot_price,
            "options_chain": _serialize_options_chain(options_chain, spot_price),
        }
        strategy_guidance = self._build_strategy_guidance(volatility_report)
        system_prompt = f"{self.prompt.rstrip()}" + strategy_guidance

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ]
        raw = self.llm.chat(messages, temperature=0.2)
        data = _safe_json_loads(raw)
        data["conviction_level"] = None
        expected_legs = _expected_leg_count(data.get("strategy_name"))
        actual_legs = len(data.get("trade_legs", []) or [])

        if data.get("generation_status") != "failed" and expected_legs is not None and actual_legs != expected_legs:
            note_reason = (
                f"Expected {expected_legs} leg(s) for strategy '{data.get('strategy_name')}', "
                f"but received {actual_legs or 0}."
            )
            data = _mark_generation_failed(data, raw, note_reason)

        return TradeProposal(**data)

    def _build_strategy_guidance(self, volatility_report: VolatilityReport) -> str:
        prefs = self.config.get("strategy_preferences") if isinstance(self.config, dict) else None
        if not isinstance(prefs, dict):
            return ""
        iv_rank = volatility_report.iv_rank
        if iv_rank is None:
            return ""
        high_threshold = prefs.get("high_iv_rank_threshold")
        low_threshold = prefs.get("low_iv_rank_threshold")
        guidance: Optional[str] = None

        if isinstance(high_threshold, (int, float)) and iv_rank >= float(high_threshold):
            bias = prefs.get("high_iv_bias")
            guidance = self._format_bias_guidance(iv_rank, bias, regime="high")
        elif isinstance(low_threshold, (int, float)) and iv_rank <= float(low_threshold):
            bias = prefs.get("low_iv_bias")
            guidance = self._format_bias_guidance(iv_rank, bias, regime="low")

        if not guidance:
            return ""
        return f"\n\n{guidance}"

    def _format_bias_guidance(self, iv_rank: float, bias: Optional[str], regime: str) -> Optional[str]:
        if not isinstance(bias, str) or not bias:
            return None
        bias_lower = bias.lower()
        if bias_lower == "credit":
            requirement = "credit spreads or premium-selling structures that collect net credit."
        elif bias_lower == "debit":
            requirement = "debit spreads or long optionality structures that pay net debit."
        else:
            requirement = f"strategies aligned with a {bias_lower} bias."

        descriptor = "elevated" if regime == "high" else "suppressed"
        return (
            f"STRATEGY BIAS: IV Rank is {descriptor} at {iv_rank:.1f}. "
            f"You MUST prioritize {requirement} Deviations require explicit justification."
        )


def _serialize_options_chain(options_chain: List[Dict[str, Any]], spot_price: float) -> List[Dict[str, Any]]:
    serialized = []
    for entry in options_chain:
        serialized.append({
            "expiration": entry.get("expiration"),
            "calls": _serialize_option_side(entry.get("calls"), spot_price),
            "puts": _serialize_option_side(entry.get("puts"), spot_price),
        })
    return serialized


def _serialize_option_side(df: Any, spot_price: float, limit: int = 12) -> List[Dict[str, Any]]:
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
    working = df.copy()
    working["_distance"] = (working["strike"] - spot_price).abs()
    trimmed = working.sort_values("_distance").head(limit)
    trimmed = trimmed[allowed_columns]
    trimmed = trimmed.where(pd.notna(trimmed), None)
    return trimmed.to_dict(orient="records")


def _safe_json_loads(raw: str) -> Dict[str, Any]:
    parsed, structured = _load_trade_json(raw)
    if parsed:
        proposal = _coerce_trade_proposal(parsed, structured, raw)
        if proposal:
            return proposal

    return _failure_payload(raw, "Unable to parse structured trade proposal from LLM response.")


JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _load_trade_json(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    candidate: Optional[Dict[str, Any]] = None
    structured: Optional[Dict[str, Any]] = None
    if not raw:
        return None, None

    try:
        data = json.loads(raw)
        candidate = data
    except json.JSONDecodeError:
        candidate = None

    if not candidate:
        for match in JSON_BLOCK_RE.finditer(raw):
            block = match.group(1)
            try:
                candidate = json.loads(block)
                break
            except json.JSONDecodeError:
                continue

    if isinstance(candidate, dict) and "trade_proposal" in candidate and isinstance(candidate["trade_proposal"], dict):
        structured = candidate["trade_proposal"]
    elif isinstance(candidate, dict):
        structured = candidate
    else:
        structured = None

    return candidate if isinstance(candidate, dict) else None, structured


def _coerce_trade_proposal(candidate: Dict[str, Any], structured: Optional[Dict[str, Any]], raw: str) -> Optional[Dict[str, Any]]:
    trade_info = structured or candidate
    if not isinstance(trade_info, dict):
        return None

    trade_legs_input = trade_info.get("trade_legs")
    if not isinstance(trade_legs_input, list) or not trade_legs_input:
        return None

    strategy = trade_info.get("strategy") or trade_info.get("strategy_name") or "Options Strategy"
    direction_text = trade_info.get("direction")
    action_raw = trade_info.get("action")
    quantity = trade_info.get("quantity") or trade_info.get("contracts") or 1
    expiration = trade_info.get("expiration_date") or trade_info.get("expiration")
    if not expiration and isinstance(trade_info.get("expiration_selection"), dict):
        expiration = trade_info["expiration_selection"].get("date")
    net_flow = (trade_info.get("net_credit_debit") or "").lower()
    action = None
    if isinstance(action_raw, str):
        action = action_raw.upper().replace(" ", "_")
    if not action:
        action = "SELL_TO_OPEN" if "credit" in net_flow else "BUY_TO_OPEN"

    normalized_legs: List[Dict[str, Any]] = []
    for leg in trade_legs_input:
        if not isinstance(leg, dict):
            continue
        normalized = _normalize_leg(leg, expiration)
        if normalized:
            normalized_legs.append(normalized)

    if not normalized_legs:
        return None

    notes_payload = raw.strip()
    if "```json" not in notes_payload:
        try:
            notes_payload = f"{notes_payload}\n```json\n{json.dumps(trade_info, ensure_ascii=False)}\n```"
        except (TypeError, ValueError):
            pass

    return {
        "strategy_name": strategy,
        "action": action,
        "quantity": int(quantity),
        "trade_legs": normalized_legs,
        "notes": notes_payload,
        "generation_status": "generated",
    }


def _normalize_leg(leg: Dict[str, Any], fallback_expiration: Optional[str]) -> Optional[Dict[str, Any]]:
    contract = leg.get("contract_symbol") or leg.get("contractSymbol")
    strike = leg.get("strike") or leg.get("strike_price")
    if strike is None:
        try:
            strike = float(leg.get("strikePrice"))
        except (TypeError, ValueError):
            strike = None
    if not contract and strike is None:
        return None

    action_raw = leg.get("action") or leg.get("direction") or "BUY"
    action_normalized = "BUY"
    if isinstance(action_raw, str):
        upper = action_raw.upper()
        if "SELL" in upper or "WRITE" in upper:
            action_normalized = "SELL"
        elif "BUY" in upper or "LONG" in upper:
            action_normalized = "BUY"
    action = action_normalized

    option_type = leg.get("option_type") or leg.get("type") or "CALL"
    option_type = option_type.upper()

    leg_expiration = leg.get("expiration") or leg.get("expiration_date") or fallback_expiration or ""
    try:
        quantity = int(leg.get("contracts") or leg.get("quantity") or 1)
    except (TypeError, ValueError):
        quantity = 1

    try:
        strike_price = float(strike) if strike is not None else None
    except (TypeError, ValueError):
        strike_price = None

    key_greeks = leg.get("key_greeks_at_selection")
    if not isinstance(key_greeks, dict):
        key_greeks = {}

    return {
        "contract_symbol": contract or "",
        "type": option_type,
        "action": action,
        "strike_price": strike_price,
        "expiration_date": leg_expiration,
        "quantity": quantity,
        "key_greeks_at_selection": key_greeks,
    }


def _expected_leg_count(strategy_name: Optional[str]) -> Optional[int]:
    if not strategy_name:
        return None
    lowered = strategy_name.strip().lower()
    if not lowered:
        return None

    specific_rules: List[Tuple[Tuple[str, ...], int]] = [
        (("iron condor",), 4),
        (("iron butterfly",), 4),
        (("broken wing butterfly",), 3),
        (("butterfly",), 4),
        (("condor",), 4),
        (("calendar",), 2),
        (("diagonal",), 2),
        (("double diagonal",), 4),
        (("strangle",), 2),
        (("straddle",), 2),
        (("collar",), 2),
        (("covered call",), 2),
        (("covered put",), 2),
        (("ladder",), 3),
        (("ratio",), None),
    ]

    for keywords, legs in specific_rules:
        for keyword in keywords:
            if keyword in lowered:
                return legs

    if "spread" in lowered or "vertical" in lowered:
        return 2
    if lowered.startswith("bull") or lowered.startswith("bear"):
        if "credit" in lowered or "debit" in lowered or "spread" in lowered:
            return 2

    if lowered.startswith("long ") or lowered.startswith("short "):
        return 1

    return None


def _mark_generation_failed(data: Dict[str, Any], raw: str, reason: str) -> Dict[str, Any]:
    base = dict(data)
    base.setdefault("trade_legs", [])
    base.setdefault("notes", "")
    note_parts = [reason]
    if base["notes"]:
        note_parts.append(base["notes"])
    else:
        note_parts.append(raw.strip())

    base["generation_status"] = "failed"
    base["quantity"] = int(base.get("quantity") or 0)
    base["notes"] = "\n\n".join(part for part in note_parts if part)
    return base


def _failure_payload(raw: str, reason: str) -> Dict[str, Any]:
    return {
        "strategy_name": "Unspecified",
        "action": "UNDEFINED",
        "quantity": 0,
        "trade_legs": [],
        "notes": f"{reason}\n\nRaw response:\n{raw.strip()}",
        "generation_status": "failed",
    }
