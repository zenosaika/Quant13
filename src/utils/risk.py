from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from datetime import datetime

import math
import pandas as pd

from src.models.schemas import TradeLeg, TradeProposal
from src.utils.greeks_aggregation import aggregate_position_greeks, assess_greeks_risk


OPTIONS_MULTIPLIER = 100.0


@dataclass
class _PriceReference:
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    strike: Optional[float]
    option_type: str
    expiration: str


def calculate_risk_metrics(
    trade: TradeProposal,
    options_chain: List[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """Compute deterministic risk metrics for a trade proposal (enhanced with Greeks)."""

    if not trade.trade_legs:
        return {"max_risk": None, "max_reward": None, "net_premium": None}

    lookup = _build_price_lookup(options_chain)

    net_premium = 0.0
    for leg in trade.trade_legs:
        price = _resolve_price(leg, lookup)
        direction = 1 if leg.action.upper().startswith("SELL") else -1
        net_premium += direction * price * leg.quantity * OPTIONS_MULTIPLIER

    max_risk, max_reward = _compute_payoff_extremes(trade.trade_legs, net_premium)

    # NEW: Add position Greeks aggregation
    spot_price = _get_spot_price_from_chain(options_chain)
    position_greeks = aggregate_position_greeks(trade.trade_legs, spot_price)

    # NEW: Assess Greeks risk level
    days_to_exp = _get_days_to_expiration(trade.trade_legs)
    greeks_risk = assess_greeks_risk(position_greeks, days_to_exp)

    return {
        "max_risk": max_risk,
        "max_reward": max_reward,
        "net_premium": net_premium,
        # NEW: Position Greeks metrics
        "position_greeks": position_greeks,
        "greeks_risk_level": greeks_risk["risk_level"],
        "greeks_warnings": greeks_risk["warnings"],
    }


def _build_price_lookup(options_chain: Iterable[Dict[str, Any]]) -> Dict[str, _PriceReference]:
    """
    Build price lookup from options chain.

    FIXED: Handles both grouped format (with DataFrames) and flat format (list of dicts).

    Grouped format:
        [{"expiration": "2024-12-20", "calls": DataFrame, "puts": DataFrame}, ...]

    Flat format (from backtesting):
        [{"type": "call", "strike": 100, "expiration": "2024-12-20", "bid": 1.0, ...}, ...]
    """
    lookup: Dict[str, _PriceReference] = {}

    for entry in options_chain:
        # Check if this is grouped format (has "calls" or "puts" DataFrames)
        if "calls" in entry or "puts" in entry:
            # Grouped format: extract from DataFrames
            expiration = str(entry.get("expiration") or "")
            for option_type, frame in (("CALL", entry.get("calls")), ("PUT", entry.get("puts"))):
                if not isinstance(frame, pd.DataFrame) or frame.empty:
                    continue
                for _, row in frame.iterrows():
                    symbol = str(row.get("contractSymbol") or "").strip()
                    if not symbol:
                        continue
                    lookup[symbol] = _PriceReference(
                        bid=_to_float(row.get("bid")),
                        ask=_to_float(row.get("ask")),
                        last=_to_float(row.get("lastPrice")),
                        strike=_to_float(row.get("strike")),
                        option_type=option_type,
                        expiration=expiration,
                    )
        else:
            # Flat format: entry is a single option dict
            symbol = str(entry.get("contractSymbol") or "").strip()
            option_type = str(entry.get("type") or "").upper()
            expiration = str(entry.get("expiration") or "")

            # Even without symbol, add by strike+type+expiration for matching
            if not symbol:
                # Create synthetic symbol for matching
                strike = entry.get("strike", 0)
                symbol = f"SYNTH_{option_type}_{expiration}_{strike}"

            lookup[symbol] = _PriceReference(
                bid=_to_float(entry.get("bid")),
                ask=_to_float(entry.get("ask")),
                last=_to_float(entry.get("lastPrice")),
                strike=_to_float(entry.get("strike")),
                option_type=option_type,
                expiration=expiration,
            )

    return lookup


def _resolve_price(leg: TradeLeg, lookup: Dict[str, _PriceReference]) -> float:
    info = lookup.get(leg.contract_symbol)
    if info is None:
        info = _match_leg_by_attributes(leg, lookup.values())
    return _mid_price(info) if info else 0.0


def _match_leg_by_attributes(leg: TradeLeg, candidates: Iterable[_PriceReference]) -> Optional[_PriceReference]:
    for candidate in candidates:
        if candidate.option_type != leg.type.upper():
            continue
        if candidate.expiration and leg.expiration_date and candidate.expiration != leg.expiration_date:
            continue
        if candidate.strike is not None and not math.isclose(candidate.strike, leg.strike_price or 0.0, rel_tol=1e-4):
            continue
        return candidate
    return None


def _mid_price(info: _PriceReference) -> float:
    bid = info.bid if info.bid is not None else 0.0
    ask = info.ask if info.ask is not None else 0.0
    if bid > 0 and ask > 0:
        return (bid + ask) / 2
    if info.last is not None and info.last > 0:
        return float(info.last)
    return max(bid, ask)


def _compute_payoff_extremes(legs: List[TradeLeg], net_premium: float) -> Tuple[Optional[float], Optional[float]]:
    if len(legs) == 1:
        leg = legs[0]
        debit = abs(net_premium) if net_premium < 0 else 0.0
        if leg.action.upper().startswith("BUY"):
            max_risk = debit if debit > 0 else _fallback_cost(leg)
            if leg.type.upper() == "CALL":
                return max_risk, None
            potential = leg.strike_price * OPTIONS_MULTIPLIER * leg.quantity - max_risk
            return max_risk, max(potential, 0.0)
        # Short single leg
        reward = net_premium if net_premium > 0 else _fallback_cost(leg)
        return None, reward

    normalized = _normalize_leg_descriptors(legs)
    if _is_vertical_spread(normalized):
        return _vertical_payoffs(normalized, net_premium)
    if _is_iron_condor(normalized):
        return _iron_condor_payoffs(normalized, net_premium)

    # Default: defined risk equals net debit if debit, reward equals net credit if credit
    if net_premium < 0:
        return abs(net_premium), None
    if net_premium > 0:
        return None, net_premium
    return None, None


def _normalize_leg_descriptors(legs: List[TradeLeg]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for leg in legs:
        normalized.append({
            "type": leg.type.upper(),
            "action": leg.action.upper(),
            "strike": leg.strike_price,
            "expiration": leg.expiration_date,
            "quantity": leg.quantity,
        })
    return normalized


def _is_vertical_spread(legs: List[Dict[str, Any]]) -> bool:
    if len(legs) != 2:
        return False
    types = {leg["type"] for leg in legs}
    expirations = {leg["expiration"] for leg in legs}
    return len(types) == 1 and len(expirations) == 1


def _vertical_payoffs(legs: List[Dict[str, Any]], net_premium: float) -> Tuple[Optional[float], Optional[float]]:
    long_leg = next((leg for leg in legs if leg["action"].startswith("BUY")), None)
    short_leg = next((leg for leg in legs if leg["action"].startswith("SELL")), None)
    if not long_leg or not short_leg:
        return None, None

    width = abs(short_leg["strike"] - long_leg["strike"]) * OPTIONS_MULTIPLIER
    contracts = min(abs(long_leg["quantity"]), abs(short_leg["quantity"]))
    width *= contracts

    if net_premium >= 0:
        max_reward = net_premium
        max_risk = max(width - net_premium, 0.0)
        return max_risk, max_reward

    debit = abs(net_premium)
    max_reward = max(width - debit, 0.0)
    return debit, max_reward


def _is_iron_condor(legs: List[Dict[str, Any]]) -> bool:
    if len(legs) != 4:
        return False
    types = [leg["type"] for leg in legs]
    expirations = {leg["expiration"] for leg in legs}
    return types.count("CALL") == 2 and types.count("PUT") == 2 and len(expirations) == 1


def _iron_condor_payoffs(legs: List[Dict[str, Any]], net_premium: float) -> Tuple[Optional[float], Optional[float]]:
    calls = [leg for leg in legs if leg["type"] == "CALL"]
    puts = [leg for leg in legs if leg["type"] == "PUT"]

    def _width(spread_legs: List[Dict[str, Any]]) -> float:
        long_leg = next((leg for leg in spread_legs if leg["action"].startswith("BUY")), None)
        short_leg = next((leg for leg in spread_legs if leg["action"].startswith("SELL")), None)
        if not long_leg or not short_leg:
            return 0.0
        contracts = min(abs(long_leg["quantity"]), abs(short_leg["quantity"]))
        return abs(short_leg["strike"] - long_leg["strike"]) * OPTIONS_MULTIPLIER * contracts

    max_width = max(_width(calls), _width(puts))
    if net_premium >= 0:
        max_reward = net_premium
        max_risk = max(max_width - net_premium, 0.0)
        return max_risk, max_reward

    debit = abs(net_premium)
    max_reward = max(max_width - debit, 0.0)
    return debit, max_reward


def _fallback_cost(leg: TradeLeg) -> float:
    if leg.key_greeks_at_selection and isinstance(leg.key_greeks_at_selection, dict):
        iv = leg.key_greeks_at_selection.get("impliedVolatility")
        if isinstance(iv, (int, float)) and iv > 0:
            return float(iv) * OPTIONS_MULTIPLIER * leg.quantity
    return 0.0


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _get_spot_price_from_chain(options_chain: List[Dict[str, Any]]) -> float:
    """Extract spot price from options chain"""
    for entry in options_chain:
        # Try to get underlying price from first entry
        underlying = entry.get("underlying_price")
        if underlying is not None:
            return float(underlying)

        # Fallback: Try to get from calls dataframe
        calls = entry.get("calls")
        if isinstance(calls, pd.DataFrame) and not calls.empty:
            # Try different column names
            for col in ["underlyingPrice", "underlying_price", "lastPrice"]:
                if col in calls.columns:
                    val = calls[col].iloc[0]
                    if val is not None and val > 0:
                        return float(val)

    # Ultimate fallback: use first strike price as proxy
    for entry in options_chain:
        calls = entry.get("calls")
        if isinstance(calls, pd.DataFrame) and not calls.empty:
            return float(calls["strike"].iloc[0])

    return 100.0  # Default fallback


def _get_days_to_expiration(trade_legs: List[TradeLeg]) -> int:
    """Calculate days to expiration for the trade"""
    if not trade_legs:
        return 30  # Default

    # Get the earliest expiration
    min_dte = 999

    for leg in trade_legs:
        if leg.expiration_date:
            try:
                # Parse expiration date
                exp_date = datetime.fromisoformat(leg.expiration_date.replace('Z', '+00:00'))
                now = datetime.now(exp_date.tzinfo) if exp_date.tzinfo else datetime.now()
                dte = (exp_date - now).days
                min_dte = min(min_dte, dte)
            except:
                pass

    return min_dte if min_dte < 999 else 30
