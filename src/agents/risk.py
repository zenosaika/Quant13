from __future__ import annotations

from typing import List, Optional

from src.models.schemas import RiskAdjustment, RiskAssessment, TradeProposal, TradeThesis, VolatilityReport


class RiskManagementTeam:
    def __init__(self, high_iv_threshold: float = 50.0, low_iv_threshold: float = 30.0) -> None:
        self.high_iv_threshold = high_iv_threshold
        self.low_iv_threshold = low_iv_threshold

    def assess(self, trade: TradeProposal, thesis: TradeThesis, volatility: VolatilityReport) -> RiskAssessment:
        adjustments: List[RiskAdjustment] = []

        iv_rank = float(volatility.iv_rank if volatility.iv_rank is not None else 0.0)
        conviction = (thesis.conviction_level or "Moderate").strip()
        conviction_lower = conviction.lower()
        stance = _normalize_stance(thesis.winning_argument)
        strategy_name = trade.strategy_name
        strategy_bias = _infer_strategy_bias(strategy_name)

        conflict_message: Optional[str] = None
        alternative: Optional[str] = None

        if strategy_bias == "debit" and iv_rank > self.high_iv_threshold:
            alternative = _suggest_alternative(stance, preferred_bias="credit")
            conflict_message = (
                f"IV Rank is high at {iv_rank:.1f}. Debit structures risk overpaying for premium."
            )
        elif strategy_bias == "credit" and iv_rank < self.low_iv_threshold:
            alternative = _suggest_alternative(stance, preferred_bias="debit")
            conflict_message = (
                f"IV Rank is low at {iv_rank:.1f}. Credit structures may offer limited premium."
            )

        if iv_rank >= self.high_iv_threshold:
            adjustments.append(RiskAdjustment(profile="Safe", recommendation="Throttle position sizing and consider hedging while implied volatility remains elevated."))
        elif iv_rank <= self.low_iv_threshold:
            adjustments.append(RiskAdjustment(profile="Safe", recommendation="Volatility discount present; maintain defined-risk sizing."))
        else:
            adjustments.append(RiskAdjustment(profile="Safe", recommendation="Operate within baseline risk budget; IV Rank is mid-range."))

        if conflict_message:
            conflict_detail = conflict_message
            if alternative:
                conflict_detail += f" Consider {alternative} to realign with volatility conditions."
            adjustments.append(RiskAdjustment(profile="Neutral", recommendation=f"Conflict: {conflict_detail}"))
        else:
            adjustments.append(RiskAdjustment(profile="Neutral", recommendation=f"Ensure {strategy_name} positioning reflects {conviction_lower or 'balanced'} conviction with appropriate Greeks."))

        if conflict_message:
            adjustments.append(RiskAdjustment(profile="Risky", recommendation="Do not scale exposure until the strategy is rebalanced toward the recommended structure."))
        elif conviction_lower in {"high", "elevated", "strong"}:
            adjustments.append(RiskAdjustment(profile="Risky", recommendation="Stagger entries to build size only if liquidity and spreads remain favorable."))
        else:
            adjustments.append(RiskAdjustment(profile="Risky", recommendation="Maintain base exposure; avoid leverage expansion without stronger conviction."))

        if conflict_message:
            final_note = f"CRITICAL: Proceed with caution. Strategy alignment conflict detected. {conflict_message}"
        else:
            final_note = f"Proceed with trade, monitoring IV Rank at {iv_rank:.1f} and respecting the {conviction_lower or 'balanced'} conviction profile."

        return RiskAssessment(adjustments=adjustments, final_recommendation=final_note)


def _normalize_stance(winning_argument: str) -> str:
    text = (winning_argument or "").strip().lower()
    if "bear" in text or "down" in text:
        return "bearish"
    if "bull" in text or "up" in text:
        return "bullish"
    return "neutral"


def _infer_strategy_bias(strategy_name: str) -> Optional[str]:
    if not strategy_name:
        return None
    lowered = strategy_name.lower()
    mapping = {
        "bull call spread": "debit",
        "bear put spread": "debit",
        "long call": "debit",
        "long put": "debit",
        "calendar": "debit",
        "diagonal": "debit",
        "bull put spread": "credit",
        "bear call spread": "credit",
        "iron condor": "credit",
        "short": "credit",
    }
    for key, bias in mapping.items():
        if key in lowered:
            return bias
    if "credit" in lowered:
        return "credit"
    if "debit" in lowered or "long" in lowered:
        return "debit"
    if "spread" in lowered and "put" in lowered and "bull" in lowered:
        return "credit"
    if "spread" in lowered and "call" in lowered and "bear" in lowered:
        return "credit"
    return None


def _suggest_alternative(stance: str, preferred_bias: str) -> str:
    stance = stance or "neutral"
    preferred_bias = preferred_bias.lower()
    if stance == "bullish":
        return "Bull Put Credit Spread" if preferred_bias == "credit" else "Bull Call Debit Spread"
    if stance == "bearish":
        return "Bear Call Credit Spread" if preferred_bias == "credit" else "Bear Put Debit Spread"
    return "Iron Condor" if preferred_bias == "credit" else "Long Straddle"
