from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.agents.base import Agent
from src.models.schemas import RiskAdjustment, RiskAssessment, TradeProposal, TradeThesis, VolatilityReport
from src.tools.llm import get_llm_client


class RiskManagementTeam(Agent):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.llm = get_llm_client()
        self.temperature: float = float(config.get("temperature", 0.15))
        self.high_iv_threshold: float = float(config.get("high_iv_threshold", 50.0))
        self.low_iv_threshold: float = float(config.get("low_iv_threshold", 30.0))

    def assess(
        self,
        trade: TradeProposal,
        thesis: TradeThesis,
        volatility: VolatilityReport,
        current_date: Optional[str] = None,
    ) -> RiskAssessment:
        state = {
            "trade_proposal": trade,
            "trade_thesis": thesis,
            "volatility_report": volatility,
            "current_date": current_date,
        }
        return self.run(state)

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        trade: TradeProposal = state["trade_proposal"]
        thesis: TradeThesis = state["trade_thesis"]
        volatility: VolatilityReport = state["volatility_report"]
        current_date: Optional[str] = state.get("current_date")

        payload = {
            "trade_thesis": thesis.model_dump(),
            "trade_proposal": trade.model_dump(),
            "volatility_report": volatility.model_dump(),
            "config": {
                "high_iv_threshold": self.high_iv_threshold,
                "low_iv_threshold": self.low_iv_threshold,
            },
            "current_date": current_date,
        }

        try:
            raw_response = self.llm.chat(
                [
                    {"role": "system", "content": self.config.get("prompt", "")},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                temperature=self.temperature,
            )
        except Exception:  # noqa: BLE001
            raw_response = ""

        parsed = _parse_risk_response(raw_response)
        if parsed is None:
            parsed = _fallback_assessment(trade, thesis, volatility, self.high_iv_threshold, self.low_iv_threshold)
            parsed["raw"] = raw_response
            return parsed

        if not parsed.get("adjustments"):
            fallback = _fallback_assessment(trade, thesis, volatility, self.high_iv_threshold, self.low_iv_threshold)
            fallback["raw"] = raw_response
            return fallback

        parsed["raw"] = raw_response
        return parsed

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> RiskAssessment:
        adjustments_payload = analysis.get("adjustments") or []
        adjustments: List[RiskAdjustment] = []
        for item in adjustments_payload:
            if isinstance(item, RiskAdjustment):
                adjustments.append(item)
            elif isinstance(item, dict) and item.get("profile") and item.get("recommendation"):
                adjustments.append(RiskAdjustment(profile=str(item["profile"]), recommendation=str(item["recommendation"])))
            elif isinstance(item, str):
                adjustments.append(RiskAdjustment(profile="General", recommendation=item))

        if not adjustments:
            fallback = _fallback_assessment(
                state["trade_proposal"],
                state["trade_thesis"],
                state["volatility_report"],
                self.high_iv_threshold,
                self.low_iv_threshold,
            )
            adjustments = [
                RiskAdjustment(profile=str(adj.get("profile", "General")), recommendation=str(adj.get("recommendation", "Review trade manually.")))
                for adj in fallback.get("adjustments", [])
            ]
            final_rec = fallback.get("final_recommendation", "Manual review required.")
        else:
            final_rec = str(analysis.get("final_recommendation") or "Manual review required.")

        return RiskAssessment(adjustments=adjustments, final_recommendation=final_rec)


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


def _parse_risk_response(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None

    if not isinstance(data, dict):
        return None

    if "risk_assessment" in data and isinstance(data["risk_assessment"], dict):
        data = data["risk_assessment"]

    adjustments_raw = data.get("adjustments")
    final_note = data.get("final_recommendation") or data.get("finalRecommendation")

    adjustments: List[Dict[str, str]] = []
    if isinstance(adjustments_raw, list):
        for entry in adjustments_raw:
            if isinstance(entry, dict):
                profile = entry.get("profile") or entry.get("label")
                recommendation = entry.get("recommendation") or entry.get("advice")
                if profile and recommendation:
                    adjustments.append({"profile": str(profile), "recommendation": str(recommendation)})
            elif isinstance(entry, str):
                adjustments.append({"profile": "General", "recommendation": entry})

    if adjustments and isinstance(final_note, str) and final_note.strip():
        return {"adjustments": adjustments, "final_recommendation": final_note.strip()}

    return None


def _fallback_assessment(
    trade: TradeProposal,
    thesis: TradeThesis,
    volatility: VolatilityReport,
    high_iv_threshold: float,
    low_iv_threshold: float,
) -> Dict[str, Any]:
    adjustments: List[Dict[str, str]] = []

    if trade.generation_status == "failed":
        return {
            "adjustments": [
                {"profile": "Critical", "recommendation": "Trade generation failed. Do not execute; request a new proposal."},
                {"profile": "Process", "recommendation": "Escalate to trader desk for manual construction of required strategy."},
            ],
            "final_recommendation": "Generation failure detected. Halt execution until a complete trade structure is available.",
        }

    iv_rank = float(volatility.iv_rank if volatility.iv_rank is not None else 0.0)
    conviction = (thesis.conviction_level or "Moderate").strip()
    conviction_lower = conviction.lower()
    stance = _normalize_stance(thesis.winning_argument)
    strategy_name = trade.strategy_name
    strategy_bias = _infer_strategy_bias(strategy_name)

    # ========================================================================
    # FIX #3: DIRECTIONAL MISMATCH DETECTION (EXPERT REVIEW FIX)
    # ========================================================================
    # Infer the strategy's directional intent from its name
    def _infer_strategy_direction(name: str) -> str:
        """Infer if strategy is bullish, bearish, or neutral"""
        name_lower = name.lower()
        if "bull" in name_lower or "long call" in name_lower:
            return "bullish"
        if "bear" in name_lower or "long put" in name_lower:
            return "bearish"
        return "neutral"

    strategy_direction = _infer_strategy_direction(strategy_name)

    conflict_message: Optional[str] = None
    alternative: Optional[str] = None

    # CHECK #1: Critical Directional Mismatch (RBLX failure mode)
    # If thesis is HIGH conviction directional, strategy MUST match
    if conviction_lower in ["high", "elevated", "strong"]:
        if stance == "bearish" and strategy_direction in ["bullish", "neutral"]:
            adjustments.append({
                "profile": "Critical",
                "recommendation": (
                    f"CRITICAL CONTRADICTION: Thesis is {stance.upper()} "
                    f"(High conviction) but strategy '{strategy_name}' is "
                    f"{strategy_direction.upper()}. DO NOT EXECUTE. "
                    f"Recommend Bear Put Spread or Bear Call Spread."
                )
            })
            conflict_message = f"Directional mismatch: {stance} thesis vs {strategy_direction} strategy"
            return {
                "adjustments": adjustments,
                "final_recommendation": "CRITICAL: Trade logic failure. Directional mismatch. HALT EXECUTION."
            }

        if stance == "bullish" and strategy_direction in ["bearish", "neutral"]:
            adjustments.append({
                "profile": "Critical",
                "recommendation": (
                    f"CRITICAL CONTRADICTION: Thesis is {stance.upper()} "
                    f"(High conviction) but strategy '{strategy_name}' is "
                    f"{strategy_direction.upper()}. DO NOT EXECUTE. "
                    f"Recommend Bull Call Spread or Bull Put Spread."
                )
            })
            conflict_message = f"Directional mismatch: {stance} thesis vs {strategy_direction} strategy"
            return {
                "adjustments": adjustments,
                "final_recommendation": "CRITICAL: Trade logic failure. Directional mismatch. HALT EXECUTION."
            }

    if strategy_bias == "debit" and iv_rank > high_iv_threshold:
        alternative = _suggest_alternative(stance, preferred_bias="credit")
        conflict_message = f"IV Rank is high at {iv_rank:.1f}. Debit structures risk overpaying for premium."
    elif strategy_bias == "credit" and iv_rank < low_iv_threshold:
        alternative = _suggest_alternative(stance, preferred_bias="debit")
        conflict_message = f"IV Rank is low at {iv_rank:.1f}. Credit structures may offer limited premium."

    if iv_rank >= high_iv_threshold:
        adjustments.append({"profile": "Safe", "recommendation": "Throttle position sizing and consider hedging while implied volatility remains elevated."})
    elif iv_rank <= low_iv_threshold:
        adjustments.append({"profile": "Safe", "recommendation": "Volatility discount present; maintain defined-risk sizing."})
    else:
        adjustments.append({"profile": "Safe", "recommendation": "Operate within baseline risk budget; IV Rank is mid-range."})

    if conflict_message:
        conflict_detail = conflict_message
        if alternative:
            conflict_detail += f" Consider {alternative} to realign with volatility conditions."
        adjustments.append({"profile": "Neutral", "recommendation": f"Conflict: {conflict_detail}"})
    else:
        adjustments.append({"profile": "Neutral", "recommendation": f"Ensure {strategy_name} positioning reflects {conviction_lower or 'balanced'} conviction with appropriate Greeks."})

    if conflict_message:
        adjustments.append({"profile": "Risky", "recommendation": "Do not scale exposure until the strategy is rebalanced toward the recommended structure."})
        final_note = f"CRITICAL: Proceed with caution. Strategy alignment conflict detected. {conflict_message}"
    elif conviction_lower in {"high", "elevated", "strong"}:
        adjustments.append({"profile": "Risky", "recommendation": "Stagger entries to build size only if liquidity and spreads remain favorable."})
        final_note = f"Proceed with trade, monitoring IV Rank at {iv_rank:.1f} and respecting the {conviction_lower} conviction profile."
    else:
        adjustments.append({"profile": "Risky", "recommendation": "Maintain base exposure; avoid leverage expansion without stronger conviction."})
        final_note = f"Proceed with trade, monitoring IV Rank at {iv_rank:.1f} and respecting the {conviction_lower or 'balanced'} conviction profile."

    if trade.max_risk is None and trade.max_reward is None:
        adjustments.append({"profile": "Data", "recommendation": "Risk metrics incomplete. Verify max risk/reward calculations before execution."})
        final_note = "Risk metrics missing; verify deterministic calculations before proceeding."

    return {"adjustments": adjustments, "final_recommendation": final_note}
