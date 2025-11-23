"""
Thesis-Strategy Alignment Validator

Enforces that trading strategies match research theses.
Prevents critical failures like proposing bullish trades on bearish theses.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

from src.models.schemas import TradeThesis, TradeProposal

logger = logging.getLogger(__name__)


class ThesisMismatchError(Exception):
    """Raised when trade proposal contradicts thesis"""

    def __init__(self, message: str, thesis_direction: str, strategy_bias: str, conviction: str):
        self.thesis_direction = thesis_direction
        self.strategy_bias = strategy_bias
        self.conviction = conviction
        super().__init__(message)


class ThesisValidator:
    """
    Validates alignment between trade thesis and proposed strategy

    Rules:
    - High conviction directional thesis MUST use matching directional strategy
    - Medium conviction directional thesis SHOULD use matching directional strategy
    - Low conviction can use neutral strategies
    - Neutral thesis should use neutral strategies
    """

    # Strategy bias mapping (from strategy_library.py)
    BEARISH_STRATEGIES = {
        "bear_put_spread", "bear_call_spread", "long_put",
        "short_call", "bear_put_debit_spread"
    }

    BULLISH_STRATEGIES = {
        "bull_call_spread", "bull_put_spread", "long_call",
        "short_put", "bull_call_debit_spread"
    }

    NEUTRAL_STRATEGIES = {
        "iron_condor", "iron_butterfly", "short_straddle",
        "short_strangle", "long_straddle", "long_strangle",
        "butterfly_spread", "calendar_spread", "diagonal_spread"
    }

    def __init__(self, strict_mode: bool = True):
        """
        Args:
            strict_mode: If True, enforce strict alignment rules
                        If False, only log warnings
        """
        self.strict_mode = strict_mode

    def validate(self, thesis: TradeThesis, proposal: TradeProposal) -> Dict[str, Any]:
        """
        Validate thesis-strategy alignment

        Args:
            thesis: Trade thesis from debate
            proposal: Proposed trade

        Returns:
            Validation result dict with:
                - valid: bool
                - severity: "pass" | "warning" | "critical"
                - message: str
                - suggestions: List[str]

        Raises:
            ThesisMismatchError: If strict_mode=True and critical mismatch detected
        """
        # Parse inputs
        thesis_direction = self._parse_direction(thesis.winning_argument)
        conviction = self._parse_conviction(thesis.conviction_level)
        strategy_bias = self._get_strategy_bias(proposal.strategy_name)

        logger.info(
            f"Validating: thesis={thesis_direction}/{conviction}, "
            f"strategy={proposal.strategy_name} ({strategy_bias})"
        )

        # Check alignment
        result = self._check_alignment(
            thesis_direction, conviction, strategy_bias, proposal.strategy_name
        )

        # Enforce in strict mode
        if self.strict_mode and result["severity"] == "critical":
            raise ThesisMismatchError(
                result["message"],
                thesis_direction=thesis_direction,
                strategy_bias=strategy_bias,
                conviction=conviction
            )

        return result

    def _check_alignment(
        self,
        thesis_direction: str,
        conviction: str,
        strategy_bias: str,
        strategy_name: str
    ) -> Dict[str, Any]:
        """Check if strategy aligns with thesis"""

        # RULE 1: High conviction directional thesis MUST match
        if conviction == "high":
            if thesis_direction == "bearish" and strategy_bias != "bearish":
                return {
                    "valid": False,
                    "severity": "critical",
                    "message": (
                        f"CRITICAL MISMATCH: Thesis is BEARISH with HIGH conviction, "
                        f"but strategy '{strategy_name}' has {strategy_bias.upper()} bias. "
                        f"This directly contradicts the research conclusion."
                    ),
                    "suggestions": [
                        "Use Bear Put Spread (debit) for limited risk",
                        "Use Bear Call Spread (credit) if IV is high",
                        "Use Long Put for maximum downside exposure"
                    ]
                }

            if thesis_direction == "bullish" and strategy_bias != "bullish":
                return {
                    "valid": False,
                    "severity": "critical",
                    "message": (
                        f"CRITICAL MISMATCH: Thesis is BULLISH with HIGH conviction, "
                        f"but strategy '{strategy_name}' has {strategy_bias.upper()} bias. "
                        f"This directly contradicts the research conclusion."
                    ),
                    "suggestions": [
                        "Use Bull Call Spread (debit) for limited risk",
                        "Use Bull Put Spread (credit) if IV is high",
                        "Use Long Call for maximum upside exposure"
                    ]
                }

        # RULE 2: Medium conviction directional thesis SHOULD match
        if conviction == "medium":
            if thesis_direction == "bearish" and strategy_bias == "bullish":
                return {
                    "valid": False,
                    "severity": "critical",
                    "message": (
                        f"CRITICAL MISMATCH: Thesis is BEARISH (medium conviction), "
                        f"but strategy '{strategy_name}' is BULLISH. "
                        f"Strategies should not contradict thesis direction."
                    ),
                    "suggestions": [
                        "Use bearish strategy (Bear Put/Call Spread)",
                        "If neutral preferred, use Iron Condor with bearish skew"
                    ]
                }

            if thesis_direction == "bullish" and strategy_bias == "bearish":
                return {
                    "valid": False,
                    "severity": "critical",
                    "message": (
                        f"CRITICAL MISMATCH: Thesis is BULLISH (medium conviction), "
                        f"but strategy '{strategy_name}' is BEARISH. "
                        f"Strategies should not contradict thesis direction."
                    ),
                    "suggestions": [
                        "Use bullish strategy (Bull Call/Put Spread)",
                        "If neutral preferred, use Iron Condor with bullish skew"
                    ]
                }

            if thesis_direction in ["bearish", "bullish"] and strategy_bias == "neutral":
                return {
                    "valid": True,
                    "severity": "warning",
                    "message": (
                        f"WARNING: Thesis is {thesis_direction.upper()} (medium conviction), "
                        f"but strategy '{strategy_name}' is NEUTRAL. "
                        f"Consider directional strategy for better alignment."
                    ),
                    "suggestions": [
                        f"Use {thesis_direction} vertical spread for directional bias",
                        "Current neutral strategy may underperform if thesis is correct"
                    ]
                }

        # RULE 3: Neutral thesis should avoid strong directional bets
        if thesis_direction == "neutral":
            if strategy_bias != "neutral" and conviction in ["high", "medium"]:
                return {
                    "valid": True,
                    "severity": "warning",
                    "message": (
                        f"WARNING: Thesis is NEUTRAL, but strategy '{strategy_name}' "
                        f"has {strategy_bias.upper()} bias. Risk if market ranges."
                    ),
                    "suggestions": [
                        "Use Iron Condor for neutral outlook",
                        "Use Iron Butterfly for tighter range assumption"
                    ]
                }

        # PASS: Alignment is good
        return {
            "valid": True,
            "severity": "pass",
            "message": (
                f"✓ Alignment validated: {thesis_direction.upper()} thesis "
                f"matches {strategy_bias.upper()} strategy bias"
            ),
            "suggestions": []
        }

    def _parse_direction(self, winning_argument: str) -> str:
        """Extract direction from thesis"""
        if not winning_argument:
            return "neutral"

        arg_lower = winning_argument.lower()

        if "bull" in arg_lower:
            return "bullish"
        elif "bear" in arg_lower:
            return "bearish"
        else:
            return "neutral"

    def _parse_conviction(self, conviction_level: str) -> str:
        """Normalize conviction level"""
        if not conviction_level:
            return "low"

        conv_lower = conviction_level.lower()

        if "high" in conv_lower:
            return "high"
        elif "medium" in conv_lower or "moderate" in conv_lower:
            return "medium"
        else:
            return "low"

    def _get_strategy_bias(self, strategy_name: str) -> str:
        """Determine strategy directional bias"""
        strategy_key = strategy_name.lower().replace(" ", "_").replace("-", "_")

        if strategy_key in self.BEARISH_STRATEGIES:
            return "bearish"
        elif strategy_key in self.BULLISH_STRATEGIES:
            return "bullish"
        elif strategy_key in self.NEUTRAL_STRATEGIES:
            return "neutral"
        else:
            # Try to infer from name
            if "bear" in strategy_key or "put" in strategy_key:
                return "bearish"
            elif "bull" in strategy_key or "call" in strategy_key:
                return "bullish"
            else:
                return "neutral"


def validate_thesis_alignment(
    thesis: TradeThesis,
    proposal: TradeProposal,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for thesis validation

    Args:
        thesis: Trade thesis
        proposal: Trade proposal
        strict: If True, raises exception on critical mismatch

    Returns:
        Validation result dict

    Raises:
        ThesisMismatchError: If strict=True and critical mismatch found
    """
    validator = ThesisValidator(strict_mode=strict)
    return validator.validate(thesis, proposal)
