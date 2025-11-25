"""
Fund Manager Agent

Final decision-making node that synthesizes Trader proposal and Risk assessment
to make the ultimate Go/No-Go decision and determine position sizing.

Based on the "Trading Agents" paper architecture where a Fund Manager reviews
conflicting inputs from the Trader (wants profit) and Risk Team (wants safety).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from src.models.schemas import RiskAssessment, TradeProposal
from src.tools.llm import get_llm_client

logger = logging.getLogger(__name__)


class FundManagerAgent:
    """
    Fund Manager: Makes final trade execution decision

    Responsibilities:
    - Reviews trade proposal from Systematic Trader
    - Reviews risk assessment from Risk Management Team
    - Makes final Go/No-Go decision
    - Determines position sizing based on risk/reward and conviction
    - Acts as tie-breaker between profit motive and risk aversion
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = get_llm_client()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make final trading decision based on proposal and risk assessment

        Args:
            state: Must contain:
                - trade_proposal: TradeProposal from trader
                - risk_assessment: RiskAssessment from risk team
                - trade_thesis: TradeThesis with conviction level

        Returns:
            Dict with:
                - execute_trade: bool (True to execute, False to skip)
                - final_sizing: str ("full", "half", "quarter", "none")
                - manager_rationale: str (reasoning for decision)
                - final_recommendation: str (summary)
        """
        logger.info("Fund Manager reviewing trade decision...")

        trade_proposal: TradeProposal = state["trade_proposal"]
        risk_assessment: RiskAssessment = state["risk_assessment"]
        trade_thesis = state.get("trade_thesis")

        # Prepare payload for LLM
        proposal_summary = {
            "strategy": trade_proposal.strategy_name,
            "action": trade_proposal.action,
            "conviction": trade_proposal.conviction_level,
            "max_risk": trade_proposal.max_risk,
            "max_reward": trade_proposal.max_reward,
            "net_premium": trade_proposal.net_premium,
            "generation_status": trade_proposal.generation_status,
        }

        risk_summary = {
            "adjustments": [adj.model_dump() for adj in risk_assessment.adjustments],
            "final_recommendation": risk_assessment.final_recommendation,
        }

        thesis_summary = {
            "direction": trade_thesis.winning_argument if trade_thesis else "unknown",
            "conviction": trade_thesis.conviction_level if trade_thesis else "unknown",
        }

        # Call LLM with manager prompt
        prompt = self._build_manager_prompt(proposal_summary, risk_summary, thesis_summary)

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self.config.get("prompt", self._default_prompt())},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.config.get("temperature", 0.1),  # Low temperature for consistent decisions
            )

            # Parse LLM response (expecting JSON)
            decision = self._parse_response(response)

            logger.info(f"  Manager decision: {decision.get('execute_trade')}")
            logger.info(f"  Position sizing: {decision.get('final_sizing')}")

            return decision

        except Exception as e:
            logger.error(f"Fund Manager decision failed: {e}")
            # Conservative fallback: reject trade
            return {
                "execute_trade": False,
                "final_sizing": "none",
                "manager_rationale": f"Decision error: {e}",
                "final_recommendation": "Trade rejected due to processing error",
            }

    def _build_manager_prompt(
        self,
        proposal: Dict[str, Any],
        risk: Dict[str, Any],
        thesis: Dict[str, Any]
    ) -> str:
        """Build prompt for fund manager decision"""

        prompt_template = self.config.get("prompt", self._default_prompt())

        context = f"""
You are the Fund Manager making the final trading decision.

## Trade Thesis
- Direction: {thesis['direction']}
- Conviction: {thesis['conviction']}

## Proposed Trade
{json.dumps(proposal, indent=2)}

## Risk Assessment
{json.dumps(risk, indent=2)}

## Your Decision Framework
1. Check if trade generation succeeded (generation_status must be "generated")
2. Evaluate alignment between thesis, strategy, and risk assessment
3. Consider conviction level (High = more aggressive, Low = more conservative)
4. Review Risk Team's concerns and recommendations
5. Make Go/No-Go decision with position sizing

## Position Sizing Guidance
- "full" (100%): High conviction, low risk, strong alignment
- "half" (50%): Medium conviction or moderate risk concerns
- "quarter" (25%): Low conviction or significant risk flags
- "none" (0%): Reject trade (thesis weak, risks too high, or generation failed)

Respond with JSON only:
{{
  "execute_trade": true/false,
  "final_sizing": "full" | "half" | "quarter" | "none",
  "manager_rationale": "Your detailed reasoning for this decision",
  "final_recommendation": "One-sentence summary"
}}
"""
        return prompt_template + "\n\n" + context

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured decision"""
        try:
            # Try to extract JSON from response
            cleaned = response.strip()

            # Remove markdown code blocks if present
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()

            decision = json.loads(cleaned)

            # Validate required fields
            required_fields = ["execute_trade", "final_sizing", "manager_rationale", "final_recommendation"]
            for field in required_fields:
                if field not in decision:
                    raise ValueError(f"Missing required field: {field}")

            # Validate sizing
            valid_sizes = ["full", "half", "quarter", "none"]
            if decision["final_sizing"] not in valid_sizes:
                logger.warning(f"Invalid sizing {decision['final_sizing']}, defaulting to 'half'")
                decision["final_sizing"] = "half"

            return decision

        except Exception as e:
            logger.error(f"Failed to parse manager response: {e}")
            logger.debug(f"Raw response: {response}")
            # Return conservative default
            return {
                "execute_trade": False,
                "final_sizing": "none",
                "manager_rationale": f"Failed to parse decision: {e}",
                "final_recommendation": "Trade rejected due to parsing error",
            }

    def _default_prompt(self) -> str:
        """Default prompt if not configured"""
        return """You are an experienced Fund Manager for an institutional options trading desk.
Your role is to make the final Go/No-Go decision on trades by balancing profit potential against risk management concerns."""
