"""
Report Refiner

Uses LLM to refine and improve report narratives for better readability.
"""

from __future__ import annotations

import json
from typing import Dict, Any
import logging

from src.tools.llm import get_llm_client

logger = logging.getLogger(__name__)


class ReportRefiner:
    """
    LLM-based report refinement for polished output
    """

    def __init__(self):
        self.llm = get_llm_client()

    def refine_executive_summary(
        self,
        trade_proposal: Dict[str, Any],
        thesis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> str:
        """
        Generate polished executive summary

        Args:
            trade_proposal: TradeProposal dict
            thesis: TradeThesis dict
            risk_assessment: RiskAssessment dict

        Returns:
            2-3 paragraph executive summary
        """
        try:
            prompt = f"""
You are a senior options strategist writing an executive summary for a client report.

TRADE ANALYSIS:

THESIS:
{json.dumps(thesis, indent=2)}

PROPOSED TRADE:
Strategy: {trade_proposal.get('strategy_name')}
Action: {trade_proposal.get('action')}
Legs: {len(trade_proposal.get('trade_legs', []))}
Max Risk: {trade_proposal.get('max_risk')}
Max Reward: {trade_proposal.get('max_reward')}

RISK ASSESSMENT:
{json.dumps(risk_assessment.get('adjustments', [])[:2], indent=2)}

TASK:
Write a concise, professional executive summary (2-3 paragraphs) that:
1. States the market outlook and conviction level
2. Describes the proposed strategy and key rationale
3. Highlights expected outcomes and primary risks

Style: Clear, professional, suitable for institutional clients.
Avoid jargon unless necessary. Explain technical terms briefly.

Return ONLY the summary text, no preamble.
"""

            messages = [
                {"role": "system", "content": "You are a professional options strategist."},
                {"role": "user", "content": prompt}
            ]

            summary = self.llm.chat(messages, temperature=0.3)
            return summary.strip()

        except Exception as e:
            logger.error(f"Executive summary refinement failed: {e}")
            return self._fallback_executive_summary(trade_proposal, thesis)

    def _fallback_executive_summary(
        self,
        trade_proposal: Dict[str, Any],
        thesis: Dict[str, Any]
    ) -> str:
        """Fallback summary when LLM fails"""
        direction = thesis.get("winning_argument", "Neutral")
        conviction = thesis.get("conviction_level", "Medium")
        strategy = trade_proposal.get("strategy_name", "Options Strategy")

        return f"""
Based on our multi-analyst review, we have developed a {conviction.lower()} conviction {direction.lower()} thesis.

The recommended approach is a {strategy}, which aligns with current market conditions and volatility levels. This strategy offers defined risk with a favorable risk-reward profile.

Key considerations include monitoring volatility changes and market catalysts that could impact the trade thesis.
"""

    def refine_risk_disclosure(
        self,
        trade_proposal: Dict[str, Any],
        max_risk: Optional[float],
        max_reward: Optional[float]
    ) -> str:
        """
        Generate standardized risk disclosure

        Args:
            trade_proposal: TradeProposal dict
            max_risk: Maximum loss
            max_reward: Maximum profit

        Returns:
            Risk disclosure paragraph
        """
        try:
            strategy_name = trade_proposal.get("strategy_name", "options strategy")

            risk_str = f"${max_risk:,.2f}" if max_risk else "unlimited"
            reward_str = f"${max_reward:,.2f}" if max_reward else "unlimited"

            prompt = f"""
Generate a professional risk disclosure statement for this options trade:

Strategy: {strategy_name}
Maximum Risk: {risk_str}
Maximum Reward: {reward_str}

The disclosure should:
1. Clearly state maximum loss potential
2. Mention key risks (volatility, time decay, directional risk)
3. Note that options can expire worthless
4. Include appropriate disclaimer language

Keep it concise (1 paragraph, 3-4 sentences) and professional.

Return ONLY the disclosure text, no preamble.
"""

            messages = [
                {"role": "system", "content": "You are a compliance officer writing risk disclosures."},
                {"role": "user", "content": prompt}
            ]

            disclosure = self.llm.chat(messages, temperature=0.1)
            return disclosure.strip()

        except Exception as e:
            logger.error(f"Risk disclosure generation failed: {e}")
            return self._fallback_risk_disclosure(max_risk, max_reward)

    def _fallback_risk_disclosure(
        self,
        max_risk: Optional[float],
        max_reward: Optional[float]
    ) -> str:
        """Fallback risk disclosure"""
        risk_str = f"${max_risk:,.2f}" if max_risk else "substantial losses"

        return f"""
RISK DISCLOSURE: This options strategy involves risk of loss up to {risk_str}.
Options trading involves significant risk and is not suitable for all investors.
Factors including time decay, volatility changes, and adverse price movements can result in partial or total loss of investment.
Options may expire worthless, resulting in 100% loss of premium paid.
Past performance does not guarantee future results.
"""
