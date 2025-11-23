"""
Systematic Trader Agent

Uses rule-based strategy selection and systematic strike selection
to generate consistent, explainable trades.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import logging

from src.models.schemas import TradeLeg, TradeProposal, TradeThesis, VolatilityReport
from src.strategies import SystematicStrategySelector, StrikeSelector
from src.strategies.strike_selector import flatten_options_chain
from src.tools.llm import get_llm_client

logger = logging.getLogger(__name__)


class SystematicTraderAgent:
    """
    Systematic trader using rule-based strategy selection

    Advantages over LLM-based approach:
    - Deterministic: same inputs → same strategy
    - Explainable: clear scoring system
    - Fast: no LLM latency for strategy selection
    - Consistent: no LLM temperature variability

    Still uses LLM for:
    - Narrative explanation generation
    - Trade rationale documentation
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.selector = SystematicStrategySelector()
        self.strike_selector = StrikeSelector()
        self.llm = get_llm_client()

    def propose_trade(
        self,
        thesis: TradeThesis,
        volatility_report: VolatilityReport,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
    ) -> TradeProposal:
        """
        Generate trade proposal using systematic approach

        Process:
        1. Systematic strategy selection (deterministic)
        2. Systematic strike selection (delta-based)
        3. LLM narrative generation (explanation)

        Args:
            thesis: Trade thesis from debate
            volatility_report: Volatility analysis
            options_chain: Enriched options chain (with computed Greeks)
            spot_price: Current underlying price

        Returns:
            TradeProposal with all details
        """
        try:
            # 1. Select strategy systematically
            ranked_strategies = self.selector.select_strategy(
                thesis=thesis,
                volatility=volatility_report,
                technical_bias=None
            )

            if not ranked_strategies:
                return self._failure_proposal(
                    "No suitable strategy found for market conditions",
                    thesis
                )

            # Get top 3 strategies for logging
            top_strategy, top_score = ranked_strategies[0]

            logger.info(f"Selected strategy: {top_strategy.name} (score={top_score:.1f})")

            if len(ranked_strategies) > 1:
                second_strategy, second_score = ranked_strategies[1]
                logger.info(f"  Runner-up: {second_strategy.name} (score={second_score:.1f})")

            # 2. Select strikes systematically
            options_df = flatten_options_chain(options_chain)

            if options_df.empty:
                return self._failure_proposal(
                    "Options chain is empty",
                    thesis
                )

            legs = self.strike_selector.select_strikes_for_strategy(
                strategy=top_strategy,
                options_chain_flat=options_df,
                spot_price=spot_price,
                target_dte=35
            )

            if not legs:
                return self._failure_proposal(
                    f"Could not select strikes for {top_strategy.name}",
                    thesis
                )

            # 3. Validate leg count
            if len(legs) != top_strategy.leg_count:
                logger.warning(
                    f"Expected {top_strategy.leg_count} legs, got {len(legs)}. "
                    f"Proceeding anyway."
                )

            # 4. Convert to TradeLeg objects
            trade_legs = [TradeLeg(**leg) for leg in legs]

            # 5. Determine action (BUY_TO_OPEN vs SELL_TO_OPEN)
            action = self._determine_action(top_strategy, legs)

            # 6. Generate LLM explanation
            explanation = self._generate_explanation(
                strategy=top_strategy,
                legs=legs,
                thesis=thesis,
                volatility=volatility_report,
                score=top_score,
                spot_price=spot_price
            )

            # 7. Build proposal
            return TradeProposal(
                agent="SystematicTraderAgent",
                strategy_name=top_strategy.name,
                action=action,
                quantity=1,
                trade_legs=trade_legs,
                notes=explanation,
                conviction_level=thesis.conviction_level,
                generation_status="generated"
            )

        except Exception as e:
            logger.error(f"Error in systematic trade generation: {e}", exc_info=True)
            return self._failure_proposal(
                f"Trade generation error: {str(e)}",
                thesis
            )

    def _determine_action(
        self,
        strategy,
        legs: List[Dict[str, Any]]
    ) -> str:
        """
        Determine if trade is BUY_TO_OPEN or SELL_TO_OPEN

        Based on net premium flow:
        - Net debit → BUY_TO_OPEN
        - Net credit → SELL_TO_OPEN
        """
        # Count buy vs sell actions
        buy_count = sum(1 for leg in legs if leg["action"] == "BUY")
        sell_count = sum(1 for leg in legs if leg["action"] == "SELL")

        # For spreads, typically:
        # - Debit spreads: more or equal buys
        # - Credit spreads: more sells

        if sell_count > buy_count:
            return "SELL_TO_OPEN"
        else:
            return "BUY_TO_OPEN"

    def _generate_explanation(
        self,
        strategy,
        legs: List[Dict[str, Any]],
        thesis: TradeThesis,
        volatility: VolatilityReport,
        score: float,
        spot_price: float
    ) -> str:
        """
        Generate LLM-based explanation for the trade

        This provides human-readable rationale even though
        the strategy selection was deterministic.
        """
        try:
            # Build context for LLM
            leg_summary = []
            for leg in legs:
                leg_summary.append({
                    "action": leg["action"],
                    "type": leg["type"],
                    "strike": leg["strike_price"],
                    "delta": leg.get("key_greeks_at_selection", {}).get("delta")
                })

            prompt = f"""
You are an options strategist explaining a trade recommendation to a client.

MARKET CONTEXT:
- Thesis: {thesis.winning_argument} (Conviction: {thesis.conviction_level})
- Current Price: ${spot_price:.2f}
- IV Rank: {volatility.iv_rank:.1f}
- Volatility Forecast: {volatility.volatility_forecast}

SELECTED STRATEGY:
- Strategy: {strategy.name}
- Selection Score: {score:.1f}/100
- Description: {strategy.description}

TRADE LEGS:
{json.dumps(leg_summary, indent=2)}

CRITICAL CONSTRAINT (EXPERT REVIEW FIX):
You MUST accept the provided Thesis: "{thesis.winning_argument}" as absolute fact.
DO NOT re-analyze the market direction or suggest a different directional bias.
Your job is ONLY to explain WHY the selected strategy aligns with this thesis.

If the strategy seems mismatched with the thesis, point that out as a concern,
but DO NOT change the thesis itself.

TASK:
Write a concise trade rationale (2-3 paragraphs) explaining:
1. WHY this strategy fits the {thesis.winning_argument} thesis
2. WHAT the trade structure is and how it works
3. KEY RISKS and what to watch for

Keep it professional and actionable. Focus on the logic, not just repeating the data.
"""

            messages = [
                {"role": "system", "content": "You are a professional options strategist."},
                {"role": "user", "content": prompt}
            ]

            explanation = self.llm.chat(messages, temperature=0.3)

            # Append structured data
            explanation += f"\n\n```json\n{json.dumps({'strategy': strategy.name, 'legs': leg_summary}, indent=2)}\n```"

            return explanation

        except Exception as e:
            logger.warning(f"LLM explanation generation failed: {e}")
            # Fallback to template
            return self._template_explanation(strategy, legs, thesis, spot_price)

    def _template_explanation(
        self,
        strategy,
        legs: List[Dict[str, Any]],
        thesis: TradeThesis,
        spot_price: float
    ) -> str:
        """Fallback template-based explanation"""
        leg_desc = ", ".join([
            f"{leg['action']} {leg['type']} ${leg['strike_price']:.2f}"
            for leg in legs
        ])

        return f"""
**{strategy.name}** selected based on {thesis.winning_argument} thesis with {thesis.conviction_level} conviction.

**Trade Structure:**
{leg_desc}

**Rationale:**
{strategy.description}

Current price: ${spot_price:.2f}

**Risk Profile:**
- Max Risk: {strategy.max_risk_type}
- Max Reward: {strategy.max_reward_type}
- Capital Required: {strategy.capital_requirement}

**Greeks Exposure:**
- Delta: {strategy.delta_exposure}
- Vega: {strategy.vega_exposure}
- Theta: {strategy.theta_exposure}
"""

    def _failure_proposal(self, reason: str, thesis: TradeThesis) -> TradeProposal:
        """Create failure proposal"""
        return TradeProposal(
            agent="SystematicTraderAgent",
            strategy_name="Unspecified",
            action="UNDEFINED",
            quantity=0,
            trade_legs=[],
            notes=f"Trade generation failed: {reason}",
            conviction_level=thesis.conviction_level if thesis else None,
            generation_status="failed"
        )
