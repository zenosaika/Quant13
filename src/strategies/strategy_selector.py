"""
Systematic Strategy Selector

Rule-based strategy selection engine that maps market conditions
to optimal options strategies.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
import logging

from src.strategies.strategy_library import (
    STRATEGY_LIBRARY,
    StrategyType,
    StrategyBlueprint,
    filter_strategies
)
from src.models.schemas import TradeThesis, VolatilityReport

logger = logging.getLogger(__name__)


class SystematicStrategySelector:
    """
    Rule-based strategy selection engine

    Maps market conditions → optimal strategy using systematic scoring

    Inputs:
        - Trade thesis (direction, conviction, timeframe)
        - Volatility report (IV Rank, forecast)
        - Technical signals (optional)

    Output:
        - Ranked list of strategies with scores
    """

    def __init__(self):
        self.library = STRATEGY_LIBRARY

    def select_strategy(
        self,
        thesis: TradeThesis,
        volatility: VolatilityReport,
        technical_bias: Optional[str] = None
    ) -> List[Tuple[StrategyBlueprint, float]]:
        """
        Select optimal strategy using systematic rules

        Returns: List of (strategy, score) tuples, sorted by score descending

        Score range: 0-100
        - 90-100: Excellent fit
        - 70-89: Good fit
        - 50-69: Acceptable fit
        - <50: Poor fit
        """
        # 1. Parse inputs
        direction = self._parse_direction(thesis.winning_argument)
        conviction = self._parse_conviction(thesis.conviction_level)
        iv_regime = self._parse_iv_regime(volatility.iv_rank)

        logger.info(
            f"Strategy selection: direction={direction}, conviction={conviction}, "
            f"iv_regime={iv_regime}, iv_rank={volatility.iv_rank:.1f}"
        )

        # ========================================================================
        # CRITICAL FIX #1: IV Rank Hard Filter (EXPERT REVIEW FIX)
        # ========================================================================
        # NEVER sell premium when IV Rank < 20 (options are too cheap)
        if volatility.iv_rank < 20:
            logger.warning(f"IV Rank {volatility.iv_rank:.1f} < 20 - TOO LOW for premium selling")
            logger.warning("Filtering out all credit/neutral strategies (variance harvesting disabled)")

            # Helper function to identify TRUE debit strategies
            # Previous bug: Credit spreads (Bull Put, Bear Call) have "limited" risk too!
            def is_debit_strategy(strategy_name: str) -> bool:
                """
                Identify debit strategies by name pattern

                Debit = You PAY to enter (Long Call, Bull Call Spread, etc.)
                Credit = You RECEIVE to enter (Bull Put Spread, Iron Condor, etc.)
                """
                name_lower = strategy_name.lower()

                # Explicit debit indicators
                if "debit" in name_lower or "long" in name_lower:
                    return True

                # Bull Call Spread = DEBIT (buy lower call, sell higher call)
                if "bull" in name_lower and "call" in name_lower and "spread" in name_lower:
                    return True

                # Bear Put Spread = DEBIT (buy higher put, sell lower put)
                if "bear" in name_lower and "put" in name_lower and "spread" in name_lower:
                    return True

                # These are CREDIT strategies (exclude them)
                if "bull put" in name_lower or "bear call" in name_lower:
                    return False
                if "iron" in name_lower or "condor" in name_lower or "butterfly" in name_lower:
                    return False
                if "short" in name_lower:
                    return False

                return False

            # Filter for TRUE debit strategies only
            debit_candidates = [
                s for s in self.library.values()
                if s.directional_bias != "neutral"  # Exclude neutral strategies in low IV
                and is_debit_strategy(s.name)  # FIX: Properly identify debit strategies
            ]

            if not debit_candidates:
                logger.warning("No suitable debit strategies found, returning None (stay in cash)")
                return []  # Return empty list = no trade

            logger.info(f"Low IV mode: {len(debit_candidates)} directional debit strategies available")
            for s in debit_candidates:
                logger.debug(f"  Allowed: {s.name}")

            candidates = debit_candidates

        # 2. Filter strategies by hard constraints
        else:
            candidates = self._filter_candidates(direction, conviction, iv_regime)

        if not candidates:
            logger.warning("No strategies meet hard constraints, relaxing filters")
            candidates = list(self.library.values())

        # 3. Score each candidate
        scored = []
        for strategy in candidates:
            score = self._score_strategy(
                strategy,
                direction=direction,
                conviction=conviction,
                iv_rank=volatility.iv_rank,
                vol_forecast=volatility.volatility_forecast,
                technical_bias=technical_bias
            )
            scored.append((strategy, score))

            logger.debug(f"  {strategy.name}: score={score:.1f}")

        # 4. Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Top strategy: {scored[0][0].name} (score={scored[0][1]:.1f})")

        return scored

    def _parse_direction(self, winning_argument: str) -> str:
        """
        Extract bullish/bearish/neutral from winning argument

        Args:
            winning_argument: "Bullish" or "Bearish" from trade thesis

        Returns:
            "bullish", "bearish", or "neutral"
        """
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
        """
        Normalize conviction to low/medium/high

        Args:
            conviction_level: Raw conviction from thesis

        Returns:
            "low", "medium", or "high"
        """
        if not conviction_level:
            return "low"

        conv_lower = conviction_level.lower()

        if "high" in conv_lower or "strong" in conv_lower:
            return "high"
        elif "medium" in conv_lower or "moderate" in conv_lower:
            return "medium"
        else:
            return "low"

    def _parse_iv_regime(self, iv_rank: float) -> str:
        """
        Classify IV regime based on IV Rank

        Args:
            iv_rank: IV Rank (0-100)

        Returns:
            "high_iv", "mid_iv", or "low_iv"
        """
        if iv_rank >= 50:
            return "high_iv"
        elif iv_rank <= 30:
            return "low_iv"
        else:
            return "mid_iv"

    def _filter_candidates(
        self,
        direction: str,
        conviction: str,
        iv_regime: str
    ) -> List[StrategyBlueprint]:
        """
        Filter strategies by hard constraints

        A strategy must match:
        - Directional bias (exact match or strategy is "any")
        - IV regime (exact match or strategy is "any" or iv_regime is "mid_iv")
        - Conviction requirements (HIGH conviction MUST use directional)

        EXPERT REVIEW FIX: High conviction directional thesis CANNOT use neutral strategies
        """
        candidates = []

        for strategy in self.library.values():
            # ========================================================================
            # FIX #2: HIGH CONVICTION HARD GATE (EXPERT REVIEW FIX)
            # ========================================================================
            # If conviction is HIGH and direction is DIRECTIONAL (not neutral),
            # HARD REJECT all neutral strategies (Iron Condor, etc.)
            #
            # This prevents RBLX failure: Bearish High + Iron Condor
            if conviction == "high" and direction != "neutral":
                if strategy.directional_bias == "neutral":
                    logger.debug(f"  REJECTED {strategy.name}: High conviction requires directional strategy")
                    continue  # Hard reject neutral strategies

            # 1. Direction match
            # VARIANCE HARVESTING: Allow neutral strategies for low/medium conviction
            if strategy.directional_bias == "neutral":
                pass  # Allow neutral strategies (unless rejected by high conviction gate above)
            elif strategy.directional_bias != direction and strategy.directional_bias != "any":
                continue

            # 2. IV regime match
            if strategy.volatility_regime not in (iv_regime, "any"):
                # Special case: mid_iv can use either high or low strategies
                if iv_regime == "mid_iv":
                    pass  # Allow mid_iv to use any strategy
                else:
                    continue

            candidates.append(strategy)

        return candidates

    def _score_strategy(
        self,
        strategy: StrategyBlueprint,
        direction: str,
        conviction: str,
        iv_rank: float,
        vol_forecast: str,
        technical_bias: Optional[str]
    ) -> float:
        """
        Score strategy: "The Variance Harvester" Logic

        STRATEGIC SHIFT: From "Sniper" (directional) to "Casino" (income)
        - Prioritizes high-probability income (Theta decay)
        - Sells options (be the house, not the gambler)
        - Only allows directional plays with extreme conviction

        Philosophy:
        - Credit strategies = Dealer odds (we have the edge)
        - Debit strategies = Gambler odds (need to be right on direction + timing)
        - In choppy markets: Casino always wins long-term

        Returns: Score 0-100
        """
        score = 0.0

        # ========================================================================
        # 1. BASELINE: Credit Strategies (EXPERT REVIEW FIX - Reduced Bias)
        # ========================================================================
        # Selling options = Being the "House" (probability edge via theta decay)
        # PREVIOUS BUG: +30 was too high, caused Iron Condor to beat directional strategies
        # FIX: Reduced to +10 to avoid over-bias toward credit

        is_credit = (
            strategy.max_reward_type == "limited" and
            strategy.capital_requirement in ["medium", "high"]
        )
        is_neutral = strategy.directional_bias == "neutral"
        is_long_option = strategy.max_reward_type == "unlimited"

        if is_credit:
            score += 10  # FIX: Reduced from 30 to 10 (less aggressive credit bias)
            logger.debug(f"  Credit strategy baseline: +10 points (variance harvesting)")

        # ========================================================================
        # 2. Volatility Regime (The Edge)
        # ========================================================================
        # High IV = Expensive Options = SELL THEM
        # Low IV = Cheap Options = BUY THEM (only time debit spreads allowed)

        if iv_rank > 30:
            # High IV regime - sell premium
            if is_credit:
                score += 20
                logger.debug(f"  High IV + Credit: +20 points (expensive premium)")
            if is_neutral:
                score += 15  # Iron Condors love High IV
                logger.debug(f"  High IV + Neutral: +15 points (Iron Condor ideal)")
        elif iv_rank < 20:
            # Low IV regime - buying is okay
            if not is_credit:
                score += 15  # Boost Debit spreads/Longs only here
                logger.debug(f"  Low IV + Debit: +15 points (cheap options)")
        else:
            # Mid IV - neutral strategies still good
            if is_neutral:
                score += 10
                logger.debug(f"  Mid IV + Neutral: +10 points")

        # ========================================================================
        # 3. Direction & Conviction (The Filter) - PHASE 2 IMPROVEMENT
        # ========================================================================
        # CRITICAL FIX: Prevent thesis-strategy mismatches (RBLX failure mode)
        #
        # New Rules:
        # 1. High conviction directional thesis CANNOT use opposite or neutral strategies
        # 2. Medium conviction directional thesis heavily penalized for opposite strategies
        # 3. Low conviction prefers neutral strategies
        #
        # This prevents:
        # - Bearish High + Iron Condor (RBLX failure)
        # - Bullish High + Neutral strategy (AMD-style contradiction)

        # RULE 1: Detect critical mismatches (opposite direction)
        is_opposite_direction = (
            (direction == "bearish" and strategy.directional_bias == "bullish") or
            (direction == "bullish" and strategy.directional_bias == "bearish")
        )

        if is_opposite_direction:
            # CRITICAL PENALTY: Strategy contradicts thesis
            if conviction.lower() == "high":
                score -= 100  # KILL this strategy completely
                logger.warning(
                    f"  CRITICAL MISMATCH: {direction.upper()} thesis + "
                    f"{strategy.directional_bias.upper()} strategy = -100 points (REJECTED)"
                )
            elif conviction.lower() == "medium":
                score -= 80  # Very heavy penalty
                logger.warning(
                    f"  MAJOR MISMATCH: {direction.upper()} thesis + "
                    f"{strategy.directional_bias.upper()} strategy = -80 points"
                )

        # RULE 2: High conviction directional - MUST use directional strategy
        if conviction.lower() == "high":
            if strategy.directional_bias == direction:
                score += 50  # Massive bonus for alignment
                logger.debug(f"  High conviction + Direction match: +50 points (PERFECT FIT)")
            elif is_neutral:
                score -= 30  # Penalty for using neutral when thesis is strong
                logger.warning(
                    f"  High conviction {direction.upper()} thesis + Neutral strategy: "
                    f"-30 points (missed opportunity)"
                )

        # RULE 3: Medium conviction directional - prefer directional, allow neutral
        elif conviction.lower() == "medium":
            if strategy.directional_bias == direction:
                score += 35  # Good bonus for alignment
                logger.debug(
                    f"  Medium conviction + Direction match: +35 points (directional play)"
                )
            elif is_neutral:
                score += 5  # Small penalty for neutral (acceptable but not ideal)
                logger.debug(f"  Medium conviction + Neutral: +5 points (conservative choice)")

        # RULE 4: Low conviction - strongly prefer neutral
        else:
            if is_neutral:
                score += 40  # High bonus for neutral (variance harvesting)
                logger.debug(f"  Low conviction + Neutral: +40 points (VARIANCE HARVEST)")
            elif strategy.directional_bias != "neutral":
                score -= 25  # Penalty for directional without conviction
                logger.debug(f"  Low conviction + Directional: -25 points (no edge)")

        # ========================================================================
        # 4. Penalize "Lottery Tickets" (Long OTM Options)
        # ========================================================================
        # Long calls/puts without high conviction = gambling

        if is_long_option and conviction != "high":
            score -= 50  # Kill long calls/puts unless we are 100% sure
            logger.debug(f"  Long option without conviction: -50 points (lottery ticket)")
        elif is_long_option and conviction == "high":
            score += 5  # Small bonus if we're very sure
            logger.debug(f"  Long option with high conviction: +5 points (directional bet)")

        # ========================================================================
        # 5. Technical Confirmation (Bonus)
        # ========================================================================
        if technical_bias:
            tech_lower = technical_bias.lower()

            if direction == "bullish" and "bull" in tech_lower:
                score += 10
            elif direction == "bearish" and "bear" in tech_lower:
                score += 10
            elif "neutral" in tech_lower and is_neutral:
                score += 15  # Extra bonus for neutral confirmation
                logger.debug(f"  Technical confirms neutral: +15 points")
            else:
                score += 5  # Partial credit

        return min(score, 100.0)  # Cap at 100
