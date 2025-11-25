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

    def __init__(self, disable_credit_spreads: bool = True):
        self.library = STRATEGY_LIBRARY
        self.disable_credit_spreads = disable_credit_spreads

        # =========================================================================
        # CRITICAL FIX: Filter out dangerous strategies
        # =========================================================================
        # 1. Credit spreads have asymmetric risk (can lose >100% of premium received)
        # 2. Naked Call has UNLIMITED risk (caused -$29,000 loss on GOOGL)
        # 3. Collar requires stock ownership (caused -$972 loss when missing stock)
        # =========================================================================

        # ALWAYS exclude unlimited-risk, broken, and UNDERPERFORMING strategies
        # Based on 180-day evaluation analysis:
        # - Long Put: -$6,903 total loss, 39% win rate, -$247 avg (WORST strategy)
        # - Bear Put Spread: -$9,765 total loss, 50% win rate, -$174 avg
        # - Bull Call Spread: -$4,700 total loss, 61% win rate, -$27 avg (BEST, keep it)
        dangerous_strategies = [
            "Naked Call",       # UNLIMITED RISK - caused -$29,000 loss
            "Collar",           # Requires stock ownership (not modeled in backtest)
            "Butterfly Spread", # BUG: Asymmetric strikes + P&L formula creates fake profits
            "Iron Butterfly",   # Similar issues with complex multi-leg pricing
            "Long Put",         # DISABLED: -$6,903 loss in 180-day eval, 39% win rate, theta bleed
            "Bear Put Spread",  # DISABLED: -$9,765 loss in 180-day eval, buying puts in bull market
            "Long Straddle",    # DISABLED: Theta bleed on both legs
            "Long Strangle",    # DISABLED: Theta bleed on both legs
        ]

        self.library = {
            k: v for k, v in self.library.items()
            if v.name not in dangerous_strategies
        }
        logger.info(f"Removed {len(dangerous_strategies)} dangerous strategies (Naked Call, Collar)")

        if self.disable_credit_spreads:
            credit_strategies = [
                "Bull Put Spread",
                "Bear Call Spread",
                "Iron Condor",
                "Iron Butterfly",
                "Short Straddle",
                "Short Strangle",
                "Cash-Secured Put",
            ]
            self.library = {
                k: v for k, v in self.library.items()
                if v.name not in credit_strategies
            }
            logger.info(f"Credit spreads DISABLED - {len(self.library)} strategies available")

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
        # 1. BASELINE: Theta-Positive & High Win Rate Strategies
        # ========================================================================
        # STRATEGY: Prioritize strategies with positive theta (time decay works for us)
        # and high probability of profit. Take profits early (20%) for higher win rate.

        is_credit = (
            strategy.max_reward_type == "limited" and
            strategy.capital_requirement in ["medium", "high"]
        )
        is_neutral = strategy.directional_bias == "neutral"
        is_long_option = strategy.max_reward_type == "unlimited"
        is_theta_positive = strategy.theta_exposure == "positive"
        is_naked_option = strategy.max_risk_type == "unlimited"  # Naked call/put/strangle

        # THETA POSITIVE BONUS: Time decay works FOR us, not against us
        if is_theta_positive:
            score += 25  # Strong bonus for theta-positive strategies
            logger.debug(f"  Theta-positive strategy: +25 points (time decay advantage)")

        # NAKED OPTIONS LEVERAGE BONUS: High IV + correct direction = high leverage
        if is_naked_option and iv_rank > 50:
            score += 20  # Bonus for selling expensive premium
            logger.debug(f"  Naked option in high IV: +20 points (leverage play)")

        if is_credit:
            score += 15  # Increased credit bias for higher win rate
            logger.debug(f"  Credit strategy baseline: +15 points (variance harvesting)")

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
        # 4. WIN RATE OPTIMIZATION: Prefer high-probability strategies
        # ========================================================================
        # Long options have low win rate (~30-40%), short options have high win rate (~60-70%)
        # For higher win rate, penalize low-probability plays

        if is_long_option:
            # Long options = low win rate (theta works against us)
            if conviction != "high":
                score -= 60  # Strong penalty - these lose most of the time
                logger.debug(f"  Long option without conviction: -60 points (low win rate)")
            else:
                score -= 20  # Still penalize even with conviction (prefer credit strategies)
                logger.debug(f"  Long option with conviction: -20 points (prefer theta-positive)")

        # SINGLE-LEG CREDIT STRATEGIES: High win rate, simpler execution
        is_single_leg_credit = (
            strategy.leg_count == 1 and
            strategy.theta_exposure == "positive"
        )
        if is_single_leg_credit:
            score += 15  # Bonus for simple, high-probability plays
            logger.debug(f"  Single-leg credit: +15 points (simple, high win rate)")

        # ========================================================================
        # 5. Technical Confirmation (ENHANCED - TREND FOLLOWING BIAS)
        # ========================================================================
        # CRITICAL IMPROVEMENT: Never fight a strong technical trend
        # If technicals say Bullish but thesis says Bearish → HEAVY PENALTY
        # This is the key insight: Technical baseline wins because it follows trends

        if technical_bias:
            tech_lower = technical_bias.lower()
            tech_is_bullish = "bull" in tech_lower
            tech_is_bearish = "bear" in tech_lower
            tech_is_neutral = "neutral" in tech_lower or (not tech_is_bullish and not tech_is_bearish)

            # TREND ALIGNMENT BONUS
            if direction == "bullish" and tech_is_bullish:
                score += 25  # Strong bonus for trend alignment (was +10)
                logger.debug(f"  Technical CONFIRMS bullish: +25 points (TREND ALIGNED)")
            elif direction == "bearish" and tech_is_bearish:
                score += 25  # Strong bonus for trend alignment (was +10)
                logger.debug(f"  Technical CONFIRMS bearish: +25 points (TREND ALIGNED)")
            elif tech_is_neutral and is_neutral:
                score += 20  # Bonus for neutral confirmation
                logger.debug(f"  Technical confirms neutral: +20 points")

            # COUNTER-TREND PENALTY (CRITICAL - Never fight the trend!)
            # Fighting the trend is the #1 reason Quant13 underperforms
            elif direction == "bullish" and tech_is_bearish:
                score -= 80  # VERY heavy penalty for fighting bearish trend
                logger.warning(f"  COUNTER-TREND: Bullish thesis vs Bearish technicals: -80 points (AVOID)")
            elif direction == "bearish" and tech_is_bullish:
                score -= 80  # VERY heavy penalty for fighting bullish trend
                logger.warning(f"  COUNTER-TREND: Bearish thesis vs Bullish technicals: -80 points (AVOID)")

            # Neutral thesis with strong technical trend - suggest directional instead
            elif is_neutral and (tech_is_bullish or tech_is_bearish):
                score -= 15  # Mild penalty for missing trend opportunity
                logger.debug(f"  Neutral strategy in trending market: -15 points")
            else:
                score += 5  # Partial credit for any technical data

            # ========================================================================
            # CRITICAL FIX: Penalize STRATEGY that fights the technical trend
            # ========================================================================
            # Even if thesis is neutral/aligned, if the STRATEGY goes against technicals, KILL IT
            # This is the root cause of QQQ Bear Call Spread loss:
            # - Thesis was Bullish (correct)
            # - Technicals were Bullish (correct)
            # - But Bear Call Spread (bearish strategy) was still selected due to credit bonuses
            strategy_fights_tech = (
                (tech_is_bullish and strategy.directional_bias == "bearish") or
                (tech_is_bearish and strategy.directional_bias == "bullish")
            )
            if strategy_fights_tech:
                score -= 100  # KILL strategies that fight the technical trend
                logger.warning(
                    f"  STRATEGY vs TECHNICALS: {strategy.directional_bias.upper()} strategy "
                    f"vs {tech_lower.upper()} technicals = -100 points (REJECTED)"
                )

        return min(max(score, -50), 100.0)  # Floor at -50, cap at 100
