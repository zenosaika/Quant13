"""
Options Strategy Library

Defines 15+ predefined options strategies with clear construction rules,
selection criteria, and risk characteristics.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional
from dataclasses import dataclass, field
from enum import Enum


class StrategyType(Enum):
    """Enumeration of all supported options strategies"""

    # Directional Strategies
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"  # Cash-secured put
    SHORT_CALL = "short_call"  # Naked call (bearish leverage)

    # Vertical Spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Volatility Strategies
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"

    # Income Strategies
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"

    # Calendar Strategies
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"

    # Advanced Strategies
    BUTTERFLY_SPREAD = "butterfly_spread"
    COLLAR = "collar"


@dataclass
class LegTemplate:
    """Template for a single option leg"""
    action: Literal["BUY", "SELL"]
    option_type: Literal["CALL", "PUT"]
    strike_selection: str  # e.g., "atm", "otm", "itm", "far_otm"
    quantity: int = 1
    expiration_offset: int = 0  # Days offset from primary expiration (for calendars)


@dataclass
class StrategyBlueprint:
    """
    Complete specification for an options strategy

    Defines everything needed to systematically select and construct a strategy.
    """

    name: str
    type: StrategyType
    description: str

    # Market condition filters (hard constraints)
    directional_bias: Literal["bullish", "bearish", "neutral"]
    volatility_regime: Literal["high_iv", "low_iv", "any"]
    conviction_required: Literal["high", "medium", "low"]

    # Position structure
    leg_count: int
    legs: List[LegTemplate]

    # Risk characteristics
    max_risk_type: Literal["limited", "unlimited"]
    max_reward_type: Literal["limited", "unlimited"]
    capital_requirement: Literal["low", "medium", "high"]

    # Greeks profile
    delta_exposure: str  # "positive", "negative", "neutral"
    vega_exposure: str  # "long_volatility", "short_volatility", "neutral"
    theta_exposure: str  # "positive" (time decay benefit), "negative"

    # Selection criteria
    min_dte: int = 14
    max_dte: int = 60
    strike_selection_rules: Dict[str, any] = field(default_factory=dict)

    # Ideal market conditions (for scoring)
    ideal_iv_rank_min: Optional[float] = None
    ideal_iv_rank_max: Optional[float] = None
    ideal_conviction: Optional[str] = None


# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

STRATEGY_LIBRARY: Dict[StrategyType, StrategyBlueprint] = {

    # ========================================================================
    # BULLISH DIRECTIONAL STRATEGIES
    # ========================================================================

    StrategyType.LONG_CALL: StrategyBlueprint(
        name="Long Call",
        type=StrategyType.LONG_CALL,
        description="Buy call option. Maximum bullish conviction play with leveraged upside.",

        directional_bias="bullish",
        volatility_regime="low_iv",  # Buy when IV is cheap
        conviction_required="high",

        leg_count=1,
        legs=[
            LegTemplate(
                action="BUY",
                option_type="CALL",
                strike_selection="slightly_otm",  # Delta ~0.60-0.70
                quantity=1
            )
        ],

        max_risk_type="limited",  # Limited to premium paid
        max_reward_type="unlimited",
        capital_requirement="low",

        delta_exposure="positive",
        vega_exposure="long_volatility",  # Benefit from IV increase
        theta_exposure="negative",  # Time decay hurts

        min_dte=21,
        max_dte=60,

        strike_selection_rules={
            "target_delta": (0.50, 0.60),  # Reduced from 0.60-0.70 to avoid deep ITM
            "max_distance_from_spot": 0.10  # Within 10% of spot
        },

        ideal_iv_rank_min=0,
        ideal_iv_rank_max=30,
        ideal_conviction="high"
    ),

    StrategyType.BULL_CALL_SPREAD: StrategyBlueprint(
        name="Bull Call Spread",
        type=StrategyType.BULL_CALL_SPREAD,
        description="Buy lower strike call, sell higher strike call. Defined risk bullish play.",

        directional_bias="bullish",
        volatility_regime="any",
        conviction_required="medium",

        leg_count=2,
        legs=[
            LegTemplate(
                action="BUY",
                option_type="CALL",
                strike_selection="slightly_otm",  # Delta ~0.60-0.70
                quantity=1
            ),
            LegTemplate(
                action="SELL",
                option_type="CALL",
                strike_selection="otm",  # Delta ~0.30-0.40
                quantity=1
            )
        ],

        max_risk_type="limited",  # Net debit
        max_reward_type="limited",  # Width - debit
        capital_requirement="low",

        delta_exposure="positive",
        vega_exposure="neutral",  # Spreads have minimal vega
        theta_exposure="negative",  # Small negative theta

        min_dte=21,
        max_dte=60,

        strike_selection_rules={
            "long_strike_delta": (0.45, 0.55),  # Reduced from 0.60-0.70 for ATM positioning
            "short_strike_delta": (0.25, 0.35),  # Adjusted proportionally
            "target_width": 5.0,  # $5 between strikes
            "max_width": 10.0
        },

        ideal_iv_rank_min=20,
        ideal_iv_rank_max=60,
        ideal_conviction="medium"
    ),

    StrategyType.BULL_PUT_SPREAD: StrategyBlueprint(
        name="Bull Put Spread",
        type=StrategyType.BULL_PUT_SPREAD,
        description="Sell OTM put, buy further OTM put. Credit spread for bullish outlook.",

        directional_bias="bullish",
        volatility_regime="high_iv",  # Sell premium when IV is rich
        conviction_required="medium",

        leg_count=2,
        legs=[
            LegTemplate(
                action="SELL",
                option_type="PUT",
                strike_selection="otm",  # Delta ~-0.30
                quantity=1
            ),
            LegTemplate(
                action="BUY",
                option_type="PUT",
                strike_selection="far_otm",  # Delta ~-0.15
                quantity=1
            )
        ],

        max_risk_type="limited",  # Width - credit
        max_reward_type="limited",  # Net credit
        capital_requirement="medium",  # Requires collateral

        delta_exposure="positive",
        vega_exposure="short_volatility",  # Benefit from IV decrease
        theta_exposure="positive",  # Collect time decay

        min_dte=30,
        max_dte=60,

        strike_selection_rules={
            "short_strike_delta": (-0.35, -0.25),
            "long_strike_delta": (-0.20, -0.10),
            "target_width": 5.0,
            "probability_otm": 0.70  # 70% probability short strike expires OTM
        },

        ideal_iv_rank_min=50,
        ideal_iv_rank_max=100,
        ideal_conviction="medium"
    ),

    # ========================================================================
    # BEARISH DIRECTIONAL STRATEGIES
    # ========================================================================

    StrategyType.LONG_PUT: StrategyBlueprint(
        name="Long Put",
        type=StrategyType.LONG_PUT,
        description="Buy put option. Maximum bearish conviction with defined risk.",

        directional_bias="bearish",
        volatility_regime="low_iv",
        conviction_required="high",

        leg_count=1,
        legs=[
            LegTemplate(
                action="BUY",
                option_type="PUT",
                strike_selection="slightly_otm",  # Delta ~-0.60 to -0.70
                quantity=1
            )
        ],

        max_risk_type="limited",
        max_reward_type="limited",  # Strike - premium
        capital_requirement="low",

        delta_exposure="negative",
        vega_exposure="long_volatility",
        theta_exposure="negative",

        min_dte=21,
        max_dte=60,

        strike_selection_rules={
            "target_delta": (-0.60, -0.50)  # Reduced from -0.70 to -0.60 for consistency
        },

        ideal_iv_rank_min=0,
        ideal_iv_rank_max=30,
        ideal_conviction="high"
    ),

    StrategyType.BEAR_CALL_SPREAD: StrategyBlueprint(
        name="Bear Call Spread",
        type=StrategyType.BEAR_CALL_SPREAD,
        description="Sell OTM call, buy further OTM call. Credit spread for bearish outlook.",

        directional_bias="bearish",
        volatility_regime="high_iv",
        conviction_required="medium",

        leg_count=2,
        legs=[
            LegTemplate(
                action="SELL",
                option_type="CALL",
                strike_selection="otm",  # Delta ~0.30
                quantity=1
            ),
            LegTemplate(
                action="BUY",
                option_type="CALL",
                strike_selection="far_otm",  # Delta ~0.15
                quantity=1
            )
        ],

        max_risk_type="limited",
        max_reward_type="limited",
        capital_requirement="medium",

        delta_exposure="negative",
        vega_exposure="short_volatility",
        theta_exposure="positive",

        min_dte=30,
        max_dte=60,

        strike_selection_rules={
            "short_strike_delta": (0.25, 0.35),
            "long_strike_delta": (0.10, 0.20),
            "target_width": 5.0
        },

        ideal_iv_rank_min=50,
        ideal_iv_rank_max=100,
        ideal_conviction="medium"
    ),

    StrategyType.BEAR_PUT_SPREAD: StrategyBlueprint(
        name="Bear Put Spread",
        type=StrategyType.BEAR_PUT_SPREAD,
        description="Buy higher strike put, sell lower strike put. Defined risk bearish play.",

        directional_bias="bearish",
        volatility_regime="any",
        conviction_required="medium",

        leg_count=2,
        legs=[
            LegTemplate(
                action="BUY",
                option_type="PUT",
                strike_selection="slightly_otm",  # Delta ~-0.60
                quantity=1
            ),
            LegTemplate(
                action="SELL",
                option_type="PUT",
                strike_selection="otm",  # Delta ~-0.30
                quantity=1
            )
        ],

        max_risk_type="limited",
        max_reward_type="limited",
        capital_requirement="low",

        delta_exposure="negative",
        vega_exposure="neutral",
        theta_exposure="negative",

        min_dte=21,
        max_dte=60,

        strike_selection_rules={
            "long_strike_delta": (-0.70, -0.60),
            "short_strike_delta": (-0.40, -0.30),
            "target_width": 5.0
        },

        ideal_iv_rank_min=20,
        ideal_iv_rank_max=60,
        ideal_conviction="medium"
    ),

    # ========================================================================
    # NEUTRAL / VOLATILITY STRATEGIES
    # ========================================================================

    StrategyType.IRON_CONDOR: StrategyBlueprint(
        name="Iron Condor",
        type=StrategyType.IRON_CONDOR,
        description="Sell OTM call spread + sell OTM put spread. Profit from range-bound market.",

        directional_bias="neutral",
        volatility_regime="high_iv",  # Sell premium when IV is rich
        conviction_required="low",

        leg_count=4,
        legs=[
            # Put spread (lower)
            LegTemplate(action="BUY", option_type="PUT", strike_selection="far_otm", quantity=1),
            LegTemplate(action="SELL", option_type="PUT", strike_selection="otm", quantity=1),
            # Call spread (upper)
            LegTemplate(action="SELL", option_type="CALL", strike_selection="otm", quantity=1),
            LegTemplate(action="BUY", option_type="CALL", strike_selection="far_otm", quantity=1),
        ],

        max_risk_type="limited",  # Max(wing width) - credit
        max_reward_type="limited",  # Net credit
        capital_requirement="medium",

        delta_exposure="neutral",
        vega_exposure="short_volatility",  # Benefit from IV crush
        theta_exposure="positive",  # Collect time decay

        min_dte=30,
        max_dte=60,

        strike_selection_rules={
            "put_spread_center_delta": -0.20,  # Center around -0.20 delta
            "call_spread_center_delta": 0.20,  # Center around +0.20 delta
            "wing_width": 5.0,  # $5 width for both spreads
            "probability_of_profit": 0.70  # Target 70% PoP
        },

        ideal_iv_rank_min=50,
        ideal_iv_rank_max=100,
        ideal_conviction="low"
    ),

    StrategyType.IRON_BUTTERFLY: StrategyBlueprint(
        name="Iron Butterfly",
        type=StrategyType.IRON_BUTTERFLY,
        description="Sell ATM call and put, buy OTM call and put wings. Tight range bet.",

        directional_bias="neutral",
        volatility_regime="high_iv",
        conviction_required="low",

        leg_count=4,
        legs=[
            LegTemplate(action="BUY", option_type="PUT", strike_selection="otm", quantity=1),
            LegTemplate(action="SELL", option_type="PUT", strike_selection="atm", quantity=1),
            LegTemplate(action="SELL", option_type="CALL", strike_selection="atm", quantity=1),
            LegTemplate(action="BUY", option_type="CALL", strike_selection="otm", quantity=1),
        ],

        max_risk_type="limited",
        max_reward_type="limited",
        capital_requirement="medium",

        delta_exposure="neutral",
        vega_exposure="short_volatility",
        theta_exposure="positive",

        min_dte=21,
        max_dte=45,

        strike_selection_rules={
            "body_strike": "atm",  # Sell ATM straddle
            "wing_distance": 5.0  # Wings $5 away
        },

        ideal_iv_rank_min=60,
        ideal_iv_rank_max=100,
        ideal_conviction="low"
    ),

    StrategyType.LONG_STRADDLE: StrategyBlueprint(
        name="Long Straddle",
        type=StrategyType.LONG_STRADDLE,
        description="Buy ATM call and put. Bet on large move in either direction.",

        directional_bias="neutral",
        volatility_regime="low_iv",  # Buy volatility when cheap
        conviction_required="low",

        leg_count=2,
        legs=[
            LegTemplate(action="BUY", option_type="CALL", strike_selection="atm", quantity=1),
            LegTemplate(action="BUY", option_type="PUT", strike_selection="atm", quantity=1),
        ],

        max_risk_type="limited",  # Total premium
        max_reward_type="unlimited",
        capital_requirement="medium",

        delta_exposure="neutral",
        vega_exposure="long_volatility",  # Benefit from IV spike
        theta_exposure="negative",  # Time decay hurts

        min_dte=14,
        max_dte=45,

        strike_selection_rules={
            "strike": "atm"
        },

        ideal_iv_rank_min=0,
        ideal_iv_rank_max=30,
        ideal_conviction="low"
    ),

    StrategyType.LONG_STRANGLE: StrategyBlueprint(
        name="Long Strangle",
        type=StrategyType.LONG_STRANGLE,
        description="Buy OTM call and put. Cheaper volatility play than straddle.",

        directional_bias="neutral",
        volatility_regime="low_iv",
        conviction_required="low",

        leg_count=2,
        legs=[
            LegTemplate(action="BUY", option_type="CALL", strike_selection="otm", quantity=1),
            LegTemplate(action="BUY", option_type="PUT", strike_selection="otm", quantity=1),
        ],

        max_risk_type="limited",
        max_reward_type="unlimited",
        capital_requirement="low",

        delta_exposure="neutral",
        vega_exposure="long_volatility",
        theta_exposure="negative",

        min_dte=14,
        max_dte=45,

        strike_selection_rules={
            "call_delta": (0.30, 0.40),
            "put_delta": (-0.40, -0.30)
        },

        ideal_iv_rank_min=0,
        ideal_iv_rank_max=30,
        ideal_conviction="low"
    ),

    StrategyType.SHORT_STRANGLE: StrategyBlueprint(
        name="Short Strangle",
        type=StrategyType.SHORT_STRANGLE,
        description="Sell OTM call and put. Collect premium from range-bound market.",

        directional_bias="neutral",
        volatility_regime="high_iv",
        conviction_required="low",

        leg_count=2,
        legs=[
            LegTemplate(action="SELL", option_type="CALL", strike_selection="otm", quantity=1),
            LegTemplate(action="SELL", option_type="PUT", strike_selection="otm", quantity=1),
        ],

        max_risk_type="unlimited",  # Naked short options
        max_reward_type="limited",
        capital_requirement="high",

        delta_exposure="neutral",
        vega_exposure="short_volatility",
        theta_exposure="positive",

        min_dte=30,
        max_dte=60,

        strike_selection_rules={
            "call_delta": (0.20, 0.30),
            "put_delta": (-0.30, -0.20)
        },

        ideal_iv_rank_min=70,
        ideal_iv_rank_max=100,
        ideal_conviction="low"
    ),

    # ========================================================================
    # INCOME STRATEGIES
    # ========================================================================

    StrategyType.SHORT_PUT: StrategyBlueprint(
        name="Cash-Secured Put",
        type=StrategyType.SHORT_PUT,
        description="Sell put option. Generate income, willing to own stock at strike.",

        directional_bias="bullish",
        volatility_regime="high_iv",
        conviction_required="low",

        leg_count=1,
        legs=[
            LegTemplate(action="SELL", option_type="PUT", strike_selection="otm", quantity=1)
        ],

        max_risk_type="limited",  # Strike - premium (if assigned)
        max_reward_type="limited",  # Premium received
        capital_requirement="medium",  # Reduced for more leverage

        delta_exposure="positive",  # Short put = bullish
        vega_exposure="short_volatility",
        theta_exposure="positive",

        min_dte=21,
        max_dte=45,

        strike_selection_rules={
            "target_delta": (-0.30, -0.20)
        },

        ideal_iv_rank_min=50,
        ideal_iv_rank_max=100,
        ideal_conviction="low"
    ),

    StrategyType.SHORT_CALL: StrategyBlueprint(
        name="Naked Call",
        type=StrategyType.SHORT_CALL,
        description="Sell call option. High leverage bearish play with unlimited risk.",

        directional_bias="bearish",
        volatility_regime="high_iv",  # Sell when IV is high
        conviction_required="high",  # Only with strong conviction

        leg_count=1,
        legs=[
            LegTemplate(action="SELL", option_type="CALL", strike_selection="otm", quantity=1)
        ],

        max_risk_type="unlimited",  # Naked call = unlimited upside risk
        max_reward_type="limited",  # Premium received
        capital_requirement="medium",  # Margin required

        delta_exposure="negative",  # Short call = bearish
        vega_exposure="short_volatility",
        theta_exposure="positive",  # Theta decay benefits

        min_dte=21,
        max_dte=45,

        strike_selection_rules={
            "target_delta": (0.20, 0.30)  # OTM call
        },

        ideal_iv_rank_min=60,
        ideal_iv_rank_max=100,
        ideal_conviction="high"
    ),

    # ========================================================================
    # ADVANCED STRATEGIES
    # ========================================================================

    StrategyType.BUTTERFLY_SPREAD: StrategyBlueprint(
        name="Butterfly Spread",
        type=StrategyType.BUTTERFLY_SPREAD,
        description="Buy 1 lower call, sell 2 middle calls, buy 1 higher call. Tight range play.",

        directional_bias="neutral",
        volatility_regime="any",
        conviction_required="low",

        leg_count=4,
        legs=[
            LegTemplate(action="BUY", option_type="CALL", strike_selection="itm", quantity=1),
            LegTemplate(action="SELL", option_type="CALL", strike_selection="atm", quantity=2),
            LegTemplate(action="BUY", option_type="CALL", strike_selection="otm", quantity=1),
        ],

        max_risk_type="limited",
        max_reward_type="limited",
        capital_requirement="low",

        delta_exposure="neutral",
        vega_exposure="short_volatility",
        theta_exposure="positive",

        min_dte=21,
        max_dte=45,

        strike_selection_rules={
            "body_strike": "atm",
            "wing_spacing": 5.0  # $5 between strikes
        },

        ideal_iv_rank_min=40,
        ideal_iv_rank_max=80,
        ideal_conviction="low"
    ),

    StrategyType.COLLAR: StrategyBlueprint(
        name="Collar",
        type=StrategyType.COLLAR,
        description="Own stock + sell OTM call + buy OTM put. Protective hedge with income.",

        directional_bias="neutral",
        volatility_regime="any",
        conviction_required="low",

        leg_count=2,  # Assumes stock ownership
        legs=[
            LegTemplate(action="SELL", option_type="CALL", strike_selection="otm", quantity=1),
            LegTemplate(action="BUY", option_type="PUT", strike_selection="otm", quantity=1),
        ],

        max_risk_type="limited",
        max_reward_type="limited",
        capital_requirement="high",  # Requires stock

        delta_exposure="positive",  # Net positive with stock
        vega_exposure="neutral",
        theta_exposure="neutral",

        min_dte=30,
        max_dte=90,

        strike_selection_rules={
            "call_delta": (0.30, 0.40),
            "put_delta": (-0.40, -0.30)
        },

        ideal_iv_rank_min=30,
        ideal_iv_rank_max=70,
        ideal_conviction="low"
    ),

}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_strategy_by_type(strategy_type: StrategyType) -> Optional[StrategyBlueprint]:
    """Get strategy blueprint by type"""
    return STRATEGY_LIBRARY.get(strategy_type)


def get_strategy_by_name(name: str) -> Optional[StrategyBlueprint]:
    """Get strategy blueprint by name (case-insensitive)"""
    name_lower = name.lower().replace(" ", "_")
    for strategy in STRATEGY_LIBRARY.values():
        if strategy.name.lower().replace(" ", "_") == name_lower:
            return strategy
    return None


def list_all_strategies() -> List[StrategyBlueprint]:
    """Get list of all available strategies"""
    return list(STRATEGY_LIBRARY.values())


def filter_strategies(
    directional_bias: Optional[str] = None,
    volatility_regime: Optional[str] = None,
    min_conviction: Optional[str] = None
) -> List[StrategyBlueprint]:
    """
    Filter strategies by criteria

    Args:
        directional_bias: "bullish", "bearish", or "neutral"
        volatility_regime: "high_iv", "low_iv", or "any"
        min_conviction: "low", "medium", or "high"

    Returns:
        List of matching strategies
    """
    results = []

    for strategy in STRATEGY_LIBRARY.values():
        # Filter by direction
        if directional_bias and strategy.directional_bias != directional_bias:
            continue

        # Filter by IV regime
        if volatility_regime and strategy.volatility_regime not in (volatility_regime, "any"):
            continue

        # Filter by conviction (hierarchical)
        if min_conviction:
            conviction_order = {"low": 0, "medium": 1, "high": 2}
            required = conviction_order.get(strategy.conviction_required, 0)
            provided = conviction_order.get(min_conviction, 0)
            if provided < required:
                continue

        results.append(strategy)

    return results
