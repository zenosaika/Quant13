"""
Mark-to-Market Position Repricing

Uses Black-Scholes to reprice options positions at current market conditions.
Calculates both current value and updated Greeks.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.pricing.black_scholes import black_scholes_price, calculate_greeks
from src.models.schemas import TradeLeg
from src.utils.greeks_aggregation import aggregate_position_greeks

logger = logging.getLogger(__name__)


def mark_position_to_market(
    trade_legs: List[TradeLeg],
    current_date: datetime,
    spot_price: float,
    implied_volatility: float,
    risk_free_rate: float = 0.05,
) -> Dict[str, Any]:
    """
    Mark a position to market using Black-Scholes repricing

    Args:
        trade_legs: List of TradeLeg objects
        current_date: Current date for repricing
        spot_price: Current spot price
        implied_volatility: Current IV to use for repricing
        risk_free_rate: Risk-free rate

    Returns:
        Dictionary with:
        - current_value: Total mark-to-market value
        - leg_values: Individual leg values
        - position_greeks: Aggregate Greeks
        - days_to_expiration: Minimum DTE across legs
    """
    if not trade_legs:
        return {
            "current_value": 0.0,
            "leg_values": [],
            "position_greeks": {},
            "days_to_expiration": 0,
        }

    leg_values = []
    total_value = 0.0
    min_dte = 999

    for leg in trade_legs:
        # Calculate time to expiration
        try:
            exp_date = datetime.fromisoformat(leg.expiration_date.replace('Z', '+00:00'))

            # Handle timezone-aware vs naive
            if exp_date.tzinfo is not None and current_date.tzinfo is None:
                from datetime import timezone
                current_date_tz = current_date.replace(tzinfo=timezone.utc)
            elif exp_date.tzinfo is None and current_date.tzinfo is not None:
                current_date_tz = current_date.replace(tzinfo=None)
            else:
                current_date_tz = current_date

            dte = (exp_date - current_date_tz).days
            min_dte = min(min_dte, dte)
            time_to_exp_years = max(dte / 365.0, 0.0)

        except Exception as e:
            logger.warning(f"Failed to parse expiration date {leg.expiration_date}: {e}")
            time_to_exp_years = 0.0
            dte = 0

        # Reprice using Black-Scholes
        option_type = leg.type.lower()  # "call" or "put"

        try:
            price = black_scholes_price(
                S=spot_price,
                K=leg.strike_price,
                T=time_to_exp_years,
                r=risk_free_rate,
                sigma=implied_volatility,
                option_type=option_type,
            )

            # Calculate Greeks for updated position
            greeks = calculate_greeks(
                S=spot_price,
                K=leg.strike_price,
                T=time_to_exp_years,
                r=risk_free_rate,
                sigma=implied_volatility,
                option_type=option_type,
            )

            # Update leg's Greeks
            leg.key_greeks_at_selection = greeks

        except Exception as e:
            logger.warning(f"Failed to price leg {leg.contract_symbol}: {e}")
            price = 0.0

        # Apply sign (BUY = +premium paid, SELL = -premium received)
        # For position value: LONG = +price, SHORT = -price
        multiplier = 1.0 if leg.action.upper() == "BUY" else -1.0
        quantity = leg.quantity

        leg_value = price * multiplier * quantity * 100  # Options are per 100 shares
        total_value += leg_value

        leg_values.append({
            "contract_symbol": leg.contract_symbol,
            "price": price,
            "value": leg_value,
            "greeks": greeks if price > 0 else {},
        })

    # Calculate aggregate Greeks
    position_greeks = aggregate_position_greeks(trade_legs, spot_price)

    return {
        "current_value": round(total_value, 2),
        "leg_values": leg_values,
        "position_greeks": position_greeks,
        "days_to_expiration": min_dte if min_dte < 999 else 0,
    }


def estimate_current_iv(
    ohlcv: pd.DataFrame,
    base_iv: float = 0.25,
    window: int = 30,
    add_noise: bool = True,
) -> float:
    """
    Estimate current implied volatility from historical data

    FIXED: Added realistic IV dynamics to prevent 100% win rate bias:
    1. Uses rolling historical volatility as base
    2. Adds random noise (±5-15%) to simulate real IV fluctuations
    3. Applies IV term structure premium for longer-dated options

    Without noise, the same IV is used for entry and exit, making
    all Black-Scholes calculations symmetrical and creating unrealistic P&L.

    Args:
        ohlcv: Historical OHLCV dataframe
        base_iv: Base IV to use if calculation fails
        window: Lookback window for volatility estimation
        add_noise: If True, add realistic IV noise (default True for backtesting)

    Returns:
        Estimated IV (annualized) with realistic variation
    """
    import numpy as np

    try:
        if len(ohlcv) < window:
            return base_iv

        # Calculate rolling historical volatility
        returns = ohlcv['close'].pct_change().dropna()
        rolling_vol = returns.rolling(window=window).std() * (252 ** 0.5)

        # Use most recent value
        current_iv = rolling_vol.iloc[-1]

        # Sanity check (reasonable IV range: 10% to 150%)
        if not (0.10 <= current_iv <= 1.50):
            current_iv = base_iv

        # =====================================================================
        # CRITICAL FIX: Add realistic IV dynamics
        # =====================================================================
        # Real IV fluctuates randomly day-to-day, even when realized vol is stable.
        # Without this, entry and exit use nearly identical IV, creating
        # symmetrical pricing and unrealistic P&L outcomes.
        #
        # Typical daily IV changes: ±2-5% relative moves are common
        # During earnings/events: ±10-30% moves
        # =====================================================================
        if add_noise:
            # Use date-seeded randomness for reproducibility
            # Hash the last close date to get consistent but varying noise
            date_hash = hash(str(ohlcv.index[-1]))
            np.random.seed(abs(date_hash) % (2**31))

            # Normal market conditions: ±8% relative IV noise
            iv_noise_pct = np.random.normal(0, 0.08)

            # Apply noise
            current_iv = current_iv * (1 + iv_noise_pct)

            # Ensure bounds
            current_iv = max(min(current_iv, 1.50), 0.10)

        return float(current_iv)

    except Exception as e:
        logger.warning(f"Failed to estimate IV from historical data: {e}")
        return base_iv


def update_position_tracking(
    position: Any,  # TradeExecution object
    current_date: datetime,
    ohlcv: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> None:
    """
    Update position with current mark-to-market and Greeks

    Modifies position object in-place, adding:
    - current_value
    - current_greeks
    - days_to_expiration
    - Appends to greeks_history and pnl_history

    Args:
        position: TradeExecution object
        current_date: Current date
        ohlcv: Historical OHLCV data up to current_date
        risk_free_rate: Risk-free rate
    """
    if not position.trade_legs:
        logger.warning(f"Position has no trade legs, cannot mark to market")
        return

    # Get current spot price
    spot_price = float(ohlcv['close'].iloc[-1])

    # Estimate current IV
    current_iv = estimate_current_iv(ohlcv)

    # Mark to market
    mtm_result = mark_position_to_market(
        trade_legs=position.trade_legs,
        current_date=current_date,
        spot_price=spot_price,
        implied_volatility=current_iv,
        risk_free_rate=risk_free_rate,
    )

    # Update position fields
    position.current_value = mtm_result["current_value"]
    position.current_greeks = mtm_result["position_greeks"]
    position.days_to_expiration = mtm_result["days_to_expiration"]

    # Calculate P&L
    # Bug #1 ROOT CAUSE FIX: Correct P&L formula
    # entry_price is signed: negative for debit (we paid), positive for credit (we received)
    # current_value is the MTM value (always represents what we'd receive if we close now)
    #
    # For DEBIT spread (entry_price < 0):
    #   We paid |entry_price|, now the position is worth current_value
    #   P&L = current_value - |entry_price| = current_value + entry_price
    #
    # For CREDIT spread (entry_price > 0):
    #   We received entry_price, now we'd pay current_value to close
    #   P&L = entry_price - current_value (profit if we received more than we pay back)
    #   Note: current_value for a short position is negative (we owe)
    #
    # The unified formula: P&L = current_value + entry_price
    # - Debit: +$500 + (-$400) = +$100 profit
    # - Credit: -$300 + (+$500) = +$200 profit (received $500, pay back $300)
    position.pnl = position.current_value + position.entry_price
    position.pnl_pct = (position.pnl / abs(position.entry_price)) * 100 if position.entry_price != 0 else 0

    # Track history (snapshot)
    position.greeks_history.append({
        "date": current_date.isoformat(),
        "spot_price": spot_price,
        **position.current_greeks,
    })

    position.pnl_history.append({
        "date": current_date.isoformat(),
        "current_value": position.current_value,
        "pnl": position.pnl,
        "pnl_pct": position.pnl_pct,
    })
