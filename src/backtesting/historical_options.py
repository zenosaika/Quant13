"""
Generate historical options chains using Black-Scholes model

Since yfinance doesn't provide historical options data, we reconstruct
options chains at historical dates using:
- Historical stock price (from OHLCV)
- Estimated historical IV (from historical returns volatility)
- Risk-free rate
- Standard expiration dates (weekly/monthly)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.pricing.black_scholes import black_scholes_price, calculate_greeks

logger = logging.getLogger(__name__)


def _get_skew_factor_for_ticker(ticker: str = None) -> float:
    """
    FIXED: Determine appropriate volatility skew based on ticker type

    Different asset classes have different skew characteristics:
    - Index ETFs (SPY, SPX, QQQ, IWM, DIA): 15-20% skew (high put demand)
    - Individual stocks: 8-12% skew (moderate skew)
    - Volatility products (VXX, UVXY): 25%+ skew (extreme)

    Args:
        ticker: Stock ticker symbol (optional)

    Returns:
        Appropriate skew factor
    """
    if ticker is None:
        return 0.10  # Default: individual stock

    ticker_upper = ticker.upper()

    # Index ETFs and major indices (higher skew)
    index_tickers = {'SPY', 'SPX', 'QQQ', 'IWM', 'DIA', 'MDY', 'EEM', 'EFA'}
    if ticker_upper in index_tickers:
        return 0.18  # 18% skew for indices

    # Volatility products (extreme skew)
    vol_tickers = {'VXX', 'UVXY', 'VIXY', 'SVXY'}
    if ticker_upper in vol_tickers:
        return 0.30  # 30% skew for vol products

    # Default: individual stock (moderate skew)
    return 0.10  # 10% skew for stocks


def _calculate_skewed_volatility(
    strike: float,
    spot_price: float,
    base_volatility: float,
    skew_factor: float,
    option_type: str
) -> float:
    """
    Apply volatility skew based on moneyness.

    Real markets exhibit "volatility smile/skew" where:
    - OTM puts (strike < spot) have higher IV
    - ATM options have baseline IV
    - OTM calls (strike > spot) may have slightly elevated IV

    Skew calibration:
    - Individual stocks: 5-10% skew per 10% moneyness
    - Index/ETF (SPX, SPY, QQQ): 15-20% skew per 10% moneyness

    Args:
        strike: Strike price
        spot_price: Current spot price
        base_volatility: Base/ATM implied volatility
        skew_factor: Skew strength parameter (0.10 for stocks, 0.15-0.20 for indices)
        option_type: "call" or "put"

    Returns:
        Adjusted volatility for this strike
    """
    # Calculate moneyness (negative for ITM, positive for OTM)
    moneyness = (strike - spot_price) / spot_price

    if option_type == "put":
        # Puts: Higher IV for lower strikes (OTM puts)
        # Negative moneyness = ITM put (strike > spot) = lower IV
        # Positive moneyness = OTM put (strike < spot would give negative moneyness for puts)
        # Actually for puts: strike < spot = OTM, we want higher IV there
        # So we use: -moneyness (when strike < spot, moneyness < 0, -moneyness > 0)
        skew_adjustment = -moneyness * skew_factor
    else:
        # Calls: Slight IV increase for far OTM calls (volatility smile effect)
        # Positive moneyness = OTM call (strike > spot) = slight increase
        skew_adjustment = abs(moneyness) * skew_factor * 0.3  # Weaker effect for calls

    # Apply adjustment (ensure non-negative)
    adjusted_vol = base_volatility * (1.0 + skew_adjustment)

    # Ensure reasonable bounds
    adjusted_vol = max(min(adjusted_vol, base_volatility * 2.0), base_volatility * 0.5)

    return adjusted_vol


def generate_historical_options_chain(
    historical_date: datetime,
    spot_price: float,
    historical_volatility: float,
    risk_free_rate: float = 0.05,
    num_strikes: int = 20,
    num_expirations: int = 2,
    apply_skew: bool = True,
    skew_factor: float = None,
    ticker: str = None,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic options chain for a historical date using Black-Scholes

    Args:
        historical_date: Date for which to generate options chain
        spot_price: Stock price on that date
        historical_volatility: Estimated IV based on historical returns (annualized)
        risk_free_rate: Risk-free rate
        num_strikes: Number of strikes to generate (centered around ATM)
        num_expirations: Number of expiration dates to generate
        apply_skew: If True, apply volatility skew (OTM puts more expensive)
        skew_factor: Controls strength of skew effect (auto-detected from ticker if None)
        ticker: Stock ticker (used for skew calibration)

    Returns:
        List of expiration groups similar to fetch_options_chain format
    """
    # FIXED: Auto-detect skew factor based on ticker type
    if skew_factor is None:
        skew_factor = _get_skew_factor_for_ticker(ticker)
        logger.info(f"Auto-detected skew factor: {skew_factor:.2f} for ticker={ticker}")
    options_chain = []

    # Generate expiration dates (next few Fridays)
    expiration_dates = _generate_expiration_dates(historical_date, num_expirations)

    # Generate strike prices around spot
    strikes = _generate_strikes(spot_price, num_strikes)

    for exp_date in expiration_dates:
        # Calculate time to expiration
        T = max((exp_date - historical_date).days / 365.0, 0.001)

        calls_data = []
        puts_data = []

        for strike in strikes:
            # Apply volatility skew if enabled
            if apply_skew:
                call_vol = _calculate_skewed_volatility(
                    strike, spot_price, historical_volatility, skew_factor, "call"
                )
                put_vol = _calculate_skewed_volatility(
                    strike, spot_price, historical_volatility, skew_factor, "put"
                )
            else:
                call_vol = historical_volatility
                put_vol = historical_volatility

            # Price calls and puts using Black-Scholes with skewed volatility
            call_price = black_scholes_price(
                S=spot_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=call_vol,
                option_type="call"
            )

            put_price = black_scholes_price(
                S=spot_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=put_vol,
                option_type="put"
            )

            # Calculate Greeks with skewed volatility
            call_greeks = calculate_greeks(
                S=spot_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=call_vol,
                option_type="call"
            )

            put_greeks = calculate_greeks(
                S=spot_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=put_vol,
                option_type="put"
            )

            # FIXED: Moneyness-dependent bid-ask spread (more realistic)
            # Calculate moneyness for each option
            call_moneyness = abs((strike - spot_price) / spot_price)  # 0 = ATM, >0 = OTM
            put_moneyness = abs((spot_price - strike) / spot_price)

            # Spread increases with moneyness (OTM options have wider spreads)
            # ATM (moneyness ~0%): 1.5% spread
            # 5% OTM: 3% spread
            # 10% OTM: 5% spread
            # 20%+ OTM: 10%+ spread
            call_spread_pct = 0.015 + (call_moneyness * 0.30)  # Scales from 1.5% to 10%+
            put_spread_pct = 0.015 + (put_moneyness * 0.30)

            # Apply spread with minimum of $0.05
            call_spread = max(call_price * call_spread_pct, 0.05)
            put_spread = max(put_price * put_spread_pct, 0.05)

            # Generate synthetic liquidity based on moneyness
            # ATM options have highest liquidity, decreases as we move OTM/ITM
            moneyness = abs(strike - spot_price) / spot_price

            if moneyness <= 0.05:  # Within 5% of ATM
                open_interest = 5000
                volume = 1000
            elif moneyness <= 0.10:  # 5-10% from ATM
                open_interest = 1000
                volume = 200
            else:  # >10% from ATM
                open_interest = 500
                volume = 100

            # Create call entry
            call_symbol = f"SYNTHETIC_C_{exp_date.strftime('%y%m%d')}_{int(strike*1000):08d}"
            calls_data.append({
                "contractSymbol": call_symbol,
                "strike": strike,
                "lastPrice": call_price,
                "bid": max(call_price - call_spread/2, 0.01),
                "ask": call_price + call_spread/2,
                "impliedVolatility": call_vol,  # Store skewed IV
                "computed_greeks": call_greeks,
                "openInterest": open_interest,  # Synthetic liquidity
                "volume": volume,  # Synthetic volume
            })

            # Create put entry
            put_symbol = f"SYNTHETIC_P_{exp_date.strftime('%y%m%d')}_{int(strike*1000):08d}"
            puts_data.append({
                "contractSymbol": put_symbol,
                "strike": strike,
                "lastPrice": put_price,
                "bid": max(put_price - put_spread/2, 0.01),
                "ask": put_price + put_spread/2,
                "impliedVolatility": put_vol,  # Store skewed IV
                "computed_greeks": put_greeks,
                "openInterest": open_interest,  # Synthetic liquidity
                "volume": volume,  # Synthetic volume
            })

        # Convert to DataFrames
        calls_df = pd.DataFrame(calls_data)
        puts_df = pd.DataFrame(puts_data)

        options_chain.append({
            "expiration": exp_date.strftime("%Y-%m-%d"),
            "calls": calls_df,
            "puts": puts_df,
            "time_to_expiration_years": T,
            "risk_free_rate": risk_free_rate,
        })

    logger.info(f"Generated {len(options_chain)} expirations with {num_strikes} strikes each")
    return options_chain


def _generate_expiration_dates(reference_date: datetime, num_expirations: int) -> List[datetime]:
    """
    Generate next N option expiration dates (Fridays) with minimum 21 DTE

    This enforces institutional-grade DTE requirements to avoid:
    - Excessive gamma risk (<14 DTE)
    - Poor liquidity (<7 DTE)
    - Unrealistic theta extraction patterns
    """
    MIN_DTE = 21  # Minimum days to expiration for realistic trading
    expirations = []

    # Start search at least MIN_DTE days in the future
    current = reference_date + timedelta(days=MIN_DTE)

    while len(expirations) < num_expirations:
        # Find next Friday
        days_until_friday = (4 - current.weekday()) % 7
        if days_until_friday == 0:
            days_until_friday = 7  # Skip today if it's Friday

        next_friday = current + timedelta(days=days_until_friday)

        # Double-check minimum DTE requirement
        days_to_expiry = (next_friday - reference_date).days
        if days_to_expiry >= MIN_DTE:
            expirations.append(next_friday)

        current = next_friday + timedelta(days=1)

    return expirations


def _generate_strikes(spot_price: float, num_strikes: int) -> List[float]:
    """
    Generate strike prices centered around spot

    Generates strikes from ~80% to ~120% of spot price
    """
    # Round spot to nearest $5 for cleaner strikes
    atm_strike = round(spot_price / 5) * 5

    # Generate strikes with $5 increments
    half_range = (num_strikes // 2)
    strikes = []

    for i in range(-half_range, half_range + 1):
        strike = atm_strike + (i * 5)
        if strike > 0:  # Ensure positive strikes
            strikes.append(float(strike))

    return sorted(strikes)


def estimate_historical_volatility(ohlcv: pd.DataFrame, window: int = 30) -> float:
    """
    Estimate implied volatility from historical returns

    Args:
        ohlcv: DataFrame with 'close' column
        window: Number of days to use for volatility calculation

    Returns:
        Annualized volatility estimate (e.g., 0.25 for 25%)
    """
    if "return" not in ohlcv.columns:
        # Calculate returns if not already present
        ohlcv = ohlcv.copy()
        ohlcv["return"] = ohlcv["close"].pct_change()

    # Use last N days
    recent_returns = ohlcv["return"].tail(window)

    # Calculate standard deviation and annualize
    daily_vol = recent_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)  # 252 trading days

    # Ensure reasonable bounds (5% to 200%)
    annualized_vol = max(min(annualized_vol, 2.0), 0.05)

    return float(annualized_vol)
