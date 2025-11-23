"""
Options chain enrichment with computed Greeks

Enhances options chain data with Black-Scholes calculated Greeks.
"""

from __future__ import annotations

from typing import Dict, List, Any
import pandas as pd
import logging

from src.pricing.black_scholes import calculate_greeks, black_scholes_price, calculate_time_to_expiration

logger = logging.getLogger(__name__)


def enrich_options_chain_with_greeks(
    options_chain: List[Dict[str, Any]],
    spot_price: float,
    risk_free_rate: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Enrich options chain with Black-Scholes calculated Greeks

    For each option in the chain:
    1. Use market IV from yfinance if available
    2. Calculate all Greeks using Black-Scholes
    3. Add computed_greeks field
    4. Calculate theoretical price

    Args:
        options_chain: List of expiration groups from fetcher
        spot_price: Current underlying price
        risk_free_rate: Risk-free rate for pricing (default 5%)

    Returns:
        Enhanced options chain with computed_greeks field added to each option

    Example:
        >>> chain = fetch_options_chain("AAPL")
        >>> enriched = enrich_options_chain_with_greeks(chain, 175.50, 0.05)
        >>> # Access computed Greeks:
        >>> call = enriched[0]["calls"].iloc[0]
        >>> print(call["computed_greeks"]["delta"])
    """
    enriched_chain = []

    for expiration_group in options_chain:
        try:
            expiration_date = expiration_group.get("expiration")
            if not expiration_date:
                logger.warning("Skipping expiration group without date")
                continue

            # Calculate time to expiration
            T = calculate_time_to_expiration(expiration_date)

            if T <= 0:
                logger.warning(f"Expiration {expiration_date} already passed, T={T}")
                # Still include but with zero time
                T = 0.0

            # Enrich calls and puts
            calls_df = expiration_group.get("calls")
            puts_df = expiration_group.get("puts")

            enriched_calls = _enrich_option_side(
                calls_df, spot_price, T, risk_free_rate, "call"
            )
            enriched_puts = _enrich_option_side(
                puts_df, spot_price, T, risk_free_rate, "put"
            )

            enriched_chain.append({
                "expiration": expiration_date,
                "calls": enriched_calls,
                "puts": enriched_puts,
                "time_to_expiration_years": T,
                "risk_free_rate": risk_free_rate
            })

            logger.info(f"Enriched expiration {expiration_date} (T={T:.3f} years)")

        except Exception as e:
            logger.error(f"Error enriching expiration group: {e}")
            # Include original group without enrichment
            enriched_chain.append(expiration_group)

    return enriched_chain


def _enrich_option_side(
    df: pd.DataFrame,
    S: float,
    T: float,
    r: float,
    option_type: str
) -> pd.DataFrame:
    """
    Add computed_greeks column to options dataframe

    Args:
        df: Options dataframe (calls or puts)
        S: Spot price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: "call" or "put"

    Returns:
        DataFrame with added 'computed_greeks' column
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    computed_greeks_list = []

    for idx, row in df.iterrows():
        try:
            K = row["strike"]

            # Use market IV if available, otherwise fallback
            sigma = row.get("impliedVolatility")

            if sigma is None or pd.isna(sigma) or sigma <= 0:
                # Fallback to 30% IV
                sigma = 0.30
                iv_source = "fallback"
            else:
                iv_source = "market"

            # Ensure sigma is reasonable (1% to 200%)
            sigma = max(min(float(sigma), 2.0), 0.01)

            # Calculate Greeks
            greeks = calculate_greeks(S, K, T, r, sigma, option_type)

            # Calculate theoretical price
            theoretical_price = black_scholes_price(S, K, T, r, sigma, option_type)

            # Compile results
            greeks["theoretical_price"] = theoretical_price
            greeks["iv_used"] = sigma
            greeks["iv_source"] = iv_source

            computed_greeks_list.append(greeks)

        except Exception as e:
            logger.warning(f"Error computing Greeks for strike {row.get('strike', 'N/A')}: {e}")
            # Append empty Greeks
            computed_greeks_list.append({
                "delta": 0.0,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theoretical_price": 0.0,
                "iv_used": 0.30,
                "iv_source": "error"
            })

    df["computed_greeks"] = computed_greeks_list

    return df


def get_atm_iv(options_chain: List[Dict[str, Any]], spot_price: float) -> float:
    """
    Extract ATM implied volatility from options chain

    Useful for getting a representative IV for the underlying.

    Args:
        options_chain: Enriched options chain
        spot_price: Current underlying price

    Returns:
        ATM implied volatility (average of ATM call and put)
    """
    if not options_chain:
        return 0.30  # Default

    try:
        # Use front month
        front_month = options_chain[0]
        calls_df = front_month.get("calls")
        puts_df = front_month.get("puts")

        atm_ivs = []

        # Find ATM call
        if calls_df is not None and not calls_df.empty:
            calls_df = calls_df.copy()
            calls_df["distance"] = (calls_df["strike"] - spot_price).abs()
            atm_call = calls_df.nsmallest(1, "distance").iloc[0]
            call_iv = atm_call.get("impliedVolatility")
            if call_iv and not pd.isna(call_iv) and call_iv > 0:
                atm_ivs.append(call_iv)

        # Find ATM put
        if puts_df is not None and not puts_df.empty:
            puts_df = puts_df.copy()
            puts_df["distance"] = (puts_df["strike"] - spot_price).abs()
            atm_put = puts_df.nsmallest(1, "distance").iloc[0]
            put_iv = atm_put.get("impliedVolatility")
            if put_iv and not pd.isna(put_iv) and put_iv > 0:
                atm_ivs.append(put_iv)

        if atm_ivs:
            return float(sum(atm_ivs) / len(atm_ivs))

    except Exception as e:
        logger.warning(f"Error extracting ATM IV: {e}")

    return 0.30  # Fallback
