"""
Black-Scholes option pricing and Greeks calculation

Provides forward-looking option pricing independent of market data providers.
"""

from __future__ import annotations

import numpy as np
from datetime import datetime
from typing import Dict, Literal


def black_scholes_price(
    S: float,  # Current stock price
    K: float,  # Strike price
    T: float,  # Time to expiration (years)
    r: float,  # Risk-free rate (annualized)
    sigma: float,  # Implied volatility (annualized)
    option_type: Literal["call", "put"]
) -> float:
    """
    Calculate Black-Scholes option price

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (e.g., 0.05 for 5%)
        sigma: Implied volatility (e.g., 0.30 for 30%)
        option_type: "call" or "put"

    Returns:
        Option price (premium)

    Example:
        >>> price = black_scholes_price(100, 105, 0.25, 0.05, 0.25, "call")
        >>> print(f"Call option price: ${price:.2f}")
    """
    if T <= 0:
        # At expiration, intrinsic value only
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    if sigma <= 0:
        # Zero volatility edge case
        if option_type == "call":
            return max(S * np.exp(-r * T) - K, 0.0)
        else:
            return max(K - S * np.exp(-r * T), 0.0)

    try:
        # Standard Black-Scholes formula
        from scipy.stats import norm

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.0)  # Ensure non-negative

    except Exception:
        # Fallback to intrinsic value if calculation fails
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"]
) -> Dict[str, float]:
    """
    Calculate all Greeks using Black-Scholes model

    Greeks measure the sensitivity of option price to various factors:
    - Delta: Price sensitivity to underlying ($1 stock move)
    - Gamma: Rate of change of delta
    - Theta: Time decay (per day)
    - Vega: Volatility sensitivity (per 1% IV change)
    - Rho: Interest rate sensitivity (per 1% rate change)

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        sigma: Implied volatility
        option_type: "call" or "put"

    Returns:
        Dictionary with all Greeks:
        {
            "delta": float,
            "gamma": float,
            "theta": float (per day),
            "vega": float (per 1% IV change),
            "rho": float (per 1% rate change)
        }
    """
    if T <= 0:
        # At expiration, Greeks are discontinuous
        if option_type == "call":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0

        return {
            "delta": delta,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }

    if sigma <= 0:
        # Zero volatility edge case
        return {
            "delta": 1.0 if option_type == "call" and S > K else 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }

    try:
        from scipy.stats import norm

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        exp_neg_rT = np.exp(-r * T)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # Delta
        if option_type == "call":
            delta = cdf_d1
        else:
            delta = cdf_d1 - 1.0

        # Gamma (same for calls and puts)
        gamma = pdf_d1 / (S * sigma * sqrt_T)

        # Theta (per day, divide by 365)
        if option_type == "call":
            theta = (
                -S * pdf_d1 * sigma / (2 * sqrt_T)
                - r * K * exp_neg_rT * cdf_d2
            ) / 365.0
        else:
            theta = (
                -S * pdf_d1 * sigma / (2 * sqrt_T)
                + r * K * exp_neg_rT * norm.cdf(-d2)
            ) / 365.0

        # Vega (per 1% change in IV, e.g., 25% to 26%)
        vega = S * pdf_d1 * sqrt_T / 100.0

        # Rho (per 1% change in interest rate)
        if option_type == "call":
            rho = K * T * exp_neg_rT * cdf_d2 / 100.0
        else:
            rho = -K * T * exp_neg_rT * norm.cdf(-d2) / 100.0

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
            "rho": float(rho)
        }

    except Exception:
        # Fallback to zero Greeks
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0
        }


def calculate_time_to_expiration(expiration_date: str) -> float:
    """
    Convert expiration date string to years

    Args:
        expiration_date: ISO format date string (e.g., "2025-12-19")

    Returns:
        Time to expiration in years (e.g., 0.25 for 3 months)
    """
    try:
        exp_dt = datetime.fromisoformat(expiration_date.replace("Z", "+00:00"))
        now = datetime.now()

        # Handle timezone-aware vs naive
        if exp_dt.tzinfo is not None and now.tzinfo is None:
            from datetime import timezone
            now = now.replace(tzinfo=timezone.utc)
        elif exp_dt.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)

        days_to_exp = (exp_dt - now).days
        return max(days_to_exp / 365.0, 0.0)

    except Exception:
        return 0.0
