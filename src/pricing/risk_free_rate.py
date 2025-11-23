"""
Risk-free rate fetching from Treasury yields

Uses yfinance to fetch current Treasury rates that match option duration.
"""

from __future__ import annotations

import yfinance as yf
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def fetch_risk_free_rate(duration_years: float = 0.25) -> float:
    """
    Fetch risk-free rate from Treasury yields

    Selects appropriate Treasury maturity based on option duration:
    - < 1 year: 13-week Treasury Bill (^IRX)
    - >= 1 year: 10-year Treasury Note (^TNX)

    Args:
        duration_years: Option duration to match Treasury maturity
                       (e.g., 0.25 for 3 months, 1.0 for 1 year)

    Returns:
        Annual risk-free rate as decimal (e.g., 0.05 for 5%)

    Example:
        >>> rate = fetch_risk_free_rate(0.25)  # 3-month option
        >>> print(f"Risk-free rate: {rate*100:.2f}%")
    """
    try:
        if duration_years < 1.0:
            # 13-week Treasury Bill rate (^IRX)
            # Note: ^IRX reports annualized yield as percentage
            ticker = yf.Ticker("^IRX")
            data = ticker.history(period="5d")  # Get last 5 days for reliability

            if not data.empty:
                # Use most recent close
                rate_percent = data["Close"].iloc[-1]
                rate = rate_percent / 100.0  # Convert percentage to decimal
                logger.info(f"Fetched 13-week T-Bill rate: {rate*100:.2f}%")
                return float(rate)

        else:
            # 10-year Treasury Note rate (^TNX)
            ticker = yf.Ticker("^TNX")
            data = ticker.history(period="5d")

            if not data.empty:
                rate_percent = data["Close"].iloc[-1]
                rate = rate_percent / 100.0
                logger.info(f"Fetched 10-year T-Note rate: {rate*100:.2f}%")
                return float(rate)

    except Exception as e:
        logger.warning(f"Failed to fetch risk-free rate: {e}")

    # Fallback to conservative estimate (5%)
    logger.info("Using fallback risk-free rate: 5.00%")
    return 0.05


def fetch_risk_free_rate_with_fallback(duration_years: float = 0.25, fallback: float = 0.05) -> float:
    """
    Fetch risk-free rate with custom fallback

    Args:
        duration_years: Option duration in years
        fallback: Fallback rate if fetch fails (default 5%)

    Returns:
        Risk-free rate as decimal
    """
    try:
        rate = fetch_risk_free_rate(duration_years)
        # Sanity check: rate should be between 0% and 20%
        if 0.0 <= rate <= 0.20:
            return rate
        else:
            logger.warning(f"Fetched rate {rate*100:.2f}% outside reasonable range, using fallback")
            return fallback
    except:
        return fallback
