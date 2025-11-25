"""
Portfolio-level Greeks aggregation

Calculates net Greeks across multiple positions for risk management.
"""

from typing import Dict, List
from src.models.schemas import TradeLeg


def aggregate_position_greeks(
    trade_legs: List[TradeLeg],
    spot_price: float
) -> Dict[str, float]:
    """
    Calculate net Greeks for a single position (multi-leg strategy)

    Args:
        trade_legs: List of TradeLeg objects
        spot_price: Current spot price

    Returns:
        Dictionary with net Greeks
    """
    net_delta = 0.0
    net_gamma = 0.0
    net_theta = 0.0
    net_vega = 0.0
    net_rho = 0.0

    for leg in trade_legs:
        greeks = leg.key_greeks_at_selection or {}

        # Multiplier: BUY = +1, SELL = -1
        multiplier = 1.0 if leg.action.upper() == "BUY" else -1.0
        quantity = leg.quantity

        net_delta += greeks.get("delta", 0.0) * multiplier * quantity
        net_gamma += greeks.get("gamma", 0.0) * multiplier * quantity
        net_theta += greeks.get("theta", 0.0) * multiplier * quantity
        net_vega += greeks.get("vega", 0.0) * multiplier * quantity
        net_rho += greeks.get("rho", 0.0) * multiplier * quantity

    # Calculate dollar values (options contract multiplier = 100)
    OPTIONS_MULTIPLIER = 100

    return {
        "net_delta": round(net_delta, 4),
        "net_gamma": round(net_gamma, 4),
        "net_theta": round(net_theta, 4),
        "net_vega": round(net_vega, 4),
        "net_rho": round(net_rho, 4),
        # Dollar exposures
        "delta_dollars": round(net_delta * spot_price * OPTIONS_MULTIPLIER, 2),
        "theta_daily_dollars": round(net_theta * OPTIONS_MULTIPLIER, 2),
        "vega_dollars": round(net_vega * OPTIONS_MULTIPLIER, 2),
    }


def aggregate_portfolio_greeks(
    positions: List[Dict],
    spot_prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate net Greeks across entire portfolio

    Args:
        positions: List of position dicts with 'ticker', 'trade_legs', 'spot_price'
        spot_prices: Dict mapping ticker -> current spot price

    Returns:
        Portfolio-level Greeks
    """
    total_delta = 0.0
    total_gamma = 0.0
    total_theta = 0.0
    total_vega = 0.0
    total_rho = 0.0
    total_delta_dollars = 0.0

    for position in positions:
        ticker = position.get("ticker")
        trade_legs = position.get("trade_legs", [])
        spot_price = spot_prices.get(ticker, position.get("spot_price", 0))

        position_greeks = aggregate_position_greeks(trade_legs, spot_price)

        total_delta += position_greeks["net_delta"]
        total_gamma += position_greeks["net_gamma"]
        total_theta += position_greeks["net_theta"]
        total_vega += position_greeks["net_vega"]
        total_rho += position_greeks["net_rho"]
        total_delta_dollars += position_greeks["delta_dollars"]

    return {
        "portfolio_delta": round(total_delta, 4),
        "portfolio_gamma": round(total_gamma, 4),
        "portfolio_theta": round(total_theta, 4),
        "portfolio_vega": round(total_vega, 4),
        "portfolio_rho": round(total_rho, 4),
        "portfolio_delta_dollars": round(total_delta_dollars, 2),
        "expected_daily_decay": round(total_theta * 100, 2),  # Dollars
    }


def assess_greeks_risk(greeks: Dict[str, float], days_to_expiration: int) -> Dict[str, str]:
    """
    Assess risk level based on Greeks

    Args:
        greeks: Position or portfolio Greeks
        days_to_expiration: Days until nearest expiration

    Returns:
        Risk assessment dict
    """
    warnings = []

    # Delta Risk (directional exposure)
    delta = abs(greeks.get("net_delta", 0) or greeks.get("portfolio_delta", 0))
    if delta > 100:
        warnings.append(f"HIGH_DELTA: {delta:.2f} contracts of directional risk")
    elif delta > 50:
        warnings.append(f"MODERATE_DELTA: {delta:.2f} contracts of directional risk")

    # Gamma Risk (especially near expiration)
    gamma = abs(greeks.get("net_gamma", 0) or greeks.get("portfolio_gamma", 0))
    if days_to_expiration <= 7 and gamma > 10:
        warnings.append(f"CRITICAL_GAMMA: {gamma:.4f} with {days_to_expiration} DTE - CLOSE POSITIONS")
    elif days_to_expiration <= 14 and gamma > 20:
        warnings.append(f"HIGH_GAMMA: {gamma:.4f} with {days_to_expiration} DTE - Monitor closely")

    # Theta Exposure (time decay)
    theta = greeks.get("net_theta", 0) or greeks.get("portfolio_theta", 0)
    if theta < -50:
        warnings.append(f"HIGH_THETA_BURN: ${abs(theta)*100:.2f}/day decay - Debit spread bleeding")
    elif theta > 50:
        warnings.append(f"HIGH_THETA_COLLECTION: ${theta*100:.2f}/day income - Credit strategy")

    # Vega Risk (IV sensitivity)
    vega = abs(greeks.get("net_vega", 0) or greeks.get("portfolio_vega", 0))
    if vega > 100:
        warnings.append(f"HIGH_VEGA: {vega:.2f} - Highly sensitive to IV changes")

    return {
        "risk_level": "CRITICAL" if any("CRITICAL" in w for w in warnings) else
                     "HIGH" if any("HIGH" in w for w in warnings) else
                     "MODERATE" if any("MODERATE" in w for w in warnings) else
                     "LOW",
        "warnings": warnings,
    }
