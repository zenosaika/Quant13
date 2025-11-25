"""
Synthetic Sentiment for Backtesting

Since historical news is not available via yfinance, we generate synthetic sentiment
based on observable historical data:
1. Price momentum (returns)
2. Volatility regime
3. Volume patterns
4. Technical indicators (RSI, MACD)

This provides a reasonable proxy for market sentiment during backtesting.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def generate_synthetic_sentiment(
    ohlcv: pd.DataFrame,
    lookback_window: int = 20,
) -> Dict[str, Any]:
    """
    Generate synthetic sentiment from price action and technical indicators

    Args:
        ohlcv: Historical OHLCV data up to current backtest date
        lookback_window: Window for calculations (default: 20 days)

    Returns:
        Dictionary with:
        - sentiment_score: Float from -1.0 (bearish) to 1.0 (bullish)
        - confidence: Float from 0.0 (low) to 1.0 (high)
        - components: Dict with individual component scores
        - reasoning: String explaining the sentiment
    """
    if len(ohlcv) < lookback_window:
        logger.warning(f"Insufficient data for sentiment: {len(ohlcv)} < {lookback_window}")
        return _neutral_sentiment("Insufficient historical data")

    # Component 1: Price Momentum (40% weight)
    momentum_score = _calculate_momentum_score(ohlcv, lookback_window)

    # Component 2: Volatility Regime (20% weight)
    volatility_score = _calculate_volatility_score(ohlcv, lookback_window)

    # Component 3: Volume Pattern (20% weight)
    volume_score = _calculate_volume_score(ohlcv, lookback_window)

    # Component 4: Technical Indicators (20% weight)
    technical_score = _calculate_technical_score(ohlcv, lookback_window)

    # Weighted average
    weights = {
        "momentum": 0.40,
        "volatility": 0.20,
        "volume": 0.20,
        "technical": 0.20,
    }

    sentiment_score = (
        momentum_score * weights["momentum"] +
        volatility_score * weights["volatility"] +
        volume_score * weights["volume"] +
        technical_score * weights["technical"]
    )

    # Calculate confidence based on alignment of components
    component_scores = [momentum_score, volatility_score, volume_score, technical_score]
    alignment = 1.0 - np.std(component_scores)  # High std = low confidence
    confidence = max(0.0, min(1.0, alignment))

    # Generate reasoning
    reasoning = _generate_reasoning(
        sentiment_score,
        {
            "momentum": momentum_score,
            "volatility": volatility_score,
            "volume": volume_score,
            "technical": technical_score,
        }
    )

    return {
        "sentiment_score": round(sentiment_score, 3),
        "confidence": round(confidence, 3),
        "components": {
            "momentum": round(momentum_score, 3),
            "volatility": round(volatility_score, 3),
            "volume": round(volume_score, 3),
            "technical": round(technical_score, 3),
        },
        "reasoning": reasoning,
        "source": "synthetic",
    }


def _calculate_momentum_score(ohlcv: pd.DataFrame, window: int) -> float:
    """
    Calculate momentum score from returns

    Returns: -1.0 (strong downtrend) to 1.0 (strong uptrend)
    """
    recent_data = ohlcv.tail(window)
    returns = recent_data['close'].pct_change().dropna()

    # Cumulative return over window
    cumulative_return = (1 + returns).prod() - 1

    # Normalize to [-1, 1] range (cap at ±30%)
    momentum_score = np.clip(cumulative_return / 0.30, -1.0, 1.0)

    return float(momentum_score)


def _calculate_volatility_score(ohlcv: pd.DataFrame, window: int) -> float:
    """
    Calculate volatility regime score

    Low volatility = positive (calm markets, bullish)
    High volatility = negative (stressed markets, bearish)

    Returns: -1.0 (very high vol) to 1.0 (very low vol)
    """
    recent_data = ohlcv.tail(window)
    returns = recent_data['close'].pct_change().dropna()

    # Calculate realized volatility
    realized_vol = returns.std() * np.sqrt(252)  # Annualized

    # Compare to longer-term volatility
    if len(ohlcv) >= window * 2:
        long_term_vol = ohlcv['close'].pct_change().tail(window * 2).std() * np.sqrt(252)
    else:
        long_term_vol = realized_vol

    # Relative volatility (-1 = much higher, 1 = much lower)
    if long_term_vol > 0:
        vol_ratio = (long_term_vol - realized_vol) / long_term_vol
        vol_score = np.clip(vol_ratio * 2, -1.0, 1.0)  # Scale to [-1, 1]
    else:
        vol_score = 0.0

    return float(vol_score)


def _calculate_volume_score(ohlcv: pd.DataFrame, window: int) -> float:
    """
    Calculate volume pattern score

    Rising volume on up days = bullish
    Rising volume on down days = bearish

    Returns: -1.0 (bearish volume) to 1.0 (bullish volume)
    """
    recent_data = ohlcv.tail(window)

    if 'volume' not in recent_data.columns or recent_data['volume'].sum() == 0:
        return 0.0

    returns = recent_data['close'].pct_change()
    volume = recent_data['volume']

    # Calculate average volume on up days vs down days
    up_days = returns > 0
    down_days = returns < 0

    if up_days.sum() > 0 and down_days.sum() > 0:
        avg_vol_up = volume[up_days].mean()
        avg_vol_down = volume[down_days].mean()

        # Relative volume (bullish if more volume on up days)
        if avg_vol_up + avg_vol_down > 0:
            vol_score = (avg_vol_up - avg_vol_down) / (avg_vol_up + avg_vol_down)
        else:
            vol_score = 0.0
    else:
        vol_score = 0.0

    return float(np.clip(vol_score, -1.0, 1.0))


def _calculate_technical_score(ohlcv: pd.DataFrame, window: int) -> float:
    """
    Calculate technical indicator score (RSI + MACD)

    Returns: -1.0 (oversold/bearish) to 1.0 (overbought/bullish)
    """
    recent_data = ohlcv.tail(window * 2)  # Need more data for indicators

    # RSI
    rsi = _calculate_rsi(recent_data['close'], period=14)
    current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50

    # Normalize RSI to [-1, 1]: 30=oversold=-0.5, 70=overbought=+0.5, 50=neutral=0
    rsi_score = (current_rsi - 50) / 50  # Range: -1 to 1

    # MACD
    macd_line, signal_line = _calculate_macd(recent_data['close'])
    if len(macd_line) > 0 and len(signal_line) > 0:
        macd_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
        # Normalize MACD (approximate range ±2)
        macd_score = np.clip(macd_diff / 2.0, -1.0, 1.0)
    else:
        macd_score = 0.0

    # Average of RSI and MACD
    technical_score = (rsi_score * 0.5 + macd_score * 0.5)

    return float(np.clip(technical_score, -1.0, 1.0))


def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    return macd_line, signal_line


def _generate_reasoning(sentiment_score: float, components: Dict[str, float]) -> str:
    """Generate human-readable reasoning for sentiment"""
    if sentiment_score > 0.5:
        direction = "Strongly Bullish"
    elif sentiment_score > 0.2:
        direction = "Moderately Bullish"
    elif sentiment_score > -0.2:
        direction = "Neutral"
    elif sentiment_score > -0.5:
        direction = "Moderately Bearish"
    else:
        direction = "Strongly Bearish"

    # Identify strongest component
    strongest = max(components.items(), key=lambda x: abs(x[1]))

    reasoning = (
        f"{direction} sentiment based on synthetic analysis. "
        f"Primary driver: {strongest[0]} ({strongest[1]:+.2f}). "
        f"Components - Momentum: {components['momentum']:+.2f}, "
        f"Volatility: {components['volatility']:+.2f}, "
        f"Volume: {components['volume']:+.2f}, "
        f"Technical: {components['technical']:+.2f}."
    )

    return reasoning


def _neutral_sentiment(reason: str) -> Dict[str, Any]:
    """Return neutral sentiment with reason"""
    return {
        "sentiment_score": 0.0,
        "confidence": 0.5,
        "components": {
            "momentum": 0.0,
            "volatility": 0.0,
            "volume": 0.0,
            "technical": 0.0,
        },
        "reasoning": f"Neutral (default): {reason}",
        "source": "synthetic",
    }
