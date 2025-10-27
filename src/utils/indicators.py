from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class IndicatorConfig:
    sma_periods: List[int]
    ema_period: int
    macd_short: int
    macd_long: int
    macd_signal: int
    rsi_window: int
    bollinger_window: int
    bollinger_std: float
    supertrend_period: int
    supertrend_multiplier: float
    candlestick_lookback: int


def compute_indicator_bundle(ohlcv: pd.DataFrame, config: IndicatorConfig) -> Dict[str, Any]:
    df = ohlcv.copy().dropna(subset=["close"])
    df = df.sort_index()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    latest_close = _last_valid(close)
    price_date = df.index[-1].isoformat() if not df.empty else None

    sma_values: Dict[str, Dict[str, Optional[float]]] = {}
    for period in config.sma_periods:
        sma_series = close.rolling(window=period).mean()
        value = _last_valid(sma_series)
        relation = _relationship(latest_close, value)
        sma_values[f"SMA_{period}"] = {
            "value": value,
            "slope": _slope(sma_series),
            "price_relationship": relation,
        }

    ema_series = close.ewm(span=config.ema_period, adjust=False).mean()
    ema_value = _last_valid(ema_series)
    ema_relation = _relationship(latest_close, ema_value)

    bollinger = _bollinger_bands(close, config.bollinger_window, config.bollinger_std, latest_close)
    macd = _macd(close, config.macd_short, config.macd_long, config.macd_signal)
    rsi_value = _rsi(close, config.rsi_window)
    rsi_regime = _rsi_regime(rsi_value)
    supertrend = _supertrend(high, low, close, config.supertrend_period, config.supertrend_multiplier)
    obv_info = _obv(close, volume)
    candlestick_patterns = _detect_candlestick_patterns(open_, high, low, close, config.candlestick_lookback)

    derived_signals = _derive_cross_signals(close, config.sma_periods)
    key_levels = _key_levels(df)

    return {
        "latest_close": latest_close,
        "price_date": price_date,
        "SMA_50": sma_values.get("SMA_50"),
        "SMA_200": sma_values.get("SMA_200"),
        "EMA_20": {
            "value": ema_value,
            "price_relationship": ema_relation,
            "slope": _slope(ema_series),
        },
        "Bollinger_Bands": bollinger,
        "MACD_Signal": macd,
        "RSI": {
            "value": rsi_value,
            "regime": rsi_regime,
        },
        "Supertrend_Signal": supertrend,
        "OBV_Trend": obv_info,
        "recent_candlestick_patterns": candlestick_patterns,
        "derived_signals": derived_signals,
        "key_levels": key_levels,
    }


def _last_valid(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if clean.empty:
        return None
    return float(clean.iloc[-1])


def _relationship(price: Optional[float], level: Optional[float]) -> Optional[str]:
    if price is None or level is None:
        return None
    if price > level:
        return "above"
    if price < level:
        return "below"
    return "at"


def _slope(series: pd.Series, window: int = 5) -> Optional[float]:
    clean = series.dropna()
    if len(clean) <= window:
        return None
    recent = clean.iloc[-window:]
    x = np.arange(len(recent))
    coeffs = np.polyfit(x, recent.values, 1)
    return float(coeffs[0])


def _bollinger_bands(close: pd.Series, window: int, std_dev: float, latest_close: Optional[float]) -> Dict[str, Optional[float]]:
    rolling_mean = close.rolling(window=window).mean()
    rolling_std = close.rolling(window=window).std(ddof=0)
    upper = rolling_mean + std_dev * rolling_std
    lower = rolling_mean - std_dev * rolling_std
    middle = rolling_mean

    upper_val = _last_valid(upper)
    lower_val = _last_valid(lower)
    middle_val = _last_valid(middle)
    width = None
    position = None
    if upper_val is not None and lower_val is not None:
        width = float(upper_val - lower_val)
        if latest_close is not None and width:
            position = float((latest_close - lower_val) / width)

    return {
        "lower": lower_val,
        "middle": middle_val,
        "upper": upper_val,
        "width": width,
        "price_position": position,
    }


def _macd(close: pd.Series, short_window: int, long_window: int, signal_window: int) -> Dict[str, Optional[float]]:
    ema_short = close.ewm(span=short_window, adjust=False).mean()
    ema_long = close.ewm(span=long_window, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_val = _last_valid(macd_line)
    signal_val = _last_valid(signal_line)
    hist_val = _last_valid(histogram)
    crossover = None
    if macd_val is not None and signal_val is not None:
        if macd_val > signal_val:
            crossover = "bullish"
        elif macd_val < signal_val:
            crossover = "bearish"
        else:
            crossover = "neutral"

    return {
        "macd": macd_val,
        "signal": signal_val,
        "histogram": hist_val,
        "crossover": crossover,
    }


def _rsi(close: pd.Series, window: int) -> Optional[float]:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return _last_valid(rsi)


def _rsi_regime(rsi_value: Optional[float]) -> Optional[str]:
    if rsi_value is None:
        return None
    if rsi_value >= 70:
        return "overbought"
    if rsi_value <= 30:
        return "oversold"
    return "neutral"


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    high_low = high - low
    high_prev_close = (high - close.shift(1)).abs()
    low_prev_close = (low - close.shift(1)).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, adjust=False).mean()
    return atr


def _supertrend(high: pd.Series, low: pd.Series, close: pd.Series, period: int, multiplier: float) -> Dict[str, Optional[float]]:
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    final_upperband = upperband.copy()
    final_lowerband = lowerband.copy()

    for i in range(1, len(close)):
        if close.iloc[i - 1] <= final_upperband.iloc[i - 1]:
            final_upperband.iloc[i] = min(upperband.iloc[i], final_upperband.iloc[i - 1])
        else:
            final_upperband.iloc[i] = upperband.iloc[i]

        if close.iloc[i - 1] >= final_lowerband.iloc[i - 1]:
            final_lowerband.iloc[i] = max(lowerband.iloc[i], final_lowerband.iloc[i - 1])
        else:
            final_lowerband.iloc[i] = lowerband.iloc[i]

    supertrend = pd.Series(index=close.index, dtype=float)
    for i in range(len(close)):
        if i < period:
            supertrend.iloc[i] = np.nan
        else:
            prev_super = supertrend.iloc[i - 1] if i > 0 else np.nan
            if close.iloc[i] > final_upperband.iloc[i]:
                supertrend.iloc[i] = final_lowerband.iloc[i]
            elif close.iloc[i] < final_lowerband.iloc[i]:
                supertrend.iloc[i] = final_upperband.iloc[i]
            else:
                supertrend.iloc[i] = prev_super if not np.isnan(prev_super) else final_lowerband.iloc[i]

    trend = None
    level = _last_valid(supertrend)
    latest_close = _last_valid(close)
    if level is not None and latest_close is not None:
        trend = "bullish" if latest_close > level else "bearish"

    return {
        "trend": trend,
        "level": level,
    }


def _obv(close: pd.Series, volume: pd.Series) -> Dict[str, Optional[float]]:
    direction = np.sign(close.diff().fillna(0))
    obv_series = (volume * direction).cumsum()
    obv_value = _last_valid(obv_series)
    trend = None
    window = obv_series.dropna().tail(5)
    if len(window) >= 2:
        if window.iloc[-1] > window.iloc[0] * 1.01:
            trend = "rising"
        elif window.iloc[-1] < window.iloc[0] * 0.99:
            trend = "falling"
        else:
            trend = "flat"

    return {
        "value": obv_value,
        "trend": trend,
    }


def _detect_candlestick_patterns(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> List[Dict[str, Any]]:
    patterns: List[Dict[str, Any]] = []
    recent_indices = open_.index[-lookback:]
    for idx in recent_indices:
        o = open_.loc[idx]
        h = high.loc[idx]
        l = low.loc[idx]
        c = close.loc[idx]
        body = abs(c - o)
        range_ = h - l
        if range_ == 0:
            continue

        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        if body <= 0.1 * range_:
            patterns.append({
                "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "pattern": "Doji",
                "direction": "neutral",
            })
        elif body > 0.6 * range_ and upper_shadow <= 0.1 * range_ and lower_shadow <= 0.1 * range_:
            direction = "bullish" if c > o else "bearish"
            patterns.append({
                "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "pattern": "Marubozu",
                "direction": direction,
            })
        elif lower_shadow >= 2 * body and upper_shadow <= 0.2 * range_ and c > o:
            patterns.append({
                "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "pattern": "Hammer",
                "direction": "bullish",
            })
        elif upper_shadow >= 2 * body and lower_shadow <= 0.2 * range_ and c < o:
            patterns.append({
                "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "pattern": "Shooting Star",
                "direction": "bearish",
            })

    # Detect engulfing patterns using previous candle
    for i in range(1, len(recent_indices)):
        idx = recent_indices[i]
        prev_idx = recent_indices[i - 1]
        o1, c1 = open_.loc[prev_idx], close.loc[prev_idx]
        o2, c2 = open_.loc[idx], close.loc[idx]
        body1 = abs(c1 - o1)
        body2 = abs(c2 - o2)
        if body1 == 0 or body2 == 0:
            continue
        if c2 > o2 and c1 < o1 and c2 >= o1 and o2 <= c1:
            patterns.append({
                "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "pattern": "Bullish Engulfing",
                "direction": "bullish",
            })
        elif c2 < o2 and c1 > o1 and o2 >= c1 and c2 <= o1:
            patterns.append({
                "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "pattern": "Bearish Engulfing",
                "direction": "bearish",
            })

    return patterns


def _derive_cross_signals(close: pd.Series, sma_periods: List[int]) -> List[str]:
    signals: List[str] = []
    if 50 in sma_periods and 200 in sma_periods:
        sma_50 = close.rolling(window=50).mean()
        sma_200 = close.rolling(window=200).mean()
        diff = sma_50 - sma_200
        diff = diff.dropna()
        if len(diff) >= 2:
            latest_diff = diff.iloc[-1]
            prev_diff = diff.iloc[-2]
            if latest_diff > 0 and prev_diff <= 0:
                signals.append("Golden Cross triggered recently")
            elif latest_diff < 0 and prev_diff >= 0:
                signals.append("Death Cross triggered recently")
    return signals


def _key_levels(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    recent = df.tail(20)
    if recent.empty:
        return {"recent_high": None, "recent_low": None}
    return {
        "recent_high": float(recent["high"].max()),
        "recent_low": float(recent["low"].min()),
    }
