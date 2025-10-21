from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf


def _create_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def fetch_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch OHLCV data for the given ticker."""
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days * 2)
    df = _create_ticker(ticker).history(start=start_date, end=end_date, interval="1d")
    if df.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}")
    return df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)


def fetch_options_chain(ticker: str, limit_expirations: int = 2) -> List[Dict[str, Any]]:
    ticker_obj = _create_ticker(ticker)
    expirations = ticker_obj.options
    chain: List[Dict[str, Any]] = []
    for expiry in expirations[:limit_expirations]:
        option_chain = ticker_obj.option_chain(expiry)
        calls = option_chain.calls[["contractSymbol", "strike", "lastPrice", "bid", "ask", "impliedVolatility"]].copy()
        puts = option_chain.puts[["contractSymbol", "strike", "lastPrice", "bid", "ask", "impliedVolatility"]].copy()
        chain.append({
            "expiration": expiry,
            "calls": calls,
            "puts": puts,
        })
    return chain


def fetch_news(ticker: str, limit: int) -> List[Dict[str, Any]]:
    ticker_obj = _create_ticker(ticker)
    raw_news = ticker_obj.news or []
    return raw_news[:limit]


def fetch_company_overview(ticker: str) -> Dict[str, Any]:
    ticker_obj = _create_ticker(ticker)
    try:
        info = ticker_obj.get_info()
    except Exception:  # pragma: no cover - network variability
        info = {}
    return info
