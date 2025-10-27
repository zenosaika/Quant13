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


def fetch_fundamental_bundle(ticker: str) -> Dict[str, Any]:
    ticker_obj = _create_ticker(ticker)
    info: Dict[str, Any] = {}
    financials = pd.DataFrame()
    balance_sheet = pd.DataFrame()
    cashflow = pd.DataFrame()
    filings = pd.DataFrame()

    try:
        info = ticker_obj.get_info()
    except Exception:  # noqa: BLE001
        info = {}

    try:
        financials = ticker_obj.get_financials()
    except Exception:  # noqa: BLE001
        financials = pd.DataFrame()

    try:
        balance_sheet = ticker_obj.get_balance_sheet()
    except Exception:  # noqa: BLE001
        balance_sheet = pd.DataFrame()

    try:
        cashflow = ticker_obj.get_cashflow()
    except Exception:  # noqa: BLE001
        cashflow = pd.DataFrame()

    filings = _fetch_filings(ticker_obj)

    return {
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "filings": filings,
    }


def _fetch_filings(ticker_obj: yf.Ticker) -> pd.DataFrame:
    candidates = [
        getattr(ticker_obj, "get_sec_filings", None),
        getattr(ticker_obj, "get_filings", None),
    ]
    for func in candidates:
        if callable(func):
            try:
                filings = func()
            except Exception:  # noqa: BLE001
                continue
            if isinstance(filings, pd.DataFrame) and not filings.empty:
                return filings
    filings_attr = getattr(ticker_obj, "filings", None)
    if isinstance(filings_attr, pd.DataFrame):
        return filings_attr
    return pd.DataFrame()
