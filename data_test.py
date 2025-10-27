from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import yfinance as yf


def summarize_dataframe(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} -> type={type(df).__name__}, shape={df.shape}")
    if not df.empty:
        print("Columns:", list(df.columns)[:15])
        print("Head:\n", df.head(2))


def main(ticker: str = "AAPL") -> None:
    print(f"Inspecting ticker: {ticker}")
    ticker_obj = yf.Ticker(ticker)

    # OHLCV data
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=365)
    ohlcv = ticker_obj.history(start=start_date, end=end_date, interval="1d")
    summarize_dataframe(ohlcv, "OHLCV (1y daily)")
    if isinstance(ohlcv.index, pd.DatetimeIndex):
        print("OHLCV index start/end:", ohlcv.index.min(), ohlcv.index.max())

    # Options chain
    expirations = list(getattr(ticker_obj, "options", []))
    print("\nOptions expirations:", expirations[:5])
    if expirations:
        sample_exp = expirations[0]
        chain = ticker_obj.option_chain(sample_exp)
        summarize_dataframe(chain.calls, f"Calls for {sample_exp}")
        summarize_dataframe(chain.puts, f"Puts for {sample_exp}")

    # News articles
    news = getattr(ticker_obj, "news", []) or []
    print(f"\nFetched {len(news)} news items")
    if news:
        print("Sample news keys:", list(news[0].keys()))
        print("Sample news item:\n", json.dumps(news[0], indent=2)[:500])

    # Company info
    info: dict[str, Any] = {}
    try:
        info = ticker_obj.get_info()
    except Exception as exc:  # noqa: BLE001
        print("Error fetching info:", exc)
    print("\nCompany info keys (sample):", list(info.keys())[:15])
    for key in [
        "longBusinessSummary",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "pegRatio",
        "priceToBook",
        "priceToSalesTrailing12Months",
    ]:
        if key in info:
            print(f"{key}: {info[key]}")

    # Financial statements
    for label, getter in {
        "financials": ticker_obj.get_financials,
        "balance_sheet": ticker_obj.get_balance_sheet,
        "cashflow": ticker_obj.get_cashflow,
    }.items():
        try:
            data = getter()
        except Exception as exc:  # noqa: BLE001
            print(f"Error fetching {label}: {exc}")
        else:
            summarize_dataframe(data, f"{label.title()} Data")

    # SEC filings
    filings_df = pd.DataFrame()
    for attr in ("filings", "get_filings", "get_sec_filings"):
        source = getattr(ticker_obj, attr, None)
        if callable(source):
            try:
                filings_df = source()
            except Exception:  # noqa: BLE001
                continue
        elif isinstance(source, pd.DataFrame):
            filings_df = source
        if isinstance(filings_df, pd.DataFrame) and not filings_df.empty:
            break
    if isinstance(filings_df, list):
        filings_df = pd.DataFrame(filings_df)
    summarize_dataframe(filings_df, "SEC Filings")
    if not filings_df.empty:
        print("Filings columns:", list(filings_df.columns))
        print("Recent filings:\n", filings_df.head(3))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect yfinance datasets for a ticker")
    parser.add_argument("ticker", nargs="?", default="AAPL")
    args = parser.parse_args()
    main(args.ticker.upper())
