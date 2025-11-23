from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import os
import pickle
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from src.data.sec import fetch_recent_filings


_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "data_fetch"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_TTL_SECONDS = int(os.getenv("DATA_FETCH_CACHE_TTL_SECONDS", "900"))


def _cache_enabled() -> bool:
    return _CACHE_TTL_SECONDS > 0


def _cache_path(namespace: str, key: str) -> Path:
    hashed = hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()
    return _CACHE_DIR / f"{namespace}_{hashed}.pkl"


def _load_from_cache(namespace: str, key: str) -> Any:
    if not _cache_enabled():
        return None
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > _CACHE_TTL_SECONDS:
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError):
        return None


def _save_to_cache(namespace: str, key: str, value: Any) -> None:
    if not _cache_enabled():
        return
    path = _cache_path(namespace, key)
    try:
        with path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PickleError):
        return


def _clone_cached(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, list):
        return [ _clone_cached(item) for item in value ]
    if isinstance(value, dict):
        return {key: _clone_cached(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return deepcopy(value)


def _create_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def fetch_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch OHLCV data for the given ticker."""
    cache_key = f"{ticker.upper()}_{lookback_days}"
    cached = _load_from_cache("ohlcv", cache_key)
    if cached is not None:
        return _clone_cached(cached)

    end_date = datetime.utcnow()
    buffer_days = max(int(lookback_days * 3), 400)
    start_date = end_date - timedelta(days=buffer_days)
    df = _create_ticker(ticker).history(start=start_date, end=end_date, interval="1d")
    if df.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}")
    subset = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
    _save_to_cache("ohlcv", cache_key, subset)
    return subset.copy(deep=True)


def fetch_options_chain(ticker: str, limit_expirations: int = 2) -> List[Dict[str, Any]]:
    cache_key = f"{ticker.upper()}_{limit_expirations}"
    cached = _load_from_cache("options_chain", cache_key)
    if cached is not None:
        return _clone_cached(cached)

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
    _save_to_cache("options_chain", cache_key, chain)
    return _clone_cached(chain)


def fetch_news(ticker: str, limit: int) -> List[Dict[str, Any]]:
    cache_key = f"{ticker.upper()}_{limit}"
    cached = _load_from_cache("news", cache_key)
    if cached is not None:
        return _clone_cached(cached)

    ticker_obj = _create_ticker(ticker)
    raw_news = ticker_obj.news or []
    articles: List[Dict[str, Any]] = []
    for item in raw_news[:limit]:
        article = dict(item)
        _hydrate_article_fields(article)
        link = _extract_article_link(article)
        if link:
            scraped = _scrape_article_content(link)
            if scraped:
                article["scraped_content"] = scraped
            if not article.get("link"):
                article["link"] = link
        _ensure_summary(article)
        articles.append(article)
    _save_to_cache("news", cache_key, articles)
    return _clone_cached(articles)


def fetch_company_overview(ticker: str) -> Dict[str, Any]:
    cache_key = ticker.upper()
    cached = _load_from_cache("company_overview", cache_key)
    if cached is not None:
        return _clone_cached(cached)

    ticker_obj = _create_ticker(ticker)
    try:
        info = ticker_obj.get_info()
    except Exception:  # pragma: no cover - network variability
        info = {}
    _save_to_cache("company_overview", cache_key, info)
    return _clone_cached(info)


def fetch_fundamental_bundle(ticker: str) -> Dict[str, Any]:
    cache_key = ticker.upper()
    cached = _load_from_cache("fundamental_bundle", cache_key)
    if cached is not None:
        return _clone_cached(cached)

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
    if isinstance(filings, pd.DataFrame) and filings.empty:
        filings = fetch_recent_filings(ticker)

    bundle = {
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "filings": filings,
    }
    _save_to_cache("fundamental_bundle", cache_key, bundle)
    return _clone_cached(bundle)


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


def _extract_article_link(article: Dict[str, Any]) -> Optional[str]:
    for candidate in (article.get("link"), article.get("url")):
        resolved = _resolve_url(candidate)
        if resolved:
            return resolved
    content = article.get("content")
    if isinstance(content, dict):
        for key in ("canonicalUrl", "clickThroughUrl", "url", "link", "href"):
            resolved = _resolve_url(content.get(key))
            if resolved:
                return resolved
        provider = content.get("provider")
        if isinstance(provider, dict):
            resolved = _resolve_url(provider.get("url"))
            if resolved:
                return resolved
    return None


def _resolve_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("url", "href", "canonicalUrl"):
            resolved = _resolve_url(value.get(key))
            if resolved:
                return resolved
    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _resolve_url(item)
            if resolved:
                return resolved
    return None


def _hydrate_article_fields(article: Dict[str, Any]) -> None:
    content = article.get("content") if isinstance(article.get("content"), dict) else {}
    provider = content.get("provider") if isinstance(content.get("provider"), dict) else {}

    if not article.get("title") and isinstance(content.get("title"), str):
        article["title"] = content["title"].strip()
    if not article.get("summary") and isinstance(content.get("summary"), str):
        article["summary"] = content["summary"].strip()
    if not article.get("publisher"):
        if isinstance(provider.get("displayName"), str):
            article["publisher"] = provider["displayName"].strip()
        elif isinstance(provider.get("name"), str):
            article["publisher"] = provider["name"].strip()
    if not article.get("providerPublishTime") and content.get("pubDate"):
        article["providerPublishTime"] = content.get("pubDate")
    if not article.get("link"):
        link = _resolve_url(content.get("canonicalUrl")) or _resolve_url(content.get("clickThroughUrl"))
        if link:
            article["link"] = link


def _ensure_summary(article: Dict[str, Any]) -> None:
    body = article.get("scraped_content")
    summary = article.get("summary")
    if summary:
        article["summary"] = _generate_summary(summary, summary)
    elif body:
        article["summary"] = _generate_summary(body, summary)


def _generate_summary(source: str, fallback: Optional[str] = None, max_sentences: int = 2, max_chars: int = 280) -> str:
    text = (source or "").strip()
    if not text and fallback:
        text = fallback.strip()
    if not text:
        return fallback or ""

    sentences = _split_sentences(text)
    if not sentences and fallback:
        sentences = _split_sentences(fallback)
    if not sentences:
        return (fallback or text)[:max_chars].strip()

    summary_parts: List[str] = []
    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if not cleaned:
            continue
        summary_parts.append(cleaned)
        joined = " ".join(summary_parts)
        if len(summary_parts) >= max_sentences or len(joined) >= max_chars:
            break

    summary = re.sub(r"\s+", " ", " ".join(summary_parts)).strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rstrip()
    return summary


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text.strip())


def _scrape_article_content(url: str) -> Optional[str]:
    headers = {"User-Agent": "Quant13NewsBot/1.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "svg", "form"]):
        tag.decompose()

    article_node = soup.find("article") or soup.find("main") or soup.body or soup
    text = article_node.get_text("\n", strip=True) if article_node else ""
    text = re.sub(r"\n{2,}", "\n\n", text).strip()
    if not text:
        return None
    return text[:5000]


def fetch_insider_transactions(ticker: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch recent insider transactions for the given ticker

    Returns list of insider transactions with:
    - date: Transaction date
    - insider_name: Name of the insider
    - transaction_type: Buy/Sell
    - shares: Number of shares
    - value: Dollar value
    - shares_owned_after: Shares owned after transaction

    Args:
        ticker: Stock ticker symbol
        limit: Maximum number of transactions to return

    Returns:
        List of insider transaction dictionaries
    """
    cache_key = f"{ticker.upper()}_{limit}"
    cached = _load_from_cache("insider_transactions", cache_key)
    if cached is not None:
        return _clone_cached(cached)

    try:
        stock = _create_ticker(ticker)

        # yfinance provides insider transactions via get_insider_transactions()
        # or insider_transactions attribute
        insider_data = None

        # Try different methods to get insider data
        try:
            # Method 1: Direct attribute
            if hasattr(stock, 'insider_transactions'):
                insider_data = stock.insider_transactions
        except Exception:
            pass

        # Method 2: Get method
        if insider_data is None:
            try:
                insider_data = stock.get_insider_transactions()
            except Exception:
                pass

        # If we got no data, return empty list
        if insider_data is None or (isinstance(insider_data, pd.DataFrame) and insider_data.empty):
            result = []
            _save_to_cache("insider_transactions", cache_key, result)
            return result

        # Convert DataFrame to list of dicts
        transactions = []

        # Handle DataFrame format
        if isinstance(insider_data, pd.DataFrame):
            # Reset index to get date as column
            df = insider_data.reset_index()

            for idx, row in df.head(limit).iterrows():
                # Extract fields (column names may vary)
                transaction = {
                    "date": row.get("Start Date", row.get("Date", "")).strftime("%Y-%m-%d") if pd.notna(row.get("Start Date", row.get("Date", ""))) else "",
                    "insider_name": str(row.get("Insider Trading", row.get("Name", "Unknown"))),
                    "transaction_type": str(row.get("Transaction", row.get("Type", ""))),
                    "shares": int(row.get("Shares", row.get("#Shares", 0))) if pd.notna(row.get("Shares", row.get("#Shares", 0))) else 0,
                    "value": float(row.get("Value ($)", row.get("Value", 0))) if pd.notna(row.get("Value ($)", row.get("Value", 0))) else 0,
                    "shares_owned_after": int(row.get("Shares Owned", 0)) if pd.notna(row.get("Shares Owned", 0)) else 0,
                }
                transactions.append(transaction)

        # Cache and return
        _save_to_cache("insider_transactions", cache_key, transactions)
        return transactions

    except Exception as e:
        # If fetch fails, return empty list (insider data is optional enhancement)
        import logging
        logging.getLogger(__name__).warning(f"Failed to fetch insider transactions for {ticker}: {e}")
        result = []
        _save_to_cache("insider_transactions", cache_key, result)
        return result
