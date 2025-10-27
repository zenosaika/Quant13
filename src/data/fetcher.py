from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup

from src.data.sec import fetch_recent_filings


def _create_ticker(ticker: str) -> yf.Ticker:
    return yf.Ticker(ticker)


def fetch_ohlcv(ticker: str, lookback_days: int) -> pd.DataFrame:
    """Fetch OHLCV data for the given ticker."""
    end_date = datetime.utcnow()
    buffer_days = max(int(lookback_days * 3), 400)
    start_date = end_date - timedelta(days=buffer_days)
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
    return articles


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
    if isinstance(filings, pd.DataFrame) and filings.empty:
        filings = fetch_recent_filings(ticker)

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
