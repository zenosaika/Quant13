from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from src.agents.base import Agent
from src.data.sec import fetch_document_text
from src.data.fetcher import fetch_insider_transactions
from src.models.schemas import FundamentalReport, QualitativeSummary
from src.tools.llm import get_llm_client


MDNA_PROMPT = (
    "Analyze this MD&A section. Summarize management's tone (optimistic, pessimistic), key performance drivers, "
    "and any stated forward-looking concerns or opportunities. Respond with a JSON object containing the keys "
    "'tone', 'performance_drivers', and 'forward_looking'."
)

RISK_PROMPT = (
    "Analyze these Risk Factors. Identify and list the top 5 most significant risks to the business. "
    "Categorize them if possible (e.g., Market, Regulatory, Operational). Respond with a JSON array of objects, "
    "each containing 'risk', 'category', and 'rationale'."
)

FUNDAMENTAL_REPORT_VERSION = 2


@dataclass
class FundamentalAnalysisConfig:
    cache_dir: Path


class FundamentalAnalyst(Agent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.llm = get_llm_client()
        cache_dir = Path(__file__).resolve().parents[2] / "cache" / "fundamental_reports"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.settings = FundamentalAnalysisConfig(cache_dir=cache_dir)

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ticker = state["ticker"]
        bundle = state.get("fundamental_bundle", {})
        info: Dict[str, Any] = bundle.get("info", {}) or state.get("company_overview", {})
        filings_data = bundle.get("filings")
        cache_path = self.settings.cache_dir / f"{ticker}.json"

        latest_filing_date = _latest_filing_date(filings_data)
        cached_report = _load_cached_report(cache_path)

        if (
            cached_report
            and cached_report.get("data_version") == FUNDAMENTAL_REPORT_VERSION
            and not _is_cache_stale(cached_report, latest_filing_date)
        ):
            cached_report["data_source"] = "cache"
            return {"report": cached_report}

        ratios = _calculate_financial_ratios(info, bundle.get("balance_sheet"), bundle.get("financials"))
        trends = _calculate_financial_trends(bundle.get("financials"), bundle.get("cashflow"))

        filing_texts = _collect_filing_sections(filings_data)
        mdna_summary = _summarize_text(self.llm, MDNA_PROMPT, filing_texts.get("mdna"), kind="mdna")
        risk_summary = _summarize_text(self.llm, RISK_PROMPT, filing_texts.get("risks"), kind="risk")

        # Fetch and analyze insider transactions
        insider_transactions = fetch_insider_transactions(ticker, limit=10)
        insider_signal = _analyze_insider_activity(insider_transactions)

        qualitative_summary = QualitativeSummary(
            mdna_summary=mdna_summary if isinstance(mdna_summary, dict) else {"summary": mdna_summary},
            risk_factors=risk_summary if isinstance(risk_summary, list) else ([{"risk": str(risk_summary)}] if risk_summary else []),
        ).model_dump()

        synthesis_payload = {
            "ticker": ticker,
            "quantitative_data": {
                "financial_ratios": ratios,
                "financial_trends": trends,
            },
            "qualitative_summaries": {
                "mdna": mdna_summary,
                "risk_factors": risk_summary,
                "insider_activity": insider_signal,
            },
            "business_overview": _extract_business_overview(info),
        }

        messages = [
            {"role": "system", "content": self.config["prompt"]},
            {"role": "user", "content": json.dumps(synthesis_payload)},
        ]
        llm_raw = self.llm.chat(messages, temperature=0.2)
        llm_summary = _safe_parse_json(llm_raw)

        generated_at = datetime.now(timezone.utc).isoformat()
        report = {
            "ticker": ticker,
            "generated_at": generated_at,
            "data_source": "fresh",
            "data_version": FUNDAMENTAL_REPORT_VERSION,
            "business_overview": synthesis_payload["business_overview"],
            "financial_ratios": ratios,
            "financial_trends": trends,
            "qualitative_summary": qualitative_summary,
            "llm_synthesis": llm_summary,
            "analysis_timestamp": generated_at,
        }

        _save_cached_report(cache_path, report)
        return {"report": report}

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> FundamentalReport:
        report_data = analysis["report"]
        return FundamentalReport(**{k: v for k, v in report_data.items() if k != "analysis_timestamp"})


def _load_cached_report(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cached_report(path: Path, payload: Dict[str, Any]) -> None:
    try:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _is_cache_stale(report: Dict[str, Any], latest_filing: Optional[datetime]) -> bool:
    if not latest_filing:
        return False
    ts = report.get("analysis_timestamp")
    if not ts:
        return True
    try:
        cached_dt = datetime.fromisoformat(ts)
    except ValueError:
        return True
    return latest_filing > cached_dt


def _latest_filing_date(filings: Any) -> Optional[datetime]:
    if filings is None:
        return None

    timestamps: List[pd.Timestamp] = []

    if isinstance(filings, list):
        for entry in filings:
            if not isinstance(entry, dict):
                continue
            date_value = entry.get("filingDate") or entry.get("filed") or entry.get("date") or entry.get("reportDate")
            ts = pd.to_datetime(date_value, errors="coerce", utc=True)
            if pd.notna(ts):
                timestamps.append(ts)
    elif isinstance(filings, pd.DataFrame) and not filings.empty:
        df = filings.copy()
        df.columns = [col.lower() for col in df.columns]
        date_column = next((col for col in ["filingdate", "filed", "date", "acceptancedatetime"] if col in df.columns), None)
        if date_column:
            type_column = next((col for col in ["filingtype", "type", "form"] if col in df.columns), None)
            relevant = df
            if type_column:
                relevant = df[df[type_column].isin(["10-K", "10-Q"])]
                if relevant.empty:
                    relevant = df
            timestamps.extend(pd.to_datetime(relevant[date_column], errors="coerce", utc=True).dropna().tolist())

    if not timestamps:
        return None
    latest = max(timestamps)
    return latest.to_pydatetime()


def _calculate_financial_ratios(info: Dict[str, Any], balance_sheet: Optional[pd.DataFrame], financials: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    ratios = {
        "pe_ratio": _safe_float(info.get("trailingPE")),
        "ps_ratio": _safe_float(info.get("priceToSalesTrailing12Months")),
        "debt_to_equity": _safe_float(info.get("debtToEquity")),
        "current_ratio": None,
    }

    if isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
        current_assets = _latest_metric(balance_sheet, ["Total Current Assets"])
        current_liabilities = _latest_metric(balance_sheet, ["Total Current Liabilities"])
        if current_assets is not None and current_liabilities not in (None, 0):
            ratios["current_ratio"] = _safe_float(current_assets / current_liabilities)

    if ratios["debt_to_equity"] is None and isinstance(balance_sheet, pd.DataFrame) and not balance_sheet.empty:
        total_debt = _latest_metric(balance_sheet, ["Total Debt", "Short Long Term Debt", "Long Term Debt"])
        equity = _latest_metric(balance_sheet, ["Total Stockholder Equity"])
        if total_debt is not None and equity not in (None, 0):
            ratios["debt_to_equity"] = _safe_float(total_debt / equity)

    if ratios["pe_ratio"] is None and isinstance(financials, pd.DataFrame) and not financials.empty:
        net_income = _latest_series(financials, ["Net Income"])
        shares_outstanding = info.get("sharesOutstanding")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if net_income is not None and shares_outstanding and price:
            eps = net_income.iloc[-1] / shares_outstanding
            trailing_eps = eps if eps else None
            if trailing_eps:
                ratios["pe_ratio"] = _safe_float(price / trailing_eps)

    return ratios


def _calculate_financial_trends(financials: Optional[pd.DataFrame], cashflow: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    trends: List[Dict[str, Any]] = []
    revenue_series = _prepared_series(financials, ["Total Revenue", "Revenue"])
    net_income_series = _prepared_series(financials, ["Net Income"])
    free_cash_flow_series = _prepared_series(cashflow, ["Free Cash Flow", "FreeCashFlow"])

    if revenue_series is not None:
        trends.append(_build_trend_payload("Revenue", revenue_series))
    if net_income_series is not None:
        trends.append(_build_trend_payload("Net Income", net_income_series))
    if revenue_series is not None and net_income_series is not None:
        aligned = revenue_series.combine(net_income_series, lambda rev, net: net / rev if rev else np.nan)
        aligned = aligned.dropna()
        if not aligned.empty:
            trends.append(_build_trend_payload("Net Margin", aligned))
    if free_cash_flow_series is not None:
        trends.append(_build_trend_payload("Free Cash Flow", free_cash_flow_series))

    return trends


def _prepared_series(df: Optional[pd.DataFrame], row_names: List[str]) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for name in row_names:
        if name in df.index:
            series = df.loc[name]
            if isinstance(series, pd.Series):
                series = series.dropna()
                series.index = _normalize_index(series.index)
                series = series.sort_index()
                return series
    return None


def _normalize_index(index: pd.Index) -> pd.Index:
    try:
        return pd.to_datetime(index)
    except Exception:  # noqa: BLE001
        return index


def _build_trend_payload(metric: str, series: pd.Series) -> Dict[str, Any]:
    values = { _format_index(idx): _safe_float(val) for idx, val in series.items() }
    direction = _trend_direction(series)
    cagr = _compute_cagr(series)
    return {
        "metric": metric,
        "values": values,
        "trend_direction": direction,
        "compound_growth_rate": cagr,
    }


def _format_index(idx: Any) -> str:
    if isinstance(idx, (pd.Timestamp, datetime)):
        return pd.Timestamp(idx).strftime("%Y-%m-%d")
    return str(idx)


def _trend_direction(series: pd.Series) -> Optional[str]:
    if series.empty:
        return None
    start = series.iloc[0]
    end = series.iloc[-1]
    if start == 0:
        return None
    change = (end - start) / abs(start) if start else np.nan
    if np.isnan(change):
        return None
    if change > 0.1:
        return "improving"
    if change < -0.1:
        return "deteriorating"
    return "stable"


def _compute_cagr(series: pd.Series) -> Optional[float]:
    if len(series) < 2:
        return None
    start, end = float(series.iloc[0]), float(series.iloc[-1])
    if start <= 0 or end <= 0:
        return None
    index = series.index
    if isinstance(index, pd.DatetimeIndex):
        years = max((index[-1] - index[0]).days / 365.25, 1.0)
    else:
        years = len(series) - 1
    if years <= 0:
        return None
    try:
        return (end / start) ** (1 / years) - 1
    except ZeroDivisionError:
        return None


def _latest_metric(df: pd.DataFrame, names: List[str]) -> Optional[float]:
    series = _prepared_series(df, names)
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[-1])


def _latest_series(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    series = _prepared_series(df, names)
    return series


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _collect_filing_sections(filings: Any) -> Dict[str, Optional[str]]:
    sections = {"mdna": None, "risks": None}
    if filings is None:
        return sections

    entries: List[Dict[str, Any]] = []
    if isinstance(filings, list):
        entries = [entry for entry in filings if isinstance(entry, dict)]
    elif isinstance(filings, pd.DataFrame) and not filings.empty:
        entries = filings.to_dict("records")

    if not entries:
        return sections

    for entry in entries:
        form = str(entry.get("form") or entry.get("filingtype") or entry.get("type") or entry.get("Form") or "").upper()
        if form not in {"10-K", "10-Q"}:
            continue
        text = _download_filing_text(entry)
        if not text:
            continue

        if form == "10-K":
            if sections["mdna"] is None:
                sections["mdna"] = _extract_item_section(
                    text,
                    "7",
                    min_length=2000,
                    keywords=["MANAGEMENT'S DISCUSSION", "MANAGEMENT’S DISCUSSION"],
                ) or text[:12000]
            if sections["risks"] is None:
                sections["risks"] = _extract_item_section(
                    text,
                    "1A",
                    min_length=1500,
                    keywords=["RISK FACTORS"],
                ) or text[:8000]
        elif form == "10-Q":
            if sections["mdna"] is None:
                sections["mdna"] = _extract_item_section(
                    text,
                    "2",
                    min_length=1500,
                    keywords=["MANAGEMENT'S DISCUSSION", "MANAGEMENT’S DISCUSSION"],
                ) or text[:8000]
            if sections["risks"] is None:
                sections["risks"] = _extract_item_section(
                    text,
                    "1A",
                    min_length=1200,
                    keywords=["RISK FACTORS"],
                ) or text[:6000]

        if sections["mdna"] and sections["risks"]:
            break

    return sections


def _download_filing_text(entry: Any) -> str:
    urls: List[str] = []
    if isinstance(entry, dict):
        candidates = {k.lower(): v for k, v in entry.items()}
        for key in ("texturl", "linktotxt", "txt", "text"):
            value = candidates.get(key)
            if isinstance(value, str):
                urls.append(value)
        for key in ("documenturl", "html", "linktohtml", "pdfurl", "url", "edgarurl"):
            value = candidates.get(key)
            if isinstance(value, str) and value not in urls:
                urls.append(value)
    elif isinstance(entry, pd.Series):
        for key in ["linktotxt", "txt", "text", "link", "texturl", "documenturl", "linktohtml", "pdfurl", "html", "url", "edgarurl", "linktotxt"]:
            value = entry.get(key)
            if isinstance(value, str):
                urls.append(value)

    if not urls:
        return ""

    raw_html = fetch_document_text(urls)
    if not raw_html:
        return ""

    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator="\n")
        return text or raw_html
    except Exception:
        return raw_html


def _extract_item_section(
    text: str,
    item_number: str,
    min_length: int = 1000,
    keywords: Optional[List[str]] = None,
) -> Optional[str]:
    if not text:
        return None
    pattern = re.compile(rf"ITEM\s+{item_number}[A-Z]?\.", re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        return _extract_by_keywords(text, keywords)

    for match in matches:
        start = match.start()
        end = _find_next_item_boundary(text, match.end())
        section = text[start:end].strip()
        if len(section) >= min_length or _looks_like_section(section):
            return section
    # fall back to longest section if none meet criteria
    candidates = [text[m.start():_find_next_item_boundary(text, m.end())].strip() for m in matches]
    longest = max(candidates, key=len, default="")
    if longest:
        return longest
    return _extract_by_keywords(text, keywords)


def _find_next_item_boundary(text: str, start: int) -> int:
    pattern = re.compile(r"\n\s*ITEM\s+\d+[A-Z]?\.", re.IGNORECASE)
    match = pattern.search(text, start)
    return match.start() if match else len(text)


def _looks_like_section(section: str) -> bool:
    if not section or len(section) < 1000:
        return False
    words = section.split()
    if len(words) < 120:
        return False
    upper = section.upper()
    keywords = ["MANAGEMENT", "DISCUSSION", "RISK", "FACTORS", "OPERATIONS"]
    return any(keyword in upper for keyword in keywords)


def _extract_by_keywords(text: str, keywords: Optional[List[str]]) -> Optional[str]:
    if not keywords:
        return None
    upper = text.upper()
    for keyword in keywords:
        idx = upper.find(keyword)
        if idx != -1:
            end = _find_next_item_boundary(text, idx + len(keyword))
            snippet = text[idx:end].strip()
            if len(snippet) > 400:
                return snippet
    return None


def _summarize_text(llm_client, prompt: str, text: Optional[str], *, kind: str = "generic") -> Any:
    if not text:
        return "Section unavailable."
    truncated = text.strip()
    if not truncated:
        return "Section unavailable."
    lower_text = truncated.lower()
    if kind == "risk" and ("no material change" in lower_text or "other than the risk factors" in lower_text):
        return [
            {
                "risk": "No material updates disclosed in the latest filing.",
                "category": "General",
                "rationale": "Management stated there were no material changes to previously reported risk factors.",
            }
        ]
    if len(truncated) > 12000:
        truncated = truncated[:12000]
    messages = [
        {"role": "system", "content": "You are a fundamental equity analyst."},
        {"role": "user", "content": prompt + "\n\n" + truncated},
    ]
    response = llm_client.chat(messages, temperature=0.3)
    try:
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        return response


def _extract_business_overview(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "longBusinessSummary": info.get("longBusinessSummary"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "marketCap": info.get("marketCap"),
        "forwardPE": info.get("forwardPE"),
        "pegRatio": info.get("pegRatio"),
    }


def _analyze_insider_activity(transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze insider transactions to generate a signal

    Insider buying is typically bullish (insiders confident in company)
    Insider selling can be neutral/bearish (but may be for personal reasons)

    Returns:
        Dict with:
            - signal: "bullish", "bearish", "neutral"
            - confidence: "high", "medium", "low"
            - summary: Text explanation
            - recent_transactions: List of recent transactions
    """
    if not transactions:
        return {
            "signal": "neutral",
            "confidence": "low",
            "summary": "No recent insider transaction data available.",
            "recent_transactions": []
        }

    # Count buys vs sells
    num_buys = 0
    num_sells = 0
    total_buy_value = 0.0
    total_sell_value = 0.0

    for txn in transactions[:10]:  # Analyze last 10
        txn_type = str(txn.get("transaction_type", "")).lower()
        value = float(txn.get("value", 0))

        if "buy" in txn_type or "purchase" in txn_type:
            num_buys += 1
            total_buy_value += abs(value)
        elif "sell" in txn_type or "sale" in txn_type:
            num_sells += 1
            total_sell_value += abs(value)

    # Determine signal
    if num_buys > num_sells * 2:  # Significantly more buying
        signal = "bullish"
        confidence = "high" if total_buy_value > 1_000_000 else "medium"
        summary = f"Strong insider buying detected: {num_buys} buys vs {num_sells} sells. Total buy value: ${total_buy_value:,.0f}. Suggests insider confidence."
    elif num_buys > num_sells:  # Moderate buying
        signal = "bullish"
        confidence = "medium"
        summary = f"Moderate insider buying: {num_buys} buys vs {num_sells} sells. Positive signal but not overwhelming."
    elif num_sells > num_buys * 2:  # Heavy selling
        signal = "bearish"
        confidence = "medium"  # Selling can be for personal reasons, so medium confidence
        summary = f"Heavy insider selling: {num_sells} sells vs {num_buys} buys. May indicate concerns, though selling can be for personal liquidity."
    elif num_sells > num_buys:
        signal = "bearish"
        confidence = "low"
        summary = f"Moderate insider selling: {num_sells} sells vs {num_buys} buys. Mildly negative but not conclusive."
    else:
        signal = "neutral"
        confidence = "low"
        summary = f"Balanced insider activity: {num_buys} buys, {num_sells} sells. No clear directional signal."

    # Format recent transactions for reporting
    recent_txns = []
    for txn in transactions[:5]:  # Include top 5 in report
        recent_txns.append({
            "date": txn.get("date", ""),
            "insider": txn.get("insider_name", "Unknown"),
            "type": txn.get("transaction_type", ""),
            "shares": txn.get("shares", 0),
            "value": txn.get("value", 0)
        })

    return {
        "signal": signal,
        "confidence": confidence,
        "summary": summary,
        "recent_transactions": recent_txns,
        "stats": {
            "num_buys": num_buys,
            "num_sells": num_sells,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value
        }
    }


def _safe_parse_json(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {
        "overall_thesis": "neutral",
        "reasoning": raw.strip(),
    }
