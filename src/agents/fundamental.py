from __future__ import annotations

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.agents.base import Agent
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
        filings: pd.DataFrame = bundle.get("filings", pd.DataFrame())
        cache_path = self.settings.cache_dir / f"{ticker}.json"

        latest_filing_date = _latest_filing_date(filings)
        cached_report = _load_cached_report(cache_path)

        if cached_report and not _is_cache_stale(cached_report, latest_filing_date):
            cached_report["data_source"] = "cache"
            return {"report": cached_report}

        ratios = _calculate_financial_ratios(info, bundle.get("balance_sheet"), bundle.get("financials"))
        trends = _calculate_financial_trends(bundle.get("financials"), bundle.get("cashflow"))

        filing_texts = _collect_filing_sections(filings)
        mdna_summary = _summarize_text(self.llm, MDNA_PROMPT, filing_texts.get("mdna"))
        risk_summary = _summarize_text(self.llm, RISK_PROMPT, filing_texts.get("risks"))

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


def _latest_filing_date(filings: pd.DataFrame) -> Optional[datetime]:
    if filings is None or filings.empty:
        return None
    df = filings.copy()
    df.columns = [col.lower() for col in df.columns]
    type_column = next((col for col in ["filingtype", "type", "form"] if col in df.columns), None)
    date_column = next((col for col in ["filingdate", "filed", "date", "acceptancedatetime"] if col in df.columns), None)
    if not date_column:
        return None
    relevant = df
    if type_column:
        relevant = df[df[type_column].isin(["10-K", "10-Q"])]
        if relevant.empty:
            relevant = df
    dates = pd.to_datetime(relevant[date_column], errors="coerce", utc=True)
    dates = dates.dropna()
    if dates.empty:
        return None
    return dates.max().to_pydatetime()


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


def _collect_filing_sections(filings: pd.DataFrame) -> Dict[str, Optional[str]]:
    sections = {"mdna": None, "risks": None}
    if filings is None or filings.empty:
        return sections
    df = filings.copy()
    df.columns = [col.lower() for col in df.columns]
    type_col = next((col for col in ["filingtype", "type", "form"] if col in df.columns), None)
    html_col = next((col for col in ["linktohtml", "pdfurl", "html", "url", "edgarurl"] if col in df.columns), None)
    text_col = next((col for col in ["linktotxt", "txt", "text", "link"] if col in df.columns), None)
    if not type_col or (not html_col and not text_col):
        return sections

    filings_sorted = df.copy()
    date_col = next((col for col in ["filingdate", "filed", "date", "acceptancedatetime"] if col in df.columns), None)
    if date_col:
        filings_sorted[date_col] = pd.to_datetime(filings_sorted[date_col], errors="coerce")
        filings_sorted = filings_sorted.sort_values(date_col, ascending=False)

    latest_10k = filings_sorted[filings_sorted[type_col] == "10-K"].head(1)
    latest_10q = filings_sorted[filings_sorted[type_col] == "10-Q"].head(1)

    if not latest_10k.empty:
        text = _download_filing_text(latest_10k.iloc[0], html_col, text_col)
        sections["mdna"] = _extract_item_section(text, "ITEM 7.") or sections["mdna"]
        sections["risks"] = _extract_item_section(text, "ITEM 1A.") or sections["risks"]

    if sections["mdna"] is None and not latest_10q.empty:
        text = _download_filing_text(latest_10q.iloc[0], html_col, text_col)
        sections["mdna"] = _extract_item_section(text, "ITEM 2.") or text[:4000]

    if sections["risks"] is None and not latest_10q.empty:
        text = _download_filing_text(latest_10q.iloc[0], html_col, text_col)
        section = _extract_item_section(text, "ITEM 1A.")
        sections["risks"] = section or text[:4000]

    return sections


def _download_filing_text(row: pd.Series, html_col: Optional[str], text_col: Optional[str]) -> str:
    url = None
    if text_col and isinstance(row.get(text_col), str):
        url = row[text_col]
    elif html_col and isinstance(row.get(html_col), str):
        url = row[html_col]
    if not url:
        return ""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/plain") or url.lower().endswith(".txt"):
            return response.text
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception:
        return ""


def _extract_item_section(text: str, item_header: str) -> Optional[str]:
    if not text:
        return None
    upper_text = text.upper()
    header = item_header.upper()
    start_idx = upper_text.find(header)
    if start_idx == -1:
        return None
    remainder = upper_text[start_idx + len(header):]
    match = re.search(r"\n\s*ITEM\s+\d+[A-Z]?\.\s", remainder)
    end_idx = start_idx + len(header) + match.start() if match else len(text)
    section = text[start_idx:end_idx]
    return section.strip()


def _summarize_text(llm_client, prompt: str, text: Optional[str]) -> Any:
    if not text:
        return "Section unavailable."
    truncated = text.strip()
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
