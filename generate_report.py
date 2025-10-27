import argparse
import base64
import io
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

import markdown
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT_FILES: Dict[str, str] = {
    "fundamental_report": "fundamental_report.json",
    "technical_report": "technical_report.json",
    "trade_decision": "trade_decision.json",
    "trade_thesis": "trade_thesis.json",
    "volatility_report": "volatility_report.json",
    "sentiment_report": "sentiment_report.json",
    "risk_assessment": "risk_assessment.json",
}

JSON_BLOCK_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
SPECIAL_ACRONYMS = {"iv", "rsi", "macd", "aws", "obv", "fcf", "ema", "sma"}


def load_all_reports(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for key, filename in REPORT_FILES.items():
        file_path = os.path.join(path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Missing required report file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as handle:
            data[key] = json.load(handle)
    return data


def extract_json_blocks(text: Optional[str]) -> List[Any]:
    if not isinstance(text, str):
        return []
    blocks: List[Any] = []
    for match in JSON_BLOCK_RE.finditer(text):
        try:
            blocks.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return blocks


def strip_code_blocks(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return JSON_BLOCK_RE.sub("", text).strip()


def render_markdown_html(text: Optional[str]) -> str:
    cleaned = strip_code_blocks(text)
    if not cleaned:
        return ""
    return markdown.markdown(
        cleaned,
        extensions=["extra", "sane_lists", "tables", "fenced_code"],
        output_format="html5",
    )


def humanize_phrase(value: Optional[str]) -> str:
    if not value:
        return ""
    text = str(value).replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    words: List[str] = []
    for word in text.split(" "):
        lower = word.lower()
        if lower in SPECIAL_ACRONYMS:
            words.append(lower.upper())
        elif lower.isupper():
            words.append(lower)
        else:
            words.append(lower.capitalize())
    return " ".join(words)


def get_nested(data: Any, path: Sequence[Any], default: Optional[Any] = None) -> Optional[Any]:
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    cleaned = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def format_timestamp(value: Optional[Any]) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = parse_iso_datetime(value)
    else:
        dt = None
    if not dt:
        return value if isinstance(value, str) and value else "N/A"
    fmt = "%B %d, %Y %H:%M %Z" if dt.tzinfo else "%B %d, %Y %H:%M"
    return dt.strftime(fmt)


def format_date(value: Optional[Any]) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = parse_iso_datetime(value)
    else:
        dt = None
    if not dt:
        return value if isinstance(value, str) and value else "N/A"
    return dt.strftime("%B %d, %Y")


def format_currency(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"${number:,.{decimals}f}"


def format_large_currency(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_value = abs(number)
    for divisor, suffix in ((1_000_000_000_000, "T"), (1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs_value >= divisor:
            return f"${number / divisor:,.2f}{suffix}"
    if abs_value >= 1000:
        return f"${number:,.0f}"
    return f"${number:,.2f}"


def format_number(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(number)


def format_percentage(value: Any, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) <= 1:
        number *= 100
    return f"{number:.{decimals}f}%"


def clean_text(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def flatten_details(details: Dict[str, Any], skip_keys: Optional[Iterable[str]] = None, prefix: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    skipped = set(skip_keys or [])
    for key, value in details.items():
        if key in skipped:
            continue
        label = humanize_phrase(key) if not prefix else f"{prefix} – {humanize_phrase(key)}"
        if isinstance(value, dict):
            rows.extend(flatten_details(value, skip_keys=skip_keys, prefix=label))
        elif isinstance(value, list):
            joined = ", ".join(format_number(item) if isinstance(item, (int, float)) else str(item) for item in value)
            rows.append({"label": label, "value": joined})
        else:
            display = format_number(value) if isinstance(value, (int, float)) else str(value)
            rows.append({"label": label, "value": display})
    return rows


def infer_report_datetime(reports: Dict[str, Any]) -> datetime:
    for key in ("trade_decision", "trade_thesis", "fundamental_report", "technical_report"):
        generated = reports.get(key, {}).get("generated_at")
        dt = parse_iso_datetime(generated)
        if dt:
            return dt
    return datetime.now()


def extract_trade_summary(trade_decision: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "max_risk": None,
        "max_reward": None,
        "justification": None,
        "details": {},
        "raw_notes": None,
        "conviction": None,
        "direction": None,
        "holding_period": None,
        "cost_per_contract": None,
    }
    notes = trade_decision.get("notes")
    proposal: Dict[str, Any] = {}
    if isinstance(notes, dict):
        proposal = notes.get("trade_proposal") or notes
    elif isinstance(notes, str):
        for block in extract_json_blocks(notes):
            if isinstance(block, dict):
                if "trade_proposal" in block and isinstance(block["trade_proposal"], dict):
                    proposal = block["trade_proposal"]
                else:
                    proposal = block
                break
        summary["raw_notes"] = notes
    if proposal:
        summary["details"] = proposal
        summary["max_risk"] = (
            proposal.get("max_risk")
            or proposal.get("max_cost")
            or proposal.get("max_cost_per_share")
            or proposal.get("max_loss")
        )
        summary["max_reward"] = proposal.get("max_reward") or proposal.get("max_profit")
        summary["justification"] = proposal.get("justification") or proposal.get("rationale")
        summary["conviction"] = proposal.get("conviction")
        summary["direction"] = proposal.get("thesis_direction") or proposal.get("direction")
        summary["holding_period"] = proposal.get("days_to_expiration") or proposal.get("holding_period")
        summary["cost_per_contract"] = proposal.get("estimated_cost_per_contract")
        if summary["max_reward"] is None and (proposal.get("net_credit_debit") or "").lower().startswith("credit"):
            summary["max_reward"] = proposal.get("net_premium")
    elif isinstance(notes, str):
        stripped = strip_code_blocks(notes)
        if stripped:
            summary["justification"] = stripped
    summary["justification"] = summary["justification"] or trade_decision.get("strategy_rationale")
    return summary


def build_derived_fields(reports: Dict[str, Any]) -> Dict[str, Any]:
    latest_close = get_nested(reports, ["technical_report", "indicators", "latest_close"])
    price_date = get_nested(reports, ["technical_report", "indicators", "price_date"])
    current_price = get_nested(reports, ["trade_thesis", "current_price"])
    return {
        "latest_close": latest_close,
        "price_date": price_date,
        "price_date_human": format_date(price_date),
        "current_price": current_price if current_price is not None else latest_close,
        "rsi_value": get_nested(reports, ["technical_report", "indicators", "RSI", "value"]),
        "rsi_regime": get_nested(reports, ["technical_report", "indicators", "RSI", "regime"]),
        "macd_histogram": get_nested(reports, ["technical_report", "indicators", "MACD_Signal", "histogram"]),
        "macd_crossover": get_nested(reports, ["technical_report", "indicators", "MACD_Signal", "crossover"]),
        "supertrend_level": get_nested(reports, ["technical_report", "indicators", "Supertrend_Signal", "level"]),
        "supertrend_trend": get_nested(reports, ["technical_report", "indicators", "Supertrend_Signal", "trend"]),
        "obv_value": get_nested(reports, ["technical_report", "indicators", "OBV_Trend", "value"]),
        "obv_trend": get_nested(reports, ["technical_report", "indicators", "OBV_Trend", "trend"]),
        "technical_bias": get_nested(reports, ["technical_report", "llm_report", "technical_bias"]),
        "iv_rank": get_nested(reports, ["volatility_report", "iv_rank"]),
        "volatility_forecast": get_nested(reports, ["volatility_report", "volatility_forecast"]),
        "sentiment_score": get_nested(reports, ["sentiment_report", "overall_sentiment_score"]),
        "sentiment_summary": get_nested(reports, ["sentiment_report", "overall_summary"]),
        "market_cap": get_nested(reports, ["fundamental_report", "business_overview", "marketCap"]),
    }


def build_market_snapshot_rows(derived: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = [
        {"label": "Price", "value": format_currency(derived.get("current_price"))},
        {"label": "Price Date", "value": derived.get("price_date_human") or "N/A"},
        {"label": "IV Rank", "value": format_number(derived.get("iv_rank"), 1) if derived.get("iv_rank") is not None else "N/A"},
        {"label": "Technical Bias", "value": humanize_phrase(derived.get("technical_bias")) or "Neutral"},
        {"label": "Sentiment", "value": derived.get("sentiment_summary") or "Neutral"},
    ]
    return rows


def build_trade_section(reports: Dict[str, Any], derived: Dict[str, Any], summary: Dict[str, Any]) -> Dict[str, Any]:
    decision = reports.get("trade_decision", {}) or {}
    summary_details = summary.get("details", {}) or {}
    direction = summary.get("direction") or summary_details.get("direction") or decision.get("action")
    strategy = summary_details.get("strategy") or decision.get("strategy_name")
    conviction = summary.get("conviction") or summary_details.get("conviction") or get_nested(reports, ["trade_thesis", "conviction_level"])
    highlights: List[Dict[str, str]] = []
    proposal = summary_details
    underlying_price = proposal.get("underlying_price")
    if underlying_price is not None:
        highlights.append({"label": "Underlying Price", "value": format_currency(underlying_price)})
    expiration = proposal.get("expiration_date") or get_nested(decision, ["trade_legs", 0, "expiration_date"])
    if expiration:
        highlights.append({"label": "Target Expiration", "value": format_date(expiration)})
    holding = summary.get("holding_period")
    if holding:
        highlights.append({"label": "Days to Expiration", "value": f"{holding} days"})
    cost = summary.get("cost_per_contract")
    if cost:
        highlights.append({"label": "Est. Cost / Contract", "value": format_currency(cost)})
    net_premium = proposal.get("net_premium")
    if net_premium is not None:
        label = "Net Credit" if (proposal.get("net_credit_debit") or "").lower().startswith("credit") else "Net Debit"
        highlights.append({"label": label, "value": format_currency(net_premium)})
    market_cap = derived.get("market_cap")
    if market_cap:
        highlights.append({"label": "Market Cap", "value": format_large_currency(market_cap)})
    max_risk = summary.get("max_risk")
    max_reward = summary.get("max_reward")
    legs_source = decision.get("trade_legs") or proposal.get("trade_legs") or []
    legs: List[Dict[str, str]] = []
    for leg in legs_source:
        if not isinstance(leg, dict):
            continue
        greeks = leg.get("key_greeks_at_selection")
        greek_parts: List[str] = []
        if isinstance(greeks, dict):
            for name, value in greeks.items():
                if value is None:
                    continue
                label = name.replace("impliedVolatility", "IV")
                greek_parts.append(f"{humanize_phrase(label)}: {format_number(value, 4)}")
        legs.append(
            {
                "action": humanize_phrase(leg.get("action")),
                "type": humanize_phrase(leg.get("type")),
                "expiration": format_date(leg.get("expiration_date")),
                "strike": format_currency(leg.get("strike_price"), 2),
                "quantity": format_number(leg.get("quantity"), 0),
                "symbol": leg.get("contract_symbol") or "—",
                "greeks": "; ".join(greek_parts) if greek_parts else "—",
            }
        )
    justification_html = render_markdown_html(summary.get("justification"))
    action_display = summary_details.get("net_credit_debit") or decision.get("action") or proposal.get("action")
    if isinstance(action_display, str):
        action_display = humanize_phrase(action_display)
    else:
        action_display = humanize_phrase(decision.get("action"))
    quantity_value = decision.get("quantity") or summary_details.get("quantity")
    return {
        "direction": humanize_phrase(direction),
        "strategy": humanize_phrase(strategy),
        "conviction": humanize_phrase(conviction),
        "action": action_display,
        "quantity": format_number(quantity_value, 0) if quantity_value is not None else "1",
        "max_risk": format_currency(max_risk) if max_risk is not None else "N/A",
        "max_reward": format_currency(max_reward) if max_reward is not None else "N/A",
        "highlights": highlights,
        "legs": legs,
        "justification_html": justification_html,
    }


def extract_headline(text: Optional[str]) -> str:
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[0] if sentences else text.strip()


def build_thesis_section(reports: Dict[str, Any]) -> Dict[str, Any]:
    thesis = reports.get("trade_thesis", {}) or {}
    summary_text = thesis.get("summary")
    payloads = extract_json_blocks(summary_text)
    primary_payload = payloads[0] if payloads and isinstance(payloads[0], dict) else {}
    narrative_plain = primary_payload.get("summary") if isinstance(primary_payload, dict) else None
    if not narrative_plain:
        narrative_plain = strip_code_blocks(summary_text)
    narrative_html = render_markdown_html(narrative_plain)
    evidence_items: List[Dict[str, str]] = []
    evidence_source = primary_payload.get("key_evidence") if isinstance(primary_payload, dict) else thesis.get("key_evidence")
    if isinstance(evidence_source, list):
        for item in evidence_source:
            if not isinstance(item, dict):
                continue
            evidence_items.append(
                {
                    "type": humanize_phrase(item.get("type")),
                    "detail_html": render_markdown_html(item.get("detail") or item.get("description")),
                }
            )
    thesis_details: Dict[str, Any] = {}
    if isinstance(primary_payload, dict):
        thesis_details = {
            k: v
            for k, v in primary_payload.items()
            if k not in {"key_evidence", "summary"} and isinstance(v, (str, int, float, list, dict))
        }
    plan_rows = flatten_details(thesis_details, skip_keys={"trade_thesis", "conviction"}) if thesis_details else []
    headline = extract_headline(narrative_plain)
    return {
        "headline": headline,
        "narrative_html": narrative_html,
        "narrative_plain": narrative_plain or "",
        "evidence": evidence_items,
        "plan_rows": plan_rows,
    }


def build_technical_section(reports: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    indicators = get_nested(reports, ["technical_report", "indicators"], {}) or {}
    rows: List[Dict[str, str]] = []
    latest_close = derived.get("latest_close")
    if latest_close is not None:
        context_note = f"Close on {derived.get('price_date_human')}" if derived.get("price_date_human") else "Latest close"
        rows.append({"label": "Latest Close", "value": format_currency(latest_close), "context": context_note})
    rsi_value = derived.get("rsi_value")
    if rsi_value is not None:
        rows.append(
            {
                "label": "RSI",
                "value": format_number(rsi_value, 1),
                "context": humanize_phrase(derived.get("rsi_regime")),
            }
        )
    macd_histogram = derived.get("macd_histogram")
    if macd_histogram is not None:
        rows.append(
            {
                "label": "MACD Histogram",
                "value": format_number(macd_histogram, 3),
                "context": humanize_phrase(derived.get("macd_crossover")),
            }
        )
    supertrend_level = derived.get("supertrend_level")
    if supertrend_level is not None:
        rows.append(
            {
                "label": "Supertrend Level",
                "value": format_currency(supertrend_level),
                "context": humanize_phrase(derived.get("supertrend_trend")),
            }
        )
    obv_value = derived.get("obv_value")
    if obv_value is not None:
        rows.append(
            {
                "label": "On-Balance Volume",
                "value": format_number(obv_value, 0),
                "context": humanize_phrase(derived.get("obv_trend")),
            }
        )
    for key in ("SMA_50", "SMA_200", "EMA_20"):
        node = indicators.get(key)
        if isinstance(node, dict):
            rows.append(
                {
                    "label": humanize_phrase(key),
                    "value": format_currency(node.get("value")),
                    "context": humanize_phrase(node.get("price_relationship")),
                }
            )
    bollinger = indicators.get("Bollinger_Bands")
    if isinstance(bollinger, dict):
        rows.append(
            {
                "label": "Bollinger Band Width",
                "value": format_number(bollinger.get("width"), 2),
                "context": f"Price position {format_percentage(bollinger.get('price_position'), 1)}",
            }
        )
    key_levels: List[Dict[str, str]] = []
    levels = indicators.get("key_levels") or {}
    if isinstance(levels, dict):
        for label, value in levels.items():
            key_levels.append({"label": humanize_phrase(label), "value": format_currency(value)})
    patterns: List[Dict[str, str]] = []
    for pattern in indicators.get("recent_candlestick_patterns", []) or []:
        if not isinstance(pattern, dict):
            continue
        patterns.append(
            {
                "date": format_date(pattern.get("date")),
                "pattern": humanize_phrase(pattern.get("pattern")),
                "signal": humanize_phrase(pattern.get("direction")),
            }
        )
    summary_html = render_markdown_html(get_nested(reports, ["technical_report", "llm_report", "summary"]))
    return {
        "bias": humanize_phrase(derived.get("technical_bias")),
        "summary_html": summary_html,
        "indicators": rows,
        "key_levels": key_levels,
        "patterns": patterns,
    }


def build_fundamental_section(reports: Dict[str, Any]) -> Dict[str, Any]:
    fundamental = reports.get("fundamental_report", {}) or {}
    business_summary_html = render_markdown_html(get_nested(fundamental, ["business_overview", "longBusinessSummary"]))
    ratios_list: List[Dict[str, str]] = []
    ratios = fundamental.get("financial_ratios") or {}
    if isinstance(ratios, dict):
        for key, value in ratios.items():
            ratios_list.append({"label": humanize_phrase(key), "value": format_number(value)})
    trend_tables: List[Dict[str, Any]] = []
    for trend in fundamental.get("financial_trends", []) or []:
        if not isinstance(trend, dict):
            continue
        values = trend.get("values")
        if not isinstance(values, dict):
            continue
        rows = [{"period": period, "value": format_large_currency(amount)} for period, amount in sorted(values.items())]
        trend_tables.append(
            {
                "metric": trend.get("metric", "Financial Trend"),
                "direction": humanize_phrase(trend.get("trend_direction")),
                "rows": rows,
            }
        )
    mdna_summary_text = get_nested(fundamental, ["qualitative_summary", "mdna_summary", "summary"])
    mdna_struct = extract_json_blocks(mdna_summary_text)
    mdna_payload = mdna_struct[0] if mdna_struct and isinstance(mdna_struct[0], dict) else {}
    mdna = {
        "tone": clean_text(get_nested(mdna_payload, ["tone"])) if mdna_payload else "",
        "performance_drivers": [clean_text(item) for item in mdna_payload.get("performance_drivers", []) if isinstance(item, str)] if mdna_payload else [],
        "opportunities": [clean_text(item) for item in get_nested(mdna_payload, ["forward_looking", "opportunities"], []) if isinstance(item, str)],
        "concerns": [clean_text(item) for item in get_nested(mdna_payload, ["forward_looking", "concerns"], []) if isinstance(item, str)],
        "narrative_html": render_markdown_html(mdna_summary_text),
    }
    risk_matrix: List[Dict[str, str]] = []
    risk_factors = get_nested(fundamental, ["qualitative_summary", "risk_factors"], [])
    if isinstance(risk_factors, list):
        for risk_entry in risk_factors:
            if not isinstance(risk_entry, dict):
                continue
            risk_text = risk_entry.get("risk")
            parsed_blocks = extract_json_blocks(risk_text)
            for block in parsed_blocks:
                if isinstance(block, list):
                    for item in block:
                        if not isinstance(item, dict):
                            continue
                        risk_matrix.append(
                            {
                                "risk": clean_text(item.get("risk")),
                                "category": humanize_phrase(item.get("category")),
                                "rationale": clean_text(item.get("rationale")),
                            }
                        )
                elif isinstance(block, dict):
                    risk_matrix.append(
                        {
                            "risk": clean_text(block.get("risk")),
                            "category": humanize_phrase(block.get("category")),
                            "rationale": clean_text(block.get("rationale")),
                        }
                    )
            break
    return {
        "business_summary_html": business_summary_html,
        "ratios": ratios_list,
        "trend_tables": trend_tables,
        "mdna": mdna,
        "risk_matrix": risk_matrix[:5],
    }


def build_volatility_section(reports: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    vol = reports.get("volatility_report", {}) or {}
    metrics = [
        {"label": "IV Rank", "value": format_number(vol.get("iv_rank"), 1) if vol.get("iv_rank") is not None else "N/A"},
        {"label": "Term Structure", "value": vol.get("term_structure", "N/A")},
        {"label": "Skew Analysis", "value": vol.get("skew_analysis", "N/A")},
    ]
    forecast = vol.get("volatility_forecast") or derived.get("volatility_forecast") or ""
    return {"metrics": metrics, "forecast": forecast}


def build_sentiment_section(reports: Dict[str, Any], derived: Dict[str, Any]) -> Dict[str, Any]:
    sentiment = reports.get("sentiment_report", {}) or {}
    articles_output: List[Dict[str, str]] = []
    for article in sentiment.get("articles", [])[:5]:
        if not isinstance(article, dict):
            continue
        articles_output.append(
            {
                "title": article.get("title", "Headline unavailable"),
                "publisher": article.get("publisher", "Unknown"),
                "published": format_timestamp(article.get("published_at")),
                "summary": clean_text(article.get("rationale")) or "Summary unavailable.",
                "link": article.get("link"),
            }
        )
    score = derived.get("sentiment_score")
    return {
        "score": format_number(score, 1) if score is not None else "N/A",
        "summary": sentiment.get("overall_summary", derived.get("sentiment_summary") or "Neutral stance"),
        "articles": articles_output,
    }


def build_risk_section(reports: Dict[str, Any]) -> Dict[str, Any]:
    risk = reports.get("risk_assessment", {}) or {}
    adjustments: List[Dict[str, str]] = []
    for item in risk.get("adjustments", []) or []:
        if not isinstance(item, dict):
            continue
        adjustments.append(
            {
                "profile": humanize_phrase(item.get("profile")),
                "recommendation": clean_text(item.get("recommendation")),
            }
        )
    return {"adjustments": adjustments, "final": risk.get("final_recommendation")}


def build_visual_assets(reports: Dict[str, Any]) -> Dict[str, Any]:
    trends = get_nested(reports, ["fundamental_report", "financial_trends"], [])
    charts: List[Dict[str, str]] = []
    if not isinstance(trends, list):
        return {"trend_charts": charts}
    for trend in trends:
        if not isinstance(trend, dict):
            continue
        values = trend.get("values")
        if not isinstance(values, dict) or not values:
            continue
        try:
            points = sorted(
                (
                    (parse_iso_datetime(period) or datetime.fromisoformat(period), float(amount))
                    for period, amount in values.items()
                ),
                key=lambda pair: pair[0],
            )
        except (ValueError, TypeError):
            continue
        if not points:
            continue
        dates, series = zip(*points)
        fig, ax = plt.subplots(figsize=(5.6, 2.8), dpi=160)
        ax.plot(dates, series, color="#1D4ED8", linewidth=2.0, marker="o", markersize=4)
        ax.fill_between(dates, series, color="#1D4ED8", alpha=0.12)
        ax.set_title(trend.get("metric", "Financial Trend"), fontsize=10, color="#0F172A", pad=10)
        ax.grid(color="#CBD5F5", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.tick_params(colors="#475569", labelsize=8)
        fig.autofmt_xdate()
        buffer = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        charts.append(
            {
                "label": trend.get("metric", "Financial Trend"),
                "image": f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}",
            }
        )
        break
    return {"trend_charts": charts}


def build_executive_summary(
    trade_section: Dict[str, Any],
    derived: Dict[str, Any],
    thesis_section: Dict[str, Any],
    fundamentals: Dict[str, Any],
    volatility: Dict[str, Any],
) -> Dict[str, Any]:
    bullets: List[str] = []
    direction = trade_section.get("direction")
    strategy = trade_section.get("strategy")
    conviction = trade_section.get("conviction")
    if direction or strategy:
        conviction_text = f" ({conviction} conviction)" if conviction else ""
        bullets.append(clean_text(f"{direction or 'Trade'} {strategy or ''}{conviction_text}"))
    bullets.append(
        clean_text(
            f"Risk envelope: {trade_section.get('max_risk')} max risk / {trade_section.get('max_reward')} potential reward."
        )
    )
    technical_bias = derived.get("technical_bias")
    if technical_bias or derived.get("rsi_value") is not None:
        bias_text = humanize_phrase(technical_bias) or "Neutral"
        rsi_text = format_number(derived.get("rsi_value"), 1) if derived.get("rsi_value") is not None else "N/A"
        macd_context = humanize_phrase(derived.get("macd_crossover")) or "steady"
        bullets.append(f"Technical view: {bias_text} bias with RSI {rsi_text} and {macd_context} MACD crossover.")
    mdna_tone = fundamentals.get("mdna", {}).get("tone")
    if mdna_tone:
        bullets.append(f"Management tone: {mdna_tone}")
    if volatility.get("forecast"):
        bullets.append(volatility["forecast"])
    headline = thesis_section.get("headline") or direction or "Trade Summary"
    bullets = [clean_text(item) for item in bullets if clean_text(item)]
    return {"headline": headline, "bullets": bullets}


def build_meta_items(
    ticker: str,
    timestamp: str,
    report_date: str,
    results_folder: str,
    trade: Dict[str, Any],
    volatility: Dict[str, Any],
    sentiment: Dict[str, Any],
) -> List[Dict[str, str]]:
    return [
        {"label": "Ticker", "value": ticker.upper()},
        {"label": "Results Folder", "value": results_folder},
        {"label": "Run Timestamp", "value": timestamp},
        {"label": "Report Generated", "value": report_date},
        {"label": "Max Risk", "value": trade.get("max_risk", "N/A")},
        {"label": "Potential Reward", "value": trade.get("max_reward", "N/A")},
        {"label": "Volatility Forecast", "value": volatility.get("forecast") or "Unavailable"},
        {"label": "Sentiment Score", "value": sentiment.get("score", "N/A")},
    ]


def build_report_context(reports: Dict[str, Any], ticker: str, timestamp: str, results_path: str) -> Dict[str, Any]:
    derived = build_derived_fields(reports)
    trade_summary = extract_trade_summary(reports.get("trade_decision", {}) or {})
    reports["trade_summary"] = trade_summary
    report_dt = infer_report_datetime(reports)
    thesis_section = build_thesis_section(reports)
    trade_section = build_trade_section(reports, derived, trade_summary)
    technical_section = build_technical_section(reports, derived)
    fundamentals = build_fundamental_section(reports)
    volatility = build_volatility_section(reports, derived)
    sentiment = build_sentiment_section(reports, derived)
    risk_section = build_risk_section(reports)
    visuals = build_visual_assets(reports)
    executive_summary = build_executive_summary(trade_section, derived, thesis_section, fundamentals, volatility)
    market_snapshot_rows = build_market_snapshot_rows(derived)
    report_date = format_timestamp(report_dt)
    results_folder = os.path.basename(results_path)
    meta_items = build_meta_items(ticker, timestamp, report_date, results_folder, trade_section, volatility, sentiment)
    return {
        "ticker": ticker.upper(),
        "timestamp": timestamp,
        "report_date": report_date,
        "results_folder": results_folder,
        "meta": meta_items,
        "market_snapshot_rows": market_snapshot_rows,
        "executive_summary": executive_summary,
        "trade": trade_section,
        "thesis": thesis_section,
        "technical": technical_section,
        "fundamentals": fundamentals,
        "volatility": volatility,
        "sentiment": sentiment,
        "risk": risk_section,
        "visuals": visuals,
        "disclaimer": "This report is prepared by the Quant13 multi-agent research framework and does not constitute investment advice.",
    }


def render_html(context: Dict[str, Any]) -> str:
    template_dir = os.path.join(BASE_DIR, "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report_template.html")
    return template.render(context=context)


def create_pdf(html_string: str, output_file: str) -> None:
    base_url = os.path.join(BASE_DIR, "templates")
    HTML(string=html_string, base_url=base_url).write_pdf(output_file)


def build_output_path(base_dir: str, ticker: str, timestamp: str, output: Optional[str]) -> str:
    if output:
        return output
    filename = f"{ticker.upper()}_{timestamp}_report.pdf"
    return os.path.join(base_dir, filename)


def generate_pdf_report(
    ticker: str,
    timestamp: str,
    base_results_dir: Optional[str] = None,
    output: Optional[str] = None,
) -> str:
    base_dir = base_results_dir or os.path.join(BASE_DIR, "results")
    results_path = os.path.join(base_dir, f"{ticker.upper()}_{timestamp}")
    if not os.path.isdir(results_path):
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    reports = load_all_reports(results_path)
    context = build_report_context(reports, ticker, timestamp, results_path)

    html = render_html(context)
    output_path = build_output_path(results_path, ticker, timestamp, output)
    create_pdf(html, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Quant13 PDF trading report.")
    parser.add_argument("ticker", help="Ticker symbol used for the run")
    parser.add_argument(
        "timestamp",
        help="Timestamp suffix used in the results directory (e.g. YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--output", help="Optional output PDF path")
    args = parser.parse_args()

    output_path = generate_pdf_report(args.ticker, args.timestamp, output=args.output)
    print(f"PDF report generated at: {output_path}")


if __name__ == "__main__":
    main()
