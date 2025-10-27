from __future__ import annotations

import json
from unittest.mock import Mock, patch

from generate_report import compose_llm_sections, build_trade_section, build_executive_summary


def _base_context() -> tuple[dict, dict]:
    context = {
        "ticker": "TEST",
        "report_date": "October 27, 2025",
        "timestamp": "20251027_120000",
        "results_folder": "TEST_20251027_120000",
        "market_snapshot": {"price": "$123.45", "iv_rank": "42.1", "technical_bias": "Bullish"},
        "executive_summary": {
            "headline": "Bullish thesis supported by cross-agent consensus",
            "bullets": ["Directional bias remains constructive", "Defined risk via debit spread"],
        },
        "trade": {
            "direction": "Bullish",
            "strategy": "Debit Call Spread",
            "conviction": "High",
            "action": "BUY_TO_OPEN",
            "quantity": "1",
            "max_risk": "$350",
            "max_reward": "$650",
            "highlights": [{"label": "Underlying Price", "value": "$120.00"}],
            "legs": [
                {
                    "action": "Buy",
                    "type": "Call",
                    "symbol": "TESTC1",
                    "expiration": "January 17, 2026",
                    "strike": "$120",
                    "greeks": "Delta: 0.55",
                }
            ],
            "justification_html": "<p>Spread aligns with bullish thesis while containing risk.</p>",
            "days_to_expiration": 90,
        },
        "thesis": {
            "headline": "Bullish argument narrowly prevails",
            "narrative_plain": "Bullish evidence outweighs risks on near-term catalysts.",
            "plan_rows": [],
        },
        "technical": {
            "bias": "Bullish",
            "summary_html": "<p>Momentum and breadth indicators confirm upside.</p>",
            "indicators": [{"label": "RSI", "value": "65", "context": "Rising"}],
            "key_levels": [{"label": "Support", "value": "$115"}],
            "patterns": [],
        },
        "fundamentals": {
            "business_summary_html": "<p>Company maintains double-digit revenue growth.</p>",
            "ratios": [{"label": "P/E", "value": "25"}],
            "trend_tables": [],
            "mdna": {
                "tone": "constructive",
                "performance_drivers": ["AI demand"],
                "opportunities": ["Cloud uptake"],
                "concerns": ["FX headwinds"],
                "narrative_html": "<p>Management highlighted AI tailwinds.</p>",
            },
            "risk_matrix": [{"risk": "Competition", "category": "Market", "rationale": "Peers invest heavily."}],
        },
        "volatility": {
            "metrics": [{"label": "IV Rank", "value": "82"}],
            "forecast": "Expect elevated implied vol to persist near-term.",
        },
        "sentiment": {
            "score": "0.4",
            "summary": "Positive skew in recent headlines.",
            "articles": [
                {
                    "title": "Analysts upgrade outlook",
                    "publisher": "Newswire",
                    "published": "October 26, 2025",
                    "summary": "Brokerages cite AI demand.",
                    "link": "https://example.com",
                }
            ],
        },
        "risk": {
            "final": "Proceed with defined-risk structure",
            "adjustments": [{"profile": "Neutral", "recommendation": "Size modestly"}],
        },
        "backtest": {
            "strategy_type": "Directional Options",
            "summary": "Simulated trades deliver a 65% win rate with controlled drawdowns.",
            "metrics": [
                {"label": "Win Rate", "value": "65.0%"},
                {"label": "Simulated Trades", "value": "10"},
            ],
        },
        "visuals": {"trend_charts": []},
        "disclaimer": "Test disclaimer",
    }
    reports = {"trade_thesis": {"winning_argument": "Bullish", "conviction_level": "Medium", "key_evidence": []}}
    return context, reports


def test_compose_llm_sections_uses_llm_output():
    context, reports = _base_context()
    mock_client = Mock()
    mock_client.chat.return_value = json.dumps({
        "hero_html": "<section><h1>Hero</h1></section>",
        "sections": [{"title": "Overview", "html": "<p>Details</p>"}],
    })

    with patch("generate_report.get_llm_client", return_value=mock_client):
        bundle = compose_llm_sections(context, reports)

    assert "Hero" in bundle["content"]["hero_html"]
    assert bundle["content"]["sections"][0]["title"] == "Overview"
    mock_client.chat.assert_called_once()


def test_compose_llm_sections_fallback_on_failure():
    context, reports = _base_context()
    mock_client = Mock()
    mock_client.chat.side_effect = RuntimeError("downstream error")

    with patch("generate_report.get_llm_client", return_value=mock_client):
        bundle = compose_llm_sections(context, reports)

    assert "Options Outlook" in bundle["content"]["hero_html"]
    assert bundle["content"]["sections"]


def test_build_trade_section_syncs_conviction_and_dte():
    reports = {
        "trade_thesis": {"conviction_level": "High"},
        "trade_decision": {
            "action": "BUY_TO_OPEN",
            "strategy_name": "Bull Call Spread",
            "trade_legs": [
                {
                    "expiration_date": "2025-12-19",
                    "strike_price": 100,
                    "contract_symbol": "TESTC1",
                    "type": "CALL",
                    "action": "BUY",
                    "quantity": 1,
                }
            ],
        },
    }
    derived = {}
    summary = {
        "details": {"strategy": "Bull Call Spread"},
        "conviction": "Low",
        "computed_dte": 45,
    }

    trade_section = build_trade_section(reports, derived, summary)
    assert trade_section["conviction"] == "High"
    dte_highlight = next((item for item in trade_section["highlights"] if item["label"] == "Days to Expiration"), None)
    assert dte_highlight is not None
    assert dte_highlight["value"] == "45 days"


def test_executive_summary_headline_strips_markdown():
    trade_section = {
        "direction": "Bullish",
        "strategy": "Bull Call Spread",
        "conviction": "High",
        "max_risk": "$100",
        "max_reward": "$200",
    }
    derived = {}
    thesis_section = {"headline": "## TRADE THESIS: LLY", "narrative_plain": ""}
    fundamentals = {}
    volatility = {}

    summary = build_executive_summary(trade_section, derived, thesis_section, fundamentals, volatility)
    assert "##" not in summary["headline"]
    assert summary["headline"].startswith("TRADE THESIS")
