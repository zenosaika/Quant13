from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.sentiment import SentimentAgent
from src.agents.technical import TechnicalAnalyst
from src.agents.volatility import VolatilityModelingAgent
from src.models.schemas import VolatilityReport
from src.orchestrator import run_pipeline


def _sample_ohlcv(rows: int = 60) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=rows, freq="B")
    prices = np.linspace(100, 120, rows)
    data = {
        "open": prices,
        "high": prices + 1,
        "low": prices - 1,
        "close": prices,
        "volume": np.full(rows, 1_000_000),
    }
    return pd.DataFrame(data, index=dates)


def _sample_options_chain() -> list[dict]:
    calls_front = pd.DataFrame({
        "contractSymbol": ["TESTC1", "TESTC2"],
        "strike": [110, 115],
        "lastPrice": [4.0, 2.5],
        "bid": [3.8, 2.3],
        "ask": [4.2, 2.7],
        "impliedVolatility": [0.35, 0.32],
    })
    puts_front = pd.DataFrame({
        "contractSymbol": ["TESTP1", "TESTP2"],
        "strike": [90, 85],
        "lastPrice": [3.5, 4.0],
        "bid": [3.3, 3.8],
        "ask": [3.7, 4.2],
        "impliedVolatility": [0.4, 0.42],
    })
    calls_back = calls_front.assign(impliedVolatility=[0.3, 0.28])
    puts_back = puts_front.assign(impliedVolatility=[0.34, 0.33])
    return [
        {"expiration": "2024-12-20", "calls": calls_front, "puts": puts_front},
        {"expiration": "2025-01-17", "calls": calls_back, "puts": puts_back},
    ]


def test_volatility_agent_outputs_report():
    agent = VolatilityModelingAgent({"garch_forecast_placeholder": "Stable"})
    report = agent.run({
        "ticker": "TEST",
        "ohlcv": _sample_ohlcv(),
        "options_chain": _sample_options_chain(),
    })
    assert isinstance(report, VolatilityReport)
    assert 0 <= report.iv_rank <= 100


def test_sentiment_agent_scores_news():
    agent = SentimentAgent({
        "prompt": "You are a News & Sentiment Analyst.",
    })
    report = agent.run({
        "ticker": "TEST",
        "news": [
            {"title": "Company announces production surge", "summary": "Strong demand"},
            {"title": "Temporary production halt reported"},
        ],
    })
    assert report.articles
    assert -1.0 <= report.overall_sentiment_score <= 1.0


def test_technical_agent_generates_summary():
    agent = TechnicalAnalyst({
        "prompt": "You are an expert Technical Analyst.",
        "sma_periods": [5, 10],
        "macd": {"short_window": 3, "long_window": 6, "signal_window": 2},
        "rsi_window": 5,
    })
    report = agent.run({
        "ticker": "TEST",
        "ohlcv": _sample_ohlcv(40),
    })
    assert report.indicators.latest_close is not None
    assert "summary" in report.llm_report

def test_pipeline_runs_end_to_end():
    def side_effect(messages, *args, **kwargs):
        user_payload = messages[-1]["content"] if messages else ""
        if '"stance": "Bullish"' in user_payload:
            return "Bullish argument with strong evidence."
        if '"stance": "Bearish"' in user_payload:
            return "Bearish argument citing risks."
        if '"transcript"' in user_payload:
            return json.dumps({
                "winning_argument": "Bullish",
                "conviction_level": "Medium",
                "summary": "Bullish wins",
                "key_evidence": ["Evidence"],
            })
        if '"options_chain"' in user_payload:
            return json.dumps({
                "strategy_name": "Call Spread",
                "action": "BUY_TO_OPEN",
                "quantity": 1,
                "trade_legs": [
                    {
                        "contract_symbol": "TESTC1",
                        "type": "CALL",
                        "action": "BUY",
                        "strike_price": 110,
                        "expiration_date": "2024-12-20",
                        "quantity": 1,
                        "key_greeks_at_selection": {"delta": 0.5, "gamma": 0.1, "theta": -0.02, "vega": 0.05, "impliedVolatility": 0.3},
                    }
                ],
                "notes": "Structured via test double.",
            })
        if "\"articles\"" in user_payload:
            return json.dumps({
                "articles": [
                    {"title": "Analysts upgrade", "sentiment_score": 0.4, "rationale": "Positive analyst commentary."}
                ],
                "overall_sentiment_score": 0.4,
                "overall_summary": "Sentiment positive.",
            })
        if "\"indicators\"" in user_payload:
            return json.dumps({
                "technical_bias": "bullish",
                "primary_trend": "Uptrend above moving averages.",
                "momentum": "RSI rising.",
                "volatility_levels": "Bands expanding.",
                "key_levels": {"support": 100, "resistance": 120},
                "summary": "Momentum supportive of upside continuation.",
            })
        if "MD&A section" in user_payload:
            return json.dumps({
                "tone": "optimistic",
                "performance_drivers": ["Strong demand"],
                "forward_looking": ["Investing in R&D"],
            })
        if "Risk Factors" in user_payload:
            return json.dumps([
                {"risk": "Competition", "category": "Market", "rationale": "Peers investing heavily."}
            ])
        if "\"qualitative_summaries\"" in user_payload:
            return json.dumps({
                "swot": {
                    "strengths": ["Brand loyalty"],
                    "weaknesses": ["High costs"],
                    "opportunities": ["New markets"],
                    "threats": ["Competition"],
                },
                "financial_health": "Strong",
                "overall_thesis": "bullish",
                "justification": "Solid growth trajectory.",
            })
        return "Fallback response"

    class DummyResponse:
        def __init__(self) -> None:
            self.text = "ITEM 7. Management discussion\nItem 8. Financial Statements"
            self.headers = {"Content-Type": "text/plain"}

        def raise_for_status(self) -> None:
            return None

    with patch("src.tools.llm.LLMClient.chat", side_effect=side_effect), \
        patch("src.orchestrator.fetch_ohlcv", return_value=_sample_ohlcv(90)), \
        patch("src.orchestrator.fetch_options_chain", return_value=_sample_options_chain()), \
        patch("src.orchestrator.fetch_news", return_value=[{"title": "Analysts upgrade"}]), \
        patch("src.orchestrator.fetch_company_overview", return_value={"trailingPE": 20, "grossMargins": 0.45, "debtToEquity": 0.5}), \
        patch("src.orchestrator.fetch_fundamental_bundle", return_value={
            "info": {"trailingPE": 20, "priceToSalesTrailing12Months": 4.0, "debtToEquity": 0.5},
            "financials": pd.DataFrame({}),
            "balance_sheet": pd.DataFrame({}),
            "cashflow": pd.DataFrame({}),
            "filings": pd.DataFrame({"filingType": ["10-K"], "filingDate": ["2024-01-01"], "linkToTxt": ["https://example.com/filing"]}),
        }), \
        patch("src.agents.fundamental.requests.get", return_value=DummyResponse()):

        results = run_pipeline("TEST")
    assert "trade_proposal" in results
    assert results["trade_proposal"].trade_legs
