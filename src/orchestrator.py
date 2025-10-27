from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from src.agents.debate import DebateOrchestrator
from src.agents.fundamental import FundamentalAnalyst
from src.agents.backtester import BacktesterAgent
from src.agents.risk import RiskManagementTeam
from src.agents.sentiment import SentimentAgent
from src.agents.technical import TechnicalAnalyst
from src.agents.trader import TraderAgent
from src.agents.volatility import VolatilityModelingAgent
from src.config import load_config
from src.data.fetcher import (
    fetch_company_overview,
    fetch_fundamental_bundle,
    fetch_news,
    fetch_ohlcv,
    fetch_options_chain,
)
from src.data.preprocessing import compute_returns

try:
    from generate_report import generate_pdf_report
except ModuleNotFoundError:  # pragma: no cover - fallback for environments without report module
    generate_pdf_report = None  # type: ignore


def run_pipeline(ticker: str) -> Dict[str, object]:
    config = load_config()
    lookback = config["data"]["default_lookback_days"]

    ohlcv = fetch_ohlcv(ticker, lookback)
    ohlcv = compute_returns(ohlcv)
    spot_price = float(ohlcv["close"].iloc[-1])
    options_chain = fetch_options_chain(ticker)
    news = fetch_news(ticker, config["data"]["news_limit"])
    fundamental_bundle = fetch_fundamental_bundle(ticker)
    overview = fundamental_bundle.get("info") or fetch_company_overview(ticker)

    base_state = {
        "ticker": ticker,
        "ohlcv": ohlcv,
        "options_chain": options_chain,
        "news": news,
        "company_overview": overview,
        "fundamental_bundle": fundamental_bundle,
    }

    volatility_agent = VolatilityModelingAgent(config["agents"]["volatility"])
    volatility_report = volatility_agent.run(base_state)

    sentiment_agent = SentimentAgent(config["agents"]["sentiment"])
    sentiment_report = sentiment_agent.run(base_state)

    technical_agent = TechnicalAnalyst(config["agents"]["technical"])
    technical_report = technical_agent.run(base_state)

    fundamental_agent = FundamentalAnalyst(config["agents"]["fundamental"])
    fundamental_report = fundamental_agent.run(base_state)

    reports_payload = {
        "volatility": volatility_report.model_dump(),
        "sentiment": sentiment_report.model_dump(),
        "technical": technical_report.model_dump(),
        "fundamental": fundamental_report.model_dump(),
    }

    debate_team = DebateOrchestrator(config["agents"]["debate"])
    trade_thesis = debate_team.conduct_debate(reports_payload)

    backtester = BacktesterAgent(config["agents"].get("backtester", {}))
    backtest_state = dict(base_state)
    backtest_state.update({
        "trade_thesis": trade_thesis,
        "volatility_report": volatility_report,
        "sentiment_report": sentiment_report,
        "technical_report": technical_report,
        "fundamental_report": fundamental_report,
    })
    backtest_report = backtester.run(backtest_state)

    trader = TraderAgent(config["agents"]["trader"])
    trade_proposal = trader.propose_trade(trade_thesis, volatility_report, options_chain, spot_price)
    trade_proposal = trade_proposal.model_copy(update={"conviction_level": trade_thesis.conviction_level})

    risk_team = RiskManagementTeam()
    risk_assessment = risk_team.assess(trade_proposal, trade_thesis, volatility_report)

    results = {
        "volatility_report": volatility_report,
        "sentiment_report": sentiment_report,
        "technical_report": technical_report,
        "fundamental_report": fundamental_report,
        "trade_thesis": trade_thesis,
        "trade_proposal": trade_proposal,
        "risk_assessment": risk_assessment,
        "backtest_report": backtest_report,
    }

    results_directory, timestamp = _persist_results(ticker, results)
    results["results_path"] = str(results_directory) if results_directory else None
    results["timestamp"] = timestamp

    if generate_pdf_report and results_directory:
        try:
            pdf_path = generate_pdf_report(
                ticker=ticker,
                timestamp=timestamp,
                base_results_dir=str(results_directory.parent),
            )
            results["report_pdf"] = pdf_path
        except Exception as exc:  # pragma: no cover - non-critical failure path
            results["report_pdf_error"] = str(exc)

    return results


def _persist_results(ticker: str, results: Dict[str, object]) -> tuple[Path | None, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_dir = Path(__file__).resolve().parents[1] / "results"
    target_dir = base_dir / f"{ticker.upper()}_{timestamp}"
    target_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "volatility_report.json": results["volatility_report"].model_dump(),
        "sentiment_report.json": results["sentiment_report"].model_dump(),
        "technical_report.json": results["technical_report"].model_dump(),
        "fundamental_report.json": results["fundamental_report"].model_dump(),
        "trade_thesis.json": results["trade_thesis"].model_dump(),
        "trade_decision.json": results["trade_proposal"].model_dump(),
        "risk_assessment.json": results["risk_assessment"].model_dump(),
        "backtest_report.json": results["backtest_report"].model_dump(),
    }

    for filename, payload in artifacts.items():
        try:
            with (target_dir / filename).open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError:
            continue

    return target_dir, timestamp
