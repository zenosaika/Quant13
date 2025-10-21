from __future__ import annotations

from typing import Dict

from src.agents.debate import DebateOrchestrator
from src.agents.fundamental import FundamentalAnalyst
from src.agents.risk import RiskManagementTeam
from src.agents.sentiment import SentimentAgent
from src.agents.technical import TechnicalAnalyst
from src.agents.trader import TraderAgent
from src.agents.volatility import VolatilityModelingAgent
from src.config import load_config
from src.data.fetcher import fetch_company_overview, fetch_news, fetch_ohlcv, fetch_options_chain
from src.data.preprocessing import compute_returns


def run_pipeline(ticker: str) -> Dict[str, object]:
    config = load_config()
    lookback = config["data"]["default_lookback_days"]

    ohlcv = fetch_ohlcv(ticker, lookback)
    ohlcv = compute_returns(ohlcv)
    options_chain = fetch_options_chain(ticker)
    news = fetch_news(ticker, config["data"]["news_limit"])
    overview = fetch_company_overview(ticker)

    base_state = {
        "ticker": ticker,
        "ohlcv": ohlcv,
        "options_chain": options_chain,
        "news": news,
        "company_overview": overview,
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

    trader = TraderAgent(config["agents"]["trader"]["prompt"])
    trade_proposal = trader.propose_trade(trade_thesis, volatility_report)

    risk_team = RiskManagementTeam()
    risk_assessment = risk_team.assess(trade_proposal, trade_thesis, volatility_report)

    return {
        "volatility_report": volatility_report,
        "sentiment_report": sentiment_report,
        "technical_report": technical_report,
        "fundamental_report": fundamental_report,
        "trade_thesis": trade_thesis,
        "trade_proposal": trade_proposal,
        "risk_assessment": risk_assessment,
    }
