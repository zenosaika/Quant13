"""
Enhanced Orchestrator with all improvements

Integrates:
- Black-Scholes Greeks calculation
- Enhanced sentiment analysis
- Systematic strategy selection
- Dual-language PDF generation
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# Original agents
from src.agents.debate import DebateOrchestrator
from src.agents.fundamental import FundamentalAnalyst
from src.agents.risk import RiskManagementTeam
from src.agents.technical import TechnicalAnalyst
from src.agents.volatility import VolatilityModelingAgent

# Enhanced agents
from src.agents.sentiment_v2 import EnhancedSentimentAgent
from src.agents.trader_v2 import SystematicTraderAgent

# Data fetchers
from src.config import load_config
from src.data.fetcher import (
    fetch_company_overview,
    fetch_fundamental_bundle,
    fetch_news,
    fetch_ohlcv,
    fetch_options_chain,
)
from src.data.preprocessing import compute_returns

# Pricing and Greeks
from src.pricing import (
    enrich_options_chain_with_greeks,
    fetch_risk_free_rate,
)

# Risk calculations
from src.utils.risk import calculate_risk_metrics

# PDF generation (if available)
try:
    from generate_report import generate_pdf_report
except ModuleNotFoundError:
    generate_pdf_report = None

# Discord integration (if enabled)
try:
    from src.integrations.discord_webhook import send_to_discord as discord_send
except ModuleNotFoundError:
    discord_send = None

logger = logging.getLogger(__name__)


def run_pipeline_v2(
    ticker: str,
    use_enhanced_sentiment: bool = True,
    use_systematic_trader: bool = True,
    generate_pdf: bool = True,
    dual_language: bool = False,
    send_to_discord: bool = False
) -> Dict[str, object]:
    """
    Enhanced pipeline with all improvements

    Pipeline Phases:
    1. Data Collection + Greeks enrichment
    2. Parallel Analysts (volatility, sentiment, technical, fundamental)
    3. Debate & Decision
    4. Trade Construction (systematic or LLM-based)
    5. Risk Metrics Calculation
    6. Risk Assessment
    7. Results Persistence
    8. PDF Generation (optional)

    Args:
        ticker: Stock ticker symbol
        use_enhanced_sentiment: Use enhanced multi-source sentiment agent
        use_systematic_trader: Use systematic rule-based trader
        generate_pdf: Generate PDF report
        dual_language: Generate both English and Thai PDFs
        send_to_discord: Send results to Discord via webhook

    Returns:
        Dictionary with all results
    """
    config = load_config()
    lookback = config["data"]["default_lookback_days"]

    logger.info(f"Starting enhanced pipeline for {ticker}")
    logger.info(f"Enhanced sentiment: {use_enhanced_sentiment}")
    logger.info(f"Systematic trader: {use_systematic_trader}")
    logger.info(f"Dual-language: {dual_language}")

    # ========================================================================
    # PHASE 1: DATA COLLECTION
    # ========================================================================
    logger.info("Phase 1: Collecting market data")

    ohlcv = fetch_ohlcv(ticker, lookback)
    ohlcv = compute_returns(ohlcv)
    spot_price = float(ohlcv["close"].iloc[-1])

    # Fetch options chain
    options_chain = fetch_options_chain(ticker)

    # Fetch risk-free rate
    try:
        risk_free_rate = fetch_risk_free_rate(duration_years=0.25)
        logger.info(f"Risk-free rate: {risk_free_rate*100:.2f}%")
    except Exception as e:
        logger.warning(f"Failed to fetch risk-free rate, using 5%: {e}")
        risk_free_rate = 0.05

    # Enrich options chain with computed Greeks
    try:
        logger.info("Enriching options chain with Black-Scholes Greeks")
        options_chain_enriched = enrich_options_chain_with_greeks(
            options_chain,
            spot_price,
            risk_free_rate
        )
        logger.info(f"Enriched {len(options_chain_enriched)} expiration groups")
    except Exception as e:
        logger.error(f"Greeks enrichment failed, using original chain: {e}")
        options_chain_enriched = options_chain

    # Fetch news and fundamental data
    news = fetch_news(ticker, config["data"]["news_limit"])
    fundamental_bundle = fetch_fundamental_bundle(ticker)
    overview = fundamental_bundle.get("info") or fetch_company_overview(ticker)

    base_state = {
        "ticker": ticker,
        "ohlcv": ohlcv,
        "options_chain": options_chain_enriched,
        "news": news,
        "company_info": overview,
        "company_overview": overview,
        "fundamental_bundle": fundamental_bundle,
        "risk_free_rate": risk_free_rate,
    }

    # ========================================================================
    # PHASE 2: PARALLEL ANALYST PHASE
    # ========================================================================
    logger.info("Phase 2: Running analyst agents in parallel")

    volatility_agent = VolatilityModelingAgent(config["agents"]["volatility"])
    technical_agent = TechnicalAnalyst(config["agents"]["technical"])
    fundamental_agent = FundamentalAnalyst(config["agents"]["fundamental"])

    # Use enhanced or original sentiment agent
    if use_enhanced_sentiment:
        logger.info("Using EnhancedSentimentAgent with multi-source analysis")
        sentiment_agent = EnhancedSentimentAgent(config["agents"]["sentiment"])
    else:
        logger.info("Using original SentimentAgent")
        from src.agents.sentiment import SentimentAgent
        sentiment_agent = SentimentAgent(config["agents"]["sentiment"])

    analyst_agents = {
        "volatility": (volatility_agent, base_state),
        "sentiment": (sentiment_agent, base_state),
        "technical": (technical_agent, base_state),
        "fundamental": (fundamental_agent, base_state),
    }

    analyst_results: Dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=len(analyst_agents)) as executor:
        futures = {
            executor.submit(agent.run, state): key
            for key, (agent, state) in analyst_agents.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                analyst_results[key] = future.result()
                logger.info(f"  {key} agent completed")
            except Exception as e:
                logger.error(f"  {key} agent failed: {e}")
                raise

    volatility_report = analyst_results["volatility"]
    sentiment_report = analyst_results["sentiment"]
    technical_report = analyst_results["technical"]
    fundamental_report = analyst_results["fundamental"]

    # ========================================================================
    # PHASE 3: DEBATE & DECISION
    # ========================================================================
    logger.info("Phase 3: Conducting debate and forming thesis")

    reports_payload = {
        "volatility": volatility_report.model_dump(),
        "sentiment": sentiment_report.model_dump(),
        "technical": technical_report.model_dump(),
        "fundamental": fundamental_report.model_dump(),
    }

    debate_team = DebateOrchestrator(config["agents"]["debate"])
    trade_thesis = debate_team.conduct_debate(reports_payload)
    logger.info(f"  Thesis: {trade_thesis.winning_argument} ({trade_thesis.conviction_level})")

    # ========================================================================
    # PHASE 4: TRADE CONSTRUCTION
    # ========================================================================
    logger.info("Phase 4: Generating trade proposal")

    if use_systematic_trader:
        logger.info("Using SystematicTraderAgent with rule-based selection")
        trader = SystematicTraderAgent(config["agents"]["trader"])
    else:
        logger.info("Using original TraderAgent (LLM-based)")
        from src.agents.trader import TraderAgent
        trader = TraderAgent(config["agents"]["trader"])

    trade_proposal = trader.propose_trade(
        trade_thesis,
        volatility_report,
        options_chain_enriched,
        spot_price
    )

    logger.info(f"  Proposed strategy: {trade_proposal.strategy_name}")
    logger.info(f"  Status: {trade_proposal.generation_status}")

    # ========================================================================
    # PHASE 4.5: THESIS-STRATEGY ALIGNMENT VALIDATION (NEW - Phase 1)
    # ========================================================================
    validation_config = config.get("validation", {})
    enable_validation = validation_config.get("enable_thesis_gate", True)
    strict_mode = validation_config.get("strict_mode", True)

    if enable_validation:
        logger.info("Phase 4.5: Validating thesis-strategy alignment")

        try:
            from src.validation import validate_thesis_alignment

            validation_result = validate_thesis_alignment(
                thesis=trade_thesis,
                proposal=trade_proposal,
                strict=strict_mode  # Use config setting
            )

            logger.info(f"  Validation: {validation_result['severity'].upper()}")
            logger.info(f"  {validation_result['message']}")

            if validation_result['suggestions']:
                for suggestion in validation_result['suggestions']:
                    logger.info(f"    • {suggestion}")

        except Exception as e:
            # If validation fails critically, log error and halt
            logger.error(f"  Validation FAILED: {e}")
            logger.error("  CRITICAL: Trade contradicts thesis!")

            if strict_mode:
                # In strict mode, raise the exception to halt pipeline
                raise
            else:
                # In non-strict mode, log warning and continue
                logger.warning("  Continuing despite validation failure (strict_mode=false)")
    else:
        logger.info("Phase 4.5: Thesis validation DISABLED (enable_thesis_gate=false)")

    # ========================================================================
    # PHASE 5: RISK METRICS CALCULATION
    # ========================================================================
    logger.info("Phase 5: Calculating risk metrics")

    risk_metrics = calculate_risk_metrics(trade_proposal, options_chain_enriched)

    trade_proposal = trade_proposal.model_copy(update={
        "conviction_level": trade_thesis.conviction_level,
        "max_risk": risk_metrics.get("max_risk"),
        "max_reward": risk_metrics.get("max_reward"),
        "net_premium": risk_metrics.get("net_premium"),
    })

    max_risk = risk_metrics.get('max_risk')
    max_reward = risk_metrics.get('max_reward')
    logger.info(f"  Max Risk: ${max_risk:,.2f}" if max_risk is not None else "  Max Risk: N/A")
    logger.info(f"  Max Reward: ${max_reward:,.2f}" if max_reward is not None else "  Max Reward: N/A")

    # ========================================================================
    # PHASE 6: RISK ASSESSMENT
    # ========================================================================
    logger.info("Phase 6: Conducting risk assessment")

    # FIX: Temporal Hallucination - Pass current date to Risk Agent
    current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    risk_team = RiskManagementTeam(config["agents"].get("risk", {}))
    risk_assessment = risk_team.assess(
        trade_proposal,
        trade_thesis,
        volatility_report,
        current_date=current_date_str
    )

    # ========================================================================
    # PHASE 6.5: FUND MANAGER DECISION (NEW - CRITICAL FIX)
    # ========================================================================
    logger.info("Phase 6.5: Fund Manager final decision")

    from src.agents.manager import FundManagerAgent

    manager = FundManagerAgent(config["agents"].get("manager", {}))
    final_decision = manager.run({
        "trade_proposal": trade_proposal,
        "risk_assessment": risk_assessment,
        "trade_thesis": trade_thesis,
    })

    logger.info(f"  Execute trade: {final_decision['execute_trade']}")
    logger.info(f"  Position sizing: {final_decision['final_sizing']}")
    logger.info(f"  Rationale: {final_decision['manager_rationale']}")

    # ========================================================================
    # PHASE 7: RESULTS PERSISTENCE
    # ========================================================================
    logger.info("Phase 7: Persisting results")

    results = {
        "volatility_report": volatility_report,
        "sentiment_report": sentiment_report,
        "technical_report": technical_report,
        "fundamental_report": fundamental_report,
        "trade_thesis": trade_thesis,
        "trade_proposal": trade_proposal,
        "risk_assessment": risk_assessment,
        "final_decision": final_decision,  # NEW: Add Fund Manager decision
        "risk_free_rate": risk_free_rate,
    }

    # If trade rejected by Fund Manager, return early
    if not final_decision["execute_trade"]:
        logger.warning("Trade REJECTED by Fund Manager")
        results["status"] = "rejected"
        results["rejection_reason"] = final_decision["manager_rationale"]

        # Still persist results for audit trail
        results_directory, timestamp = _persist_results(ticker, results)
        results["results_path"] = str(results_directory) if results_directory else None
        results["timestamp"] = timestamp

        return results

    # Apply position sizing for approved trades
    sizing_multiplier = {
        "full": 1.0,
        "half": 0.5,
        "quarter": 0.25,
        "none": 0.0
    }.get(final_decision["final_sizing"], 0.5)

    results["position_sizing_multiplier"] = sizing_multiplier
    logger.info(f"  Applied sizing: {sizing_multiplier*100:.0f}%")

    results_directory, timestamp = _persist_results(ticker, results)
    results["results_path"] = str(results_directory) if results_directory else None
    results["timestamp"] = timestamp

    # ========================================================================
    # PHASE 8: PDF GENERATION (Optional)
    # ========================================================================
    if generate_pdf and generate_pdf_report and results_directory:
        try:
            logger.info("Phase 8: Generating PDF report(s)")

            if dual_language:
                logger.info("  Generating dual-language PDFs (English + Thai)")
                # TODO: Implement dual-language generation
                # For now, generate English only
                pdf_path = generate_pdf_report(
                    ticker=ticker,
                    timestamp=timestamp,
                    base_results_dir=str(results_directory.parent),
                )
                results["report_pdf_english"] = pdf_path
                results["report_pdf_thai"] = None  # Placeholder
                logger.info(f"  English PDF: {pdf_path}")
            else:
                logger.info("  Generating English PDF")
                pdf_path = generate_pdf_report(
                    ticker=ticker,
                    timestamp=timestamp,
                    base_results_dir=str(results_directory.parent),
                )
                results["report_pdf"] = pdf_path
                logger.info(f"  PDF: {pdf_path}")

        except Exception as exc:
            logger.error(f"PDF generation failed: {exc}", exc_info=True)
            results["report_pdf_error"] = str(exc)

    # ========================================================================
    # PHASE 9: DISCORD NOTIFICATION (Optional)
    # ========================================================================
    if send_to_discord and discord_send and results_directory:
        try:
            logger.info("Phase 9: Sending Discord notification")

            # Check if Discord is enabled in config
            discord_config = config.get("discord", {})
            if not discord_config.get("enabled", False):
                logger.warning("  Discord is disabled in config.yaml. Skipping notification.")
            else:
                discord_success = discord_send(
                    ticker=ticker,
                    result_dir=results_directory
                )
                results["discord_sent"] = discord_success
                if discord_success:
                    logger.info("  Discord notification sent successfully!")
                else:
                    logger.warning("  Discord notification failed")
        except Exception as exc:
            logger.error(f"Discord notification failed: {exc}", exc_info=True)
            results["discord_error"] = str(exc)

    logger.info(f"Pipeline completed successfully for {ticker}")
    return results


def _persist_results(ticker: str, results: Dict[str, object]) -> tuple[Path | None, str]:
    """
    Persist all results to JSON files

    Args:
        ticker: Stock ticker
        results: Results dictionary

    Returns:
        Tuple of (results_directory, timestamp)
    """
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
        "final_decision.json": results["final_decision"],  # NEW: Save Fund Manager decision
    }

    for filename, payload in artifacts.items():
        try:
            with (target_dir / filename).open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                logger.debug(f"  Saved {filename}")
        except OSError as e:
            logger.error(f"  Failed to save {filename}: {e}")
            continue

    logger.info(f"Results saved to: {target_dir}")
    return target_dir, timestamp


# Convenience function to run with defaults
def run_pipeline(ticker: str) -> Dict[str, object]:
    """
    Run pipeline with all enhancements enabled (backward compatible)

    Args:
        ticker: Stock ticker symbol

    Returns:
        Results dictionary
    """
    return run_pipeline_v2(
        ticker,
        use_enhanced_sentiment=True,
        use_systematic_trader=True,
        generate_pdf=True,
        dual_language=False
    )
