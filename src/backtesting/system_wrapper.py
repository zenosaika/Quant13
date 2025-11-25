"""
Wrapper to run the main Quant13 system in backtest mode

This module provides a function that runs the main multi-agent system
with historical data only (no future data leakage) and handles the
no-news limitation for backtesting.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.agents.debate import DebateOrchestrator
from src.agents.fundamental import FundamentalAnalyst
from src.agents.technical import TechnicalAnalyst
from src.agents.volatility import VolatilityModelingAgent
from src.config import load_config
from src.models.schemas import SentimentReport, TradeProposal, ArticleSentiment
from src.pricing.greeks_engine import enrich_options_chain_with_greeks

# Use systematic trader for deterministic results
from src.agents.trader_v2 import SystematicTraderAgent

logger = logging.getLogger(__name__)


def generate_synthetic_sentiment(ohlcv: pd.DataFrame, ticker: str) -> SentimentReport:
    """
    Generates sentiment based on Mean Reversion logic.

    VARIANCE HARVESTING STRATEGY:
    - Default to NEUTRAL (0.0 score) → Forces Iron Condors
    - Only directional if STRONG trend confirmed (>3% SMA separation)
    - Uses RSI for mean reversion signals

    Philosophy: If price is chopping, output NEUTRAL → System sells premium
    This beats trying to predict direction in choppy markets.

    Args:
        ohlcv: Historical OHLCV dataframe
        ticker: Stock ticker symbol

    Returns:
        SentimentReport optimized for variance harvesting
    """
    if len(ohlcv) < 50:
        return SentimentReport(
            agent="SyntheticMeanReversion",
            ticker=ticker,
            overall_sentiment_score=0.0,
            overall_summary="Insufficient data",
            articles=[]
        )

    # 1. Trend Strength (SMA Separation)
    sma20 = ohlcv['close'].rolling(20).mean().iloc[-1]
    sma50 = ohlcv['close'].rolling(50).mean().iloc[-1]
    trend_strength = abs(sma20 - sma50) / sma50

    # 2. RSI (Mean Reversion)
    delta = ohlcv['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]

    # 3. Bollinger Bandwidth (Volatility Context)
    recent = ohlcv.iloc[-20:]
    std = recent['close'].std()
    mean_price = recent['close'].mean()
    bb_width = (4 * std) / mean_price  # Normalized volatility

    # 4. MACD (Momentum - CRITICAL FIX #2)
    # Calculate MACD to avoid trading against momentum
    ema12 = ohlcv['close'].ewm(span=12, adjust=False).mean()
    ema26 = ohlcv['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_crossover = "bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "bearish"

    # ========================================================================
    # FIXED: SIMPLIFIED SIGNAL HIERARCHY (Clear Priority)
    # ========================================================================
    # Priority Order:
    # 1. RSI Extremes (>70 or <30) → NEUTRAL (avoid chasing parabolic moves)
    # 2. Strong Trends (>7% SMA separation + neutral RSI) → DIRECTIONAL
    # 3. MACD Confirmation → BOOST directional signals
    # 4. Default → NEUTRAL (variance harvesting)
    # ========================================================================
    score = 0.0
    regime = "Choppy/Neutral"
    summary = ""
    articles = []

    # PRIORITY 1: RSI Extremes → Force Neutral (highest priority)
    if curr_rsi > 70 or curr_rsi < 30:
        score = 0.0  # NEUTRAL → Iron Condor
        regime = "Overbought" if curr_rsi > 70 else "Oversold"
        summary = f"RSI extreme ({curr_rsi:.1f}). Avoid directional bias. Default to neutral strategies."
        articles.append(ArticleSentiment(
            title=f"{ticker} RSI extreme - neutral strategy",
            publisher="Mean Reversion Logic",
            sentiment_score=0.0,
            rationale=f"RSI {curr_rsi:.1f} {'overbought' if curr_rsi > 70 else 'oversold'}. Don't chase extremes."
        ))

    # PRIORITY 2: Strong Trend (only if RSI is neutral 30-70)
    elif trend_strength > 0.07 and 30 <= curr_rsi <= 70:
        # Very strong trend (>7% SMA separation)
        direction = 1 if sma20 > sma50 else -1
        score = 0.6 * direction
        regime = "Strong Uptrend" if direction > 0 else "Strong Downtrend"
        summary = f"Strong trend. SMA separation: {trend_strength:.1%}, RSI: {curr_rsi:.1f}."
        articles.append(ArticleSentiment(
            title=f"{ticker} strong trend",
            publisher="Mean Reversion Logic",
            sentiment_score=score,
            rationale=f"Strong trend ({trend_strength:.1%}) with neutral RSI ({curr_rsi:.1f})."
        ))

    # PRIORITY 3: Moderate Trend (5-7% SMA separation)
    elif trend_strength > 0.05 and 30 <= curr_rsi <= 70:
        direction = 1 if sma20 > sma50 else -1
        score = 0.4 * direction
        regime = "Moderate Uptrend" if direction > 0 else "Moderate Downtrend"
        summary = f"Moderate trend. SMA separation: {trend_strength:.1%}, RSI: {curr_rsi:.1f}."
        articles.append(ArticleSentiment(
            title=f"{ticker} moderate trend",
            publisher="Mean Reversion Logic",
            sentiment_score=score,
            rationale=f"Moderate trend ({trend_strength:.1%}) with neutral RSI ({curr_rsi:.1f})."
        ))

    # PRIORITY 4: Default Neutral (most common case)
    else:
        score = 0.0
        regime = "Range-Bound"
        summary = f"Choppy market. RSI: {curr_rsi:.1f}, Trend: {trend_strength:.1%}."
        articles.append(ArticleSentiment(
            title=f"{ticker} range-bound",
            publisher="Mean Reversion Logic",
            sentiment_score=0.0,
            rationale=f"No clear trend. Favor neutral strategies."
        ))

    # ========================================================================
    # MACD CONFIRMATION: Boost directional signals if MACD agrees
    # ========================================================================
    # Only boost existing directional signals, never override neutral signals
    if score > 0.3 and macd_crossover == "bullish":
        logger.info(f"MACD confirms bullish trend (score {score:.2f}) → boosting to 0.7")
        score = 0.7
        regime = "MACD Confirmed Uptrend"
        summary = f"MACD bullish confirms uptrend. Strong buy signal."

    elif score < -0.3 and macd_crossover == "bearish":
        logger.info(f"MACD confirms bearish trend (score {score:.2f}) → boosting to -0.7")
        score = -0.7
        regime = "MACD Confirmed Downtrend"
        summary = f"MACD bearish confirms downtrend. Strong sell signal."

    return SentimentReport(
        agent="SyntheticMeanReversion",
        ticker=ticker,
        overall_sentiment_score=score,
        overall_summary=summary,
        articles=articles
    )


def run_system_backtest(
    ticker: str,
    ohlcv: pd.DataFrame,
    options_chain: list[Dict[str, Any]],
    spot_price: float,
    fundamental_bundle: Dict[str, Any] = None,
    risk_free_rate: float = 0.05,
    return_reports: bool = False,
) -> TradeProposal | tuple[TradeProposal, Dict[str, Any]]:
    """
    Run the Quant13 system for backtesting

    This function runs the main multi-agent system but with:
    - No news data (sentiment agent returns neutral sentiment)
    - Only historical data up to the backtest date
    - Systematic trader for deterministic results

    Args:
        ticker: Stock ticker
        ohlcv: Historical OHLCV data up to backtest date
        options_chain: Historical options chain
        spot_price: Current spot price
        fundamental_bundle: Fundamental data (optional, will use empty if not provided)
        risk_free_rate: Risk-free rate
        return_reports: If True, return (trade_proposal, reports_dict)

    Returns:
        TradeProposal if return_reports=False
        (TradeProposal, reports_dict) if return_reports=True
    """
    config = load_config()

    # ========================================================================
    # DTE FILTERING: Avoid short-dated options (institutional minimum 21 DTE)
    # ========================================================================
    # Expert recommendation: 21-60 DTE to avoid "weekly lottery" behavior
    # Gives trades time to work, reduces gamma risk
    MIN_DTE = 21
    MAX_DTE = 60

    # Filter options chain for institutional DTE range
    valid_chains = []
    for expiry in options_chain:
        # Check if already enriched
        if "time_to_expiration_years" in expiry:
            dte = expiry.get('time_to_expiration_years', 0) * 365
        else:
            # Bug #10 FIX: Use 'expiration' key (not 'expiration_date')
            # historical_options.py uses "expiration": exp_date.strftime("%Y-%m-%d")
            exp_date_str = expiry.get('expiration') or expiry.get('expiration_date', '')
            if exp_date_str:
                try:
                    exp_date = datetime.fromisoformat(exp_date_str.replace('Z', '+00:00'))
                    current_date = ohlcv.index[-1]
                    # Handle timezone-aware index
                    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
                        current_date = current_date.replace(tzinfo=None)
                    dte = (exp_date - current_date).days
                except:
                    dte = 30  # Default fallback
            else:
                dte = 30  # Default fallback

        # Filter for institutional DTE range
        if MIN_DTE <= dte <= MAX_DTE:
            valid_chains.append(expiry)

    # Fallback if filtering removes everything (keep at least some options)
    if len(valid_chains) == 0:
        logger.warning(f"DTE filtering removed all options, using original chain")
        filtered_chain = options_chain
    else:
        logger.info(f"DTE filtering: {len(options_chain)} → {len(valid_chains)} expirations ({MIN_DTE}-{MAX_DTE} days)")
        filtered_chain = valid_chains

    # Enrich options chain with Greeks if not already enriched
    if not filtered_chain or "time_to_expiration_years" not in filtered_chain[0]:
        logger.info("Enriching options chain with Greeks")
        filtered_chain = enrich_options_chain_with_greeks(
            filtered_chain, spot_price, risk_free_rate
        )

    # Use filtered chain for all downstream processing
    options_chain = filtered_chain

    # Prepare base state
    base_state = {
        "ticker": ticker,
        "ohlcv": ohlcv,
        "options_chain": options_chain,
        "news": [],  # No news for backtesting
        "company_info": fundamental_bundle.get("info", {}) if fundamental_bundle else {},
        "company_overview": fundamental_bundle.get("info", {}) if fundamental_bundle else {},
        "fundamental_bundle": fundamental_bundle or {},
        "risk_free_rate": risk_free_rate,
    }

    # ========================================================================
    # PHASE 1: PARALLEL ANALYST PHASE
    # ========================================================================
    logger.info("Running analyst agents (synthetic sentiment mode)")

    volatility_agent = VolatilityModelingAgent(config["agents"]["volatility"])
    technical_agent = TechnicalAnalyst(config["agents"]["technical"])
    fundamental_agent = FundamentalAnalyst(config["agents"]["fundamental"])

    # Generate synthetic sentiment from price/volume action
    sentiment_report = generate_synthetic_sentiment(ohlcv, ticker)
    logger.info(f"  Synthetic sentiment: {sentiment_report.overall_sentiment_score:.2f}")

    # Run other analysts in parallel
    analyst_agents = {
        "volatility": (volatility_agent, base_state),
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
    technical_report = analyst_results["technical"]
    fundamental_report = analyst_results["fundamental"]

    # ========================================================================
    # PHASE 2: DEBATE & DECISION
    # ========================================================================
    logger.info("Conducting debate")

    reports_payload = {
        "volatility": volatility_report.model_dump(),
        "sentiment": sentiment_report.model_dump(),
        "technical": technical_report.model_dump(),
        "fundamental": fundamental_report.model_dump(),
    }

    debate_team = DebateOrchestrator(config["agents"]["debate"])
    trade_thesis = debate_team.conduct_debate(reports_payload)
    logger.info(f"  Thesis: {trade_thesis.winning_argument}")
    logger.info(f"  Conviction (original): {trade_thesis.conviction_level}")

    # ========================================================================
    # CRITICAL FIX: CONVICTION & DIRECTION ALIGNMENT
    # ========================================================================
    # In backtesting, the LLM (Debate) often defaults to "Low" conviction because
    # it sees no "Real News". We must override this if the Synthetic Signal (Math) is strong.
    #
    # The "Hallucination Gap" Problem:
    # - Synthetic Sentiment says: "Score: 0.85 (Strong Bullish)" or "-0.85 (Strong Bearish)"
    # - Debate Agent sees: No actual news articles to analyze
    # - LLM concludes: "Indicators are good, but insufficient textual evidence → Low"
    # - Result: All trades killed by conviction filter
    #
    # Solution: In backtest mode, align BOTH conviction AND direction with mathematical reality

    # 1. Get the raw score (Signed: Positive = Bullish, Negative = Bearish)
    raw_score = sentiment_report.overall_sentiment_score
    synthetic_magnitude = abs(raw_score)
    original_conviction = trade_thesis.conviction_level
    original_direction = trade_thesis.winning_argument

    # 2. Align Conviction Level
    if synthetic_magnitude >= 0.8:
        trade_thesis.conviction_level = "High"
        logger.info(f"  [Conviction Override] {original_conviction} → High (Signal Magnitude: {synthetic_magnitude:.2f})")
    elif synthetic_magnitude >= 0.4 and trade_thesis.conviction_level.lower() == "low":
        trade_thesis.conviction_level = "Medium"
        logger.info(f"  [Conviction Override] {original_conviction} → Medium (Signal Magnitude: {synthetic_magnitude:.2f})")
    else:
        logger.info(f"  [Conviction Unchanged] {trade_thesis.conviction_level} (Signal Magnitude: {synthetic_magnitude:.2f})")

    # 3. Align Direction (CRITICAL: Forces bearish strategies in downtrends)
    # If math says "CRASH" (-0.85) but Debate says "Bullish", FORCE Bearish
    if raw_score <= -0.4:
        logger.info(f"  [Direction Override] {original_direction} → BEARISH (Raw Score: {raw_score:.2f})")
        trade_thesis.winning_argument = "Bearish"
        # Update summary to ensure Trader agent picks bearish strategies
        trade_thesis.summary = f"Strong bearish technical and sentiment signals (Score: {raw_score:.2f}). Downside momentum confirmed. Recommend bearish positioning."
    elif raw_score >= 0.4:
        if original_direction.lower() != "bullish":
            logger.info(f"  [Direction Override] {original_direction} → BULLISH (Raw Score: {raw_score:.2f})")
            trade_thesis.winning_argument = "Bullish"
            trade_thesis.summary = f"Strong bullish technical and sentiment signals (Score: {raw_score:.2f}). Upside momentum confirmed. Recommend bullish positioning."
    else:
        logger.info(f"  [Direction Unchanged] {trade_thesis.winning_argument} (Raw Score: {raw_score:.2f})")

    # ========================================================================
    # 4. TECHNICAL ALIGNMENT (Moved to after trade construction - see below)
    # ========================================================================
    # Technical alignment now happens AFTER we extract tech_bias properly
    # in the trade construction phase (lines ~459-492)

    # ========================================================================
    # ADAPTIVE STRATEGY SELECTION: Match strategy to market regime
    # ========================================================================
    # CRITICAL FIX #5: Don't force variance harvesting in ALL markets
    #
    # Market Regime Detection:
    # 1. High IV + Choppy → Variance Harvesting (Iron Condor, sell premium)
    # 2. Low IV + Trending → Directional Trading (Bull/Bear Call Spreads)
    # 3. Low IV + Choppy → Cash (no edge either way)
    #
    # The strategy selector will handle this via the IV < 20 filter that
    # prevents selling premium. We just need to allow trades to proceed
    # so it can choose directional strategies when appropriate.
    #
    # Removed the hard IV < 30 rejection to allow participation in trending markets.

    # ========================================================================
    # CONVICTION FILTER: Don't trade with low conviction!
    # ========================================================================
    # CRITICAL FIX #4: Allow low conviction trades if signal is NEUTRAL
    #
    # Reasoning:
    # - Low conviction + directional = BAD (don't guess direction)
    # - Low conviction + neutral (0.00 score) = GOOD (Iron Condor opportunity!)
    #
    # When score is near 0.00, it means "choppy/uncertain direction" which is
    # EXACTLY when variance harvesting strategies (Iron Condor, Iron Butterfly)
    # perform best. We should TRADE these setups, not skip them!
    #
    is_neutral_signal = abs(raw_score) < 0.2  # Score within ±0.2 of neutral

    if trade_thesis.conviction_level.lower() in ["low", "weak", "uncertain"]:
        if is_neutral_signal:
            # Low conviction + neutral signal = Iron Condor opportunity
            logger.info(f"✅ Low conviction BUT neutral signal ({raw_score:.2f}) - allowing neutral strategies (Iron Condor/Butterfly)")
            # Let the strategy selector choose Iron Condor, don't skip
        else:
            # Low conviction + directional = Skip (don't guess direction)
            logger.info(f"🚫 Skipping trade: Low Conviction ({trade_thesis.conviction_level}) with directional bias ({raw_score:.2f})")
            if return_reports:
                reports = {
                    "volatility_report": volatility_report,
                    "sentiment_report": sentiment_report,
                    "technical_report": technical_report,
                    "fundamental_report": fundamental_report,
                    "trade_thesis": trade_thesis,
                }
                return None, reports
            return None

    # ========================================================================
    # PHASE 3: TRADE CONSTRUCTION (ENHANCED WITH TECHNICAL BIAS)
    # ========================================================================
    logger.info("Generating trade proposal")

    # Extract technical bias from technical report for trend alignment
    # This is the KEY improvement - pass technical signals to strategy selector
    technical_bias = None
    if technical_report:
        # Method 1: Try llm_report dict
        if hasattr(technical_report, 'llm_report') and isinstance(technical_report.llm_report, dict):
            technical_bias = technical_report.llm_report.get('technical_bias')

        # Method 2: If llm_report bias is neutral or None, check the raw LLM output for actual bias
        # The raw output often contains the real technical_bias in JSON format
        if (technical_bias is None or technical_bias == 'neutral') and hasattr(technical_report, 'llm_raw'):
            raw = technical_report.llm_raw or ""
            # Look for explicit "technical_bias" field in JSON output
            import re
            match = re.search(r'"technical_bias"\s*:\s*"(\w+)"', raw, re.IGNORECASE)
            if match:
                extracted = match.group(1).lower()
                if extracted in ['bullish', 'bearish', 'neutral']:
                    technical_bias = extracted
                    logger.info(f"  [Technical bias extracted from llm_raw: {extracted}]")

        # Method 3: Fall back to looking for bias keywords in summary
        if technical_bias is None or technical_bias == 'neutral':
            if hasattr(technical_report, 'llm_raw'):
                raw = (technical_report.llm_raw or "").lower()
                # Count bullish vs bearish mentions to determine bias
                bullish_count = raw.count('bullish')
                bearish_count = raw.count('bearish')
                if bullish_count > bearish_count + 2:  # Clear bullish majority
                    technical_bias = "bullish"
                elif bearish_count > bullish_count + 2:  # Clear bearish majority
                    technical_bias = "bearish"

    # ========================================================================
    # COMPUTE TECHNICAL BIAS FROM PRICE ACTION (Most Reliable Method)
    # ========================================================================
    # Use the SAME logic as the Technical Baseline strategy for consistency!
    # This ensures Quant13 never fights the same trends that Technical follows.
    price_based_bias = None
    try:
        current_price = ohlcv['close'].iloc[-1]

        # SMA 20 (same as Technical baseline)
        sma_20 = ohlcv['close'].rolling(20).mean().iloc[-1] if len(ohlcv) >= 20 else ohlcv['close'].mean()

        # MACD (same as Technical baseline)
        ema_12 = ohlcv['close'].ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = ohlcv['close'].ewm(span=26, adjust=False).mean().iloc[-1]
        macd_line = ema_12 - ema_26

        # Use EXACTLY the same conditions as Technical baseline
        # Bullish: Price > SMA20 AND MACD > 0
        # Bearish: Price < SMA20 AND MACD < 0
        if current_price > sma_20 and macd_line > 0:
            price_based_bias = "bullish"
        elif current_price < sma_20 and macd_line < 0:
            price_based_bias = "bearish"
        else:
            price_based_bias = "neutral"

        logger.info(f"  Price-based technical bias: {price_based_bias} (price vs SMA20: {current_price/sma_20:.2%}, MACD: {macd_line:.2f})")
    except Exception as e:
        logger.warning(f"  Could not compute price-based bias: {e}")
        price_based_bias = technical_bias  # Fallback to extracted bias

    # Use price-based bias if available, otherwise use extracted
    final_technical_bias = price_based_bias if price_based_bias else technical_bias
    logger.info(f"  Final technical bias for strategy selection: {final_technical_bias}")

    # ========================================================================
    # TECHNICAL ALIGNMENT: Override thesis direction to match technical trend
    # ========================================================================
    # This is the CRITICAL fix - if technicals are bullish, don't go bearish!
    if final_technical_bias and final_technical_bias != 'neutral':
        current_direction = trade_thesis.winning_argument.lower()

        if final_technical_bias == 'bullish' and current_direction != 'bullish':
            if raw_score > -0.7:  # Only keep bearish if sentiment is EXTREMELY bearish
                logger.info(f"  [TECHNICAL ALIGNMENT] {trade_thesis.winning_argument} → Bullish (aligning with tech trend)")
                trade_thesis.winning_argument = "Bullish"
                trade_thesis.summary = f"Technical trend bullish. Aligning thesis with price momentum."
                if trade_thesis.conviction_level.lower() == 'low':
                    trade_thesis.conviction_level = 'Medium'
                    logger.info(f"  [CONVICTION BOOST] Low → Medium (trend-aligned)")

        elif final_technical_bias == 'bearish' and current_direction != 'bearish':
            if raw_score < 0.7:  # Only keep bullish if sentiment is EXTREMELY bullish
                logger.info(f"  [TECHNICAL ALIGNMENT] {trade_thesis.winning_argument} → Bearish (aligning with tech trend)")
                trade_thesis.winning_argument = "Bearish"
                trade_thesis.summary = f"Technical trend bearish. Aligning thesis with price momentum."
                if trade_thesis.conviction_level.lower() == 'low':
                    trade_thesis.conviction_level = 'Medium'
                    logger.info(f"  [CONVICTION BOOST] Low → Medium (trend-aligned)")

    # Update technical_bias for strategy selector
    technical_bias = final_technical_bias

    # Use systematic trader for deterministic results
    trader = SystematicTraderAgent(config["agents"]["trader"], backtest_mode=True)
    trade_proposal = trader.propose_trade(
        trade_thesis,
        volatility_report,
        options_chain,
        spot_price,
        technical_bias=technical_bias  # NEW: Pass technical bias for trend alignment
    )

    logger.info(f"  Proposed: {trade_proposal.strategy_name}")

    # Calculate risk metrics
    from src.utils.risk import calculate_risk_metrics
    risk_metrics = calculate_risk_metrics(trade_proposal, options_chain)

    trade_proposal = trade_proposal.model_copy(update={
        "conviction_level": trade_thesis.conviction_level,
        "max_risk": risk_metrics.get("max_risk"),
        "max_reward": risk_metrics.get("max_reward"),
        "net_premium": risk_metrics.get("net_premium"),
    })

    if return_reports:
        # Return all agent reports for debugging/analysis
        reports = {
            "volatility_report": volatility_report,
            "sentiment_report": sentiment_report,
            "technical_report": technical_report,
            "fundamental_report": fundamental_report,
            "trade_thesis": trade_thesis,
        }
        return trade_proposal, reports

    return trade_proposal


def quant13_strategy_wrapper(
    ticker: str,
    ohlcv: pd.DataFrame,
    options_chain: list[Dict[str, Any]],
    spot_price: float,
    **kwargs
) -> TradeProposal | tuple[TradeProposal, Dict[str, Any]]:
    """
    Wrapper function for backtesting framework compatibility

    This function has the signature expected by the backtesting framework
    and calls run_system_backtest internally.

    Args:
        ticker: Stock ticker
        ohlcv: Historical OHLCV data
        options_chain: Options chain
        spot_price: Current spot price
        **kwargs: Additional arguments (risk_free_rate, etc.)

    Returns:
        (TradeProposal, agent_reports) tuple for evaluation runs
    """
    # Suppress noisy logs during backtesting
    logging.getLogger("src.tools.llm").setLevel(logging.WARNING)

    risk_free_rate = kwargs.get("risk_free_rate", 0.05)

    # Always return reports for evaluation analysis
    return run_system_backtest(
        ticker=ticker,
        ohlcv=ohlcv,
        options_chain=options_chain,
        spot_price=spot_price,
        risk_free_rate=risk_free_rate,
        return_reports=True,  # Enable report export
    )
