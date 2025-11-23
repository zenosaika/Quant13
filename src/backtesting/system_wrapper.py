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
    # LOGIC: Default to Neutral (Iron Condor) unless proven otherwise
    # ========================================================================
    # VARIANCE HARVESTING LOGIC: Mean Reversion FIRST, Trend Following LAST
    # Priority: RSI extremes > MACD Filter > Default Neutral > Trend (only if clean)
    # ========================================================================
    score = 0.0
    regime = "Choppy/Neutral"
    summary = ""
    articles = []

    # FIRST: Mean Reversion (HIGHEST PRIORITY - overrides trend)
    # If RSI is extreme, default to NEUTRAL (Iron Condor) to avoid chasing parabolic moves
    if curr_rsi > 70:  # Lowered threshold (was 75) - more conservative
        score = 0.0  # NEUTRAL → Iron Condor (don't chase overbought)
        regime = "Overbought (Iron Condor)"
        summary = f"Overbought (RSI {curr_rsi:.1f}). Default to Iron Condor. Trend: {trend_strength:.1%}."
        articles.append(ArticleSentiment(
            title=f"{ticker} overbought - avoid directional",
            publisher="Mean Reversion Logic",
            sentiment_score=0.0,
            rationale=f"RSI {curr_rsi:.1f} overbought. Don't chase. Sell premium (Iron Condor)."
        ))

    elif curr_rsi < 30:  # Lowered threshold (was 25) - more conservative
        score = 0.0  # NEUTRAL → Iron Condor (don't chase oversold)
        regime = "Oversold (Iron Condor)"
        summary = f"Oversold (RSI {curr_rsi:.1f}). Default to Iron Condor. Trend: {trend_strength:.1%}."
        articles.append(ArticleSentiment(
            title=f"{ticker} oversold - avoid directional",
            publisher="Mean Reversion Logic",
            sentiment_score=0.0,
            rationale=f"RSI {curr_rsi:.1f} oversold. Don't chase. Sell premium (Iron Condor)."
        ))

    # SECOND: Strong Trend (Only if RSI is NOT extreme AND separation > 5%)
    # CRITICAL FIX #3: Increased directional strength for trending markets
    elif trend_strength > 0.07 and 30 <= curr_rsi <= 70:
        # Very strong trend (7%+) - increase score to 0.6
        if sma20 > sma50:
            score = 0.6  # Strong Bullish
            regime = "Strong Uptrend"
            summary = f"Strong uptrend. SMA separation: {trend_strength:.1%}. RSI: {curr_rsi:.1f}. MACD: {macd_crossover}."
            articles.append(ArticleSentiment(
                title=f"{ticker} strong uptrend",
                publisher="Mean Reversion Logic",
                sentiment_score=0.6,
                rationale=f"Very strong trend ({trend_strength:.1%}) with neutral RSI ({curr_rsi:.1f})."
            ))
        else:
            score = -0.6  # Strong Bearish
            regime = "Strong Downtrend"
            summary = f"Strong downtrend. SMA separation: {trend_strength:.1%}. RSI: {curr_rsi:.1f}. MACD: {macd_crossover}."
            articles.append(ArticleSentiment(
                title=f"{ticker} strong downtrend",
                publisher="Mean Reversion Logic",
                sentiment_score=-0.6,
                rationale=f"Very strong downtrend ({trend_strength:.1%}) with neutral RSI ({curr_rsi:.1f})."
            ))

    elif trend_strength > 0.05 and 30 <= curr_rsi <= 70:
        # Moderate trend (5-7%)
        if sma20 > sma50:
            score = 0.4  # Moderate Bullish
            regime = "Confirmed Uptrend"
            summary = f"Uptrend confirmed. SMA separation: {trend_strength:.1%}. RSI: {curr_rsi:.1f}. MACD: {macd_crossover}."
            articles.append(ArticleSentiment(
                title=f"{ticker} uptrend",
                publisher="Mean Reversion Logic",
                sentiment_score=0.4,
                rationale=f"Moderate trend ({trend_strength:.1%}) with neutral RSI ({curr_rsi:.1f})."
            ))
        else:
            score = -0.4  # Moderate Bearish
            regime = "Confirmed Downtrend"
            summary = f"Downtrend confirmed. SMA separation: {trend_strength:.1%}. RSI: {curr_rsi:.1f}. MACD: {macd_crossover}."
            articles.append(ArticleSentiment(
                title=f"{ticker} downtrend",
                publisher="Mean Reversion Logic",
                sentiment_score=-0.4,
                rationale=f"Moderate downtrend ({trend_strength:.1%}) with neutral RSI ({curr_rsi:.1f})."
            ))

    # DEFAULT: Choppy/Neutral (This should be the MOST COMMON case)
    else:
        score = 0.0  # NEUTRAL → Iron Condor
        regime = "Range-Bound/Choppy"
        summary = f"Market choppy/neutral. RSI: {curr_rsi:.1f}, Trend: {trend_strength:.1%}, BB: {bb_width:.2f}."
        articles.append(ArticleSentiment(
            title=f"{ticker} range-bound",
            publisher="Mean Reversion Logic",
            sentiment_score=0.0,
            rationale=f"No strong trend ({trend_strength:.1%}). Favor Iron Condor (sell premium)."
        ))

    # ========================================================================
    # CRITICAL FIX #2: MACD Momentum Enhancement (REVISED)
    # ========================================================================
    # Instead of BLOCKING trades when MACD disagrees, use MACD to CONFIRM
    # strong trends and BOOST the signal.
    #
    # If MACD agrees with trend: BOOST signal (helps catch trends earlier)
    # If MACD disagrees: Keep original signal (don't force neutral)
    #
    if macd_crossover == "bullish" and score > 0.3:
        # MACD confirms uptrend - boost signal
        logger.info(f"MACD bullish confirming uptrend (score {score:.2f}) - boosting to 0.7")
        score = 0.7  # Strong bullish
        regime = "MACD Confirmed Uptrend"
        summary = f"MACD bullish ({macd_line.iloc[-1]:.2f} > {signal_line.iloc[-1]:.2f}) confirms uptrend. Strong buy signal."

    elif macd_crossover == "bearish" and score < -0.3:
        # MACD confirms downtrend - boost signal
        logger.info(f"MACD bearish confirming downtrend (score {score:.2f}) - boosting to -0.7")
        score = -0.7  # Strong bearish
        regime = "MACD Confirmed Downtrend"
        summary = f"MACD bearish ({macd_line.iloc[-1]:.2f} < {signal_line.iloc[-1]:.2f}) confirms downtrend. Strong sell signal."

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
            # Estimate DTE from expiration_date if available
            exp_date_str = expiry.get('expiration_date', '')
            if exp_date_str:
                try:
                    exp_date = datetime.fromisoformat(exp_date_str.replace('Z', '+00:00'))
                    current_date = ohlcv.index[-1]
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
    # PHASE 3: TRADE CONSTRUCTION
    # ========================================================================
    logger.info("Generating trade proposal")

    # Use systematic trader for deterministic results
    trader = SystematicTraderAgent(config["agents"]["trader"])
    trade_proposal = trader.propose_trade(
        trade_thesis,
        volatility_report,
        options_chain,
        spot_price
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
