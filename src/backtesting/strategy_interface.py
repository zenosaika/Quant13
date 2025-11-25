"""
Strategy Interface for Multi-Ticker Evaluation

Provides a unified interface for all strategies (Quant13, Technical, Chimpanzee)
to ensure consistent logging and evaluation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.schemas import TradeProposal

logger = logging.getLogger(__name__)


class StrategyInterface(ABC):
    """
    Base interface for all trading strategies

    All strategies must implement:
    - generate_signal(): Generate trade signal
    - get_name(): Return strategy name
    - get_agent_reports(): Return agent reports (if applicable)
    """

    def __init__(self, name: str):
        self.name = name
        self.agent_reports: Optional[Dict[str, Any]] = None

    @abstractmethod
    def generate_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
        risk_free_rate: float = 0.05,
        **kwargs
    ) -> Optional[TradeProposal]:
        """
        Generate trade signal

        Args:
            ticker: Stock ticker
            ohlcv: Historical OHLCV data (up to current date)
            options_chain: Available options chain
            spot_price: Current spot price
            risk_free_rate: Risk-free rate
            **kwargs: Additional parameters

        Returns:
            TradeProposal if trade generated, None otherwise
        """
        pass

    def get_name(self) -> str:
        """Get strategy name"""
        return self.name

    def get_agent_reports(self) -> Optional[Dict[str, Any]]:
        """
        Get agent reports (for Quant13 only)

        Returns:
            Dictionary with all agent reports, or None for baseline strategies
        """
        return self.agent_reports

    def attach_agent_reports(self, reports: Dict[str, Any]) -> None:
        """
        Attach agent reports to strategy

        Used by Quant13 wrapper to save full agent outputs for logging
        """
        self.agent_reports = reports


class Quant13Strategy(StrategyInterface):
    """
    Wrapper for Quant13 multi-agent system in BACKTEST MODE

    CRITICAL: This wrapper runs the backtest-safe version of the Quant13 pipeline
    that uses:
    - Synthetic sentiment (from price action, NOT live news)
    - Historical OHLCV data passed in (NO data fetching)
    - Historical options chain passed in (NO live options fetching)

    This prevents:
    - Data leakage from future data
    - Spurious folder creation in results/
    - Live API calls during backtesting
    """

    def __init__(self):
        super().__init__("Quant13")

    def generate_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
        risk_free_rate: float = 0.05,
        **kwargs
    ) -> Optional[TradeProposal]:
        """
        Run Quant13 pipeline in BACKTEST MODE

        CRITICAL FIX: Uses run_system_backtest() instead of run_pipeline_v2()
        - Does NOT fetch live news (uses synthetic sentiment from price action)
        - Does NOT create {TICKER}_{timestamp} folders
        - Uses the historical data passed in (prevents data leakage)

        Returns:
            TradeProposal if trade generated, None otherwise
        """
        try:
            # Import the BACKTEST wrapper (NOT the live pipeline)
            from src.backtesting.system_wrapper import run_system_backtest

            # Convert flat options chain back to grouped format if needed
            # run_system_backtest expects grouped format with DataFrames
            grouped_chain = self._convert_to_grouped_chain(options_chain)

            # Run backtest-safe pipeline with historical data only
            result = run_system_backtest(
                ticker=ticker,
                ohlcv=ohlcv,  # Historical OHLCV (no future leakage)
                options_chain=grouped_chain,  # Historical options (synthetic)
                spot_price=spot_price,
                risk_free_rate=risk_free_rate,
                return_reports=True,  # Get agent reports for logging
            )

            # Handle result (can be None, TradeProposal, or tuple)
            if result is None:
                logger.info(f"Quant13: No trade signal (low conviction or unclear setup)")
                return None

            if isinstance(result, tuple):
                trade_proposal, reports = result
                # Store agent reports for logging
                self.agent_reports = {
                    "volatility": reports.get("volatility_report"),
                    "sentiment": reports.get("sentiment_report"),  # Synthetic!
                    "technical": reports.get("technical_report"),
                    "fundamental": reports.get("fundamental_report"),
                    "trade_thesis": reports.get("trade_thesis"),
                }
            else:
                trade_proposal = result

            if trade_proposal is None:
                logger.info(f"Quant13: No trade proposal generated")
                return None

            return trade_proposal

        except Exception as e:
            logger.error(f"Quant13 strategy failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _convert_to_grouped_chain(self, flat_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert flat options list back to grouped format

        The hybrid framework converts grouped chains to flat format for baselines.
        We need to convert back for run_system_backtest which expects grouped.

        CRITICAL: Must include time_to_expiration_years to prevent re-enrichment
        with datetime.now() which breaks historical backtesting.

        Args:
            flat_chain: List of options with 'type' and 'expiration' fields

        Returns:
            Grouped chain with 'calls' and 'puts' DataFrames per expiration
        """
        if not flat_chain:
            return []

        # Check if already in grouped format (has 'calls' key)
        if flat_chain and isinstance(flat_chain[0], dict) and 'calls' in flat_chain[0]:
            return flat_chain

        # Group by expiration
        from collections import defaultdict
        expiration_groups = defaultdict(lambda: {"calls": [], "puts": [], "time_to_exp": None})

        for opt in flat_chain:
            exp = opt.get("expiration", "unknown")
            opt_type = opt.get("type", "call").lower()

            if opt_type == "call":
                expiration_groups[exp]["calls"].append(opt)
            else:
                expiration_groups[exp]["puts"].append(opt)

            # Capture time_to_expiration_years from any option in this expiration group
            if expiration_groups[exp]["time_to_exp"] is None:
                expiration_groups[exp]["time_to_exp"] = opt.get("time_to_expiration_years")

        # Convert to grouped format
        grouped_chain = []
        for exp, options in sorted(expiration_groups.items()):
            calls_df = pd.DataFrame(options["calls"]) if options["calls"] else pd.DataFrame()
            puts_df = pd.DataFrame(options["puts"]) if options["puts"] else pd.DataFrame()

            group = {
                "expiration": exp,
                "calls": calls_df,
                "puts": puts_df,
            }

            # CRITICAL: Include time_to_expiration_years to signal already enriched
            # This prevents system_wrapper from calling enrich_options_chain_with_greeks
            # which would use datetime.now() and break historical backtesting
            if options["time_to_exp"] is not None:
                group["time_to_expiration_years"] = options["time_to_exp"]
            else:
                # Estimate from expiration date if not available
                # Use a reasonable default (30 days = ~0.082 years)
                group["time_to_expiration_years"] = 0.082

            grouped_chain.append(group)

        return grouped_chain


class TechnicalBaselineStrategy(StrategyInterface):
    """
    Technical baseline strategy using systematic rules

    FIXED: More reasonable conditions that actually generate trades:
    - Bullish: RSI < 50 AND price > SMA20 (trend following with momentum confirmation)
    - Bearish: RSI > 50 AND price < SMA20 (trend following with momentum confirmation)

    Previous bug: RSI < 40 AND price > SMA50 was too restrictive
    (RSI rarely goes below 40 when price is above SMA50 - contradictory conditions)
    """

    def __init__(self):
        super().__init__("Technical")

    def generate_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
        risk_free_rate: float = 0.05,
        **kwargs
    ) -> Optional[TradeProposal]:
        """
        Generate trade based on simple technical rules

        FIXED CONDITIONS:
        - Bullish: Price > SMA20 AND RSI between 30-60 (uptrend with room to run)
        - Bearish: Price < SMA20 AND RSI between 40-70 (downtrend with room to fall)
        - Skip: RSI extremes (>70 or <30) - likely mean reversion zone

        Returns:
            TradeProposal with debit spread, or None if no signal
        """
        try:
            # Calculate technical indicators
            from src.utils.indicators import _rsi

            # RSI (14-period)
            current_rsi = _rsi(ohlcv['close'], window=14)
            if current_rsi is None:
                current_rsi = 50

            # SMA 20 (shorter-term trend)
            sma_20 = ohlcv['close'].rolling(20).mean().iloc[-1] if len(ohlcv) >= 20 else ohlcv['close'].mean()

            # SMA 50 (medium-term trend)
            sma_50 = ohlcv['close'].rolling(50).mean().iloc[-1] if len(ohlcv) >= 50 else ohlcv['close'].mean()

            # MACD for momentum confirmation
            ema_12 = ohlcv['close'].ewm(span=12, adjust=False).mean().iloc[-1]
            ema_26 = ohlcv['close'].ewm(span=26, adjust=False).mean().iloc[-1]
            macd_line = ema_12 - ema_26

            # Determine signal with FIXED conditions
            signal = None
            reason = ""

            # Skip RSI extremes (likely reversal zones)
            if current_rsi > 70:
                logger.info(f"Technical: Overbought RSI={current_rsi:.1f}, skipping")
                return None
            elif current_rsi < 30:
                logger.info(f"Technical: Oversold RSI={current_rsi:.1f}, skipping")
                return None

            # Bullish: Uptrend (price > SMA20) with positive momentum
            if spot_price > sma_20 and macd_line > 0:
                signal = "Bullish"
                reason = f"Uptrend (Price > SMA20) with positive MACD, RSI={current_rsi:.1f}"
            # Bearish: Downtrend (price < SMA20) with negative momentum
            elif spot_price < sma_20 and macd_line < 0:
                signal = "Bearish"
                reason = f"Downtrend (Price < SMA20) with negative MACD, RSI={current_rsi:.1f}"

            if signal is None:
                logger.info(f"Technical: No clear signal (RSI={current_rsi:.1f}, MACD={macd_line:.2f})")
                return None

            logger.info(f"Technical: {signal} signal - {reason}")

            # Select strikes for debit spread using simple delta-based selection
            if signal == "Bullish":
                # Bull Call Spread: Buy ATM call, Sell OTM call
                calls = [
                    opt for opt in options_chain
                    if opt.get('type') == 'call' and opt.get('openInterest', 0) >= 100
                ]

                if len(calls) < 2:
                    logger.info("Technical: Not enough liquid call options")
                    return None

                # Sort by delta (descending - higher delta = more ITM)
                calls_sorted = sorted(calls, key=lambda x: x.get('delta', 0), reverse=True)

                # Buy call near 0.50 delta (ATM)
                long_call = self._select_by_target_delta(calls_sorted, target_delta=0.50)
                # Sell call near 0.30 delta (OTM)
                short_call = self._select_by_target_delta(calls_sorted, target_delta=0.30)

                if long_call is None or short_call is None:
                    logger.info("Technical: Could not find suitable call strikes")
                    return None

                # Ensure long call has higher delta (more ITM) than short call
                if long_call.get('strike', 0) >= short_call.get('strike', 0):
                    logger.info("Technical: Invalid spread structure, swapping")
                    long_call, short_call = short_call, long_call

                strategy_name = "Bull Call Spread"
                legs = [
                    self._create_leg(long_call, "BUY", "CALL"),
                    self._create_leg(short_call, "SELL", "CALL"),
                ]

            else:  # Bearish
                # Bear Put Spread: Buy ATM put, Sell OTM put
                puts = [
                    opt for opt in options_chain
                    if opt.get('type') == 'put' and opt.get('openInterest', 0) >= 100
                ]

                if len(puts) < 2:
                    logger.info("Technical: Not enough liquid put options")
                    return None

                # Sort by delta (ascending - more negative delta = more ITM for puts)
                puts_sorted = sorted(puts, key=lambda x: x.get('delta', 0))

                # Buy put near -0.50 delta (ATM)
                long_put = self._select_by_target_delta(puts_sorted, target_delta=-0.50)
                # Sell put near -0.30 delta (OTM)
                short_put = self._select_by_target_delta(puts_sorted, target_delta=-0.30)

                if long_put is None or short_put is None:
                    logger.info("Technical: Could not find suitable put strikes")
                    return None

                # Ensure long put has higher strike than short put (proper bear put spread)
                if long_put.get('strike', 0) <= short_put.get('strike', 0):
                    logger.info("Technical: Invalid spread structure, swapping")
                    long_put, short_put = short_put, long_put

                strategy_name = "Bear Put Spread"
                legs = [
                    self._create_leg(long_put, "BUY", "PUT"),
                    self._create_leg(short_put, "SELL", "PUT"),
                ]

            # Create trade proposal
            from src.models.schemas import TradeLeg

            trade_legs = [TradeLeg(**leg) for leg in legs]

            return TradeProposal(
                agent="TechnicalBaseline",
                strategy_name=strategy_name,
                action="OPEN",
                quantity=1,
                trade_legs=trade_legs,
                notes=f"{signal} signal based on RSI={current_rsi:.1f}",
                conviction_level="Medium",
                generation_status="generated",
            )

        except Exception as e:
            logger.error(f"Technical baseline failed: {e}")
            return None

    def _select_by_target_delta(
        self,
        options: List[Dict[str, Any]],
        target_delta: float
    ) -> Optional[Dict[str, Any]]:
        """
        Select option closest to target delta

        Args:
            options: List of option dicts with 'delta' field
            target_delta: Target delta value (e.g., 0.50 for calls, -0.50 for puts)

        Returns:
            Option dict closest to target delta, or None if no options
        """
        if not options:
            return None

        # Find option with delta closest to target
        best_option = None
        best_diff = float('inf')

        for opt in options:
            delta = opt.get('delta', 0)
            if delta is None:
                continue

            diff = abs(delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best_option = opt

        return best_option

    def _create_leg(self, option: Dict[str, Any], action: str, option_type: str) -> Dict[str, Any]:
        """Create trade leg from option data"""
        return {
            "contract_symbol": option.get('contractSymbol', f"{option['strike']}{option_type[0]}"),
            "type": option_type,
            "action": action,
            "strike_price": option['strike'],
            "expiration_date": option['expiration'],
            "quantity": 1,
            "key_greeks_at_selection": {
                "delta": option.get('delta', 0),
                "gamma": option.get('gamma', 0),
                "theta": option.get('theta', 0),
                "vega": option.get('vega', 0),
            },
        }


class RetailTraderBaseline(StrategyInterface):
    """
    Retail Trader Baseline - Mimics typical retail trader behavior

    This represents how most retail traders actually trade options (poorly):

    1. MOMENTUM CHASING: Buy calls after big up days, puts after big down days
       (Classic "buy high, sell low" behavior - chasing moves that already happened)

    2. NO SPREAD KNOWLEDGE: Buys naked long options instead of spreads
       (Doesn't understand theta decay, pays full premium)

    3. WEEKLY OPTIONS: Prefers short-dated options for "leverage"
       (Theta decay destroys these positions rapidly)

    4. STRIKE SELECTION: Buys cheap OTM options
       (Low probability of profit, but "cheap" appeals to retail)

    5. NO IV CONSIDERATION: Ignores implied volatility
       (Buys expensive options after big moves when IV is crushed or elevated)

    6. RECENCY BIAS: Last 3 days determine direction
       (Ignores larger trends, gets whipsawed constantly)

    This baseline should LOSE money consistently, representing the average
    retail trader who subsidizes market makers and professional traders.

    Expected win rate: ~25-35% (typical retail trader performance)
    """

    def __init__(self):
        super().__init__("RetailTrader")

    def generate_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
        risk_free_rate: float = 0.05,
        **kwargs
    ) -> Optional[TradeProposal]:
        """
        Generate trade based on typical retail trader behavior

        RETAIL TRADER LOGIC:
        - If stock went up last 3 days → BUY CALLS (momentum chasing)
        - If stock went down last 3 days → BUY PUTS (panic/FOMO)
        - Pick cheap OTM options (lottery ticket mentality)
        - Prefer weekly options (maximum theta decay exposure)
        """
        try:
            # Retail traders look at recent price action (last 3 days)
            if len(ohlcv) < 5:
                return None

            # Calculate 3-day return (recency bias)
            three_day_return = (ohlcv['close'].iloc[-1] / ohlcv['close'].iloc[-4] - 1) * 100

            # Retail trader threshold: "If it moved 1%, I need to chase!"
            if three_day_return > 1.0:
                signal = "Bullish"
                reason = f"Stock up {three_day_return:.1f}% last 3 days - BUYING CALLS!"
            elif three_day_return < -1.0:
                signal = "Bearish"
                reason = f"Stock down {three_day_return:.1f}% last 3 days - BUYING PUTS!"
            else:
                # Retail trader sits out during "boring" periods
                logger.info(f"RetailTrader: No action - stock only moved {three_day_return:.1f}%")
                return None

            logger.info(f"RetailTrader: {signal} signal - {reason}")

            # Find options - prefer SHORT dated (retail loves weeklies)
            # and CHEAP OTM options (lottery tickets)
            if signal == "Bullish":
                calls = [
                    opt for opt in options_chain
                    if opt.get('type') == 'call'
                    and opt.get('openInterest', 0) >= 50  # Retail doesn't check liquidity carefully
                ]

                if not calls:
                    return None

                # Retail prefers CHEAP OTM calls (lottery ticket mentality)
                # Sort by price (ascending) and pick something cheap and OTM
                otm_calls = [c for c in calls if c.get('strike', 0) > spot_price * 1.02]  # >2% OTM
                if not otm_calls:
                    otm_calls = calls

                # Pick a cheap one (retail loves "cheap" options)
                otm_calls_sorted = sorted(otm_calls, key=lambda x: x.get('lastPrice', x.get('ask', 100)))
                selected_option = otm_calls_sorted[min(2, len(otm_calls_sorted)-1)]  # Not the cheapest (that's too OTM), but cheap

                strategy_name = "Long Call"
                legs = [self._create_leg(selected_option, "BUY", "CALL")]

            else:  # Bearish
                puts = [
                    opt for opt in options_chain
                    if opt.get('type') == 'put'
                    and opt.get('openInterest', 0) >= 50
                ]

                if not puts:
                    return None

                # Retail prefers CHEAP OTM puts
                otm_puts = [p for p in puts if p.get('strike', 0) < spot_price * 0.98]  # >2% OTM
                if not otm_puts:
                    otm_puts = puts

                otm_puts_sorted = sorted(otm_puts, key=lambda x: x.get('lastPrice', x.get('ask', 100)))
                selected_option = otm_puts_sorted[min(2, len(otm_puts_sorted)-1)]

                strategy_name = "Long Put"
                legs = [self._create_leg(selected_option, "BUY", "PUT")]

            # Create trade proposal
            from src.models.schemas import TradeLeg

            trade_legs = [TradeLeg(**leg) for leg in legs]

            return TradeProposal(
                agent="RetailTraderBaseline",
                strategy_name=strategy_name,
                action="OPEN",
                quantity=1,
                trade_legs=trade_legs,
                notes=f"Retail trader: {reason}",
                conviction_level="Medium",  # Retail always thinks they're right
                generation_status="generated",
            )

        except Exception as e:
            logger.error(f"RetailTrader baseline failed: {e}")
            return None

    def _create_leg(self, option: Dict[str, Any], action: str, option_type: str) -> Dict[str, Any]:
        """Create trade leg from option data"""
        return {
            "contract_symbol": option.get('contractSymbol', f"{option['strike']}{option_type[0]}"),
            "type": option_type,
            "action": action,
            "strike_price": option['strike'],
            "expiration_date": option['expiration'],
            "quantity": 1,
            "key_greeks_at_selection": {
                "delta": option.get('delta', 0),
                "gamma": option.get('gamma', 0),
                "theta": option.get('theta', 0),
                "vega": option.get('vega', 0),
            },
        }


class ChimpanzeeStrategy(StrategyInterface):
    """
    Chimpanzee (random) baseline strategy

    Randomly selects:
    - Direction (bullish/bearish)
    - Strategy type (debit spread)
    - Strikes (random liquid options)

    Used as baseline to ensure Quant13 beats random selection
    """

    def __init__(self):
        super().__init__("Chimpanzee")
        self.rng = None

    def generate_signal(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
        risk_free_rate: float = 0.05,
        **kwargs
    ) -> Optional[TradeProposal]:
        """
        Generate random trade

        Returns:
            Random TradeProposal
        """
        try:
            import random
            import hashlib

            if self.rng is None:
                # Use ticker-specific seed for reproducibility per ticker
                # This ensures different tickers get different random sequences
                ticker_hash = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
                self.rng = random.Random(ticker_hash)

            # Random direction
            signal = self.rng.choice(["Bullish", "Bearish"])

            # Filter liquid options (relaxed for synthetic options)
            # Synthetic options have OI 500-5000, volume 100-1000
            liquid_options = [
                opt for opt in options_chain
                if opt.get('openInterest', 0) >= 100  # Relaxed: only check OI, not volume
            ]

            if len(liquid_options) < 4:
                logger.info(f"Chimpanzee: Not enough liquid options ({len(liquid_options)} found, need 4)")
                return None

            if signal == "Bullish":
                # Random bull call spread
                calls = [opt for opt in liquid_options if opt['type'] == 'call']
                if len(calls) < 2:
                    return None

                calls = sorted(calls, key=lambda x: x['strike'])
                long_call = self.rng.choice(calls[:len(calls)//2])  # Lower half
                short_call = self.rng.choice([c for c in calls if c['strike'] > long_call['strike']])

                strategy_name = "Bull Call Spread"
                legs = [
                    self._create_leg(long_call, "BUY", "CALL"),
                    self._create_leg(short_call, "SELL", "CALL"),
                ]

            else:  # Bearish
                # Random bear put spread
                puts = [opt for opt in liquid_options if opt['type'] == 'put']
                if len(puts) < 2:
                    return None

                puts = sorted(puts, key=lambda x: x['strike'], reverse=True)
                long_put = self.rng.choice(puts[:len(puts)//2])  # Upper half
                short_put = self.rng.choice([p for p in puts if p['strike'] < long_put['strike']])

                strategy_name = "Bear Put Spread"
                legs = [
                    self._create_leg(long_put, "BUY", "PUT"),
                    self._create_leg(short_put, "SELL", "PUT"),
                ]

            # Create trade proposal
            from src.models.schemas import TradeLeg

            trade_legs = [TradeLeg(**leg) for leg in legs]

            return TradeProposal(
                agent="ChimpanzeeBaseline",
                strategy_name=strategy_name,
                action="OPEN",
                quantity=1,
                trade_legs=trade_legs,
                notes=f"Random {signal} selection",
                conviction_level="Low",
                generation_status="generated",
            )

        except Exception as e:
            logger.error(f"Chimpanzee baseline failed: {e}")
            return None

    def _create_leg(self, option: Dict[str, Any], action: str, option_type: str) -> Dict[str, Any]:
        """Create trade leg from option data"""
        return {
            "contract_symbol": option.get('contractSymbol', f"{option['strike']}{option_type[0]}"),
            "type": option_type,
            "action": action,
            "strike_price": option['strike'],
            "expiration_date": option['expiration'],
            "quantity": 1,
            "key_greeks_at_selection": {
                "delta": option.get('delta', 0),
                "gamma": option.get('gamma', 0),
                "theta": option.get('theta', 0),
                "vega": option.get('vega', 0),
            },
        }
