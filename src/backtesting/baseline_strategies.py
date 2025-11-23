"""
Baseline trading strategies for comparison

Implements simple strategies to benchmark against the main system:
1. Chimpanzee Strategy: Random options trading
2. Technical Baseline: Simple technical indicator-based trading
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.schemas import TradeProposal, TradeLeg


class ChimpanzeeStrategy:
    """
    Random options trading strategy (the chimpanzee)

    Randomly selects:
    - Strategy type (call, put, spread)
    - Strike prices
    - Expiration dates
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_trade(
        self,
        ticker: str,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
    ) -> TradeProposal:
        """
        Generate a random options trade

        Args:
            ticker: Stock ticker
            options_chain: Available options chain
            spot_price: Current stock price

        Returns:
            TradeProposal with random strategy
        """
        # Randomly choose strategy type
        strategy_types = [
            "long_call",
            "long_put",
            "bull_call_spread",
            "bear_put_spread",
        ]
        strategy = random.choice(strategy_types)

        # Randomly choose expiration (prefer front month)
        exp_weights = [0.7] + [0.3 / (len(options_chain) - 1)] * (len(options_chain) - 1)
        exp_idx = np.random.choice(len(options_chain), p=exp_weights)
        expiration_group = options_chain[exp_idx]

        # Generate trade based on strategy
        if strategy == "long_call":
            trade_legs = self._random_long_call(expiration_group, spot_price)
            action = "buy"
        elif strategy == "long_put":
            trade_legs = self._random_long_put(expiration_group, spot_price)
            action = "buy"
        elif strategy == "bull_call_spread":
            trade_legs = self._random_bull_call_spread(expiration_group, spot_price)
            action = "buy"
        else:  # bear_put_spread
            trade_legs = self._random_bear_put_spread(expiration_group, spot_price)
            action = "buy"

        return TradeProposal(
            agent="ChimpanzeeStrategy",
            strategy_name=strategy,
            action=action,
            quantity=1,
            trade_legs=trade_legs,
            notes="Randomly generated trade (chimpanzee algorithm)",
            conviction_level="random",
            generation_status="generated",
        )

    def _random_long_call(
        self, expiration_group: Dict[str, Any], spot_price: float
    ) -> List[TradeLeg]:
        """Buy a random call option (slightly OTM preferred)"""
        calls_df = expiration_group["calls"]

        # Prefer strikes 0-10% OTM
        otm_calls = calls_df[calls_df["strike"] >= spot_price * 1.0]
        if not otm_calls.empty and random.random() < 0.7:
            option = otm_calls.sample(1).iloc[0]
        else:
            option = calls_df.sample(1).iloc[0]

        return [
            TradeLeg(
                contract_symbol=option["contractSymbol"],
                type="call",
                action="buy",
                strike_price=option["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=option.get("computed_greeks", {}),
            )
        ]

    def _random_long_put(
        self, expiration_group: Dict[str, Any], spot_price: float
    ) -> List[TradeLeg]:
        """Buy a random put option (slightly OTM preferred)"""
        puts_df = expiration_group["puts"]

        # Prefer strikes 0-10% OTM
        otm_puts = puts_df[puts_df["strike"] <= spot_price * 1.0]
        if not otm_puts.empty and random.random() < 0.7:
            option = otm_puts.sample(1).iloc[0]
        else:
            option = puts_df.sample(1).iloc[0]

        return [
            TradeLeg(
                contract_symbol=option["contractSymbol"],
                type="put",
                action="buy",
                strike_price=option["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=option.get("computed_greeks", {}),
            )
        ]

    def _random_bull_call_spread(
        self, expiration_group: Dict[str, Any], spot_price: float
    ) -> List[TradeLeg]:
        """Random bull call spread"""
        calls_df = expiration_group["calls"]

        # Select two random calls with different strikes
        selected = calls_df.sample(min(2, len(calls_df)))
        if len(selected) < 2:
            # Fallback to long call
            return self._random_long_call(expiration_group, spot_price)

        selected = selected.sort_values("strike")
        lower_call = selected.iloc[0]
        upper_call = selected.iloc[1]

        return [
            TradeLeg(
                contract_symbol=lower_call["contractSymbol"],
                type="call",
                action="buy",
                strike_price=lower_call["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=lower_call.get("computed_greeks", {}),
            ),
            TradeLeg(
                contract_symbol=upper_call["contractSymbol"],
                type="call",
                action="sell",
                strike_price=upper_call["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=upper_call.get("computed_greeks", {}),
            ),
        ]

    def _random_bear_put_spread(
        self, expiration_group: Dict[str, Any], spot_price: float
    ) -> List[TradeLeg]:
        """Random bear put spread"""
        puts_df = expiration_group["puts"]

        # Select two random puts with different strikes
        selected = puts_df.sample(min(2, len(puts_df)))
        if len(selected) < 2:
            # Fallback to long put
            return self._random_long_put(expiration_group, spot_price)

        selected = selected.sort_values("strike", ascending=False)
        upper_put = selected.iloc[0]
        lower_put = selected.iloc[1]

        return [
            TradeLeg(
                contract_symbol=upper_put["contractSymbol"],
                type="put",
                action="buy",
                strike_price=upper_put["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=upper_put.get("computed_greeks", {}),
            ),
            TradeLeg(
                contract_symbol=lower_put["contractSymbol"],
                type="put",
                action="sell",
                strike_price=lower_put["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=lower_put.get("computed_greeks", {}),
            ),
        ]


class TechnicalBaselineStrategy:
    """
    Simple technical analysis based options strategy

    Uses basic indicators to make directional bets:
    - RSI: Oversold (< 30) -> Buy calls, Overbought (> 70) -> Buy puts
    - MACD: Bullish cross -> Buy calls, Bearish cross -> Buy puts
    - Moving averages: Price > SMA200 -> Bias long, Price < SMA200 -> Bias short
    """

    def generate_trade(
        self,
        ticker: str,
        ohlcv: pd.DataFrame,
        options_chain: List[Dict[str, Any]],
        spot_price: float,
    ) -> TradeProposal:
        """
        Generate trade based on technical indicators

        Args:
            ticker: Stock ticker
            ohlcv: Historical OHLCV data
            options_chain: Available options chain
            spot_price: Current stock price

        Returns:
            TradeProposal based on technical analysis
        """
        # Calculate indicators
        signals = self._calculate_signals(ohlcv)

        # Determine direction
        bullish_score = sum([
            signals["rsi"] < 30,  # Oversold
            signals["macd_signal"] == "bullish",
            signals["price_vs_sma200"] == "above",
        ])

        bearish_score = sum([
            signals["rsi"] > 70,  # Overbought
            signals["macd_signal"] == "bearish",
            signals["price_vs_sma200"] == "below",
        ])

        # Choose strategy based on signals
        if bullish_score > bearish_score:
            strategy = "bull_call_spread"
            action = "buy"
            trade_legs = self._create_bull_call_spread(options_chain, spot_price)
        elif bearish_score > bullish_score:
            strategy = "bear_put_spread"
            action = "buy"
            trade_legs = self._create_bear_put_spread(options_chain, spot_price)
        else:
            # Neutral - sell iron condor or do nothing (long call as fallback)
            strategy = "long_call"
            action = "buy"
            trade_legs = self._create_long_call(options_chain, spot_price)

        return TradeProposal(
            agent="TechnicalBaselineStrategy",
            strategy_name=strategy,
            action=action,
            quantity=1,
            trade_legs=trade_legs,
            notes=f"Technical signals: RSI={signals['rsi']:.1f}, MACD={signals['macd_signal']}, Price vs SMA200={signals['price_vs_sma200']}",
            conviction_level="medium" if abs(bullish_score - bearish_score) >= 2 else "low",
            generation_status="generated",
        )

    def _calculate_signals(self, ohlcv: pd.DataFrame) -> Dict[str, Any]:
        """Calculate technical indicators"""
        df = ohlcv.copy()

        # RSI
        rsi = self._calculate_rsi(df["close"], period=14)

        # MACD
        macd_signal = self._calculate_macd_signal(df["close"])

        # SMA200
        sma_200 = df["close"].rolling(window=200).mean().iloc[-1]
        current_price = df["close"].iloc[-1]

        price_vs_sma200 = "above" if current_price > sma_200 else "below"

        return {
            "rsi": rsi,
            "macd_signal": macd_signal,
            "price_vs_sma200": price_vs_sma200,
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1])

    def _calculate_macd_signal(self, prices: pd.Series) -> str:
        """Calculate MACD signal (bullish, bearish, neutral)"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()

        # Check for crossover
        current_diff = macd.iloc[-1] - signal.iloc[-1]
        prev_diff = macd.iloc[-2] - signal.iloc[-2]

        if current_diff > 0 and prev_diff <= 0:
            return "bullish"
        elif current_diff < 0 and prev_diff >= 0:
            return "bearish"
        elif current_diff > 0:
            return "bullish"
        else:
            return "bearish"

    def _create_bull_call_spread(
        self, options_chain: List[Dict[str, Any]], spot_price: float
    ) -> List[TradeLeg]:
        """Create bull call spread (buy ATM call, sell OTM call)"""
        expiration_group = options_chain[0]  # Front month
        calls_df = expiration_group["calls"]

        # Buy ATM call
        atm_call = calls_df.iloc[(calls_df["strike"] - spot_price).abs().argsort()[0]]

        # Sell 5-10% OTM call
        otm_calls = calls_df[calls_df["strike"] > spot_price * 1.05]
        if not otm_calls.empty:
            short_call = otm_calls.iloc[0]
        else:
            short_call = calls_df.iloc[-1]

        return [
            TradeLeg(
                contract_symbol=atm_call["contractSymbol"],
                type="call",
                action="buy",
                strike_price=atm_call["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=atm_call.get("computed_greeks", {}),
            ),
            TradeLeg(
                contract_symbol=short_call["contractSymbol"],
                type="call",
                action="sell",
                strike_price=short_call["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=short_call.get("computed_greeks", {}),
            ),
        ]

    def _create_bear_put_spread(
        self, options_chain: List[Dict[str, Any]], spot_price: float
    ) -> List[TradeLeg]:
        """Create bear put spread (buy ATM put, sell OTM put)"""
        expiration_group = options_chain[0]  # Front month
        puts_df = expiration_group["puts"]

        # Buy ATM put
        atm_put = puts_df.iloc[(puts_df["strike"] - spot_price).abs().argsort()[0]]

        # Sell 5-10% OTM put
        otm_puts = puts_df[puts_df["strike"] < spot_price * 0.95]
        if not otm_puts.empty:
            short_put = otm_puts.iloc[-1]
        else:
            short_put = puts_df.iloc[0]

        return [
            TradeLeg(
                contract_symbol=atm_put["contractSymbol"],
                type="put",
                action="buy",
                strike_price=atm_put["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=atm_put.get("computed_greeks", {}),
            ),
            TradeLeg(
                contract_symbol=short_put["contractSymbol"],
                type="put",
                action="sell",
                strike_price=short_put["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=short_put.get("computed_greeks", {}),
            ),
        ]

    def _create_long_call(
        self, options_chain: List[Dict[str, Any]], spot_price: float
    ) -> List[TradeLeg]:
        """Create simple long call position"""
        expiration_group = options_chain[0]  # Front month
        calls_df = expiration_group["calls"]

        # Buy ATM call
        atm_call = calls_df.iloc[(calls_df["strike"] - spot_price).abs().argsort()[0]]

        return [
            TradeLeg(
                contract_symbol=atm_call["contractSymbol"],
                type="call",
                action="buy",
                strike_price=atm_call["strike"],
                expiration_date=expiration_group["expiration"],
                quantity=1,
                key_greeks_at_selection=atm_call.get("computed_greeks", {}),
            )
        ]
