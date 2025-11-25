"""
Multi-Ticker Evaluator

Orchestrates parallel evaluation across multiple tickers and strategies.
Combines hybrid backtesting framework with comprehensive logging.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.backtesting.hybrid_framework import run_hybrid_backtest
from src.backtesting.strategy_interface import (
    Quant13Strategy,
    TechnicalBaselineStrategy,
    RetailTraderBaseline,
    ChimpanzeeStrategy,
)
from src.evaluation.trade_logger import TradeLogger
from src.evaluation.result_aggregator import ResultAggregator
from src.data.fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for multi-ticker evaluation"""

    # Tickers to evaluate
    tickers: List[str] = field(default_factory=list)

    # Strategies to evaluate
    strategies: List[str] = field(default_factory=lambda: ["quant13", "technical", "chimpanzee"])

    # Date range
    start_date: datetime = field(default_factory=lambda: datetime(2024, 1, 1))
    end_date: datetime = field(default_factory=lambda: datetime(2024, 12, 31))

    # Backtest parameters
    initial_capital: float = 10000.0
    position_size_pct: float = 0.90  # AGGRESSIVE: Use 90% of capital per trade for maximum P&L
    risk_free_rate: float = 0.05

    # Position management - TIGHT STOPS FOR CAPITAL PRESERVATION
    # Exit losers quickly (25% stop) while letting winners hit 50% target
    profit_target_pct: float = 0.50  # 50% profit target
    stop_loss_pct: float = 0.25  # 25% stop loss - EXIT FAST on losers!
    min_dte_close: int = 7  # Close at 7 DTE

    # Hybrid rebalancing
    signal_frequency: str = "weekly"  # How often to generate new signals
    monitor_frequency: str = "daily"  # How often to check positions

    # Execution
    parallel_workers: int = 4  # Number of parallel processes
    output_dir: Optional[Path] = None

    # Predefined ticker lists
    @classmethod
    def mixed_portfolio(cls) -> EvaluationConfig:
        """Mixed portfolio: indices, tech, defensive"""
        return cls(
            tickers=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "JNJ", "PG", "XLE", "GLD"],
        )

    @classmethod
    def tech_focused(cls) -> EvaluationConfig:
        """Tech-focused portfolio"""
        return cls(
            tickers=["QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "NFLX"],
        )

    @classmethod
    def diverse_sectors(cls) -> EvaluationConfig:
        """Diverse sectors portfolio"""
        return cls(
            tickers=["SPY", "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"],
        )

    @classmethod
    def quick_test(cls) -> EvaluationConfig:
        """Quick test with 3 tickers, 6 months"""
        return cls(
            tickers=["SPY", "AAPL", "QQQ"],
            start_date=datetime(2024, 6, 1),
            end_date=datetime(2024, 12, 31),
        )

    @classmethod
    def quick_test_1month(cls) -> EvaluationConfig:
        """Quick test with 3 tickers, 1 month (cost-optimized)"""
        return cls(
            tickers=["SPY", "AAPL", "QQQ"],
            start_date=datetime(2024, 11, 1),
            end_date=datetime(2024, 12, 1),
        )

    @classmethod
    def quick_test_3month(cls) -> EvaluationConfig:
        """Quick test with 3 tickers, 3 months (more statistical significance)"""
        return cls(
            tickers=["SPY", "AAPL", "QQQ"],
            start_date=datetime(2024, 9, 1),
            end_date=datetime(2024, 12, 1),
        )

    @classmethod
    def baseline_only(cls) -> EvaluationConfig:
        """Baseline strategies only (no LLM calls) - 3 tickers, 1 month"""
        return cls(
            tickers=["SPY", "AAPL", "QQQ"],
            strategies=["technical", "chimpanzee"],  # Skip Quant13
            start_date=datetime(2024, 11, 1),
            end_date=datetime(2024, 12, 1),
        )


class MultiTickerEvaluator:
    """
    Orchestrates multi-ticker evaluation with parallel execution

    Workflow:
    1. For each ticker:
       - Run hybrid backtest for each strategy (quant13, technical, chimpanzee)
       - Log individual trades to JSON
       - Save backtest results
    2. Aggregate results across all tickers and strategies
    3. Generate summary tables and visualizations
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize evaluator

        Args:
            config: Evaluation configuration
        """
        self.config = config

        # Create output directory
        if config.output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"results/evaluation_{timestamp}")
        else:
            self.output_dir = Path(config.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize utilities
        self.trade_logger = TradeLogger(self.output_dir)
        self.result_aggregator = ResultAggregator(self.output_dir)

        # Log configuration
        self._save_config()

    def _save_config(self) -> None:
        """Save configuration to file"""
        import json

        config_path = self.output_dir / "config.json"
        config_dict = {
            "tickers": self.config.tickers,
            "strategies": self.config.strategies,
            "start_date": self.config.start_date.isoformat(),
            "end_date": self.config.end_date.isoformat(),
            "initial_capital": self.config.initial_capital,
            "position_size_pct": self.config.position_size_pct,
            "signal_frequency": self.config.signal_frequency,
            "monitor_frequency": self.config.monitor_frequency,
            "parallel_workers": self.config.parallel_workers,
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Saved config: {config_path}")

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run full multi-ticker evaluation

        Returns:
            Summary dictionary with results
        """
        logger.info("=" * 80)
        logger.info("MULTI-TICKER EVALUATION STARTED")
        logger.info("=" * 80)
        logger.info(f"Tickers: {len(self.config.tickers)} ({', '.join(self.config.tickers)})")
        logger.info(f"Strategies: {len(self.config.strategies)} ({', '.join(self.config.strategies)})")
        logger.info(f"Date range: {self.config.start_date.date()} to {self.config.end_date.date()}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 80)

        # Generate tasks
        tasks = []
        for ticker in self.config.tickers:
            for strategy_name in self.config.strategies:
                tasks.append((ticker, strategy_name))

        logger.info(f"Total tasks: {len(tasks)}")

        # Run tasks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self._run_single_backtest,
                    ticker,
                    strategy_name,
                ): (ticker, strategy_name)
                for ticker, strategy_name in tasks
            }

            # Collect results as they complete
            for future in as_completed(futures):
                ticker, strategy_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✓ Completed: {ticker} {strategy_name}")
                except Exception as e:
                    logger.error(f"✗ Failed: {ticker} {strategy_name} - {e}")
                    continue

        logger.info(f"Completed {len(results)}/{len(tasks)} backtests")

        # Generate summaries
        logger.info("Generating aggregate summaries...")
        summaries = self.result_aggregator.generate_all_summaries()

        # Export summary report
        report_path = self.result_aggregator.export_summary_report()

        logger.info("=" * 80)
        logger.info("MULTI-TICKER EVALUATION COMPLETE")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Summary report: {report_path}")
        logger.info("=" * 80)

        return {
            "output_dir": str(self.output_dir),
            "num_results": len(results),
            "summaries": summaries,
            "report_path": str(report_path),
        }

    def _run_single_backtest(
        self,
        ticker: str,
        strategy_name: str,
    ) -> Dict[str, Any]:
        """
        Run single backtest for ticker × strategy

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name

        Returns:
            Result dictionary
        """
        logger.info(f"Running backtest: {ticker} {strategy_name}")

        try:
            # Fetch OHLCV data
            lookback_days = (self.config.end_date - self.config.start_date).days + 120
            ohlcv = fetch_ohlcv(ticker, lookback_days=lookback_days)

            if ohlcv.empty or len(ohlcv) < 120:
                logger.warning(f"Insufficient data for {ticker}, skipping")
                return {"ticker": ticker, "strategy": strategy_name, "status": "insufficient_data"}

            # Create strategy instance
            strategy = self._create_strategy(strategy_name)

            # Run hybrid backtest
            result = run_hybrid_backtest(
                ticker=ticker,
                strategy_func=strategy.generate_signal,
                ohlcv=ohlcv,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_capital=self.config.initial_capital,
                position_size_pct=self.config.position_size_pct,
                risk_free_rate=self.config.risk_free_rate,
                output_dir=self.output_dir,
                profit_target_pct=self.config.profit_target_pct,
                stop_loss_pct=self.config.stop_loss_pct,
                min_dte_close=self.config.min_dte_close,
                signal_frequency=self.config.signal_frequency,
                monitor_frequency=self.config.monitor_frequency,
                strategy_name=strategy_name,  # Pass explicit strategy name
            )

            # Attach agent reports to trades (for Quant13)
            if strategy_name.lower() == "quant13":
                for trade in result.trades:
                    trade.agent_reports = strategy.get_agent_reports()

            # Log trades
            self.trade_logger.log_trades_batch(
                ticker=ticker,
                strategy_name=strategy_name,
                trades=result.trades,
            )

            # Save backtest result
            self.result_aggregator.save_backtest_result(
                ticker=ticker,
                strategy_name=strategy_name,
                result=result,
            )

            return {
                "ticker": ticker,
                "strategy": strategy_name,
                "status": "success",
                "total_return_pct": result.total_return_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "num_trades": result.num_trades,
            }

        except Exception as e:
            logger.error(f"Backtest failed for {ticker} {strategy_name}: {e}")
            return {
                "ticker": ticker,
                "strategy": strategy_name,
                "status": "error",
                "error": str(e),
            }

    def _create_strategy(self, strategy_name: str):
        """
        Create strategy instance

        Args:
            strategy_name: Strategy name (quant13, technical, chimpanzee)

        Returns:
            Strategy instance
        """
        strategy_map = {
            "quant13": Quant13Strategy,
            "technical": TechnicalBaselineStrategy,
            "retailtrader": RetailTraderBaseline,
            "chimpanzee": ChimpanzeeStrategy,
        }

        strategy_class = strategy_map.get(strategy_name.lower())
        if strategy_class is None:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategy_class()


def run_quick_test() -> Dict[str, Any]:
    """Run quick test with 3 tickers"""
    config = EvaluationConfig.quick_test()
    evaluator = MultiTickerEvaluator(config)
    return evaluator.run_evaluation()


def run_full_evaluation(ticker_set: str = "mixed") -> Dict[str, Any]:
    """
    Run full evaluation with predefined ticker set

    Args:
        ticker_set: One of "mixed", "tech", "diverse"

    Returns:
        Evaluation results
    """
    if ticker_set == "mixed":
        config = EvaluationConfig.mixed_portfolio()
    elif ticker_set == "tech":
        config = EvaluationConfig.tech_focused()
    elif ticker_set == "diverse":
        config = EvaluationConfig.diverse_sectors()
    else:
        raise ValueError(f"Unknown ticker set: {ticker_set}")

    evaluator = MultiTickerEvaluator(config)
    return evaluator.run_evaluation()
