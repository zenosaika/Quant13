"""Backtesting framework for options trading strategies"""

from src.backtesting.historical_options import (
    generate_historical_options_chain,
    estimate_historical_volatility,
)
from src.backtesting.framework import run_backtest, BacktestResult
from src.backtesting.baseline_strategies import ChimpanzeeStrategy, TechnicalBaselineStrategy
from src.backtesting.system_wrapper import quant13_strategy_wrapper
from src.backtesting.visualization import create_evaluation_report, plot_comparison

__all__ = [
    "generate_historical_options_chain",
    "estimate_historical_volatility",
    "run_backtest",
    "BacktestResult",
    "ChimpanzeeStrategy",
    "TechnicalBaselineStrategy",
    "quant13_strategy_wrapper",
    "create_evaluation_report",
    "plot_comparison",
]
