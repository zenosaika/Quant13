"""
Result Aggregator for Multi-Ticker Evaluation

Aggregates backtest results across multiple tickers and strategies:
- Combines individual backtest results
- Generates performance matrix (tickers × strategies)
- Creates aggregate metrics CSV
- Computes portfolio-level statistics
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.backtesting.framework import BacktestResult

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates backtest results across tickers and strategies

    Directory structure:
        output_dir/
        ├── summary/
        │   ├── aggregate_metrics.csv
        │   ├── performance_matrix.csv
        │   ├── strategy_comparison.csv
        │   └── figures/
        ├── tickers/
        │   ├── SPY/
        │   │   ├── quant13/
        │   │   │   ├── backtest_result.json
        │   │   │   ├── equity_curve.csv
    """

    def __init__(self, output_dir: Path):
        """
        Initialize result aggregator

        Args:
            output_dir: Base directory for evaluation results
        """
        self.output_dir = Path(output_dir)
        self.summary_dir = self.output_dir / "summary"
        self.summary_dir.mkdir(parents=True, exist_ok=True)

        self.tickers_dir = self.output_dir / "tickers"

    def save_backtest_result(
        self,
        ticker: str,
        strategy_name: str,
        result: BacktestResult,
    ) -> None:
        """
        Save individual backtest result to JSON and CSV

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name
            result: BacktestResult object
        """
        # Create directory
        strategy_dir = self.tickers_dir / ticker / strategy_name.lower()
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Save backtest result JSON
        result_dict = {
            "ticker": result.ticker,
            "strategy_name": result.strategy_name,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "total_return_pct": result.total_return_pct,
            "num_trades": result.num_trades,
            "num_winners": result.num_winners,
            "num_losers": result.num_losers,
            "win_rate": result.win_rate,
            "average_pnl": result.average_pnl,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
        }

        result_path = strategy_dir / "backtest_result.json"
        with open(result_path, 'w') as f:
            json.dump(result_dict, f, indent=2)

        # Save equity curve CSV
        if not result.equity_curve.empty:
            equity_path = strategy_dir / "equity_curve.csv"
            result.equity_curve.to_csv(equity_path, index=False)

        logger.info(f"Saved backtest result: {ticker} {strategy_name}")

    def load_backtest_result(
        self,
        ticker: str,
        strategy_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Load individual backtest result from JSON

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name

        Returns:
            Backtest result dictionary or None if not found
        """
        result_path = self.tickers_dir / ticker / strategy_name.lower() / "backtest_result.json"

        if not result_path.exists():
            logger.warning(f"Backtest result not found: {result_path}")
            return None

        try:
            with open(result_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load backtest result {result_path}: {e}")
            return None

    def load_all_results(self) -> pd.DataFrame:
        """
        Load all backtest results into a DataFrame

        Returns:
            DataFrame with columns: ticker, strategy_name, total_return_pct, sharpe_ratio, etc.
        """
        results = []

        # Search for all backtest_result.json files
        for result_file in self.tickers_dir.glob("*/*/backtest_result.json"):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                logger.warning(f"Skipping invalid result {result_file}: {e}")
                continue

        if not results:
            logger.warning("No backtest results found")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        logger.info(f"Loaded {len(df)} backtest results")
        return df

    def generate_aggregate_metrics(self) -> pd.DataFrame:
        """
        Generate aggregate metrics across all tickers and strategies

        Returns:
            DataFrame with aggregate statistics
        """
        df = self.load_all_results()

        if df.empty:
            return pd.DataFrame()

        # Group by strategy and calculate aggregate metrics
        agg_metrics = df.groupby('strategy_name').agg({
            'total_return_pct': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std', 'min', 'max'],
            'win_rate': ['mean', 'std'],
            'max_drawdown': ['mean', 'min'],  # min because drawdown is negative
            'num_trades': 'sum',
        }).round(2)

        # Flatten column names
        agg_metrics.columns = ['_'.join(col).strip() for col in agg_metrics.columns.values]
        agg_metrics = agg_metrics.reset_index()

        # Save to CSV
        metrics_path = self.summary_dir / "aggregate_metrics.csv"
        agg_metrics.to_csv(metrics_path, index=False)
        logger.info(f"Saved aggregate metrics: {metrics_path}")

        return agg_metrics

    def generate_performance_matrix(self) -> pd.DataFrame:
        """
        Generate performance matrix (tickers × strategies)

        Returns:
            DataFrame with tickers as rows, strategies as columns, values = total_return_pct
        """
        df = self.load_all_results()

        if df.empty:
            return pd.DataFrame()

        # Pivot table: tickers × strategies
        matrix = df.pivot_table(
            index='ticker',
            columns='strategy_name',
            values='total_return_pct',
            aggfunc='first',
        ).round(2)

        # Add summary row (mean across tickers)
        matrix.loc['MEAN'] = matrix.mean()

        # Save to CSV
        matrix_path = self.summary_dir / "performance_matrix.csv"
        matrix.to_csv(matrix_path)
        logger.info(f"Saved performance matrix: {matrix_path}")

        return matrix

    def generate_strategy_comparison(self) -> pd.DataFrame:
        """
        Generate strategy comparison table

        Returns:
            DataFrame comparing key metrics across strategies
        """
        df = self.load_all_results()

        if df.empty:
            return pd.DataFrame()

        # Group by strategy
        comparison = df.groupby('strategy_name').agg({
            'total_return_pct': 'mean',
            'sharpe_ratio': 'mean',
            'win_rate': 'mean',
            'max_drawdown': 'mean',
            'num_trades': 'sum',
            'num_winners': 'sum',
            'num_losers': 'sum',
        }).round(2)

        # Add derived metrics
        comparison['avg_trades_per_ticker'] = (
            comparison['num_trades'] / df['ticker'].nunique()
        ).round(0)

        # Sort by Sharpe ratio
        comparison = comparison.sort_values('sharpe_ratio', ascending=False)

        # Save to CSV
        comp_path = self.summary_dir / "strategy_comparison.csv"
        comparison.to_csv(comp_path)
        logger.info(f"Saved strategy comparison: {comp_path}")

        return comparison

    def generate_all_summaries(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all summary tables

        Returns:
            Dictionary with:
            - aggregate_metrics: Aggregate statistics
            - performance_matrix: Tickers × strategies matrix
            - strategy_comparison: Strategy comparison table
        """
        logger.info("Generating all summary tables...")

        summaries = {
            "aggregate_metrics": self.generate_aggregate_metrics(),
            "performance_matrix": self.generate_performance_matrix(),
            "strategy_comparison": self.generate_strategy_comparison(),
        }

        logger.info("All summaries generated successfully")
        return summaries

    def get_best_strategy(self, metric: str = "sharpe_ratio") -> Optional[str]:
        """
        Get best performing strategy based on metric

        Args:
            metric: Metric to optimize (sharpe_ratio, total_return_pct, win_rate)

        Returns:
            Best strategy name or None
        """
        df = self.load_all_results()

        if df.empty:
            return None

        # Group by strategy and get mean metric
        strategy_metrics = df.groupby('strategy_name')[metric].mean()

        # Get best strategy
        best_strategy = strategy_metrics.idxmax()
        best_value = strategy_metrics.max()

        logger.info(f"Best strategy ({metric}): {best_strategy} = {best_value:.2f}")
        return best_strategy

    def get_ticker_rankings(self, strategy_name: str, metric: str = "total_return_pct") -> pd.DataFrame:
        """
        Get ticker rankings for a specific strategy

        Args:
            strategy_name: Strategy to analyze
            metric: Metric to rank by

        Returns:
            DataFrame with ticker rankings
        """
        df = self.load_all_results()

        if df.empty:
            return pd.DataFrame()

        # Filter by strategy
        strategy_df = df[df['strategy_name'] == strategy_name].copy()

        # Sort by metric
        rankings = strategy_df.sort_values(metric, ascending=False)[
            ['ticker', metric, 'sharpe_ratio', 'win_rate', 'num_trades']
        ]

        logger.info(f"Generated rankings for {strategy_name}")
        return rankings

    def export_summary_report(self) -> Path:
        """
        Export comprehensive summary report to text file

        Returns:
            Path to summary report
        """
        report_path = self.summary_dir / "summary_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MULTI-TICKER EVALUATION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Load data
            df = self.load_all_results()

            if df.empty:
                f.write("No results found.\n")
                return report_path

            f.write(f"Total Results: {len(df)}\n")
            f.write(f"Tickers: {df['ticker'].nunique()} ({', '.join(sorted(df['ticker'].unique()))})\n")
            f.write(f"Strategies: {df['strategy_name'].nunique()} ({', '.join(sorted(df['strategy_name'].unique()))})\n\n")

            # Strategy comparison
            f.write("-" * 80 + "\n")
            f.write("STRATEGY COMPARISON\n")
            f.write("-" * 80 + "\n\n")

            comparison = self.generate_strategy_comparison()
            f.write(comparison.to_string())
            f.write("\n\n")

            # Best strategy
            best_sharpe = self.get_best_strategy("sharpe_ratio")
            best_return = self.get_best_strategy("total_return_pct")

            f.write("-" * 80 + "\n")
            f.write("BEST STRATEGIES\n")
            f.write("-" * 80 + "\n\n")
            f.write(f"Best Sharpe Ratio: {best_sharpe}\n")
            f.write(f"Best Total Return: {best_return}\n\n")

            # Performance matrix
            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE MATRIX (Total Return %)\n")
            f.write("-" * 80 + "\n\n")

            matrix = self.generate_performance_matrix()
            f.write(matrix.to_string())
            f.write("\n\n")

        logger.info(f"Exported summary report: {report_path}")
        return report_path
