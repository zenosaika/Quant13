"""
Visualization module for backtesting results

Creates charts and reports comparing strategy performance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtesting.framework import BacktestResult


def plot_comparison(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    """
    Create comparison visualizations for all strategies

    Args:
        results: Dictionary of strategy_name -> BacktestResult
        output_dir: Directory to save plots
    """
    # Check if we have any results
    if not results:
        print("No results to visualize")
        return

    # Create figures directory
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Equity curves comparison
    _plot_equity_curves(results, figures_dir)

    # 2. Returns comparison (bar chart)
    _plot_returns_comparison(results, figures_dir)

    # 3. Win rate comparison
    _plot_win_rates(results, figures_dir)

    # 4. Risk metrics comparison
    _plot_risk_metrics(results, figures_dir)

    # 5. Trade distribution
    _plot_trade_distributions(results, figures_dir)


def _plot_equity_curves(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    """Plot equity curves for all strategies"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for strategy_name, result in results.items():
        if not result.equity_curve.empty:
            equity_curve = result.equity_curve.copy()
            # Normalize to percentage
            equity_curve["equity_pct"] = (
                (equity_curve["equity"] / result.initial_capital - 1) * 100
            )

            ax.plot(
                equity_curve["date"],
                equity_curve["equity_pct"],
                label=strategy_name,
                linewidth=2
            )

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Return (%)", fontsize=12)
    ax.set_title("Strategy Performance Comparison - Equity Curves", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_returns_comparison(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    """Plot returns comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = list(results.keys())
    returns = [results[s].total_return_pct for s in strategies]

    colors = ['green' if r > 0 else 'red' for r in returns]

    bars = ax.bar(strategies, returns, color=colors, alpha=0.7)

    # Add value labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{ret:.2f}%',
            ha='center',
            va='bottom' if ret > 0 else 'top',
            fontweight='bold'
        )

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Total Return (%)", fontsize=12)
    ax.set_title("Total Return Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "returns_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_win_rates(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    """Plot win rates comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = list(results.keys())
    win_rates = [results[s].win_rate * 100 for s in strategies]

    bars = ax.bar(strategies, win_rates, color='steelblue', alpha=0.7)

    # Add value labels
    for bar, wr in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{wr:.1f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (Break-even)')
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate Comparison", fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "win_rates.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_risk_metrics(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    """Plot risk metrics comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    strategies = list(results.keys())

    # Max Drawdown
    drawdowns = [abs(results[s].max_drawdown) * 100 for s in strategies]
    bars1 = ax1.bar(strategies, drawdowns, color='crimson', alpha=0.7)

    for bar, dd in zip(bars1, drawdowns):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{dd:.2f}%',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    ax1.set_ylabel("Max Drawdown (%)", fontsize=12)
    ax1.set_title("Maximum Drawdown", fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    # Sharpe Ratio
    sharpes = [results[s].sharpe_ratio for s in strategies]
    colors = ['green' if s > 0 else 'red' for s in sharpes]
    bars2 = ax2.bar(strategies, sharpes, color=colors, alpha=0.7)

    for bar, sharpe in zip(bars2, sharpes):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{sharpe:.2f}',
            ha='center',
            va='bottom' if sharpe > 0 else 'top',
            fontweight='bold'
        )

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylabel("Sharpe Ratio", fontsize=12)
    ax2.set_title("Risk-Adjusted Returns (Sharpe Ratio)", fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "risk_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()


def _plot_trade_distributions(results: Dict[str, BacktestResult], output_dir: Path) -> None:
    """Plot P&L distribution for each strategy"""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))

    if len(results) == 1:
        axes = [axes]

    for ax, (strategy_name, result) in zip(axes, results.items()):
        pnls = [trade.pnl for trade in result.trades if trade.pnl is not None]

        if pnls:
            ax.hist(pnls, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
            ax.axvline(
                x=np.mean(pnls),
                color='green',
                linestyle='-',
                linewidth=2,
                label=f'Mean: ${np.mean(pnls):.2f}'
            )

            ax.set_xlabel("P&L ($)", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(f"{strategy_name}\nP&L Distribution", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No trades", ha='center', va='center', fontsize=14)
            ax.set_title(strategy_name, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "pnl_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_evaluation_report(
    results: Dict[str, BacktestResult],
    summary: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Create a detailed markdown report

    Args:
        results: Dictionary of BacktestResults
        summary: Summary dictionary
        output_dir: Output directory
    """
    report_path = output_dir / "EVALUATION_REPORT.md"

    with open(report_path, "w") as f:
        # Header
        f.write("# Options Trading Strategy Evaluation Report\n\n")
        f.write(f"**Ticker:** {summary['ticker']}  \n")
        f.write(f"**Period:** {summary['start_date']} to {summary['end_date']} ({summary['duration_days']} days)  \n")
        f.write(f"**Initial Capital:** ${summary['initial_capital']:,.2f}  \n")
        f.write(f"**Evaluation Date:** {summary['timestamp']}  \n\n")

        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].total_return_pct,
            reverse=True
        )

        f.write("### Performance Ranking\n\n")
        f.write("| Rank | Strategy | Total Return | Final Capital | Win Rate | Sharpe Ratio |\n")
        f.write("|------|----------|--------------|---------------|----------|-------------|\n")

        for rank, (strategy_name, result) in enumerate(sorted_results, 1):
            f.write(
                f"| {rank} | {strategy_name} | {result.total_return_pct:+.2f}% "
                f"(${result.total_return:+,.2f}) | ${result.final_capital:,.2f} | "
                f"{result.win_rate:.1%} | {result.sharpe_ratio:.2f} |\n"
            )

        f.write("\n---\n\n")

        # Detailed Results
        f.write("## Detailed Strategy Analysis\n\n")

        for strategy_name, result in results.items():
            f.write(f"### {strategy_name}\n\n")

            f.write("**Performance Metrics:**\n\n")
            f.write(f"- Total Return: **{result.total_return_pct:+.2f}%** (${result.total_return:+,.2f})  \n")
            f.write(f"- Final Capital: ${result.final_capital:,.2f}  \n")
            f.write(f"- Number of Trades: {result.num_trades}  \n")
            f.write(f"- Winners: {result.num_winners} | Losers: {result.num_losers}  \n")
            f.write(f"- Win Rate: {result.win_rate:.1%}  \n")
            f.write(f"- Average P&L per Trade: ${result.average_pnl:,.2f}  \n")
            f.write(f"- Maximum Drawdown: {abs(result.max_drawdown):.2%}  \n")
            f.write(f"- Sharpe Ratio: {result.sharpe_ratio:.2f}  \n\n")

            # Trade details
            if result.trades:
                f.write("**Trade Details:**\n\n")
                f.write("| Date | Strategy Type | Entry | Exit | P&L | P&L % |\n")
                f.write("|------|---------------|-------|------|-----|-------|\n")

                for trade in result.trades[:10]:  # Show first 10 trades
                    entry_date = trade.date.strftime("%Y-%m-%d")
                    exit_date = trade.closed_date.strftime("%Y-%m-%d") if trade.closed_date else "Open"
                    pnl = trade.pnl if trade.pnl else 0.0
                    pnl_pct = trade.pnl_pct if trade.pnl_pct else 0.0

                    f.write(
                        f"| {entry_date} | {trade.strategy_name} | "
                        f"${trade.entry_price:.2f} | {exit_date} | "
                        f"${pnl:+.2f} | {pnl_pct:+.1f}% |\n"
                    )

                if len(result.trades) > 10:
                    f.write(f"\n*Showing 10 of {len(result.trades)} trades*\n")

            f.write("\n")

        # Visualizations
        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("### Equity Curves\n\n")
        f.write("![Equity Curves](figures/equity_curves.png)\n\n")
        f.write("### Returns Comparison\n\n")
        f.write("![Returns](figures/returns_comparison.png)\n\n")
        f.write("### Win Rates\n\n")
        f.write("![Win Rates](figures/win_rates.png)\n\n")
        f.write("### Risk Metrics\n\n")
        f.write("![Risk Metrics](figures/risk_metrics.png)\n\n")
        f.write("### P&L Distributions\n\n")
        f.write("![P&L Distributions](figures/pnl_distributions.png)\n\n")

        # Methodology
        f.write("---\n\n")
        f.write("## Methodology\n\n")
        f.write("### Data Sources\n\n")
        f.write("- **Historical Prices:** yfinance OHLCV data  \n")
        f.write("- **Options Chain:** Black-Scholes synthetic options (historical IV from price volatility)  \n")
        f.write("- **Rebalancing:** Weekly  \n")
        f.write("- **Position Sizing:** 20% of capital per trade  \n\n")

        f.write("### Strategies\n\n")
        f.write("1. **Quant13 System:** Multi-agent framework with technical, fundamental, and volatility analysis (no news)  \n")
        f.write("2. **Technical Baseline:** Simple RSI + MACD + SMA200 based options trading  \n")
        f.write("3. **Chimpanzee:** Random options selection (control)  \n\n")

        f.write("### Data Leakage Prevention\n\n")
        f.write("- Only data up to each rebalance date is used  \n")
        f.write("- No future information or news data  \n")
        f.write("- Options chains synthetically generated using Black-Scholes with historical volatility  \n\n")

        f.write("---\n\n")
        f.write("*Report generated by Quant13 Backtesting Framework*\n")

    print(f"Report saved: {report_path}")
