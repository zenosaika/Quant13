#!/usr/bin/env python3
"""
Evaluation Visualizations for Quant13

Creates charts and plots to visualize strategy performance.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def create_evaluation_visualizations(results_dir: str, output_dir: str = None) -> Dict[str, str]:
    """
    Create all visualizations for evaluation results.

    Args:
        results_dir: Path to evaluation results directory
        output_dir: Output directory for charts (defaults to results_dir/charts)

    Returns:
        Dictionary of chart paths
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'charts')

    os.makedirs(output_dir, exist_ok=True)

    # Load summary data
    summary_path = os.path.join(results_dir, 'summary', 'strategy_comparison.csv')
    perf_matrix_path = os.path.join(results_dir, 'summary', 'performance_matrix.csv')

    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return {}

    summary_df = pd.read_csv(summary_path, index_col=0)
    perf_matrix_df = pd.read_csv(perf_matrix_path, index_col=0) if os.path.exists(perf_matrix_path) else None

    charts = {}

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Strategy Comparison Bar Chart
    charts['strategy_comparison'] = _create_strategy_comparison_chart(summary_df, output_dir)

    # 2. Win Rate Comparison
    charts['win_rate'] = _create_win_rate_chart(summary_df, output_dir)

    # 3. Performance by Ticker Heatmap
    if perf_matrix_df is not None:
        charts['ticker_heatmap'] = _create_ticker_heatmap(perf_matrix_df, output_dir)

    # 4. Risk-Adjusted Returns (Sharpe)
    charts['sharpe_comparison'] = _create_sharpe_chart(summary_df, output_dir)

    # 5. Trade Statistics
    charts['trade_stats'] = _create_trade_stats_chart(summary_df, output_dir)

    # 6. Comprehensive Dashboard
    charts['dashboard'] = _create_dashboard(summary_df, perf_matrix_df, output_dir)

    return charts


def _create_strategy_comparison_chart(df: pd.DataFrame, output_dir: str) -> str:
    """Create strategy comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = df.index.tolist()
    returns = df['total_return_pct'].values

    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]

    bars = ax.bar(strategies, returns, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars, returns):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('Total Return (%)', fontsize=12)
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_title('Strategy Performance Comparison\n30-Day Evaluation (Nov 2024)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'strategy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path


def _create_win_rate_chart(df: pd.DataFrame, output_dir: str) -> str:
    """Create win rate comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = df.index.tolist()
    win_rates = df['win_rate'].values * 100  # Convert to percentage

    colors = ['#3498db', '#e67e22', '#9b59b6']

    bars = ax.bar(strategies, win_rates, color=colors[:len(strategies)], edgecolor='black', linewidth=1.2)

    # Add 50% reference line
    ax.axhline(y=50, color='red', linestyle='--', linewidth=1.5, label='50% (Random)')

    # Add value labels
    for bar, val in zip(bars, win_rates):
        ax.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

    ax.set_ylabel('Win Rate (%)', fontsize=12)
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_title('Win Rate Comparison\nHigher is Better', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend(loc='lower right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'win_rate_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path


def _create_ticker_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Create performance by ticker heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Remove MEAN row if exists
    if 'MEAN' in df.index:
        df = df.drop('MEAN')

    # Create heatmap data
    data = df.values

    # Create heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-30, vmax=30)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Return (%)', rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns, fontsize=11)
    ax.set_yticklabels(df.index, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = data[i, j]
            color = 'white' if abs(val) > 15 else 'black'
            text = ax.text(j, i, f'{val:.1f}%', ha="center", va="center",
                          color=color, fontsize=10, fontweight='bold')

    ax.set_title('Performance by Ticker (%)\nGreen = Profit, Red = Loss', fontsize=14, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, 'ticker_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path


def _create_sharpe_chart(df: pd.DataFrame, output_dir: str) -> str:
    """Create Sharpe ratio comparison chart"""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = df.index.tolist()
    sharpe = df['sharpe_ratio'].values

    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpe]

    bars = ax.barh(strategies, sharpe, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, val in zip(bars, sharpe):
        width = bar.get_width()
        ax.annotate(f'{val:.2f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if width >= 0 else -30, 0),
                    textcoords="offset points",
                    ha='left' if width >= 0 else 'right', va='center',
                    fontsize=12, fontweight='bold')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=1, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (>1)')
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Poor (<-1)')

    ax.set_xlabel('Sharpe Ratio', fontsize=12)
    ax.set_ylabel('Strategy', fontsize=12)
    ax.set_title('Risk-Adjusted Returns (Sharpe Ratio)\nHigher is Better', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'sharpe_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path


def _create_trade_stats_chart(df: pd.DataFrame, output_dir: str) -> str:
    """Create trade statistics chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    strategies = df.index.tolist()

    # Left: Number of trades (winners vs losers)
    ax1 = axes[0]
    winners = df['num_winners'].values
    losers = df['num_losers'].values

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax1.bar(x - width/2, winners, width, label='Winners', color='#2ecc71', edgecolor='black')
    bars2 = ax1.bar(x + width/2, losers, width, label='Losers', color='#e74c3c', edgecolor='black')

    ax1.set_ylabel('Number of Trades', fontsize=11)
    ax1.set_title('Winners vs Losers', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies)
    ax1.legend()

    # Add value labels
    for bar in bars1:
        ax1.annotate(f'{int(bar.get_height())}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    for bar in bars2:
        ax1.annotate(f'{int(bar.get_height())}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # Right: Max Drawdown
    ax2 = axes[1]
    drawdowns = df['max_drawdown'].values * 100  # Convert to percentage

    bars = ax2.bar(strategies, drawdowns, color='#e74c3c', edgecolor='black', linewidth=1.2)

    for bar, val in zip(bars, drawdowns):
        ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, -15), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax2.set_title('Maximum Drawdown\nLower is Better', fontsize=12, fontweight='bold')
    ax2.set_ylim(min(drawdowns) * 1.3, 0)

    plt.tight_layout()
    path = os.path.join(output_dir, 'trade_stats.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path


def _create_dashboard(summary_df: pd.DataFrame, perf_df: pd.DataFrame, output_dir: str) -> str:
    """Create comprehensive dashboard"""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Total Returns (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    strategies = summary_df.index.tolist()
    returns = summary_df['total_return_pct'].values
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
    bars = ax1.bar(strategies, returns, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_title('Total Return (%)', fontweight='bold')
    for bar, val in zip(bars, returns):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3 if val >= 0 else -12), textcoords="offset points",
                     ha='center', fontsize=9, fontweight='bold')

    # 2. Win Rate (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    win_rates = summary_df['win_rate'].values * 100
    colors = ['#3498db', '#e67e22', '#9b59b6'][:len(strategies)]
    bars = ax2.bar(strategies, win_rates, color=colors, edgecolor='black')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Win Rate (%)', fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, val in zip(bars, win_rates):
        ax2.annotate(f'{val:.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # 3. Sharpe Ratio (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    sharpe = summary_df['sharpe_ratio'].values
    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sharpe]
    bars = ax3.barh(strategies, sharpe, color=colors, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_title('Sharpe Ratio', fontweight='bold')
    for bar, val in zip(bars, sharpe):
        ax3.annotate(f'{val:.2f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                     xytext=(3 if val >= 0 else -25, 0), textcoords="offset points",
                     ha='left' if val >= 0 else 'right', va='center', fontsize=9, fontweight='bold')

    # 4. Performance Heatmap (middle, spanning 2 columns)
    if perf_df is not None:
        ax4 = fig.add_subplot(gs[1, :2])
        perf_data = perf_df.drop('MEAN') if 'MEAN' in perf_df.index else perf_df
        im = ax4.imshow(perf_data.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
        ax4.set_xticks(np.arange(len(perf_data.columns)))
        ax4.set_yticks(np.arange(len(perf_data.index)))
        ax4.set_xticklabels(perf_data.columns, fontsize=9)
        ax4.set_yticklabels(perf_data.index, fontsize=9)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        ax4.set_title('Return by Ticker (%)', fontweight='bold')
        for i in range(len(perf_data.index)):
            for j in range(len(perf_data.columns)):
                val = perf_data.values[i, j]
                color = 'white' if abs(val) > 15 else 'black'
                ax4.text(j, i, f'{val:.0f}', ha="center", va="center", color=color, fontsize=8)

    # 5. Trade Count (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    winners = summary_df['num_winners'].values
    losers = summary_df['num_losers'].values
    x = np.arange(len(strategies))
    width = 0.35
    ax5.bar(x - width/2, winners, width, label='Win', color='#2ecc71')
    ax5.bar(x + width/2, losers, width, label='Loss', color='#e74c3c')
    ax5.set_xticks(x)
    ax5.set_xticklabels(strategies, fontsize=9)
    ax5.set_title('Wins vs Losses', fontweight='bold')
    ax5.legend(fontsize=8)

    # 6. Summary Table (bottom)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    # Create summary text
    best_return = summary_df['total_return_pct'].idxmax()
    best_winrate = summary_df['win_rate'].idxmax()
    best_sharpe = summary_df['sharpe_ratio'].idxmax()

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                           EVALUATION SUMMARY                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Best Total Return:    {best_return:<12} ({summary_df.loc[best_return, 'total_return_pct']:+.2f}%)                           ║
    ║  Best Win Rate:        {best_winrate:<12} ({summary_df.loc[best_winrate, 'win_rate']*100:.0f}%)                               ║
    ║  Best Sharpe Ratio:    {best_sharpe:<12} ({summary_df.loc[best_sharpe, 'sharpe_ratio']:.2f})                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Period: November 2024 (30 days) | Tickers: 10 | Initial Capital: $10,000    ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    ax6.text(0.5, 0.5, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Quant13 Strategy Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.98)

    path = os.path.join(output_dir, 'dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find latest results
        import glob
        results_dirs = sorted(glob.glob('results/evaluation_*'))
        if results_dirs:
            results_dir = results_dirs[-1]
        else:
            print("No results found")
            sys.exit(1)

    print(f"Creating visualizations for: {results_dir}")
    charts = create_evaluation_visualizations(results_dir)
    print(f"Created {len(charts)} charts:")
    for name, path in charts.items():
        print(f"  - {name}: {path}")
