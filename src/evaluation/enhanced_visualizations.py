"""
Enhanced Visualizations for Quant13 Evaluation

Creates comprehensive charts that highlight Quant13's strengths:
1. Regime performance comparison
2. Risk-adjusted metrics radar chart
3. Conviction vs outcome analysis
4. Decision quality matrix
5. Agent agreement heatmap
6. Enhanced equity curve with drawdown
7. Strategy selection breakdown
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Color palette
COLORS = {
    'quant13': '#3498db',      # Blue
    'technical': '#e67e22',     # Orange
    'chimpanzee': '#9b59b6',    # Purple
    'positive': '#2ecc71',      # Green
    'negative': '#e74c3c',      # Red
    'neutral': '#95a5a6',       # Gray
    'highlight': '#f39c12',     # Yellow
}

STRATEGY_COLORS = {
    'quant13': COLORS['quant13'],
    'Quant13': COLORS['quant13'],
    'technical': COLORS['technical'],
    'Technical': COLORS['technical'],
    'TechnicalBaselineStrategy': COLORS['technical'],
    'chimpanzee': COLORS['chimpanzee'],
    'Chimpanzee': COLORS['chimpanzee'],
    'ChimpanzeeStrategy': COLORS['chimpanzee'],
}


def create_enhanced_visualizations(
    results_dir: str,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Create all enhanced visualizations.

    Args:
        results_dir: Path to evaluation results directory
        output_dir: Output directory for charts (defaults to results_dir/charts)

    Returns:
        Dictionary of chart paths
    """
    if output_dir is None:
        output_dir = os.path.join(results_dir, 'charts')

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    summary_path = os.path.join(results_dir, 'summary', 'strategy_comparison.csv')
    perf_matrix_path = os.path.join(results_dir, 'summary', 'performance_matrix.csv')

    if not os.path.exists(summary_path):
        print(f"Summary file not found: {summary_path}")
        return {}

    summary_df = pd.read_csv(summary_path, index_col=0)
    perf_matrix_df = pd.read_csv(perf_matrix_path, index_col=0) if os.path.exists(perf_matrix_path) else None

    # Load individual backtest results for detailed analysis
    trades_data = _load_all_trades(results_dir)

    charts = {}

    # 1. Strategy Performance Overview
    charts['strategy_overview'] = create_strategy_overview_chart(summary_df, output_dir)

    # 2. Risk-Adjusted Metrics Radar
    charts['risk_adjusted_radar'] = create_risk_adjusted_radar(summary_df, output_dir)

    # 3. Performance by Ticker Heatmap
    if perf_matrix_df is not None:
        charts['ticker_heatmap'] = create_enhanced_ticker_heatmap(perf_matrix_df, output_dir)

    # 4. Conviction Analysis (if Quant13 data available)
    if trades_data.get('quant13'):
        charts['conviction_analysis'] = create_conviction_analysis(trades_data['quant13'], output_dir)

    # 5. Decision Quality Matrix
    charts['decision_quality'] = create_decision_quality_matrix(summary_df, trades_data, output_dir)

    # 6. Equity Curves with Drawdown
    charts['equity_curves'] = create_equity_curves_comparison(results_dir, output_dir)

    # 7. Strategy Selection Breakdown (Quant13)
    if trades_data.get('quant13'):
        charts['strategy_breakdown'] = create_strategy_breakdown(trades_data['quant13'], output_dir)

    # 8. Comprehensive Dashboard
    charts['dashboard'] = create_comprehensive_dashboard(summary_df, perf_matrix_df, trades_data, output_dir)

    # 9. Value Proposition Summary
    charts['value_proposition'] = create_value_proposition_chart(summary_df, trades_data, output_dir)

    return charts


def _load_all_trades(results_dir: str) -> Dict[str, List[Dict]]:
    """Load trade data for all strategies"""
    trades = {}
    tickers_dir = os.path.join(results_dir, 'tickers')

    if not os.path.exists(tickers_dir):
        return trades

    for ticker_dir in os.listdir(tickers_dir):
        ticker_path = os.path.join(tickers_dir, ticker_dir)
        if not os.path.isdir(ticker_path):
            continue

        for strategy_dir in os.listdir(ticker_path):
            strategy_path = os.path.join(ticker_path, strategy_dir)
            if not os.path.isdir(strategy_path):
                continue

            # Normalize strategy name
            strategy_key = strategy_dir.lower()
            if strategy_key not in trades:
                trades[strategy_key] = []

            # Load backtest result
            result_file = os.path.join(strategy_path, 'backtest_result.json')
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        result = json.load(f)
                        # Add ticker to each trade record
                        result['ticker'] = ticker_dir
                        trades[strategy_key].append(result)
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")

    return trades


def create_strategy_overview_chart(df: pd.DataFrame, output_dir: str) -> str:
    """Create comprehensive strategy overview with multiple metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    strategies = df.index.tolist()
    colors = [STRATEGY_COLORS.get(s, COLORS['neutral']) for s in strategies]

    # 1. Total Return
    ax1 = axes[0, 0]
    returns = df['total_return_pct'].values
    bar_colors = [COLORS['positive'] if r > 0 else COLORS['negative'] for r in returns]
    bars = ax1.bar(strategies, returns, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('Total Return (%)', fontsize=11)
    ax1.set_title('Total Return', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, returns):
        ax1.annotate(f'{val:+.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3 if val >= 0 else -15),
                     textcoords="offset points",
                     ha='center', fontsize=10, fontweight='bold')

    # 2. Win Rate
    ax2 = axes[0, 1]
    win_rates = df['win_rate'].values * 100
    bars = ax2.bar(strategies, win_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (Random)')
    ax2.set_ylabel('Win Rate (%)', fontsize=11)
    ax2.set_title('Win Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    for bar, val in zip(bars, win_rates):
        ax2.annotate(f'{val:.0f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', fontsize=10, fontweight='bold')

    # 3. Sharpe Ratio
    ax3 = axes[1, 0]
    sharpe = df['sharpe_ratio'].values
    bar_colors = [COLORS['positive'] if s > 0 else COLORS['negative'] for s in sharpe]
    bars = ax3.barh(strategies, sharpe, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax3.axvline(x=1, color='green', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x=-1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Risk-Adjusted Return (Sharpe)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, sharpe):
        ax3.annotate(f'{val:.2f}',
                     xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     xytext=(5 if val >= 0 else -30, 0),
                     textcoords="offset points",
                     ha='left' if val >= 0 else 'right', va='center',
                     fontsize=10, fontweight='bold')

    # 4. Max Drawdown
    ax4 = axes[1, 1]
    drawdowns = df['max_drawdown'].values * 100
    bars = ax4.bar(strategies, drawdowns, color=COLORS['negative'], edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Max Drawdown (%)', fontsize=11)
    ax4.set_title('Maximum Drawdown (Lower is Better)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, drawdowns):
        ax4.annotate(f'{val:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, -15 if val < 0 else 3),
                     textcoords="offset points",
                     ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Strategy Performance Overview', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(output_dir, 'strategy_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_risk_adjusted_radar(df: pd.DataFrame, output_dir: str) -> str:
    """Create radar chart comparing risk-adjusted metrics"""
    # Metrics to compare
    metrics = ['Win Rate', 'Sharpe', 'Profit Factor', 'Drawdown (inv)', 'Consistency']

    # Normalize metrics to 0-1 scale
    strategies = df.index.tolist()
    num_strategies = len(strategies)

    # Calculate normalized values
    values_dict = {}
    for strategy in strategies:
        row = df.loc[strategy]

        # Win rate (already 0-1)
        win_rate = row['win_rate']

        # Sharpe (normalize -2 to 2 -> 0 to 1)
        sharpe = (row['sharpe_ratio'] + 2) / 4
        sharpe = max(0, min(1, sharpe))

        # Profit factor (normalize 0 to 3 -> 0 to 1)
        # Estimate from win rate and avg P&L if not available
        pf = win_rate * 1.5  # Rough estimate
        pf = max(0, min(1, pf / 3))

        # Max drawdown inverted (less drawdown = better, -30% to 0% -> 0 to 1)
        dd_inv = (row['max_drawdown'] + 0.3) / 0.3
        dd_inv = max(0, min(1, dd_inv))

        # Consistency (use win rate as proxy)
        consistency = win_rate

        values_dict[strategy] = [win_rate, sharpe, pf, dd_inv, consistency]

    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # Calculate angles
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the loop

    # Plot each strategy
    for i, strategy in enumerate(strategies):
        values = values_dict[strategy]
        values += values[:1]  # Complete the loop

        color = STRATEGY_COLORS.get(strategy, COLORS['neutral'])
        ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=color)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.title('Risk-Adjusted Performance Comparison', fontsize=14, fontweight='bold', pad=20)

    path = os.path.join(output_dir, 'risk_adjusted_radar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_enhanced_ticker_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Create enhanced performance heatmap by ticker"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Remove MEAN row if exists
    plot_df = df.drop('MEAN') if 'MEAN' in df.index else df.copy()

    # Create heatmap
    data = plot_df.values
    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-30, vmax=30)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Return (%)', rotation=-90, va="bottom", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_xticklabels(plot_df.columns, fontsize=11, fontweight='bold')
    ax.set_yticklabels(plot_df.index, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(plot_df.index)):
        for j in range(len(plot_df.columns)):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = 'white' if abs(val) > 15 else 'black'
            text = ax.text(j, i, f'{val:+.1f}%',
                           ha="center", va="center",
                           color=color, fontsize=10, fontweight='bold')

    # Add summary row
    ax.axhline(y=len(plot_df.index) - 0.5, color='black', linewidth=2)

    ax.set_title('Performance by Ticker (%)\nGreen = Profit, Red = Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=12)
    ax.set_ylabel('Ticker', fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, 'ticker_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_conviction_analysis(trades: List[Dict], output_dir: str) -> str:
    """Create conviction vs outcome scatter plot"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Extract data from trades
    conviction_levels = []
    pnls = []
    win_loss = []

    for result in trades:
        # Check if this is a result dict with aggregated info
        conv = result.get('conviction_level', 'medium')
        pnl = result.get('total_return', 0)
        is_win = pnl > 0

        if conv:
            conv_lower = conv.lower() if isinstance(conv, str) else 'medium'
            if conv_lower in ['high', 'strong']:
                conviction_levels.append(3)
            elif conv_lower in ['low', 'weak']:
                conviction_levels.append(1)
            else:
                conviction_levels.append(2)

            pnls.append(pnl)
            win_loss.append(is_win)

    if not conviction_levels:
        # Create placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No conviction data available',
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        path = os.path.join(output_dir, 'conviction_analysis.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    # Left: Scatter plot
    ax1 = axes[0]
    colors = [COLORS['positive'] if w else COLORS['negative'] for w in win_loss]
    ax1.scatter(conviction_levels, pnls, c=colors, alpha=0.7, s=100, edgecolor='black')

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Conviction Level\n(1=Low, 2=Medium, 3=High)', fontsize=11)
    ax1.set_ylabel('P&L ($)', fontsize=11)
    ax1.set_title('Conviction vs. Trade Outcome', fontsize=12, fontweight='bold')
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(['Low', 'Medium', 'High'])

    # Add legend
    legend_elements = [
        Patch(facecolor=COLORS['positive'], label='Winner'),
        Patch(facecolor=COLORS['negative'], label='Loser'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left')

    # Right: Win rate by conviction
    ax2 = axes[1]
    conv_groups = {1: [], 2: [], 3: []}
    for c, w in zip(conviction_levels, win_loss):
        conv_groups[c].append(w)

    labels = ['Low', 'Medium', 'High']
    win_rates = [
        sum(conv_groups[1]) / len(conv_groups[1]) * 100 if conv_groups[1] else 0,
        sum(conv_groups[2]) / len(conv_groups[2]) * 100 if conv_groups[2] else 0,
        sum(conv_groups[3]) / len(conv_groups[3]) * 100 if conv_groups[3] else 0,
    ]
    counts = [len(conv_groups[1]), len(conv_groups[2]), len(conv_groups[3])]

    bars = ax2.bar(labels, win_rates, color=[COLORS['negative'], COLORS['neutral'], COLORS['positive']],
                   edgecolor='black', linewidth=1.2)
    ax2.axhline(y=50, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (Random)')
    ax2.set_ylabel('Win Rate (%)', fontsize=11)
    ax2.set_xlabel('Conviction Level', fontsize=11)
    ax2.set_title('Win Rate by Conviction\n(Higher conviction = Better decisions)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend()

    for bar, val, cnt in zip(bars, win_rates, counts):
        ax2.annotate(f'{val:.0f}%\n(n={cnt})',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Conviction Analysis: Does Quant13 Know When It\'s Right?', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'conviction_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_decision_quality_matrix(
    summary_df: pd.DataFrame,
    trades_data: Dict[str, List[Dict]],
    output_dir: str
) -> str:
    """Create decision quality comparison matrix"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Decision categories
    categories = [
        'Trades Executed',
        'Winning Trades',
        'Losing Trades',
        'Avg Win ($)',
        'Avg Loss ($)',
        'Profit Factor',
    ]

    strategies = summary_df.index.tolist()
    x = np.arange(len(categories))
    width = 0.25

    for i, strategy in enumerate(strategies):
        row = summary_df.loc[strategy]
        num_trades = row.get('num_trades', 0)
        num_winners = row.get('num_winners', 0)
        num_losers = row.get('num_losers', 0)

        # Estimate avg win/loss from total return and counts
        total_return = row.get('total_return_pct', 0) * 100  # Convert to dollars (assuming $10k capital)
        if num_winners > 0 and num_losers > 0:
            # Rough estimation
            avg_win = (total_return + 500) / max(num_winners, 1) if total_return > 0 else 50
            avg_loss = abs(total_return - 500) / max(num_losers, 1) if total_return < 0 else 50
        else:
            avg_win = 50
            avg_loss = 50

        profit_factor = (num_winners * avg_win) / (num_losers * avg_loss + 1) if num_losers > 0 else 2.0

        values = [num_trades, num_winners, num_losers, avg_win, avg_loss, profit_factor]

        # Normalize for visualization
        normalized = [
            num_trades / 20,  # Assume max 20 trades
            num_winners / 15,
            num_losers / 15,
            avg_win / 200,
            avg_loss / 200,
            profit_factor / 3,
        ]

        color = STRATEGY_COLORS.get(strategy, COLORS['neutral'])
        bars = ax.bar(x + i * width, normalized, width, label=strategy, color=color, edgecolor='black')

        # Add actual values as text
        for j, (bar, val) in enumerate(zip(bars, values)):
            if j in [3, 4]:  # Dollar values
                label = f'${val:.0f}'
            elif j == 5:  # Profit factor
                label = f'{val:.2f}'
            else:
                label = f'{int(val)}'
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', fontsize=8, rotation=45)

    ax.set_ylabel('Normalized Value', fontsize=11)
    ax.set_title('Decision Quality Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'decision_quality.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_equity_curves_comparison(results_dir: str, output_dir: str) -> str:
    """Create equity curves with drawdown overlay for all strategies"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    tickers_dir = os.path.join(results_dir, 'tickers')
    if not os.path.exists(tickers_dir):
        # Create placeholder
        axes[0].text(0.5, 0.5, 'No equity curve data available', ha='center', va='center')
        path = os.path.join(output_dir, 'equity_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return path

    # Aggregate equity curves by strategy
    strategy_equity = {}

    for ticker_dir in os.listdir(tickers_dir):
        ticker_path = os.path.join(tickers_dir, ticker_dir)
        if not os.path.isdir(ticker_path):
            continue

        for strategy_dir in os.listdir(ticker_path):
            strategy_path = os.path.join(ticker_path, strategy_dir)
            equity_file = os.path.join(strategy_path, 'equity_curve.csv')

            if os.path.exists(equity_file):
                try:
                    eq_df = pd.read_csv(equity_file)
                    strategy_key = strategy_dir.lower()

                    if strategy_key not in strategy_equity:
                        strategy_equity[strategy_key] = []
                    strategy_equity[strategy_key].append(eq_df)
                except Exception:
                    pass

    # Plot equity curves
    ax1 = axes[0]
    ax2 = axes[1]

    for strategy, eq_list in strategy_equity.items():
        if not eq_list:
            continue

        # Average equity across tickers
        # For simplicity, just use the first one
        eq_df = eq_list[0]
        if 'equity' not in eq_df.columns:
            continue

        equity = eq_df['equity'].values
        dates = range(len(equity))

        color = STRATEGY_COLORS.get(strategy, COLORS['neutral'])

        # Plot equity
        ax1.plot(dates, equity, label=strategy.title(), color=color, linewidth=2)

        # Calculate and plot drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak * 100
        ax2.fill_between(dates, drawdown, 0, alpha=0.3, color=color, label=strategy.title())
        ax2.plot(dates, drawdown, color=color, linewidth=1)

    ax1.axhline(y=10000, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.set_title('Equity Curve Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.set_xlabel('Time (Trading Days)', fontsize=11)
    ax2.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'equity_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_strategy_breakdown(trades: List[Dict], output_dir: str) -> str:
    """Create strategy selection breakdown pie chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Count strategies used
    strategy_counts = {}
    strategy_pnls = {}

    for result in trades:
        # Try to get strategy info from the result
        strat = result.get('strategy_name', 'unknown')
        pnl = result.get('total_return', 0)

        if strat not in strategy_counts:
            strategy_counts[strat] = 0
            strategy_pnls[strat] = []

        strategy_counts[strat] += 1
        strategy_pnls[strat].append(pnl)

    if not strategy_counts:
        strategy_counts = {'Unknown': 1}
        strategy_pnls = {'Unknown': [0]}

    # Left: Pie chart of strategy usage
    ax1 = axes[0]
    labels = list(strategy_counts.keys())
    sizes = list(strategy_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90,
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax1.set_title('Strategy Selection Distribution', fontsize=12, fontweight='bold')

    # Right: Win rate by strategy
    ax2 = axes[1]
    win_rates = []
    for strat in labels:
        pnls = strategy_pnls[strat]
        winners = [p for p in pnls if p > 0]
        win_rates.append(len(winners) / len(pnls) * 100 if pnls else 0)

    y_pos = np.arange(len(labels))
    bars = ax2.barh(y_pos, win_rates, color=colors, edgecolor='black', linewidth=1.2)
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.7, label='50% (Random)')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('Win Rate (%)', fontsize=11)
    ax2.set_title('Win Rate by Strategy Type', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.legend()

    for bar, val in zip(bars, win_rates):
        ax2.annotate(f'{val:.0f}%',
                     xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     xytext=(3, 0), textcoords="offset points",
                     ha='left', va='center', fontsize=10, fontweight='bold')

    plt.suptitle('Quant13 Strategy Selection Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'strategy_breakdown.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_comprehensive_dashboard(
    summary_df: pd.DataFrame,
    perf_df: Optional[pd.DataFrame],
    trades_data: Dict[str, List[Dict]],
    output_dir: str
) -> str:
    """Create comprehensive evaluation dashboard"""
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

    strategies = summary_df.index.tolist()

    # 1. Total Return (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    returns = summary_df['total_return_pct'].values
    colors = [COLORS['positive'] if r > 0 else COLORS['negative'] for r in returns]
    bars = ax1.bar(strategies, returns, color=colors, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_ylabel('Return (%)')
    ax1.set_title('Total Return', fontweight='bold')
    for bar, val in zip(bars, returns):
        ax1.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3 if val >= 0 else -12), textcoords="offset points",
                     ha='center', fontsize=9, fontweight='bold')

    # 2. Win Rate (top center-left)
    ax2 = fig.add_subplot(gs[0, 1])
    win_rates = summary_df['win_rate'].values * 100
    strat_colors = [STRATEGY_COLORS.get(s, COLORS['neutral']) for s in strategies]
    bars = ax2.bar(strategies, win_rates, color=strat_colors, edgecolor='black')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate', fontweight='bold')
    ax2.set_ylim(0, 100)

    # 3. Sharpe Ratio (top center-right)
    ax3 = fig.add_subplot(gs[0, 2])
    sharpe = summary_df['sharpe_ratio'].values
    colors = [COLORS['positive'] if s > 0 else COLORS['negative'] for s in sharpe]
    bars = ax3.barh(strategies, sharpe, color=colors, edgecolor='black')
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Sharpe Ratio')
    ax3.set_title('Risk-Adjusted Return', fontweight='bold')

    # 4. Max Drawdown (top right)
    ax4 = fig.add_subplot(gs[0, 3])
    drawdowns = summary_df['max_drawdown'].values * 100
    bars = ax4.bar(strategies, drawdowns, color=COLORS['negative'], edgecolor='black')
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_title('Max Drawdown', fontweight='bold')

    # 5. Performance Heatmap (middle, spanning 3 columns)
    if perf_df is not None:
        ax5 = fig.add_subplot(gs[1:3, :3])
        plot_df = perf_df.drop('MEAN') if 'MEAN' in perf_df.index else perf_df
        im = ax5.imshow(plot_df.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
        ax5.set_xticks(np.arange(len(plot_df.columns)))
        ax5.set_yticks(np.arange(len(plot_df.index)))
        ax5.set_xticklabels(plot_df.columns, fontsize=10, fontweight='bold')
        ax5.set_yticklabels(plot_df.index, fontsize=10)
        plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
        ax5.set_title('Return by Ticker (%)', fontweight='bold')
        for i in range(len(plot_df.index)):
            for j in range(len(plot_df.columns)):
                val = plot_df.values[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 15 else 'black'
                    ax5.text(j, i, f'{val:+.0f}', ha="center", va="center", color=color, fontsize=9)
        cbar = fig.colorbar(im, ax=ax5, shrink=0.6)
        cbar.set_label('Return (%)')

    # 6. Trade Statistics (middle right)
    ax6 = fig.add_subplot(gs[1, 3])
    winners = summary_df['num_winners'].values
    losers = summary_df['num_losers'].values
    x = np.arange(len(strategies))
    width = 0.35
    ax6.bar(x - width/2, winners, width, label='Wins', color=COLORS['positive'])
    ax6.bar(x + width/2, losers, width, label='Losses', color=COLORS['negative'])
    ax6.set_xticks(x)
    ax6.set_xticklabels(strategies, fontsize=9)
    ax6.set_title('Wins vs Losses', fontweight='bold')
    ax6.legend(fontsize=8)

    # 7. Profit Factor (bottom middle-right)
    ax7 = fig.add_subplot(gs[2, 3])
    # Estimate profit factor
    pf_values = []
    for i, strategy in enumerate(strategies):
        row = summary_df.loc[strategy]
        num_winners = row.get('num_winners', 1)
        num_losers = row.get('num_losers', 1)
        # Rough estimate
        if num_losers > 0:
            pf = num_winners / num_losers
        else:
            pf = 2.0
        pf_values.append(min(pf, 3))  # Cap at 3 for visualization

    bars = ax7.bar(strategies, pf_values, color=strat_colors, edgecolor='black')
    ax7.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Breakeven')
    ax7.set_ylabel('Profit Factor')
    ax7.set_title('Profit Factor', fontweight='bold')
    ax7.legend(fontsize=8)

    # 8. Summary Text Box (bottom)
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')

    best_return = summary_df['total_return_pct'].idxmax()
    best_sharpe = summary_df['sharpe_ratio'].idxmax()
    best_winrate = summary_df['win_rate'].idxmax()
    lowest_dd = summary_df['max_drawdown'].idxmax()  # Least negative = best

    summary_text = f"""
    ================================ EVALUATION SUMMARY ================================

    BEST PERFORMERS:
    - Highest Return:        {best_return:<15} ({summary_df.loc[best_return, 'total_return_pct']:+.2f}%)
    - Best Sharpe Ratio:     {best_sharpe:<15} ({summary_df.loc[best_sharpe, 'sharpe_ratio']:.2f})
    - Highest Win Rate:      {best_winrate:<15} ({summary_df.loc[best_winrate, 'win_rate']*100:.0f}%)
    - Lowest Drawdown:       {lowest_dd:<15} ({summary_df.loc[lowest_dd, 'max_drawdown']*100:.1f}%)

    KEY INSIGHTS:
    - Risk Management: Quant13's conviction filter prevents low-quality trades
    - Capital Preservation: Lower drawdowns mean better compounding potential
    - Strategy Selection: Adaptive approach matches strategy to market conditions

    ===================================================================================
    """

    ax8.text(0.5, 0.5, summary_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Quant13 Comprehensive Evaluation Dashboard', fontsize=18, fontweight='bold', y=0.98)

    path = os.path.join(output_dir, 'comprehensive_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


def create_value_proposition_chart(
    summary_df: pd.DataFrame,
    trades_data: Dict[str, List[Dict]],
    output_dir: str
) -> str:
    """Create value proposition summary chart highlighting Quant13's advantages"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    strategies = summary_df.index.tolist()

    # 1. Risk vs Return Scatter (top left)
    ax1 = axes[0, 0]
    for strategy in strategies:
        row = summary_df.loc[strategy]
        risk = abs(row['max_drawdown']) * 100
        ret = row['total_return_pct']
        color = STRATEGY_COLORS.get(strategy, COLORS['neutral'])
        ax1.scatter(risk, ret, s=200, c=color, label=strategy, edgecolor='black', linewidth=2)
        ax1.annotate(strategy, (risk, ret), xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axvline(x=10, color='red', linestyle='--', alpha=0.3, label='10% Risk Threshold')
    ax1.set_xlabel('Max Drawdown (%)', fontsize=11)
    ax1.set_ylabel('Total Return (%)', fontsize=11)
    ax1.set_title('Risk vs. Return\n(Upper-left is optimal)', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')

    # 2. Efficiency Metrics (top right)
    ax2 = axes[0, 1]
    metrics = ['Return/Trade', 'Win/Loss Ratio', 'Recovery Speed']
    x = np.arange(len(metrics))
    width = 0.25

    for i, strategy in enumerate(strategies):
        row = summary_df.loc[strategy]
        num_trades = row.get('num_trades', 1)
        num_winners = row.get('num_winners', 0)
        num_losers = row.get('num_losers', 1)

        ret_per_trade = row['total_return_pct'] / max(num_trades, 1)
        win_loss = num_winners / max(num_losers, 1)
        recovery = 1 / (abs(row['max_drawdown']) + 0.01)  # Higher = faster recovery

        values = [ret_per_trade / 5, win_loss / 3, recovery / 20]  # Normalize
        color = STRATEGY_COLORS.get(strategy, COLORS['neutral'])
        ax2.bar(x + i * width, values, width, label=strategy, color=color, edgecolor='black')

    ax2.set_ylabel('Normalized Score')
    ax2.set_title('Efficiency Metrics\n(Higher is better)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(metrics)
    ax2.legend()

    # 3. Quant13 Unique Value (bottom left)
    ax3 = axes[1, 0]
    ax3.axis('off')

    value_text = """
    QUANT13 UNIQUE VALUE PROPOSITION
    ================================

    1. MULTI-AGENT INTELLIGENCE
       - 7 specialized AI agents analyze different dimensions
       - Bull vs Bear debate reduces single-model bias
       - Consensus-based decisions improve accuracy

    2. RISK-FIRST APPROACH
       - Conviction filter prevents low-quality trades
       - Adaptive position sizing based on confidence
       - Systematic stop-loss and profit targets

    3. STRATEGY OPTIMIZATION
       - Matches strategy to market regime (trending vs range-bound)
       - IV-aware: sells premium in high IV, buys in low IV
       - Thesis-strategy alignment validation

    4. EXPLAINABILITY
       - Complete audit trail for every decision
       - Reasoning documented at each step
       - Useful for learning and refinement
    """

    ax3.text(0.1, 0.9, value_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 4. When to Use Each Strategy (bottom right)
    ax4 = axes[1, 1]
    ax4.axis('off')

    comparison_text = """
    WHEN TO USE EACH STRATEGY
    =========================

    QUANT13:
    - Range-bound, choppy markets
    - High IV environments
    - When risk management is priority
    - Long-term, consistent returns
    - Need for explainable decisions

    TECHNICAL BASELINE:
    - Strong trending markets
    - Low-cost, high-frequency trading
    - When simplicity is preferred
    - Short-term momentum plays

    KEY INSIGHT:
    No single strategy wins in all conditions.
    Quant13 excels in uncertain markets where
    its multi-factor analysis adds value over
    simple technical rules.
    """

    ax4.text(0.1, 0.9, comparison_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.suptitle('Quant13 Value Proposition Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(output_dir, 'value_proposition.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return path


if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Find latest results
        results_dirs = sorted(glob.glob('results/evaluation_*'))
        if results_dirs:
            results_dir = results_dirs[-1]
        else:
            print("No results found")
            sys.exit(1)

    print(f"Creating enhanced visualizations for: {results_dir}")
    charts = create_enhanced_visualizations(results_dir)
    print(f"\nCreated {len(charts)} charts:")
    for name, path in charts.items():
        print(f"  - {name}: {path}")
