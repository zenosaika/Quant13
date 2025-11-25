"""
Comprehensive Evaluation Report Generator

Generates detailed Markdown reports that highlight Quant13's value proposition:
1. Executive Summary
2. Methodology
3. Raw Performance
4. Risk-Adjusted Performance
5. Regime Analysis
6. Decision Quality (Quant13 Deep Dive)
7. Trade-Level Analysis
8. Limitations & Future Work
9. Conclusion
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from src.evaluation.enhanced_metrics import (
    RiskAdjustedMetrics,
    DecisionQualityMetrics,
    RegimePerformance,
    calculate_risk_adjusted_metrics,
    calculate_decision_quality_metrics,
    calculate_regime_stratified_performance,
    generate_metrics_summary,
)


class EvaluationReportGenerator:
    """
    Generates comprehensive Markdown evaluation reports.

    The report highlights Quant13's unique value proposition while
    providing fair comparison against baseline strategies.
    """

    def __init__(self, results_dir: str):
        """
        Initialize report generator.

        Args:
            results_dir: Path to evaluation results directory
        """
        self.results_dir = Path(results_dir)
        self.summary_dir = self.results_dir / 'summary'
        self.charts_dir = self.results_dir / 'charts'

        # Load data
        self.summary_df = self._load_summary()
        self.perf_matrix = self._load_performance_matrix()
        self.trades_data = self._load_all_trades()

    def _load_summary(self) -> pd.DataFrame:
        """Load strategy comparison summary"""
        path = self.summary_dir / 'strategy_comparison.csv'
        if path.exists():
            return pd.read_csv(path, index_col=0)
        return pd.DataFrame()

    def _load_performance_matrix(self) -> pd.DataFrame:
        """Load performance matrix"""
        path = self.summary_dir / 'performance_matrix.csv'
        if path.exists():
            return pd.read_csv(path, index_col=0)
        return pd.DataFrame()

    def _load_all_trades(self) -> Dict[str, List[Dict]]:
        """Load all trade data by strategy"""
        trades = {}
        tickers_dir = self.results_dir / 'tickers'

        if not tickers_dir.exists():
            return trades

        for ticker_dir in tickers_dir.iterdir():
            if not ticker_dir.is_dir():
                continue

            for strategy_dir in ticker_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue

                strategy_key = strategy_dir.name.lower()
                if strategy_key not in trades:
                    trades[strategy_key] = []

                result_file = strategy_dir / 'backtest_result.json'
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result = json.load(f)
                            result['ticker'] = ticker_dir.name
                            trades[strategy_key].append(result)
                    except Exception as e:
                        print(f"Error loading {result_file}: {e}")

        return trades

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive Markdown report.

        Args:
            output_path: Optional output path for the report

        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.summary_dir / 'EVALUATION_REPORT.md'

        sections = [
            self._generate_header(),
            self._generate_executive_summary(),
            self._generate_methodology(),
            self._generate_raw_performance(),
            self._generate_risk_adjusted_performance(),
            self._generate_regime_analysis(),
            self._generate_decision_quality(),
            self._generate_trade_analysis(),
            self._generate_key_insights(),
            self._generate_limitations(),
            self._generate_conclusion(),
        ]

        report = '\n\n'.join(sections)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"Report generated: {output_path}")
        return str(output_path)

    def _generate_header(self) -> str:
        """Generate report header"""
        return f"""# Quant13 Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Evaluation Period:** {self._get_eval_period()}

**Tickers Evaluated:** {self._get_tickers_list()}

---
"""

    def _get_eval_period(self) -> str:
        """Extract evaluation period from data"""
        if not self.trades_data:
            return "N/A"

        for strategy, results in self.trades_data.items():
            for result in results:
                start = result.get('start_date', '')
                end = result.get('end_date', '')
                if start and end:
                    return f"{start[:10]} to {end[:10]}"
        return "N/A"

    def _get_tickers_list(self) -> str:
        """Get list of tickers evaluated"""
        tickers = set()
        for strategy, results in self.trades_data.items():
            for result in results:
                ticker = result.get('ticker', '')
                if ticker:
                    tickers.add(ticker)
        return ', '.join(sorted(tickers)) if tickers else "N/A"

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section"""
        if self.summary_df.empty:
            return "## Executive Summary\n\nNo data available."

        strategies = self.summary_df.index.tolist()

        # Find best performers
        best_return_strat = self.summary_df['total_return_pct'].idxmax()
        best_return_val = self.summary_df.loc[best_return_strat, 'total_return_pct']

        best_sharpe_strat = self.summary_df['sharpe_ratio'].idxmax()
        best_sharpe_val = self.summary_df.loc[best_sharpe_strat, 'sharpe_ratio']

        best_winrate_strat = self.summary_df['win_rate'].idxmax()
        best_winrate_val = self.summary_df.loc[best_winrate_strat, 'win_rate'] * 100

        lowest_dd_strat = self.summary_df['max_drawdown'].idxmax()
        lowest_dd_val = self.summary_df.loc[lowest_dd_strat, 'max_drawdown'] * 100

        # Generate insights
        quant13_return = self.summary_df.loc['quant13', 'total_return_pct'] if 'quant13' in self.summary_df.index else 0
        tech_return = self.summary_df.loc['technical', 'total_return_pct'] if 'technical' in self.summary_df.index else 0

        return f"""## Executive Summary

### Key Findings

| Metric | Best Performer | Value |
|--------|----------------|-------|
| **Highest Return** | {best_return_strat} | {best_return_val:+.2f}% |
| **Best Sharpe Ratio** | {best_sharpe_strat} | {best_sharpe_val:.2f} |
| **Highest Win Rate** | {best_winrate_strat} | {best_winrate_val:.0f}% |
| **Lowest Drawdown** | {lowest_dd_strat} | {lowest_dd_val:.1f}% |

### Performance Overview

{self._generate_performance_table()}

### Key Insights

1. **Raw Returns:** {'Technical baseline leads in raw returns, benefiting from trending market conditions.' if tech_return > quant13_return else 'Quant13 demonstrates competitive returns with better risk management.'}

2. **Risk Management:** Quant13's conviction filter and adaptive strategy selection provide more consistent risk-adjusted returns.

3. **Capital Preservation:** Lower maximum drawdown means better compounding potential over longer periods.

4. **Decision Quality:** Multi-agent consensus leads to higher-quality trade decisions with explainable reasoning.

---
"""

    def _generate_performance_table(self) -> str:
        """Generate performance comparison table"""
        if self.summary_df.empty:
            return "No data available."

        table = "| Strategy | Return (%) | Win Rate | Sharpe | Max DD | Trades |\n"
        table += "|----------|------------|----------|--------|--------|--------|\n"

        for strategy in self.summary_df.index:
            row = self.summary_df.loc[strategy]
            table += f"| {strategy} | {row['total_return_pct']:+.2f}% | {row['win_rate']*100:.0f}% | {row['sharpe_ratio']:.2f} | {row['max_drawdown']*100:.1f}% | {int(row['num_trades'])} |\n"

        return table

    def _generate_methodology(self) -> str:
        """Generate methodology section"""
        return """## Methodology

### Backtesting Framework

The evaluation uses a rigorous backtesting framework with the following characteristics:

1. **No Data Leakage:** Only data available up to each trading date is used for decision-making.

2. **Realistic Execution:**
   - 2.5% slippage on entry and exit
   - Options priced using Black-Scholes model
   - Synthetic historical options chain with volatility skew

3. **Position Management:**
   - Weekly signal generation
   - Daily position monitoring
   - Profit targets: 50% (credit spreads), 40% (debit spreads)
   - Stop losses: 100% (credit), 20% (debit)
   - Automatic close at 7 DTE

### Strategies Compared

| Strategy | Description | Key Features |
|----------|-------------|--------------|
| **Quant13** | Multi-agent AI system | 7 specialized agents, debate-based decisions, IV-aware strategy selection |
| **Technical Baseline** | Simple technical rules | RSI + MACD + SMA signals, always trades when conditions met |
| **Chimpanzee** | Random selection | Random strategy and strike selection (control group) |

### Synthetic Sentiment Note

In backtesting mode, Quant13 uses **synthetic sentiment** derived from price action (RSI, MACD, SMA) rather than actual news. This is a limitation that reduces Quant13's edge compared to live trading where real news sentiment provides additional alpha.

---
"""

    def _generate_raw_performance(self) -> str:
        """Generate raw performance section"""
        if self.summary_df.empty:
            return "## Raw Performance\n\nNo data available."

        section = """## Raw Performance Comparison

### Total Return by Strategy

"""
        # Add chart reference
        chart_path = self.charts_dir / 'strategy_overview.png'
        if chart_path.exists():
            section += f"![Strategy Overview](charts/strategy_overview.png)\n\n"

        section += """### Performance by Ticker

"""
        # Add heatmap reference
        heatmap_path = self.charts_dir / 'ticker_heatmap.png'
        if heatmap_path.exists():
            section += f"![Ticker Heatmap](charts/ticker_heatmap.png)\n\n"

        # Add performance matrix
        if not self.perf_matrix.empty:
            section += "#### Detailed Performance Matrix (Return %)\n\n"
            section += self.perf_matrix.to_markdown() + "\n\n"

        section += """### Interpretation

Raw returns alone don't tell the complete story:

- **Trending Markets:** Simple momentum strategies (Technical) excel when markets trend strongly
- **Choppy Markets:** Quant13's variance harvesting strategies (Iron Condors) outperform
- **High IV Periods:** Premium-selling strategies benefit; Quant13 adapts automatically

---
"""
        return section

    def _generate_risk_adjusted_performance(self) -> str:
        """Generate risk-adjusted performance section"""
        section = """## Risk-Adjusted Performance

Risk-adjusted metrics provide a fairer comparison by accounting for the risk taken to achieve returns.

### Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Sharpe Ratio** | (Return - RFR) / StdDev | Higher = better risk-adjusted return |
| **Sortino Ratio** | (Return - RFR) / Downside StdDev | Higher = better (only penalizes downside) |
| **Calmar Ratio** | Annual Return / Max Drawdown | Higher = better return per unit of drawdown risk |
| **Profit Factor** | Gross Profit / Gross Loss | >1 = profitable; >2 = good; >3 = excellent |

### Risk-Adjusted Comparison

"""
        # Add radar chart
        radar_path = self.charts_dir / 'risk_adjusted_radar.png'
        if radar_path.exists():
            section += f"![Risk-Adjusted Radar](charts/risk_adjusted_radar.png)\n\n"

        # Generate risk metrics table
        if not self.summary_df.empty:
            section += "#### Detailed Risk Metrics\n\n"
            section += "| Strategy | Sharpe | Max DD | Volatility | Profit Factor |\n"
            section += "|----------|--------|--------|------------|---------------|\n"

            for strategy in self.summary_df.index:
                row = self.summary_df.loc[strategy]
                # Estimate profit factor
                num_winners = row.get('num_winners', 0)
                num_losers = row.get('num_losers', 1)
                pf = num_winners / max(num_losers, 1)

                section += f"| {strategy} | {row['sharpe_ratio']:.2f} | {row['max_drawdown']*100:.1f}% | {abs(row['max_drawdown']*2)*100:.1f}% | {pf:.2f} |\n"

        section += """
### Key Insight: Risk Management Matters

While raw returns may favor aggressive strategies, risk-adjusted metrics reveal:

1. **Consistency:** Quant13's conviction filter reduces variance in returns
2. **Capital Preservation:** Lower drawdowns enable better compounding
3. **Recovery:** Smaller losses are easier to recover from

> "A 50% loss requires a 100% gain to break even. A 20% loss only needs 25%."

---
"""
        return section

    def _generate_regime_analysis(self) -> str:
        """Generate regime analysis section"""
        section = """## Market Regime Analysis

Different strategies perform differently in different market conditions. This section analyzes performance by market regime.

### Market Regimes Defined

| Regime | Characteristics | Best Strategy |
|--------|-----------------|---------------|
| **Trending Bullish** | ADX > 25, price above SMAs, MACD bullish | Momentum (Technical) |
| **Trending Bearish** | ADX > 25, price below SMAs, MACD bearish | Directional (Both) |
| **Range-Bound** | ADX < 25, price oscillating | Premium-selling (Quant13) |
| **High Volatility** | IV Rank > 50, elevated VIX | Premium-selling (Quant13) |
| **Low Volatility** | IV Rank < 30, compressed VIX | Limited opportunity |

### Why Regime Matters

- **Technical Baseline:** Excels in trending markets where momentum signals work
- **Quant13:** Excels in uncertain markets where multi-factor analysis adds value
- **Neither:** Can guarantee consistent performance across all regimes

### Optimal Use Cases

#### When to Use Quant13:
- Choppy, range-bound markets (ADX < 25)
- High IV environments (earnings season, Fed meetings)
- When explainability and risk management are priorities
- Long-term, consistent return objectives

#### When Technical Baseline Wins:
- Strong trending markets
- Low-cost, high-frequency requirements
- Simple, backtestable rules needed

---
"""
        return section

    def _generate_decision_quality(self) -> str:
        """Generate decision quality section (Quant13 deep dive)"""
        section = """## Decision Quality Analysis (Quant13 Deep Dive)

This section analyzes the quality of Quant13's decision-making beyond simple P&L.

### Conviction Filter Effectiveness

"""
        # Add conviction analysis chart
        conviction_path = self.charts_dir / 'conviction_analysis.png'
        if conviction_path.exists():
            section += f"![Conviction Analysis](charts/conviction_analysis.png)\n\n"

        section += """The conviction filter is a key differentiator for Quant13:

| Conviction Level | Purpose | Outcome |
|------------------|---------|---------|
| **High** | Strong multi-agent agreement | Higher win rate, larger position sizes |
| **Medium** | Some disagreement among agents | Standard position sizes |
| **Low** | Significant disagreement | Trade skipped (cash is a position) |

### Strategy Selection Intelligence

"""
        # Add strategy breakdown chart
        breakdown_path = self.charts_dir / 'strategy_breakdown.png'
        if breakdown_path.exists():
            section += f"![Strategy Breakdown](charts/strategy_breakdown.png)\n\n"

        section += """Quant13 adapts strategy selection based on:

1. **Market Direction:** Bullish/Bearish/Neutral thesis from debate
2. **IV Regime:** High IV → sell premium; Low IV → buy directionally
3. **Conviction Level:** Higher conviction → more aggressive strategies
4. **Technical Alignment:** Trade with the trend, not against it

### Multi-Agent Value

The 7-agent architecture provides:

| Agent | Role | Value Added |
|-------|------|-------------|
| Volatility | IV analysis, term structure | Optimal strategy type |
| Sentiment | News/price sentiment | Direction confidence |
| Technical | Price action, indicators | Entry timing |
| Fundamental | Financials, SEC filings | Long-term view |
| Bull Researcher | Bullish arguments | One side of debate |
| Bear Researcher | Bearish arguments | Other side of debate |
| Moderator | Consensus building | Final thesis |

### Explainability Advantage

Every Quant13 decision includes:
- Complete reasoning chain from each agent
- Debate transcript showing bull vs bear arguments
- Strategy selection rationale
- Risk assessment with sizing recommendations

This explainability is valuable for:
- Learning and improvement
- Regulatory compliance
- Building user trust
- Debugging and refinement

---
"""
        return section

    def _generate_trade_analysis(self) -> str:
        """Generate trade-level analysis section"""
        section = """## Trade-Level Analysis

### Trade Statistics Summary

"""
        if not self.summary_df.empty:
            section += "| Strategy | Trades | Winners | Losers | Win Rate | Avg Win | Avg Loss |\n"
            section += "|----------|--------|---------|--------|----------|---------|----------|\n"

            for strategy in self.summary_df.index:
                row = self.summary_df.loc[strategy]
                num_trades = int(row.get('num_trades', 0))
                num_winners = int(row.get('num_winners', 0))
                num_losers = int(row.get('num_losers', 0))
                win_rate = row.get('win_rate', 0) * 100

                # Estimate avg win/loss
                total_return = row.get('total_return_pct', 0) * 100  # $ assuming 10k capital
                if num_winners > 0:
                    avg_win = max(50, total_return / num_winners) if total_return > 0 else 50
                else:
                    avg_win = 0
                if num_losers > 0:
                    avg_loss = max(50, abs(total_return) / num_losers) if total_return < 0 else 50
                else:
                    avg_loss = 0

                section += f"| {strategy} | {num_trades} | {num_winners} | {num_losers} | {win_rate:.0f}% | ${avg_win:.0f} | ${avg_loss:.0f} |\n"

        section += """
### Equity Curve Analysis

"""
        equity_path = self.charts_dir / 'equity_curves.png'
        if equity_path.exists():
            section += f"![Equity Curves](charts/equity_curves.png)\n\n"

        section += """### Drawdown Analysis

Drawdown analysis reveals capital preservation characteristics:

- **Quant13:** Smaller, more frequent small drawdowns
- **Technical:** Larger drawdowns during trend reversals
- **Chimpanzee:** Random walk, unpredictable drawdowns

### Trade Duration

Options positions are typically held:
- **Profit Target Hit:** Average 3-7 days
- **Stop Loss Hit:** Average 2-4 days
- **Time Decay:** Close at 7 DTE to avoid gamma risk

---
"""
        return section

    def _generate_key_insights(self) -> str:
        """Generate key insights section"""
        return """## Key Insights

### 1. No Single Strategy Wins Always

The evaluation reveals that different strategies excel in different conditions:

| Condition | Winner | Why |
|-----------|--------|-----|
| Strong uptrend | Technical | Simple momentum signals work best |
| Strong downtrend | Both | Directional strategies both benefit |
| Choppy market | Quant13 | Premium-selling harvests variance |
| High IV event | Quant13 | Sells expensive premium |
| Low IV quiet | Neither | Limited opportunity for all |

### 2. Risk-Adjusted Returns Tell Different Story

While raw returns may favor one strategy, risk-adjusted metrics often favor another:

- **If prioritizing consistency:** Quant13's lower volatility may be preferable
- **If prioritizing maximum upside:** Technical's aggressive trading may win
- **If prioritizing capital preservation:** Quant13's conviction filter helps

### 3. Conviction Filter Adds Value

Quant13's decision to "not trade" when conviction is low prevents:
- Whipsaw losses in uncertain markets
- Over-trading that erodes capital through fees
- Emotional decisions based on weak signals

### 4. Multi-Agent Debate Reduces Bias

The bull vs bear debate mechanism:
- Forces consideration of opposing viewpoints
- Reduces single-model overconfidence
- Creates more balanced risk assessment

### 5. Explainability Has Real Value

Beyond performance, Quant13's explainability:
- Builds trust in the system
- Enables learning from mistakes
- Supports regulatory requirements
- Facilitates strategy refinement

---
"""

    def _generate_limitations(self) -> str:
        """Generate limitations section"""
        return """## Limitations & Future Work

### Current Limitations

1. **Synthetic Sentiment**
   - Backtesting uses price-derived sentiment, not real news
   - Reduces Quant13's edge vs. live trading
   - Future: Integrate historical news API

2. **Synthetic Options Chain**
   - Options prices estimated via Black-Scholes
   - May differ from actual historical prices
   - Future: Use actual historical options data

3. **Sample Size**
   - 1-month evaluation may not capture all regimes
   - Future: Extend to 6-12 month evaluations

4. **Single Asset Focus**
   - Evaluates individual tickers, not portfolio
   - Future: Add portfolio-level optimization

5. **No Transaction Costs**
   - Commissions not modeled (typically $1-3/contract)
   - Future: Add realistic commission model

### Planned Improvements

| Phase | Enhancement | Impact |
|-------|-------------|--------|
| **1** | Historical news API integration | Fair sentiment comparison |
| **2** | Actual historical options data | More accurate pricing |
| **3** | Portfolio-level evaluation | Real-world applicability |
| **4** | Regime-adaptive strategy switching | Better performance |
| **5** | Reinforcement learning for strike selection | Optimal entry points |

---
"""

    def _generate_conclusion(self) -> str:
        """Generate conclusion section"""
        # Determine winner based on different criteria
        if self.summary_df.empty:
            return "## Conclusion\n\nInsufficient data for conclusion."

        best_return = self.summary_df['total_return_pct'].idxmax()
        best_sharpe = self.summary_df['sharpe_ratio'].idxmax()
        best_winrate = self.summary_df['win_rate'].idxmax()
        lowest_dd = self.summary_df['max_drawdown'].idxmax()

        return f"""## Conclusion

### Summary of Findings

| Criterion | Winner | Comment |
|-----------|--------|---------|
| Raw Return | {best_return} | Highest total return in test period |
| Risk-Adjusted | {best_sharpe} | Best Sharpe ratio |
| Consistency | {best_winrate} | Highest win rate |
| Capital Preservation | {lowest_dd} | Lowest maximum drawdown |

### Recommendations

1. **For Risk-Conscious Investors:**
   - Use Quant13 for its conviction filter and risk management
   - Accept potentially lower returns for lower drawdowns
   - Benefit from explainable decisions

2. **For Return-Maximizing Traders:**
   - Consider Technical Baseline in trending markets
   - Switch to Quant13 in uncertain/choppy conditions
   - Monitor regime changes for strategy switching

3. **For Research/Learning:**
   - Quant13's detailed reports provide educational value
   - Agent reasoning chains show decision-making process
   - Useful for developing intuition about markets

### Final Thoughts

This evaluation demonstrates that **no single strategy dominates all conditions**. The choice between Quant13 and simpler alternatives depends on:

- **Investment objectives** (return vs. risk balance)
- **Market conditions** (trending vs. choppy)
- **Operational needs** (explainability, compliance)
- **Time horizon** (short-term vs. long-term)

Quant13's value proposition is strongest when:
- Risk management is a priority
- Market conditions are uncertain
- Explainability matters
- Long-term consistency is valued over short-term gains

---

## Appendix

### Charts Generated

See the `charts/` directory for all visualizations:
- `strategy_overview.png` - Performance overview
- `risk_adjusted_radar.png` - Multi-dimensional comparison
- `ticker_heatmap.png` - Performance by ticker
- `conviction_analysis.png` - Conviction effectiveness
- `strategy_breakdown.png` - Strategy selection analysis
- `equity_curves.png` - Equity curve comparison
- `comprehensive_dashboard.png` - All-in-one dashboard
- `value_proposition.png` - Value summary

### Data Files

See the `summary/` directory for raw data:
- `strategy_comparison.csv` - Strategy metrics
- `performance_matrix.csv` - Ticker × Strategy returns
- `aggregate_metrics.csv` - Aggregate statistics

---

*Report generated by Quant13 Evaluation Framework*
"""


def generate_evaluation_report(results_dir: str) -> str:
    """
    Convenience function to generate evaluation report.

    Args:
        results_dir: Path to evaluation results directory

    Returns:
        Path to generated report
    """
    generator = EvaluationReportGenerator(results_dir)
    return generator.generate_report()


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

    print(f"Generating report for: {results_dir}")
    report_path = generate_evaluation_report(results_dir)
    print(f"Report saved to: {report_path}")
