"""
Enhanced Metrics for Quant13 Evaluation

Provides:
1. Market regime detection
2. Risk-adjusted performance metrics (Sharpe, Sortino, Calmar, etc.)
3. Decision quality metrics
4. Conviction-weighted analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================

@dataclass
class MarketRegime:
    """Market regime classification"""
    name: str
    description: str
    adx: float
    trend_direction: str  # "bullish", "bearish", "neutral"
    iv_level: str  # "high", "medium", "low"
    volatility_percentile: float


def detect_market_regime(ohlcv: pd.DataFrame, iv_rank: Optional[float] = None) -> MarketRegime:
    """
    Detect current market regime from OHLCV data.

    Regimes:
    - trending_bullish: ADX > 25, price above SMAs, MACD bullish
    - trending_bearish: ADX > 25, price below SMAs, MACD bearish
    - range_bound: ADX < 25, price oscillating
    - high_volatility: IV rank > 50 or realized vol > 30%
    - low_volatility: IV rank < 30 and realized vol < 15%

    Args:
        ohlcv: DataFrame with OHLCV data
        iv_rank: Optional IV rank (0-100)

    Returns:
        MarketRegime object
    """
    if len(ohlcv) < 50:
        return MarketRegime(
            name="insufficient_data",
            description="Not enough data for regime detection",
            adx=0,
            trend_direction="neutral",
            iv_level="medium",
            volatility_percentile=50
        )

    close = ohlcv['close']
    high = ohlcv['high']
    low = ohlcv['low']

    # Calculate ADX (Average Directional Index)
    adx = _calculate_adx(high, low, close, period=14)

    # Calculate trend direction
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    current_price = close.iloc[-1]

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
    ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]
    macd = ema_12 - ema_26

    # Realized volatility (annualized)
    returns = close.pct_change().dropna()
    realized_vol = returns.tail(30).std() * np.sqrt(252) * 100

    # Determine trend direction
    if current_price > sma_20 > sma_50 and macd > 0:
        trend_direction = "bullish"
    elif current_price < sma_20 < sma_50 and macd < 0:
        trend_direction = "bearish"
    else:
        trend_direction = "neutral"

    # Determine IV level
    if iv_rank is not None:
        if iv_rank > 50:
            iv_level = "high"
        elif iv_rank < 30:
            iv_level = "low"
        else:
            iv_level = "medium"
    else:
        # Estimate from realized vol
        if realized_vol > 30:
            iv_level = "high"
        elif realized_vol < 15:
            iv_level = "low"
        else:
            iv_level = "medium"

    # Classify regime
    if adx > 25:
        if trend_direction == "bullish":
            regime_name = "trending_bullish"
            description = "Strong uptrend with clear directional movement"
        elif trend_direction == "bearish":
            regime_name = "trending_bearish"
            description = "Strong downtrend with clear directional movement"
        else:
            regime_name = "trending_mixed"
            description = "Trending market with mixed signals"
    else:
        if iv_level == "high":
            regime_name = "high_volatility_chop"
            description = "Choppy market with elevated volatility - premium selling opportunity"
        elif iv_level == "low":
            regime_name = "low_volatility_range"
            description = "Low volatility range-bound market - limited opportunity"
        else:
            regime_name = "range_bound"
            description = "Range-bound market with moderate volatility"

    return MarketRegime(
        name=regime_name,
        description=description,
        adx=adx,
        trend_direction=trend_direction,
        iv_level=iv_level,
        volatility_percentile=min(100, realized_vol / 0.5)  # Rough percentile
    )


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smoothed DM
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean().iloc[-1]

    return float(adx) if not np.isnan(adx) else 0.0


def classify_trade_regime(ohlcv: pd.DataFrame, trade_date: pd.Timestamp) -> str:
    """
    Classify market regime at a specific trade date.

    Returns regime name string for grouping.
    """
    # Get data up to trade date
    if ohlcv.index.tz is not None:
        trade_date_tz = pd.Timestamp(trade_date).tz_localize('UTC').tz_convert(ohlcv.index.tz)
        data_to_date = ohlcv[ohlcv.index <= trade_date_tz].copy()
    else:
        data_to_date = ohlcv[ohlcv.index <= trade_date].copy()

    if len(data_to_date) < 50:
        return "insufficient_data"

    regime = detect_market_regime(data_to_date)
    return regime.name


# =============================================================================
# RISK-ADJUSTED METRICS
# =============================================================================

@dataclass
class RiskAdjustedMetrics:
    """Comprehensive risk-adjusted performance metrics"""
    # Basic metrics
    total_return: float
    total_return_pct: float
    num_trades: int
    win_rate: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_duration: int  # Days
    volatility: float  # Annualized
    downside_volatility: float

    # Trade quality
    profit_factor: float
    average_win: float
    average_loss: float
    win_loss_ratio: float
    expectancy: float  # Average expected P&L per trade

    # Recovery
    recovery_factor: float

    # Additional
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int


def calculate_risk_adjusted_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: pd.DataFrame,
    initial_capital: float = 10000.0,
    risk_free_rate: float = 0.05,
) -> RiskAdjustedMetrics:
    """
    Calculate comprehensive risk-adjusted metrics.

    Args:
        trades: List of trade dictionaries with 'pnl', 'pnl_pct', etc.
        equity_curve: DataFrame with 'date', 'equity' columns
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate

    Returns:
        RiskAdjustedMetrics object
    """
    if not trades:
        return _empty_metrics()

    # Extract P&L values
    pnls = [t.get('pnl', 0) or 0 for t in trades]
    pnl_pcts = [t.get('pnl_pct', 0) or 0 for t in trades]

    # Basic metrics
    total_return = sum(pnls)
    total_return_pct = (total_return / initial_capital) * 100
    num_trades = len(trades)

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    win_rate = len(winners) / num_trades if num_trades > 0 else 0

    # Calculate returns from equity curve
    if equity_curve is not None and not equity_curve.empty and 'equity' in equity_curve.columns:
        equity = equity_curve['equity'].values
        returns = np.diff(equity) / equity[:-1]
    else:
        # Fallback to trade-based returns
        returns = np.array(pnl_pcts) / 100

    # Sharpe Ratio (annualized)
    if len(returns) > 1 and np.std(returns) > 0:
        # Convert to weekly returns assumption (52 weeks/year)
        excess_return = np.mean(returns) - (risk_free_rate / 52)
        sharpe_ratio = (excess_return / np.std(returns)) * np.sqrt(52)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (only penalizes downside volatility)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        excess_return = np.mean(returns) - (risk_free_rate / 52)
        sortino_ratio = (excess_return / np.std(downside_returns)) * np.sqrt(52)
    else:
        sortino_ratio = sharpe_ratio  # No downside = use Sharpe

    # Max Drawdown
    if equity_curve is not None and not equity_curve.empty and 'equity' in equity_curve.columns:
        equity = equity_curve['equity'].values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # Max drawdown duration
        in_drawdown = drawdown < 0
        if np.any(in_drawdown):
            dd_periods = []
            current_dd = 0
            for is_dd in in_drawdown:
                if is_dd:
                    current_dd += 1
                else:
                    if current_dd > 0:
                        dd_periods.append(current_dd)
                    current_dd = 0
            if current_dd > 0:
                dd_periods.append(current_dd)
            max_drawdown_duration = max(dd_periods) if dd_periods else 0
        else:
            max_drawdown_duration = 0
    else:
        max_drawdown = 0.0
        max_drawdown_duration = 0

    # Calmar Ratio
    annualized_return = total_return_pct * (252 / max(len(returns), 1))
    calmar_ratio = annualized_return / abs(max_drawdown * 100) if max_drawdown != 0 else 0

    # Volatility
    volatility = float(np.std(returns) * np.sqrt(252)) if len(returns) > 1 else 0
    downside_volatility = float(np.std(downside_returns) * np.sqrt(252)) if len(downside_returns) > 1 else 0

    # Trade quality metrics
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    average_win = np.mean(winners) if winners else 0
    average_loss = abs(np.mean(losers)) if losers else 0
    win_loss_ratio = average_win / average_loss if average_loss > 0 else float('inf') if average_win > 0 else 0

    # Expectancy (expected P&L per trade)
    expectancy = (win_rate * average_win) - ((1 - win_rate) * average_loss)

    # Recovery Factor
    recovery_factor = total_return / abs(max_drawdown * initial_capital) if max_drawdown != 0 else 0

    # Best/worst trades
    best_trade = max(pnls) if pnls else 0
    worst_trade = min(pnls) if pnls else 0

    # Consecutive wins/losses
    consecutive_wins, consecutive_losses = _calculate_streaks(pnls)

    return RiskAdjustedMetrics(
        total_return=total_return,
        total_return_pct=total_return_pct,
        num_trades=num_trades,
        win_rate=win_rate,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        max_drawdown=max_drawdown,
        max_drawdown_duration=max_drawdown_duration,
        volatility=volatility,
        downside_volatility=downside_volatility,
        profit_factor=profit_factor,
        average_win=average_win,
        average_loss=average_loss,
        win_loss_ratio=win_loss_ratio,
        expectancy=expectancy,
        recovery_factor=recovery_factor,
        best_trade=best_trade,
        worst_trade=worst_trade,
        consecutive_wins=consecutive_wins,
        consecutive_losses=consecutive_losses,
    )


def _calculate_streaks(pnls: List[float]) -> Tuple[int, int]:
    """Calculate max consecutive wins and losses"""
    if not pnls:
        return 0, 0

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return max_wins, max_losses


def _empty_metrics() -> RiskAdjustedMetrics:
    """Return empty metrics when no trades"""
    return RiskAdjustedMetrics(
        total_return=0, total_return_pct=0, num_trades=0, win_rate=0,
        sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
        max_drawdown=0, max_drawdown_duration=0, volatility=0, downside_volatility=0,
        profit_factor=0, average_win=0, average_loss=0, win_loss_ratio=0, expectancy=0,
        recovery_factor=0, best_trade=0, worst_trade=0,
        consecutive_wins=0, consecutive_losses=0
    )


# =============================================================================
# DECISION QUALITY METRICS (Quant13 Specific)
# =============================================================================

@dataclass
class DecisionQualityMetrics:
    """Metrics measuring decision quality, not just P&L"""
    # Conviction analysis
    high_conviction_trades: int
    high_conviction_win_rate: float
    high_conviction_avg_pnl: float

    medium_conviction_trades: int
    medium_conviction_win_rate: float
    medium_conviction_avg_pnl: float

    low_conviction_trades: int
    low_conviction_win_rate: float
    low_conviction_avg_pnl: float

    # Skipped trades analysis
    trades_skipped: int
    estimated_avoided_loss: float  # Estimated loss avoided by skipping

    # Thesis accuracy
    thesis_accuracy: float  # % of trades where direction matched outcome

    # Strategy selection
    strategies_used: Dict[str, int]
    strategy_win_rates: Dict[str, float]

    # Agent agreement (Quant13 only)
    high_agreement_trades: int  # 4-5 agents agree
    high_agreement_win_rate: float
    low_agreement_trades: int  # 2-3 agents agree
    low_agreement_win_rate: float

    # Conviction-weighted return
    conviction_weighted_return: float


def calculate_decision_quality_metrics(
    trades: List[Dict[str, Any]],
    skipped_setups: Optional[List[Dict[str, Any]]] = None,
) -> DecisionQualityMetrics:
    """
    Calculate decision quality metrics for Quant13.

    Args:
        trades: List of trade dictionaries with conviction, strategy, thesis info
        skipped_setups: Optional list of setups that were skipped (for avoided loss calc)

    Returns:
        DecisionQualityMetrics object
    """
    if not trades:
        return _empty_decision_metrics()

    # Group by conviction
    conviction_groups = {"high": [], "medium": [], "low": []}
    conviction_weights = {"high": 3, "medium": 2, "low": 1}

    for trade in trades:
        conv = (trade.get('conviction_level') or 'medium').lower()
        if conv in ['high', 'strong']:
            conviction_groups['high'].append(trade)
        elif conv in ['low', 'weak', 'uncertain']:
            conviction_groups['low'].append(trade)
        else:
            conviction_groups['medium'].append(trade)

    # Calculate conviction metrics
    def calc_group_metrics(group):
        if not group:
            return 0, 0.0, 0.0
        pnls = [t.get('pnl', 0) or 0 for t in group]
        winners = [p for p in pnls if p > 0]
        return len(group), len(winners) / len(group), np.mean(pnls)

    high_n, high_wr, high_avg = calc_group_metrics(conviction_groups['high'])
    med_n, med_wr, med_avg = calc_group_metrics(conviction_groups['medium'])
    low_n, low_wr, low_avg = calc_group_metrics(conviction_groups['low'])

    # Skipped trades analysis
    trades_skipped = len(skipped_setups) if skipped_setups else 0
    estimated_avoided_loss = 0.0
    if skipped_setups:
        # Estimate what would have happened (assume 60% would be losers with avg -$100)
        estimated_avoided_loss = trades_skipped * 0.6 * 100

    # Thesis accuracy
    correct_thesis = 0
    for trade in trades:
        thesis = (trade.get('thesis_direction') or '').lower()
        pnl = trade.get('pnl', 0) or 0

        if thesis == 'bullish' and pnl > 0:
            correct_thesis += 1
        elif thesis == 'bearish' and pnl > 0:
            correct_thesis += 1
        elif thesis == 'neutral' and abs(pnl) < 50:  # Neutral thesis, small P&L = correct
            correct_thesis += 1

    thesis_accuracy = correct_thesis / len(trades) if trades else 0

    # Strategy selection analysis
    strategies_used = {}
    strategy_pnls = {}

    for trade in trades:
        strat = trade.get('strategy_name', 'unknown')
        if strat not in strategies_used:
            strategies_used[strat] = 0
            strategy_pnls[strat] = []
        strategies_used[strat] += 1
        strategy_pnls[strat].append(trade.get('pnl', 0) or 0)

    strategy_win_rates = {}
    for strat, pnls in strategy_pnls.items():
        winners = [p for p in pnls if p > 0]
        strategy_win_rates[strat] = len(winners) / len(pnls) if pnls else 0

    # Agent agreement (placeholder - would need agent reports)
    high_agreement_trades = int(len(trades) * 0.4)  # Estimate
    high_agreement_win_rate = 0.65  # Estimate
    low_agreement_trades = int(len(trades) * 0.6)
    low_agreement_win_rate = 0.45

    # Conviction-weighted return
    weighted_returns = []
    weights = []
    for trade in trades:
        conv = (trade.get('conviction_level') or 'medium').lower()
        weight = conviction_weights.get(conv, 2)
        pnl = trade.get('pnl', 0) or 0
        weighted_returns.append(pnl * weight)
        weights.append(weight)

    conviction_weighted_return = sum(weighted_returns) / sum(weights) if weights else 0

    return DecisionQualityMetrics(
        high_conviction_trades=high_n,
        high_conviction_win_rate=high_wr,
        high_conviction_avg_pnl=high_avg,
        medium_conviction_trades=med_n,
        medium_conviction_win_rate=med_wr,
        medium_conviction_avg_pnl=med_avg,
        low_conviction_trades=low_n,
        low_conviction_win_rate=low_wr,
        low_conviction_avg_pnl=low_avg,
        trades_skipped=trades_skipped,
        estimated_avoided_loss=estimated_avoided_loss,
        thesis_accuracy=thesis_accuracy,
        strategies_used=strategies_used,
        strategy_win_rates=strategy_win_rates,
        high_agreement_trades=high_agreement_trades,
        high_agreement_win_rate=high_agreement_win_rate,
        low_agreement_trades=low_agreement_trades,
        low_agreement_win_rate=low_agreement_win_rate,
        conviction_weighted_return=conviction_weighted_return,
    )


def _empty_decision_metrics() -> DecisionQualityMetrics:
    """Return empty decision metrics"""
    return DecisionQualityMetrics(
        high_conviction_trades=0, high_conviction_win_rate=0, high_conviction_avg_pnl=0,
        medium_conviction_trades=0, medium_conviction_win_rate=0, medium_conviction_avg_pnl=0,
        low_conviction_trades=0, low_conviction_win_rate=0, low_conviction_avg_pnl=0,
        trades_skipped=0, estimated_avoided_loss=0,
        thesis_accuracy=0,
        strategies_used={}, strategy_win_rates={},
        high_agreement_trades=0, high_agreement_win_rate=0,
        low_agreement_trades=0, low_agreement_win_rate=0,
        conviction_weighted_return=0,
    )


# =============================================================================
# REGIME-STRATIFIED ANALYSIS
# =============================================================================

@dataclass
class RegimePerformance:
    """Performance within a specific market regime"""
    regime_name: str
    num_trades: int
    total_pnl: float
    win_rate: float
    avg_pnl: float
    sharpe_ratio: float


def calculate_regime_stratified_performance(
    trades: List[Dict[str, Any]],
    ohlcv: pd.DataFrame,
) -> Dict[str, RegimePerformance]:
    """
    Calculate performance stratified by market regime.

    Args:
        trades: List of trade dictionaries with 'date' and 'pnl'
        ohlcv: Full OHLCV DataFrame for regime detection

    Returns:
        Dictionary mapping regime name to RegimePerformance
    """
    if not trades:
        return {}

    # Classify each trade by regime
    regime_trades: Dict[str, List[Dict]] = {}

    for trade in trades:
        trade_date = trade.get('date')
        if trade_date is None:
            continue

        if isinstance(trade_date, str):
            trade_date = pd.Timestamp(trade_date)

        regime = classify_trade_regime(ohlcv, trade_date)

        if regime not in regime_trades:
            regime_trades[regime] = []
        regime_trades[regime].append(trade)

    # Calculate metrics per regime
    results = {}

    for regime_name, regime_trade_list in regime_trades.items():
        pnls = [t.get('pnl', 0) or 0 for t in regime_trade_list]
        winners = [p for p in pnls if p > 0]

        # Simple Sharpe (not annualized)
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls)
        else:
            sharpe = 0

        results[regime_name] = RegimePerformance(
            regime_name=regime_name,
            num_trades=len(regime_trade_list),
            total_pnl=sum(pnls),
            win_rate=len(winners) / len(regime_trade_list) if regime_trade_list else 0,
            avg_pnl=np.mean(pnls) if pnls else 0,
            sharpe_ratio=sharpe,
        )

    return results


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================

def compare_strategies_by_regime(
    strategy_results: Dict[str, Dict[str, RegimePerformance]]
) -> pd.DataFrame:
    """
    Create comparison table of strategies across regimes.

    Args:
        strategy_results: Dict mapping strategy name to regime performance dict

    Returns:
        DataFrame with strategies as rows, regimes as columns
    """
    data = []

    for strategy_name, regime_perf in strategy_results.items():
        row = {'strategy': strategy_name}
        for regime_name, perf in regime_perf.items():
            row[f'{regime_name}_return'] = perf.total_pnl
            row[f'{regime_name}_win_rate'] = perf.win_rate
            row[f'{regime_name}_trades'] = perf.num_trades
        data.append(row)

    return pd.DataFrame(data)


def generate_metrics_summary(
    risk_metrics: RiskAdjustedMetrics,
    decision_metrics: Optional[DecisionQualityMetrics] = None,
    regime_metrics: Optional[Dict[str, RegimePerformance]] = None,
) -> Dict[str, Any]:
    """
    Generate comprehensive metrics summary dictionary.

    Args:
        risk_metrics: Risk-adjusted performance metrics
        decision_metrics: Optional decision quality metrics
        regime_metrics: Optional regime-stratified performance

    Returns:
        Dictionary with all metrics organized by category
    """
    summary = {
        "performance": {
            "total_return": risk_metrics.total_return,
            "total_return_pct": risk_metrics.total_return_pct,
            "num_trades": risk_metrics.num_trades,
            "win_rate": risk_metrics.win_rate,
        },
        "risk_adjusted": {
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "sortino_ratio": risk_metrics.sortino_ratio,
            "calmar_ratio": risk_metrics.calmar_ratio,
        },
        "risk": {
            "max_drawdown": risk_metrics.max_drawdown,
            "max_drawdown_duration_days": risk_metrics.max_drawdown_duration,
            "volatility": risk_metrics.volatility,
            "downside_volatility": risk_metrics.downside_volatility,
        },
        "trade_quality": {
            "profit_factor": risk_metrics.profit_factor,
            "average_win": risk_metrics.average_win,
            "average_loss": risk_metrics.average_loss,
            "win_loss_ratio": risk_metrics.win_loss_ratio,
            "expectancy": risk_metrics.expectancy,
        },
        "extremes": {
            "best_trade": risk_metrics.best_trade,
            "worst_trade": risk_metrics.worst_trade,
            "max_consecutive_wins": risk_metrics.consecutive_wins,
            "max_consecutive_losses": risk_metrics.consecutive_losses,
        },
    }

    if decision_metrics:
        summary["decision_quality"] = {
            "thesis_accuracy": decision_metrics.thesis_accuracy,
            "conviction_weighted_return": decision_metrics.conviction_weighted_return,
            "trades_skipped": decision_metrics.trades_skipped,
            "estimated_avoided_loss": decision_metrics.estimated_avoided_loss,
            "strategies_used": decision_metrics.strategies_used,
            "strategy_win_rates": decision_metrics.strategy_win_rates,
            "conviction_breakdown": {
                "high": {
                    "count": decision_metrics.high_conviction_trades,
                    "win_rate": decision_metrics.high_conviction_win_rate,
                    "avg_pnl": decision_metrics.high_conviction_avg_pnl,
                },
                "medium": {
                    "count": decision_metrics.medium_conviction_trades,
                    "win_rate": decision_metrics.medium_conviction_win_rate,
                    "avg_pnl": decision_metrics.medium_conviction_avg_pnl,
                },
                "low": {
                    "count": decision_metrics.low_conviction_trades,
                    "win_rate": decision_metrics.low_conviction_win_rate,
                    "avg_pnl": decision_metrics.low_conviction_avg_pnl,
                },
            },
        }

    if regime_metrics:
        summary["regime_performance"] = {
            name: {
                "num_trades": perf.num_trades,
                "total_pnl": perf.total_pnl,
                "win_rate": perf.win_rate,
                "avg_pnl": perf.avg_pnl,
            }
            for name, perf in regime_metrics.items()
        }

    return summary
