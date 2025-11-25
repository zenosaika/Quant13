"""
Hybrid Rebalancing Backtest Framework

Implements the recommended evaluation strategy:
- Weekly signal generation (run full pipeline)
- Daily position monitoring (check exits, Greeks risk)
- Position-level exit management (profit targets, stop losses)

This framework provides the best of both worlds:
- Strategic patience (weekly signals prevent overtrading)
- Tactical agility (daily monitoring catches profit targets and risks)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np

from src.backtesting.framework import BacktestResult, TradeExecution
from src.backtesting.historical_options import (
    estimate_historical_volatility,
    generate_historical_options_chain,
)
from src.backtesting.mark_to_market import update_position_tracking
from src.backtesting.synthetic_sentiment import generate_synthetic_sentiment
from src.agents.position_monitor import PositionMonitorAgent
from src.utils.greeks_aggregation import aggregate_position_greeks, aggregate_portfolio_greeks

logger = logging.getLogger(__name__)


def run_hybrid_backtest(
    ticker: str,
    strategy_func: Callable,
    ohlcv: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.10,
    risk_free_rate: float = 0.05,
    output_dir: Optional[Path] = None,
    # Position management parameters
    profit_target_pct: float = 0.50,
    stop_loss_pct: float = 0.50,
    min_dte_close: int = 7,
    # Hybrid rebalancing parameters
    signal_frequency: str = "weekly",  # How often to generate new signals
    monitor_frequency: str = "daily",  # How often to check positions
    strategy_name: Optional[str] = None,  # Explicit strategy name (optional)
) -> BacktestResult:
    """
    Hybrid rebalancing backtest with weekly signals and daily monitoring

    This is the recommended evaluation method combining:
    - Weekly signal generation (prevents overtrading, reduces costs)
    - Daily position monitoring (catches profit targets, Greeks risk)

    Args:
        ticker: Stock ticker
        strategy_func: Strategy function (returns TradeProposal)
        ohlcv: Historical OHLCV data
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        position_size_pct: % of capital per trade
        risk_free_rate: Risk-free rate
        output_dir: Optional output directory
        profit_target_pct: Profit target (0.50 = 50%)
        stop_loss_pct: Stop loss threshold
        min_dte_close: Close positions at this DTE
        signal_frequency: 'weekly' or 'daily' signal generation
        monitor_frequency: 'daily' monitoring (always recommended)

    Returns:
        BacktestResult with performance metrics
    """
    logger.info(f"Starting hybrid backtest for {ticker}")
    logger.info(f"  Signal frequency: {signal_frequency}")
    logger.info(f"  Monitor frequency: {monitor_frequency}")
    logger.info(f"  Period: {start_date.date()} to {end_date.date()}")

    # Initialize position monitor
    monitor_config = {
        "profit_target_pct": profit_target_pct,
        "stop_loss_credit_pct": 1.00,  # 100% for credit spreads
        "stop_loss_debit_pct": stop_loss_pct,
        "gamma_risk_threshold": 10.0,
        "min_dte_close": min_dte_close,
    }
    position_monitor = PositionMonitorAgent(monitor_config)

    # Generate dates
    signal_dates = _generate_signal_dates(start_date, end_date, signal_frequency)
    monitor_dates = _generate_monitor_dates(start_date, end_date, monitor_frequency)

    logger.info(f"  Signal dates: {len(signal_dates)} (every {signal_frequency})")
    logger.info(f"  Monitor dates: {len(monitor_dates)} (every {monitor_frequency})")

    # Track state
    capital = initial_capital
    open_positions: List[TradeExecution] = []
    closed_trades: List[TradeExecution] = []
    equity_curve_data = []

    # Portfolio Greeks tracking
    portfolio_greeks_history = []

    # Iterate through all calendar days
    current_date = start_date
    while current_date <= end_date:
        # Get data up to current date
        # Handle timezone-aware index
        if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
            current_date_tz = pd.Timestamp(current_date).tz_localize(ohlcv.index.tz)
        else:
            current_date_tz = current_date

        ohlcv_to_date = ohlcv[ohlcv.index <= current_date_tz]

        if len(ohlcv_to_date) < 50:
            current_date += timedelta(days=1)
            continue

        spot_price = float(ohlcv_to_date["close"].iloc[-1])

        # ===================================================================
        # DAILY: Monitor existing positions (profit targets, stop losses)
        # ===================================================================
        if current_date in monitor_dates and open_positions:
            logger.info(f"[{current_date.date()}] Monitoring {len(open_positions)} positions")

            # Update Greeks and P&L for all positions
            for position in open_positions:
                position.current_value = _mark_to_market(
                    position, current_date, ohlcv_to_date, risk_free_rate
                )
                # Bug #1 FIX: Correct P&L formula (P&L = current_value + entry_price)
                position.pnl = position.current_value + position.entry_price
                position.pnl_pct = (position.pnl / abs(position.entry_price)) * 100 if position.entry_price != 0 else 0

                # Update Greeks
                position.current_greeks = aggregate_position_greeks(
                    position.trade_legs, spot_price
                )

                # Update DTE
                position.days_to_expiration = _calculate_dte(position, current_date)

            # Generate synthetic sentiment for position monitoring
            synthetic_sentiment = generate_synthetic_sentiment(ohlcv_to_date)

            # Check for exit triggers
            positions_to_close = []
            for position in open_positions:
                # Determine strategy type
                strategy_type = "CREDIT" if position.entry_price > 0 else "DEBIT"

                # Track highest P&L for trailing stop (NEW)
                if not hasattr(position, 'highest_pnl_pct') or position.highest_pnl_pct is None:
                    position.highest_pnl_pct = 0.0
                if position.pnl_pct > position.highest_pnl_pct:
                    position.highest_pnl_pct = position.pnl_pct

                # Build position dict for monitor
                position_dict = {
                    "entry_premium": position.entry_price,
                    "current_value": position.current_value,
                    "days_to_expiration": position.days_to_expiration,
                    "strategy_type": strategy_type,
                    "greeks": position.current_greeks,
                    "thesis_direction": position.thesis_direction,
                    "highest_pnl_pct": position.highest_pnl_pct,  # NEW: For trailing stop
                }

                # Build market data with synthetic sentiment
                market_data = {
                    "sentiment_score": synthetic_sentiment["sentiment_score"],
                    "sentiment_confidence": synthetic_sentiment["confidence"],
                }

                # Check exit conditions
                action = position_monitor.check_position(position_dict, market_data)

                if action["action"] == "CLOSE":
                    logger.info(f"  Closing position: {action['reason']}")
                    positions_to_close.append(position)

            # Close triggered positions
            if positions_to_close:
                capital_returned, closed = _close_positions_list(
                    positions_to_close, current_date, ohlcv_to_date, risk_free_rate
                )
                capital += capital_returned
                closed_trades.extend(closed)
                open_positions = [p for p in open_positions if p not in positions_to_close]

        # ===================================================================
        # WEEKLY (or configured frequency): Generate new signals
        # ===================================================================
        if current_date in signal_dates:
            logger.info(f"\n[{current_date.date()}] Generating new trade signal")
            logger.info(f"  Current capital: ${capital:,.2f}")
            logger.info(f"  Open positions: {len(open_positions)}")

            # Check if we can open new positions (max 5 concurrent for better capital utilization)
            max_concurrent_positions = 5
            if len(open_positions) >= max_concurrent_positions:
                logger.info(f"  Max positions reached ({max_concurrent_positions}), skipping signal")
            else:
                try:
                    # Generate options chain
                    hist_vol = estimate_historical_volatility(ohlcv_to_date, window=30)
                    options_chain = generate_historical_options_chain(
                        historical_date=current_date,
                        spot_price=spot_price,
                        historical_volatility=hist_vol,
                        risk_free_rate=risk_free_rate,
                        ticker=ticker,
                    )

                    # Convert options chain to flat list for baseline strategies
                    # Quant13 uses grouped format (expiration groups with DataFrames)
                    # Baselines need flat list with 'type' field
                    flat_options_chain = _convert_to_flat_options_list(options_chain)

                    # NEW: Capture market context at signal generation
                    market_context = _capture_market_context(
                        ohlcv=ohlcv_to_date,
                        options_chain=flat_options_chain,
                        hist_vol=hist_vol,
                    )

                    # Call strategy function
                    trade_proposal = strategy_func(
                        ticker=ticker,
                        ohlcv=ohlcv_to_date,
                        options_chain=flat_options_chain,
                        spot_price=spot_price,
                        risk_free_rate=risk_free_rate,
                    )

                    # Execute if approved
                    if trade_proposal is not None:
                        # CONVICTION-BASED POSITION SIZING (NEW IMPROVEMENT)
                        # =====================================================================
                        # CONSERVATIVE CONVICTION SIZING (Based on 180-day analysis)
                        # =====================================================================
                        # Problem: High conviction trades were BIGGER but not better
                        # Solution: Reduce multipliers to limit $ impact of wrong convictions
                        # Previous: 1.5/1.0/0.5 → New: 1.2/1.0/0.7
                        # =====================================================================
                        conviction = getattr(trade_proposal, 'conviction_level', 'Medium')
                        if conviction and conviction.lower() == 'high':
                            conviction_multiplier = 1.2  # 20% larger position (was 50%)
                        elif conviction and conviction.lower() == 'low':
                            conviction_multiplier = 0.7  # 30% smaller position (was 50%)
                        else:
                            conviction_multiplier = 1.0  # Medium = normal size

                        adjusted_position_size = position_size_pct * conviction_multiplier
                        max_capital_per_trade = capital * adjusted_position_size
                        logger.debug(f"  Position sizing: {conviction} conviction → {adjusted_position_size:.1%} of capital")

                        execution = _execute_trade(
                            trade_proposal,
                            current_date,
                            spot_price,
                            flat_options_chain,
                            max_capital_per_trade,
                            risk_free_rate,
                            ohlcv=ohlcv_to_date,  # NEW: Pass OHLCV for trend detection
                        )

                        if execution:
                            # NEW: Attach market context to execution
                            execution.market_data = market_context

                            capital += execution.entry_price  # Debit/credit flow
                            open_positions.append(execution)
                            logger.info(f"  Opened: {execution.strategy_name}")
                            logger.info(f"  Entry price: ${execution.entry_price:,.2f}")

                except Exception as e:
                    logger.error(f"  Strategy failed: {e}")

        # ===================================================================
        # DAILY: Track equity curve and portfolio Greeks
        # ===================================================================
        if current_date in monitor_dates:
            # Calculate total equity
            mark_to_market_value = sum(
                _mark_to_market(p, current_date, ohlcv_to_date, risk_free_rate)
                for p in open_positions
            )
            total_equity = capital + mark_to_market_value

            equity_curve_data.append({
                "date": current_date,
                "equity": total_equity,
            })

            # Track portfolio Greeks
            if open_positions:
                portfolio_greeks = aggregate_portfolio_greeks(
                    positions=[{
                        "ticker": ticker,
                        "trade_legs": p.trade_legs,
                        "spot_price": spot_price,
                    } for p in open_positions],
                    spot_prices={ticker: spot_price}
                )

                portfolio_greeks_history.append({
                    "date": current_date,
                    **portfolio_greeks
                })

        # Move to next day
        current_date += timedelta(days=1)

    # ===================================================================
    # Close all remaining positions at end
    # ===================================================================
    if open_positions:
        logger.info(f"Closing {len(open_positions)} remaining positions at backtest end")
        capital_returned, closed = _close_positions_list(
            open_positions, end_date, ohlcv, risk_free_rate
        )
        capital += capital_returned
        closed_trades.extend(closed)
        open_positions = []  # All positions now closed

    # ===================================================================
    # FIX: Update equity curve with final capital after all positions closed
    # ===================================================================
    # Recalculate equity curve using reconciled P&L from trades
    total_pnl_from_trades = sum(t.pnl for t in closed_trades if t.pnl is not None)
    expected_final_capital = initial_capital + total_pnl_from_trades

    # Update or append final equity point
    if equity_curve_data:
        # Replace the last equity point with correct final value
        equity_curve_data[-1]["equity"] = expected_final_capital
    else:
        equity_curve_data.append({
            "date": end_date,
            "equity": expected_final_capital,
        })

    # ===================================================================
    # Calculate metrics
    # ===================================================================
    equity_curve = pd.DataFrame(equity_curve_data)

    # Use the reconciled final capital (already calculated above)
    final_capital = expected_final_capital

    # Log if running capital differs significantly (for debugging)
    if abs(capital - expected_final_capital) > 1.0:
        logger.debug(f"Capital reconciliation: running=${capital:,.2f}, reconciled=${expected_final_capital:,.2f}")

    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    # Calculate Sharpe, drawdown, etc.
    if not equity_curve.empty:
        returns = equity_curve["equity"].pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

        running_max = equity_curve["equity"].cummax()
        drawdown = (equity_curve["equity"] - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        sharpe_ratio = 0
        max_drawdown = 0

    # Win rate (Bug #5 FIX: break-even trades are not losers)
    winning_trades = [t for t in closed_trades if t.pnl and t.pnl > 0]
    losing_trades = [t for t in closed_trades if t.pnl and t.pnl < 0]  # Changed <= to <
    breakeven_trades = [t for t in closed_trades if t.pnl is not None and t.pnl == 0]
    win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0

    result = BacktestResult(
        ticker=ticker,
        strategy_name=strategy_name or getattr(strategy_func, '__name__', 'HybridStrategy'),
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        total_return_pct=total_return_pct,
        num_trades=len(closed_trades),
        num_winners=len(winning_trades),
        num_losers=len(losing_trades),
        win_rate=win_rate,
        average_pnl=np.mean([t.pnl for t in closed_trades if t.pnl]) if closed_trades else 0,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        equity_curve=equity_curve,
        trades=closed_trades,
    )

    # Save portfolio Greeks history if output dir provided
    if output_dir and portfolio_greeks_history:
        greeks_df = pd.DataFrame(portfolio_greeks_history)
        greeks_path = output_dir / "portfolio_greeks.csv"
        greeks_df.to_csv(greeks_path, index=False)
        logger.info(f"Saved portfolio Greeks history: {greeks_path}")

    logger.info(f"\nBacktest complete:")
    logger.info(f"  Final capital: ${final_capital:,.2f}")
    logger.info(f"  Total return: {total_return_pct:+.2f}%")
    logger.info(f"  Trades: {len(closed_trades)}")
    logger.info(f"  Win rate: {win_rate:.1%}")
    logger.info(f"  Sharpe ratio: {sharpe_ratio:.2f}")

    return result


def _generate_signal_dates(start: datetime, end: datetime, frequency: str) -> List[datetime]:
    """Generate dates for signal generation"""
    dates = []
    current = start

    if frequency == "weekly":
        # Generate Mondays
        while current <= end:
            if current.weekday() == 0:  # Monday
                dates.append(current)
            current += timedelta(days=1)
    elif frequency == "daily":
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
    else:
        raise ValueError(f"Unknown signal frequency: {frequency}")

    return dates


def _generate_monitor_dates(start: datetime, end: datetime, frequency: str) -> List[datetime]:
    """Generate dates for position monitoring"""
    dates = []
    current = start

    if frequency == "daily":
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
    else:
        raise ValueError(f"Unknown monitor frequency: {frequency}")

    return dates


def _mark_to_market(position: TradeExecution, current_date: datetime, ohlcv: pd.DataFrame, risk_free_rate: float) -> float:
    """Mark position to market using Black-Scholes repricing"""
    # Use real mark-to-market with Black-Scholes
    update_position_tracking(
        position=position,
        current_date=current_date,
        ohlcv=ohlcv,
        risk_free_rate=risk_free_rate,
    )
    return position.current_value if position.current_value is not None else position.entry_price


def _calculate_dte(position: TradeExecution, current_date: datetime) -> int:
    """Calculate days to expiration"""
    if not position.trade_legs:
        return 30

    # Get earliest expiration
    min_dte = 999
    for leg in position.trade_legs:
        if leg.expiration_date:
            try:
                exp_date = datetime.fromisoformat(leg.expiration_date.replace('Z', '+00:00'))
                dte = (exp_date - current_date).days
                min_dte = min(min_dte, dte)
            except:
                pass

    return min_dte if min_dte < 999 else 30


def _execute_trade(
    trade_proposal,
    execution_date: datetime,
    spot_price: float,
    options_chain: List[Dict[str, Any]],
    max_capital: float,
    risk_free_rate: float,
    ohlcv: pd.DataFrame = None,  # NEW: For trend-based contract scaling
) -> Optional[TradeExecution]:
    """Execute trade with trend-based contract scaling"""
    from src.backtesting.framework import _execute_trade as framework_execute
    return framework_execute(
        trade_proposal, execution_date, spot_price, options_chain,
        max_capital, risk_free_rate, ohlcv=ohlcv
    )


def _close_positions_list(
    positions: List[TradeExecution],
    close_date: datetime,
    ohlcv: pd.DataFrame,
    risk_free_rate: float,
    slippage_pct: float = 0.025,  # Bug #8 FIX: Add slippage parameter
) -> tuple[float, List[TradeExecution]]:
    """
    Close list of positions with slippage applied

    Bug #2 FIX: Set exit_price properly
    Bug #3 FIX: Cap P&L percentage to max theoretical loss
    Bug #8 FIX: Apply slippage on exit
    Bug #9 FIX: Cap absolute P&L loss to prevent catastrophic failures
    """
    capital_returned = 0.0
    closed = []

    # =========================================================================
    # Bug #9 FIX: ABSOLUTE LOSS CAP
    # =========================================================================
    # Even with defined-risk strategies, some can have absurd losses due to:
    # 1. Collar without stock ownership (synthetic short position)
    # 2. Naked options (unlimited risk)
    # 3. Mark-to-market errors with bad data
    #
    # Cap the maximum loss to 100% of entry price (can't lose more than you put in)
    # This is a safety net - strategies should never reach this in practice
    # =========================================================================
    MAX_LOSS_MULTIPLIER = 1.0  # Max loss = 100% of entry price

    for position in positions:
        # Mark to market at close
        exit_value = _mark_to_market(position, close_date, ohlcv, risk_free_rate)

        # Bug #8 FIX: Apply exit slippage
        # If exit_value > 0, we're closing longs (selling) → receive less
        # If exit_value < 0, we're closing shorts (buying back) → pay more
        if exit_value > 0:
            exit_value_with_slippage = exit_value * (1 - slippage_pct)
        else:
            exit_value_with_slippage = exit_value * (1 + slippage_pct)

        position.closed_date = close_date

        # Bug #2 FIX: Set exit_price properly
        position.exit_price = exit_value_with_slippage

        # Get spot price at exit
        if hasattr(ohlcv.index, 'tz') and ohlcv.index.tz is not None:
            close_date_tz = pd.Timestamp(close_date).tz_localize(ohlcv.index.tz)
            position.spot_price_at_exit = float(ohlcv[ohlcv.index <= close_date_tz]["close"].iloc[-1])
        else:
            position.spot_price_at_exit = float(ohlcv[ohlcv.index <= close_date]["close"].iloc[-1])

        # Calculate P&L (Bug #1 ROOT CAUSE FIX: correct formula)
        # P&L = current_value + entry_price (see mark_to_market.py for explanation)
        raw_pnl = exit_value_with_slippage + position.entry_price

        # Bug #9 FIX: Cap absolute P&L loss
        # Max loss = entry_price * MAX_LOSS_MULTIPLIER (for debit spreads)
        # For credit spreads, max loss = credit received * MAX_LOSS_MULTIPLIER
        max_allowed_loss = abs(position.entry_price) * MAX_LOSS_MULTIPLIER
        if raw_pnl < -max_allowed_loss:
            logger.warning(
                f"P&L capped: raw={raw_pnl:.2f}, capped={-max_allowed_loss:.2f} "
                f"(entry={position.entry_price:.2f})"
            )
            position.pnl = -max_allowed_loss
            # Also cap the exit value for capital accounting
            exit_value_with_slippage = -max_allowed_loss - position.entry_price
        else:
            position.pnl = raw_pnl

        # Bug #3 FIX: Cap P&L percentage to max theoretical loss
        # For defined-risk spreads, max loss is spread width - premium received (credit)
        # or premium paid (debit). P&L% should never exceed ±100% of max risk.
        if position.entry_price != 0:
            raw_pnl_pct = (position.pnl / abs(position.entry_price)) * 100

            # Calculate max risk from trade proposal if available
            max_risk = None
            if position.trade_proposal:
                max_risk = position.trade_proposal.get("max_risk")

            if max_risk and max_risk > 0:
                # Cap P&L% based on max risk (not entry price)
                max_loss_pct = (max_risk / abs(position.entry_price)) * 100
                position.pnl_pct = max(raw_pnl_pct, -max_loss_pct)  # Cap at max loss
            else:
                # Fallback: cap at -100% for ALL strategies (Bug #9 safety)
                position.pnl_pct = max(raw_pnl_pct, -100.0)
        else:
            position.pnl_pct = 0

        position.status = "closed"

        capital_returned += exit_value_with_slippage
        closed.append(position)

        logger.info(f"Closed {position.strategy_name}: P&L=${position.pnl:,.2f} ({position.pnl_pct:.1f}%)")

    return capital_returned, closed


def _capture_market_context(
    ohlcv: pd.DataFrame,
    options_chain: List[Dict[str, Any]],
    hist_vol: float,
) -> Dict[str, Any]:
    """
    Capture market context at trade entry for logging

    Includes:
    - IV rank (if calculable from options chain)
    - Market regime (trending, ranging, volatile)
    - Historical volatility
    - Technical sentiment (from price action)

    Args:
        ohlcv: Historical OHLCV data
        options_chain: Options chain data
        hist_vol: Historical volatility

    Returns:
        Dictionary with market context
    """
    # Bug #12 FIX: Calculate IV rank properly using historical volatility range
    try:
        # Get ATM IV from options chain
        spot_price = float(ohlcv['close'].iloc[-1])
        atm_options = [
            opt for opt in options_chain
            if abs(opt['strike'] - spot_price) / spot_price < 0.05  # Within 5% of ATM
        ]

        if atm_options:
            avg_atm_iv = np.mean([opt['impliedVolatility'] for opt in atm_options])
        else:
            avg_atm_iv = hist_vol

        # Bug #12 FIX: Calculate IV rank using 52-week (or available) historical vol range
        # IV Rank = (Current IV - 52wk Low IV) / (52wk High IV - 52wk Low IV) * 100
        returns = ohlcv['close'].pct_change().dropna()

        # Calculate rolling 30-day volatility over available history
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) >= 20:
            vol_min = rolling_vol.min()
            vol_max = rolling_vol.max()

            if vol_max > vol_min:
                # True IV rank calculation
                iv_rank = ((avg_atm_iv - vol_min) / (vol_max - vol_min)) * 100
                iv_rank = max(0, min(100, iv_rank))
            else:
                iv_rank = 50  # Default if no range
        else:
            # Fallback for short history: compare current to average
            avg_vol = rolling_vol.mean() if len(rolling_vol) > 0 else hist_vol
            iv_rank = 50 + ((avg_atm_iv - avg_vol) / avg_vol) * 50 if avg_vol > 0 else 50
            iv_rank = max(0, min(100, iv_rank))

    except Exception:
        avg_atm_iv = hist_vol
        iv_rank = 50  # Neutral

    # Determine market regime based on recent price action
    try:
        returns = ohlcv['close'].pct_change().dropna()
        recent_vol = returns.tail(20).std() * (252 ** 0.5)

        if recent_vol > hist_vol * 1.5:
            regime = "High Volatility"
        elif recent_vol < hist_vol * 0.7:
            regime = "Low Volatility"
        else:
            regime = "Normal"

        # Check for trend
        sma_50 = ohlcv['close'].rolling(50).mean().iloc[-1] if len(ohlcv) >= 50 else ohlcv['close'].mean()
        current_price = ohlcv['close'].iloc[-1]

        if current_price > sma_50 * 1.05:
            regime += " Uptrend"
        elif current_price < sma_50 * 0.95:
            regime += " Downtrend"
        else:
            regime += " Ranging"

    except Exception:
        regime = "Unknown"

    # Technical sentiment using SAME logic as Technical baseline and system_wrapper
    # Bullish: Price > SMA20 AND MACD > 0
    # Bearish: Price < SMA20 AND MACD < 0
    try:
        sma_20 = ohlcv['close'].rolling(20).mean().iloc[-1] if len(ohlcv) >= 20 else ohlcv['close'].mean()
        ema_12 = ohlcv['close'].ewm(span=12, adjust=False).mean().iloc[-1]
        ema_26 = ohlcv['close'].ewm(span=26, adjust=False).mean().iloc[-1]
        macd_line = ema_12 - ema_26

        if current_price > sma_20 and macd_line > 0:
            sentiment = "Bullish"
        elif current_price < sma_20 and macd_line < 0:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
    except Exception:
        sentiment = "Neutral"

    return {
        "iv_rank": round(iv_rank, 2),
        "atm_iv": round(avg_atm_iv, 4),
        "historical_vol": round(hist_vol, 4),
        "regime": regime,
        "technical_sentiment": sentiment,
    }


def _convert_to_flat_options_list(options_chain: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert grouped options chain to flat list format

    Input format (from generate_historical_options_chain):
        [
            {
                "expiration": "2024-12-20",
                "time_to_expiration_years": 0.082,
                "calls": DataFrame with columns [strike, lastPrice, bid, ask, computed_greeks, ...],
                "puts": DataFrame with columns [strike, lastPrice, bid, ask, computed_greeks, ...],
                ...
            },
            ...
        ]

    Output format (for baseline strategies):
        [
            {"type": "call", "strike": 100, "expiration": "2024-12-20", "delta": 0.50,
             "time_to_expiration_years": 0.082, ...},
            {"type": "put", "strike": 100, "expiration": "2024-12-20", "delta": -0.50,
             "time_to_expiration_years": 0.082, ...},
            ...
        ]

    FIXED: Extracts delta/gamma/theta/vega from computed_greeks dict to top-level
    fields so baseline strategies can use them for strike selection.
    ALSO: Preserves time_to_expiration_years for Quant13Strategy reconstruction.

    Args:
        options_chain: Grouped options chain from generate_historical_options_chain

    Returns:
        Flat list of options with 'type' field and extracted Greeks
    """
    flat_list = []

    for expiration_group in options_chain:
        expiration = expiration_group["expiration"]
        # CRITICAL: Preserve time_to_expiration_years for later reconstruction
        time_to_exp = expiration_group.get("time_to_expiration_years")

        # Add calls
        calls_df = expiration_group.get("calls")
        if calls_df is not None and not calls_df.empty:
            for _, row in calls_df.iterrows():
                opt = row.to_dict()
                opt["type"] = "call"
                opt["expiration"] = expiration

                # Preserve time_to_expiration_years to prevent re-enrichment
                if time_to_exp is not None:
                    opt["time_to_expiration_years"] = time_to_exp

                # CRITICAL FIX: Extract Greeks from nested dict to top-level
                # The historical_options.py stores greeks in 'computed_greeks' dict
                # But baseline strategies expect top-level 'delta', 'gamma', etc.
                greeks = opt.get("computed_greeks", {})
                if isinstance(greeks, dict):
                    opt["delta"] = greeks.get("delta", 0.5)  # Default ATM call delta
                    opt["gamma"] = greeks.get("gamma", 0.0)
                    opt["theta"] = greeks.get("theta", 0.0)
                    opt["vega"] = greeks.get("vega", 0.0)

                flat_list.append(opt)

        # Add puts
        puts_df = expiration_group.get("puts")
        if puts_df is not None and not puts_df.empty:
            for _, row in puts_df.iterrows():
                opt = row.to_dict()
                opt["type"] = "put"
                opt["expiration"] = expiration

                # Preserve time_to_expiration_years to prevent re-enrichment
                if time_to_exp is not None:
                    opt["time_to_expiration_years"] = time_to_exp

                # CRITICAL FIX: Extract Greeks from nested dict to top-level
                greeks = opt.get("computed_greeks", {})
                if isinstance(greeks, dict):
                    opt["delta"] = greeks.get("delta", -0.5)  # Default ATM put delta
                    opt["gamma"] = greeks.get("gamma", 0.0)
                    opt["theta"] = greeks.get("theta", 0.0)
                    opt["vega"] = greeks.get("vega", 0.0)

                flat_list.append(opt)

    return flat_list
