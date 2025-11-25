"""
Backtesting framework for options trading strategies

Runs strategies on historical data with weekly rebalancing and tracks P&L.
Ensures no data leakage by only using data available up to each backtest date.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.backtesting.historical_options import (
    estimate_historical_volatility,
    generate_historical_options_chain,
)
from src.data.fetcher import fetch_ohlcv
from src.data.preprocessing import compute_returns
from src.models.schemas import TradeProposal
from src.pricing.black_scholes import black_scholes_price
from src.utils.risk import calculate_risk_metrics

logger = logging.getLogger(__name__)


@dataclass
class TradeExecution:
    """Record of a single trade execution"""
    date: datetime
    strategy_name: str
    agent_name: str
    trade_proposal: Dict[str, Any]
    entry_price: float  # Net premium paid/received
    spot_price_at_entry: float
    expiration_date: str
    closed_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    spot_price_at_exit: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"  # open, closed, expired
    highest_pnl_pct: float = 0.0  # For trailing stop logic

    # NEW: Enhanced logging for multi-ticker evaluation
    market_data: Optional[Dict[str, Any]] = None  # IV rank, regime, sentiment at entry
    agent_reports: Optional[Dict[str, Any]] = None  # All agent outputs (for Quant13 only)
    trade_legs: List[Any] = field(default_factory=list)  # TradeLeg objects for Greeks calculation
    thesis_direction: Optional[str] = None  # "Bullish" or "Bearish"

    # Mark-to-market tracking
    current_value: Optional[float] = None  # Current mark-to-market value
    current_greeks: Optional[Dict[str, float]] = None  # Current position Greeks
    days_to_expiration: Optional[int] = None  # Updated daily

    # Historical tracking
    greeks_history: List[Dict[str, Any]] = field(default_factory=list)  # Daily Greeks snapshots
    pnl_history: List[Dict[str, Any]] = field(default_factory=list)  # Daily P&L snapshots

    def to_trade_log(self) -> Dict[str, Any]:
        """Convert to detailed trade log format for JSON export"""
        return {
            "trade_id": f"{self.date.strftime('%Y%m%d')}_{self.strategy_name}",
            "execution": {
                "entry_date": self.date.isoformat(),
                "closed_date": self.closed_date.isoformat() if self.closed_date else None,
                "expiration_date": self.expiration_date,
                "days_held": (self.closed_date - self.date).days if self.closed_date else None,
            },
            "market_context": self.market_data or {},
            "trade_proposal": self.trade_proposal,
            "position_details": {
                "strategy_name": self.strategy_name,
                "agent_name": self.agent_name,
                "trade_legs": [leg.__dict__ if hasattr(leg, '__dict__') else leg for leg in self.trade_legs],
                "thesis_direction": self.thesis_direction,
            },
            "pricing": {
                "entry_price": self.entry_price,
                "exit_price": self.exit_price,
                "spot_at_entry": self.spot_price_at_entry,
                "spot_at_exit": self.spot_price_at_exit,
            },
            "performance": {
                "pnl": self.pnl,
                "pnl_pct": self.pnl_pct,
                "status": self.status,
                "highest_pnl_pct": self.highest_pnl_pct,
            },
            "position_greeks": self.current_greeks or {},
            "agent_reports": self.agent_reports or {},  # Full agent reports (Quant13 only)
            "greeks_history": self.greeks_history,
            "pnl_history": self.pnl_history,
        }


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    ticker: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    num_trades: int
    num_winners: int
    num_losers: int
    win_rate: float
    average_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[TradeExecution] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    weekly_results: List[Dict[str, Any]] = field(default_factory=list)


def run_backtest(
    ticker: str,
    start_date: datetime,
    end_date: datetime,
    strategy_func: Callable,
    strategy_name: str,
    rebalance_frequency: str = "weekly",
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.10,  # Use 10% of capital per trade (variance harvesting - tail risk protection)
    risk_free_rate: float = 0.05,
    output_dir: Optional[Path] = None,
    profit_target_pct: float = 0.25,  # Take profit at 25% gain (realistic for spreads)
    stop_loss_pct: float = 0.50,  # Stop loss at 50% loss (tighter risk management)
    hold_to_expiry: bool = False,  # If True, use old behavior (close weekly)
    slippage_pct: float = 0.025,  # 2.5% slippage on entry/exit (realistic for liquid equity options)
) -> BacktestResult:
    """
    Run backtest for a strategy with improved position management

    Args:
        ticker: Stock ticker to trade
        start_date: Backtest start date
        end_date: Backtest end date
        strategy_func: Function that generates trade proposals
                      Should accept (ticker, ohlcv_up_to_date, options_chain, spot_price, **kwargs)
        strategy_name: Name of the strategy
        rebalance_frequency: "weekly" or "daily"
        initial_capital: Starting capital
        position_size_pct: Percentage of capital to use per trade (default: 20%)
        risk_free_rate: Risk-free rate for pricing
        output_dir: Directory to save weekly results (optional)
        profit_target_pct: Take profit when position gains this % (default: 25% for spreads)
        stop_loss_pct: Cut losses when position loses this % (default: 50%)
        hold_to_expiry: If True, use old behavior (close all positions weekly)
        slippage_pct: Transaction cost as % of price (default: 2.5% for liquid equity options)

    Returns:
        BacktestResult with performance metrics
    """
    logger.info(f"Starting backtest: {strategy_name} on {ticker}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Rebalance: {rebalance_frequency}")

    # Fetch full historical data (we'll slice it to prevent leakage)
    # Fetch more data than needed for technical indicators
    lookback_days = (end_date - start_date).days + 400
    full_ohlcv = fetch_ohlcv(ticker, lookback_days)
    full_ohlcv = compute_returns(full_ohlcv)

    # Generate rebalance dates
    rebalance_dates = _generate_rebalance_dates(start_date, end_date, rebalance_frequency)

    # ========================================================================
    # CAPITAL ACCOUNTING EXPLANATION
    # ========================================================================
    # Capital tracks AVAILABLE CASH, not total portfolio value
    #
    # Credit Spreads (e.g., Bull Put Spread):
    #   - Entry: Receive premium → capital INCREASES (entry_price > 0)
    #   - Exit: Buy back spread → capital DECREASES
    #   - Example: Sell spread for $200 credit
    #     * At entry: capital += $200 (we receive cash)
    #     * At exit: capital -= exit_value (we pay to close)
    #
    # Debit Spreads (e.g., Bull Call Spread):
    #   - Entry: Pay premium → capital DECREASES (entry_price < 0)
    #   - Exit: Sell spread → capital INCREASES
    #   - Example: Buy spread for -$300 debit
    #     * At entry: capital -= $300 (we pay cash)
    #     * At exit: capital += exit_value (we receive cash from sale)
    #
    # Position Sizing:
    #   - position_size_pct = 10% means allocate 10% of capital per trade
    #   - max_concurrent_positions = 3 allows 30% total allocation
    #   - With credit spreads, capital can INCREASE above initial_capital
    #   - This is realistic: selling premium generates cash inflow
    # ========================================================================
    capital = initial_capital
    open_positions: List[TradeExecution] = []
    closed_trades: List[TradeExecution] = []
    equity_curve_data = []
    weekly_results = []

    for i, rebalance_date in enumerate(rebalance_dates):
        logger.info(f"\n{'='*60}")
        logger.info(f"Rebalance #{i+1}: {rebalance_date.date()}")
        logger.info(f"{'='*60}")

        # Get data up to this point (prevent future data leakage)
        # Handle timezone-aware index
        if full_ohlcv.index.tz is not None:
            # Make rebalance_date timezone-aware
            rebalance_date_tz = pd.Timestamp(rebalance_date).tz_localize('UTC').tz_convert(full_ohlcv.index.tz)
            ohlcv_up_to_date = full_ohlcv[full_ohlcv.index <= rebalance_date_tz].copy()
        else:
            ohlcv_up_to_date = full_ohlcv[full_ohlcv.index <= rebalance_date].copy()

        if len(ohlcv_up_to_date) < 50:
            logger.warning(f"Insufficient data at {rebalance_date.date()}, skipping")
            continue

        spot_price = float(ohlcv_up_to_date["close"].iloc[-1])

        # Close expired positions
        capital_returned, closed = _close_expired_positions(
            open_positions, rebalance_date, ohlcv_up_to_date, risk_free_rate, slippage_pct
        )
        # CAPITAL FLOW: Add exit value back to capital
        # - For long positions: exit_value > 0 (we sell and receive cash)
        # - For short positions: exit_value < 0 (we buy back and pay cash)
        capital += capital_returned
        closed_trades.extend(closed)
        open_positions = [p for p in open_positions if p.status == "open"]

        logger.info(f"Current capital: ${capital:,.2f}")
        logger.info(f"Open positions: {len(open_positions)}")

        # Improved position management: Check for profit targets and stop losses
        if not hold_to_expiry and open_positions:
            positions_to_close = []

            for position in open_positions:
                current_value = _mark_to_market(position, rebalance_date, ohlcv_up_to_date, risk_free_rate)
                pnl = current_value - position.entry_price
                pnl_pct = pnl / abs(position.entry_price) if position.entry_price != 0 else 0

                # Check profit target (50% gain)
                if pnl_pct >= profit_target_pct:
                    logger.info(f"  Profit target hit: {pnl_pct:.1%} gain, closing position")
                    positions_to_close.append(position)
                    continue

                # Check stop loss (80% loss - aggressive to cut losers)
                if pnl_pct <= -stop_loss_pct:
                    logger.info(f"  Stop loss hit: {pnl_pct:.1%} loss, closing position")
                    positions_to_close.append(position)
                    continue

                # Check if position has < 7 days to expiration (close early to avoid gamma risk)
                try:
                    exp_date = datetime.strptime(position.expiration_date, "%Y-%m-%d")
                    days_to_exp = (exp_date - rebalance_date).days
                    if days_to_exp <= 7:
                        logger.info(f"  Position near expiry ({days_to_exp} days), closing")
                        positions_to_close.append(position)
                except Exception:
                    pass

            # Close positions that hit targets/stops
            if positions_to_close:
                capital_returned, closed = _close_positions(
                    positions_to_close, rebalance_date, ohlcv_up_to_date, risk_free_rate, slippage_pct
                )
                capital += capital_returned
                closed_trades.extend(closed)
                open_positions = [p for p in open_positions if p not in positions_to_close]
                logger.info(f"  Closed {len(closed)} positions, capital returned: ${capital_returned:,.2f}")

        # Old behavior: close all positions weekly (only if hold_to_expiry=True)
        elif hold_to_expiry and open_positions:
            capital_returned, closed = _close_positions(
                open_positions, rebalance_date, ohlcv_up_to_date, risk_free_rate, slippage_pct
            )
            capital += capital_returned  # Add returned capital, don't replace!
            closed_trades.extend(closed)
            open_positions = []

        # Generate historical options chain
        hist_vol = estimate_historical_volatility(ohlcv_up_to_date, window=30)
        options_chain = generate_historical_options_chain(
            historical_date=rebalance_date,
            spot_price=spot_price,
            historical_volatility=hist_vol,
            risk_free_rate=risk_free_rate,
            ticker=ticker,  # Pass ticker for skew calibration
        )

        # Generate trade proposal from strategy
        # Only open new positions if we have capacity (not at max positions)
        max_concurrent_positions = 3  # Limit to 3 concurrent positions
        should_open_new = len(open_positions) < max_concurrent_positions

        agent_reports = None
        trade_proposal = None

        if should_open_new:
            try:
                result = strategy_func(
                    ticker=ticker,
                    ohlcv=ohlcv_up_to_date,
                    options_chain=options_chain,
                    spot_price=spot_price,
                )

                # Check if strategy returned reports along with proposal
                if isinstance(result, tuple):
                    trade_proposal, agent_reports = result
                else:
                    trade_proposal = result
                    agent_reports = None

                # Handle None (no trade signal from strategy due to low conviction)
                if trade_proposal is None:
                    logger.info(f"Strategy returned no trade (low conviction or unclear setup)")
                    # Skip to next period - cash is a position!
                    continue

                # Execute trade
                position_size = capital * position_size_pct
                execution = _execute_trade(
                    trade_proposal,
                    rebalance_date,
                    spot_price,
                    options_chain,
                    position_size,
                    risk_free_rate,
                    slippage_pct,
                )

                if execution:
                    # CAPITAL FLOW: Add entry_price to capital
                    # - Debit spread (entry_price < 0): Capital DECREASES (we pay premium)
                    # - Credit spread (entry_price > 0): Capital INCREASES (we receive premium)
                    capital += execution.entry_price
                    open_positions.append(execution)
                    logger.info(f"Opened: {execution.strategy_name}")
                    logger.info(f"  Entry price: ${execution.entry_price:,.2f} ({'debit' if execution.entry_price < 0 else 'credit'})")
                    logger.info(f"  Remaining capital: ${capital:,.2f}")

            except Exception as e:
                logger.error(f"Strategy execution failed at {rebalance_date.date()}: {e}")
                # Continue to next period
        else:
            logger.info(f"At max positions ({len(open_positions)}/{max_concurrent_positions}), skipping new trade")

        # Monitor positions DAILY between this rebalance and the next
        # This prevents "bleeding out" from theta decay and adverse moves
        if open_positions and i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i + 1]
            open_positions, closed_trades, capital = _monitor_positions_daily(
                open_positions=open_positions,
                closed_trades=closed_trades,
                capital=capital,
                start_date=rebalance_date,
                end_date=next_rebalance,
                full_ohlcv=full_ohlcv,
                risk_free_rate=risk_free_rate,
                profit_target_pct=profit_target_pct,
                stop_loss_pct=stop_loss_pct,
                slippage_pct=slippage_pct,
            )

        # Record equity curve point
        total_equity = capital + sum(
            _mark_to_market(p, rebalance_date, full_ohlcv, risk_free_rate)
            for p in open_positions
        )
        equity_curve_data.append({
            "date": rebalance_date,
            "equity": total_equity,
            "capital": capital,
            "open_positions": len(open_positions),
        })

        # Save weekly result
        weekly_result = {
            "date": rebalance_date.isoformat(),
            "spot_price": spot_price,
            "capital": capital,
            "total_equity": total_equity,
            "open_positions": len(open_positions),
            "closed_trades_this_period": len(closed),
            "trade_proposal": trade_proposal.model_dump() if trade_proposal else None,
        }
        weekly_results.append(weekly_result)

        # Save to file if output_dir provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            week_file = output_dir / f"week_{i+1}_{rebalance_date.strftime('%Y%m%d')}.json"
            with open(week_file, "w") as f:
                json.dump(weekly_result, f, indent=2)

            # Save agent reports if available
            if agent_reports:
                reports_dir = output_dir / "agent_reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                reports_file = reports_dir / f"week_{i+1}_{rebalance_date.strftime('%Y%m%d')}.json"

                # Convert Pydantic models to dicts
                reports_dict = {}
                for key, report in agent_reports.items():
                    if hasattr(report, "model_dump"):
                        reports_dict[key] = report.model_dump()
                    else:
                        reports_dict[key] = report

                with open(reports_file, "w") as f:
                    json.dump(reports_dict, f, indent=2)
                logger.debug(f"Saved agent reports to {reports_file}")

    # Close any remaining positions at end date
    if open_positions:
        capital_returned, closed = _close_positions(
            open_positions, end_date, full_ohlcv, risk_free_rate, slippage_pct
        )
        capital += capital_returned  # Add returned capital, don't replace!
        closed_trades.extend(closed)

    # Calculate performance metrics
    final_capital = capital
    total_return = final_capital - initial_capital
    total_return_pct = (total_return / initial_capital) * 100

    num_winners = sum(1 for t in closed_trades if t.pnl and t.pnl > 0)
    num_losers = sum(1 for t in closed_trades if t.pnl and t.pnl < 0)
    win_rate = num_winners / len(closed_trades) if closed_trades else 0.0

    average_pnl = sum(t.pnl for t in closed_trades if t.pnl) / len(closed_trades) if closed_trades else 0.0

    # Calculate max drawdown
    equity_curve = pd.DataFrame(equity_curve_data)
    if not equity_curve.empty:
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["peak"]) / equity_curve["peak"]
        max_drawdown = equity_curve["drawdown"].min()
    else:
        max_drawdown = 0.0

    # Calculate Sharpe ratio (simplified)
    if not equity_curve.empty and len(equity_curve) > 1:
        equity_curve["returns"] = equity_curve["equity"].pct_change()
        avg_return = equity_curve["returns"].mean()
        std_return = equity_curve["returns"].std()
        sharpe_ratio = (avg_return / std_return) * np.sqrt(52) if std_return > 0 else 0.0  # Weekly to annual
    else:
        sharpe_ratio = 0.0

    result = BacktestResult(
        strategy_name=strategy_name,
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        total_return_pct=total_return_pct,
        num_trades=len(closed_trades),
        num_winners=num_winners,
        num_losers=num_losers,
        win_rate=win_rate,
        average_pnl=average_pnl,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        trades=closed_trades,
        equity_curve=equity_curve,
        weekly_results=weekly_results,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST COMPLETE: {strategy_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Total Return: ${total_return:,.2f} ({total_return_pct:.2f}%)")
    logger.info(f"Win Rate: {win_rate:.1%}")
    logger.info(f"Avg P&L: ${average_pnl:,.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    return result


def _monitor_positions_daily(
    open_positions: List[TradeExecution],
    closed_trades: List[TradeExecution],
    capital: float,
    start_date: datetime,
    end_date: datetime,
    full_ohlcv: pd.DataFrame,
    risk_free_rate: float,
    profit_target_pct: float,
    stop_loss_pct: float,
    slippage_pct: float,
) -> tuple[List[TradeExecution], List[TradeExecution], float]:
    """
    Monitor open positions DAILY for stop-loss and profit targets.

    This function checks positions every single day between rebalance dates,
    not just at weekly intervals. This prevents positions from "bleeding out"
    due to time decay and adverse price movements during the week.

    Args:
        open_positions: List of currently open positions
        closed_trades: List of closed trades (will be appended to)
        capital: Current capital
        start_date: Start of monitoring period
        end_date: End of monitoring period
        full_ohlcv: Full OHLCV dataframe
        risk_free_rate: Risk-free rate
        profit_target_pct: Take profit threshold (e.g., 0.25 for 25%)
        stop_loss_pct: Stop loss threshold (e.g., 0.50 for 50%)
        slippage_pct: Slippage percentage

    Returns:
        Tuple of (open_positions, closed_trades, capital)
    """
    if not open_positions:
        return open_positions, closed_trades, capital

    # Generate daily dates for monitoring
    current_date = start_date + timedelta(days=1)  # Start day after entry

    while current_date < end_date and open_positions:
        # Get OHLCV data up to current date
        if full_ohlcv.index.tz is not None:
            current_date_tz = pd.Timestamp(current_date).tz_localize('UTC').tz_convert(full_ohlcv.index.tz)
            ohlcv_to_date = full_ohlcv[full_ohlcv.index <= current_date_tz].copy()
        else:
            ohlcv_to_date = full_ohlcv[full_ohlcv.index <= current_date].copy()

        if len(ohlcv_to_date) == 0:
            current_date += timedelta(days=1)
            continue

        positions_to_close = []

        for position in open_positions:
            # Check if position has expired
            if _is_expired(position, current_date):
                positions_to_close.append(position)
                logger.info(f"  [{current_date.date()}] Position expired, closing")
                continue

            # Mark to market
            current_value = _mark_to_market(position, current_date, ohlcv_to_date, risk_free_rate)
            # FIXED: Correct P&L accounting without abs()
            # For debit spreads (entry_price < 0): we paid, so pnl = current_value - entry_price
            # For credit spreads (entry_price > 0): we received, so pnl = current_value - entry_price
            # In both cases: pnl = exit_value - entry_price (standard accounting)
            pnl = current_value - position.entry_price
            pnl_pct = pnl / abs(position.entry_price) if position.entry_price != 0 else 0

            # Update highest P&L for trailing stop
            position.highest_pnl_pct = max(position.highest_pnl_pct, pnl_pct)

            # ================================================================
            # DIFFERENTIATED PROFIT TARGETS: Credit vs Debit Spreads
            # ================================================================
            # Credit spreads (entry_price > 0): We received premium
            # - TP: 50% of credit (capture half the premium decay)
            # - SL: 100% of credit (risk all premium + spread width)
            #
            # Debit spreads (entry_price < 0): We paid premium
            # - TP: 40% of debit (capture significant move)
            # - SL: 20% of debit (tight stop to preserve capital)
            is_credit_spread = position.entry_price > 0

            if is_credit_spread:
                # Credit Spread: Theta-positive strategy
                credit_tp_pct = 0.50  # Take 50% of max profit
                credit_sl_pct = 1.00  # Risk full credit received

                if pnl_pct >= credit_tp_pct:
                    logger.info(f"  [{current_date.date()}] Credit TP hit: {pnl_pct:.1%} gain, closing")
                    positions_to_close.append(position)
                    continue

                if pnl_pct <= -credit_sl_pct:
                    logger.info(f"  [{current_date.date()}] Credit SL hit: {pnl_pct:.1%} loss, closing")
                    positions_to_close.append(position)
                    continue
            else:
                # Debit Spread: Directional strategy
                debit_tp_pct = 0.40  # Take 40% profit
                debit_sl_pct = 0.20  # Stop at 20% loss

                if pnl_pct >= debit_tp_pct:
                    logger.info(f"  [{current_date.date()}] Debit TP hit: {pnl_pct:.1%} gain, closing")
                    positions_to_close.append(position)
                    continue

                if pnl_pct <= -debit_sl_pct:
                    logger.info(f"  [{current_date.date()}] Debit SL hit: {pnl_pct:.1%} loss, closing")
                    positions_to_close.append(position)
                    continue

            # ================================================================
            # TRAILING STOP: Protect profits from turning into losses
            # ================================================================
            # If we were ever up 30%, lock in 15% minimum profit
            # This prevents big winners from becoming losers
            TRAILING_TRIGGER = 0.30  # Start trailing at +30%
            TRAILING_STOP = 0.15     # Lock in at least +15%

            if position.highest_pnl_pct >= TRAILING_TRIGGER:
                if pnl_pct < TRAILING_STOP:
                    logger.info(f"  [{current_date.date()}] Trailing stop hit: was {position.highest_pnl_pct:.1%}, now {pnl_pct:.1%}, closing")
                    positions_to_close.append(position)
                    continue

        # Close positions that hit targets/stops
        if positions_to_close:
            capital_returned, newly_closed = _close_positions(
                positions_to_close, current_date, ohlcv_to_date, risk_free_rate, slippage_pct
            )
            capital += capital_returned
            closed_trades.extend(newly_closed)
            open_positions = [p for p in open_positions if p not in positions_to_close]
            logger.info(f"  [{current_date.date()}] Closed {len(newly_closed)} positions via daily monitoring")

        current_date += timedelta(days=1)

    return open_positions, closed_trades, capital


def _generate_rebalance_dates(
    start_date: datetime, end_date: datetime, frequency: str
) -> List[datetime]:
    """Generate rebalancing dates"""
    dates = []
    current = start_date

    if frequency == "weekly":
        delta = timedelta(days=7)
    else:  # daily
        delta = timedelta(days=1)

    while current <= end_date:
        dates.append(current)
        current += delta

    return dates


def _detect_trend_strength(ohlcv: pd.DataFrame) -> str:
    """
    Detect market trend strength for contract scaling.

    Returns:
        "strong_trend": Clear directional move (scale up to 3 contracts)
        "weak_trend": Some direction but not clear (scale to 2 contracts)
        "no_trend": Choppy/sideways market (use 1 contract only)
    """
    if ohlcv is None or len(ohlcv) < 20:
        return "no_trend"

    try:
        close = ohlcv['close']

        # Calculate indicators
        sma_10 = close.rolling(10).mean().iloc[-1]
        sma_20 = close.rolling(20).mean().iloc[-1]
        current_price = close.iloc[-1]

        # Calculate ADX-like trend strength (simplified)
        # Use price position relative to SMAs and SMA alignment
        price_above_sma10 = current_price > sma_10
        price_above_sma20 = current_price > sma_20
        sma10_above_sma20 = sma_10 > sma_20

        # Calculate recent volatility vs directional move
        returns_10d = (close.iloc[-1] / close.iloc[-10] - 1) * 100 if len(close) >= 10 else 0
        returns_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0

        # Strong trend: All aligned + significant move (>3% in 10 days)
        all_bullish = price_above_sma10 and price_above_sma20 and sma10_above_sma20
        all_bearish = not price_above_sma10 and not price_above_sma20 and not sma10_above_sma20

        if (all_bullish or all_bearish) and abs(returns_10d) > 3:
            return "strong_trend"

        # Weak trend: Some alignment or moderate move
        if (all_bullish or all_bearish) or abs(returns_10d) > 2:
            return "weak_trend"

        # No trend: Mixed signals, choppy market
        return "no_trend"

    except Exception:
        return "no_trend"


def _execute_trade(
    trade_proposal: TradeProposal,
    execution_date: datetime,
    spot_price: float,
    options_chain: List[Dict[str, Any]],
    max_capital: float,
    risk_free_rate: float,
    slippage_pct: float = 0.025,
    ohlcv: pd.DataFrame = None,  # NEW: For trend detection
) -> Optional[TradeExecution]:
    """
    Execute a trade and return execution record

    Calculates net premium and creates execution record with slippage applied

    Slippage logic:
    - When buying (paying premium), we pay MORE: premium * (1 + slippage)
    - When selling (receiving premium), we receive LESS: premium * (1 - slippage)
    - Net effect: slippage always hurts the trader

    Contract scaling based on trend:
    - Strong trend: Up to 3 contracts (capture the move)
    - Weak trend: Up to 2 contracts (moderate exposure)
    - No trend: 1 contract only (preserve capital in chop)
    """
    try:
        # Calculate risk metrics
        risk_metrics = calculate_risk_metrics(trade_proposal, options_chain)

        net_premium = risk_metrics.get("net_premium", 0.0)
        if net_premium is None:
            net_premium = 0.0

        # Apply slippage to entry
        # If net_premium < 0, we're paying (debit), so we pay more
        # If net_premium > 0, we're receiving (credit), so we receive less
        if net_premium < 0:
            # Debit trade: pay more due to slippage
            net_premium_with_slippage = net_premium * (1 + slippage_pct)
        else:
            # Credit trade: receive less due to slippage
            net_premium_with_slippage = net_premium * (1 - slippage_pct)

        logger.info(f"  Entry slippage: ${abs(net_premium_with_slippage - net_premium):.2f} ({slippage_pct:.1%})")

        # =========================================================================
        # CONTRACT SCALING: Trade multiple contracts to maximize capital utilization
        # =========================================================================
        # For debit trades: net_premium < 0 (we pay)
        # For credit trades: net_premium > 0 (we receive, but max_risk is the margin required)

        max_risk = risk_metrics.get("max_risk", 0)

        if net_premium_with_slippage < 0:
            # Debit trade: capital needed = premium paid per contract
            capital_per_contract = abs(net_premium_with_slippage)
        else:
            # Credit trade: capital needed = max_risk (spread width - credit received)
            # max_risk is typically spread_width * 100 - credit_received
            if max_risk and max_risk > 0:
                capital_per_contract = max_risk
            else:
                # Fallback: estimate max risk as 5x the credit received
                capital_per_contract = abs(net_premium_with_slippage) * 5

        # =========================================================================
        # NORMALIZED POSITION SIZING: All strategies use same % of capital
        # =========================================================================
        # This ensures fair comparison - a $10 option and $500 spread get equal weight
        # based on capital allocation, not arbitrary contract limits.

        trend_strength = _detect_trend_strength(ohlcv)

        # Target position size as % of available capital based on trend
        # =====================================================================
        # ULTRA CONSERVATIVE SIZING (Based on 180-day analysis)
        # =====================================================================
        # Problem: Avg Loss = $402 (2.5x bigger than Avg Win = $164)
        # Solution: Reduce position sizes to limit $ impact of losers
        # Previous: 20/15/10% → New: 15/10/8%
        # =====================================================================
        if trend_strength == "strong_trend":
            target_position_pct = 0.15  # Use 15% of allocated capital in strong trends (was 20%)
        elif trend_strength == "weak_trend":
            target_position_pct = 0.10  # Use 10% in weak trends (was 15%)
        else:  # no_trend
            target_position_pct = 0.08  # Use only 8% in choppy markets (was 10%)

        # Calculate target dollar amount
        target_position_size = max_capital * target_position_pct

        # Calculate contracts needed to reach target (minimum 1)
        if capital_per_contract > 0:
            num_contracts = max(1, int(target_position_size / capital_per_contract))
        else:
            num_contracts = 1

        # Safety cap: Never exceed 5 contracts to limit catastrophic losses
        num_contracts = min(num_contracts, 5)

        actual_position_size = capital_per_contract * num_contracts
        logger.info(f"  Trend: {trend_strength} → target {target_position_pct:.0%} of ${max_capital:,.0f} = ${target_position_size:,.0f}")
        logger.info(f"  Contract scaling: {num_contracts} contracts @ ${capital_per_contract:,.0f} = ${actual_position_size:,.0f}")

        # Scale the premium by number of contracts
        net_premium_with_slippage = net_premium_with_slippage * num_contracts

        # =========================================================================
        # CRITICAL FIX: Scale leg quantities to match contract count
        # =========================================================================
        # Without this, mark-to-market only values 1 contract while we paid for N
        # This caused massive losses because exit value was 1/N of entry value
        for leg in trade_proposal.trade_legs:
            leg.quantity = num_contracts

        # Check if we have enough capital (use slippage-adjusted premium)
        if net_premium_with_slippage < 0 and abs(net_premium_with_slippage) > max_capital:
            logger.warning(f"Insufficient capital for trade: need ${abs(net_premium_with_slippage):,.2f}, have ${max_capital:,.2f}")
            return None

        # Find expiration date
        expiration_date = trade_proposal.trade_legs[0].expiration_date if trade_proposal.trade_legs else "unknown"

        # Bug #7 FIX: Determine thesis direction from strategy name (not conviction level)
        # Bullish strategies: Bull Call Spread, Bull Put Spread, Long Call, Cash-Secured Put
        # Bearish strategies: Bear Call Spread, Bear Put Spread, Long Put
        # Neutral strategies: Iron Condor, Butterfly, Straddle, Strangle
        strategy_name_lower = trade_proposal.strategy_name.lower()
        if any(x in strategy_name_lower for x in ['bull', 'long call', 'cash-secured put', 'covered call']):
            thesis_dir = "Bullish"
        elif any(x in strategy_name_lower for x in ['bear', 'long put']):
            thesis_dir = "Bearish"
        else:
            thesis_dir = "Neutral"

        execution = TradeExecution(
            date=execution_date,
            strategy_name=trade_proposal.strategy_name,
            agent_name=trade_proposal.agent,
            trade_proposal=trade_proposal.model_dump(),
            entry_price=net_premium_with_slippage,  # Use slippage-adjusted price
            spot_price_at_entry=spot_price,
            expiration_date=expiration_date,
            status="open",
            # NEW: Add trade legs for Greeks calculation
            trade_legs=trade_proposal.trade_legs,
            # Bug #7 FIX: Use actual direction, not conviction level
            thesis_direction=thesis_dir,
        )

        return execution

    except Exception as e:
        logger.error(f"Failed to execute trade: {e}")
        return None


def _close_positions(
    positions: List[TradeExecution],
    close_date: datetime,
    ohlcv: pd.DataFrame,
    risk_free_rate: float,
    slippage_pct: float = 0.025,
) -> tuple[float, List[TradeExecution]]:
    """
    Close all positions and return (capital_returned, closed_positions)

    Applies slippage on exit:
    - When closing long positions (selling), receive LESS
    - When closing short positions (buying back), pay MORE
    """
    capital_returned = 0.0
    closed = []

    for position in positions:
        exit_value = _mark_to_market(position, close_date, ohlcv, risk_free_rate)

        # Apply exit slippage
        # If exit_value > 0, we're closing longs (selling) → receive less
        # If exit_value < 0, we're closing shorts (buying back) → pay more
        if exit_value > 0:
            exit_value_with_slippage = exit_value * (1 - slippage_pct)
        else:
            exit_value_with_slippage = exit_value * (1 + slippage_pct)

        # Update position
        position.closed_date = close_date
        position.exit_price = exit_value_with_slippage  # Use slippage-adjusted exit

        # Handle timezone-aware index
        if ohlcv.index.tz is not None:
            close_date_tz = pd.Timestamp(close_date).tz_localize('UTC').tz_convert(ohlcv.index.tz)
            position.spot_price_at_exit = float(ohlcv[ohlcv.index <= close_date_tz]["close"].iloc[-1])
        else:
            position.spot_price_at_exit = float(ohlcv[ohlcv.index <= close_date]["close"].iloc[-1])

        # Bug #1 ROOT CAUSE FIX: Correct P&L calculation
        # entry_price is signed: negative for debit (cash out), positive for credit (cash in)
        # exit_value is what we receive when closing (positive for selling, negative for buying back)
        #
        # P&L = current_value + entry_price
        # - Debit spread: paid $400 (entry=-400), now worth $500 (exit=+500) → P&L = 500 + (-400) = +$100
        # - Credit spread: received $300 (entry=+300), now pay $100 to close (exit=-100) → P&L = -100 + 300 = +$200
        position.pnl = exit_value_with_slippage + position.entry_price
        position.pnl_pct = (position.pnl / abs(position.entry_price)) * 100 if position.entry_price != 0 else 0
        position.status = "closed"

        capital_returned += exit_value_with_slippage  # Return slippage-adjusted value
        closed.append(position)

        logger.info(f"Closed {position.strategy_name}: P&L=${position.pnl:,.2f} ({position.pnl_pct:.1f}%)")

    return capital_returned, closed


def _close_expired_positions(
    positions: List[TradeExecution],
    current_date: datetime,
    ohlcv: pd.DataFrame,
    risk_free_rate: float,
    slippage_pct: float = 0.025,
) -> tuple[float, List[TradeExecution]]:
    """Close positions that have expired"""
    expired = [p for p in positions if _is_expired(p, current_date)]

    if not expired:
        return 0.0, []

    return _close_positions(expired, current_date, ohlcv, risk_free_rate, slippage_pct)


def _is_expired(position: TradeExecution, current_date: datetime) -> bool:
    """Check if position has expired"""
    try:
        exp_date = datetime.fromisoformat(position.expiration_date)
        return current_date >= exp_date
    except Exception:
        return False


def _mark_to_market(
    position: TradeExecution,
    mark_date: datetime,
    ohlcv: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """
    Calculate current market value of a position

    Uses Black-Scholes to price options at the mark date
    """
    try:
        # Get spot price at mark date
        # Handle timezone-aware index
        if ohlcv.index.tz is not None:
            mark_date_tz = pd.Timestamp(mark_date).tz_localize('UTC').tz_convert(ohlcv.index.tz)
            spot_price = float(ohlcv[ohlcv.index <= mark_date_tz]["close"].iloc[-1])
        else:
            spot_price = float(ohlcv[ohlcv.index <= mark_date]["close"].iloc[-1])

        # Calculate time to expiration
        exp_date = datetime.fromisoformat(position.expiration_date)
        days_to_exp = (exp_date - mark_date).days
        T = max(days_to_exp / 365.0, 0.0)

        # Get IV from entry
        trade_legs = position.trade_proposal.get("trade_legs", [])
        if not trade_legs:
            return 0.0

        # Estimate IV from recent volatility
        # Handle timezone-aware index
        if ohlcv.index.tz is not None:
            mark_date_tz = pd.Timestamp(mark_date).tz_localize('UTC').tz_convert(ohlcv.index.tz)
            recent_ohlcv = ohlcv[ohlcv.index <= mark_date_tz].tail(30)
        else:
            recent_ohlcv = ohlcv[ohlcv.index <= mark_date].tail(30)
        from src.backtesting.historical_options import estimate_historical_volatility
        sigma = estimate_historical_volatility(recent_ohlcv, window=min(30, len(recent_ohlcv)))

        # Value each leg
        total_value = 0.0
        OPTIONS_MULTIPLIER = 100.0  # Standard options contract size

        for leg in trade_legs:
            option_type = leg.get("type")  # "CALL" or "PUT" (uppercase from trade proposal)
            strike = leg.get("strike_price")
            action = leg.get("action")  # "BUY" or "SELL"
            quantity = leg.get("quantity", 1)

            # Convert to lowercase for black_scholes_price function
            option_type_lower = option_type.lower() if option_type else "call"

            # Price the option (per share)
            price = black_scholes_price(
                S=spot_price,
                K=strike,
                T=T,
                r=risk_free_rate,
                sigma=sigma,
                option_type=option_type_lower,
            )

            # Add or subtract based on action (handle both uppercase and lowercase)
            # Multiply by OPTIONS_MULTIPLIER to match entry pricing
            if action and action.upper() == "BUY":
                total_value += price * quantity * OPTIONS_MULTIPLIER
            else:  # SELL
                total_value -= price * quantity * OPTIONS_MULTIPLIER

        return total_value

    except Exception as e:
        logger.warning(f"Failed to mark position to market: {e}")
        return 0.0


# Import numpy for Sharpe calculation
import numpy as np
