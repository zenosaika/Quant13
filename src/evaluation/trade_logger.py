"""
Trade Logger for Multi-Ticker Evaluation

Logs individual trades to JSON files with comprehensive details including:
- Market context (IV rank, regime, sentiment)
- Trade proposal and execution
- Position Greeks
- Agent reports (for Quant13)
- Historical tracking (Greeks history, P&L history)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.backtesting.framework import TradeExecution

logger = logging.getLogger(__name__)


class TradeLogger:
    """
    Logs individual trades to JSON files

    Directory structure:
        output_dir/
        ├── tickers/
        │   ├── SPY/
        │   │   ├── quant13/
        │   │   │   ├── trades/
        │   │   │   │   ├── trade_20240115_001.json
        │   │   │   │   ├── trade_20240122_002.json
        │   │   ├── technical/
        │   │   │   ├── trades/
        │   │   ├── chimpanzee/
        │   │   │   ├── trades/
    """

    def __init__(self, output_dir: Path):
        """
        Initialize trade logger

        Args:
            output_dir: Base directory for all evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.tickers_dir = self.output_dir / "tickers"
        self.tickers_dir.mkdir(parents=True, exist_ok=True)

    def log_trade(
        self,
        ticker: str,
        strategy_name: str,
        trade: TradeExecution,
    ) -> Path:
        """
        Log individual trade to JSON file

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name (quant13, technical, chimpanzee)
            trade: TradeExecution object

        Returns:
            Path to saved trade log file
        """
        # Create directory structure
        strategy_dir = self.tickers_dir / ticker / strategy_name.lower() / "trades"
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        trade_date = trade.date.strftime("%Y%m%d")
        trade_id = f"trade_{trade_date}_{trade.strategy_name.replace(' ', '_')}.json"
        trade_path = strategy_dir / trade_id

        # Convert to log format
        trade_log = trade.to_trade_log()

        # Save to JSON
        try:
            with open(trade_path, 'w') as f:
                json.dump(trade_log, f, indent=2, default=str)

            logger.debug(f"Logged trade: {trade_path}")
            return trade_path

        except Exception as e:
            logger.error(f"Failed to log trade {trade_id}: {e}")
            raise

    def log_trades_batch(
        self,
        ticker: str,
        strategy_name: str,
        trades: List[TradeExecution],
    ) -> List[Path]:
        """
        Log multiple trades in batch

        Args:
            ticker: Stock ticker
            strategy_name: Strategy name
            trades: List of TradeExecution objects

        Returns:
            List of paths to saved trade log files
        """
        paths = []
        for trade in trades:
            try:
                path = self.log_trade(ticker, strategy_name, trade)
                paths.append(path)
            except Exception as e:
                logger.error(f"Failed to log trade in batch: {e}")
                continue

        logger.info(f"Logged {len(paths)}/{len(trades)} trades for {ticker} {strategy_name}")
        return paths

    def load_trade_log(self, trade_path: Path) -> Dict[str, Any]:
        """
        Load trade log from JSON file

        Args:
            trade_path: Path to trade log file

        Returns:
            Trade log dictionary
        """
        try:
            with open(trade_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load trade log {trade_path}: {e}")
            raise

    def load_all_trades(
        self,
        ticker: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all trade logs (optionally filtered by ticker and strategy)

        Args:
            ticker: Optional ticker filter
            strategy_name: Optional strategy filter

        Returns:
            List of trade log dictionaries
        """
        trade_logs = []

        # Build search path
        if ticker and strategy_name:
            search_dirs = [self.tickers_dir / ticker / strategy_name.lower() / "trades"]
        elif ticker:
            ticker_dir = self.tickers_dir / ticker
            if ticker_dir.exists():
                search_dirs = list(ticker_dir.glob("*/trades"))
            else:
                search_dirs = []
        elif strategy_name:
            search_dirs = list(self.tickers_dir.glob(f"*/{strategy_name.lower()}/trades"))
        else:
            search_dirs = list(self.tickers_dir.glob("*/*/trades"))

        # Load all trade JSON files
        for trades_dir in search_dirs:
            if not trades_dir.exists():
                continue

            for trade_file in trades_dir.glob("trade_*.json"):
                try:
                    trade_log = self.load_trade_log(trade_file)
                    trade_logs.append(trade_log)
                except Exception as e:
                    logger.warning(f"Skipping invalid trade log {trade_file}: {e}")
                    continue

        logger.info(f"Loaded {len(trade_logs)} trade logs")
        return trade_logs

    def get_trade_summary(
        self,
        ticker: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get summary statistics for trades

        Args:
            ticker: Optional ticker filter
            strategy_name: Optional strategy filter

        Returns:
            Summary statistics dictionary
        """
        trades = self.load_all_trades(ticker, strategy_name)

        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
            }

        # Calculate statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.get('performance', {}).get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('performance', {}).get('pnl', 0) <= 0]

        pnls = [t.get('performance', {}).get('pnl', 0) for t in trades]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
        }
