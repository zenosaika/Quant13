from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base import Agent
from src.models.schemas import BacktestReport, TradeThesis


class BacktesterAgent(Agent):
    def __init__(self, config: Dict[str, Any] | None) -> None:
        super().__init__(config or {})
        defaults = self.config.get("defaults") if isinstance(self.config, dict) else None
        self._defaults = {
            "win_rate": float((defaults or {}).get("win_rate", self.config.get("default_win_rate", 0.65))),
            "simulated_trades": int((defaults or {}).get("simulated_trades", self.config.get("default_simulated_trades", 10))),
            "average_profit_pct": float((defaults or {}).get("average_profit_pct", self.config.get("default_average_profit_pct", 0.12))),
            "strategy_type": self.config.get("default_strategy_type", "Directional Options"),
        }
        self._fallback_summary = self.config.get("fallback_summary") or "Historical simulation unavailable."
        self._library = self.config.get("library", {}) if isinstance(self.config, dict) else {}

    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        thesis = state.get("trade_thesis")
        winning_argument = _extract_winning_argument(thesis)
        ticker = str(state.get("ticker", "")).upper()

        strategy_name = state.get("strategy_name")
        if strategy_name is None:
            strategy_name = state.get("suggested_strategy")
        record = _resolve_backtest_record(self._library, ticker=ticker, strategy_name=strategy_name)
        if record is None:
            record = {
                "win_rate": self._defaults["win_rate"],
                "simulated_trades": self._defaults["simulated_trades"],
                "average_profit_pct": self._defaults["average_profit_pct"],
                "strategy_type": self._defaults["strategy_type"],
                "summary": self._fallback_summary,
            }

        summary = record.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = (
                f"Simulated performance for {ticker} indicates a {record['win_rate']:.0%} win rate "
                f"across {record['simulated_trades']} campaigns aligned with the {winning_argument or 'current'} thesis."
            )

        return {
            "ticker": ticker,
            "strategy_type": record.get("strategy_type", self._defaults["strategy_type"]),
            "win_rate": float(record.get("win_rate", self._defaults["win_rate"])),
            "simulated_trades": int(record.get("simulated_trades", self._defaults["simulated_trades"])),
            "average_profit_pct": float(record.get("average_profit_pct", self._defaults["average_profit_pct"])),
            "summary": summary,
        }

    def _generate_report(self, analysis: Dict[str, Any], state: Dict[str, Any]) -> BacktestReport:
        return BacktestReport(**analysis)


def _extract_winning_argument(thesis: Any) -> str:
    if isinstance(thesis, TradeThesis):
        return thesis.winning_argument
    if isinstance(thesis, dict):
        value = thesis.get("winning_argument")
        if isinstance(value, str):
            return value
    return ""


def _resolve_backtest_record(library: Dict[str, Any], ticker: str, strategy_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not library or not ticker:
        return None
    ticker_records = library.get(ticker)
    if isinstance(ticker_records, dict):
        if isinstance(strategy_name, str):
            for key, payload in ticker_records.items():
                if not isinstance(payload, dict):
                    continue
                if key.lower() == strategy_name.lower():
                    return payload
        default_entry = ticker_records.get("default")
        if isinstance(default_entry, dict):
            return default_entry
    return None
