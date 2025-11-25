"""
Position Monitoring Agent

Runs daily to check open positions for exit triggers:
- Profit targets
- Stop losses
- Greeks risk (Gamma explosion)
- Thesis reversal
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PositionMonitorAgent:
    """
    Daily position monitoring for exit triggers

    This agent should run DAILY (not weekly) to catch:
    1. Profit targets (50% of max profit)
    2. Stop losses (-100% of credit, -50% of debit)
    3. Gamma risk (<7 DTE with high Gamma)
    4. Thesis reversal (sentiment flip)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # ULTRA TIGHT RISK MANAGEMENT: Cut losers IMMEDIATELY
        # Analysis: META lost 49% (-$1351) because stop wasn't tight enough
        # New rule: If trade goes against you 15%, you're probably wrong - GET OUT
        # =====================================================================
        # TIGHTENED RISK MANAGEMENT (Based on 180-day analysis)
        # =====================================================================
        # Problem: Win/Loss ratio = 0.41 (need 0.77 to break even)
        # - Avg Win: $163.85
        # - Avg Loss: $402.24 (2.5x bigger than wins!)
        # Solution: Cut losers FASTER to improve win/loss ratio
        # =====================================================================
        self.profit_target_pct = config.get("profit_target_pct", 0.30)  # 30% profit target (take profits earlier!)
        self.stop_loss_credit_pct = config.get("stop_loss_credit_pct", 0.20)  # 20% stop for credit (was 25%)
        self.stop_loss_debit_pct = config.get("stop_loss_debit_pct", 0.10)  # 10% stop for debit (was 15% - TIGHTER!)
        self.gamma_risk_threshold = config.get("gamma_risk_threshold", 10.0)
        self.min_dte_close = config.get("min_dte_close", 7)  # Close at 7 DTE

    def check_position(
        self,
        position: Dict[str, Any],
        current_market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if position should be closed or adjusted

        Args:
            position: Position dict with entry data, current P&L, Greeks
            current_market_data: Latest OHLCV, options chain, sentiment

        Returns:
            Action dict: {"action": "HOLD|CLOSE|ADJUST", "reason": str}
        """
        # Extract position details
        entry_premium = position.get("entry_premium", 0)
        current_value = position.get("current_value", 0)
        days_to_expiration = position.get("days_to_expiration", 999)
        strategy_type = position.get("strategy_type", "UNKNOWN")  # CREDIT or DEBIT
        position_greeks = position.get("greeks", {})
        entry_thesis = position.get("thesis_direction", "NEUTRAL")

        # Calculate P&L (FIXED: use correct formula)
        # entry_premium is signed: negative for debit (paid), positive for credit (received)
        # current_value is the MTM value (what we'd receive if we close now)
        # P&L = current_value + entry_premium
        # Example debit: entry=-500 (paid), current=600 → pnl = 600 + (-500) = +100
        # Example credit: entry=+300 (received), current=-100 → pnl = -100 + 300 = +200
        pnl = current_value + entry_premium
        pnl_pct = pnl / abs(entry_premium) if entry_premium != 0 else 0

        # ===================================================================
        # CHECK 1: Gamma Risk (HIGHEST PRIORITY - can blow up account)
        # ===================================================================
        gamma = abs(position_greeks.get("net_gamma", 0))

        if days_to_expiration <= self.min_dte_close and gamma > self.gamma_risk_threshold:
            return {
                "action": "CLOSE",
                "reason": f"GAMMA_RISK: {gamma:.2f} Gamma with {days_to_expiration} DTE",
                "urgency": "CRITICAL",
            }

        # ===================================================================
        # CHECK 2: Profit Target
        # ===================================================================
        if pnl_pct >= self.profit_target_pct:
            return {
                "action": "CLOSE",
                "reason": f"PROFIT_TARGET: {pnl_pct:.1%} gain (target: {self.profit_target_pct:.1%})",
                "urgency": "NORMAL",
            }

        # ===================================================================
        # CHECK 2.5: TRAILING STOP (NEW - Protect Winners)
        # ===================================================================
        # If position was up >25% at some point, tighten stop to break-even
        # This prevents winners from turning into losers
        highest_pnl_pct = position.get("highest_pnl_pct", 0)

        # Track highest P&L (caller should update this field)
        if pnl_pct > highest_pnl_pct:
            highest_pnl_pct = pnl_pct
            position["highest_pnl_pct"] = highest_pnl_pct  # Update for next check

        # Trailing stop logic (ORDER MATTERS - check larger gains first)
        if highest_pnl_pct >= 0.50:  # Was up 50%+ at some point
            # Lock in at least 60% of the gains (very profitable trade)
            if pnl_pct <= highest_pnl_pct * 0.6:
                return {
                    "action": "CLOSE",
                    "reason": f"TRAILING_STOP: Was up {highest_pnl_pct:.1%}, now at {pnl_pct:.1%}. Locking in 60% of gains.",
                    "urgency": "HIGH",
                }
        elif highest_pnl_pct >= 0.30:  # Was up 30%+ at some point
            # Lock in at least half the gains
            if pnl_pct <= highest_pnl_pct * 0.5:
                return {
                    "action": "CLOSE",
                    "reason": f"TRAILING_STOP: Was up {highest_pnl_pct:.1%}, now at {pnl_pct:.1%}. Taking profits.",
                    "urgency": "NORMAL",
                }
        elif highest_pnl_pct >= 0.20:  # Was up 20%+ at some point
            # Don't let winners become losers
            if pnl_pct <= 0.05:  # Allow small buffer (5%)
                return {
                    "action": "CLOSE",
                    "reason": f"TRAILING_STOP: Was up {highest_pnl_pct:.1%}, now at {pnl_pct:.1%}. Protecting gains.",
                    "urgency": "HIGH",
                }

        # ===================================================================
        # CHECK 3: Stop Loss (Differentiated by strategy type)
        # ===================================================================
        if strategy_type == "CREDIT":
            # Credit spread: received premium, risk is spread width
            # Stop at -100% of credit (lost all premium)
            if pnl_pct <= -self.stop_loss_credit_pct:
                return {
                    "action": "CLOSE",
                    "reason": f"STOP_LOSS: {pnl_pct:.1%} loss (credit strategy)",
                    "urgency": "HIGH",
                }
        else:  # DEBIT
            # Debit spread: paid premium, want tight stop
            # Stop at -50% of debit (half the premium gone)
            if pnl_pct <= -self.stop_loss_debit_pct:
                return {
                    "action": "CLOSE",
                    "reason": f"STOP_LOSS: {pnl_pct:.1%} loss (debit strategy)",
                    "urgency": "HIGH",
                }

        # ===================================================================
        # CHECK 4: Time-Based Close (Avoid last week Gamma risk)
        # ===================================================================
        if days_to_expiration <= self.min_dte_close:
            return {
                "action": "CLOSE",
                "reason": f"TIME_EXIT: {days_to_expiration} DTE (close before Gamma risk)",
                "urgency": "NORMAL",
            }

        # ===================================================================
        # CHECK 5: Thesis Reversal (Sentiment flip)
        # ===================================================================
        current_sentiment = current_market_data.get("sentiment_score", 0)

        # If entry was bullish but now strongly bearish (or vice versa)
        if entry_thesis == "Bullish" and current_sentiment < -0.5:
            return {
                "action": "CLOSE",
                "reason": f"THESIS_REVERSAL: Entry bullish, now bearish ({current_sentiment:.2f})",
                "urgency": "HIGH",
            }
        elif entry_thesis == "Bearish" and current_sentiment > 0.5:
            return {
                "action": "CLOSE",
                "reason": f"THESIS_REVERSAL: Entry bearish, now bullish ({current_sentiment:.2f})",
                "urgency": "HIGH",
            }

        # ===================================================================
        # CHECK 6: Theta Burn Warning (Debit spreads bleeding)
        # ===================================================================
        theta = position_greeks.get("net_theta", 0)
        if strategy_type == "DEBIT" and theta < -50:
            return {
                "action": "REVIEW",
                "reason": f"HIGH_THETA_BURN: ${abs(theta)*100:.2f}/day decay",
                "urgency": "NORMAL",
            }

        # Default: Hold position
        return {
            "action": "HOLD",
            "reason": f"Position within parameters (P&L: {pnl_pct:+.1%}, DTE: {days_to_expiration})",
            "urgency": "NONE",
        }

    def monitor_portfolio(
        self,
        positions: List[Dict[str, Any]],
        current_market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check all open positions and return actions

        Args:
            positions: List of open positions
            current_market_data: Current market data for all tickers

        Returns:
            List of actions to take
        """
        actions = []

        for position in positions:
            ticker = position.get("ticker")
            ticker_data = current_market_data.get(ticker, {})

            action = self.check_position(position, ticker_data)

            if action["action"] != "HOLD":
                actions.append({
                    "ticker": ticker,
                    "position_id": position.get("id"),
                    **action
                })
                logger.info(
                    f"[{ticker}] {action['action']}: {action['reason']} "
                    f"(urgency: {action['urgency']})"
                )

        return actions
