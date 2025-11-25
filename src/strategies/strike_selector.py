"""
Strike Selection Engine

Systematic strike selection based on delta targets, probability, and Greeks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import logging

from src.strategies.strategy_library import StrategyBlueprint, LegTemplate

logger = logging.getLogger(__name__)


class StrikeSelector:
    """
    Systematic strike selection engine

    Selects optimal strikes for a strategy based on:
    - Delta targets
    - Distance from spot
    - Probability of profit
    - Greeks characteristics
    - Liquidity (Open Interest, Volume)
    """

    def filter_liquid_options(
        self,
        options_df: pd.DataFrame,
        min_open_interest: int = 100,
        min_volume: int = 50,
        min_bid: float = 0.05
    ) -> pd.DataFrame:
        """
        Filter out illiquid options before strike selection

        Args:
            options_df: Options dataframe
            min_open_interest: Minimum OI to consider liquid
            min_volume: Minimum daily volume
            min_bid: Minimum bid price (avoid pennies)

        Returns:
            Filtered dataframe with only liquid options
        """
        # Check if liquidity columns exist
        has_oi = 'openInterest' in options_df.columns
        has_volume = 'volume' in options_df.columns

        if not has_oi and not has_volume:
            logger.warning("No liquidity data available (OI/Volume missing)")
            return options_df

        # Build filter conditions
        conditions = []

        if has_oi:
            # Handle NaN values
            oi_values = options_df['openInterest'].fillna(0)
            conditions.append(oi_values >= min_open_interest)

        if has_volume:
            vol_values = options_df['volume'].fillna(0)
            conditions.append(vol_values >= min_volume)

        # Bid price filter
        if 'bid' in options_df.columns:
            conditions.append(options_df['bid'] >= min_bid)

        # Combine conditions
        if conditions:
            combined = conditions[0]
            for condition in conditions[1:]:
                combined &= condition

            filtered = options_df[combined].copy()

            logger.info(
                f"Liquidity filter: {len(options_df)} → {len(filtered)} options "
                f"(OI≥{min_open_interest}, Vol≥{min_volume})"
            )

            return filtered

        return options_df

    def select_strikes_for_strategy(
        self,
        strategy: StrategyBlueprint,
        options_chain_flat: pd.DataFrame,
        spot_price: float,
        target_dte: int = 35
    ) -> List[Dict[str, Any]]:
        """
        Select optimal strikes for all legs of a strategy (enhanced with liquidity filtering)

        Args:
            strategy: Strategy blueprint
            options_chain_flat: Flattened options chain DataFrame
            spot_price: Current underlying price
            target_dte: Target days to expiration

        Returns:
            List of selected legs with specific strikes and contracts
        """
        # CRITICAL FIX: Ensure all legs use the same expiration
        # Select target expiration first, then filter options to that expiration only
        target_expiration = self._select_target_expiration(options_chain_flat, target_dte)

        if target_expiration is None:
            logger.error("No valid expiration found in options chain")
            return []

        # Filter to only the target expiration
        expiration_filtered = options_chain_flat[
            options_chain_flat["expiration"] == target_expiration
        ].copy()

        if expiration_filtered.empty:
            logger.error(f"No options found for expiration {target_expiration}")
            return []

        # NEW: Filter by liquidity BEFORE strike selection
        liquid_options = self.filter_liquid_options(
            expiration_filtered,
            min_open_interest=100,
            min_volume=50,
            min_bid=0.05
        )

        if liquid_options.empty:
            logger.warning("No liquid options found, using all options")
            liquid_options = expiration_filtered

        logger.info(f"Selected expiration {target_expiration} for all legs")

        selected_legs = []

        for leg_template in strategy.legs:
            try:
                selected = self._select_leg_strike(
                    leg_template,
                    liquid_options,  # Use liquidity-filtered dataframe
                    spot_price,
                    strategy.strike_selection_rules
                )

                if selected is not None:
                    selected_legs.append(selected)
                else:
                    logger.warning(
                        f"Could not select strike for {leg_template.action} "
                        f"{leg_template.option_type} {leg_template.strike_selection}"
                    )

            except Exception as e:
                logger.error(f"Error selecting strike for leg: {e}")
                continue

        return selected_legs

    def _select_target_expiration(
        self,
        options_chain_flat: pd.DataFrame,
        target_dte: int = 35
    ) -> Optional[str]:
        """
        Select the expiration date closest to target DTE

        Args:
            options_chain_flat: Flattened options chain
            target_dte: Target days to expiration

        Returns:
            Expiration date string (YYYY-MM-DD) or None
        """
        if options_chain_flat.empty:
            return None

        # Get unique expirations
        expirations = options_chain_flat["expiration"].unique()

        if len(expirations) == 0:
            return None

        # If only one expiration, use it
        if len(expirations) == 1:
            return expirations[0]

        # Calculate DTE for each expiration
        from datetime import datetime, timedelta

        today = datetime.now().date()
        expiration_dtes = {}

        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                expiration_dtes[exp_str] = dte
            except Exception as e:
                logger.warning(f"Could not parse expiration {exp_str}: {e}")
                continue

        if not expiration_dtes:
            return None

        # Select expiration closest to target_dte
        closest_exp = min(
            expiration_dtes.items(),
            key=lambda x: abs(x[1] - target_dte)
        )

        logger.info(
            f"Selected expiration {closest_exp[0]} (DTE: {closest_exp[1]}) "
            f"closest to target DTE {target_dte}"
        )

        return closest_exp[0]

    def _select_leg_strike(
        self,
        leg_template: LegTemplate,
        options_df: pd.DataFrame,
        spot_price: float,
        rules: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Select strike for a single leg

        Args:
            leg_template: Leg specification
            options_df: Options chain dataframe
            spot_price: Current price
            rules: Strategy-specific selection rules

        Returns:
            Dict with leg details including selected strike
        """
        # Filter to correct option type
        side_df = options_df[options_df["type"] == leg_template.option_type].copy()

        if side_df.empty:
            return None

        # Select based on strike_selection rule
        strike_rule = leg_template.strike_selection

        if strike_rule == "atm":
            selected_row = self._select_atm(side_df, spot_price)

        elif strike_rule == "itm":
            selected_row = self._select_itm(side_df, spot_price, leg_template.option_type)

        elif strike_rule == "slightly_otm":
            # Use delta if available in rules
            if leg_template.option_type == "CALL":
                delta_range = rules.get("long_strike_delta", (0.60, 0.70))
            else:  # PUT
                delta_range = rules.get("long_strike_delta", (-0.70, -0.60))

            selected_row = self._select_by_delta(side_df, leg_template.option_type, delta_range)

        elif strike_rule == "otm":
            if leg_template.option_type == "CALL":
                delta_range = rules.get("short_strike_delta", (0.30, 0.40))
            else:
                delta_range = rules.get("short_strike_delta", (-0.40, -0.30))

            selected_row = self._select_by_delta(side_df, leg_template.option_type, delta_range)

        elif strike_rule == "far_otm":
            if leg_template.option_type == "CALL":
                delta_range = (0.10, 0.20)
            else:
                delta_range = (-0.20, -0.10)

            selected_row = self._select_by_delta(side_df, leg_template.option_type, delta_range)

        else:
            # Default to ATM
            logger.warning(f"Unknown strike rule '{strike_rule}', using ATM")
            selected_row = self._select_atm(side_df, spot_price)

        if selected_row is None:
            return None

        # Build leg dict - use bracket notation for pandas Series
        # Access computed_greeks correctly from pandas Series
        computed_greeks = selected_row["computed_greeks"] if "computed_greeks" in selected_row.index else None

        leg_dict = {
            "contract_symbol": selected_row.get("contractSymbol", ""),
            "type": leg_template.option_type,
            "action": leg_template.action,
            "strike_price": float(selected_row["strike"]),
            "expiration_date": selected_row.get("expiration", ""),
            "quantity": leg_template.quantity,
            "key_greeks_at_selection": {}
        }

        # Extract Greeks from computed_greeks or fallback to yfinance
        if isinstance(computed_greeks, dict):
            # Use computed Greeks (from Black-Scholes)
            leg_dict["key_greeks_at_selection"] = {
                "delta": computed_greeks.get("delta"),
                "gamma": computed_greeks.get("gamma"),
                "theta": computed_greeks.get("theta"),
                "vega": computed_greeks.get("vega"),
                "impliedVolatility": computed_greeks.get("iv_used")
            }
        else:
            # Fallback to yfinance Greeks (usually not available)
            leg_dict["key_greeks_at_selection"] = {
                "delta": selected_row.get("delta"),
                "gamma": selected_row.get("gamma"),
                "theta": selected_row.get("theta"),
                "vega": selected_row.get("vega"),
                "impliedVolatility": selected_row.get("impliedVolatility")
            }

        return leg_dict

    def _select_atm(self, df: pd.DataFrame, spot_price: float) -> Optional[pd.Series]:
        """Select strike closest to spot price (ATM)"""
        if df.empty:
            return None

        df = df.copy()
        df["distance"] = (df["strike"] - spot_price).abs()
        return df.nsmallest(1, "distance").iloc[0]

    def _select_itm(
        self,
        df: pd.DataFrame,
        spot_price: float,
        option_type: str
    ) -> Optional[pd.Series]:
        """
        Select in-the-money strike

        For calls: strike < spot
        For puts: strike > spot
        """
        if df.empty:
            return None

        df = df.copy()

        if option_type == "CALL":
            # ITM calls: strike < spot
            itm_df = df[df["strike"] < spot_price]
            if itm_df.empty:
                # Fallback to ATM
                return self._select_atm(df, spot_price)

            # Select strike close to spot but ITM (delta ~0.60-0.70)
            itm_df["distance"] = (spot_price - itm_df["strike"]).abs()
            return itm_df.nsmallest(1, "distance").iloc[0]

        else:  # PUT
            # ITM puts: strike > spot
            itm_df = df[df["strike"] > spot_price]
            if itm_df.empty:
                return self._select_atm(df, spot_price)

            itm_df["distance"] = (itm_df["strike"] - spot_price).abs()
            return itm_df.nsmallest(1, "distance").iloc[0]

    def _select_by_delta(
        self,
        df: pd.DataFrame,
        option_type: str,
        delta_range: Tuple[float, float]
    ) -> Optional[pd.Series]:
        """
        Select strike by target delta range

        Uses computed_greeks if available, falls back to yfinance delta

        Args:
            df: Options dataframe
            option_type: "CALL" or "PUT"
            delta_range: (min_delta, max_delta) tuple

        Returns:
            Selected option row
        """
        if df.empty:
            return None

        df = df.copy()

        # Extract delta from computed_greeks or fallback
        def extract_delta(row):
            computed = row.get("computed_greeks")
            if isinstance(computed, dict) and "delta" in computed:
                return computed["delta"]
            # Fallback to yfinance delta
            return row.get("delta", 0.0)

        df["abs_delta"] = df.apply(lambda row: abs(extract_delta(row)), axis=1)

        # Filter to delta range
        min_delta, max_delta = delta_range
        min_abs = min(abs(min_delta), abs(max_delta))
        max_abs = max(abs(min_delta), abs(max_delta))

        in_range = df[(df["abs_delta"] >= min_abs) & (df["abs_delta"] <= max_abs)].copy()

        if in_range.empty:
            # Fallback: select closest to target delta
            target_delta = (min_delta + max_delta) / 2.0
            target_abs = abs(target_delta)

            df = df.copy()
            df["delta_diff"] = (df["abs_delta"] - target_abs).abs()
            return df.nsmallest(1, "delta_diff").iloc[0]

        # Select middle of range
        target_delta = (min_delta + max_delta) / 2.0
        target_abs = abs(target_delta)

        in_range["delta_diff"] = (in_range["abs_delta"] - target_abs).abs()
        return in_range.nsmallest(1, "delta_diff").iloc[0]


def flatten_options_chain(options_chain: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten enriched options chain into single DataFrame

    Args:
        options_chain: Enriched options chain from greeks_engine

    Returns:
        DataFrame with all options (calls and puts) from all expirations
    """
    all_options = []

    for expiration_group in options_chain:
        expiration_date = expiration_group.get("expiration")

        # Process calls
        calls_df = expiration_group.get("calls")
        if calls_df is not None and not calls_df.empty:
            calls_df = calls_df.copy()
            calls_df["type"] = "CALL"
            calls_df["expiration"] = expiration_date
            all_options.append(calls_df)

        # Process puts
        puts_df = expiration_group.get("puts")
        if puts_df is not None and not puts_df.empty:
            puts_df = puts_df.copy()
            puts_df["type"] = "PUT"
            puts_df["expiration"] = expiration_date
            all_options.append(puts_df)

    if all_options:
        return pd.concat(all_options, ignore_index=True)
    else:
        return pd.DataFrame()
