import logging

import numpy as np

logger = logging.getLogger(__name__)


class StopLossEngine:
    """
    Stop loss calculation engine with regime-aware logic.

    Features:
    - ATR-based initial stops
    - Regime-aware multipliers (wider in volatile regimes)
    - Trailing stops with volatility adaptation
    - Gap risk protection
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        risk_config = config.get("risk", {}) if config else {}

        # Base multipliers
        self.atr_multiplier = risk_config.get("atr_multiplier", 1.5)
        self.trail_multiplier = risk_config.get("trail_multiplier", 1.0)

        # Regime-specific multipliers
        self.regime_multipliers = {
            "quiet": 1.0,  # Normal stops
            "trending": 1.2,  # Wider in trends to avoid whipsaw
            "volatile": 1.5,  # Much wider in volatile markets
            "crisis": 2.0,  # Maximum width in crisis
            "unknown": 1.2,
        }

        # Minimum stop distance (as % of price)
        self.min_stop_pct = risk_config.get("min_stop_pct", 0.005)  # 0.5%
        # Maximum stop distance (as % of price)
        self.max_stop_pct = risk_config.get("max_stop_pct", 0.03)  # 3%

    def calculate_stop(
        self,
        entry_price: float,
        volatility: float,
        regime: str = "unknown",
        is_long: bool = True,
        atr: float = None,
    ) -> float:
        """
        Calculate initial stop loss price.

        Args:
            entry_price: Entry price
            volatility: Annualized volatility (or ATR if atr is provided)
            regime: Current market regime
            is_long: True for long position, False for short
            atr: ATR value (preferred over volatility)

        Returns:
            Stop loss price
        """
        if atr is not None and atr > 0:
            # Use ATR-based stop
            stop_distance = (
                atr * self.atr_multiplier * self.regime_multipliers.get(regime, 1.0)
            )
        elif volatility > 0:
            # Convert annualized volatility to per-bar (assuming 15min bars, ~25/day)
            # For intraday, use per-bar volatility
            per_bar_vol = volatility / np.sqrt(252 * 25)  # Rough conversion
            stop_distance = (
                entry_price
                * per_bar_vol
                * self.atr_multiplier
                * self.regime_multipliers.get(regime, 1.0)
            )
        else:
            # Fallback
            stop_distance = entry_price * 0.01  # 1%

        # Enforce min/max bounds
        min_dist = entry_price * self.min_stop_pct
        max_dist = entry_price * self.max_stop_pct
        stop_distance = max(min_dist, min(stop_distance, max_dist))

        if is_long:
            stop_price = entry_price - stop_distance
        else:
            stop_price = entry_price + stop_distance

        return round(stop_price, 2)

    def calculate_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        volatility: float,
        regime: str = "unknown",
        is_long: bool = True,
        atr: float = None,
    ) -> float:
        """
        Calculate trailing stop - only moves in favorable direction.

        Args:
            current_price: Current market price
            current_stop: Current stop price
            volatility: Annualized volatility
            regime: Current market regime
            is_long: True for long position
            atr: ATR value

        Returns:
            New stop price (only moves favorably)
        """
        if atr is not None and atr > 0:
            trail_distance = (
                atr * self.trail_multiplier * self.regime_multipliers.get(regime, 1.0)
            )
        elif volatility > 0:
            per_bar_vol = volatility / np.sqrt(252 * 25)
            trail_distance = (
                current_price
                * per_bar_vol
                * self.trail_multiplier
                * self.regime_multipliers.get(regime, 1.0)
            )
        else:
            trail_distance = current_price * 0.005

        if is_long:
            # For long: trail stop UP as price rises
            new_stop = current_price - trail_distance
            if new_stop > current_stop:
                return round(new_stop, 2)
        else:
            # For short: trail stop DOWN as price falls
            new_stop = current_price + trail_distance
            if new_stop < current_stop:
                return round(new_stop, 2)

        return current_stop

    def check_gap_risk(
        self, entry_price: float, prev_close: float, is_long: bool
    ) -> bool:
        """
        Check for gap risk at market open.

        Args:
            entry_price: Position entry price
            prev_close: Previous day's close
            is_long: True for long position

        Returns:
            True if gap risk detected (position should be reduced/closed)
        """
        if prev_close <= 0:
            return False

        gap_pct = (entry_price - prev_close) / prev_close

        # Long position with gap down > 2%
        if is_long and gap_pct < -0.02:
            logger.warning(f"Gap down risk detected for LONG: {gap_pct:.2%}")
            return True

        # Short position with gap up > 2%
        if not is_long and gap_pct > 0.02:
            logger.warning(f"Gap up risk detected for SHORT: {gap_pct:.2%}")
            return True

        return False

    def get_stop_info(
        self,
        entry_price: float,
        volatility: float,
        regime: str,
        is_long: bool,
        atr: float = None,
    ) -> dict:
        """Get detailed stop information for logging/debugging."""
        initial_stop = self.calculate_stop(
            entry_price, volatility, regime, is_long, atr
        )
        trail_distance = 0
        if atr is not None and atr > 0:
            trail_distance = (
                atr * self.trail_multiplier * self.regime_multipliers.get(regime, 1.0)
            )
        elif volatility > 0:
            per_bar_vol = volatility / np.sqrt(252 * 25)
            trail_distance = (
                entry_price
                * per_bar_vol
                * self.trail_multiplier
                * self.regime_multipliers.get(regime, 1.0)
            )

        return {
            "initial_stop": initial_stop,
            "trail_distance": round(trail_distance, 2),
            "regime_multiplier": self.regime_multipliers.get(regime, 1.0),
            "atr_multiplier": self.atr_multiplier,
            "trail_multiplier": self.trail_multiplier,
        }


class VolatilityStopEngine(StopLossEngine):
    """Alias for backward compatibility."""

    pass
