import logging
from datetime import datetime, time

from intraday_quant_system.deployment.config import RiskConfig, get_config

logger = logging.getLogger(__name__)


class RiskMonitor:
    """
    Real-time risk monitoring with hard limits and kill switch.
    Tracks: daily P&L, weekly P&L, max drawdown, position limits, VIX exposure.
    """

    def __init__(self, config: RiskConfig = None, initial_capital: float = None):
        self.config = config or get_config().risk
        self.initial_capital = initial_capital or get_config().max_capital
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital

        # P&L tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.daily_start_capital = self.initial_capital
        self.weekly_start_capital = self.initial_capital
        self.last_day = datetime.now().date()
        self.last_week_start = datetime.now().date()

        # State
        self.trading_halted = False
        self.halt_reason = ""
        self.exposure_multiplier = 1.0
        self.max_drawdown_pct = 0.0

        logger.info(f"RiskMonitor initialized: capital={self.initial_capital:,.0f}")

    def update_capital(self, new_capital: float):
        """Update current capital and recalculate risk metrics."""
        self.current_capital = new_capital

        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        # Calculate drawdown
        if self.peak_capital > 0:
            self.max_drawdown_pct = (self.peak_capital - new_capital) / self.peak_capital

        # Daily P&L
        today = datetime.now().date()
        if today != self.last_day:
            self.daily_start_capital = new_capital
            self.daily_pnl = 0.0
            self.last_day = today
        self.daily_pnl = new_capital - self.daily_start_capital

        # Weekly P&L
        if today != self.last_week_start and (today - self.last_week_start).days >= 7:
            self.weekly_start_capital = new_capital
            self.weekly_pnl = 0.0
            self.last_week_start = today
        self.weekly_pnl = new_capital - self.weekly_start_capital

        # Check limits
        self._check_limits()

    def _check_limits(self):
        """Check all risk limits and trigger kill switch if breached."""
        # Daily loss limit
        daily_loss_pct = -self.daily_pnl / self.daily_start_capital if self.daily_start_capital > 0 else 0
        if daily_loss_pct >= self.config.daily_loss_limit:
            self._halt_trading(f"Daily loss limit breached: {daily_loss_pct:.2%} >= {self.config.daily_loss_limit:.2%}")
            return

        # Weekly loss limit
        weekly_loss_pct = -self.weekly_pnl / self.weekly_start_capital if self.weekly_start_capital > 0 else 0
        if weekly_loss_pct >= self.config.weekly_loss_limit:
            self._halt_trading(f"Weekly loss limit breached: {weekly_loss_pct:.2%} >= {self.config.weekly_loss_limit:.2%}")
            return

        # Max drawdown limit
        if self.max_drawdown_pct >= self.config.max_drawdown_limit:
            self._halt_trading(f"Max drawdown limit breached: {self.max_drawdown_pct:.2%} >= {self.config.max_drawdown_limit:.2%}")
            return

        # Update exposure multiplier based on VIX (will be set externally)
        # This is a placeholder - actual VIX check happens in OrderManager

    def _halt_trading(self, reason: str):
        """Trigger kill switch - halt all new trading."""
        if not self.trading_halted:
            self.trading_halted = True
            self.halt_reason = reason
            self.exposure_multiplier = 0.0
            logger.critical(f"RISK KILL SWITCH ACTIVATED: {reason}")

    def check_vix_exposure(self, vix: float):
        """Update exposure multiplier based on VIX level."""
        if vix >= self.config.vix_cutoff:
            self.exposure_multiplier = 0.5  # Cut exposure by 50%
            logger.warning(f"VIX {vix:.1f} >= {self.config.vix_cutoff}: exposure cut to 50%")
        elif vix >= self.config.vix_cutoff * 0.8:
            self.exposure_multiplier = 0.75
        else:
            self.exposure_multiplier = 1.0

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed. Returns (allowed, reason)."""
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"

        now = datetime.now().time()
        market_open = time(9, 15)
        market_close = time(15, 15)

        if now < market_open or now > market_close:
            return False, "Outside market hours"

        return True, "OK"

    def validate_order(self, symbol: str, quantity: int, price: float, side: str) -> tuple[bool, str]:
        """Validate a single order against risk limits."""
        # Check position size limit
        max_position_value = self.current_capital * self.config.max_risk_per_trade
        order_value = quantity * price
        if order_value > max_position_value:
            return False, f"Order value {order_value:,.0f} exceeds max risk per trade {max_position_value:,.0f}"

        return True, "OK"

    def reset_daily(self):
        """Call at market open to reset daily counters."""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0.0
        self.last_day = datetime.now().date()

    def get_status(self) -> dict:
        """Return current risk status for monitoring."""
        return {
            "current_capital": self.current_capital,
            "peak_capital": self.peak_capital,
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "daily_pnl_pct": self.daily_pnl / self.daily_start_capital if self.daily_start_capital > 0 else 0,
            "weekly_pnl_pct": self.weekly_pnl / self.weekly_start_capital if self.weekly_start_capital > 0 else 0,
            "trading_halted": self.trading_halted,
            "halt_reason": self.halt_reason,
            "exposure_multiplier": self.exposure_multiplier,
        }
