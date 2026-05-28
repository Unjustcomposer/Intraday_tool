import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PortfolioRiskMonitor:
    """
    Monitors aggregate portfolio risk and triggers kill switches.
    
    Production notes:
      - Uses independent if-checks (not elif chain) to correctly identify
        all simultaneous breaches.
    """
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Pull from config dict if available, otherwise use defaults
        risk_params = config.get('risk', {})
        self.daily_loss_limit = risk_params.get('daily_loss_limit', 0.03)
        self.weekly_loss_limit = risk_params.get('weekly_loss_limit', 0.06)
        self.max_drawdown = risk_params.get('max_drawdown_limit', 0.10)
        self.vix_cutoff = risk_params.get('vix_cutoff', 25.0)
        
        self.initial_capital = config.get('max_capital', 100000.0)
        self.current_capital = self.initial_capital
        self.high_water_mark = self.initial_capital
        
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        
        # Kill switch status
        self.trading_halted = False
        self.halt_reason = ""
        
        self.exposure_multiplier = 1.0
        
        # Intraday EWMA VaR settings
        self.ewma_alpha = 0.94
        self.ewma_cov = {}  # Map of (sym_a, sym_b) -> covariance_value
        self.latest_intraday_var = 0.0

    def update_pnl(self, realized_pnl: float, unrealized_pnl: float = 0.0):
        total_pnl = realized_pnl + unrealized_pnl
        self.daily_pnl += total_pnl
        self.weekly_pnl += total_pnl
        self.current_capital += total_pnl
        
        if self.current_capital > self.high_water_mark:
            self.high_water_mark = self.current_capital
            
        self._check_kill_switches()

    def update_market_context(self, vix: float):
        if vix > self.vix_cutoff:
            logger.warning(f"VIX ({vix}) above cutoff ({self.vix_cutoff}). Reducing exposure by 50%.")
            self.exposure_multiplier = 0.5
        else:
            self.exposure_multiplier = 1.0
            
        self._check_kill_switches(vix=vix)

    def _check_kill_switches(self, vix: float = None):
        """
        Check all risk limits independently. 
        Multiple breaches can trigger simultaneously.
        """
        daily_loss_pct = abs(self.daily_pnl) / self.initial_capital if self.daily_pnl < 0 else 0
        weekly_loss_pct = abs(self.weekly_pnl) / self.initial_capital if self.weekly_pnl < 0 else 0
        drawdown_pct = (self.high_water_mark - self.current_capital) / self.high_water_mark
        
        breaches = []
        
        if daily_loss_pct >= self.daily_loss_limit:
            breaches.append(f"Daily loss limit breached: {daily_loss_pct:.2%} >= {self.daily_loss_limit:.2%}")
            
        if weekly_loss_pct >= self.weekly_loss_limit:
            breaches.append(f"Weekly loss limit breached: {weekly_loss_pct:.2%} >= {self.weekly_loss_limit:.2%}")
            
        if drawdown_pct >= self.max_drawdown:
            breaches.append(f"Max drawdown breached: {drawdown_pct:.2%} >= {self.max_drawdown:.2%}")
            
        if vix and vix > (self.vix_cutoff * 1.5): # Severe market stress
            breaches.append(f"Extreme VIX level: {vix} > {self.vix_cutoff * 1.5}")
            
        if breaches:
            self.trading_halted = True
            self.halt_reason = " | ".join(breaches)
            logger.critical(f"KILL SWITCH TRIGGERED: {self.halt_reason}")

    def reset_daily(self):
        self.daily_pnl = 0.0
        # If halted due to daily limit, reset. If DD or weekly, stay halted.
        if self.trading_halted and "Daily loss limit" in self.halt_reason and "Max drawdown" not in self.halt_reason and "Weekly" not in self.halt_reason:
            self.trading_halted = False
            self.halt_reason = ""
            logger.info("Daily risk limits reset. Trading resumed.")

    def reset_weekly(self):
        self.weekly_pnl = 0.0
        self.reset_daily()
        if self.trading_halted and "Weekly" in self.halt_reason and "Max drawdown" not in self.halt_reason:
            self.trading_halted = False
            self.halt_reason = ""
            logger.info("Weekly risk limits reset. Trading resumed.")
            
    def get_status(self) -> Dict[str, Any]:
        return {
            'halted': self.trading_halted,
            'reason': self.halt_reason,
            'capital': self.current_capital,
            'drawdown': (self.high_water_mark - self.current_capital) / self.high_water_mark,
            'daily_pnl_pct': self.daily_pnl / self.initial_capital,
            'exposure_mult': self.exposure_multiplier,
            'intraday_var': self.latest_intraday_var
        }

    def update_ewma_cov(self, returns_dict: Dict[str, float]):
        """
        Update EWMA covariance matrix of 5-minute returns.
        returns_dict: dict mapping symbol -> latest 5-minute return (float)
        """
        symbols = list(returns_dict.keys())
        for i in range(len(symbols)):
            sym_a = symbols[i]
            r_a = returns_dict[sym_a]
            
            for j in range(i, len(symbols)):
                sym_b = symbols[j]
                r_b = returns_dict[sym_b]
                
                key = (sym_a, sym_b) if sym_a <= sym_b else (sym_b, sym_a)
                
                # Get previous covariance or initialize with default
                # Initial default variance = 0.0001 (approx 1% standard deviation per bar)
                # Initial default covariance = 0.0
                prev_cov = self.ewma_cov.get(key, 0.0001 if sym_a == sym_b else 0.0)
                
                # EWMA update rule: Cov_t = (1 - alpha) * r_a * r_b + alpha * Cov_{t-1}
                new_cov = (1.0 - self.ewma_alpha) * r_a * r_b + self.ewma_alpha * prev_cov
                self.ewma_cov[key] = new_cov

    def calculate_intraday_var(self, positions_dict: Dict[str, float], confidence_level: float = 0.95) -> float:
        """
        Calculate 5-minute Parametric VaR based on the EWMA covariance matrix.
        positions_dict: dict mapping symbol -> current absolute exposure in currency (INR)
        """
        if not positions_dict or not self.ewma_cov:
            self.latest_intraday_var = 0.0
            return 0.0
            
        symbols = list(positions_dict.keys())
        portfolio_variance = 0.0
        
        for i in range(len(symbols)):
            sym_a = symbols[i]
            exp_a = positions_dict[sym_a]
            
            for j in range(len(symbols)):
                sym_b = symbols[j]
                exp_b = positions_dict[sym_b]
                
                key = (sym_a, sym_b) if sym_a <= sym_b else (sym_b, sym_a)
                cov = self.ewma_cov.get(key, 0.0001 if sym_a == sym_b else 0.0)
                
                # portfolio variance = sum_i sum_j exp_i * exp_j * cov_{i,j}
                portfolio_variance += exp_a * exp_b * cov
                
        if portfolio_variance <= 0:
            self.latest_intraday_var = 0.0
            return 0.0
            
        portfolio_std_dev = np.sqrt(portfolio_variance)
        
        # Z-score for 95% standard normal is 1.645
        z_score = 1.645 if confidence_level == 0.95 else 2.326 # 99%
        
        self.latest_intraday_var = z_score * portfolio_std_dev
        return self.latest_intraday_var

