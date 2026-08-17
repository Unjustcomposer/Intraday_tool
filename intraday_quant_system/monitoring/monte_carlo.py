"""
Monte Carlo Stress Tester
=========================

Path-dependent Monte Carlo simulation for trade returns.
Estimates tail risk, ruin probability, and expected shortfall.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo stress test"""
    n_simulations: int
    n_trades: int
    initial_capital: float
    
    # Terminal capital statistics
    median_terminal_capital: float
    mean_terminal_capital: float
    std_terminal_capital: float
    worst_5pct_capital: float
    best_5pct_capital: float
    percentiles: Dict[str, float]
    
    # Risk metrics
    probability_of_profit: float
    probability_of_ruin_50pct: float
    probability_of_ruin_25pct: float
    
    # Drawdown statistics
    median_max_drawdown: float
    mean_max_drawdown: float
    max_drawdown_percentiles: Dict[str, float]
    
    # VaR / CVaR
    var_95: float
    cvar_95: float
    
    # Sharpe ratio of simulated paths
    sharpe_ratio: float
    
    # Path-dependent statistics
    median_trade_duration: float
    median_max_consecutive_losses: float
    
    def to_dict(self) -> Dict:
        return {
            "n_simulations": self.n_simulations,
            "n_trades": self.n_trades,
            "initial_capital": self.initial_capital,
            "median_terminal_capital": self.median_terminal_capital,
            "mean_terminal_capital": self.mean_terminal_capital,
            "std_terminal_capital": self.std_terminal_capital,
            "worst_5pct_capital": self.worst_5pct_capital,
            "best_5pct_capital": self.best_5pct_capital,
            "percentiles": self.percentiles,
            "probability_of_profit": self.probability_of_profit,
            "probability_of_ruin_50pct": self.probability_of_ruin_50pct,
            "probability_of_ruin_25pct": self.probability_of_ruin_25pct,
            "median_max_drawdown": self.median_max_drawdown,
            "mean_max_drawdown": self.mean_max_drawdown,
            "max_drawdown_percentiles": self.max_drawdown_percentiles,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe_ratio": self.sharpe_ratio,
            "median_trade_duration": self.median_trade_duration,
            "median_max_consecutive_losses": self.median_max_consecutive_losses,
        }


class MonteCarloStressTester:
    """
    Monte Carlo stress tester for trade returns.
    
    Performs bootstrap resampling of historical trades to simulate
    alternative equity curves and estimate tail risk.
    
    Features:
    - Path-dependent simulation (preserves trade duration effects)
    - Block bootstrap option for autocorrelated returns
    - Ruin probability estimation
    - VaR / CVaR calculation
    - Drawdown analysis
    """
    
    def __init__(
        self,
        n_simulations: int = 5000,
        confidence_levels: Optional[List[float]] = None,
        initial_capital: float = 1_000_000.0,
        risk_per_trade: float = 0.02,
        block_size: Optional[int] = None,  # For block bootstrap
        random_seed: int = 42,
    ):
        """
        Initialize stress tester.
        
        Args:
            n_simulations: Number of Monte Carlo paths
            confidence_levels: Percentiles to report (default: [0.05, 0.25, 0.5, 0.75, 0.95])
            initial_capital: Starting capital for simulation
            risk_per_trade: Fraction of capital risked per trade (for position sizing)
            block_size: Block size for block bootstrap (None = simple bootstrap)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.05, 0.25, 0.5, 0.75, 0.95]
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.block_size = block_size
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
    
    def run(
        self, 
        trades_df: pd.DataFrame, 
        initial_capital: Optional[float] = None,
        capital_at_risk_pct: Optional[float] = None
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation on historical trades.
        
        Args:
            trades_df: DataFrame with columns ['return', 'duration_bars', 'side']
                       return = net return per trade (decimal, e.g., 0.02 = 2%)
            initial_capital: Override initial capital
            capital_at_risk_pct: Override risk per trade fraction
            
        Returns:
            MonteCarloResult with all stress test statistics
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        if trades_df.empty:
            logger.warning("Empty trades DataFrame provided")
            return self._empty_result()
        
        if "return" not in trades_df.columns:
            logger.error("trades_df must contain 'return' column")
            return self._empty_result()
        
        returns = trades_df["return"].values
        n_trades = len(returns)
        
        if n_trades < 10:
            logger.warning(f"Only {n_trades} trades - results may be unreliable")
        
        # Get trade durations if available
        durations = trades_df.get("duration_bars", pd.Series([1] * n_trades)).values
        sides = trades_df.get("side", pd.Series(["long"] * n_trades)).values
        
        # Run simulations
        terminal_capitals = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        trade_durations = []
        max_consecutive_losses = []
        
        for i in range(self.n_simulations):
            # Bootstrap sample with replacement
            if self.block_size and self.block_size > 1:
                # Block bootstrap for autocorrelated returns
                sampled_returns, sampled_durations, sampled_sides = self._block_bootstrap(
                    returns, durations, sides, self.block_size
                )
            else:
                # Simple bootstrap
                idx = np.random.choice(n_trades, size=n_trades, replace=True)
                sampled_returns = returns[idx]
                sampled_durations = durations[idx]
                sampled_sides = sides[idx]
            
            # Simulate equity curve
            equity = self.initial_capital
            peak = self.initial_capital
            max_dd = 0.0
            consecutive_losses = 0
            max_consecutive_losses = 0
            max_consecutive_losses_list = []
            
            for ret, dur, side in zip(sampled_returns, sampled_durations, sampled_sides):
                equity *= (1 + ret)
                trade_durations.append(dur)
                
                # Track drawdown
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
                
                # Track consecutive losses
                if ret < 0:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            terminal_capitals[i] = equity
            max_drawdowns[i] = max_dd
            max_consecutive_losses_list.append(max_consecutive_losses)
        
        # Calculate statistics
        profit_prob = (terminal_capitals > self.initial_capital).mean()
        ruin_50_prob = (max_drawdowns > 0.5).mean()
        ruin_25_prob = (max_drawdowns > 0.25).mean()
        
        percentiles = np.percentile(terminal_capitals, [c * 100 for c in self.confidence_levels])
        dd_percentiles = np.percentile(max_drawdowns, [c * 100 for c in self.confidence_levels])
        
        # VaR / CVaR
        var_95 = self.initial_capital - np.percentile(terminal_capitals, 5)
        cvar_95 = self.initial_capital - terminal_capitals[terminal_capitals <= np.percentile(terminal_capitals, 5)].mean()
        
        # Sharpe ratio of terminal returns
        terminal_returns = (terminal_capitals - self.initial_capital) / self.initial_capital
        if np.std(terminal_returns) > 0:
            sharpe = np.mean(terminal_returns) / np.std(terminal_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            n_trades=n_trades,
            initial_capital=self.initial_capital,
            median_terminal_capital=float(np.median(terminal_capitals)),
            mean_terminal_capital=float(np.mean(terminal_capitals)),
            std_terminal_capital=float(np.std(terminal_capitals)),
            worst_5pct_capital=float(percentiles[0]),
            best_5pct_capital=float(percentiles[-1]),
            percentiles={f"p{int(c*100)}": float(p) for c, p in zip(self.confidence_levels, percentiles)},
            probability_of_profit=float(profit_prob),
            probability_of_ruin_50pct=float(ruin_50_prob),
            probability_of_ruin_25pct=float(ruin_25_prob),
            median_max_drawdown=float(np.median(max_drawdowns)),
            mean_max_drawdown=float(np.mean(max_drawdowns)),
            max_drawdown_percentiles={f"p{int(c*100)}": float(p) for c, p in zip(self.confidence_levels, dd_percentiles)},
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            sharpe_ratio=float(sharpe),
            median_trade_duration=float(np.median(trade_durations)) if trade_durations else 0.0,
            median_max_consecutive_losses=float(np.median(max_consecutive_losses_list)) if max_consecutive_losses_list else 0.0,
        )
    
    def _block_bootstrap(
        self, 
        returns: np.ndarray, 
        durations: np.ndarray, 
        sides: np.ndarray,
        block_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Block bootstrap for autocorrelated returns.
        Samples blocks of consecutive trades with replacement.
        """
        n = len(returns)
        n_blocks = int(np.ceil(n / block_size))
        
        sampled_returns = []
        sampled_durations = []
        sampled_sides = []
        
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_size + 1)
            end_idx = min(start_idx + block_size, n)
            sampled_returns.extend(returns[start_idx:end_idx])
            sampled_durations.extend(durations[start_idx:end_idx])
            sampled_sides.extend(sides[start_idx:end_idx])
        
        # Truncate to original length
        return (
            np.array(sampled_returns[:n]),
            np.array(sampled_durations[:n]),
            np.array(sampled_sides[:n])
        )
    
    def _empty_result(self) -> MonteCarloResult:
        """Return empty result structure"""
        return MonteCarloResult(
            n_simulations=self.n_simulations,
            n_trades=0,
            initial_capital=self.initial_capital,
            median_terminal_capital=self.initial_capital,
            mean_terminal_capital=self.initial_capital,
            std_terminal_capital=0.0,
            worst_5pct_capital=self.initial_capital,
            best_5pct_capital=self.initial_capital,
            percentiles={f"p{int(c*100)}": self.initial_capital for c in self.confidence_levels},
            probability_of_profit=0.0,
            probability_of_ruin_50pct=1.0,
            probability_of_ruin_25pct=1.0,
            median_max_drawdown=0.0,
            mean_max_drawdown=0.0,
            max_drawdown_percentiles={f"p{int(c*100)}": 0.0 for c in self.confidence_levels},
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            median_trade_duration=0.0,
            median_max_consecutive_losses=0.0,
        )
    
    def simulate_paths(
        self, 
        trades_df: pd.DataFrame, 
        n_paths: int = 100
    ) -> pd.DataFrame:
        """
        Generate equity curve paths for visualization.
        
        Returns:
            DataFrame with columns: path_id, step, equity
        """
        if trades_df.empty or "return" not in trades_df.columns:
            return pd.DataFrame()
        
        returns = trades_df["return"].values
        n_trades = len(returns)
        
        paths = []
        for path_id in range(min(n_paths, self.n_simulations)):
            idx = np.random.choice(n_trades, size=n_trades, replace=True)
            sampled_returns = returns[idx]
            
            equity = self.initial_capital
            for step, ret in enumerate(sampled_returns):
                equity *= (1 + ret)
                paths.append({"path_id": path_id, "step": step, "equity": equity})
        
        return pd.DataFrame(paths)