import numpy as np
import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MonteCarloStressTester:
    """
    Monte Carlo analysis by resampling actual historical trades.
    
    Production fix: Replaces fake normally distributed data with 
    actual bootstrap resampling of the backtest trade log.
    """
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        
    def run(self, trades_df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict:
        """
        Run Monte Carlo simulations by resampling the trade history.
        
        Args:
            trades_df: DataFrame containing at least a 'pnl_pct' or 'pnl' column
            initial_capital: Starting capital
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations")
        
        if trades_df.empty:
            logger.warning("Empty trade log provided to Monte Carlo.")
            return {}
            
        # Extract trade returns
        if 'pnl_pct' in trades_df.columns:
            trade_returns = trades_df['pnl_pct'].values
        elif 'pnl' in trades_df.columns:
            trade_returns = (trades_df['pnl'] / initial_capital).values
        else:
            logger.warning("Trade log missing return columns ('pnl_pct' or 'pnl')")
            return {}
            
        n_trades = len(trade_returns)
        if n_trades < 30:
            logger.warning(f"Only {n_trades} trades available. Monte Carlo results will be unreliable.")
            
        # 1. Standard Bootstrap: Resample trades with replacement
        # Shape: (n_simulations, n_trades)
        simulated_returns = np.random.choice(trade_returns, size=(self.n_simulations, n_trades), replace=True)
        
        # Calculate equity curves: cumulative product of (1 + return)
        # Shape: (n_simulations, n_trades)
        equity_curves = np.cumprod(1 + simulated_returns, axis=1) * initial_capital
        
        # Calculate terminal wealth
        terminal_wealth = equity_curves[:, -1]
        
        # Calculate max drawdowns for each simulation
        max_drawdowns = np.zeros(self.n_simulations)
        for i in range(self.n_simulations):
            curve = equity_curves[i]
            running_max = np.maximum.accumulate(curve)
            drawdowns = (running_max - curve) / running_max
            max_drawdowns[i] = np.max(drawdowns)
            
        # Calculate Ruin Probability
        # Defined as hitting a 50% drawdown
        ruin_prob = np.mean(max_drawdowns >= 0.50)
        
        metrics = {
            'median_terminal_capital': float(np.median(terminal_wealth)),
            'worst_5pct_capital': float(np.percentile(terminal_wealth, 5)),
            'best_5pct_capital': float(np.percentile(terminal_wealth, 95)),
            
            'median_max_drawdown': float(np.median(max_drawdowns)),
            'worst_5pct_max_drawdown': float(np.percentile(max_drawdowns, 95)),  # 95th percentile is WORST drawdown
            
            'probability_of_ruin_50pct': float(ruin_prob),
            'probability_of_profit': float(np.mean(terminal_wealth > initial_capital)),
            
            'n_trades_simulated': n_trades
        }
        
        logger.info(f"MC Results: Median Return {(metrics['median_terminal_capital']/initial_capital - 1):.2%}, "
                    f"95% Worst Drawdown {metrics['worst_5pct_max_drawdown']:.2%}")
                    
        return metrics
