"""
Monte Carlo Alpha Stress Test
==============================
Runs 10,000 permutations of a trading system's equity curve based on empirical win rates and risk:reward profiles to calculate risk of ruin and maximum drawdown.
"""
import numpy as np
import pandas as pd
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger("MonteCarlo")

def run_monte_carlo(win_rate: float, avg_win: float, avg_loss: float, n_trades: int = 250, n_simulations: int = 10000, initial_capital: float = 100000.0, risk_per_trade: float = 0.02):
    """
    Run Monte Carlo stress test.
    """
    logger.info(f"Starting Monte Carlo Simulation ({n_simulations} permutations, {n_trades} trades each)")
    logger.info(f"Metrics: Win Rate={win_rate:.1%}, Avg Win={avg_win:.2f}%, Avg Loss={avg_loss:.2f}%")
    
    terminal_capitals = []
    max_drawdowns = []
    ruin_count_50pct = 0
    ruin_count_100pct = 0
    
    start_time = time.time()
    
    for i in range(n_simulations):
        # Generate random array of 0s and 1s based on win rate
        outcomes = np.random.binomial(1, win_rate, n_trades)
        
        # Convert outcomes to percentage returns
        # If win, return is avg_win; if loss, return is avg_loss
        returns = np.where(outcomes == 1, avg_win / 100, avg_loss / 100)
        
        # Calculate equity curve assuming fixed fractional risk (risk_per_trade of current capital)
        # However, to match the simple metrics, we assume flat returns on capital.
        # Let's use simple compounding for equity curve:
        equity_curve = np.cumprod(1 + returns) * initial_capital
        
        terminal_capitals.append(equity_curve[-1])
        
        # Calculate max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = np.max(drawdowns)
        max_drawdowns.append(max_dd)
        
        # Check for ruin
        if np.any(equity_curve <= initial_capital * 0.5):
            ruin_count_50pct += 1
        if np.any(equity_curve <= 0):
            ruin_count_100pct += 1
            
        if (i+1) % 2500 == 0:
            logger.info(f"  Processed {i+1}/{n_simulations} simulations...")
            
    exec_time = time.time() - start_time
    logger.info(f"Simulation completed in {exec_time:.2f} seconds.\n")
    
    # Calculate statistics
    median_terminal = np.median(terminal_capitals)
    worst_5pct = np.percentile(terminal_capitals, 5)
    prob_profit = np.mean(np.array(terminal_capitals) > initial_capital)
    prob_ruin_50 = ruin_count_50pct / n_simulations
    median_dd = np.median(max_drawdowns)
    
    logger.info("="*50)
    logger.info("  MONTE CARLO STRESS TEST RESULTS")
    logger.info("="*50)
    logger.info(f"  Risk of Ruin (50% Loss):    {prob_ruin_50:.2%}")
    logger.info(f"  Risk of Ruin (100% Loss):   {ruin_count_100pct / n_simulations:.2%}")
    logger.info(f"  Probability of Profit:      {prob_profit:.2%}")
    logger.info(f"  Median Maximum Drawdown:    {median_dd:.2%}")
    logger.info(f"  Median Terminal Capital:    ₹{median_terminal:,.2f}")
    logger.info(f"  Worst 5th Percentile Cap:   ₹{worst_5pct:,.2f}")
    logger.info("="*50)
    
    return {
        "prob_ruin_50": prob_ruin_50,
        "prob_profit": prob_profit,
        "median_dd": median_dd,
        "median_terminal": median_terminal
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo Stress Test")
    parser.add_argument("--win-rate", type=float, default=0.442, help="Empirical win rate (0-1)")
    parser.add_argument("--avg-win", type=float, default=1.85, help="Average percentage win")
    parser.add_argument("--avg-loss", type=float, default=-0.92, help="Average percentage loss (negative)")
    parser.add_argument("--sims", type=int, default=10000, help="Number of simulations to run")
    args = parser.parse_args()
    
    run_monte_carlo(
        win_rate=args.win_rate,
        avg_win=args.avg_win,
        avg_loss=args.avg_loss,
        n_simulations=args.sims
    )
