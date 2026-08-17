"""
Go/No-Go Decision Framework
===========================
Evaluates the cumulative paper trading log against the strict thresholds
defined in config.py (PaperTradingValidationConfig). Determines if the system 
is mathematically ready for LIVE capital deployment.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.config import get_config
from scripts.generate_paper_report import generate_report # We can reuse the logic if we adapt it, but simpler to just re-compute

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("GoNoGo")

def evaluate_readiness():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paper_log_path = os.path.join(base_dir, "data", "paper_trades_log.csv")
    
    config = get_config()
    val_cfg = config.paper_trading
    
    print("\n" + "═"*60)
    print(" 🚦 GO / NO-GO DECISION FRAMEWORK")
    print("═"*60)
    
    if not os.path.exists(paper_log_path):
        print("[FAIL] Paper trades log not found. Have you run paper trading yet?")
        return
        
    trades = pd.read_csv(paper_log_path)
    
    if trades.empty:
        print("[FAIL] Zero paper trades executed.")
        return
        
    # In a real implementation, we would parse the exact PnL from the report generator.
    # For now, we will do a rough estimation based on simulated outcomes, or we can just 
    # look at the number of unique days traded for the Go/No-Go.
    
    trades['date'] = pd.to_datetime(trades['timestamp']).dt.date
    unique_days = trades['date'].nunique()
    
    print(f"📊 Evaluated {len(trades)} trades over {unique_days} trading days.")
    
    # Check 1: Minimum Days
    if unique_days < val_cfg.min_paper_trading_days:
        print(f"❌ [FAIL] Insufficient Paper Trading Days: {unique_days} / {val_cfg.min_paper_trading_days}")
        passed_days = False
    else:
        print(f"✅ [PASS] Minimum Trading Days: {unique_days} >= {val_cfg.min_paper_trading_days}")
        passed_days = True
        
    # Because we don't have the full post-trade tick data here (it's in the report generator),
    # we will require the user to run the report generator first, or we assume they have.
    # For the sake of the framework, we will mock the PnL check based on the fill prices
    # just as a structural demonstration, assuming 50% hit target and 50% hit SL for the mock.
    
    # To truly do this, `generate_paper_report.py` should save a `cumulative_results.csv`.
    # Let's check if there is a cumulative report we can read.
    # If not, we will just print the structural requirements.
    
    print("\n⚠️ Note: Full PnL Go/No-Go requires cumulative tick simulation.")
    print(f"Target Thresholds for LIVE Deployment:")
    print(f" - Min Win Rate: {val_cfg.min_win_rate * 100}%")
    print(f" - Min Net Profit Factor: {val_cfg.min_net_profit_factor}")
    print(f" - Max Drawdown: {val_cfg.max_paper_drawdown * 100}%\n")
    
    if passed_days:
        print("💡 STATUS: IN PROGRESS (Awaiting PnL validation)")
    else:
        print("💡 STATUS: NO-GO (Insufficient data)")
        
    print("═"*60 + "\n")

if __name__ == "__main__":
    evaluate_readiness()
