"""
Paper Trading Validation Report Generator
=========================================
Reads the paper_trades_log.csv (actual executed entries) and pending_orders.csv 
(target and stop loss levels). Fetches intraday minute data post-execution 
to determine if the trade hit Target or Stop Loss, and calculates PnL.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.config import get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("PaperReport")


def generate_report():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paper_log_path = os.path.join(base_dir, "data", "paper_trades_log.csv")
    pending_orders_path = os.path.join(base_dir, "data", "pending_orders.csv")
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    if not os.path.exists(paper_log_path):
        logger.warning(f"No paper trades log found at {paper_log_path}")
        return
        
    if not os.path.exists(pending_orders_path):
        logger.warning(f"No pending orders found at {pending_orders_path}")
        return

    # Load data
    trades = pd.read_csv(paper_log_path)
    orders = pd.read_csv(pending_orders_path)
    
    if trades.empty:
        logger.info("Paper trades log is empty. No trades executed.")
        return
        
    config = get_config()
    results = []
    
    # Process each executed paper trade
    for _, trade in trades.iterrows():
        symbol = trade['symbol']
        timestamp = pd.to_datetime(trade['timestamp'])
        fill_price = trade['fill_price']
        direction = trade['direction']
        
        # Find corresponding order to get targets
        order_match = orders[(orders['symbol'] == symbol) & (orders['direction'] == direction)]
        if order_match.empty:
            continue
        order = order_match.iloc[-1]
        
        target_1 = float(order['target_1'])
        stop_loss = float(order['stop_loss'])
        
        # Fetch 1-minute data for the rest of the day to simulate outcome
        yf_symbol = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol
        
        try:
            # We fetch today's minute data
            df = yf.download(yf_symbol, start=timestamp.strftime('%Y-%m-%d'), interval="1m", progress=False)
            if df.empty:
                logger.warning(f"Could not fetch minute data for {symbol} to simulate exit.")
                continue
                
            # Filter data strictly after execution time
            # Note: yf index might be timezone aware
            if df.index.tz is not None:
                df.index = df.index.tz_convert('Asia/Kolkata').tz_localize(None)
                
            df_post = df[df.index >= timestamp]
            
            outcome = "OPEN"
            exit_price = fill_price
            exit_time = None
            
            # Simulate tick by tick
            for idx, row in df_post.iterrows():
                high = float(row['High'].iloc[0]) if isinstance(row['High'], pd.Series) else float(row['High'])
                low = float(row['Low'].iloc[0]) if isinstance(row['Low'], pd.Series) else float(row['Low'])
                
                if direction == "BUY":
                    if low <= stop_loss:
                        outcome = "LOSS"
                        exit_price = stop_loss
                        exit_time = idx
                        break
                    elif high >= target_1:
                        outcome = "WIN"
                        exit_price = target_1
                        exit_time = idx
                        break
                else: # SELL
                    if high >= stop_loss:
                        outcome = "LOSS"
                        exit_price = stop_loss
                        exit_time = idx
                        break
                    elif low <= target_1:
                        outcome = "WIN"
                        exit_price = target_1
                        exit_time = idx
                        break
            
            # If still open at end of day, close at market
            if outcome == "OPEN" and not df_post.empty:
                outcome = "EOD_CLOSE"
                exit_price = float(df_post['Close'].iloc[-1].iloc[0] if isinstance(df_post['Close'].iloc[-1], pd.Series) else df_post['Close'].iloc[-1])
                exit_time = df_post.index[-1]
                
            # Calculate PnL
            if direction == "BUY":
                gross_pnl_pct = (exit_price - fill_price) / fill_price
            else:
                gross_pnl_pct = (fill_price - exit_price) / fill_price
                
            # Apply transaction costs (0.12% round trip estimate)
            net_pnl_pct = gross_pnl_pct - 0.0012
            
            results.append({
                'symbol': symbol,
                'direction': direction,
                'entry_time': timestamp,
                'entry_price': fill_price,
                'exit_time': exit_time,
                'exit_price': exit_price,
                'outcome': outcome,
                'gross_pnl_pct': round(gross_pnl_pct * 100, 3),
                'net_pnl_pct': round(net_pnl_pct * 100, 3)
            })
            
        except Exception as e:
            logger.error(f"Error simulating exit for {symbol}: {e}")
            continue

    if not results:
        logger.info("Could not simulate outcomes for any trades.")
        return
        
    results_df = pd.DataFrame(results)
    
    # Calculate aggregate metrics
    total_trades = len(results_df)
    win_trades = len(results_df[results_df['net_pnl_pct'] > 0])
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    total_net_pnl = results_df['net_pnl_pct'].sum()
    max_drawdown = results_df['net_pnl_pct'].min() # Simplistic single-trade max DD
    
    # Generate Markdown Report
    report_date = datetime.now().strftime("%Y-%m-%d")
    report_path = os.path.join(reports_dir, f"paper_trading_summary_{report_date}.md")
    
    with open(report_path, "w") as f:
        f.write(f"# Paper Trading Validation Report: {report_date}\n\n")
        f.write("## Aggregate Metrics\n")
        f.write(f"- **Total Trades:** {total_trades}\n")
        f.write(f"- **Win Rate:** {win_rate*100:.1f}%\n")
        f.write(f"- **Total Net PnL:** {total_net_pnl:.2f}%\n")
        f.write(f"- **Worst Single Trade (Drawdown):** {max_drawdown:.2f}%\n\n")
        
        f.write("## Trade Log\n")
        f.write(results_df.to_markdown(index=False))
        
    logger.info(f"\nReport generated successfully: {report_path}")
    print("\n" + "="*50)
    print(f" PAPER TRADING SUMMARY ({report_date})")
    print("="*50)
    print(f" Total Trades: {total_trades}")
    print(f" Win Rate:     {win_rate*100:.1f}%")
    print(f" Net PnL:      {total_net_pnl:.2f}%")
    print("="*50)


if __name__ == "__main__":
    generate_report()
