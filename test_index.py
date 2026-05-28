import yfinance as yf
import vectorbt as vbt
import pandas as pd
import numpy as np

def run_nifty_backtest():
    symbol = "^NSEI"
    print(f"Fetching historical data for {symbol}...")
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="30d", interval="5m")
    
    if data.empty:
        print("No data fetched. Check connectivity.")
        return

    print(f"Fetched {len(data)} bars of 5-minute data.")
    
    # Calculate EMA indicators
    fast_ema = data['Close'].ewm(span=9, adjust=False).mean()
    slow_ema = data['Close'].ewm(span=21, adjust=False).mean()
    
    # Generate Crossover Signals
    entries = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    exits = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    
    # Programmatic 1-tick penalty on market orders (Nifty tick size is 0.05)
    tick_size = 0.05
    price = data['Close'].copy()
    price[entries] = price[entries] + tick_size
    price[exits] = price[exits] - tick_size
    
    # Run backtest
    print("Running VectorBT backtest with 1-tick execution penalty...")
    portfolio = vbt.Portfolio.from_signals(
        close=data['Close'],
        entries=entries,
        exits=exits,
        price=price,
        init_cash=100000.0,
        fees=0.0003, # 3 bps transaction fee
        freq="5m"
    )
    
    # Print metrics
    print("\n" + "=" * 50)
    print("  VectorBT Backtest Results (NSE Nifty Index)")
    print("  1-Tick Slippage Penalty: 0.05 Index Points")
    print("=" * 50)
    
    stats = portfolio.stats()
    print(stats.to_string())
    print("=" * 50 + "\n")

if __name__ == "__main__":
    run_nifty_backtest()
