"""
End-to-End Backtest Runner (Microstructure Alpha Edition)
=========================================================
Connects the new Microstructure Alpha Model and Queue Simulator 
to produce a validated performance report on 1-minute data.

Usage:
    python -m scripts.run_full_backtest --symbols RELIANCE.NS HDFCBANK.NS --days 60
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import mlflow

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # adds intraday_quant_system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) # adds intraday

from deployment.config import TransactionCosts, get_config
from data.market_data import MarketDataEngine

from models.microstructure_alpha import MicrostructureAlphaModel
from execution.queue_simulator import QueuePositionSimulator

from intraday_quant_system.backtesting.dsr_metrics import calculate_dsr as DeflatedSharpeRatio
from intraday_quant_system.backtesting.monte_carlo import MonteCarloStressTester
from intraday_quant_system.backtesting.walk_forward import PurgedWalkForwardValidator as WalkForwardValidator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger("FullBacktest")

# ─── DATA ────────────────────────────────────────────────────────────────

def fetch_data(symbols: list, days: int) -> dict:
    """Fetch historical 1-minute data for all symbols"""
    engine = MarketDataEngine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching {days} days of 1-minute data for {symbol}...")
        df = engine.fetch_intraday_data(
            symbol, start_date, end_date, interval="1minute"
        )
        if df is not None and not df.empty:
            validation = engine.validate_data(df, symbol)
            logger.info(
                f"  {symbol}: {len(df)} bars, valid={validation['valid']}, "
                f"issues={len(validation['issues'])}"
            )
            all_data[symbol] = df
        else:
            logger.warning(f"  {symbol}: No data returned")

    return all_data

# ─── WALK-FORWARD WITH MICROSTRUCTURE COMPONENTS ─────────────────────────

def run_walk_forward(symbol: str, df: pd.DataFrame, config: dict) -> dict:
    """
    Walk-forward validation using:
    MicrostructureAlphaModel -> QueuePositionSimulator -> P&L
    """
    tc = TransactionCosts()
    round_trip_cost = tc.total_round_trip_pct()
    
    # Instantiate models
    alpha_model = MicrostructureAlphaModel(vpin_window=50, tsc_window=20)
    
    # Pre-compute scores for the whole series for efficiency
    df = df.copy()
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
        
    # Generate scores
    df['alpha_score'] = alpha_model.generate_scores(df)

    def train_and_evaluate(train_df, val_df):
        """
        Microstructure is purely statistical, so 'train' is mostly irrelevant,
        but we still evaluate strictly out-of-sample on 'val_df'.
        """
        if len(val_df) < 50:
            return _empty_result()
            
        # We process the validation period bar-by-bar
        trades = []
        position = 0
        entry_price = 0.0
        entry_bar = 0
        adverse_selections = []
        
        closes = val_df["close"].values
        opens = val_df["open"].values
        highs = val_df["high"].values
        lows = val_df["low"].values
        volumes = val_df["volume"].values
        scores = val_df["alpha_score"].values
        
        # Calculate proxy spread based on volatility
        atrs = (pd.Series(highs) - pd.Series(lows)).rolling(14).mean().bfill().values
        med_vol = np.nanmedian(volumes) if len(volumes) > 0 else 1.0
        vol_ratio = np.clip(med_vol / np.maximum(volumes, 1.0), 0.5, 3.0)
        spreads = atrs * vol_ratio * 0.1
        
        queue_sim = QueuePositionSimulator(queue_ahead_fraction=0.5)
        
        # To calculate adverse selection (M2M after 5 bars)
        post_fill_trackers = []
        
        last_exit_bar = -1
        for i in range(len(val_df)):
            current_open = opens[i]
            current_close = closes[i]
            current_vol = volumes[i]
            current_spread = spreads[i]
            current_score = scores[i]
            
            # Record M2M for adverse selection
            to_remove = []
            for tracker in post_fill_trackers:
                tracker["elapsed"] += 1
                if tracker["elapsed"] == 5: # 5 bars post fill
                    m2m_pnl = tracker["side_mult"] * (current_close - tracker["fill_price"]) / tracker["fill_price"]
                    adverse_selections.append(m2m_pnl)
                    to_remove.append(tracker)
            for r in to_remove:
                post_fill_trackers.remove(r)
                
            # Process Execution Queue for active limit orders
            fill_event = queue_sim.update(val_df.iloc[i])
            if fill_event:
                if position == 0:
                    # Entry fill
                    position = 1 if fill_event["side"] == "buy" else -1
                    entry_price = fill_event["fill_price"]

                    # 2.1 Perpetual Limit Order Fix: Apply spread/2 slippage for timeout market orders
                    if fill_event.get("fill_type") == "timeout_market":
                        if position == 1:
                            entry_price += (current_spread / 2)
                        else:
                            entry_price -= (current_spread / 2)

                    entry_bar = i
                    
                    # Track adverse selection
                    post_fill_trackers.append({
                        "fill_price": entry_price,
                        "side_mult": position,
                        "elapsed": 0
                    })
                else:
                    # Exit fill
                    exit_price = fill_event["fill_price"]
                    # 2.1 Apply spread/2 slippage on exit timeout market orders
                    if fill_event.get("fill_type") == "timeout_market":
                        if position == 1:
                            exit_price -= (current_spread / 2)
                        else:
                            exit_price += (current_spread / 2)

                    trade_return = position * (exit_price - entry_price) / entry_price
                    flat_fee_pct = 40.0 / 100000.0
                    trade_return -= round_trip_cost + flat_fee_pct
                    
                    trades.append({
                        "return": trade_return,
                        "side": "long" if position == 1 else "short",
                        "duration_bars": i - entry_bar,
                        "fill_type": fill_event.get("fill_type", "unknown")
                    })
                    position = 0
                    entry_price = 0.0
                    last_exit_bar = i

            # Signal Generation (at the close)
            # Only if we don't have an active order
            if not queue_sim.active_order:
                signal = alpha_model.get_signal(current_score, dynamic_threshold=0.1)
                
                if position != 0:
                    # Exit logic
                    exit_now = False
                    current_atr = atrs[i]
                    unrealized_pnl = position * (current_close - entry_price)
                    
                    # Take-profit or dynamic stop-loss
                    if unrealized_pnl > (2.0 * current_atr):
                        exit_now = True
                    elif unrealized_pnl < (-1.0 * max(current_atr, current_spread * 2)):
                        exit_now = True
                    elif (position == 1 and signal == "sell") or (position == -1 and signal == "buy"):
                        exit_now = True
                        
                    if exit_now:
                        # Place limit order to exit at current close
                        exit_side = "sell" if position == 1 else "buy"
                        queue_sim.place_order(exit_side, current_close, 100, current_vol)
                else:
                    # Entry logic
                    # 2.2 Wash-Trade Churn Fix: Strict 15-bar (15m) cooldown after exit
                    if signal in ("buy", "sell") and i > last_exit_bar + 15:
                        queue_sim.place_order(signal, current_close, 100, current_vol)
                        
        # Force close at end
        if position != 0:
            final_price = closes[-1]
            final_price -= (spreads[-1] / 2) if position == 1 else -(spreads[-1] / 2)
            trade_return = position * (final_price - entry_price) / entry_price
            trade_return -= round_trip_cost + (40.0 / 100000.0)
            trades.append({
                "return": trade_return,
                "side": "long" if position == 1 else "short",
                "duration_bars": len(closes) - entry_bar,
            })

        metrics = _compute_metrics(trades)
        
        # Log mean adverse selection
        mean_adv = np.mean(adverse_selections) if adverse_selections else 0.0
        metrics["adverse_selection"] = float(mean_adv)
        
        return metrics

    wf = WalkForwardValidator(
        n_splits=5,
        train_size=0.7,
        val_size=0.15,
        purge_bars=24,
        embargo_bars=24,
    )

    results = wf.run(df, train_and_evaluate)
    aggregated = wf.aggregate_results(results)

    return {
        "symbol": symbol,
        "n_splits": len(results),
        "individual_results": results,
        "aggregated": aggregated,
    }


def _empty_result():
    return {
        "sharpe": 0, "win_rate": 0, "trade_count": 0, "net_return": 0, 
        "profit_factor": 0, "avg_duration": 0, "long_count": 0, 
        "short_count": 0, "max_drawdown": 0, "adverse_selection": 0
    }


def _compute_metrics(trades: list) -> dict:
    """Compute comprehensive trade-level metrics"""
    if not trades:
        return _empty_result()

    returns = np.array([t["return"] for t in trades])
    durations = np.array([t["duration_bars"] for t in trades])

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 1
    profit_factor = (
        (np.sum(wins) / np.sum(np.abs(losses)))
        if len(losses) > 0 and np.sum(np.abs(losses)) > 0 else 0
    )

    if np.std(returns) > 0:
        avg_duration_bars = np.mean(durations) if len(durations) > 0 else 10
        # 1-minute bars: ~375 per day
        trades_per_day = 375.0 / max(avg_duration_bars, 1)
        trades_per_year = trades_per_day * 252 
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max 
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    long_trades = [t for t in trades if t["side"] == "long"]
    short_trades = [t for t in trades if t["side"] == "short"]

    return {
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "trade_count": len(trades),
        "long_count": len(long_trades),
        "short_count": len(short_trades),
        "net_return": float(np.sum(returns)),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "avg_duration": float(np.mean(durations)),
        "max_drawdown": float(max_dd),
        "adverse_selection": 0.0 # Will be populated by caller
    }


def run_monte_carlo(trades_pnl: np.ndarray, initial_capital: float = 100000.0) -> dict:
    if len(trades_pnl) < 10:
        logger.warning(f"Only {len(trades_pnl)} trades for Monte Carlo. Results unreliable.")
    mc = MonteCarloStressTester(n_simulations=5000)
    trades_df = pd.DataFrame({"pnl_pct": trades_pnl})
    return mc.run(trades_df, initial_capital)

def generate_report(wf_results: dict, mc_results: dict, config) -> dict:
    success = config.success_metrics
    agg = wf_results.get("aggregated", {})

    avg_sharpe = agg.get("avg_sharpe", 0)
    avg_win_rate = agg.get("avg_win_rate", 0)
    pct_positive = agg.get("pct_positive_sharpe", 0)
    avg_profit_factor = agg.get("avg_profit_factor", 0)
    
    # We also log adverse selection
    avg_adv_sel = agg.get("avg_adverse_selection", 0)

    checks = {
        "sharpe_ratio": {
            "value": avg_sharpe, "threshold": success.min_sharpe_ratio,
            "pass": avg_sharpe >= success.min_sharpe_ratio,
        },
        "win_rate": {
            "value": avg_win_rate, "threshold": success.min_win_rate,
            "pass": avg_win_rate >= success.min_win_rate,
        },
        "profit_factor": {
            "value": avg_profit_factor, "threshold": 1.0,
            "pass": avg_profit_factor > 1.0,
        },
        "adverse_selection": {
            "value": avg_adv_sel, "threshold": 0.0,
            "pass": avg_adv_sel >= 0.0, # We want M2M to be positive (price moved in our favor)
        }
    }

    if mc_results:
        checks["ruin_probability"] = {
            "value": mc_results.get("probability_of_ruin_50pct", 1.0),
            "threshold": 0.05,
            "pass": mc_results.get("probability_of_ruin_50pct", 1.0) < 0.05,
        }

    all_pass = all(c["pass"] for c in checks.values())

    return {
        "go_no_go": "GO" if all_pass else "NO-GO",
        "checks": checks,
        "mc_results": mc_results,
        "recommendation": "Strategy passes minimum thresholds." if all_pass else "Strategy FAILS thresholds."
    }

def main():
    parser = argparse.ArgumentParser(description="Run Microstructure Alpha backtest")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"])
    parser.add_argument("--days", type=int, default=14) # Default to 14 days for 1m data (yfinance limit is usually 7d, but we request 14)
    args = parser.parse_args()

    config = get_config()
    mlflow.set_tracking_uri("file:./data/mlruns")
    mlflow.set_experiment("Intraday_Microstructure_Backtest")

    logger.info("=" * 70)
    logger.info("  MICROSTRUCTURE ALPHA END-TO-END BACKTEST")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info("=" * 70)

    with mlflow.start_run():
        all_data = fetch_data(args.symbols, args.days)
        if not all_data:
            logger.error("No data fetched.")
            return

        all_wf_results = {}
        all_trade_returns = []

        for symbol, df in all_data.items():
            if len(df) < 500:
                logger.warning(f"Not enough data for {symbol}.")
                continue
                
            logger.info(f"Walk-Forward Validation: {symbol}")
            wf_result = run_walk_forward(symbol, df, config)
            all_wf_results[symbol] = wf_result

            for sr in wf_result.get("individual_results", []):
                if "trade_returns" in sr:
                    all_trade_returns.extend(sr["trade_returns"])
                logger.info(
                    f"  Split: Sharpe={sr.get('sharpe',0):.2f} "
                    f"WR={sr.get('win_rate',0):.1%} "
                    f"PF={sr.get('profit_factor',0):.2f} "
                    f"Trades={sr.get('trade_count',0)} "
                    f"AdvSel={sr.get('adverse_selection',0):.4f}"
                )

        mc_results = {}
        if all_trade_returns:
            trades_arr = np.array(all_trade_returns)
            mc_results = run_monte_carlo(trades_arr, 100000.0)

        combined_agg = {}
        for _symbol, wf in all_wf_results.items():
            for key, val in wf.get("aggregated", {}).items():
                if key not in combined_agg:
                    combined_agg[key] = []
                combined_agg[key].append(val)

        avg_agg = {}
        for k, v in combined_agg.items():
            try:
                if len(v) > 0 and isinstance(v[0], int | float | np.number):
                    avg_agg[k] = np.mean(v)
            except Exception:
                pass

        report = generate_report({"aggregated": avg_agg}, mc_results, config)

        logger.info(f"\n{'#'*70}")
        logger.info(f"  VERDICT: {report['go_no_go']}")
        logger.info(f"{'#'*70}")
        
        for check_name, check_data in report["checks"].items():
            status = "✅ PASS" if check_data["pass"] else "❌ FAIL"
            logger.info(f"  {status} {check_name}: {check_data['value']:.4f} (thr: {check_data['threshold']})")

if __name__ == "__main__":
    main()
