"""
End-to-End Backtest Runner
==========================
Connects ALL system components to produce a validated performance report.

Usage:
    python -m scripts.run_full_backtest --symbols RELIANCE HDFCBANK --days 180

Flow:
    1. Fetch real historical data (yfinance / Kite API)
    2. Compute all features via FeatureStore
    3. Generate labels (triple-barrier method)
    4. Run Walk-Forward validation with PURGE + EMBARGO
    5. Inside each split: train LGBM → meta-labeler → ensemble → signal filters
    6. Run Monte Carlo stress test on resulting trades
    7. Produce GO / NO-GO report against success metrics
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import MarketDataEngine
from features.feature_store import FeatureStore
from features.volatility_features import atr as compute_atr
from models.lgbm_model import LGBMAlphaModel
from models.tabnet_model import TabNetModel
from models.tft_model import TemporalFusionTransformerModel
from models.catboost_meta_labeler import MetaLabeler
from signals.ensemble import EnsembleScorer
from regime.hmm_regime import RegimeDetector
from backtesting.walk_forward import WalkForwardValidator
from backtesting.monte_carlo import MonteCarloStressTester
from backtesting.dsr_metrics import DeflatedSharpeRatio
from deployment.config import get_config, TransactionCosts

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s', force=True)
logger = logging.getLogger("FullBacktest")


# ─── DATA ────────────────────────────────────────────────────────────────

def fetch_data(symbols: list, days: int) -> dict:
    """Fetch historical data for all symbols"""
    engine = MarketDataEngine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching {days} days of data for {symbol}...")
        df = engine.fetch_historical_data(symbol, start_date, end_date, interval='15minute')
        if not df.empty:
            validation = engine.validate_data(df, symbol)
            logger.info(f"  {symbol}: {len(df)} bars, valid={validation['valid']}, "
                       f"issues={len(validation['issues'])}")
            all_data[symbol] = df
        else:
            logger.warning(f"  {symbol}: No data returned")
    
    return all_data


def compute_features_and_labels(symbol: str, df: pd.DataFrame, feature_store: FeatureStore) -> pd.DataFrame:
    """Compute features + labels for a single symbol"""
    if 'timestamp' in df.columns:
        df_indexed = df.set_index('timestamp')
    else:
        df_indexed = df
    
    features_df = feature_store.compute_all(symbol, df_indexed)
    
    if 'atr' not in features_df.columns:
        features_df['atr'] = compute_atr(features_df)
    
    # Generate labels using triple-barrier method
    labels = LGBMAlphaModel.make_labels(features_df, atr_mult_up=1.5, atr_mult_down=1.5, horizon_minutes=120)
    features_df['label'] = labels
    
    # Drop rows with NaN features (warmup period)
    features_df = features_df.dropna()
    
    return features_df


# ─── WALK-FORWARD WITH FULL SYSTEM COMPONENTS ───────────────────────────

def run_walk_forward(symbol: str, df: pd.DataFrame, feature_cols: list, config: dict) -> dict:
    """
    Walk-forward validation using ACTUAL system components:
    LGBM → Meta-labeler confidence → Ensemble scorer → Signal filters → P&L
    """
    ensemble = EnsembleScorer()
    tc = TransactionCosts()
    round_trip_cost = tc.total_round_trip_pct()
    
    def train_and_evaluate(train_df, val_df):
        """Train on train_df, evaluate on val_df using full signal pipeline"""
        # Filter to available feature columns
        avail_train = [c for c in feature_cols if c in train_df.columns]
        avail_val = [c for c in feature_cols if c in val_df.columns]
        common_cols = list(set(avail_train) & set(avail_val))
        
        if len(common_cols) < 3:
            return _empty_result()
        
        X_train = train_df[common_cols]
        y_train = train_df['label']
        X_val = val_df[common_cols]
        y_val = val_df['label']
        
        if len(X_train) < 200 or len(X_val) < 50:
            return _empty_result()
            
        # === NEW: Correlation Feature Dropping (fitted strictly on train) ===
        corr_matrix = X_train.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
        if to_drop:
            X_train = X_train.drop(columns=to_drop)
            X_val = X_val.drop(columns=[c for c in to_drop if c in X_val.columns])
            common_cols = [c for c in common_cols if c not in to_drop]
            
        # === NEW: Sequential Train/Meta Split (70/30) ===
        split_meta = int(len(X_train) * 0.7)
        X_primary_train = X_train.iloc[:split_meta]
        y_primary_train = y_train.iloc[:split_meta]
        
        X_meta_train = X_train.iloc[split_meta:]
        y_meta_train = y_train.iloc[split_meta:]
        
        # 1. Train Alpha Models (on Primary 70%)
        lgbm_model = LGBMAlphaModel(config=config.get('models', {}).get('lgbm', {}))
        lgbm_model.train(X_primary_train, y_primary_train, val_size=0.15)
        
        # TabNet (Deep Neural Network) to decorrelate the ensemble
        from models.tabnet_model import TabNetModel
        tabnet_model = TabNetModel(model_type='classifier')
        X_tr_np = X_primary_train.values.astype(np.float32)
        y_tr_np = y_primary_train.astype(np.int64)
        X_va_np = X_val.values.astype(np.float32)
        y_va_np = y_val.astype(np.int64)
        tabnet_model.fit(X_train=X_tr_np, y_train=y_tr_np, X_valid=X_va_np, y_valid=y_va_np, max_epochs=20, patience=5)
        
        # Save feature importances to local data/ directory for logging as artifacts
        try:
            lgbm_imp = lgbm_model.feature_importance()
            if not lgbm_imp.empty:
                os.makedirs("./data", exist_ok=True)
                lgbm_imp.to_csv(f"./data/feature_importance_{symbol}_lgbm.csv", index=False)
            
            # tabnet_imp = tabnet_model.get_feature_importance()
            # if not tabnet_imp.empty:
            #     os.makedirs("./data", exist_ok=True)
            #     tabnet_imp.to_csv(f"./data/feature_importance_{symbol}_tabnet.csv", index=False)
        except Exception as e:
            logger.warning(f"Could not save feature importance: {e}")
        
        # TFT Model
        tft_model = TemporalFusionTransformerModel()
        seq_len = tft_model.sequence_length
        # Prepare sequences for TFT
        if len(X_primary_train) > seq_len:
            try:
                train_df_primary = train_df.iloc[:split_meta]
                X_seq_train, y_seq_train, _ = tft_model.prepare_sequences(train_df_primary, common_cols, seq_len=seq_len)
                tft_model.train(X_seq_train, y_seq_train, val_size=0.15, epochs=10)
            except Exception as e:
                logger.warning(f"TFT training failed: {e}")
        
        # 2. Train MetaLabeler (using primary predictions on the out-of-sample meta slice)
        meta_labeler = MetaLabeler()
        lgbm_meta_preds = lgbm_model.predict(X_meta_train)
        
        # Meta-label: 1 if primary prediction was correct, 0 if incorrect
        y_meta_outcome = (lgbm_meta_preds == y_meta_train).astype(int)
        meta_labeler.train(lgbm_meta_preds, X_meta_train, y_meta_outcome, val_size=0.2)
        
        # 3. Fit regime detector
        regime_detector = RegimeDetector()
        try:
            regime_detector.fit(train_df)
            regime_series = regime_detector.predict(val_df)
        except Exception:
            regime_series = pd.Series('unknown', index=val_df.index)
        
        # 4. Generate predictions on validation set
        lgbm_probs = lgbm_model.predict_proba(X_val)
        tabnet_preds = tabnet_model.predict_proba(X_va_np)
        tabnet_probs = tabnet_preds[:, 1] if tabnet_preds.shape[1] > 1 else tabnet_preds[:, 0]
        
        tft_probs = np.ones(len(X_val)) * 0.5
        if tft_model.is_trained and len(val_df) > seq_len:
            try:
                X_seq_val, _, val_indices = tft_model.prepare_sequences(val_df, common_cols, seq_len=seq_len)
                tft_preds = tft_model.predict_proba(X_seq_val)
                # Align predictions back to full length (pad start with neutral)
                tft_probs[val_indices] = tft_preds
            except Exception as e:
                logger.warning(f"TFT predict failed: {e}")
                
        lgbm_preds = (lgbm_probs > 0.5).astype(int)
        meta_probs = meta_labeler.predict_proba(X_val, primary_preds=lgbm_preds)
        
        # Simulate trades bar-by-bar (with 1-bar execution delay and queue simulation)
        trades = []
        position = 0
        entry_price = 0.0
        entry_bar = 0
        closes = val_df['close'].values if 'close' in val_df.columns else np.zeros(len(val_df))
        opens = val_df['open'].values if 'open' in val_df.columns else np.zeros(len(val_df))
        highs = val_df['high'].values if 'high' in val_df.columns else closes
        lows = val_df['low'].values if 'low' in val_df.columns else closes
        volumes = val_df['volume'].values if 'volume' in val_df.columns else np.ones(len(val_df)) * 10000
        
        pending_entry = None  # dict with side, price, queue_pos
        pending_exit = None   # dict with price, queue_pos
        atrs = val_df['atr'].values if 'atr' in val_df.columns else np.ones(len(val_df))
        
        # Fix #9: True Realized Spread calculation
        # If true spread is missing, calculate a dynamic proxy based on volume liquidity rather than a static ATR percentage.
        if 'spread' in val_df.columns and not val_df['spread'].isna().all():
            spreads = val_df['spread'].values
        else:
            vols = val_df['volume'].values if 'volume' in val_df.columns else np.ones(len(val_df))
            med_vol = np.nanmedian(vols) if len(vols) > 0 else 1.0
            vol_ratio = np.clip(med_vol / np.maximum(vols, 1.0), 0.5, 3.0)
            spreads = atrs * vol_ratio * 0.05
        
        for i in range(len(X_val)):
            current_open = opens[i]
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            current_vol = volumes[i]
            current_spread = spreads[i]
            
            if current_open <= 0 or current_close <= 0:
                continue
                
            # 1. Execute pending EXITS
            if pending_exit and position != 0:
                exit_price = pending_exit['price']
                is_gap = (current_open < exit_price) if position == 1 else (current_open > exit_price)
                # Fix #3: Queue Position Fill Assumption
                # Require price to trade *through* the limit price to assume a fill
                is_cross = (current_low < exit_price and current_high >= exit_price) if position == 1 else (current_high > exit_price and current_low <= exit_price)
                
                if is_gap:
                    pending_exit['queue_pos'] = 0
                    fill_price = current_open
                elif is_cross:
                    # Reduce queue by 15% of the bar's volume
                    pending_exit['queue_pos'] -= (current_vol * 0.15)
                    fill_price = exit_price
                    
                if pending_exit['queue_pos'] <= 0 or is_gap:
                    # Dynamic Tiered Slippage: Spread/2 represents expected slippage cost to cross the book
                    dynamic_slippage = (current_spread / 2) / fill_price
                    trade_return = position * (fill_price - entry_price) / entry_price
                    
                    # Fix #20: Tax Ignorance in Backtest
                    # Absolute flat fee deduction for tax/brokerage (₹40 round trip)
                    # Assuming a base ₹1,00,000 per trade, ₹40 = 0.04% absolute drag per trade
                    flat_fee_pct = 40.0 / 100000.0
                    trade_return -= (dynamic_slippage + round_trip_cost + flat_fee_pct)
                    
                    trades.append({
                        'return': trade_return,
                        'side': 'long' if position == 1 else 'short',
                        'duration_bars': i - entry_bar
                    })
                    position = 0
                    entry_price = 0.0
                    pending_exit = None
                
            # 1.5 Execute pending ENTRIES
            if pending_entry and position == 0:
                entry_target = pending_entry['price']
                # For long, we want to buy at limit (price must drop to or below limit)
                if pending_entry['side'] == 'buy':
                    is_gap = current_open < entry_target
                    # Fix #3: Queue Position Fill Assumption
                    is_cross = current_low < entry_target and current_high >= entry_target
                else:
                    is_gap = current_open > entry_target
                    is_cross = current_high > entry_target and current_low <= entry_target
                    
                if is_gap:
                    pending_entry['queue_pos'] = 0
                    fill_price = current_open
                elif is_cross:
                    pending_entry['queue_pos'] -= (current_vol * 0.15)
                    fill_price = entry_target
                    
                if pending_entry['queue_pos'] <= 0 or is_gap:
                    position = 1 if pending_entry['side'] == 'buy' else -1
                    entry_price = fill_price
                    entry_bar = i
                    pending_entry = None
                
            # 2. Compute signal at the CLOSE of the candle
            regime = regime_series.iloc[i] if i < len(regime_series) else 'unknown'
            
            score = ensemble.compute_score(
                lgbm_prob=lgbm_probs[i],
                tabnet_prob=tabnet_probs[i],
                tft_prob=tft_probs[i],
                meta_prob=meta_probs[i],
                sentiment_score=0.0, # Deep historical news is unavailable, default to 0.0 for backtesting.
                regime_score=0.5,
                symbol=symbol
            )
            current_vix = val_df['realized_vol'].iloc[i] * 100 if 'realized_vol' in val_df.columns else 15.0
            signal = ensemble.get_signal(score, symbol=symbol, regime=regime, vix=current_vix, meta_confidence=meta_probs[i])
            
            # Check exit/entry conditions for the next bar's open
            if position != 0:
                exit_now = False
                current_atr = atrs[i]
                
                # Take-Profit logic (1.5x ATR)
                unrealized_pnl = position * (current_close - entry_price)
                if unrealized_pnl > (1.5 * current_atr):
                    exit_now = True
                    
                # Dynamic Stop-Loss logic (Scales with spread to prevent whipsaw in illiquid names)
                dynamic_sl = max(1.0 * current_atr, current_spread * 2)
                if unrealized_pnl < (-1.0 * dynamic_sl):
                    exit_now = True
                
                # Exit if opposing signal
                if (position == 1 and signal == 'sell') or (position == -1 and signal == 'buy'):
                    exit_now = True
                    pending_signal = signal  # Queue the reverse entry
                # Exit if held too long (max 168 bars = 1 week on 1H)
                elif i - entry_bar >= 168:
                    exit_now = True
                    
                if exit_now and not pending_exit:
                    pending_exit = {
                        'price': current_close,
                        'queue_pos': current_vol * 0.1 # Queue is 10% of current bar vol
                    }
            else:
                if signal in ('buy', 'sell') and not pending_entry:
                    pending_entry = {
                        'side': signal,
                        'price': current_close,
                        'queue_pos': current_vol * 0.1
                    }
        
        # Close any open position at end of validation (at the last close price)
        if position != 0 and len(closes) > 0:
            final_price = closes[-1]
            final_spread = spreads[-1]
            dynamic_slippage = (final_spread / 2) / final_price
            
            trade_return = position * (final_price - entry_price) / entry_price
            trade_return -= (dynamic_slippage + round_trip_cost)
            trades.append({
                'return': trade_return,
                'side': 'long' if position == 1 else 'short',
                'duration_bars': len(closes) - entry_bar
            })
        
        # Compute metrics from trade-level results (using lgbm model as proxy for the return model obj)
        return _compute_metrics(trades, lgbm_model, y_val, lgbm_probs)
    
    # Run walk-forward with purge + embargo
    wf = WalkForwardValidator(
        training_window=20,
        validation_window=5,
        step_size=5,
        purge_bars=24,
        embargo_bars=24
    )
    
    results = wf.run(df, train_and_evaluate)
    aggregated = wf.aggregate_results(results)
    
    return {
        'symbol': symbol,
        'n_splits': len(results),
        'individual_results': results,
        'aggregated': aggregated
    }


def _empty_result():
    return {
        'sharpe': 0, 'win_rate': 0, 'trade_count': 0,
        'net_return': 0, 'profit_factor': 0, 'avg_duration': 0,
        'val_auc': 0.5, 'long_count': 0, 'short_count': 0,
        'max_drawdown': 0
    }


def _compute_metrics(trades: list, model, y_val, probs) -> dict:
    """Compute comprehensive trade-level metrics"""
    if not trades:
        result = _empty_result()
        try:
            from sklearn.metrics import roc_auc_score
            result['val_auc'] = float(roc_auc_score(y_val, probs))
        except (ValueError, ImportError):
            pass
        return result
    
    returns = np.array([t['return'] for t in trades])
    durations = np.array([t['duration_bars'] for t in trades])
    
    # Core metrics
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    
    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 1
    profit_factor = (np.sum(wins) / np.sum(np.abs(losses))) if len(losses) > 0 and np.sum(np.abs(losses)) > 0 else 0
    
    # Sharpe from trade returns (annualize using actual trade frequency)
    # NOTE: Requires minimum 30 trades for statistical significance
    if np.std(returns) > 0 and len(returns) >= 30:
        # Estimate trades per year from actual duration
        avg_duration_bars = np.mean(durations) if len(durations) > 0 else 10
        trades_per_day = 24.0 / max(avg_duration_bars, 1)  # 24 bars per day
        trades_per_year = trades_per_day * 365 # Crypto 365 days
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)
    elif np.std(returns) > 0:
        # Compute but flag as unreliable
        avg_duration_bars = np.mean(durations) if len(durations) > 0 else 10
        trades_per_year = (24.0 / max(avg_duration_bars, 1)) * 365
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)
        logger.warning(f"Sharpe computed from only {len(returns)} trades (<30) — NOT statistically significant")
    else:
        sharpe = 0
    
    # Max drawdown on compounded equity curve (not cumulative sum)
    # cumsum of % returns is incorrect for large returns; cumprod reflects real equity
    equity_curve = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max  # Percentage drawdown
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Sortino ratio (proper downside deviation)
    mar = 0.0  # minimum acceptable return
    downside = np.minimum(returns - mar, 0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev > 0:
        avg_duration_bars = np.mean(durations) if len(durations) > 0 else 10
        trades_per_year = (24.0 / max(avg_duration_bars, 1)) * 365
        sortino = np.mean(returns - mar) / downside_dev * np.sqrt(trades_per_year)
    else:
        sortino = sharpe
    
    # Long/short breakdown
    long_trades = [t for t in trades if t['side'] == 'long']
    short_trades = [t for t in trades if t['side'] == 'short']
    
    # Validation AUC
    try:
        from sklearn.metrics import roc_auc_score
        val_auc = float(roc_auc_score(y_val, probs))
    except (ValueError, ImportError):
        val_auc = 0.5
    
    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'trade_count': len(trades),
        'long_count': len(long_trades),
        'short_count': len(short_trades),
        'net_return': float(np.sum(returns)),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'avg_duration': float(np.mean(durations)),
        'max_drawdown': float(max_dd),
        'val_auc': float(val_auc),
    }


# ─── MONTE CARLO ────────────────────────────────────────────────────────

def run_monte_carlo(trades_pnl: np.ndarray, initial_capital: float = 1000000.0) -> dict:
    """Monte Carlo stress test on trade returns"""
    if len(trades_pnl) < 10:
        logger.warning(f"Only {len(trades_pnl)} trades for Monte Carlo. Results unreliable.")
    mc = MonteCarloStressTester(n_simulations=5000)
    trades_df = pd.DataFrame({'pnl_pct': trades_pnl})
    return mc.run(trades_df, initial_capital)


# ─── REPORT ──────────────────────────────────────────────────────────────

def generate_report(wf_results: dict, mc_results: dict, config) -> dict:
    """Generate GO / NO-GO report against success metrics"""
    success = config.success_metrics
    agg = wf_results.get('aggregated', {})
    
    avg_sharpe = agg.get('avg_sharpe', 0)
    avg_win_rate = agg.get('avg_win_rate', 0)
    pct_positive = agg.get('pct_positive_sharpe', 0)
    avg_profit_factor = agg.get('avg_profit_factor', 0)
    avg_max_dd = agg.get('avg_max_drawdown', 1.0)
    
    checks = {
        'sharpe_ratio': {
            'value': avg_sharpe,
            'threshold': success.min_sharpe_ratio,
            'pass': avg_sharpe >= success.min_sharpe_ratio
        },
        'win_rate': {
            'value': avg_win_rate,
            'threshold': success.min_win_rate,
            'pass': avg_win_rate >= success.min_win_rate
        },
        'walk_forward_splits_passing': {
            'value': pct_positive,
            'threshold': success.min_walk_forward_splits_passing,
            'pass': pct_positive >= success.min_walk_forward_splits_passing
        },
        'profit_factor': {
            'value': avg_profit_factor,
            'threshold': 1.0,
            'pass': avg_profit_factor > 1.0
        },
    }
    
    # Add Deflated Sharpe Ratio check
    if 'dsr' in wf_results:
        dsr_val = wf_results['dsr']
        checks['deflated_sharpe_ratio'] = {
            'value': dsr_val,
            'threshold': 0.95, # 95% confidence alpha is > 0
            'pass': dsr_val >= 0.95
        }
    
    if mc_results:
        checks['ruin_probability'] = {
            'value': mc_results.get('probability_of_ruin_50pct', 1.0),
            'threshold': 0.05,
            'pass': mc_results.get('probability_of_ruin_50pct', 1.0) < 0.05
        }
    
    all_pass = all(c['pass'] for c in checks.values())
    
    return {
        'go_no_go': 'GO' if all_pass else 'NO-GO',
        'checks': checks,
        'mc_results': mc_results,
        'recommendation': (
            "Strategy passes minimum thresholds. Proceed to paper trading."
            if all_pass else
            "Strategy FAILS minimum thresholds. Do NOT deploy. Review weaknesses above."
        )
    }


# ─── MAIN ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run full end-to-end backtest')
    parser.add_argument('--symbols', nargs='+', default=['RELIANCE', 'HDFCBANK', 'TCS', 'INFY'],
                       help='Symbols to backtest')
    parser.add_argument('--days', type=int, default=180,
                       help='Number of days of historical data')
    args = parser.parse_args()
    
    config = get_config()
    
    # Initialize MLflow tracking
    import mlflow
    mlflow.set_tracking_uri("file:./data/mlruns")
    mlflow.set_experiment("Intraday_Ensemble_Backtest")
    
    logger.info("=" * 70)
    logger.info("  FULL END-TO-END BACKTEST (v2 — Full Pipeline)")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Period: {args.days} days")
    logger.info(f"  Embargo: 10 bars | Purge: 10 bars")
    logger.info("=" * 70)
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params({
            'symbols': ",".join(args.symbols),
            'days': args.days,
            'lgbm_num_leaves': getattr(config.models.lgbm, 'num_leaves', 63),
            'catboost_iterations': getattr(config.models.catboost, 'iterations', 800),
            'catboost_depth': getattr(config.models.catboost, 'depth', 5),
            'catboost_l2_leaf_reg': getattr(config.models.catboost, 'l2_leaf_reg', 10.0),
        })
        
        # 1. Fetch data
        all_data = fetch_data(args.symbols, args.days)
        if not all_data:
            logger.error("No data fetched. Aborting.")
            return
        
        # 2. Compute features + labels for each symbol
        feature_store = FeatureStore(bars_per_day=config.intraday.bars_per_day)
        all_features = {}
        
        for symbol, df in all_data.items():
            logger.info(f"\nComputing features for {symbol}...")
            features_df = compute_features_and_labels(symbol, df, feature_store)
            all_features[symbol] = features_df
            logger.info(f"  {symbol}: {len(features_df)} samples, "
                        f"label_rate={features_df['label'].mean():.3f}")
        
        # 3. Walk-forward validation per symbol
        feature_cols = feature_store.get_feature_columns()
        feature_cols = [c for c in feature_cols if c != 'label']
        
        all_wf_results = {}
        all_trade_returns = []
        
        for symbol, features_df in all_features.items():
            available_cols = [c for c in feature_cols if c in features_df.columns]
            if len(available_cols) < 3:
                logger.warning(f"Insufficient features for {symbol}. Skipping.")
                continue
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Walk-Forward Validation: {symbol}")
            logger.info(f"  Features: {len(available_cols)}")
            logger.info(f"  Samples: {len(features_df)}")
            logger.info(f"{'='*50}")
            
            wf_result = run_walk_forward(
                symbol, features_df, available_cols,
                {'models': {'lgbm': {'num_leaves': 63}}}
            )
            all_wf_results[symbol] = wf_result
            
            # Per-split summary — collect individual trade returns for Monte Carlo
            for sr in wf_result.get('individual_results', []):
                if 'trade_returns' in sr:
                    all_trade_returns.extend(sr['trade_returns'])  # Individual trade P&Ls
                elif 'net_return' in sr:
                    all_trade_returns.append(sr['net_return'])  # Fallback
                logger.info(f"  Split {sr.get('split_id','?')}: "
                           f"Sharpe={sr.get('sharpe',0):.2f} "
                           f"WR={sr.get('win_rate',0):.1%} "
                           f"PF={sr.get('profit_factor',0):.2f} "
                           f"Trades={sr.get('trade_count',0)} "
                           f"(L={sr.get('long_count',0)} S={sr.get('short_count',0)}) "
                           f"AUC={sr.get('val_auc',0.5):.3f}")
            
            # Log symbol metrics to MLflow
            agg = wf_result.get('aggregated', {})
            for key, val in agg.items():
                if isinstance(val, (int, float, np.number)):
                    mlflow.log_metric(f"{symbol}_{key}", float(val))
                    
            # Log feature importances as artifacts if saved
            lgbm_imp_file = f"./data/feature_importance_{symbol}_lgbm.csv"
            tabnet_imp_file = f"./data/feature_importance_{symbol}_tabnet.csv"
            
            if os.path.exists(lgbm_imp_file):
                mlflow.log_artifact(lgbm_imp_file)
            if os.path.exists(tabnet_imp_file):
                mlflow.log_artifact(tabnet_imp_file)
        
        # 4. Monte Carlo stress test & DSR Calculation
        logger.info(f"\n{'='*50}")
        logger.info("Monte Carlo Stress Test & Statistical Validation")
        logger.info(f"{'='*50}")
        
        mc_results = {}
        dsr_prob = 0.0
        if all_trade_returns:
            trades_arr = np.array(all_trade_returns)
            mc_results = run_monte_carlo(trades_arr, config.max_capital)
            
            # Assume ~10,000 independent trials given hyperparameter search space across 4 models
            # and feature selection. We estimate variance of SRs conservatively as 0.5.
            trades_per_year = (24.0 / 10) * 365 # Approximation
            dsr_prob = DeflatedSharpeRatio.calculate_dsr(
                returns=trades_arr, 
                n_trials=10000, 
                variance_of_sharpes=0.5, 
                annualization_factor=trades_per_year
            )
        
        # 5. Generate report
        combined_agg = {}
        for symbol, wf in all_wf_results.items():
            for key, val in wf.get('aggregated', {}).items():
                if key not in combined_agg:
                    combined_agg[key] = []
                combined_agg[key].append(val)
        
        avg_agg = {}
        for k, v in combined_agg.items():
            try:
                if len(v) > 0 and isinstance(v[0], (int, float, np.number)):
                    avg_agg[k] = np.mean(v)
                else:
                    avg_agg[k] = v[0] if len(v) > 0 else None
            except Exception:
                pass
                
        report = generate_report(
            {'aggregated': avg_agg, 'dsr': dsr_prob},
            mc_results,
            config
        )
        
        # Log report metrics to MLflow
        mlflow.log_param("verdict", report['go_no_go'])
        for check_name, check_data in report['checks'].items():
            mlflow.log_metric(f"check_{check_name}_value", float(check_data['value']))
            mlflow.log_param(f"check_{check_name}_pass", int(check_data['pass']))
            
        if mc_results:
            mlflow.log_metrics({
                'mc_median_terminal_capital': float(mc_results.get('median_terminal_capital', 0)),
                'mc_worst_5pct_capital': float(mc_results.get('worst_5pct_capital', 0)),
                'mc_probability_of_profit': float(mc_results.get('probability_of_profit', 0)),
                'mc_probability_of_ruin_50pct': float(mc_results.get('probability_of_ruin_50pct', 0)),
                'mc_median_max_drawdown': float(mc_results.get('median_max_drawdown', 0)),
            })
            
        # Print report
        logger.info(f"\n{'#'*70}")
        logger.info("  BACKTEST REPORT")
        logger.info(f"{'#'*70}")
        
        logger.info(f"\n  VERDICT: {report['go_no_go']}")
        logger.info(f"  {report['recommendation']}\n")
        
        for check_name, check_data in report['checks'].items():
            status = "✅ PASS" if check_data['pass'] else "❌ FAIL"
            logger.info(f"  {status} {check_name}: {check_data['value']:.4f} "
                        f"(threshold: {check_data['threshold']:.4f})")
        
        if mc_results:
            logger.info(f"\n  Monte Carlo ({mc_results.get('n_trades_simulated',0)} trades × 5000 sims):")
            logger.info(f"    Median terminal capital: ₹{mc_results.get('median_terminal_capital', 0):,.0f}")
            logger.info(f"    5th percentile (worst):   ₹{mc_results.get('worst_5pct_capital', 0):,.0f}")
            logger.info(f"    Probability of profit:    {mc_results.get('probability_of_profit', 0):.1%}")
            logger.info(f"    Prob of ruin (50% DD):     {mc_results.get('probability_of_ruin_50pct', 0):.1%}")
            logger.info(f"    Median max drawdown:       {mc_results.get('median_max_drawdown', 0):.1%}")
        
        # Cross-symbol summary
        if all_wf_results:
            logger.info(f"\n  Per-Symbol Summary:")
            for sym, wf in all_wf_results.items():
                agg = wf.get('aggregated', {})
                logger.info(f"    {sym}: Sharpe={agg.get('avg_sharpe',0):.2f} | "
                           f"WR={agg.get('avg_win_rate',0):.1%} | "
                           f"PF={agg.get('avg_profit_factor',0):.2f} | "
                           f"Splits={wf.get('n_splits',0)}")
        
        logger.info(f"\n{'#'*70}\n")


if __name__ == "__main__":
    main()
