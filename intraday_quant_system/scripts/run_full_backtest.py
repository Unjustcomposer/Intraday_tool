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
from models.xgboost_model import XGBoostAlphaModel
from models.tft_model import TemporalFusionTransformerModel
from models.catboost_meta_labeler import MetaLabeler
from signals.ensemble import EnsembleScorer
from regime.hmm_regime import RegimeDetector
from backtesting.walk_forward import WalkForwardValidator
from backtesting.monte_carlo import MonteCarloStressTester
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
        df = engine.fetch_historical_data(symbol, start_date, end_date, interval='5minute')
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
    labels = LGBMAlphaModel.make_labels(features_df, atr_mult_up=2.0, atr_mult_down=1.0, horizon_minutes=45)
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
        
        # 1. Train Alpha Models
        lgbm_model = LGBMAlphaModel(config=config.get('models', {}).get('lgbm', {}))
        lgbm_model.train(X_train, y_train, val_size=0.15)
        
        xgb_model = XGBoostAlphaModel()
        xgb_model.train(X_train, y_train, val_size=0.15)
        
        # TFT Model
        tft_model = TemporalFusionTransformerModel()
        seq_len = tft_model.sequence_length
        # Prepare sequences for TFT
        if len(train_df) > seq_len:
            try:
                X_seq_train, y_seq_train, _ = tft_model.prepare_sequences(train_df, common_cols, seq_len=seq_len)
                tft_model.train(X_seq_train, y_seq_train, val_size=0.15, epochs=10)
            except Exception as e:
                logger.warning(f"TFT training failed: {e}")
        
        # 2. Train MetaLabeler (using primary predictions on a validation slice as proxy)
        meta_labeler = MetaLabeler()
        split_meta = int(len(X_train) * 0.7)
        X_meta_train = X_train.iloc[split_meta:]
        y_meta_train = y_train.iloc[split_meta:]
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
        xgb_probs = xgb_model.predict_proba(X_val)
        
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
        
        # Simulate trades bar-by-bar
        trades = []
        position = 0
        entry_price = 0.0
        entry_bar = 0
        closes = val_df['close'].values if 'close' in val_df.columns else np.zeros(len(val_df))
        
        for i in range(len(X_val)):
            regime = regime_series.iloc[i] if i < len(regime_series) else 'unknown'
            
            score = ensemble.compute_score(
                lgbm_prob=lgbm_probs[i],
                xgboost_prob=xgb_probs[i],
                tft_prob=tft_probs[i],
                meta_prob=meta_probs[i],
                sentiment_score=0.0,
                regime_score=0.5
            )
            signal = ensemble.get_signal(score, regime=regime)
            
            current_price = closes[i]
            if current_price <= 0:
                continue
            
            # Flat → entry
            if position == 0 and signal in ('buy', 'sell'):
                position = 1 if signal == 'buy' else -1
                entry_price = current_price
                entry_bar = i
            
            # Position → exit (reversal or opposing signal)
            elif position != 0:
                exit_now = False
                
                # Exit if opposite signal
                if (position == 1 and signal == 'sell') or (position == -1 and signal == 'buy'):
                    exit_now = True
                
                # Exit if held too long (max 45 bars = 3.75 hours)
                if i - entry_bar >= 45:
                    exit_now = True
                
                if exit_now:
                    trade_return = position * (current_price - entry_price) / entry_price
                    trade_return -= round_trip_cost  # Deduct transaction costs
                    trades.append({
                        'return': trade_return,
                        'side': 'long' if position == 1 else 'short',
                        'duration_bars': i - entry_bar
                    })
                    position = 0
                    entry_price = 0.0
        
        # Close any open position at end of validation
        if position != 0 and len(closes) > 0:
            trade_return = position * (closes[-1] - entry_price) / entry_price
            trade_return -= round_trip_cost
            trades.append({
                'return': trade_return,
                'side': 'long' if position == 1 else 'short',
                'duration_bars': len(closes) - entry_bar
            })
        
        # Compute metrics from trade-level results (using lgbm model as proxy for the return model obj)
        return _compute_metrics(trades, lgbm_model, y_val, lgbm_probs)
    
    # Run walk-forward with purge + embargo
    wf = WalkForwardValidator(
        training_window=20,   # Reduced from 35 to 20 to guarantee splits within 40 trading days
        validation_window=5,  # Reduced from 10 to 5
        step_size=5,
        purge_bars=10,
        embargo_bars=10       # NEW: embargo at start of validation
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
    if np.std(returns) > 0:
        # Estimate trades per year from actual duration
        avg_duration_bars = np.mean(durations) if len(durations) > 0 else 10
        trades_per_day = 75.0 / max(avg_duration_bars, 1)  # 75 bars per day
        trades_per_year = trades_per_day * 252
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(trades_per_year)
    else:
        sharpe = 0
    
    # Max drawdown on cumulative equity curve
    cum_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = running_max - cum_returns
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Sortino ratio (proper downside deviation)
    mar = 0.0  # minimum acceptable return
    downside = np.minimum(returns - mar, 0)
    downside_dev = np.sqrt(np.mean(downside ** 2))
    if downside_dev > 0:
        avg_duration_bars = np.mean(durations) if len(durations) > 0 else 10
        trades_per_year = (75.0 / max(avg_duration_bars, 1)) * 252
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
    
    logger.info("=" * 70)
    logger.info("  FULL END-TO-END BACKTEST (v2 — Full Pipeline)")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Period: {args.days} days")
    logger.info(f"  Embargo: 10 bars | Purge: 10 bars")
    logger.info("=" * 70)
    
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
    
    # 4. Monte Carlo stress test
    logger.info(f"\n{'='*50}")
    logger.info("Monte Carlo Stress Test (5,000 simulations)")
    logger.info(f"{'='*50}")
    
    mc_results = {}
    if all_trade_returns:
        mc_results = run_monte_carlo(np.array(all_trade_returns), config.max_capital)
    
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
            # Check if elements are numeric
            if len(v) > 0 and isinstance(v[0], (int, float, np.number)):
                avg_agg[k] = np.mean(v)
            else:
                avg_agg[k] = v[0] if len(v) > 0 else None
        except Exception:
            pass
    report = generate_report(
        {'aggregated': avg_agg},
        mc_results,
        config
    )
    
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
