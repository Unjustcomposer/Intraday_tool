import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import MarketDataEngine
from features.feature_store import FeatureStore
from features.volatility_features import atr as compute_atr
from models.lgbm_model import LGBMAlphaModel
from models.tabnet_model import TabNetModel
from models.catboost_meta_labeler import MetaLabeler
from signals.ensemble import EnsembleScorer
from signals.call_generator import CallGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s', force=True)
logger = logging.getLogger("Simulator")

def simulate(symbol: str, days: int):
    logger.info(f"Fetching {days} days of data for {symbol}...")
    engine = MarketDataEngine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = engine.fetch_historical_data(symbol, start_date, end_date, interval='15minute')
    if df.empty:
        logger.error("No data fetched.")
        return

    # 1. Feature Engineering
    logger.info("Computing features...")
    feature_store = FeatureStore()
    features_df = feature_store.compute_all(symbol, df)
    
    # 2. Extract arrays
    close_prices = features_df['close'].values
    high_prices = features_df['high'].values
    low_prices = features_df['low'].values
    times = features_df.index.values if hasattr(features_df, 'index') else np.arange(len(features_df))
    atr = compute_atr(features_df).values
    
    # We will simulate a rolling window. Train on first 70%, test on 30%.
    train_size = int(len(features_df) * 0.7)
    
    feature_cols = [c for c in features_df.columns if c not in ['timestamp', 'symbol', 'label', 'target', 'open', 'high', 'low', 'close', 'volume', 'date']]
    
    # Fake a target for training (just for simulation purposes)
    labels = LGBMAlphaModel.make_labels(features_df, atr_mult_up=1.5, atr_mult_down=1.5, horizon_minutes=120)
    features_df['label'] = labels
    
    train_df = features_df.iloc[:train_size].dropna()
    test_df = features_df.iloc[train_size:]
    
    if len(train_df) < 100:
        logger.error("Not enough training data.")
        return
        
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['label'].values
    X_test = test_df[feature_cols].fillna(0)
    
    logger.info("Training LGBM...")
    lgbm = LGBMAlphaModel()
    lgbm.train(X_train, pd.Series(y_train))
    lgbm_probs = lgbm.predict_proba(X_test)
    
    logger.info("Training TabNet...")
    tabnet = TabNetModel(model_type='classifier')
    tabnet.fit(X_train.values.astype(np.float32), y_train.astype(np.int64), max_epochs=10, patience=3)
    tabnet_preds = tabnet.predict_proba(X_test.values.astype(np.float32))
    tabnet_probs = tabnet_preds[:, 1] if tabnet_preds.shape[1] > 1 else tabnet_preds[:, 0]
    
    ensemble = EnsembleScorer()
    
    call_gen = CallGenerator(min_risk_reward=1.2, min_confidence=0.60)
    
    trades = []
    
    logger.info("Simulating test period...")
    for i in range(len(test_df)):
        if i + 5 >= len(test_df): # Need future bars to resolve
            break
            
        prob_lgbm = lgbm_probs[i]
        prob_tabnet = tabnet_probs[i]
        
        # Simple ensemble
        score = ensemble.compute_score(lgbm_prob=prob_lgbm, tabnet_prob=prob_tabnet, meta_prob=0.65)
        
        if score > 0.65:
            signal = 'buy'
        elif score < 0.35:
            signal = 'sell'
        else:
            signal = 'hold'
            
        if signal in ['buy', 'sell']:
            c = test_df['close'].iloc[i]
            a = atr[train_size + i]
            call = call_gen.generate_call(symbol, signal, c, a, score, 'normal', 15.0, test_df.index[i])
            
            if call:
                # Resolve trade
                entry = call.entry_price
                sl = call.stop_loss
                tp1 = call.target_1
                
                # Check next 15 bars (intraday)
                outcome = 'expired'
                pnl = 0.0
                for j in range(1, min(15, len(test_df) - i)):
                    fut_h = test_df['high'].iloc[i+j]
                    fut_l = test_df['low'].iloc[i+j]
                    
                    if call.direction == 'BUY':
                        if fut_l <= sl:
                            outcome = 'loss'
                            pnl = -call.risk_pct
                            break
                        if fut_h >= tp1:
                            outcome = 'win'
                            pnl = call.reward_pct_t1
                            break
                    else:
                        if fut_h >= sl:
                            outcome = 'loss'
                            pnl = -call.risk_pct
                            break
                        if fut_l <= tp1:
                            outcome = 'win'
                            pnl = call.reward_pct_t1
                            break
                            
                trades.append({
                    'timestamp': call.timestamp,
                    'direction': call.direction,
                    'pnl': pnl,
                    'outcome': outcome,
                    'confidence': call.confidence,
                    'rr': call.risk_reward
                })

    df_trades = pd.DataFrame(trades)
    if not df_trades.empty:
        wins = len(df_trades[df_trades['outcome'] == 'win'])
        losses = len(df_trades[df_trades['outcome'] == 'loss'])
        total = len(df_trades[df_trades['outcome'] != 'expired'])
        win_rate = (wins / total) * 100 if total > 0 else 0
        total_pnl = df_trades['pnl'].sum()
        
        logger.info(f"\n{'='*50}")
        logger.info(f"SIMULATION RESULTS: {symbol}")
        logger.info(f"{'='*50}")
        logger.info(f"Total Trades Taken: {len(df_trades)}")
        logger.info(f"Resolved Trades:    {total}")
        logger.info(f"Wins / Losses:      {wins} / {losses}")
        logger.info(f"Win Rate:           {win_rate:.2f}%")
        logger.info(f"Total PnL (Unlevered): +{total_pnl:.2f}%")
        logger.info(f"Avg R:R per trade:  1:{df_trades['rr'].mean():.2f}")
        logger.info(f"{'='*50}\n")
    else:
        logger.info("No trades generated during test period. Filters were strictly active.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+')
    parser.add_argument('--days', type=int, default=120)
    args = parser.parse_args()
    for s in args.symbols:
        simulate(s, args.days)
