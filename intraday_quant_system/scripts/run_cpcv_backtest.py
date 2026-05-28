"""
CPCV Backtest Runner — Uses Real Data and Labels
=================================================
Replaces the mock random-label script with a proper CPCV validator
that uses real yfinance data and triple-barrier labels.

Usage:
    python -m scripts.run_cpcv_backtest --symbols RELIANCE HDFCBANK --days 120
"""

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
from backtesting.cpcv_validator import CPCVValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("CPCVBacktest")


def main():
    parser = argparse.ArgumentParser(description='Run CPCV backtest with real data')
    parser.add_argument('--symbols', nargs='+', default=['RELIANCE', 'HDFCBANK'],
                       help='Symbols to validate')
    parser.add_argument('--days', type=int, default=120,
                       help='Number of days of historical data')
    args = parser.parse_args()
    
    engine = MarketDataEngine()
    feature_store = FeatureStore(bars_per_day=75)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    for symbol in args.symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"CPCV Validation: {symbol}")
        logger.info(f"{'='*60}")
        
        # 1. Fetch real data
        df = engine.fetch_historical_data(symbol, start_date, end_date, interval='5minute')
        if df.empty:
            logger.error(f"No data for {symbol}. Skipping.")
            continue
        
        # 2. Compute features
        if 'timestamp' in df.columns:
            df_indexed = df.set_index('timestamp')
        else:
            df_indexed = df
        
        features_df = feature_store.compute_all(symbol, df_indexed)
        
        # 3. Compute ATR and generate REAL labels (not random!)
        if 'atr' not in features_df.columns:
            features_df['atr'] = compute_atr(features_df)
        
        labels = LGBMAlphaModel.make_labels(
            features_df, atr_mult_up=2.0, atr_mult_down=1.0, horizon_minutes=45
        )
        features_df['target'] = labels
        features_df = features_df.dropna()
        
        logger.info(f"  Samples: {len(features_df)}")
        logger.info(f"  Label rate: {features_df['target'].mean():.3f}")
        
        # 4. Get feature columns
        feature_cols = feature_store.get_feature_columns()
        feature_cols = [c for c in feature_cols if c in features_df.columns and c != 'target']
        
        if len(feature_cols) < 3:
            logger.error(f"Insufficient features for {symbol}.")
            continue
        
        logger.info(f"  Features: {len(feature_cols)}")
        
        # 5. Run CPCV
        model = LGBMAlphaModel()
        validator = CPCVValidator(n_groups=6, k_test_groups=2)
        
        logger.info("  Running Combinatorial Purged Cross-Validation...")
        results = validator.run_backtest(model, features_df, feature_cols, 'target')
        
        if not results:
            logger.warning(f"  No CPCV results for {symbol}.")
            continue
        
        # 6. Compute AUC per path
        from sklearn.metrics import roc_auc_score
        all_scores = []
        for i, res in enumerate(results):
            try:
                auc = roc_auc_score(res['y_true'], res['y_prob'])
                all_scores.append(auc)
            except ValueError:
                logger.warning(f"  Path {i+1}: Cannot compute AUC (single class)")
        
        if not all_scores:
            logger.warning(f"  No valid AUC scores for {symbol}.")
            continue
        
        # 7. Report
        mean_auc = np.mean(all_scores)
        std_auc = np.std(all_scores)
        pct_above_chance = np.mean(np.array(all_scores) > 0.5) * 100
        
        logger.info(f"\n  --- CPCV RESULTS: {symbol} ---")
        logger.info(f"  Paths evaluated:     {len(all_scores)}")
        logger.info(f"  Mean AUC:            {mean_auc:.4f}")
        logger.info(f"  Std Dev AUC:         {std_auc:.4f}")
        logger.info(f"  AUC Sharpe:          {mean_auc / max(std_auc, 1e-6):.2f}")
        logger.info(f"  % paths > chance:    {pct_above_chance:.0f}%")
        
        # Interpretation
        if mean_auc > 0.55 and pct_above_chance > 70:
            logger.info(f"  ✅ SIGNAL DETECTED — Mean AUC {mean_auc:.3f} with {pct_above_chance:.0f}% consistency")
        elif mean_auc > 0.52:
            logger.info(f"  ⚠️  WEAK SIGNAL — Mean AUC {mean_auc:.3f}, needs further validation")
        else:
            logger.info(f"  ❌ NO SIGNAL — Mean AUC {mean_auc:.3f} indistinguishable from random")


if __name__ == "__main__":
    main()
