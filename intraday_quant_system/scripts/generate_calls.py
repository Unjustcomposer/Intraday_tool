"""
Intraday Call Generator — Indian Stocks (NSE)
==============================================
Scans a watchlist of NSE stocks and generates structured trade calls
with Entry, Target, Stop-Loss, and Risk:Reward analysis.

Usage:
    python -m scripts.generate_calls
    python -m scripts.generate_calls --symbols RELIANCE.NS HDFCBANK.NS
    python -m scripts.generate_calls --days 59
"""

import sys
import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.market_data import MarketDataEngine
from features.feature_store import FeatureStore
from features.volatility_features import atr as compute_atr
from models.lgbm_model import LGBMAlphaModel
from models.xgboost_model import XGBoostAlphaModel
from models.catboost_meta_labeler import MetaLabeler
from signals.ensemble import EnsembleScorer
import shap
from signals.call_generator import CallGenerator
from regime.hmm_regime import RegimeDetector
from deployment.config import get_config
from features.news_sentiment import NewsSentimentEngine
from features.volume_profile import calculate_vpvr, get_closest_support_resistance
from data.screener import DynamicScreener, FNO_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    force=True
)
logger = logging.getLogger("CallGenerator")

# Full NSE F&O Universe watchlist is now located in data.screener
DEFAULT_WATCHLIST = FNO_UNIVERSE[:10] # Default to top 10 for quick static runs


def fetch_india_vix() -> float:
    """Fetch real India VIX value from yfinance."""
    try:
        import yfinance as yf
        vix_data = yf.Ticker('^INDIAVIX').history(period='5d', interval='1d')
        if not vix_data.empty:
            vix_val = float(vix_data['Close'].iloc[-1])
            logger.info(f"India VIX: {vix_val:.2f}")
            return vix_val
    except Exception as e:
        logger.warning(f"Failed to fetch India VIX: {e}")
    return 15.0  # Conservative fallback


def fetch_stock_data(symbols: list, days: int = 59) -> dict:
    """Fetch historical 15-min bar data for NSE stocks via yfinance."""
    engine = MarketDataEngine()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching {days} days of 15-min data for {symbol}...")
        try:
            df = engine.fetch_historical_data(symbol, start_date, end_date, interval='15minute')
            if not df.empty:
                all_data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} bars loaded")
            else:
                logger.warning(f"  {symbol}: No data returned")
        except Exception as e:
            logger.error(f"  {symbol}: Failed to fetch data — {e}")
    
    return all_data


def run_pipeline(symbol: str, df: pd.DataFrame, config: dict,
                 india_vix: float = 15.0, news_engine=None) -> list:
    """
    Run the full ML pipeline on a single stock:
      1. Feature engineering
      2. Label generation
      3. Train ensemble (LGBM + XGBoost + Meta-Labeler)
      4. Generate signals on latest bars
      5. Convert signals to trade calls
    
    Returns list of TradeCall objects.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"  Processing: {symbol}")
    logger.info(f"{'='*60}")
    
    # --- 1. Feature Engineering ---
    feature_store = FeatureStore()
    features_df = feature_store.compute_all(symbol, df)
    
    if features_df.empty or len(features_df) < 100:
        logger.warning(f"  {symbol}: Insufficient data ({len(features_df)} bars). Skipping.")
        return []
    
    # --- 2. Label Generation (Triple Barrier) ---
    close = features_df['close'].values
    atr_series = compute_atr(features_df)
    
    # Forward return label: 1 if price goes up by > 0.5*ATR before going down 0.5*ATR
    labels = []
    for i in range(len(close) - 1):
        future_returns = (close[i+1:min(i+26, len(close))] - close[i]) / close[i]
        if len(future_returns) == 0:
            labels.append(0)
            continue
        max_up = future_returns.max()
        max_down = future_returns.min()
        atr_pct = atr_series.iloc[i] / close[i] if close[i] > 0 and not np.isnan(atr_series.iloc[i]) else 0.01
        
        if max_up > 0.5 * atr_pct and max_up > abs(max_down):
            labels.append(1)  # Bullish
        elif abs(max_down) > 0.5 * atr_pct and abs(max_down) > max_up:
            labels.append(0)  # Bearish
        else:
            labels.append(0)  # Neutral → bearish bucket
    labels.append(0)  # Last bar has no forward data
    
    features_df['label'] = labels
    features_df['atr'] = atr_series
    
    # --- 3. Train/Val Split with PURGE GAP (prevents forward label leakage) ---
    split_idx = int(len(features_df) * 0.70)
    purge_gap = 30  # Skip 30 bars between train/val to prevent label leakage
    train_df = features_df.iloc[:split_idx].copy()
    val_df = features_df.iloc[split_idx + purge_gap:].copy()
    
    if len(train_df) < 50 or len(val_df) < 20:
        logger.warning(f"  {symbol}: Not enough data for train/val split. Skipping.")
        return []
    
    # Feature columns (exclude metadata)
    exclude_cols = ['label', 'symbol', 'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'vwap', 'bid_price', 'ask_price', 'bid_volume',
                    'ask_volume', 'oi', 'spread', 'trade_count', 'aggressor_side',
                    'buy_volume', 'sell_volume', 'options_pcr', 'options_max_pain',
                    'options_unusual_oi', 'nifty_futures_basis', 'fii_net_flow',
                    'dii_net_flow', 'atr']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols
                    and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    if len(feature_cols) < 5:
        logger.warning(f"  {symbol}: Only {len(feature_cols)} features. Skipping.")
        return []
    
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df['label'].values
    X_val = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_val = val_df['label'].values
    
    # --- 4. Train Models ---
    # LightGBM
    lgbm = LGBMAlphaModel()
    lgbm.train(X_train, pd.Series(y_train))
    lgbm_probs = lgbm.predict_proba(X_val)
    
    # XGBoost
    xgb = XGBoostAlphaModel()
    xgb.train(X_train, pd.Series(y_train))
    
    # Meta-Labeler: Use out-of-fold predictions to avoid feeding overfit predictions
    meta = MetaLabeler()
    try:
        from sklearn.model_selection import TimeSeriesSplit
        oof_preds = np.zeros(len(X_train))
        tscv_meta = TimeSeriesSplit(n_splits=3, gap=15)
        fold_list = list(tscv_meta.split(X_train))
        for tr_idx, va_idx in fold_list:
            lgbm_oof = LGBMAlphaModel()
            lgbm_oof.train(X_train.iloc[tr_idx], pd.Series(y_train[tr_idx]))
            oof_preds[va_idx] = lgbm_oof.predict_proba(X_train.iloc[va_idx])
        # Fill initial bars that weren't in any val fold with simple predictions
        first_val_start = fold_list[0][1][0]
        oof_preds[:first_val_start] = lgbm.predict_proba(X_train.iloc[:first_val_start])
        meta.train(oof_preds, X_train, pd.Series(y_train))
        meta_probs = meta.predict_proba(X_val, lgbm_probs)
    except Exception as e:
        logger.warning(f"  {symbol}: Meta-labeler training failed: {e}. Using default confidence.")
        meta_probs = np.full(len(X_val), 0.7)
    
    # Regime Detection
    regime_detector = RegimeDetector()
    try:
        regime_detector.fit(train_df[['close']].copy())
        regimes = regime_detector.predict(val_df[['close']].copy())
    except Exception:
        regimes = ['unknown'] * len(val_df)
        
    # VPVR: Only use TRAINING data to prevent lookahead bias
    vpvr_data = calculate_vpvr(train_df)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(lgbm.base_model)
    
    # NLP Sentiment
    live_sentiment = 0.0
    if news_engine:
        live_sentiment = news_engine.compute_sentiment(symbol)
    
    # --- 5. Generate Signals on Latest Bars ---
    ensemble = EnsembleScorer()
    call_gen = CallGenerator(min_risk_reward=1.5, min_confidence=0.55)
    
    # Get conformal threshold from meta-labeler
    conformal_threshold = meta.conformal_threshold if hasattr(meta, 'conformal_threshold') else 0.5
    
    calls = []
    # Only look at the last 5 bars for actionable signals
    start_idx = max(0, len(X_val) - 5)
    
    for i in range(start_idx, len(X_val)):
        # Get regime for this bar
        regime = 'unknown'
        if hasattr(regimes, 'iloc') and i < len(regimes):
            regime = regimes.iloc[i]
        elif isinstance(regimes, list) and i < len(regimes):
            regime = regimes[i]
        
        score = ensemble.compute_score(
            lgbm_prob=lgbm_probs[i],
            meta_prob=meta_probs[i],
            sentiment_score=live_sentiment,
            regime_score=0.5,
            meta_gate=conformal_threshold,
            regime=regime,
            symbol=symbol
        )
        
        signal = ensemble.get_signal(
            score, symbol=symbol, regime=regime,
            vix=india_vix, meta_confidence=meta_probs[i],
            sentiment_score=live_sentiment,
            conformal_threshold=conformal_threshold
        )
        
        if signal in ('buy', 'sell'):
            current_close = val_df['close'].iloc[i]
            current_atr = val_df['atr'].iloc[i] if not np.isnan(val_df['atr'].iloc[i]) else current_close * 0.015
            
            # VPVR Support/Resistance
            support, resistance = get_closest_support_resistance(current_close, vpvr_data)
            
            # SHAP
            features_row = X_val.iloc[i:i+1]
            shap_values = explainer.shap_values(features_row)
            
            if isinstance(shap_values, list):
                shap_vals = shap_values[1][0]
            else:
                if len(shap_values.shape) == 3:
                    shap_vals = shap_values[0, :, 1]
                else:
                    shap_vals = shap_values[0]
                    
            top_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]
            feature_names = X_val.columns
            top_features = [f"{'+' if shap_vals[idx] > 0 else '-'}{feature_names[idx]}" for idx in top_idx]
            shap_str = ", ".join(top_features)
            
            ts = None
            if 'timestamp' in val_df.columns:
                ts = pd.to_datetime(val_df['timestamp'].iloc[i])
            elif hasattr(val_df.index, 'to_pydatetime'):
                ts = val_df.index[i]
            
            call = call_gen.generate_call(
                symbol=symbol,
                signal=signal,
                close=current_close,
                atr=current_atr,
                confidence=meta_probs[i],
                regime=regime,
                vix=india_vix,
                vpvr_support=support,
                vpvr_resistance=resistance,
                news_sentiment=live_sentiment,
                shap_features=shap_str,
                timestamp=ts or datetime.now()
            )
            
            if call is not None:
                calls.append(call)
                print(call_gen.format_call(call))
    
    return calls


def main():
    parser = argparse.ArgumentParser(description="Generate Intraday Trade Calls")
    parser.add_argument('--symbols', nargs='+', help="Specific symbols to process")
    parser.add_argument('--days', type=int, default=59, help="Days of history to fetch")
    parser.add_argument('--dynamic-screener', action='store_true', help="Use pre-market gap screener to pick top stocks dynamically")
    parser.add_argument('--top-n', type=int, default=15, help="Number of stocks to pick if using dynamic screener")
    parser.add_argument('--output', type=str, default='calls_log.csv', help='Output CSV file for call log')
    args = parser.parse_args()
    
    if args.dynamic_screener:
        screener = DynamicScreener(top_n=args.top_n)
        symbols = screener.scan_pre_market()
    else:
        symbols = args.symbols or DEFAULT_WATCHLIST
    config = get_config()
    
    # Fetch real India VIX
    india_vix = fetch_india_vix()
    
    print("\n" + "=" * 60)
    print("  [IN] NSE INTRADAY CALL GENERATOR")
    print(f"  Scanning {len(symbols)} stocks | {args.days} days lookback")
    print(f"  India VIX = {india_vix:.1f}")
    print(f"  Minimum Risk:Reward = 1:1.5")
    print("=" * 60)
    
    # --- Fetch Data ---
    all_data = fetch_stock_data(symbols, days=args.days)
    
    if not all_data:
        logger.error("No data fetched for any symbol. Exiting.")
        return
    
    logger.info(f"\nSuccessfully fetched data for {len(all_data)}/{len(symbols)} symbols.")
    
    # --- Process Each Stock ---
    all_calls = []
    call_gen = CallGenerator(min_risk_reward=1.5, min_confidence=0.55)
    news_engine = NewsSentimentEngine(hours_lookback=24)
    
    for symbol, df in all_data.items():
        try:
            calls = run_pipeline(symbol, df, config, india_vix=india_vix, news_engine=news_engine)
            all_calls.extend(calls)
        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- Summary ---
    call_gen.calls_today = all_calls
    print(call_gen.get_summary())
    
    # --- Export to CSV ---
    if all_calls:
        calls_df = call_gen.to_dataframe()
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
        calls_df.to_csv(output_path, index=False)
        logger.info(f"\nCalls exported to: {output_path}")
    else:
        print("\n[!] No calls generated. This could mean:")
        print("   - All signals were below confidence threshold")
        print("   - Risk:Reward ratios were insufficient")
        print("   - Market regime is unfavorable (crisis/chop)")
        print("   - Insufficient training data for the models")


if __name__ == '__main__':
    main()
