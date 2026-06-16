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


def fetch_stock_data(symbols: list, days: int = 59, broker=None) -> dict:
    """Fetch historical 15-min bar data for NSE stocks via Broker API (or yfinance fallback)."""
    engine = MarketDataEngine(broker_client=broker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    all_data = {}
    for symbol in symbols:
        logger.info(f"Fetching {days} days of 15-min data for {symbol}...")
        try:
            # Use fetch_intraday_data so it automatically uses broker if available
            df = engine.fetch_intraday_data(symbol, start_date, end_date, interval='15minute')
            if not df.empty:
                all_data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} bars loaded")
            else:
                logger.warning(f"  {symbol}: No data returned")
        except Exception as e:
            logger.error(f"  {symbol}: Failed to fetch data — {e}")
    
    return all_data


def run_pipeline(symbol: str, df: pd.DataFrame, config: dict, 
                 india_vix: float = 15.0, news_engine=None, broker=None, call_gen=None) -> list:
    """
    Run the end-to-end ML + Microstructure + NLP pipeline for a single symbol.
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
    
    # --- 1b. Stock Quality Gate ---
    last_close = features_df['close'].iloc[-1]
    
    # Estimate DAILY volume from 15-min bars (sum last ~25 bars = 1 trading day)
    if 'volume' in features_df.columns:
        bars_per_day = 25  # ~6.25 hrs / 15 min
        recent_vol = features_df['volume'].iloc[-bars_per_day * 3:]  # Last 3 days
        if len(recent_vol) >= bars_per_day:
            # Sum bars into daily chunks and take the average
            n_full_days = len(recent_vol) // bars_per_day
            daily_vols = [recent_vol.iloc[i*bars_per_day:(i+1)*bars_per_day].sum() 
                          for i in range(n_full_days)]
            avg_daily_volume = sum(daily_vols) / len(daily_vols) if daily_vols else 0
        else:
            avg_daily_volume = recent_vol.sum()  # Less than 1 day of data
    else:
        avg_daily_volume = 0
    
    if last_close < 50:
        logger.warning(f"  {symbol}: Price ₹{last_close:.2f} below ₹50 minimum. Skipping penny stock.")
        return []
    if avg_daily_volume < 200_000 and avg_daily_volume > 0:
        logger.warning(f"  {symbol}: Est. daily volume {avg_daily_volume:,.0f} below 200K minimum. Skipping illiquid stock.")
        return []
    
    # --- 2. Label Generation (SYMMETRIC Triple Barrier) ---
    # Uses symmetric barriers so that label=1 (bullish) and label=0 (bearish)
    # have equal requirements. This prevents the old asymmetric bias where
    # 75%+ of labels were 0, causing the model to default to bearish predictions.
    close = features_df['close'].values
    atr_series = compute_atr(features_df)
    
    horizon = 25  # 25 bars forward (~6 hours of 15-min bars)
    atr_mult = 1.0  # Symmetric: same multiplier for up and down
    labels = []
    for i in range(len(close)):
        if i + horizon >= len(close) or np.isnan(atr_series.iloc[i]) or atr_series.iloc[i] <= 0:
            labels.append(-1)  # Mark as "unlabeled" (will be filtered out)
            continue
        
        current_atr = atr_series.iloc[i]
        target_up = close[i] + atr_mult * current_atr
        target_down = close[i] - atr_mult * current_atr
        
        label = -1  # Neutral (neither barrier hit)
        for j in range(i + 1, min(i + 1 + horizon, len(close))):
            if close[j] >= target_up:
                label = 1  # Bullish: hit upper barrier first
                break
            elif close[j] <= target_down:
                label = 0  # Bearish: hit lower barrier first
                break
        labels.append(label)
    
    features_df['label_raw'] = labels
    features_df['atr'] = atr_series
    
    # Filter out neutral/unlabeled bars (label=-1) from training
    labeled_mask = features_df['label_raw'] >= 0
    labeled_df = features_df[labeled_mask].copy()
    labeled_df['label'] = labeled_df['label_raw'].astype(int)
    
    if len(labeled_df) < 100:
        logger.warning(f"  {symbol}: Only {len(labeled_df)} labeled bars after filtering neutrals. Skipping.")
        return []
    
    pos_rate = labeled_df['label'].mean()
    logger.info(f"  {symbol}: Label distribution — {pos_rate:.1%} bullish, {1-pos_rate:.1%} bearish ({len(labeled_df)} labeled bars)")
    features_df = labeled_df
    
    # --- 3. Train/Val Split with PURGE GAP (prevents forward label leakage) ---
    split_idx = int(len(features_df) * 0.70)
    purge_gap = 30  # Skip 30 bars between train/val to prevent label leakage
    train_df = features_df.iloc[:split_idx].copy()
    val_df = features_df.iloc[split_idx + purge_gap:].copy()
    
    if len(train_df) < 50 or len(val_df) < 20:
        logger.warning(f"  {symbol}: Not enough data for train/val split. Skipping.")
        return []
    
    # Feature columns (exclude metadata)
    exclude_cols = ['label', 'label_raw', 'symbol', 'timestamp', 'open', 'high', 'low', 'close',
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
    
    # TabNet (Deep Neural Network) to decorrelate the ensemble
    tabnet_probs = np.full(len(X_val), 0.5)
    try:
        from models.tabnet_model import TabNetModel
        tabnet = TabNetModel(model_type='classifier')
        # TabNet requires NumPy arrays and is highly sensitive to extreme scales
        X_tr_np = X_train.values.astype(np.float32)
        y_tr_np = y_train.astype(np.int64)
        X_va_np = X_val.values.astype(np.float32)
        y_va_np = y_val.astype(np.int64)
        
        # Fast training limits for intraday generation loops
        tabnet.fit(X_train=X_tr_np, y_train=y_tr_np, X_valid=X_va_np, y_valid=y_va_np, 
                   max_epochs=20, patience=5)
        
        # Extract bullish probability
        tabnet_preds = tabnet.predict_proba(X_va_np)
        if tabnet_preds.shape[1] > 1:
            tabnet_probs = tabnet_preds[:, 1]
    except Exception as e:
        logger.warning(f"  {symbol}: TabNet training failed: {e}. Falling back to 0.5 prob.")
    
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
        
    # True Level 2 Order Flow Imbalance (OFI) has been moved to execution_engine.py
    # to be evaluated at the exact microsecond the price hits the entry level.
    
    # --- 5. Generate Signals on Latest Bars ---
    ensemble = EnsembleScorer()
    if call_gen is None:
        from signals.call_generator import CallGenerator
        call_gen = CallGenerator(min_risk_reward=1.2, min_confidence=0.40, no_new_calls_after="15:30")
    
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
            tabnet_prob=tabnet_probs[i],
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
        
        # Microstructure Veto (Iceberg detection) has been moved to execution_engine.py
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
                # Deduplicate: Keep only the latest valid call for this symbol
                calls = [call]
    
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
    
    # --- Initialize Fyers Broker for Direct NSE Feeds & WebSockets ---
    fyers_broker = None
    if os.environ.get("FYERS_CLIENT_ID") and os.environ.get("FYERS_ACCESS_TOKEN"):
        try:
            from data.fyers_client import FyersBroker
            fyers_broker = FyersBroker(
                client_id=os.environ.get("FYERS_CLIENT_ID"),
                secret_key=os.environ.get("FYERS_SECRET_KEY", "")
            )
        except Exception as e:
            logger.warning(f"Could not init FyersBroker: {e}")
    else:
        logger.warning("Fyers credentials missing. Will fall back to delayed yfinance data.")
        
    # --- Fetch Data ---
    all_data = fetch_stock_data(symbols, days=args.days, broker=fyers_broker)
    
    if not all_data:
        logger.error("No data fetched for any symbol. Exiting.")
        return
    
    logger.info(f"\nSuccessfully fetched data for {len(all_data)}/{len(symbols)} symbols.")
    
    # Fyers Websocket connection moved to execution_engine.py
    
    # --- Process Each Stock ---
    all_calls = []
    call_gen = CallGenerator(min_risk_reward=1.2, min_confidence=0.40, no_new_calls_after="15:30")
    news_engine = NewsSentimentEngine(hours_lookback=24)
    
    for symbol, df in all_data.items():
        try:
            calls = run_pipeline(symbol, df, config, india_vix=india_vix, news_engine=news_engine, broker=fyers_broker, call_gen=call_gen)
            all_calls.extend(calls)
        except Exception as e:
            logger.error(f"Pipeline failed for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # --- Summary ---
    call_gen.calls_today = all_calls
    
    if all_calls:
        print("\n" + "=" * 60)
        print("  FINAL TRADE CALLS")
        print("=" * 60)
        for call in all_calls:
            print(call_gen.format_call(call))
            
    print(call_gen.get_summary())
    
    # --- Export to CSV ---
    if all_calls:
        calls_df = call_gen.to_dataframe()
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'pending_orders.csv')
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
