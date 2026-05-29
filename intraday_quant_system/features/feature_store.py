import pandas as pd
import numpy as np
import logging
import os
from .flow_features import relative_volume, vwap_deviation, volume_delta, index_relative_strength, kyles_lambda, amihud_illiquidity, trade_size_distribution
from .trend_features import ema_slope, adx, momentum
from .volatility_features import atr, realized_volatility, bollinger_width, volatility_percentile, range_expansion, garman_klass_volatility
from .time_features import time_of_day_features
from .sentiment_features import encode_sentiment
from .market_context_features import options_pcr_ratio, options_max_pain_deviation, options_unusual_oi_signal, nifty_futures_basis_pct, fii_dii_net_flow_momentum

logger = logging.getLogger(__name__)


class FeatureStore:
    def __init__(self, storage_dir: str = "./data/feature_store", bars_per_day: int = 75):
        self.storage_dir = storage_dir
        self.bars_per_day = bars_per_day
        self.bars_per_year = bars_per_day * 252
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Track feature names for downstream consumers (e.g. Transformer input dim)
        self._feature_columns: list = []
        
    def compute_all(self, symbol: str, df: pd.DataFrame, sentiment_data: dict = None) -> pd.DataFrame:
        """
        Compute all features for a given symbol.
        
        Args:
            symbol: Stock symbol
            df: OHLCV DataFrame with columns [open, high, low, close, volume, vwap, ...]
            sentiment_data: Optional dict from FinBERT {sentiment: str, score: float}
        """
        if df.empty:
            return df
            
        logger.info(f"Computing features for {symbol}")
        features = df.copy()
        
        # === Flow Features ===
        features['rvol'] = relative_volume(df)
        features['vwap_dev'] = vwap_deviation(df)
        features['vol_delta'] = volume_delta(df)
        features['kyles_lambda'] = kyles_lambda(df)
        features['amihud_illiquidity'] = amihud_illiquidity(df)
        features['trade_size_dist'] = trade_size_distribution(df)
        
        # === Options Flow Features ===
        features['options_pcr_ratio'] = options_pcr_ratio(df)
        features['options_max_pain_dev'] = options_max_pain_deviation(df)
        features['options_unusual_oi'] = options_unusual_oi_signal(df)
        
        # === Cross-Asset Context Features ===
        features['nifty_basis_pct'] = nifty_futures_basis_pct(df)
        features['fii_dii_flow_mom'] = fii_dii_net_flow_momentum(df)
        
        # === Trend Features ===
        features['ema_slope'] = ema_slope(df)
        features['adx'] = adx(df)
        features['momentum'] = momentum(df)
        features['rel_strength'] = index_relative_strength(df) # Optionally pass index_df here if available
        
        # === Volatility Features (with corrected annualization) ===
        features['atr'] = atr(df)
        features['realized_vol'] = realized_volatility(df, bars_per_year=self.bars_per_year)
        features['bb_width'] = bollinger_width(df)
        features['gk_vol'] = garman_klass_volatility(df, bars_per_year=self.bars_per_year)
        # range_expansion uses ATR internally — keep as it provides unique info
        features['range_exp'] = range_expansion(df)
        # Drop vol_percentile in favor of gk_vol to reduce correlation
        # features['vol_percentile'] = volatility_percentile(df, ...)
        
        # === Time-of-Day Features ===
        time_feats = time_of_day_features(df)
        for col in time_feats.columns:
            features[col] = time_feats[col].values
        
        # === Sentiment Features ===
        if sentiment_data:
            features['sentiment_score'] = encode_sentiment(sentiment_data)
        else:
            features['sentiment_score'] = 0.0
        
        # === Clean up with daily-boundary-aware forward fill ===
        features = features.replace([np.inf, -np.inf], np.nan)
        features = self._daily_aware_fill(features)
        
        # === Drop highly correlated features ===
        features = self._drop_correlated_features(features, threshold=0.95)
        
        # Track feature columns (exclude raw OHLCV, metadata, AND unimplemented mock data columns)
        raw_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'vwap', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume',
                    'oi', 'spread', 'trade_count', 'aggressor_side']
        # Columns that require real data sources not yet implemented.
        # These are NaN placeholders and must NOT be used as model features.
        unimplemented_cols = ['options_pcr', 'options_max_pain', 'options_unusual_oi',
                              'nifty_futures_basis', 'fii_net_flow', 'dii_net_flow',
                              'options_pcr_ratio', 'options_max_pain_dev', 
                              'options_unusual_oi', 'nifty_basis_pct', 'fii_dii_flow_mom']
        exclude_cols = set(raw_cols) | set(unimplemented_cols)
        
        # Also exclude any all-NaN columns as a safety net
        all_nan_cols = [c for c in features.columns if features[c].isna().all()]
        if all_nan_cols:
            logger.warning(f"Excluding {len(all_nan_cols)} all-NaN feature columns: {all_nan_cols}")
        exclude_cols.update(all_nan_cols)
        
        self._feature_columns = [c for c in features.columns if c not in exclude_cols]
        
        return features
    
    def get_feature_columns(self) -> list:
        """Return the list of computed feature column names"""
        return self._feature_columns
    
    def get_feature_matrix(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Extract only computed feature columns (for model input)"""
        if not self._feature_columns:
            # Infer feature columns
            raw_cols = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'vwap', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume',
                        'oi', 'spread', 'trade_count', 'aggressor_side']
            self._feature_columns = [c for c in features_df.columns if c not in raw_cols]
        
        available = [c for c in self._feature_columns if c in features_df.columns]
        return features_df[available]
    
    @staticmethod
    def _daily_aware_fill(df: pd.DataFrame) -> pd.DataFrame:
        """
        Forward-fill NaN values within each trading day only.
        Prevents overnight information leakage into intraday features.
        
        Post-fill NaN strategy:
          - Volatility features (atr, gk_vol, realized_vol) → expanding mean
            (prevents 0-vol → infinite position sizing on first bar)
          - All other numeric features → 0
        """
        if hasattr(df.index, 'date'):
            dates = df.index.date
        elif 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp']).dt.date.values
        else:
            # Fallback: simple ffill + fillna(0)
            return df.ffill().fillna(0)
        
        date_series = pd.Series(dates, index=df.index)
        result = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for _, group_idx in date_series.groupby(date_series).groups.items():
            mask = df.index.isin(group_idx)
            result.loc[mask, numeric_cols] = df.loc[mask, numeric_cols].ffill()
        
        # Context-aware NaN fill: volatility features use expanding mean (never 0)
        vol_cols = [c for c in result.columns if c in
                    ('atr', 'gk_vol', 'realized_vol', 'bb_width', 'range_exp')]
        for col in vol_cols:
            if col in result.columns:
                expanding_mean = result[col].expanding(min_periods=1).mean()
                result[col] = result[col].fillna(expanding_mean)
        
        # All remaining NaN → 0 (safe for momentum, sentiment, session flags)
        result = result.fillna(0)
        return result
    
    @staticmethod
    def _drop_correlated_features(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Drop features with correlation > threshold to reduce multicollinearity.
        Keeps the first feature in each correlated pair.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return df
        
        # Only check computed features, not raw OHLCV
        raw_cols = {'open', 'high', 'low', 'close', 'volume', 'vwap'}
        check_cols = [c for c in numeric_cols if c not in raw_cols]
        
        if len(check_cols) < 2:
            return df
        
        corr_matrix = df[check_cols].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
        
        return df
    
    def normalize_for_model(self, df: pd.DataFrame, method: str = 'zscore') -> tuple:
        """
        Normalize features for models that require it (e.g. Transformer).
        
        Returns:
            (normalized_df, stats_dict) — stats_dict contains mean/std for inverse transform
        """
        feature_cols = self.get_feature_columns()
        available = [c for c in feature_cols if c in df.columns]
        
        if not available:
            return df, {}
        
        result = df.copy()
        stats = {}
        
        if method == 'zscore':
            for col in available:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    result[col] = (df[col] - mean) / std
                    stats[col] = {'mean': mean, 'std': std}
                else:
                    result[col] = 0.0
                    stats[col] = {'mean': mean, 'std': 1.0}
        elif method == 'minmax':
            for col in available:
                min_val = df[col].min()
                max_val = df[col].max()
                range_val = max_val - min_val
                if range_val > 0:
                    result[col] = (df[col] - min_val) / range_val
                    stats[col] = {'min': min_val, 'max': max_val}
                else:
                    result[col] = 0.0
                    stats[col] = {'min': min_val, 'max': min_val + 1.0}
        
        return result, stats
        
    def save(self, symbol: str, features_df: pd.DataFrame, date: str):
        """Persist computed features to Parquet (partitioned by date/symbol) with atomic writes"""
        if features_df.empty:
            return
            
        date_dir = os.path.join(self.storage_dir, f"date={date}")
        os.makedirs(date_dir, exist_ok=True)
        
        filepath = os.path.join(date_dir, f"{symbol}.parquet")
        temp_path = filepath + '.tmp'
        
        try:
            features_df.to_parquet(temp_path, engine='pyarrow')
            if os.path.exists(filepath):
                os.remove(filepath)
            os.rename(temp_path, filepath)
            logger.debug(f"Saved features to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save features: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    def load(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """Load features for backtesting"""
        dates = pd.date_range(start, end).strftime('%Y-%m-%d')
        dfs = []
        
        for d in dates:
            filepath = os.path.join(self.storage_dir, f"date={d}", f"{symbol}.parquet")
            if os.path.exists(filepath):
                try:
                    dfs.append(pd.read_parquet(filepath))
                except Exception as e:
                    logger.error(f"Failed to read {filepath}: {e}")
                
        if dfs:
            return pd.concat(dfs).sort_index()
        return pd.DataFrame()
