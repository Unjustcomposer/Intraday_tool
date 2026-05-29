import pandas as pd
import numpy as np

def relative_volume(df: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    """Time-of-Day RVOL = CurrentVol / AvgVol(for this specific time of day over last N days)"""
    if 'volume' not in df.columns:
        return pd.Series(1.0, index=df.index)
    
    df = df.copy()
    
    # Handle timestamp in index or column
    if hasattr(df.index, 'hour') and pd.api.types.is_datetime64_any_dtype(df.index):
        df['_time'] = df.index.time
    elif 'timestamp' in df.columns:
        df['_time'] = pd.to_datetime(df['timestamp']).dt.time
    else:
        return pd.Series(1.0, index=df.index)
    
    # Calculate rolling mean of volume for the same time of day
    avg_vol = df.groupby('_time')['volume'].transform(lambda x: x.rolling(window=lookback_days, min_periods=1).mean().shift(1))
    
    result = df['volume'] / avg_vol.replace(0, np.nan)
    return result

def index_relative_strength(df: pd.DataFrame, index_df: pd.DataFrame = None, window: int = 20) -> pd.Series:
    """Stock returns vs Index returns over a rolling window"""
    if 'close' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
        
    stock_ret = df['close'].pct_change().rolling(window=window).sum()
    
    if index_df is None or 'close' not in index_df.columns:
        # Fallback if index not provided
        return stock_ret
        
    index_ret = index_df['close'].pct_change().rolling(window=window).sum()
    return stock_ret - index_ret

def vwap_deviation(df: pd.DataFrame) -> pd.Series:
    """(price - vwap) / vwap"""
    if 'close' not in df.columns or 'vwap' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return (df['close'] - df['vwap']) / df['vwap'].replace(0, np.nan)



def volume_delta(df: pd.DataFrame) -> pd.Series:
    """cumulative buy_vol - sell_vol, reset daily to prevent monotonic growth.
    Approximated via aggressor side or price change if aggressor side not available."""
    # Check if aggressor_side has ACTUAL data (not just the column with all NaN)
    if ('aggressor_side' in df.columns and 'volume' in df.columns 
            and df['aggressor_side'].notna().any()):
        signed_vol = df['volume'] * df['aggressor_side']
    elif 'close' in df.columns and 'open' in df.columns and 'volume' in df.columns:
        # Tick rule approximation: sign volume by intra-bar direction
        direction = np.sign(df['close'] - df['open'])
        signed_vol = df['volume'] * direction
    else:
        return pd.Series(index=df.index, dtype=float)
    
    # Daily reset: cumsum within each trading day only
    if hasattr(df.index, 'date'):
        dates = df.index.date
    elif 'timestamp' in df.columns:
        dates = pd.to_datetime(df['timestamp']).dt.date
    else:
        # Fallback: no date info, cumsum entire series (legacy behavior)
        return signed_vol.cumsum()
    
    date_series = pd.Series(dates, index=df.index)
    return signed_vol.groupby(date_series).cumsum()

def kyles_lambda(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Kyle's Lambda: short-term price impact parameter.
    Lambda = Cov(Price_Diff, Signed_Volume) / Var(Signed_Volume)
    """
    if 'close' not in df.columns or 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index)
        
    price_diff = df['close'].diff()
    
    # Approximate signed volume
    if 'aggressor_side' in df.columns and df['aggressor_side'].notna().any():
        signed_vol = df['volume'] * df['aggressor_side'].fillna(0)
    else:
        # Use returns to sign volume
        direction = np.sign(price_diff)
        signed_vol = df['volume'] * direction
        
    # Compute EWMA covariance and variance to reduce statistical noise
    cov = price_diff.ewm(span=window, min_periods=window//2).cov(signed_vol)
    var = signed_vol.ewm(span=window, min_periods=window//2).var()
    
    # Fill NAs and scale lambda
    k_lambda = cov / var.replace(0, np.nan)
    return k_lambda.fillna(0.0)

def amihud_illiquidity(df: pd.DataFrame, window: int = 60) -> pd.Series:
    """
    Amihud Illiquidity Ratio: |Return| / Dollar (Rupee) Volume
    """
    if 'close' not in df.columns or 'volume' not in df.columns:
        return pd.Series(0.0, index=df.index)
        
    returns = df['close'].pct_change().abs()
    rupee_volume = df['volume'] * df['close']
    
    ratio = returns / rupee_volume.replace(0, np.nan)
    amihud = ratio.ewm(span=window, min_periods=window//2).mean()
    return amihud.fillna(0.0)

def trade_size_distribution(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Trade Size Distribution: Average trade size vs rolling average.
    Tracks presence of large institutional blocks.
    """
    if 'volume' not in df.columns:
        return pd.Series(1.0, index=df.index)
        
    # If trade_count is not available, default it by assuming an average of 100 shares per trade
    trade_count = df['trade_count'] if 'trade_count' in df.columns else pd.Series(np.nan, index=df.index)
    trade_count = trade_count.fillna(df['volume'] / 100.0).replace(0, 1.0)
    
    avg_trade_size = df['volume'] / trade_count
    rolling_avg_size = avg_trade_size.rolling(window=window).mean().replace(0, np.nan)
    
    size_ratio = avg_trade_size / rolling_avg_size
    return size_ratio.fillna(1.0)


