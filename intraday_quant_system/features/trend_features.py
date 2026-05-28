import pandas as pd
import numpy as np

def relative_strength(stock_ret: pd.Series, index_ret: pd.Series) -> pd.Series:
    """stock_ret - index_ret"""
    return stock_ret - index_ret

def ema_slope(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Slope of EMA normalized by price"""
    if 'close' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    ema = df['close'].ewm(span=period, adjust=False).mean()
    # Slope as percentage change over 1 period
    return ema.pct_change()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    if not all(c in df.columns for c in ['high', 'low', 'close']):
        return pd.Series(index=df.index, dtype=float)
        
    high = df['high']
    low = df['low']
    close = df['close']
    
    # +DM and -DM
    up_move = high.diff()
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed True Range and +DM, -DM
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx_series

def momentum(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Price rate of change over period"""
    if 'close' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return df['close'].pct_change(periods=period)

def sector_strength(df: pd.DataFrame, sector_ret: pd.Series) -> pd.Series:
    """Relative strength vs sector"""
    if 'close' not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    stock_ret = df['close'].pct_change()
    return relative_strength(stock_ret, sector_ret)

def beta(stock_ret: pd.Series, index_ret: pd.Series, window: int = 60) -> pd.Series:
    """Rolling beta relative to index"""
    cov = stock_ret.rolling(window=window).cov(index_ret)
    var = index_ret.rolling(window=window).var()
    return cov / var.replace(0, np.nan)
