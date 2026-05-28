import pandas as pd
import numpy as np

def nifty_trend(nifty_df: pd.DataFrame) -> str:
    """'up' | 'down' | 'sideways'"""
    if 'close' not in nifty_df.columns:
        return 'sideways'
        
    ema_20 = nifty_df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = nifty_df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    
    if ema_20 > ema_50 * 1.005:
        return 'up'
    elif ema_20 < ema_50 * 0.995:
        return 'down'
    return 'sideways'

def vix_level(vix_df: pd.DataFrame) -> float:
    """Returns latest VIX level"""
    if 'close' in vix_df.columns:
        return float(vix_df['close'].iloc[-1])
    return 15.0 # fallback

def market_breadth(universe_df: pd.DataFrame) -> float:
    """advance/decline ratio"""
    if 'close' not in universe_df.columns or 'open' not in universe_df.columns:
        return 1.0
        
    advances = (universe_df['close'] > universe_df['open']).sum()
    declines = (universe_df['close'] < universe_df['open']).sum()
    
    if declines == 0:
        return float(advances) if advances > 0 else 1.0
        
    return advances / declines

def usd_inr(fx_df: pd.DataFrame) -> float:
    if 'close' in fx_df.columns:
        return float(fx_df['close'].iloc[-1])
    return 83.0 # fallback

def crude_oil_price() -> float:
    # In reality this would fetch from a data source
    return 80.0

def sector_rotation_score(sector_returns: pd.DataFrame) -> dict:
    """Calculates relative momentum of sectors"""
    scores = {}
    if sector_returns.empty:
        return scores
        
    for col in sector_returns.columns:
        # Assuming last 20 days cumulative return as score
        scores[col] = float(sector_returns[col].tail(20).sum())
        
    return scores

def options_pcr_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling average of Put/Call Ratio (PCR)"""
    if 'options_pcr' not in df.columns:
        return pd.Series(1.0, index=df.index)
    return df['options_pcr'].rolling(window=window).mean().fillna(1.0)

def options_max_pain_deviation(df: pd.DataFrame) -> pd.Series:
    """Deviation of current price from options max pain strike"""
    if 'close' not in df.columns or 'options_max_pain' not in df.columns:
        return pd.Series(0.0, index=df.index)
    return ((df['close'] - df['options_max_pain']) / df['close']).fillna(0.0)

def options_unusual_oi_signal(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """Indicator of recent unusual options open interest buildup"""
    if 'options_unusual_oi' not in df.columns:
        return pd.Series(0.0, index=df.index)
    return df['options_unusual_oi'].rolling(window=window).sum().fillna(0.0)

def nifty_futures_basis_pct(df: pd.DataFrame) -> pd.Series:
    """Nifty Futures Basis as % of asset close price"""
    if 'close' not in df.columns or 'nifty_futures_basis' not in df.columns:
        return pd.Series(0.0, index=df.index)
    return (df['nifty_futures_basis'] / df['close']).fillna(0.0)

def fii_dii_net_flow_momentum(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Rolling momentum of combined FII and DII daily net flows (Cr INR)"""
    if 'fii_net_flow' not in df.columns or 'dii_net_flow' not in df.columns:
        return pd.Series(0.0, index=df.index)
    combined = df['fii_net_flow'] + df['dii_net_flow']
    return combined.rolling(window=window).mean().fillna(0.0)

