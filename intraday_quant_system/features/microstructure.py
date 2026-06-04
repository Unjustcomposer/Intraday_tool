import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def volume_synchronized_probability_of_informed_trading(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Approximates VPIN (Volume-Synchronized Probability of Informed Trading) using OHLCV data.
    Uses Bulk Volume Classification (BVC) to estimate buy and sell volume.
    
    VPIN = |V_buy - V_sell| / V_total over a rolling volume bucket.
    """
    if 'close' not in df.columns or 'volume' not in df.columns:
        return pd.Series(np.nan, index=df.index)
        
    # Bulk Volume Classification (BVC) based on standard normal CDF of price change
    delta_p = df['close'].diff()
    sigma = delta_p.rolling(window=window, min_periods=5).std().replace(0, 1e-5)
    
    # Estimate probability of a trade being buyer-initiated
    # Using normal CDF approximation
    from scipy.stats import norm
    # Handling initial NaNs
    valid_mask = ~delta_p.isna() & ~sigma.isna()
    
    p_buy = pd.Series(0.5, index=df.index)
    if valid_mask.any():
        p_buy.loc[valid_mask] = norm.cdf(delta_p.loc[valid_mask] / sigma.loc[valid_mask])
        
    v_buy = df['volume'] * p_buy
    v_sell = df['volume'] * (1 - p_buy)
    
    # VPIN is |V_buy - V_sell| / V_total over a rolling window
    rolling_v_buy = v_buy.rolling(window=window, min_periods=window//2).sum()
    rolling_v_sell = v_sell.rolling(window=window, min_periods=window//2).sum()
    rolling_v_total = df['volume'].rolling(window=window, min_periods=window//2).sum()
    
    vpin = (rolling_v_buy - rolling_v_sell).abs() / rolling_v_total.replace(0, np.nan)
    # FIXED: Was bfill() which leaked future data into past VPIN values
    return vpin.ffill().fillna(0)


def trade_sign_correlation(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Measures the rolling autocorrelation of trade signs to detect institutional slicing (iceberg orders).
    Since we don't have tick data, we use the sign of the close-to-close return as a proxy for the dominant trade sign.
    """
    if 'close' not in df.columns:
        return pd.Series(np.nan, index=df.index)
        
    trade_signs = np.sign(df['close'].diff()).replace(0, np.nan).ffill().fillna(1)
    
    # Compute rolling lag-1 autocorrelation
    lag_1 = trade_signs.shift(1)
    
    # Manual rolling correlation
    rolling_mean_x = trade_signs.rolling(window=window).mean()
    rolling_mean_y = lag_1.rolling(window=window).mean()
    
    rolling_cov = ((trade_signs - rolling_mean_x) * (lag_1 - rolling_mean_y)).rolling(window=window).mean()
    rolling_var_x = ((trade_signs - rolling_mean_x) ** 2).rolling(window=window).mean()
    rolling_var_y = ((lag_1 - rolling_mean_y) ** 2).rolling(window=window).mean()
    
    autocorr = rolling_cov / np.sqrt(rolling_var_x * rolling_var_y).replace(0, np.nan)
    return autocorr.fillna(0)
