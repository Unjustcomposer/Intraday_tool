import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def hurst_exponent(
    df: pd.DataFrame, window: int = 100, min_lags: int = 2, max_lags: int = 20
) -> pd.Series:
    """
    Calculates rolling Hurst Exponent to identify market regime:
    H < 0.5: Mean Reverting
    H = 0.5: Random Walk
    H > 0.5: Trending

    Uses an optimized rolling variance ratio approximation for speed.
    """
    if "close" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Shift by 1 to prevent lookahead bias
    log_prices = np.log(df["close"].shift(1).replace(0, np.nan))
    returns = log_prices.diff().fillna(0)

    # We compute this over a rolling window, but a full R/S analysis in a loop is too slow in Pandas.
    # We will use an approximation based on the scaling of volatility.

    # Variance of lag-1 returns
    var_1 = returns.shift(1).rolling(window=window, min_periods=window // 2).var()

    # Variance of lag-k returns
    k = max_lags
    returns_k = log_prices.diff(periods=k).fillna(0)
    var_k = returns_k.shift(1).rolling(window=window, min_periods=window // 2).var()

    # Var(r_k) ~ k^(2H) * Var(r_1)
    # 2H * log(k) = log(Var(r_k) / Var(r_1))
    # H = log(Var(r_k) / Var(r_1)) / (2 * log(k))
    ratio = (var_k / var_1.replace(0, np.nan)).replace([np.inf, -np.inf, 0], np.nan)

    h_approx = np.log(ratio.astype(float)) / (2 * np.log(k))

    # Clip to theoretical bounds
    return h_approx.clip(0.0, 1.0).fillna(0.5)


def statistical_divergence(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates divergence between price action and momentum (RSI).
    Positive divergence = Price making lower lows but RSI making higher lows (Bullish)
    Negative divergence = Price making higher highs but RSI making lower highs (Bearish)
    """
    if "close" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Shift by 1 to prevent lookahead bias
    delta = df["close"].shift(1).diff()
    gain = (delta.where(delta > 0, 0)).shift(1).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).shift(1).rolling(window=window).mean()

    rs = gain / loss.replace(0, 1e-5)
    rsi = 100 - (100 / (1 + rs))

    # Look for local extrema over a short window
    lookback = 5
    price_min = df["close"].shift(1).rolling(window=lookback).min()
    price_max = df["close"].shift(1).rolling(window=lookback).max()
    rsi_min = rsi.shift(1).rolling(window=lookback).min()
    rsi_max = rsi.shift(1).rolling(window=lookback).max()

    # Shift to compare current extreme with previous extreme
    prev_price_min = price_min.shift(lookback)
    prev_price_max = price_max.shift(lookback)
    prev_rsi_min = rsi_min.shift(lookback)
    prev_rsi_max = rsi_max.shift(lookback)

    # Bullish Divergence: Lower low in price, higher low in RSI
    bull_div = ((df["close"] < prev_price_min) & (rsi > prev_rsi_min)).astype(float)

    # Bearish Divergence: Higher high in price, lower high in RSI
    bear_div = ((df["close"] > prev_price_max) & (rsi < prev_rsi_max)).astype(float)

    # Net divergence signal
    divergence = bull_div - bear_div
    return divergence.fillna(0)
