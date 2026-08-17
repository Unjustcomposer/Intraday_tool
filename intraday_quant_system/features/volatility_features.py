import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    if not all(c in df.columns for c in ["high", "low", "close"]):
        return pd.Series(index=df.index, dtype=float)

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Shift TR by 1 to prevent lookahead bias in rolling mean
    return tr.shift(1).rolling(window=period).mean()


def realized_volatility(
    df: pd.DataFrame, window: int = 20, bars_per_year: int = 18900
) -> pd.Series:
    """
    Annualized realized volatility using close-to-close returns.

    Args:
        bars_per_year: For 5-min NSE data: 75 bars/day * 252 days = 18,900.
                       For 1-min data: 375 bars/day * 252 = 94,500.
                       For daily data: 252.
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    returns = df["close"].pct_change()
    # Shift returns by 1 to prevent lookahead bias
    return returns.shift(1).rolling(window=window).std() * np.sqrt(bars_per_year)


def bollinger_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """(Upper Band - Lower Band) / Middle Band"""
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    # Shift close by 1 to prevent lookahead bias in rolling stats
    close_shifted = df["close"].shift(1)
    middle_band = close_shifted.rolling(window=period).mean()
    std_dev = close_shifted.rolling(window=period).std()

    upper_band = middle_band + (2 * std_dev)
    lower_band = middle_band - (2 * std_dev)

    return (upper_band - lower_band) / middle_band.replace(0, np.nan)


def volatility_percentile(
    df: pd.DataFrame, lookback_days: int = 60, bars_per_day: int = 75
) -> pd.Series:
    """
    0-100 percentile of current realized vol relative to lookback window.

    Args:
        lookback_days: Number of trading days to look back.
        bars_per_day: Number of intraday bars per trading day (75 for 5-min NSE).
    """
    if "close" not in df.columns:
        return pd.Series(index=df.index, dtype=float)

    lookback_bars = lookback_days * bars_per_day
    vol = realized_volatility(df, window=20)

    # Rolling percentile
    def calc_percentile(x):
        if np.isnan(x[-1]):
            return np.nan
        return pd.Series(x).rank(pct=True).iloc[-1] * 100

    # Use fixed window for stationarity (min_periods prevents NaN for short data)
    # Shift by 1 to prevent lookahead
    return (
        vol.shift(1)
        .rolling(window=lookback_bars, min_periods=lookback_bars // 4)
        .apply(calc_percentile, raw=True)
    )


def range_expansion(df: pd.DataFrame) -> pd.Series:
    """(High - Low) / ATR"""
    if not all(c in df.columns for c in ["high", "low"]):
        return pd.Series(index=df.index, dtype=float)

    current_range = df["high"] - df["low"]
    atr_val = atr(df, period=14)

    # Shift ATR by 1 to prevent lookahead bias
    return current_range / atr_val.shift(1).replace(0, np.nan)


def garman_klass_volatility(
    df: pd.DataFrame, window: int = 20, bars_per_year: int = 18900
) -> pd.Series:
    """
    Garman-Klass volatility estimator — more efficient than close-to-close.
    Uses OHLC data for better variance estimation.
    """
    if not all(c in df.columns for c in ["open", "high", "low", "close"]):
        return pd.Series(index=df.index, dtype=float)

    # Shift OHLC by 1 to prevent lookahead bias
    high = df["high"].shift(1)
    low = df["low"].shift(1)
    open_ = df["open"].shift(1)
    close = df["close"].shift(1)

    log_hl = (np.log(high / low)) ** 2
    log_co = (np.log(close / open_)) ** 2

    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

    return np.sqrt(gk_var.rolling(window=window).mean() * bars_per_year)
