import numpy as np
import pandas as pd


def nifty_trend(nifty_df: pd.DataFrame) -> str:
    """'up' | 'down' | 'sideways'"""
    if "close" not in nifty_df.columns:
        return "sideways"

    ema_20 = nifty_df["close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema_50 = nifty_df["close"].ewm(span=50, adjust=False).mean().iloc[-1]

    if ema_20 > ema_50 * 1.005:
        return "up"
    elif ema_20 < ema_50 * 0.995:
        return "down"
    return "sideways"


def vix_level(vix_df: pd.DataFrame) -> float:
    """Returns latest VIX level"""
    if "close" in vix_df.columns:
        return float(vix_df["close"].iloc[-1])
    return 15.0  # fallback


def market_breadth(universe_df: pd.DataFrame) -> float:
    """advance/decline ratio"""
    if "close" not in universe_df.columns or "open" not in universe_df.columns:
        return 1.0

    advances = (universe_df["close"] > universe_df["open"]).sum()
    declines = (universe_df["close"] < universe_df["open"]).sum()

    if declines == 0:
        return float(advances) if advances > 0 else 1.0

    return advances / declines


def usd_inr(fx_df: pd.DataFrame) -> float:
    if "close" in fx_df.columns:
        return float(fx_df["close"].iloc[-1])
    return 83.0  # fallback


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


def nifty_futures_basis_pct(df: pd.DataFrame) -> pd.Series:
    """Nifty Futures Basis proxy (using moving average divergence as a proxy for basis expansion/contraction)."""
    if "close" not in df.columns:
        return pd.Series(0.0, index=df.index)

    # Proxy: If EMA 10 is sharply above EMA 50, implies contango / positive basis
    ema_short = df["close"].ewm(span=10, adjust=False).mean()
    ema_long = df["close"].ewm(span=50, adjust=False).mean()

    proxy_basis = (ema_short - ema_long) / ema_long
    return proxy_basis


def fii_dii_net_flow_momentum(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Rolling momentum proxy for institutional flows (uses cumulative volume delta)."""
    if "close" not in df.columns or "volume" not in df.columns:
        return pd.Series(0.0, index=df.index)

    # Proxy: Directional volume (close > open -> positive flow)
    if "open" in df.columns:
        flow = np.where(df["close"] > df["open"], df["volume"], -df["volume"])
    else:
        flow = df["volume"] * np.sign(df["close"].diff().fillna(0))

    proxy_flow = pd.Series(flow, index=df.index).rolling(window=window).mean()
    # Normalize
    mean_vol = df["volume"].rolling(window=window).mean().replace(0, 1)
    return proxy_flow / mean_vol
