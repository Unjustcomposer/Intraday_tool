import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from features.microstructure import (
    trade_sign_correlation,
    volume_synchronized_probability_of_informed_trading,
)
from features.volume_profile import calculate_vpvr, get_closest_support_resistance

logger = logging.getLogger(__name__)


class MicrostructureAlphaModel:
    """
    Evaluates short-term order flow toxicity and structural imbalances.
    Uses 1-minute OHLCV data to generate an 'Order Flow Pressure' score.
    
    Score ranges from -1.0 (Extreme Sell Pressure / Toxic) to 1.0 (Extreme Buy Pressure).
    """

    def __init__(
        self, vpin_window: int = 50, tsc_window: int = 20, vpvr_bins: int = 50
    ):
        self.vpin_window = vpin_window
        self.tsc_window = tsc_window
        self.vpvr_bins = vpvr_bins

    def generate_scores(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes the combined microstructure alpha score across the DataFrame.
        Supports both 1-minute OHLCV (legacy fallback) and MBP-1 L2 data.
        """
        logger.info(
            f"Generating Microstructure Alpha scores for {len(df)} records..."
        )

        # 1. Order Book Imbalance (OBI) - Requires Level-2 MBP-1 data
        if "bid_sz_00" in df.columns and "ask_sz_00" in df.columns:
            logger.info("Level-2 Data detected. Calculating Order Book Imbalance (OBI).")
            bid_vol = df["bid_sz_00"]
            ask_vol = df["ask_sz_00"]
            obi = (bid_vol - ask_vol) / (bid_vol + ask_vol).replace(0, 1)
        else:
            logger.warning("Level-2 Data not found! OBI will default to 0. Falling back to VPIN.")
            obi = pd.Series(0, index=df.index)

        # 2. Volume-Synchronized Probability of Informed Trading (VPIN) (Fallback)
        if "volume" in df.columns and "close" in df.columns:
            vpin = volume_synchronized_probability_of_informed_trading(
                df, window=self.vpin_window
            )
            tsc = trade_sign_correlation(df, window=self.tsc_window)
            ret = df["close"].pct_change().fillna(0)
        else:
            vpin = pd.Series(0.5, index=df.index)
            tsc = pd.Series(0, index=df.index)
            # If MBP-1 data is used, compute returns from mid-price
            if "bid_px_00" in df.columns:
                mid = (df["bid_px_00"] + df["ask_px_00"]) / 2
                ret = mid.pct_change().fillna(0)
            else:
                ret = pd.Series(0, index=df.index)

        ema_ret = ret.ewm(span=10).mean()
        direction = pd.Series(np.where(ema_ret > 0, 1.0, np.where(ema_ret < 0, -1.0, 0.0)), index=df.index)

        # 3. Support/Resistance Context via Volume Profile
        # We need price and volume to calculate VPVR
        if "close" in df.columns and "volume" in df.columns:
            profile = calculate_vpvr(df, bins=self.vpvr_bins)
            # Find closest support and resistance for the entire series (vectorized approximation)
            # To avoid look-ahead bias, this should ideally be rolling, but for MVP we compute on rolling blocks
            # Here we just use a trailing VPVR proxy context
            dist_to_vwap = (df["close"] - df["close"].rolling(375).mean()) / df["close"].rolling(375).mean()
            vpvr_context = -np.sign(dist_to_vwap) * np.clip(np.abs(dist_to_vwap) * 100, 0, 1)
        else:
            vpvr_context = pd.Series(0, index=df.index)

        # Combine OBI and traditional toxicity
        toxicity_amplifier = 0.5 + (obi * 0.5) 
        
        # In MBP-1 tick data, OBI provides direct structural pressure. 
        # Positive OBI = Buy Pressure, Negative OBI = Sell Pressure
        if (obi != 0).any():
            raw_score = obi + (vpvr_context * 0.1)
        else:
            # Legacy fallback: Abandon naive VPIN trend-following which gets murdered by spread.
            # Instead, we implement a strict Intraday Mean Reversion (RSI + VWAP Z-score)
            # This allows limit orders to sit passively at extremes and collect the spread.
            if "close" in df.columns:
                delta = df["close"].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.nan)
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.fillna(50)
                
                vwap_series = df.get("vwap", df["close"].rolling(375).mean())
                vwap_std = df["close"].rolling(375).std().replace(0, 1)
                z_score = (df["close"] - vwap_series) / vwap_std
                
                # We want to BUY when z_score is very negative (<-2) and RSI is oversold (<30)
                # We want to SELL when z_score is very positive (>2) and RSI is overbought (>70)
                
                mean_reversion_score = pd.Series(0.0, index=df.index)
                
                buy_condition = (z_score < -1.5) & (rsi < 30)
                sell_condition = (z_score > 1.5) & (rsi > 70)
                
                mean_reversion_score[buy_condition] = 1.0  # Max Buy Pressure
                mean_reversion_score[sell_condition] = -1.0 # Max Sell Pressure
                
                raw_score = mean_reversion_score
            else:
                raw_score = pd.Series(0, index=df.index)

        smoothed_score = raw_score.ewm(span=3).mean().clip(-1.0, 1.0)

        return smoothed_score

    def get_signal(self, current_score: float, dynamic_threshold: float = 0.5) -> str:
        if current_score > dynamic_threshold:
            return "buy"
        elif current_score < -dynamic_threshold:
            return "sell"
        return "no_trade"
