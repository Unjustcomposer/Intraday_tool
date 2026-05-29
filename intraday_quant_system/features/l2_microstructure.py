import pandas as pd
import numpy as np

class L2Microstructure:
    """
    Computes Level 2 Market Microstructure Features.
    Provides institutional metrics like Order Imbalance and VPIN.
    """
    @staticmethod
    def order_imbalance(bid_vols: list, ask_vols: list, levels: int = 5) -> float:
        """OIB = (Total Bid Vol - Total Ask Vol) / (Total Bid Vol + Total Ask Vol)"""
        bid_sum = sum(bid_vols[:levels])
        ask_sum = sum(ask_vols[:levels])
        if bid_sum + ask_sum == 0:
            return 0.0
        return (bid_sum - ask_sum) / (bid_sum + ask_sum)

    @staticmethod
    def quote_slope(bid_prices: list, ask_prices: list, bid_vols: list, ask_vols: list) -> float:
        """Slope of liquidity on bid vs ask side"""
        if len(bid_prices) < 2 or len(ask_prices) < 2:
            return 0.0
        bid_slope = (bid_vols[-1] - bid_vols[0]) / (bid_prices[0] - bid_prices[-1] + 1e-6)
        ask_slope = (ask_vols[-1] - ask_vols[0]) / (ask_prices[-1] - ask_prices[0] + 1e-6)
        return bid_slope - ask_slope

    @staticmethod
    def vpin_proxy(df_ticks: pd.DataFrame, bucket_vol: int = 50000) -> float:
        """
        Proxy for Volume-Synchronized Probability of Informed Trading (VPIN).
        Requires tick data with 'price', 'volume', 'aggressor_side'.
        """
        if df_ticks.empty or 'volume' not in df_ticks.columns or 'aggressor_side' not in df_ticks.columns:
            return 0.0
            
        df = df_ticks.copy()
        df['cum_vol'] = df['volume'].cumsum()
        df['bucket'] = df['cum_vol'] // bucket_vol
        
        # Calculate buy/sell volume per bucket
        df['buy_vol'] = np.where(df['aggressor_side'] == 1, df['volume'], 0)
        df['sell_vol'] = np.where(df['aggressor_side'] == -1, df['volume'], 0)
        
        buckets = df.groupby('bucket').agg({'buy_vol': 'sum', 'sell_vol': 'sum'})
        if buckets.empty:
            return 0.0
            
        vpin = abs(buckets['buy_vol'] - buckets['sell_vol']).sum() / (buckets['buy_vol'].sum() + buckets['sell_vol'].sum())
        return vpin
