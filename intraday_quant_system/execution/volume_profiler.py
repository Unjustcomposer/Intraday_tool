import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger(__name__)

class VolumeProfiler:
    """
    Computes normalized historical volume profiles (e.g., U-shaped curve)
    for use in VWAP execution slicing.
    """
    
    @staticmethod
    def compute_intraday_profile(df: pd.DataFrame, time_interval: str = '15min') -> List[float]:
        """
        Takes a historical intraday DataFrame and computes the average percentage 
        of daily volume that occurs in each time bucket.
        
        Returns a list of fractions that sum to 1.0.
        """
        if df.empty or 'volume' not in df.columns:
            logger.warning("Empty dataframe or no volume column. Returning flat profile.")
            return VolumeProfiler.flat_profile(25) # Default 25 bins (375 mins / 15 mins)
            
        # Ensure index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'timestamp' in df.columns:
                df = df.set_index(pd.to_datetime(df['timestamp']))
            else:
                logger.warning("No datetime index found. Returning flat profile.")
                return VolumeProfiler.flat_profile(25)
                
        # Group by time of day
        df['_time'] = df.index.time
        df['_date'] = df.index.date
        
        # Calculate total daily volume to normalize
        daily_vol = df.groupby('_date')['volume'].sum()
        df['_daily_total'] = df['_date'].map(daily_vol)
        
        if (df['_daily_total'] == 0).all():
            return VolumeProfiler.flat_profile(25)
            
        # Fraction of daily volume for each bar
        df['_vol_fraction'] = df['volume'] / df['_daily_total'].replace(0, np.nan)
        
        # Average fraction per time bucket across all days
        avg_profile = df.groupby('_time')['_vol_fraction'].mean()
        
        # Normalize to ensure it sums exactly to 1.0
        sum_frac = avg_profile.sum()
        if sum_frac == 0 or np.isnan(sum_frac):
            return VolumeProfiler.flat_profile(len(avg_profile) if len(avg_profile) > 0 else 25)
            
        normalized_profile = (avg_profile / sum_frac).tolist()
        
        # Clean up temporary columns
        df.drop(columns=['_time', '_date', '_daily_total', '_vol_fraction'], inplace=True, errors='ignore')
        
        return normalized_profile
        
    @staticmethod
    def flat_profile(bins: int = 25) -> List[float]:
        """Returns a flat (TWAP) profile"""
        if bins <= 0:
            return [1.0]
        return [1.0 / bins] * bins
