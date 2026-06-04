import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def calculate_vpvr(df: pd.DataFrame, bins: int = 50) -> dict:
    """
    Calculate Volume Profile Visible Range (VPVR).
    Returns High Volume Nodes (HVNs) and Low Volume Nodes (LVNs).
    
    Vectorized implementation — avoids iterrows() for performance.
    """
    if len(df) == 0:
        return {'hvns': [], 'lvns': [], 'poc': 0.0}
        
    price_min = df['low'].min()
    price_max = df['high'].max()
    
    if price_min == price_max:
        return {'hvns': [price_min], 'lvns': [], 'poc': price_min}
        
    # Create price bins
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_profile = np.zeros(bins)
    
    # Vectorized volume distribution across bins
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values
    
    for i in range(len(df)):
        p_high = highs[i]
        p_low = lows[i]
        p_vol = volumes[i]
        
        # Determine overlapping bins using vectorized comparison
        overlap_mask = (bin_edges[:-1] <= p_high) & (bin_edges[1:] >= p_low)
        num_overlapping = overlap_mask.sum()
        
        if num_overlapping > 0:
            vol_per_bin = p_vol / num_overlapping
            vol_profile[overlap_mask] += vol_per_bin
            
    # Find peaks (HVNs) and valleys (LVNs)
    # distance=3 ensures we don't pick adjacent bins
    hvn_idx, _ = find_peaks(vol_profile, distance=3)
    lvn_idx, _ = find_peaks(-vol_profile, distance=3)
    
    hvns = sorted(bin_centers[hvn_idx].tolist())
    lvns = sorted(bin_centers[lvn_idx].tolist())
    
    # Point of Control (POC)
    poc_idx = np.argmax(vol_profile)
    poc = bin_centers[poc_idx]
    
    return {
        'hvns': hvns,
        'lvns': lvns,
        'poc': poc
    }

def get_closest_support_resistance(price: float, vpvr: dict) -> tuple:
    """Find the closest structural support (below) and resistance (above)."""
    hvns = np.array(vpvr['hvns'])
    if len(hvns) == 0:
        return None, None
        
    supports = hvns[hvns < price]
    resistances = hvns[hvns > price]
    
    support = supports[-1] if len(supports) > 0 else None
    resistance = resistances[0] if len(resistances) > 0 else None
    
    return support, resistance
