import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from features.flow_features import relative_volume, vwap_deviation

def test_relative_volume():
    # Setup mock data
    df = pd.DataFrame({
        'volume': [100] * 19 + [200]
    })
    # Set daily datetime index so all times are 09:15:00
    df.index = pd.date_range('2023-01-01 09:15:00', periods=20, freq='1D')
    
    rvol = relative_volume(df, lookback_days=20)
    
    # average volume of previous 19 days is 100. So rvol should be 200 / 100 = 2.0
    assert np.isclose(rvol.iloc[-1], 2.0)

def test_vwap_deviation():
    df = pd.DataFrame({
        'close': [105.0],
        'vwap': [100.0]
    })
    
    dev = vwap_deviation(df)
    assert np.isclose(dev.iloc[0], 0.05)
