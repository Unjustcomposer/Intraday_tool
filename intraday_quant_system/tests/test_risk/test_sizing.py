import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from risk.position_sizing import kelly_fraction, volatility_adjusted_size, PortfolioLimits

def test_kelly_fraction():
    win_prob = 0.55
    win_loss_ratio = 2.0
    
    kf = kelly_fraction(win_prob, win_loss_ratio)
    
    # f = 0.55 - 0.45/2.0 = 0.55 - 0.225 = 0.325
    # half-kelly = 0.1625
    assert abs(kf - 0.1625) < 0.0001

def test_volatility_adjusted_size():
    target_vol = 0.15
    asset_vol = 0.30
    capital = 100000.0
    
    size = volatility_adjusted_size(target_vol, asset_vol, capital)
    
    assert size == 50000.0

def test_portfolio_limits():
    # Test valid trade
    valid = PortfolioLimits.check_trade(
        capital=100000,
        current_positions=2,
        proposed_risk=1500, # 1.5%
        current_sector_exposure=0.10,
        current_portfolio_exposure=0.50
    )
    assert valid is True
    
    # Test max risk breach
    invalid_risk = PortfolioLimits.check_trade(
        capital=100000,
        current_positions=2,
        proposed_risk=2500, # 2.5% > 2% limit
        current_sector_exposure=0.10,
        current_portfolio_exposure=0.50
    )
    assert invalid_risk is False
