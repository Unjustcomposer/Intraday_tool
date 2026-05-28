class VolatilityStopEngine:
    """
    Adaptive Stop Loss Engine based on Garman-Klass Volatility and Market Regime.
    
    Production features:
      - Trailing stop functionality
      - Gap risk detection
      - Handles both annualized % vol (GK, realized) and price-unit vol (ATR)
    """
    def __init__(self):
        # Volatility multipliers for different regimes
        self.regime_multipliers = {
            'quiet': 2.0,
            'bull_volatile': 3.0,
            'bear_volatile': 3.0,
            'unknown': 2.0
        }
    
    @staticmethod
    def _vol_to_price_units(volatility: float, reference_price: float) -> float:
        """
        Convert volatility to price units.
        If vol < 1.0, treat as annualized percentage (e.g., 0.25 = 25%) and convert:
            daily_vol = annual_vol / sqrt(252)
            bar_vol = daily_vol / sqrt(75)  # 75 bars per day
            price_move = reference_price * bar_vol
        If vol >= 1.0, treat as already in price units (ATR).
        """
        if volatility < 1.0:
            # Annualized percentage vol → per-bar price movement
            daily_vol = volatility / (252 ** 0.5)
            bar_vol = daily_vol / (75 ** 0.5)
            return reference_price * bar_vol * 10  # ~10 bars of movement for stop
        else:
            # Already in price units (ATR)
            return volatility
        
    def calculate_stop(self, entry_price: float, volatility: float, regime: str, is_long: bool) -> float:
        """Calculate initial static stop loss price"""
        multiplier = self.regime_multipliers.get(regime, 2.0)
        vol_price = self._vol_to_price_units(volatility, entry_price)
        
        if is_long:
            return entry_price - (vol_price * multiplier)
        else:
            return entry_price + (vol_price * multiplier)

    def calculate_trailing_stop(self, current_price: float, current_stop: float, volatility: float, regime: str, is_long: bool) -> float:
        """Calculate trailing stop — only moves in direction of profit"""
        multiplier = self.regime_multipliers.get(regime, 2.0)
        vol_price = self._vol_to_price_units(volatility, current_price)
        
        if is_long:
            new_stop = current_price - (vol_price * multiplier)
            # Stop only moves up
            return max(current_stop, new_stop)
        else:
            new_stop = current_price + (vol_price * multiplier)
            # Stop only moves down
            if current_stop == 0:
                return new_stop
            return min(current_stop, new_stop)
            
    def check_gap_risk(self, open_price: float, previous_close: float, volatility: float, position_is_long: bool) -> dict:
        """
        Evaluate if market gapped significantly against the position at open.
        """
        vol_price = self._vol_to_price_units(volatility, previous_close)
        gap_size = abs(open_price - previous_close)
        is_adverse = (position_is_long and open_price < previous_close) or (not position_is_long and open_price > previous_close)
        
        if is_adverse and gap_size > (vol_price * 2.5):
            return {
                'critical_gap': True,
                'gap_size_vol': gap_size / max(vol_price, 0.01),
                'action': 'immediate_market_exit'
            }
            
        return {
            'critical_gap': False,
            'gap_size_vol': gap_size / max(vol_price, 0.01),
            'action': 'hold'
        }

