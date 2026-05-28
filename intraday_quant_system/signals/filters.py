def regime_filter(signal: str, regime: str) -> str:
    """
    Filter signals based on current market regime.
    Prevents taking trades in hostile environments.
    
    Regime names from HMM: 'quiet', 'bull_volatile', 'bear_volatile', 'unknown'
    """
    if regime == 'bear_volatile':
        # In bear volatile, block new long entries (high risk of drawdown)
        if signal == 'buy':
            return 'hold'
    
    if regime == 'unknown':
        # Unknown regime — reduce signal confidence, only take strong signals
        # Let it through but signal downstream to reduce size
        pass
        
    return signal

def volatility_filter(signal: str, vix: float, threshold: float = 25.0) -> str:
    """Filter trades if VIX is too high"""
    if vix > threshold:
        return 'hold'
    return signal

def confidence_filter(signal: str, meta_confidence: float, threshold: float = 0.5) -> str:
    """Only take trade if meta-labeler confidence is high enough"""
    if meta_confidence < threshold:
        return 'hold'
    return signal

