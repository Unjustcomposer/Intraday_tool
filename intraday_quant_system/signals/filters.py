def regime_filter(signal: str, regime: str) -> str:
    """
    Filter signals based on current market regime.
    Prevents taking trades in hostile environments.
    
    Regime names from GMM detector: 'quiet', 'trending', 'volatile', 'crisis', 'unknown'
    """
    if regime == 'crisis':
        # Crisis regime — block ALL new trades. Capital preservation is paramount.
        if signal in ('buy', 'sell'):
            return 'hold'
    
    if regime == 'volatile':
        # In volatile regime, block new long entries (high risk of drawdown)
        if signal == 'buy':
            return 'hold'
    
    if regime == 'unknown':
        # Unknown regime — let signals through but downstream filters will gate
        pass
        
    return signal

def volatility_filter(signal: str, vix: float, threshold: float = 30.0) -> str:
    """Filter trades if VIX is too high"""
    if vix > threshold:
        return 'hold'
    return signal

def confidence_filter(signal: str, meta_confidence: float, threshold: float = 0.40) -> str:
    """Only take trade if meta-labeler confidence is high enough"""
    if meta_confidence < threshold:
        return 'hold'
    return signal
