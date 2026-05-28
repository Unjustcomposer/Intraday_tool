import logging
from signals.filters import regime_filter, volatility_filter, confidence_filter

logger = logging.getLogger(__name__)

class EnsembleScorer:
    """
    EnsembleScorer — Symmetric Signal Generation
    
    Architecture:
      1. Separate long/short score computation (eliminates long bias)
      2. Meta-labeler confidence gate (≥0.65 to take any trade)
      3. Signal filters integrated (regime, volatility, confidence)
    
    Weights:
      lgbm_weight       = 0.60
      meta_weight       = 0.20
      sentiment_weight  = 0.10
      regime_weight     = 0.10
    """
    def __init__(self):
        # Regime-conditional weights mapping
        self.regime_weights = {
            'quiet': {
                'lgbm': 0.40,
                'xgboost': 0.30,
                'tft': 0.10,
                'meta': 0.10,
                'sentiment': 0.05,
                'regime': 0.05
            },
            'bull_volatile': {
                'lgbm': 0.20,
                'xgboost': 0.20,
                'tft': 0.30,
                'meta': 0.10,
                'sentiment': 0.10,
                'regime': 0.10
            },
            'bear_volatile': {
                'lgbm': 0.20,
                'xgboost': 0.20,
                'tft': 0.30,
                'meta': 0.10,
                'sentiment': 0.10,
                'regime': 0.10
            },
            'crisis': {
                'lgbm': 0.10,
                'xgboost': 0.10,
                'tft': 0.40,
                'meta': 0.20,
                'sentiment': 0.10,
                'regime': 0.10
            },
            'unknown': {
                'lgbm': 0.30,
                'xgboost': 0.20,
                'tft': 0.10,
                'meta': 0.20,
                'sentiment': 0.10,
                'regime': 0.10
            }
        }
        
        # Thresholds — calibrate via walk-forward, never on in-sample data
        self.long_threshold = 0.70    # Lowered from 0.82 (was too restrictive)
        self.short_threshold = 0.30   # Raised from 0.18 (symmetric from center)
        self.meta_gate = 0.65
        
        # Verify all regime weights sum to 1.0
        for regime, w_dict in self.regime_weights.items():
            assert abs(sum(w_dict.values()) - 1.0) < 1e-6, f"Weights for regime '{regime}' must sum to 1.0"

    @property
    def weights(self):
        """Returns the default weights (for 'unknown' regime) for backward compatibility"""
        return self.regime_weights['unknown']

    def compute_score(self, lgbm_prob: float, xgboost_prob: float = 0.5, tft_prob: float = 0.5,
                      meta_prob: float = 0.5, sentiment_score: float = 0.0, regime_score: float = 0.5,
                      regime: str = 'unknown') -> float:
        """
        Calculate weighted ensemble score from multiple alpha models + meta + sentiment.
        Uses regime-conditional weights based on the current market regime.
        
        Returns score in [0, 1] where:
          > long_threshold  → long signal
          < short_threshold → short signal
          between           → no trade
        """
        # Meta-labeler confidence gate
        if meta_prob < self.meta_gate:
            return 0.5  # Return neutral (no trade zone) instead of 0.0
        
        # Normalize sentiment_score from [-1, 1] to [0, 1] for consistent weighting
        sentiment_normalized = (sentiment_score + 1.0) / 2.0
        
        # Fetch weights for the current regime
        w = self.regime_weights.get(regime, self.regime_weights['unknown'])
        
        score = (
            lgbm_prob * w['lgbm'] +
            xgboost_prob * w['xgboost'] +
            tft_prob * w['tft'] +
            meta_prob * w['meta'] +
            sentiment_normalized * w['sentiment'] +
            regime_score * w['regime']
        )
        
        return score

    def get_signal(self, score: float, regime: str = 'unknown',
                   vix: float = 0.0, meta_confidence: float = 1.0) -> str:
        """
        Generate signal with integrated filters.
        
        Thresholds are symmetric around 0.5:
          score > 0.70 → 'buy'
          score < 0.30 → 'sell'
          else         → 'no_trade'
        
        Filters applied post-threshold:
          - Regime filter (blocks trades in crisis/chop)
          - VIX filter (blocks if VIX > 25)
          - Confidence filter (requires meta > 0.60)
        """
        if score > self.long_threshold:
            signal = 'buy'
        elif score < self.short_threshold:
            signal = 'sell'
        else:
            return 'no_trade'
        
        # Apply signal filters (previously in filters.py but never called)
        signal = regime_filter(signal, regime)
        if signal == 'hold':
            return 'no_trade'
        
        if vix > 0:
            signal = volatility_filter(signal, vix)
            if signal == 'hold':
                return 'no_trade'
        
        signal = confidence_filter(signal, meta_confidence)
        if signal == 'hold':
            return 'no_trade'
        
        return signal
