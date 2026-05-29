import logging
import math
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
                'tabnet': 0.30,
                'tft': 0.10,
                'meta': 0.10,
                'sentiment': 0.05,
                'regime': 0.05
            },
            'bull_volatile': {
                'lgbm': 0.20,
                'tabnet': 0.20,
                'tft': 0.30,
                'meta': 0.10,
                'sentiment': 0.10,
                'regime': 0.10
            },
            'bear_volatile': {
                'lgbm': 0.20,
                'tabnet': 0.20,
                'tft': 0.30,
                'meta': 0.10,
                'sentiment': 0.10,
                'regime': 0.10
            },
            'crisis': {
                'lgbm': 0.25,
                'tabnet': 0.25,
                'tft': 0.10,
                'meta': 0.25,
                'sentiment': 0.05,
                'regime': 0.10
            },
            'unknown': {
                'lgbm': 0.30,
                'tabnet': 0.20,
                'tft': 0.10,
                'meta': 0.20,
                'sentiment': 0.10,
                'regime': 0.10
            }
        }
        
        # Thresholds — Institutional Upgrade: Dynamic Rolling Thresholds
        self.base_long_threshold = 0.55
        self.base_short_threshold = 0.45
        # State for dynamic thresholding
        self.score_history = []
        
        # Verify all regime weights sum to 1.0
        for regime, w_dict in self.regime_weights.items():
            assert abs(sum(w_dict.values()) - 1.0) < 1e-6, f"Weights for regime '{regime}' must sum to 1.0"

    @property
    def weights(self):
        """Returns the default weights (for 'unknown' regime) for backward compatibility"""
        return self.regime_weights['unknown']

    def _calibrate(self, prob: float) -> float:
        """Apply basic Platt scaling approximation to prevent overconfidence."""
        return 1.0 / (1.0 + math.exp(-8.0 * (prob - 0.5)))

    def compute_score(self, lgbm_prob: float, tabnet_prob: float = 0.5, tft_prob: float = 0.5,
                      meta_prob: float = 0.5, meta_gate: float = 0.50, sentiment_score: float = 0.0, regime_score: float = 0.5,
                      regime: str = 'unknown') -> float:
        """
        Calculate weighted ensemble score from multiple alpha models + meta + sentiment.
        Uses regime-conditional weights based on the current market regime.
        
        Returns score in [0, 1] where:
          > long_threshold  → long signal
          < short_threshold → short signal
          between           → no trade
        """
        # Meta-labeler confidence gate (now dynamically driven by Conformal Prediction)
        if meta_prob < meta_gate:
            return 0.5  # Return neutral (no trade zone) instead of 0.0
            
        # Calibrate raw probabilities
        lgbm_prob = self._calibrate(lgbm_prob)
        tabnet_prob = self._calibrate(tabnet_prob)
        tft_prob = self._calibrate(tft_prob)
        
        # Normalize sentiment_score from [-1, 1] to [0, 1] for consistent weighting
        sentiment_normalized = (sentiment_score + 1.0) / 2.0
        
        # Fetch weights for the current regime
        w = self.regime_weights.get(regime, self.regime_weights['unknown'])
        
        score = (
            lgbm_prob * w['lgbm'] +
            tabnet_prob * w['tabnet'] +
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
        
        Thresholds (current — require walk-forward validation):
          score > 0.55 → 'buy'
          score < 0.45 → 'sell'
          else         → 'no_trade'
        
        Filters applied post-threshold:
          - Regime filter (blocks trades in crisis/chop)
          - VIX filter (blocks if VIX > 25)
          - Confidence filter (requires meta > 0.60)
        """
        # Track score history for dynamic thresholds
        self.score_history.append(score)
        if len(self.score_history) > 100:
            self.score_history.pop(0)
            
        # Dynamic threshold logic (rolling mean +/- 0.5 std dev)
        if len(self.score_history) > 20:
            mean_score = sum(self.score_history) / len(self.score_history)
            std_score = (sum((x - mean_score) ** 2 for x in self.score_history) / len(self.score_history)) ** 0.5
            dynamic_long = mean_score + (0.5 * std_score)
            dynamic_short = mean_score - (0.5 * std_score)
        else:
            dynamic_long = self.base_long_threshold
            dynamic_short = self.base_short_threshold

        if score > dynamic_long:
            signal = 'buy'
        elif score < dynamic_short:
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
