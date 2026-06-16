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
        # Regime-conditional weights mapping (LGBM + TabNet Decorrelation)
        self.regime_weights = {
            'quiet': {
                'lgbm': 0.30, 'tabnet': 0.30, 'meta': 0.20, 'regime': 0.10, 'sentiment': 0.10
            },
            'trending': {
                'lgbm': 0.25, 'tabnet': 0.25, 'meta': 0.25, 'regime': 0.15, 'sentiment': 0.10
            },
            'volatile': {
                'lgbm': 0.20, 'tabnet': 0.20, 'meta': 0.35, 'regime': 0.15, 'sentiment': 0.10
            },
            'crisis': {
                'lgbm': 0.15, 'tabnet': 0.15, 'meta': 0.45, 'regime': 0.15, 'sentiment': 0.10
            },
            'unknown': {
                'lgbm': 0.25, 'tabnet': 0.25, 'meta': 0.30, 'regime': 0.10, 'sentiment': 0.10
            }
        }
        
        # Thresholds — Institutional Upgrade: Dynamic Rolling Thresholds
        self.base_long_threshold = 0.51
        self.base_short_threshold = 0.49
        # State for dynamic thresholding
        self.score_history = {}
        
        # Verify all regime weights sum to 1.0
        for regime, w_dict in self.regime_weights.items():
            assert abs(sum(w_dict.values()) - 1.0) < 1e-6, f"Weights for regime '{regime}' must sum to 1.0"

    @property
    def weights(self):
        """Returns the default weights (for 'unknown' regime) for backward compatibility"""
        return self.regime_weights['unknown']

    def compute_score(self, lgbm_prob: float, tabnet_prob: float = 0.5, meta_prob: float = 0.5,
                      sentiment_score: float = 0.0, regime_score: float = 0.5,
                      meta_gate: float = 0.0, regime: str = 'unknown', symbol: str = 'unknown') -> float:
        """
        Calculate weighted ensemble score from lgbm + tabnet + meta + regime + sentiment.
        Uses regime-conditional weights based on the current market regime.
        Raw probabilities are used directly (already Isotonic-calibrated upstream).
        
        Returns score in [0, 1] where:
          > long_threshold  → long signal
          < short_threshold → short signal
          between           → no trade
        """
        sentiment_normalized = (sentiment_score + 1.0) / 2.0
        
        w = self.regime_weights.get(regime, self.regime_weights['unknown'])
        
        score = (
            lgbm_prob * w['lgbm'] +
            tabnet_prob * w['tabnet'] +
            meta_prob * w['meta'] +
            regime_score * w['regime'] +
            sentiment_normalized * w['sentiment']
        )
        return score

    def get_signal(self, score: float, symbol: str = 'unknown', regime: str = 'unknown',
                   vix: float = 0.0, meta_confidence: float = 1.0, sentiment_score: float = 0.5,
                   conformal_threshold: float = 0.5) -> str:
        """
        Generate signal with integrated filters.
        
        Thresholds: dynamically calculated based on rolling history.
        
        Filters applied post-threshold:
          - Regime filter (blocks trades in crisis/chop)
          - VIX filter (blocks if VIX > 25)
          - Confidence filter (requires meta > 0.60)
          - Pre-market NLP filter (blocks if sentiment is extremely negative)
        """
        # --- Conformal Prediction confidence gate ---
        if abs(score - 0.5) < abs(conformal_threshold - 0.5):
            logger.info(f"[{symbol}] Score {score:.4f} inside conformal band (threshold={conformal_threshold:.4f}), no trade")
            return 'no_trade'

        # --- NLP Pre-Market Risk Filter ---
        # If sentiment is disastrously low, block the trade entirely.
        if sentiment_score < -0.6:  # Normalized to -1 to 1. Severe negative news blocks all trades.
            logger.info(f"[{symbol}] Trade blocked due to severe negative NLP sentiment ({sentiment_score})")
            return 'no_trade'
        if symbol not in self.score_history:
            self.score_history[symbol] = []
            
        # Track score history for dynamic thresholds
        self.score_history[symbol].append(score)
        if len(self.score_history[symbol]) > 100:
            self.score_history[symbol].pop(0)
            
        # Dynamic threshold logic (rolling mean +/- 1.5 std dev)
        history = self.score_history[symbol]
        if len(history) > 20:
            mean_score = sum(history) / len(history)
            std_score = (sum((x - mean_score) ** 2 for x in history) / len(history)) ** 0.5
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
