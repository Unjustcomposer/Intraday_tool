import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from signals.ensemble import EnsembleScorer

def test_ensemble_weights():
    scorer = EnsembleScorer()
    
    # weights sum to 1.0
    assert sum(scorer.weights.values()) == 1.0
    
def test_ensemble_scoring():
    scorer = EnsembleScorer()
    
    score = scorer.compute_score(
        lgbm_prob=0.9,      # * 0.30 = 0.27
        xgboost_prob=0.5,   # * 0.20 = 0.10 (default)
        tft_prob=0.5,       # * 0.10 = 0.05 (default)
        meta_prob=0.7,      # * 0.20 = 0.14
        sentiment_score=0.8,# normalized to 0.9 * 0.10 = 0.09
        regime_score=1.0    # * 0.10 = 0.10
    )
    # expected: 0.27 + 0.10 + 0.05 + 0.14 + 0.09 + 0.10 = 0.75
    
    assert abs(score - 0.75) < 0.001
    
    signal = scorer.get_signal(score)
    assert signal == 'buy'
