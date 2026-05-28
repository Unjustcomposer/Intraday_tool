import pandas as pd
import numpy as np

def encode_sentiment(finbert_output: dict) -> float:
    """float -1 to 1 based on finbert output {sentiment: str, score: float}"""
    if not finbert_output:
        return 0.0
        
    sentiment = finbert_output.get('sentiment', 'neutral').lower()
    score = finbert_output.get('score', 0.0)
    
    if sentiment == 'positive':
        return score
    elif sentiment == 'negative':
        return -score
    else:
        return 0.0

def earnings_surprise_score(actual: float, estimate: float) -> float:
    """(actual - estimate) / abs(estimate)"""
    if estimate == 0:
        return np.sign(actual) * 1.0 # arbitrary cap
    return (actual - estimate) / abs(estimate)

def guidance_change_score(event_text: str) -> float:
    """Mock function. In reality, via FinBERT"""
    # Simple keyword based mock
    text = event_text.lower()
    if 'raise' in text or 'upgrade' in text:
        return 0.8
    elif 'lower' in text or 'downgrade' in text or 'cut' in text:
        return -0.8
    return 0.0

def event_severity(event_type: str) -> float:
    """lookup table"""
    severity_map = {
        'earnings': 0.9,
        'guidance': 1.0,
        'merger': 0.8,
        'regulatory': 0.7,
        'macro': 0.6,
        'rating_change': 0.5,
        'insider': 0.4,
        'other': 0.1
    }
    return severity_map.get(event_type.lower(), 0.1)
