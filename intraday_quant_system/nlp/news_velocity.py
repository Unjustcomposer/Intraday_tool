import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta

def compute_velocity(articles: List[Dict], window_minutes: int = 30) -> float:
    """
    Articles per minute for the symbol in the rolling window
    """
    if not articles:
        return 0.0
        
    now = datetime.now()
    cutoff = now - timedelta(minutes=window_minutes)
    
    count = 0
    for article in articles:
        pub_time = article.get('published_at')
        if pub_time and pub_time >= cutoff:
            count += 1
            
    return count / window_minutes

def social_sentiment_score(symbol: str, lookback_minutes: int = 60) -> float:
    """
    Mock function. In a real system, would query a social media API (e.g. Twitter/StockTwits)
    """
    # Placeholder
    return 0.5
