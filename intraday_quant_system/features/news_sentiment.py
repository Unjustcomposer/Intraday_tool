import logging
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import urllib.parse
import os

logger = logging.getLogger(__name__)

# Lazy loading of transformers to save memory if NLP is disabled
_nlp_pipeline = None

def get_finbert_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        try:
            from transformers import pipeline
            import warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            logger.info("Loading FinBERT model into memory (this may take a moment)...")
            # Using ProsusAI/finbert as it is specifically trained on financial text
            _nlp_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            return None
    return _nlp_pipeline

class NewsSentimentEngine:
    """
    Fetches real-time news via RSS and applies FinBERT for sentiment analysis.
    """
    
    MAX_CACHE_SIZE = 200
    CACHE_TTL_SECONDS = 900  # 15 minutes
    MIN_REQUEST_INTERVAL = 2.0  # seconds between RSS requests
    
    def __init__(self, hours_lookback: int = 48):
        self.hours_lookback = hours_lookback
        self.cache = {}  # {symbol: {'data': [...], 'timestamp': datetime}}
        self._last_request_time = None
        
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still fresh."""
        if symbol not in self.cache:
            return False
        entry = self.cache[symbol]
        age = (datetime.now() - entry['timestamp']).total_seconds()
        return age < self.CACHE_TTL_SECONDS
    
    def _evict_oldest(self):
        """Evict oldest cache entries if over size limit."""
        if len(self.cache) >= self.MAX_CACHE_SIZE:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
    def fetch_news(self, symbol: str) -> list:
        """Fetch news from Google News RSS for a specific NSE stock."""
        if self._is_cache_valid(symbol):
            return self.cache[symbol]['data']
        
        # Rate limiting
        import time
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.MIN_REQUEST_INTERVAL:
                time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            
        clean_sym = symbol.replace('.NS', '')
        query = urllib.parse.quote(f"{clean_sym} NSE stock")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        try:
            self._last_request_time = time.time()
            feed = feedparser.parse(url)
            articles = []
            cutoff_time = datetime.now().astimezone() - timedelta(hours=self.hours_lookback)
            
            for entry in feed.entries[:10]:  # Limit to top 10 to save processing time
                try:
                    # feedparser dates are parsed as struct_time
                    from email.utils import parsedate_to_datetime
                    pub_date = parsedate_to_datetime(entry.published)
                    if pub_date >= cutoff_time:
                        articles.append({
                            'title': entry.title,
                            'published_at': pub_date,
                            'source': entry.source.title if hasattr(entry, 'source') else 'Unknown'
                        })
                except Exception:
                    continue
            
            self._evict_oldest()
            self.cache[symbol] = {'data': articles, 'timestamp': datetime.now()}
            return articles
            
        except Exception as e:
            logger.error(f"[{symbol}] Error fetching news: {e}")
            return []

    def compute_sentiment(self, symbol: str) -> float:
        """
        Compute a continuous sentiment score (-1.0 to 1.0).
        Positive = 1.0, Neutral = 0.0, Negative = -1.0.
        Scores are volume-weighted by confidence and time-decayed.
        """
        articles = self.fetch_news(symbol)
        if not articles:
            return 0.0
            
        nlp = get_finbert_pipeline()
        if not nlp:
            return 0.0
            
        total_score = 0.0
        weight_sum = 0.0
        now = datetime.now().astimezone()
        
        for article in articles:
            text = article['title']
            try:
                result = nlp(text)[0]
                label = result['label']
                score = result['score']
                
                # Convert label to numeric multiplier
                if label == 'positive':
                    val = score
                elif label == 'negative':
                    val = -score
                else:
                    val = 0.0
                    
                # Exponential time decay based on hours old
                hours_old = (now - article['published_at']).total_seconds() / 3600.0
                decay_weight = max(0.1, 1.0 * (0.9 ** hours_old))
                
                total_score += val * decay_weight
                weight_sum += decay_weight
                
            except Exception as e:
                logger.debug(f"Failed to score article '{text}': {e}")
                continue
                
        if weight_sum == 0:
            return 0.0
            
        final_sentiment = total_score / weight_sum
        return round(final_sentiment, 3)

    def generate_news_features(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        For a full historical backtest, fetching 60 days of news is impossible via RSS.
        In this implementation, we only use live sentiment for the CURRENT bar veto.
        For the feature dataframe, we just fill 0 (as backtesting NLP without a massive DB is not feasible).
        """
        df['finbert_sentiment'] = 0.0
        return df
