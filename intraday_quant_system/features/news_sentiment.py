import logging
import yfinance as yf
from datetime import datetime, timedelta
import feedparser

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

logger = logging.getLogger(__name__)

class NewsSentimentEngine:
    """
    Fetches real-time news via yfinance and macro feeds.
    Uses FinBERT (ProsusAI/finbert) for institutional-grade financial NLP sentiment analysis.
    """
    
    def __init__(self, hours_lookback: int = 48):
        self.hours_lookback = hours_lookback
        self.macro_sentiment = 0.0
        self.macro_fetched = False
        
        if pipeline:
            logger.info("Initializing FinBERT NLP pipeline (ProsusAI/finbert)...")
            try:
                # Use a specific financial sentiment model
                self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                self.nlp = None
        else:
            logger.warning("transformers not installed. Returning neutral sentiment 0.0.")
            self.nlp = None

    def _score_text(self, text: str) -> float:
        """Scores a text using FinBERT. Returns [-1.0 to 1.0]"""
        if not self.nlp or not text:
            return 0.0
        try:
            res = self.nlp(text[:512])[0] # Truncate to 512 chars
            label = res['label'].lower()
            score = res['score']
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:
                return 0.0
        except Exception:
            return 0.0

    def fetch_macro_sentiment(self) -> float:
        """Fetches macro news using NIFTY 50 and global feeds to gauge systemic sentiment."""
        if self.macro_fetched:
            return self.macro_sentiment
            
        logger.info("Fetching systemic macroeconomic sentiment...")
        try:
            # Proxy macro news via India ETF and major index (yfinance news)
            macro_tickers = ['^NSEI', 'INDA']
            total_score = 0.0
            count = 0
            
            for t in macro_tickers:
                ticker = yf.Ticker(t)
                news = ticker.news
                if not news:
                    continue
                for item in news:
                    title = item.get("title", "")
                    if title:
                        total_score += self._score_text(title)
                        count += 1
                        
            if count > 0:
                self.macro_sentiment = total_score / count
            
            self.macro_fetched = True
            logger.info(f"Macro Systemic Sentiment: {self.macro_sentiment:.2f} based on {count} events.")
            return self.macro_sentiment
        except Exception as e:
            logger.error(f"Macro sentiment fetch failed: {e}")
            return 0.0

    def compute_sentiment(self, symbol: str) -> float:
        """
        Computes weighted NLP sentiment (70% Micro / 30% Macro).
        """
        if not self.nlp:
            return 0.0
            
        # Ensure macro is fetched once per run
        macro_score = self.fetch_macro_sentiment()
            
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            
            if not news_items:
                return round(macro_score * 0.3, 2)
                
            total_polarity = 0.0
            count = 0
            
            for item in news_items:
                title = item.get("title", "")
                if title:
                    total_polarity += self._score_text(title)
                    count += 1
                    
            if count == 0:
                micro_score = 0.0
            else:
                micro_score = total_polarity / count
                
            # Blend 70% micro, 30% macro
            blended_score = (micro_score * 0.7) + (macro_score * 0.3)
            logger.debug(f"[{symbol}] FinBERT Blended Sentiment: {blended_score:.2f} (Micro: {micro_score:.2f}, Macro: {macro_score:.2f})")
            return round(blended_score, 2)
            
        except Exception as e:
            logger.error(f"[{symbol}] Failed to fetch micro sentiment: {e}")
            return round(macro_score * 0.3, 2)
