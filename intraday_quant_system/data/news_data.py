import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta
import feedparser

logger = logging.getLogger(__name__)


class NewsFetcher:
    """
    Fetches news from configured RSS/API feeds.
    Stores raw article text with symbol tag and timestamp.
    
    Production notes:
      - Uses feedparser for real RSS feed parsing
      - Falls back to empty list on network errors (never blocks trading)
      - Caches articles to avoid re-processing
    """
    def __init__(self, feeds: List[str] = None):
        self.feeds = feeds or [
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.moneycontrol.com/rss/marketreports.xml",
        ]
        self._article_cache: Dict[str, dict] = {}  # keyed by headline hash to avoid duplicates
        
    def fetch_recent_news(self, hours_back: int = 24, symbols: List[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch news from RSS feeds and format them.
        Output: List[dict] with keys [headline, body, symbol, published_at, source]
        
        Non-blocking: returns empty list on any network error.
        """
        logger.info(f"Fetching news from {len(self.feeds)} feeds")
        articles = []
        cutoff = datetime.now() - timedelta(hours=hours_back)
        symbols_lower = [s.lower() for s in (symbols or [])]
        
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                if feed.bozo and not feed.entries:
                    logger.warning(f"Feed parse error for {feed_url}: {feed.bozo_exception}")
                    continue
                
                for entry in feed.entries:
                    headline = entry.get('title', '')
                    body = entry.get('summary', entry.get('description', ''))
                    
                    # Parse published date
                    pub_time = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            pub_time = datetime(*entry.published_parsed[:6])
                        except (TypeError, ValueError):
                            pub_time = datetime.now()
                    else:
                        pub_time = datetime.now()
                    
                    # Skip old articles
                    if pub_time < cutoff:
                        continue
                    
                    # Try to match symbol from headline/body
                    matched_symbol = self._match_symbol(headline + " " + body, symbols_lower, symbols or [])
                    
                    # Deduplicate by headline
                    headline_key = headline.strip().lower()
                    if headline_key in self._article_cache:
                        continue
                    
                    article = {
                        "headline": headline,
                        "body": body,
                        "symbol": matched_symbol,
                        "published_at": pub_time,
                        "source": feed_url
                    }
                    
                    self._article_cache[headline_key] = article
                    articles.append(article)
                    
            except Exception as e:
                # Non-blocking: log and continue, never halt trading for news
                logger.warning(f"Failed to fetch feed {feed_url}: {e}")
                continue
        
        logger.info(f"Fetched {len(articles)} new articles")
        return articles
    
    @staticmethod
    def _match_symbol(text: str, symbols_lower: List[str], symbols_original: List[str]) -> str:
        """Match stock symbol mentioned in text. Returns empty string if no match."""
        text_lower = text.lower()
        for sym_lower, sym_orig in zip(symbols_lower, symbols_original):
            # Check both the symbol and common variations
            clean_sym = sym_lower.replace('.ns', '').replace('.bo', '')
            if clean_sym in text_lower:
                return sym_orig
        return ""
        
    def store_news(self, articles: List[Dict[str, Any]]):
        """Store articles in database"""
        if not articles:
            return
        logger.info(f"Storing {len(articles)} articles to database")
        # DB insertion logic here — in production, write to TimescaleDB or similar
    
    def clear_cache(self):
        """Clear article cache (call at end of trading day)"""
        self._article_cache.clear()
