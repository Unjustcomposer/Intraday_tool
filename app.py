"""
News-Driven Intraday Trading Scanner
=====================================
Scans live news feeds for market shocks, identifies affected sectors,
analyzes all major stocks in those sectors, and suggests high-margin trades.

Flow: News Scan -> Sentiment + Shock Detection -> Sector Identification
      -> Technical Analysis on affected stocks -> Ranked Trade Suggestions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import re
import time
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Suppress noisy yfinance/peewee logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)


# ============================================================================
#  SECTOR UNIVERSE — All major NSE stocks mapped by sector
# ============================================================================
SECTOR_STOCKS = {
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM",
        "MPHASIS", "COFORGE", "PERSISTENT", "OFSS"
    ],
    "BANKING": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "BANKBARODA", "PNB", "INDUSINDBK", "FEDERALBNK", "IDFCFIRSTB"
    ],
    "AUTO": [
        "TATAMOTORS", "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT",
        "HEROMOTOCO", "ASHOKLEY", "TVSMOTOR", "BALKRISIND", "MOTHERSON"
    ],
    "PHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
        "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM"
    ],
    "ENERGY": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "ADANIGREEN",
        "TATAPOWER", "BPCL", "IOC", "GAIL", "COALINDIA"
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "GODREJCP",
        "DABUR", "MARICO", "COLPAL", "TATACONSUM", "VBL"
    ],
    "METALS": [
        "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "COALINDIA",
        "NMDC", "SAIL", "NATIONALUM", "JINDALSTEL", "APLAPOLLO"
    ],
    "FINANCE": [
        "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
        "CHOLAFIN", "MUTHOOTFIN", "SHRIRAMFIN", "PFC", "RECLTD"
    ],
    "REALTY": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
        "PHOENIXLTD", "SOBHA", "LODHA", "SUNTECK"
    ],
    "TELECOM": [
        "BHARTIARTL", "IDEA", "TATACOMM", "ROUTE"
    ],
    "INFRA": [
        "LT", "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "GRASIM",
        "AMBUJACEM", "ACC", "SHREECEM", "DALBHARAT"
    ],
    "DEFENCE": [
        "HAL", "BEL", "SOLARINDS", "BHEL", "COCHINSHIP",
        "MAZAGON", "GRSE"
    ],
}

# Flatten for quick lookup: stock -> sector
STOCK_TO_SECTOR = {}
for sector, stocks in SECTOR_STOCKS.items():
    for stock in stocks:
        STOCK_TO_SECTOR[stock.upper()] = sector

ALL_STOCKS = list(STOCK_TO_SECTOR.keys())


# ============================================================================
#  NEWS SCANNER — Fetches and parses live financial news
# ============================================================================
NEWS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.livemint.com/rss/markets",
    "https://feeds.feedburner.com/ndtvprofit-latest",
]

# Keywords that signal a market shock or major event
SHOCK_KEYWORDS = {
    "high_impact": [
        "crash", "plunge", "plummet", "tank", "tumble", "collapse",
        "surge", "soar", "skyrocket", "rally", "boom", "breakout",
        "halt", "ban", "fraud", "scam", "default", "bankruptcy",
        "fda approval", "patent", "merger", "acquisition", "takeover",
        "rate hike", "rate cut", "rbi", "fed", "tariff", "sanction",
        "downgrade", "upgrade", "block deal", "bulk deal",
        "results beat", "results miss", "profit warning",
        "order win", "contract", "deal signed",
    ],
    "medium_impact": [
        "beats estimate", "misses estimate", "guidance", "outlook",
        "expansion", "layoff", "restructuring", "buyback", "dividend",
        "ipo", "listing", "delisting", "stake sale", "investment",
        "revenue growth", "profit growth", "margin expansion",
        "market share", "new product", "launch",
    ]
}

# Sector keywords to detect which sector the news is about
SECTOR_KEYWORDS = {
    "IT": ["it sector", "tech", "software", "digital", "saas", "cloud",
           "cybersecurity", "ai ", "artificial intelligence", "outsourcing",
           "tcs", "infosys", "wipro", "hcl tech", "tech mahindra"],
    "BANKING": ["banking", "bank", "npa", "credit growth", "loan", "deposit",
                "rbi", "interest rate", "monetary policy", "nbfc", "hdfc",
                "icici", "sbi", "kotak", "axis bank"],
    "AUTO": ["auto", "automobile", "vehicle", "ev ", "electric vehicle", "car",
             "two-wheeler", "sales volume", "maruti", "tata motors", "mahindra"],
    "PHARMA": ["pharma", "drug", "fda", "clinical trial", "generic",
               "hospital", "healthcare", "medical", "api ", "patent expiry",
               "sun pharma", "dr reddy", "cipla"],
    "ENERGY": ["energy", "oil", "gas", "crude", "petrol", "diesel", "refinery",
               "power", "solar", "wind", "renewable", "opec", "reliance",
               "ongc", "ntpc", "adani green"],
    "FMCG": ["fmcg", "consumer goods", "food", "beverage", "personal care",
             "rural demand", "urban consumption", "hindustan unilever", "itc",
             "nestle", "britannia"],
    "METALS": ["metal", "steel", "aluminium", "aluminum", "copper", "zinc",
               "iron ore", "mining", "commodity", "tata steel", "hindalco",
               "jsw steel", "vedanta"],
    "FINANCE": ["finance", "insurance", "mutual fund", "amc", "lending",
                "microfinance", "bajaj finance", "sbi life"],
    "REALTY": ["realty", "real estate", "housing", "property", "construction",
               "dlf", "godrej properties", "oberoi"],
    "TELECOM": ["telecom", "5g", "spectrum", "arpu", "broadband", "airtel",
                "jio", "vodafone"],
    "INFRA": ["infra", "infrastructure", "cement", "road", "highway", "port",
              "airport", "railway", "l&t", "adani", "ultratech"],
    "DEFENCE": ["defence", "defense", "military", "missile", "radar",
                "hal", "bel", "bhel", "cochin shipyard"],
}


def sanitize_unicode(text: str) -> str:
    """Sanitize text to prevent UnicodeEncodeError crashes on Windows CP1252 consoles."""
    if not text:
        return ""
    # Remove zero-width spaces
    text = text.replace('\u200b', '')
    # Map fancy/high unicode quotes and hyphens to standard equivalents
    replacements = {
        '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-',
        '\u2022': '*', '\u2026': '...'
    }
    for orig, rep in replacements.items():
        text = text.replace(orig, rep)
        
    try:
        enc = sys.stdout.encoding or 'cp1252'
        return text.encode(enc, errors='replace').decode(enc)
    except Exception:
        return text.encode('ascii', errors='replace').decode('ascii')


async def fetch_feed_async(session, url: str):
    """Asynchronously fetch RSS feed content."""
    try:
        async with session.get(url, timeout=10) as response:
            content = await response.read()
            # parse from string/bytes in a non-blocking way
            return feedparser.parse(content)
    except Exception:
        return None


async def fetch_news_async(hours_back=6):
    """Fetch recent news from multiple RSS feeds concurrently using aiohttp."""
    import aiohttp
    articles = []
    cutoff = datetime.now() - timedelta(hours=hours_back)

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_feed_async(session, url) for url in NEWS_FEEDS]
        feeds = await asyncio.gather(*tasks, return_exceptions=True)
        
        for feed_url, feed in zip(NEWS_FEEDS, feeds):
            if feed is None or isinstance(feed, Exception) or not hasattr(feed, 'entries'):
                continue
                
            for entry in feed.entries:
                headline = entry.get("title", "")
                body = entry.get("summary", entry.get("description", ""))
                body = re.sub(r"<[^>]+>", "", body)

                pub_time = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    try:
                        pub_time = datetime(*entry.published_parsed[:6])
                    except (TypeError, ValueError):
                        pub_time = datetime.now()
                else:
                    pub_time = datetime.now()

                if pub_time < cutoff:
                    continue

                articles.append({
                    "headline": sanitize_unicode(headline),
                    "body": sanitize_unicode(body[:500]),
                    "published_at": pub_time,
                    "source": feed_url.split("/")[2],
                })

    # Deduplicate by headline similarity
    seen = set()
    unique = []
    for a in articles:
        key = a["headline"].strip().lower()[:60]
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


# ============================================================================
#  SENTIMENT & SHOCK DETECTION — Keyword-based (fast, no GPU needed)
# ============================================================================
POSITIVE_WORDS = {
    "surge", "soar", "rally", "boom", "gain", "rise", "jump", "up",
    "profit", "beat", "outperform", "bullish", "upgrade", "buy",
    "growth", "expansion", "positive", "strong", "high", "record",
    "breakout", "approval", "deal", "win", "order", "launch",
    "optimistic", "recovery", "rebound"
}

NEGATIVE_WORDS = {
    "crash", "plunge", "fall", "drop", "slip", "decline", "loss",
    "miss", "weak", "bearish", "downgrade", "sell", "fraud", "scam",
    "default", "bankruptcy", "halt", "ban", "layoff", "warning",
    "negative", "concern", "risk", "threat", "cut", "slash",
    "disappointing", "pressure", "correction", "collapse"
}


def analyze_sentiment(text):
    """Fast keyword-based sentiment scoring. Returns score between -1 and +1."""
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 2)


def detect_shock(headline, body=""):
    """Detect if a news item represents a market shock. Returns (is_shock, impact_level, score)."""
    text = (headline + " " + body).lower()
    high_hits = sum(1 for kw in SHOCK_KEYWORDS["high_impact"] if kw in text)
    med_hits = sum(1 for kw in SHOCK_KEYWORDS["medium_impact"] if kw in text)
    score = high_hits * 3 + med_hits * 1
    if high_hits >= 1:
        return True, "HIGH", score
    elif med_hits >= 2:
        return True, "MEDIUM", score
    return False, "LOW", score


def detect_sectors(headline, body=""):
    """Identify which sectors a news headline is about."""
    text = (headline + " " + body).lower()
    matched = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits >= 1:
            matched[sector] = hits
    # Sort by relevance
    return sorted(matched.keys(), key=lambda s: matched[s], reverse=True)


def detect_specific_stocks(headline, body=""):
    """Find specific stock names mentioned in the news."""
    text = (headline + " " + body).upper()
    found = []
    for stock in ALL_STOCKS:
        # Check for the stock name as a whole word
        if re.search(r'\b' + re.escape(stock) + r'\b', text):
            found.append(stock)
    return found


# ============================================================================
#  TECHNICAL ANALYZER — Same core logic, cleaner interface
# ============================================================================
class IntradayAnalyzer:
    """Analyzes a single stock using technical indicators."""
    def __init__(self, symbol):
        self.symbol = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        self.data = None

    async def fetch_and_analyze(self, max_retries=2):
        """Fetch data + calculate indicators + generate signal. Returns result dict or None."""
        if self.data is not None and not self.data.empty:
            data = self.data
        else:
            data = None
            for attempt in range(max_retries):
                try:
                    ticker = yf.Ticker(self.symbol)
                    data = await asyncio.to_thread(ticker.history, period="5d", interval="5m")
                    if data is not None and not data.empty and len(data) >= 30:
                        break
                    data = None
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.5)

        if data is None or data.empty or len(data) < 30:
            return None
        self.data = data
        df = self.data.copy()

        # Core indicators
        df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

        # RSI using Wilder's smoothing (EMA with alpha=1/14, equivalent to com=13)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).ewm(com=13, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

        # VWAP with proper daily reset (cumulative within each trading day only)
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        vol_tp = df["Volume"] * typical_price
        if hasattr(df.index, 'date'):
            day_groups = pd.Series(df.index.date, index=df.index)
        else:
            day_groups = pd.Series(0, index=df.index)  # Fallback: treat as single day
        df["VWAP"] = vol_tp.groupby(day_groups).cumsum() / df["Volume"].groupby(day_groups).cumsum()

        hl = df["High"] - df["Low"]
        hc = np.abs(df["High"] - df["Close"].shift())
        lc = np.abs(df["Low"] - df["Close"].shift())
        df["ATR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

        df["Vol_MA"] = df["Volume"].rolling(20).mean()
        df["Vol_Ratio"] = df["Volume"] / df["Vol_MA"]

        df["BB_Mid"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
        df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        buy = 0
        sell = 0
        reasons = []

        # EMA
        if latest["EMA_9"] > latest["EMA_21"] and prev["EMA_9"] <= prev["EMA_21"]:
            buy += 2; reasons.append("[+] EMA 9 crossed above EMA 21 (Bullish Crossover)")
        elif latest["EMA_9"] < latest["EMA_21"] and prev["EMA_9"] >= prev["EMA_21"]:
            sell += 2; reasons.append("[-] EMA 9 crossed below EMA 21 (Bearish Crossover)")
        elif latest["EMA_9"] > latest["EMA_21"]:
            buy += 1; reasons.append("[+] EMA 9 > EMA 21 (Uptrend)")
        else:
            sell += 1; reasons.append("[-] EMA 9 < EMA 21 (Downtrend)")

        # RSI
        if latest["RSI"] < 30:
            buy += 2; reasons.append(f"[+] RSI oversold ({latest['RSI']:.1f})")
        elif latest["RSI"] > 70:
            sell += 2; reasons.append(f"[-] RSI overbought ({latest['RSI']:.1f})")

        # MACD
        if latest["MACD"] > latest["Signal_Line"] and prev["MACD"] <= prev["Signal_Line"]:
            buy += 2; reasons.append("[+] MACD bullish crossover")
        elif latest["MACD"] < latest["Signal_Line"] and prev["MACD"] >= prev["Signal_Line"]:
            sell += 2; reasons.append("[-] MACD bearish crossover")

        # VWAP
        if latest["Close"] > latest["VWAP"]:
            buy += 1; reasons.append(f"[+] Price above VWAP ({latest['VWAP']:.2f})")
        else:
            sell += 1; reasons.append(f"[-] Price below VWAP ({latest['VWAP']:.2f})")

        # Bollinger
        if latest["Close"] < latest["BB_Lower"]:
            buy += 1; reasons.append("[+] Price at lower Bollinger Band")
        elif latest["Close"] > latest["BB_Upper"]:
            sell += 1; reasons.append("[-] Price at upper Bollinger Band")

        # Volume
        if latest["Vol_Ratio"] > 1.5:
            reasons.append(f"[+] High volume ({latest['Vol_Ratio']:.1f}x avg)")
            if latest["Close"] > prev["Close"]:
                buy += 1
            else:
                sell += 1

        price = latest["Close"]
        atr = latest["ATR"]
        day_high = df["High"].iloc[-20:].max()
        day_low = df["Low"].iloc[-20:].min()
        support = df["Low"].iloc[-40:].min()
        resistance = df["High"].iloc[-40:].max()

        # --- High-profit signal logic with aggressive targets ---
        if buy >= 5 and buy > sell:
            sig = "STRONG BUY"
            entry = round(price - 0.3 * atr, 2)  # Slight dip entry
            sl = round(price - 2.0 * atr, 2)
            t1 = round(price + 2.5 * atr, 2)
            t2 = round(price + 4.0 * atr, 2)
            t3 = round(min(resistance, price + 5.0 * atr), 2)
        elif buy > sell and buy >= 2:
            sig = "BUY"
            entry = round(price, 2)
            sl = round(price - 1.5 * atr, 2)
            t1 = round(price + 2.0 * atr, 2)
            t2 = round(price + 3.0 * atr, 2)
            t3 = round(min(resistance, price + 4.0 * atr), 2)
        elif sell >= 5 and sell > buy:
            sig = "STRONG SELL"
            entry = round(price + 0.3 * atr, 2)
            sl = round(price + 2.0 * atr, 2)
            t1 = round(price - 2.5 * atr, 2)
            t2 = round(price - 4.0 * atr, 2)
            t3 = round(max(support, price - 5.0 * atr), 2)
        elif sell > buy and sell >= 2:
            sig = "SELL"
            entry = round(price, 2)
            sl = round(price + 1.5 * atr, 2)
            t1 = round(price - 2.0 * atr, 2)
            t2 = round(price - 3.0 * atr, 2)
            t3 = round(max(support, price - 4.0 * atr), 2)
        else:
            sig = "HOLD"
            entry = sl = t1 = t2 = t3 = price

        risk = abs(entry - sl)
        reward_t1 = abs(t1 - entry)
        reward_t2 = abs(t2 - entry)
        profit_pct_t1 = round((reward_t1 / entry * 100), 2) if entry > 0 else 0
        profit_pct_t2 = round((reward_t2 / entry * 100), 2) if entry > 0 else 0
        rr = round(reward_t1 / risk, 2) if risk > 0 else 0
        # Suggested qty for INR 1,00,000 capital with 2% risk
        capital = 100000
        risk_amount = capital * 0.02
        qty = int(risk_amount / risk) if risk > 0 else 0
        profit_t1 = round(reward_t1 * qty, 2)
        profit_t2 = round(reward_t2 * qty, 2)

        return {
            "symbol": self.symbol.replace(".NS", ""),
            "signal": sig,
            "price": round(price, 2),
            "entry": entry,
            "stop_loss": sl,
            "target1": t1, "target2": t2, "target3": t3,
            "rr_ratio": rr,
            "profit_pct_t1": profit_pct_t1,
            "profit_pct_t2": profit_pct_t2,
            "qty": qty,
            "profit_t1": profit_t1,
            "profit_t2": profit_t2,
            "risk_per_share": round(risk, 2),
            "buy_score": buy, "sell_score": sell,
            "rsi": round(latest["RSI"], 1),
            "atr": round(atr, 2),
            "vol_ratio": round(latest["Vol_Ratio"], 1),
            "support": round(support, 2), "resistance": round(resistance, 2),
            "reasons": reasons,
            "sector": STOCK_TO_SECTOR.get(self.symbol.replace(".NS", "").upper(), "OTHER"),
        }


# ============================================================================
#  ASYNC QUEUE PIPELINE WORKERS
# ============================================================================

async def news_producer(
    news_queue: asyncio.Queue,
    hours_back: int,
    directly_mentioned_stocks: set,
    affected_sectors: dict,
    shock_articles: list
):
    """Stage 1: Fetch recent news, extract sentiment, and identify shocks/sectors."""
    print("\n[1/4] Scanning live news feeds...")
    articles = await fetch_news_async(hours_back=hours_back)
    print(f"      Found {len(articles)} recent articles from {len(NEWS_FEEDS)} sources")

    if not articles:
        print("\n  [!] No recent news articles found. Falling back to full market scan.")
        articles = [{
            "headline": "Market overview scan",
            "body": "",
            "published_at": datetime.now(),
            "source": "fallback"
        }]

    print("\n[2/4] Analyzing news for shocks and sector impact...")
    for article in articles:
        is_shock, impact, score = detect_shock(article["headline"], article["body"])
        sentiment = analyze_sentiment(article["headline"] + " " + article["body"])
        sectors = detect_sectors(article["headline"], article["body"])
        stocks = detect_specific_stocks(article["headline"], article["body"])

        if is_shock or abs(sentiment) >= 0.3:
            item = {
                **article,
                "impact": impact,
                "shock_score": score,
                "sentiment": sentiment,
                "sectors": sectors,
                "stocks": stocks,
            }
            shock_articles.append(item)
            for s in sectors:
                affected_sectors[s] = affected_sectors.get(s, 0) + score
            for st in stocks:
                directly_mentioned_stocks.add(st)
                
            await news_queue.put(item)
            
    # Sentinel
    await news_queue.put(None)


async def market_data_producer(
    news_queue: asyncio.Queue,
    market_data_queue: asyncio.Queue,
    directly_mentioned_stocks: set,
    affected_sectors: dict
):
    """Stage 2: Process shock news, build stock list, download intraday candles asynchronously."""
    # Retrieve all shock items from news queue
    news_items = []
    while True:
        item = await news_queue.get()
        if item is None:
            news_queue.task_done()
            break
        news_items.append(item)
        news_queue.task_done()

    print("\n[3/4] Building scan list from affected sectors...")
    stocks_to_scan = set()
    stocks_to_scan.update(directly_mentioned_stocks)

    # Sort sectors by total shock score
    sorted_sectors = sorted(affected_sectors.items(), key=lambda x: x[1], reverse=True)
    if sorted_sectors:
        for sector, score in sorted_sectors[:4]:
            sector_stocks = SECTOR_STOCKS.get(sector, [])
            stocks_to_scan.update(sector_stocks)
            print(f"      + {sector} sector ({len(sector_stocks)} stocks, shock score: {score})")
    else:
        print("      No sector-specific shocks. Scanning top stocks from all sectors...")
        for sector, stocks in SECTOR_STOCKS.items():
            stocks_to_scan.update(stocks[:3])

    print(f"\n      Total stocks to analyze: {len(stocks_to_scan)}")
    print("\n[4/4] Ingesting market data and running technical analysis...")

    # Fetch data concurrently using a semaphore to avoid rate limit bans
    sem = asyncio.Semaphore(5)
    
    async def process_single_stock(symbol):
        async with sem:
            await asyncio.sleep(0.5)  # Rate limiting throttle
            analyzer = IntradayAnalyzer(symbol)
            result = await analyzer.fetch_and_analyze()
            if result:
                # Boost if mentioned directly in news
                result["news_boost"] = result["symbol"].upper() in directly_mentioned_stocks
                await market_data_queue.put(result)

    tasks = [asyncio.create_task(process_single_stock(s)) for s in stocks_to_scan]
    if tasks:
        await asyncio.gather(*tasks)

    # Sentinel to signal completion
    await market_data_queue.put(None)


async def analysis_consumer(market_data_queue: asyncio.Queue, results: list):
    """Stage 3: Consumer that collects and tracks signals generated."""
    done_count = 0
    while True:
        result = await market_data_queue.get()
        if result is None:
            market_data_queue.task_done()
            break
        results.append(result)
        done_count += 1
        sys.stdout.write(f"\r      Analyzed {done_count} setups...")
        sys.stdout.flush()
        market_data_queue.task_done()


# ============================================================================
#  MAIN ORCHESTRATOR
# ============================================================================
async def run_scanner():
    print("\n" + "=" * 74)
    print("  NEWS-DRIVEN INTRADAY TRADING SCANNER (ASYNCHRONOUS)")
    print("  Scans live news -> Detects shocks -> Analyzes affected sectors")
    print("  WARNING: For educational purposes only. Trade at your own risk.")
    print("=" * 74)

    # Initialize Queues
    news_queue = asyncio.Queue()
    market_data_queue = asyncio.Queue()

    shock_articles = []
    affected_sectors = {}
    directly_mentioned_stocks = set()
    results = []

    # Run producers and consumer concurrently
    prod_task = asyncio.create_task(news_producer(
        news_queue, 12, directly_mentioned_stocks, affected_sectors, shock_articles
    ))
    
    data_task = asyncio.create_task(market_data_producer(
        news_queue, market_data_queue, directly_mentioned_stocks, affected_sectors
    ))
    
    cons_task = asyncio.create_task(analysis_consumer(
        market_data_queue, results
    ))

    await asyncio.gather(prod_task, data_task, cons_task)
    print(f"\n      Completed: {len(results)} stocks successfully analyzed.")

    # Sort and print shocks
    if shock_articles:
        print(f"\n      Detected {len(shock_articles)} market-moving news items:\n")
        for i, sa in enumerate(shock_articles[:10], 1):
            sent_label = "POSITIVE" if sa["sentiment"] > 0 else "NEGATIVE" if sa["sentiment"] < 0 else "NEUTRAL"
            print(f"      {i}. [{sa['impact']}] [{sent_label}] {sa['headline'][:90]}")
            if sa["sectors"]:
                print(f"         Sectors: {', '.join(sa['sectors'])}")
            if sa["stocks"]:
                print(f"         Stocks:  {', '.join(sa['stocks'])}")
    else:
        print("      No major shocks detected.")

    # ========================================================================
    #  RESULTS — Ranked by PROFIT potential
    # ========================================================================
    actionable = [r for r in results if r["signal"] != "HOLD"]
    high_profit = [r for r in actionable if r["profit_pct_t1"] >= 0.3]
    high_profit.sort(key=lambda x: x["profit_pct_t2"], reverse=True)

    buys = [r for r in high_profit if "BUY" in r["signal"]]
    sells = [r for r in high_profit if "SELL" in r["signal"]]

    def print_trade_card(r, rank):
        news_tag = " << NEWS DRIVEN >>" if r.get("news_boost") else ""
        action = "BUY AT" if "BUY" in r["signal"] else "SELL AT"
        symbol = sanitize_unicode(r['symbol'])
        sector = sanitize_unicode(r['sector'])
        signal = sanitize_unicode(r['signal'])
        
        print(f"\n  {'='*68}")
        print(f"  #{rank}  {symbol} | {sector} | {signal}{news_tag}")
        print(f"  {'='*68}")
        print(f"  |  CMP (Current Price)  :  INR {r['price']}")
        print(f"  |  {action:<22}:  INR {r['entry']}")
        print(f"  |  STOP LOSS            :  INR {r['stop_loss']}  (Risk: INR {r['risk_per_share']}/share)")
        print(f"  |  TARGET 1             :  INR {r['target1']}  (+{r['profit_pct_t1']}%)")
        print(f"  |  TARGET 2             :  INR {r['target2']}  (+{r['profit_pct_t2']}%)")
        print(f"  |  TARGET 3 (Aggressive):  INR {r['target3']}")
        print(f"  |  SUPPORT / RESISTANCE :  INR {r['support']} / INR {r['resistance']}")
        print(f"  |")
        print(f"  |  Risk:Reward          :  1:{r['rr_ratio']}")
        print(f"  |  Suggested Qty (1L)   :  {r['qty']} shares")
        print(f"  |  Profit at T1         :  INR {r['profit_t1']:,.0f}")
        print(f"  |  Profit at T2         :  INR {r['profit_t2']:,.0f}")
        print(f"  |")
        print(f"  |  RSI: {r['rsi']}  |  ATR: {r['atr']}  |  Volume: {r['vol_ratio']}x avg")
        for reason in r["reasons"]:
            print(f"  |    {sanitize_unicode(reason)}")
        print(f"  {'='*68}")

    print("\n\n" + "#" * 74)
    print("  HIGH PROFIT INTRADAY TRADE SETUPS")
    print(f"  Capital Assumed: INR 1,00,000 | Risk per Trade: 2%")
    print("#" * 74)

    if buys:
        print(f"\n  >>> LONG (BUY) SETUPS -- {len(buys)} found <<<")
        for i, r in enumerate(buys[:6], 1):
            print_trade_card(r, i)

    if sells:
        print(f"\n  >>> SHORT (SELL) SETUPS -- {len(sells)} found <<<")
        for i, r in enumerate(sells[:6], 1):
            print_trade_card(r, i)

    if not buys and not sells:
        print("\n  No high-profit setups found. Market may be range-bound.")

    # Quick summary table
    print("\n" + "=" * 74)
    print("  TRADE SUMMARY (sorted by profit potential)")
    print("=" * 74)
    hdr = f"  {'#':<3} {'STOCK':<12} {'SIGNAL':<11} {'ENTRY':>8} {'SL':>8} {'TGT1':>8} {'TGT2':>8} {'PROFIT%':>8}"
    print(hdr)
    print("  " + "-" * 71)
    for i, r in enumerate(high_profit[:20], 1):
        tag = "*" if r.get("news_boost") else " "
        symbol = sanitize_unicode(r['symbol'])
        signal = sanitize_unicode(r['signal'])
        print(f" {tag}{i:<3} {symbol:<12} {signal:<11} {r['entry']:>8} {r['stop_loss']:>8} {r['target1']:>8} {r['target2']:>8} {r['profit_pct_t2']:>7}%")

    if not high_profit:
        print("  No actionable trades.")

    # Watchlist
    watching = [r for r in results if r["signal"] == "HOLD" and r.get("news_boost")]
    if watching:
        print(f"\n  WATCHLIST (news-mentioned, waiting for entry):")
        for r in watching:
            symbol = sanitize_unicode(r['symbol'])
            sector = sanitize_unicode(r['sector'])
            print(f"    {symbol:<12} {sector:<10} CMP: INR {r['price']:>8}  RSI:{r['rsi']}  Support: {r['support']}")

    print("\n" + "=" * 74)
    print(f"  Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 74 + "\n")


# ============================================================================
#  LIVE WEBSOCKET SCANNER (SIMULATED EVENT-DRIVEN FEED)
# ============================================================================

class LiveWebsocketScanner:
    """
    Simulates a live WebSocket feed, aggregates ticks into OHLCV bars,
    and feeds closed bars directly to the IntradayAnalyzer.
    """
    def __init__(self, symbols, interval_mins=1):
        self.symbols = [s if s.endswith(".NS") else f"{s}.NS" for s in symbols]
        self.interval_mins = interval_mins
        self.data_cache = {}  # {symbol: pd.DataFrame}
        self.active_bars = {}  # {symbol: active_bar_dict}
        self.running = False

    async def bootstrap(self):
        """Bootstraps the analyzer with historical candles so technical indicators can be calculated immediately."""
        print(f"\n[Bootstrap] Downloading historical {self.interval_mins}m bars to warm up indicators...")
        sem = asyncio.Semaphore(5)
        
        async def fetch_one(symbol):
            async with sem:
                try:
                    ticker = yf.Ticker(symbol)
                    # Use ThreadPoolExecutor to run the blocking yfinance call
                    df = await asyncio.to_thread(ticker.history, period="5d", interval=f"{self.interval_mins}m")
                    if df is not None and not df.empty and len(df) >= 30:
                        df.index.name = "Datetime"
                        self.data_cache[symbol] = df.tail(60)
                        print(f"      [OK] {symbol} warmed up with {len(self.data_cache[symbol])} bars.")
                    else:
                        print(f"      [FAIL] {symbol} insufficient history ({len(df) if df is not None else 0} bars).")
                except Exception as e:
                    print(f"      [FAIL] {symbol} error during warmup: {e}")
                    
        tasks = [fetch_one(s) for s in self.symbols]
        await asyncio.gather(*tasks)

    async def tick_producer(self, tick_queue: asyncio.Queue):
        """Generates continuous mock ticks simulating a live WebSocket stream."""
        import random
        prices = {}
        for symbol in self.symbols:
            if symbol in self.data_cache and not self.data_cache[symbol].empty:
                prices[symbol] = float(self.data_cache[symbol]['Close'].iloc[-1])
            else:
                prices[symbol] = 1500.0
                
        print("\n[WebSocket] Simulated feed connected. Producing ticks...")
        while self.running:
            # Generate ticks for 1-3 random symbols every 200ms
            active_subset = random.sample(self.symbols, k=random.randint(1, min(3, len(self.symbols))))
            for symbol in active_subset:
                # Random walk with slight drift
                prices[symbol] += random.uniform(-0.4, 0.4)
                tick = {
                    'symbol': symbol,
                    'price': round(prices[symbol], 2),
                    'volume': random.randint(5, 50),
                    'timestamp': datetime.now()
                }
                await tick_queue.put(tick)
            await asyncio.sleep(0.2)

async def run_live_websocket_scanner(universe, interval_mins=1, duration_sec=0):
    """
    Runs the live event-driven scanner using simulated websocket tick queues.
    """
    print("\n" + "=" * 74)
    print(f"  LIVE WEBSOCKET INTRADAY SCANNER (SIMULATED FEED)")
    print(f"  Aggregating ticks into {interval_mins}-minute bars")
    print(f"  Universe: {', '.join(universe)}")
    print("=" * 74)

    scanner = LiveWebsocketScanner(universe, interval_mins=interval_mins)
    await scanner.bootstrap()
    
    tick_queue = asyncio.Queue()
    scanner.running = True
    
    producer_task = asyncio.create_task(scanner.tick_producer(tick_queue))
    start_time = time.time()
    
    print("\n[Live Scanner] Active. Monitoring bars. Press Ctrl+C to stop.")
    
    try:
        while scanner.running:
            if duration_sec > 0 and (time.time() - start_time) > duration_sec:
                print(f"\n[Live Scanner] Target duration of {duration_sec}s reached. Stopping scanner...")
                break
                
            try:
                tick = await asyncio.wait_for(tick_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
                
            symbol = tick['symbol']
            price = tick['price']
            volume = tick['volume']
            ts = tick['timestamp']
            
            # Align tick timestamp to the bar boundary
            bar_ts = ts.replace(second=0, microsecond=0)
            bar_ts = bar_ts - timedelta(minutes=bar_ts.minute % scanner.interval_mins)
            
            if symbol not in scanner.active_bars:
                scanner.active_bars[symbol] = {
                    'timestamp': bar_ts,
                    'Open': price,
                    'High': price,
                    'Low': price,
                    'Close': price,
                    'Volume': volume
                }
                tick_queue.task_done()
                continue
                
            active = scanner.active_bars[symbol]
            
            # If the tick time is greater than the active bar boundary, finalize it!
            if bar_ts > active['timestamp']:
                finalized = active
                ts_str = finalized['timestamp'].strftime('%H:%M:%S')
                clean_symbol = sanitize_unicode(symbol.replace(".NS", ""))
                msg = f"\n[BAR CLOSED] {clean_symbol} at {ts_str} | OHLCV: {finalized['Open']:.2f}/{finalized['High']:.2f}/{finalized['Low']:.2f}/{finalized['Close']:.2f} | Vol: {finalized['Volume']}"
                print(msg)
                
                # Update the history cache
                if symbol in scanner.data_cache:
                    df = scanner.data_cache[symbol]
                    new_row = pd.DataFrame([{
                        'Open': finalized['Open'],
                        'High': finalized['High'],
                        'Low': finalized['Low'],
                        'Close': finalized['Close'],
                        'Volume': finalized['Volume']
                    }], index=[finalized['timestamp']])
                    new_row.index.name = "Datetime"
                    
                    df = pd.concat([df, new_row])
                    if len(df) > 100:
                        df = df.iloc[-100:]
                    scanner.data_cache[symbol] = df
                    
                    # Run analysis on the updated data (with REST call bypassed since analyzer.data is populated)
                    analyzer = IntradayAnalyzer(symbol)
                    analyzer.data = df
                    result = await analyzer.fetch_and_analyze()
                    if result and result['signal'] != "HOLD":
                        reasons_str = " | ".join(result['reasons'][:2])
                        alert_msg = f"  >>> SIGNAL ALERT: {result['signal']} for {result['symbol']} at CMP INR {result['price']}! SL: {result['stop_loss']} | T1: {result['target1']} ({reasons_str})"
                        print(sanitize_unicode(alert_msg))
                
                # Start the next bar
                scanner.active_bars[symbol] = {
                    'timestamp': bar_ts,
                    'Open': price,
                    'High': price,
                    'Low': price,
                    'Close': price,
                    'Volume': volume
                }
            else:
                # Update high/low/close/volume
                active['High'] = max(active['High'], price)
                active['Low'] = min(active['Low'], price)
                active['Close'] = price
                active['Volume'] += volume
                
            tick_queue.task_done()
            
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        print("\n[Live Scanner] Interrupted by user. Exiting...")
    finally:
        scanner.running = False
        producer_task.cancel()
        print("\n[Live Scanner] Shutdown complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="News-Driven Intraday Trading Scanner")
    parser.add_argument("--mode", type=str, default="scan", choices=["scan", "live"],
                        help="Execution mode: 'scan' (default, fetches recent RSS news and prints trade recommendations) or 'live' (runs simulated live websocket scanner)")
    parser.add_argument("--universe", type=str, default=None,
                        help="Comma-separated list of symbols to scan in live mode")
    parser.add_argument("--interval", type=int, default=1,
                        help="Bar aggregation interval in minutes for live mode (default: 1)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Duration to run the live scanner in seconds (0 = run indefinitely)")
    args = parser.parse_args()
    
    if args.mode == "live":
        if args.universe:
            universe = [s.strip() for s in args.universe.split(",")]
        else:
            try:
                from deployment.config import get_config
                cfg = get_config("config.yaml")
                universe = cfg.universe
            except Exception:
                universe = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "TATAMOTORS", "WIPRO"]
        
        asyncio.run(run_live_websocket_scanner(universe, interval_mins=args.interval, duration_sec=args.duration))
    else:
        asyncio.run(run_scanner())