import pandas as pd
import numpy as np
try:
    import redis
    HAS_REDIS = True
except ImportError:
    redis = None
    HAS_REDIS = False
import json
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BarAggregator:
    """
    In-memory tick-to-bar aggregator.
    
    Subscribes to live ticks and publishes finalized OHLCV bars to Redis.
    This ensures Process B (Inference) triggers exactly when a bar closes.
    """
    def __init__(self, redis_url: str, interval_mins: int = 5):
        self.r = redis.from_url(redis_url)
        self.interval_mins = interval_mins
        
        # State: {symbol: {open, high, low, close, volume, vwap_num, vwap_den, last_ts}}
        self.current_bars = {}
        
    def on_tick(self, tick: dict):
        """
        Process a single tick.
        tick format: {symbol, price, volume, timestamp}
        """
        symbol = tick['symbol']
        price = tick['price']
        volume = tick.get('volume', 0)
        ts = tick['timestamp']
        
        if isinstance(ts, str):
            ts = pd.to_datetime(ts)
            
        # Determine bar boundary (e.g. 09:20:00 for 5 min bars starting 09:15)
        bar_ts = ts.replace(second=0, microsecond=0)
        bar_ts = bar_ts - timedelta(minutes=bar_ts.minute % self.interval_mins)
        
        if symbol not in self.current_bars:
            self._start_new_bar(symbol, price, volume, bar_ts)
            return
            
        active_bar = self.current_bars[symbol]
        
        # If tick belongs to a new bar, finalize the old one
        if bar_ts > active_bar['timestamp']:
            self._finalize_bar(symbol)
            self._start_new_bar(symbol, price, volume, bar_ts)
        else:
            # Update active bar
            active_bar['high'] = max(active_bar['high'], price)
            active_bar['low'] = min(active_bar['low'], price)
            active_bar['close'] = price
            active_bar['volume'] += volume
            active_bar['vwap_num'] += (price * volume)
            active_bar['vwap_den'] += volume

    def _start_new_bar(self, symbol: str, price: float, volume: float, ts: datetime):
        self.current_bars[symbol] = {
            'timestamp': ts,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume,
            'vwap_num': price * volume,
            'vwap_den': volume
        }

    def _finalize_bar(self, symbol: str):
        bar = self.current_bars[symbol]
        
        # Calculate VWAP
        if bar['vwap_den'] > 0:
            bar['vwap'] = bar['vwap_num'] / bar['vwap_den']
        else:
            bar['vwap'] = bar['close']
            
        # Prepare payload
        payload = {
            'symbol': symbol,
            'timestamp': bar['timestamp'].isoformat(),
            'open': bar['open'],
            'high': bar['high'],
            'low': bar['low'],
            'close': bar['close'],
            'volume': bar['volume'],
            'vwap': bar['vwap']
        }
        
        # Publish to Redis
        channel = f"bar_ready:{symbol}"
        self.r.publish(channel, json.dumps(payload))
        
        # Also store latest bar in Redis for quick access
        self.r.set(f"latest_bar:{symbol}", json.dumps(payload))
        
        logger.info(f"Finalized {self.interval_mins}m bar for {symbol} at {bar['timestamp']}")

    def force_finalize_all(self):
        """Used at end of day or on shutdown"""
        symbols = list(self.current_bars.keys())
        for sym in symbols:
            self._finalize_bar(sym)
        self.current_bars.clear()
