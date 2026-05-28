import logging
import os
import signal
import sys
import json
import time
import threading
try:
    from kiteconnect import KiteTicker
    HAS_KITE = True
except ImportError:
    KiteTicker = None
    HAS_KITE = False
from data.bar_aggregator import BarAggregator
from data.market_data import MarketDataEngine
from deployment.config import get_config
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TickerProcess")

class TickerProcess:
    """
    Process A: WebSocket Ingestion.
    Responsible for maintaining WebSocket connection and aggregating ticks.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = get_config(config_path)
        self.aggregator = BarAggregator(redis_url=self.config.redis_url)
        self.market_engine = MarketDataEngine(api_key=self.config.zerodha_api_key)
        
        # Resolve tokens dynamically
        self.symbol_map = self.market_engine.get_instrument_tokens(self.config.universe)
        self.token_map = {v: k for k, v in self.symbol_map.items()}
        
        self.kws = KiteTicker(
            api_key=self.config.zerodha_api_key,
            access_token=os.getenv("ZERODHA_ACCESS_TOKEN", "mock_token")
        )
        
    def on_ticks(self, ws, ticks):
        for tick in ticks:
            token = tick.get('instrument_token')
            symbol = self.token_map.get(token)
            if not symbol:
                continue
                
            self.aggregator.on_tick({
                'symbol': symbol,
                'price': tick['last_price'],
                'volume': tick.get('last_traded_quantity', 0),
                'timestamp': tick.get('timestamp') or datetime.now()
            })

    def on_connect(self, ws, response):
        # Subscribe to tokens from universe
        tokens = list(self.token_map.keys())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        logger.info(f"Connected and subscribed to {len(tokens)} tokens")

    def on_close(self, ws, code, reason):
        logger.warning(f"Connection closed: {code} - {reason}")
        if code == 1006 or "403" in str(reason):
            logger.info("Switching to SIMULATION MODE due to connection failure.")
            self._run_simulation()

    def run(self):
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
        
        logger.info("Starting Process A: WebSocket Ingestion")
        try:
            self.kws.connect()
        except Exception as e:
            logger.warning(f"Real WebSocket connection failed: {e}. Switching to SIMULATION MODE.")
            self._run_simulation()

    def _run_simulation(self):
        """
        Generates mock ticks for the universe when real connection is unavailable.
        """
        import time
        import random
        
        logger.info("Simulation Mode active: Generating mock ticks for universe...")
        prices = {sym: 2000.0 for sym in self.config.universe}
        
        while True:
            for symbol in self.config.universe:
                # Random walk simulation
                prices[symbol] += random.uniform(-0.5, 0.5)
                
                tick = {
                    'symbol': symbol,
                    'price': prices[symbol],
                    'volume': random.randint(1, 50),
                    'timestamp': datetime.now()
                }
                self.aggregator.on_tick(tick)
            
            time.sleep(1) # 1 second tick frequency in simulation

if __name__ == "__main__":
    ticker = TickerProcess()
    
    # Handle graceful shutdown
    def handle_exit(sig, frame):
        logger.info("Shutting down ticker process...")
        ticker.aggregator.force_finalize_all()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_exit)
    ticker.run()
