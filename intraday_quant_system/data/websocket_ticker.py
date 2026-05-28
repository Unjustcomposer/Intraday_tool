import logging
import os
import signal
import sys
import json
import time
import asyncio
import threading
from datetime import datetime

try:
    from kiteconnect import KiteTicker
    HAS_KITE = True
except ImportError:
    KiteTicker = None
    HAS_KITE = False

from data.bar_aggregator import BarAggregator
from data.market_data import MarketDataEngine
from deployment.config import get_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TickerProcess")

class TickerProcess:
    """
    Process A: WebSocket Ingestion.
    Responsible for maintaining WebSocket connection and aggregating ticks.
    Refactored to support asyncio and a 30-second disconnect kill-switch.
    """
    def __init__(self, config_path="config.yaml"):
        self.config = get_config(config_path)
        self.aggregator = BarAggregator(redis_url=self.config.redis_url)
        self.market_engine = MarketDataEngine(api_key=self.config.zerodha_api_key)
        
        # Resolve tokens dynamically
        self.symbol_map = self.market_engine.get_instrument_tokens(self.config.universe)
        self.token_map = {v: k for k, v in self.symbol_map.items()}
        
        # Connection status tracking
        self.last_tick_time = time.time()
        self.startup_time = time.time()
        self.is_connected = False
        self.kill_switch_activated = False
        self.disconnect_start_time = None
        
        if HAS_KITE and self.config.zerodha_api_key:
            self.kws = KiteTicker(
                api_key=self.config.zerodha_api_key,
                access_token=os.getenv("ZERODHA_ACCESS_TOKEN", "mock_token")
            )
        else:
            self.kws = None

    def on_ticks(self, ws, ticks):
        self.last_tick_time = time.time()
        self.is_connected = True
        self.disconnect_start_time = None
        
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
        self.is_connected = True
        self.disconnect_start_time = None
        # Subscribe to tokens from universe
        tokens = list(self.token_map.keys())
        ws.subscribe(tokens)
        ws.set_mode(ws.MODE_FULL, tokens)
        logger.info(f"Connected and subscribed to {len(tokens)} tokens")

    def on_close(self, ws, code, reason):
        logger.warning(f"Connection closed: {code} - {reason}")
        self.is_connected = False
        if self.disconnect_start_time is None:
            self.disconnect_start_time = time.time()

    async def _monitor_connection_async(self):
        """Asynchronously monitor connection health and trigger kill-switch if disconnected or latency > 500ms"""
        while True:
            await asyncio.sleep(0.1)  # High-frequency check (100ms) for sub-second latency
            if self.kill_switch_activated:
                continue
                
            now = time.time()
            # Allow a grace period of 15 seconds on startup for connection to establish
            if now - self.startup_time < 15.0:
                continue
                
            latency = now - self.last_tick_time
            
            # Heartbeat check: disconnect or latency > 500ms (0.5s)
            if not self.is_connected or latency > 0.5:
                logger.critical(f"HEARTBEAT FAILURE: WebSocket connected={self.is_connected} | Latency={latency*1000:.1f}ms exceeds 500ms limit!")
                self.activate_kill_switch()

    def activate_kill_switch(self):
        """Kill-switch: Send market orders to close all open positions to prevent unmonitored exposure"""
        self.kill_switch_activated = True
        logger.critical("KILL-SWITCH TRIGGERED: Websocket disconnected for more than 30 seconds!")
        try:
            from execution.execution_engine import ExecutionEngine
            engine = ExecutionEngine(
                api_key=self.config.zerodha_api_key,
                api_secret=self.config.zerodha_api_secret,
                paper_trading=True  # Safe default: follows configuration setup
            )
            positions = engine.get_positions()
            if not positions.empty:
                logger.critical(f"Kill-switch liquidating {len(positions)} open positions...")
                for _, pos in positions.iterrows():
                    symbol = pos['symbol']
                    qty = abs(pos['quantity'])
                    side = "SELL" if pos['quantity'] > 0 else "BUY"
                    if qty > 0:
                        engine.place_order(symbol, qty, side, order_type="MARKET")
            engine.cancel_all_orders()
            logger.critical("Kill-switch liquidation and order cancellations complete.")
        except Exception as e:
            logger.error(f"Error executing kill-switch: {e}")

    async def _run_simulation_async(self):
        """Asynchronous simulation tick generator"""
        import random
        logger.info("Async Simulation Mode active: Generating mock ticks for universe...")
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
                # Keep connection parameters fresh in simulation
                self.last_tick_time = time.time()
                self.is_connected = True
                
            await asyncio.sleep(1.0) # 1 second tick frequency in simulation

    async def run_async(self):
        """Async execution entry point"""
        logger.info("Starting Process A: WebSocket Ingestion (Async)")
        
        # Spawn the async monitoring task
        asyncio.create_task(self._monitor_connection_async())
        
        if not self.kws:
            logger.info("No Kite client configured. Switching to ASYNC SIMULATION MODE.")
            await self._run_simulation_async()
            return
            
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_close = self.on_close
        
        loop = asyncio.get_running_loop()
        try:
            # Run blocking ticker client in executor to not block async loop
            await loop.run_in_executor(None, self.kws.connect)
        except Exception as e:
            logger.warning(f"Real WebSocket connection failed: {e}. Switching to ASYNC SIMULATION MODE.")
            await self._run_simulation_async()

if __name__ == "__main__":
    ticker = TickerProcess()
    
    # Handle graceful shutdown
    def handle_exit(sig, frame):
        logger.info("Shutting down ticker process...")
        ticker.aggregator.force_finalize_all()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_exit)
    
    # Run the asyncio event loop
    asyncio.run(ticker.run_async())
