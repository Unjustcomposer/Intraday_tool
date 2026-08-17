"""
Paper Broker Integration
========================
Provides a PaperBroker class that acts as a drop-in replacement for FyersBroker.
It intercepts place_order calls and writes them to a local CSV log, simulating execution
while using live Fyers WebSocket data for pricing and OFI computation.
"""

import os
import csv
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger("PaperBroker")

class PaperBroker:
    """
    Mock broker that implements the exact interface as FyersBroker but
    simulates executions locally in a CSV file.
    """
    def __init__(self, live_broker, paper_trades_csv="data/paper_trades_log.csv"):
        """
        Initializes the PaperBroker.
        Args:
            live_broker: An instantiated FyersBroker object used to proxy WS connections.
            paper_trades_csv: Path to save the executed paper trades.
        """
        self.live_broker = live_broker
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.paper_trades_csv = os.path.join(base_dir, paper_trades_csv)
        
        # Initialize the CSV if it doesn't exist
        os.makedirs(os.path.dirname(self.paper_trades_csv), exist_ok=True)
        if not os.path.exists(self.paper_trades_csv):
            with open(self.paper_trades_csv, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'direction', 'quantity', 
                    'fill_price', 'simulated_slippage', 'ofi_at_execution', 'status'
                ])
                
        # Proxy attributes so execution_engine can access l2_cache seamlessly
        self.l2_cache = self.live_broker.l2_cache

    def connect_websocket(self, symbols):
        """Proxy to the live broker's websocket."""
        logger.info("[PaperBroker] Proxying WebSocket connection to live broker...")
        self.live_broker.connect_websocket(symbols)

    def get_ofi(self, symbol):
        """Proxy to the live broker's OFI computation."""
        return self.live_broker.get_ofi(symbol)

    def place_order(self, symbol: str, quantity: int, side: str, order_type: str, price: float = 0.0) -> str:
        """
        Simulates placing an order. Instead of hitting the API, it calculates a 
        simulated fill price based on the live LTP and logs it.
        """
        cache = self.l2_cache.get(symbol, {})
        live_ltp = cache.get("ltp", 0.0)
        
        if live_ltp == 0.0:
            logger.error(f"[PaperBroker] Cannot place order for {symbol}, live LTP is 0.0")
            return None
            
        # Simulate slippage (e.g., 0.02% adverse movement)
        slippage_pct = 0.0002
        simulated_slippage = live_ltp * slippage_pct
        
        if side == "BUY":
            fill_price = live_ltp + simulated_slippage
        else:
            fill_price = live_ltp - simulated_slippage
            
        current_ofi = self.get_ofi(symbol)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log trade to CSV
        with open(self.paper_trades_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, symbol, side, quantity, 
                round(fill_price, 2), round(simulated_slippage, 2), round(current_ofi, 2), 'FILLED'
            ])
            
        logger.info(f"[PaperBroker] 📝 PAPER TRADE LOGGED: {side} {quantity} {symbol} @ ₹{fill_price:.2f}")
        
        # Return a fake order ID
        return f"PAPER_ORD_{int(datetime.now().timestamp())}"
        
    def cancel_order(self, order_id: str) -> bool:
        logger.info(f"[PaperBroker] Simulated cancellation of {order_id}")
        return True
