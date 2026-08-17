import os
import sys
import time
import logging
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import pandas as pd

from data.fyers_client import FyersBroker
from execution.order_manager import LiveOMS, OrderState
from models.microstructure_alpha import MicrostructureAlphaModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("PaperTraderDaemon")

class PaperFyersClient:
    """
    Mocks Fyers API execution endpoints for Paper Trading.
    Keeps track of fake order IDs and simulates fills using real LTP from WebSocket cache.
    """
    def __init__(self, real_broker: FyersBroker):
        self.broker = real_broker
        self.orders = {}
        self.order_counter = 1000

    def place_order(self, symbol, quantity, side, order_type, price=0.0):
        order_id = f"PAPER_{self.order_counter}"
        self.order_counter += 1
        
        self.orders[order_id] = {
            "symbol": symbol,
            "quantity": quantity,
            "side": side,
            "order_type": order_type,
            "price": price,
            "status": "OPEN",
            "filledQty": 0
        }
        logger.info(f"[PAPER] Placed {side} {quantity} {symbol} @ {price} (Type: {order_type}) -> ID: {order_id}")
        return order_id

    def modify_order(self, order_id, order_type=None, price=None, quantity=None):
        if order_id in self.orders:
            if order_type is not None:
                self.orders[order_id]["order_type"] = "MARKET" if order_type == 2 else "LIMIT"
                logger.info(f"[PAPER] Modified Order {order_id} to {self.orders[order_id]['order_type']}")
            if price is not None:
                self.orders[order_id]["price"] = price
            return {"s": "ok", "id": order_id}
        return {"s": "error", "message": "Order not found"}
        
    def cancel_order(self, order_id):
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELLED"
            logger.info(f"[PAPER] Cancelled Order {order_id}")
            return True
        return False

def get_current_ist():
    # In a real environment, datetime.now() is sufficient if system is IST.
    return datetime.now()

def run():
    load_dotenv(os.path.expanduser("~/.env"))
    
    client_id = os.getenv("FYERS_APP_ID")
    secret_key = os.getenv("FYERS_SECRET_ID")
    access_token = os.getenv("FYERS_ACCESS_TOKEN")
    
    if not client_id or not access_token:
        logger.error("Missing Fyers credentials in ~/.env")
        return

    logger.info("Initializing Fyers Broker...")
    broker = FyersBroker(client_id, secret_key)
    # Patch SDK bug manually just in case
    if broker.fyers:
        broker.fyers.token = access_token
    
    # We don't call broker.authenticate() because we rely on the env token being valid.
    broker.access_token = access_token
    
    paper_client = PaperFyersClient(broker)
    oms = LiveOMS(fyers_client=paper_client)
    
    alpha_model = MicrostructureAlphaModel(vpin_window=50, tsc_window=20)
    
    symbols = ["TCS.NS", "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS"]
    logger.info(f"Connecting WebSocket for {symbols}")
    broker.connect_websocket(symbols)
    
    time.sleep(3) # Wait for WS connection
    
    last_minute = -1
    
    logger.info("🚀 PAPER TRADING DAEMON STARTED")
    
    try:
        while True:
            now = get_current_ist()
            
            # Auto-shutdown at 15:30 IST
            if now.hour == 15 and now.minute >= 30:
                logger.info("Market Closed (15:30). Shutting down...")
                break
                
            # Run Alpha Model once per minute
            if now.minute != last_minute and now.second > 2:
                last_minute = now.minute
                logger.info(f"Fetching 1m bars for {now.strftime('%H:%M')}...")
                
                start_dt = now - timedelta(days=3)
                
                for symbol in symbols:
                    df = broker.get_historical_data(symbol, "1m", start_dt, now)
                    if df is not None and not df.empty:
                        # Feed the latest bar to alpha model
                        latest_bar = df.iloc[-1]
                        
                        # Generate signal
                        # NOTE: alpha_model.on_bar is designed for backtesting (stateful).
                        # We should ideally feed the whole dataframe or reconstruct state.
                        # For now, we will re-init the alpha model or calculate VWAP directly.
                        # To keep it simple, we just run the alpha model on the last 50 bars.
                        
                        # Re-run VWAP manually for the latest signal
                        typical_price = (df['high'] + df['low'] + df['close']) / 3
                        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
                        df['vwap'] = vwap
                        
                        current_px = df['close'].iloc[-1]
                        current_vwap = df['vwap'].iloc[-1]
                        
                        # Fallback: update broker's l2_cache using history in case WS is down
                        std_sym = symbol.replace("NSE:", "").replace("-EQ", "")
                        if std_sym not in broker.l2_cache:
                            broker.l2_cache[std_sym] = {}
                        broker.l2_cache[std_sym]["ltp"] = current_px
                        
                        # Mean Reversion Signal
                        dist = (current_px - current_vwap) / current_vwap
                        threshold = 0.005 # 0.5% deviation
                        
                        signal = 0
                        if dist > threshold:
                            signal = -1 # Sell (Mean Reversion)
                        elif dist < -threshold:
                            signal = 1  # Buy
                            
                        # Execute Order if no active order exists for symbol
                        has_active = any(o.symbol == symbol and o.state in [OrderState.OPEN, OrderState.SUBMITTED, OrderState.PARTIAL] for o in oms.active_orders.values())
                        
                        if signal != 0 and not has_active:
                            side = "BUY" if signal == 1 else "SELL"
                            limit_price = current_px - 2.0 if side == "BUY" else current_px + 2.0
                            logger.info(f"[ALPHA] {symbol} generated {side} signal! Dist to VWAP: {dist*100:.2f}%. Limit: {limit_price:.2f}")
                            oms.submit_order(symbol, side, 10, limit_price)
                            
            # Process Paper Fills from live LTP
            for order_id, order in list(paper_client.orders.items()):
                if order["status"] == "OPEN":
                    sym = order["symbol"]
                    std_sym = sym.replace("NSE:", "").replace("-EQ", "")
                    if ".NS" not in std_sym:
                        std_sym += ".NS"
                        
                    cache = broker.l2_cache.get(std_sym, {})
                    ltp = cache.get("ltp", 0.0)
                    
                    if ltp > 0:
                        # Check if limit price is hit
                        is_filled = False
                        fill_price = ltp
                        
                        if order["order_type"] == "MARKET":
                            # Market orders fill immediately
                            is_filled = True
                        elif order["side"] == "BUY" and ltp <= order["price"]:
                            is_filled = True
                            fill_price = order["price"]
                        elif order["side"] == "SELL" and ltp >= order["price"]:
                            is_filled = True
                            fill_price = order["price"]
                            
                        if is_filled:
                            logger.info(f"[PAPER FILL] Order {order_id} ({sym}) filled at {fill_price} (LTP: {ltp})")
                            paper_client.orders[order_id]["status"] = "FILLED"
                            paper_client.orders[order_id]["filledQty"] = order["quantity"]
                            
                            # Update OMS
                            oms.on_order_update({
                                "id": order_id, 
                                "status": "FILLED", 
                                "filledQty": order["quantity"], 
                                "tradedPrice": fill_price
                            })
                            
            time.sleep(1) # sleep 1 second for event loop
            
    except KeyboardInterrupt:
        logger.info("Daemon interrupted by user.")
    except Exception as e:
        logger.error(f"Daemon crashed: {e}")
        
    logger.info("Paper Trading Daemon shutdown complete.")

if __name__ == "__main__":
    run()
