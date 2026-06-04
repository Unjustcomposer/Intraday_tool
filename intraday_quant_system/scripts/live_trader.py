import sys
import os
import time
import logging
import argparse
import subprocess
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fyers_client import FyersBroker
from data.screener import DynamicScreener
from data.market_data import MarketDataEngine
# from execution.order_manager import OrderManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    force=True
)
logger = logging.getLogger("LiveTrader")

def main():
    parser = argparse.ArgumentParser(description="Live Intraday Trading Daemon")
    parser.add_argument('--top-n', type=int, default=10, help="Number of stocks to track")
    args = parser.parse_args()
    
    load_dotenv()
    
    # 1. Initialize Fyers Broker
    client_id = os.environ.get("FYERS_APP_ID")
    secret_key = os.environ.get("FYERS_SECRET_KEY")
    
    if not client_id or not secret_key:
        logger.error("Missing FYERS_APP_ID or FYERS_SECRET_KEY in environment.")
        return
        
    logger.info("Initializing Fyers Broker Client...")
    broker = FyersBroker(client_id=client_id, secret_key=secret_key)
    
    if not broker.authenticate():
        logger.warning("Fyers authentication failed or token missing. Attempting auto-login...")
        
        # Run auto_login script
        auto_login_path = os.path.join(os.path.dirname(__file__), 'fyers_auto_login.py')
        try:
            result = subprocess.run(
                [sys.executable, auto_login_path], 
                capture_output=True, 
                text=True
            )
            if "LOGIN SUCCESSFUL" in result.stdout:
                logger.info("Auto-login successful! Reloading environment variables...")
                load_dotenv(override=True)
                # Re-init broker with new token
                broker = FyersBroker(client_id=client_id, secret_key=secret_key)
                if not broker.authenticate():
                    logger.error("Still failed to authenticate after successful auto-login script.")
                    return
            else:
                logger.error(f"Auto-login failed. Make sure FYERS_CLIENT_ID, FYERS_TOTP_SECRET, and FYERS_PIN are correct in .env.\nOutput: {result.stdout}\n{result.stderr}")
                return
        except Exception as e:
            logger.error(f"Exception while running auto-login script: {e}")
            return
            
    logger.info("Broker authenticated successfully. Ready for live trading.")
    
    # 2. Run Pre-Market Screener (assuming we start at 09:15)
    logger.info("Running Dynamic Pre-Market Screener to find Stocks in Play...")
    screener = DynamicScreener(top_n=args.top_n)
    symbols = screener.scan_pre_market()
    
    if not symbols:
        logger.error("Screener returned no symbols. Exiting.")
        return
        
    logger.info(f"Today's active watchlist: {symbols}")
    
    # 3. Initialize Engines
    data_engine = MarketDataEngine(broker_client=broker)
    
    # Note: Full execution engine and OMS initialization goes here in production
    
    logger.info("Entering live tick loop...")
    
    try:
        while True:
            now = datetime.now()
            
            # Simple simulation of a tick loop
            # In a real Fyers integration, this would be an async WebSocket callback
            logger.debug(f"Tick loop running at {now.strftime('%H:%M:%S')}")
            
            # If past 15:15, exit loop
            if now.time() >= datetime.strptime("15:15", "%H:%M").time():
                logger.info("Market close approaching. Exiting live trading loop.")
                break
                
            time.sleep(1) # Sleep to prevent CPU spinning in this scaffold
            
    except KeyboardInterrupt:
        logger.info("Live trader interrupted by user. Shutting down gracefully...")
        # broker.cancel_all_orders() -> Handle clean shutdown

if __name__ == "__main__":
    main()
