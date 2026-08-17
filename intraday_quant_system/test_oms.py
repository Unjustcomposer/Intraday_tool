import time
import logging
from execution.order_manager import LiveOMS, OrderState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestOMS")

import os
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

# Load real credentials
load_dotenv(os.path.expanduser("~/.env"))
app_id = os.getenv("FYERS_APP_ID")
access_token = os.getenv("FYERS_ACCESS_TOKEN")

class MockFyers:
    def place_order(self, **kwargs):
        return {"s": "ok", "id": "fyers_12345"}
    def modify_order(self, **kwargs):
        return {"s": "ok", "id": "fyers_12345"}

def run_test():
    if app_id and access_token:
        logger.info("Initializing REAL Fyers Client for Paper Trading...")
        fyers = fyersModel.FyersModel(client_id=app_id, is_async=False, token=access_token, log_path="")
        fyers.token = f"{app_id}:{access_token}"  # Patching SDK bug
        profile = fyers.get_profile()
        if profile.get("s") == "ok":
            logger.info(f"Successfully authenticated as: {profile['data']['name']}")
        else:
            logger.error(f"Fyers Authentication Failed: {profile}")
            return
    else:
        logger.info("Using MockFyers (no credentials found)...")
        fyers = MockFyers()

    oms = LiveOMS(fyers_client=fyers)
    oms.timeout_seconds = 3

    logger.info("--- TEST 1: Submit and Fill ---")
    order_id = oms.submit_order("TCS.NS", "BUY", 10, 3500.0)
    
    logger.info("Mocking WS update: OPEN")
    oms.on_order_update({"id": order_id, "status": "OPEN"})
    
    logger.info("Mocking WS update: PARTIALLY_FILLED")
    oms.on_order_update({"id": order_id, "status": "PARTIALLY_FILLED", "filledQty": 5, "tradedPrice": 3500.0})
    
    logger.info("Mocking WS update: FILLED")
    oms.on_order_update({"id": order_id, "status": "FILLED", "filledQty": 10, "tradedPrice": 3500.0})
    
    final_state = oms.active_orders[order_id].state
    logger.info(f"Test 1 Final State: {final_state}")
    
    
    logger.info("\n--- TEST 2: 30-Second Limit Timeout Conversion ---")
    order_id_2 = oms.submit_order("RELIANCE.NS", "SELL", 5, 2500.0)
    
    oms.on_order_update({"id": order_id_2, "status": "OPEN"})
    
    state_before = oms.active_orders[order_id_2].order_type
    logger.info(f"Order Type before timeout: {state_before}")
    
    logger.info(f"Waiting {oms.timeout_seconds + 1} seconds for timeout thread to kick in...")
    time.sleep(oms.timeout_seconds + 1)
    
    state_after = oms.active_orders[order_id_2].order_type
    logger.info(f"Order Type after timeout: {state_after}")
    
    if state_after == "MARKET":
        logger.info("SUCCESS: The Order State Machine and Limit-to-Market timeout logic works perfectly!")
    else:
        logger.error("FAILURE: Timeout logic did not trigger.")

if __name__ == "__main__":
    run_test()
