import logging
import math
import time
import pandas as pd
from typing import Dict, Any, List
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class APICircuitBreaker:
    """
    Lightweight circuit breaker for broker API calls.
    If 3 failures occur within 30 seconds, the circuit opens for 60 seconds.
    """
    def __init__(self, max_failures: int = 3, reset_timeout: float = 60.0, window: float = 30.0):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.window = window
        self.failure_timestamps = []
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_state_change = time.time()

    def check_call(self) -> bool:
        """Returns True if call is allowed, False otherwise"""
        now = time.time()
        if self.state == "OPEN":
            if now - self.last_state_change > self.reset_timeout:
                logger.warning("Circuit breaker transitioning to HALF-OPEN")
                self.state = "HALF-OPEN"
                self.last_state_change = now
                return True
            return False
        return True

    def record_success(self):
        self.failure_timestamps = []
        if self.state != "CLOSED":
            logger.info("Circuit breaker successfully reset to CLOSED")
            self.state = "CLOSED"
            self.last_state_change = time.time()

    def record_failure(self):
        now = time.time()
        self.failure_timestamps.append(now)
        # Filter failures outside the window
        self.failure_timestamps = [t for t in self.failure_timestamps if now - t <= self.window]
        
        if len(self.failure_timestamps) >= self.max_failures:
            if self.state != "OPEN":
                logger.critical(f"APICircuitBreaker: {len(self.failure_timestamps)} failures in {self.window}s. OPENING CIRCUIT!")
                self.state = "OPEN"
                self.last_state_change = now

class ExecutionEngine:
    """
    Execution Engine interacting with the broker API.
    
    Production features:
      - Explicit paper_trading mode
      - Robust slippage model
      - Position state tracking
    """
    def __init__(self, api_key: str = "", api_secret: str = "", paper_trading: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite = None  # In prod: KiteConnect(api_key=self.api_key)
        self.paper_trading = paper_trading
        
        # State tracking for paper trading
        self._mock_positions: Dict[str, Dict] = {}
        self._mock_orders: Dict[str, Dict] = {} # Keyed by order_id
        self._mock_balance = 1000000.0
        
        # Circuit breaker
        self.circuit_breaker = APICircuitBreaker()
        
        if self.paper_trading:
            logger.info("ExecutionEngine initialized in PAPER TRADING mode")
        else:
            logger.warning("ExecutionEngine initialized in LIVE TRADING mode")

    def authenticate(self, request_token: str = ""):
        if self.paper_trading:
            logger.info("Mock authentication successful")
            return
            
        if self.kite:
            try:
                # data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                # self.kite.set_access_token(data["access_token"])
                logger.info("Authenticated with live broker API")
            except Exception as e:
                logger.error(f"Broker authentication failed: {e}")
                raise

    def place_order(self, symbol: str, quantity: int, side: str, order_type: str = "MARKET", price: float = 0.0) -> str:
        """
        Place an order to the broker.
        side: "BUY" or "SELL"
        order_type: "MARKET" or "LIMIT"
        """
        if not self.circuit_breaker.check_call():
            logger.critical("API call blocked by Circuit Breaker (State: OPEN). Switching to monitoring-only.")
            return ""

        if quantity <= 0:
            logger.error(f"Invalid quantity {quantity} for {symbol}")
            return ""
            
        logger.info(f"Placing {order_type} {side} order for {quantity} {symbol} @ {price}")
        
        if self.paper_trading:
            # Paper trading always succeeds
            self.circuit_breaker.record_success()
            return self._place_mock_order(symbol, quantity, side, order_type, price)
            
        # Live trading logic
        try:
            # order_id = self.kite.place_order(
            #     variety=self.kite.VARIETY_REGULAR,
            #     exchange=self.kite.EXCHANGE_NSE,
            #     tradingsymbol=symbol,
            #     transaction_type=self.kite.TRANSACTION_TYPE_BUY if side == "BUY" else self.kite.TRANSACTION_TYPE_SELL,
            #     quantity=quantity,
            #     product=self.kite.PRODUCT_MIS,
            #     order_type=self.kite.ORDER_TYPE_MARKET if order_type == "MARKET" else self.kite.ORDER_TYPE_LIMIT,
            #     price=price if order_type == "LIMIT" else None,
            #     validity=self.kite.VALIDITY_DAY
            # )
            # self.circuit_breaker.record_success()
            # return order_id
            raise NotImplementedError("Live trading not fully implemented in this stub")
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            self.circuit_breaker.record_failure()
            return ""
            
    def _place_mock_order(self, symbol: str, quantity: int, side: str, order_type: str, price: float) -> str:
        order_id = str(uuid.uuid4())
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'side': side,
            'type': order_type,
            'price': price,
            'status': 'OPEN' if order_type == 'LIMIT' else 'COMPLETE',
            'average_price': 0.0,
            'timestamp': datetime.now()
        }
        
        if order_type == "MARKET":
            # Simulate execution price (add slippage)
            exec_price = price
            if price > 0:
                slippage = self.estimate_slippage(quantity * price, 100000000, side=side) # Mock ADV
                exec_price = price + (price * slippage) if side == "BUY" else price - (price * slippage)
            order['average_price'] = exec_price
            self._update_position(symbol, quantity, side, exec_price)
            logger.info(f"Mock MARKET order executed: {side} {quantity} {symbol} @ {exec_price:.2f}")
        else:
            logger.info(f"Mock LIMIT order placed: {side} {quantity} {symbol} @ {price:.2f}")
            
        self._mock_orders[order_id] = order
        return order_id

    def _update_position(self, symbol: str, quantity: int, side: str, exec_price: float):
        current_pos = self._mock_positions.get(symbol, {'quantity': 0, 'average_price': 0.0})
        curr_qty = current_pos['quantity']
        curr_avg = current_pos['average_price']
        
        if side == "BUY":
            new_qty = curr_qty + quantity
        else:
            new_qty = curr_qty - quantity
        
        # Check for flat position FIRST to prevent division by zero
        if new_qty == 0:
            if symbol in self._mock_positions:
                del self._mock_positions[symbol]
            return
        
        # Compute new average price
        if side == "BUY":
            if curr_qty >= 0:
                new_avg = ((curr_qty * curr_avg) + (quantity * exec_price)) / new_qty
            else:
                new_avg = curr_avg if new_qty < 0 else exec_price
        else:
            if curr_qty <= 0:
                new_avg = ((abs(curr_qty) * curr_avg) + (quantity * exec_price)) / abs(new_qty)
            else:
                new_avg = curr_avg if new_qty > 0 else exec_price
        
        self._mock_positions[symbol] = {'quantity': new_qty, 'average_price': new_avg}

    def convert_to_market(self, order_id: str, current_price: float):
        """Force execution of an open LIMIT order by converting to MARKET"""
        if order_id not in self._mock_orders:
            return
            
        order = self._mock_orders[order_id]
        if order['status'] != 'OPEN':
            return
            
        logger.info(f"Converting LIMIT order {order_id} to MARKET at {current_price}")
        order['type'] = 'MARKET'
        order['status'] = 'COMPLETE'
        
        slippage = self.estimate_slippage(order['quantity'] * current_price, 100000000, side=order['side'])
        exec_price = current_price + (current_price * slippage) if order['side'] == "BUY" else current_price - (current_price * slippage)
        order['average_price'] = exec_price
        
        self._update_position(order['symbol'], order['quantity'], order['side'], exec_price)


    def get_positions(self) -> pd.DataFrame:
        """Get current open positions"""
        if self.paper_trading:
            if not self._mock_positions:
                return pd.DataFrame()
            
            data = []
            for sym, pos in self._mock_positions.items():
                data.append({
                    'symbol': sym,
                    'quantity': pos['quantity'],
                    'average_price': pos['average_price']
                })
            return pd.DataFrame(data)
            
        try:
            # positions = self.kite.positions()
            # net_positions = positions.get("net", [])
            # return pd.DataFrame(net_positions)
            logger.warning("Live positions not implemented")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return pd.DataFrame()

    def estimate_slippage(self, order_size: float, adv: float, side: str = "BUY") -> float:
        """
        Estimate slippage using an asymmetric Almgren-Chriss style square-root impact model.
        Buying (demanding liquidity in thin books) is modeled as more expensive than selling.
        """
        if adv <= 0:
            return 0.005 # Default 50bps for unknown volume
            
        participation_rate = order_size / adv
        # k is an empirical constant for Indian equities
        k = 0.1 
        
        # Asymmetric impact: buying is 1.5x more expensive than selling in typical markets
        asymmetry_multiplier = 1.5 if side.upper() == "BUY" else 0.8
        
        impact = k * math.sqrt(participation_rate) * asymmetry_multiplier
        # Cap at 1% to prevent absurd estimates
        return min(impact, 0.01)

    def cancel_all_orders(self, symbol: str = None):
        """Cancel all open orders (e.g. during a kill switch event)"""
        logger.info(f"Canceling all open orders {f'for {symbol}' if symbol else ''}")
        if self.paper_trading:
            # Paper mode: mark all matching open orders as CANCELLED
            cancelled_count = 0
            for order_id, order in list(self._mock_orders.items()):
                if order['status'] == 'OPEN':
                    if symbol is None or order['symbol'] == symbol:
                        order['status'] = 'CANCELLED'
                        cancelled_count += 1
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} paper orders")
            return
            
        # Live implementation
        # try:
        #     orders = self.kite.orders()
        #     for order in orders:
        #         if order['status'] == 'OPEN':
        #             if symbol is None or order['tradingsymbol'] == symbol:
        #                 self.kite.cancel_order(self.kite.VARIETY_REGULAR, order['order_id'])
        # except Exception as e:
        #     logger.error(f"Failed to cancel orders: {e}")
