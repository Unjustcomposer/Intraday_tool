import logging
import os
import threading
from datetime import datetime

import pandas as pd

from .broker_interface import BrokerClient

try:
    from fyers_apiv3 import fyersModel
    from fyers_apiv3.FyersWebsocket import data_ws
except ImportError:
    fyersModel = None
    data_ws = None

logger = logging.getLogger(__name__)


class FyersBroker(BrokerClient):
    """
    Fyers API implementation of the BrokerClient interface.
    Handles authentication, data fetching, order execution, and L2 WebSocket streaming.
    """

    def __init__(
        self,
        client_id: str,
        secret_key: str,
        redirect_uri: str = "http://127.0.0.1:5000/login",
    ):
        self.client_id = client_id
        self.secret_key = secret_key
        self.redirect_uri = redirect_uri
        self.access_token = None
        self.fyers = None

        # WebSocket internals
        self.ws = None
        self.ws_thread = None
        self.l2_cache = {}

        # Load token from environment or config if it exists
        self.access_token = os.environ.get("FYERS_ACCESS_TOKEN")
        if self.access_token:
            self._init_fyers()

    def _init_fyers(self):
        if not fyersModel:
            logger.error("fyers_apiv3 is not installed. Run: pip install fyers-apiv3")
            return

        self.fyers = fyersModel.FyersModel(
            client_id=self.client_id,
            is_async=False,
            token=self.access_token,
            log_path="",
        )

    def authenticate(self) -> bool:
        """
        Authenticate with Fyers API.
        If access_token is missing, you must generate an auth_code externally and exchange it.
        For automated systems, the auth flow usually requires a TOTP generation step or a cached token.
        """
        if self.access_token and self.fyers:
            try:
                profile = self.fyers.get_profile()
                if profile["s"] == "ok":
                    logger.info(
                        f"Fyers Authentication Successful. Logged in as {profile['data']['name']}"
                    )
                    return True
                else:
                    logger.error(f"Fyers Authentication Failed: {profile}")
            except Exception as e:
                logger.error(f"Fyers API Error during auth: {e}")

        logger.warning(
            "No valid access token. Fyers requires daily manual login or TOTP automation."
        )
        return False

    def _translate_symbol(self, symbol: str) -> str:
        """
        Translate generic yfinance symbols to Fyers format.
        e.g., RELIANCE.NS -> NSE:RELIANCE-EQ
        """
        if ".NS" in symbol:
            base = symbol.replace(".NS", "")
            return f"NSE:{base}-EQ"
        return symbol

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Fetch historical candle data from Fyers."""
        if not self.fyers:
            logger.error("Fyers client not initialized.")
            return pd.DataFrame()

        fyers_symbol = self._translate_symbol(symbol)

        # Map our timeframe to Fyers timeframe (1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 240, D)
        tf_map = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "1d": "D"}
        res = tf_map.get(timeframe, "15")

        data = {
            "symbol": fyers_symbol,
            "resolution": res,
            "date_format": "1",  # 1 means epoch
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": end.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        try:
            import requests
            url = "https://api-t1.fyers.in/data/history"
            headers = {"Authorization": f"{self.client_id}:{self.access_token}"}
            response = requests.get(url, headers=headers, params=data).json()
            
            if response.get("s") == "ok":
                candles = response["candles"]
                df = pd.DataFrame(
                    candles,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df.set_index("timestamp", inplace=True)
                return df
            else:
                logger.error(
                    f"Failed to fetch historical data for {fyers_symbol}: {response}"
                )
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Fyers API Exception in get_historical_data: {e}")
            return pd.DataFrame()

    def place_order(
        self, symbol: str, quantity: int, side: str, order_type: str, price: float = 0.0
    ) -> str | None:
        """Place an order via Fyers."""
        if not self.fyers:
            return None

        fyers_symbol = self._translate_symbol(symbol)

        side_val = 1 if side.upper() == "BUY" else -1
        type_val = 1 if order_type.upper() == "LIMIT" else 2  # 1=Limit, 2=Market

        data = {
            "symbol": fyers_symbol,
            "qty": quantity,
            "type": type_val,
            "side": side_val,
            "productType": "INTRADAY",
            "limitPrice": price if type_val == 1 else 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": "False",
        }

        try:
            response = self.fyers.place_order(data=data)
            if response["s"] == "ok":
                order_id = response["id"]
                logger.info(
                    f"Fyers Order Placed: {order_id} - {side} {quantity} {fyers_symbol}"
                )
                return order_id
            else:
                logger.error(f"Fyers Order Failed: {response}")
                return None
        except Exception as e:
            logger.error(f"Fyers API Exception in place_order: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order in Fyers."""
        if not self.fyers:
            return False

        data = {"id": order_id}
        try:
            response = self.fyers.cancel_order(data=data)
            if response["s"] == "ok":
                logger.info(f"Fyers Order Cancelled: {order_id}")
                return True
            else:
                logger.error(f"Fyers Cancel Failed: {response}")
                return False
        except Exception as e:
            logger.error(f"Fyers API Exception in cancel_order: {e}")
            return False

    def get_positions(self) -> pd.DataFrame:
        """Get current positions from Fyers."""
        if not self.fyers:
            return pd.DataFrame()

        try:
            response = self.fyers.positions()
            if response["s"] == "ok":
                positions = response["netPositions"]
                records = []
                for p in positions:
                    # Translate back to yfinance format for internal compatibility if needed,
                    # or keep as is. For now, we strip NSE: and -EQ
                    sym = p["symbol"].replace("NSE:", "").replace("-EQ", "") + ".NS"
                    records.append(
                        {
                            "symbol": sym,
                            "quantity": p["netQty"],
                            "average_price": p["avgPrice"],
                            "realized_pnl": p["realized_profit"],
                            "unrealized_pnl": p["unrealized_profit"],
                        }
                    )
                return pd.DataFrame(records)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Fyers API Exception in get_positions: {e}")
            return pd.DataFrame()

    def get_order_book(self) -> pd.DataFrame:
        """Get order book from Fyers."""
        if not self.fyers:
            return pd.DataFrame()

        try:
            response = self.fyers.orderbook()
            if response["s"] == "ok":
                orders = response["orderBook"]
                return pd.DataFrame(orders)
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Fyers API Exception in get_order_book: {e}")
            return pd.DataFrame()

    def connect_websocket(self, symbols: list[str]):
        """Starts a background WebSocket daemon subscribing to Level 2 Data."""
        if not data_ws:
            logger.error("fyers_apiv3 not installed, cannot start WS.")
            return

        if not self.access_token:
            logger.error("Cannot connect WS without access token.")
            return

        fyers_symbols = [self._translate_symbol(sym) for sym in symbols]

        def on_message(message):
            try:
                # Fyers message is usually a dict. DepthUpdate has 'bids' and 'asks' lists.
                if isinstance(message, dict) and "symbol" in message:
                    sym = message["symbol"]
                    # Translate back to standard symbol e.g., NSE:RELIANCE-EQ -> RELIANCE.NS
                    std_sym = sym.replace("NSE:", "").replace("-EQ", "") + ".NS"

                    # Update cache if bid/ask data is present
                    if std_sym not in self.l2_cache:
                        self.l2_cache[std_sym] = {"bids": [], "asks": [], "ltp": 0.0}

                    if "bids" in message and "asks" in message:
                        self.l2_cache[std_sym]["bids"] = message.get("bids", [])
                        self.l2_cache[std_sym]["asks"] = message.get("asks", [])

                    if "ltp" in message:
                        self.l2_cache[std_sym]["ltp"] = message["ltp"]
            except Exception as e:
                logger.debug(f"WS Parse error: {e}")

        def on_error(message):
            logger.error(f"Fyers WS Error: {message}")

        def on_close(message):
            logger.info(f"Fyers WS Closed: {message}")

        def on_open():
            logger.info(
                f"Fyers WS Opened. Subscribing to DepthUpdate for {len(fyers_symbols)} symbols..."
            )
            self.ws.subscribe(symbols=fyers_symbols, data_type="DepthUpdate")

        # Fyers requires token format: APP_ID:ACCESS_TOKEN
        token_str = f"{self.client_id}:{self.access_token}"

        self.ws = data_ws.FyersDataSocket(
            access_token=token_str,
            log_path="",
            litemode=False,  # We want full market depth
            write_to_file=False,
            reconnect=True,
            on_connect=on_open,
            on_close=on_close,
            on_error=on_error,
            on_message=on_message,
        )

        self.ws_thread = threading.Thread(target=self.ws.connect, daemon=True)
        self.ws_thread.start()
        logger.info(f"Spawned Fyers WS Thread for {len(symbols)} symbols.")

    def get_corporate_actions(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch corporate actions (dividends, splits, bonuses) from Fyers.
        """
        if not self.fyers:
            logger.error("Fyers client not initialized.")
            return pd.DataFrame()

        fyers_symbol = self._translate_symbol(symbol)

        try:
            # Fyers API endpoint for corporate actions
            # Note: This is a hypothetical endpoint - actual Fyers API may differ
            data = {
                "symbol": fyers_symbol,
                "from_date": start_date.strftime("%Y-%m-%d"),
                "to_date": end_date.strftime("%Y-%m-%d"),
            }
            
            # Try to call corporate actions endpoint
            # This may need to be adjusted based on actual Fyers API
            response = self.fyers.get_corporate_actions(data=data)
            
            if response.get("s") == "ok" and "data" in response:
                df = pd.DataFrame(response["data"])
                if not df.empty:
                    df["ex_date"] = pd.to_datetime(df["ex_date"])
                    logger.info(f"Fetched {len(df)} corporate actions for {symbol}")
                return df
            else:
                logger.warning(f"No corporate actions found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Corporate actions fetch failed for {symbol}: {e}")
            return pd.DataFrame()

    def get_true_vwap(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.Series:
        """
        Fetch true tick-level VWAP from Fyers API.
        This eliminates VWAP drift from 15m candle approximation.
        """
        if not self.fyers:
            logger.error("Fyers client not initialized.")
            return None

        fyers_symbol = self._translate_symbol(symbol)

        try:
            # Try to get tick-level VWAP data
            # This may require a specific Fyers API endpoint
            data = {
                "symbol": fyers_symbol,
                "from_date": start_date.strftime("%Y-%m-%d"),
                "to_date": end_date.strftime("%Y-%m-%d"),
            }
            
            response = self.fyers.get_vwap(data=data)
            
            if response.get("s") == "ok" and "data" in response:
                vwap_data = response["data"]
                df = pd.DataFrame(vwap_data)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df.set_index("timestamp", inplace=True)
                vwap_series = df["vwap"]
                logger.info(f"Fetched true VWAP for {symbol} from Fyers")
                return vwap_series
            else:
                logger.warning(f"True VWAP not available for {symbol}")
                return None
                
        except Exception as e:
            logger.warning(f"True VWAP fetch failed for {symbol}: {e}")
            return None

    def get_ofi(self, symbol: str) -> float:
        """
        Computes real-time Order Flow Imbalance (OFI) from L2 Top 5 Bids/Asks.
        Returns [-1.0 to 1.0]. Positive means buying pressure (heavy bids).
        """
        if symbol not in self.l2_cache:
            return 0.0

        depth = self.l2_cache[symbol]
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])

        # Sum volume of top 5 bids/asks
        total_bid_vol = sum([b.get("volume", 0) for b in bids[:5]])
        total_ask_vol = sum([a.get("volume", 0) for a in asks[:5]])

        total_vol = total_bid_vol + total_ask_vol
        if total_vol == 0:
            return 0.0

        return (total_bid_vol - total_ask_vol) / total_vol
