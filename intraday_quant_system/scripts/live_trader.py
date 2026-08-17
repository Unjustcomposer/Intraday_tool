import argparse
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from datetime import time as dt_time

from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fyers_client import FyersBroker
from data.market_data import MarketDataEngine
from data.screener import DynamicScreener
from deployment.config import get_config
from execution.execution_engine import ExecutionEngine
from execution.order_manager import OrderManager
from execution.stop_loss import StopLossEngine
from risk.risk_monitor import RiskMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger("LiveTrader")


class LiveTrader:
    """
    Production Live Trading Daemon.

    Components:
    - FyersBroker: Authentication, order placement, L2 WebSocket
    - MarketDataEngine: Data fetching, feature computation
    - RiskMonitor: Kill switch, daily/weekly limits, exposure control
    - ExecutionEngine: Order execution with circuit breakers
    - OrderManager: Position management, stop-loss, VWAP execution
    - StopLossEngine: Trailing stops, regime-aware stops
    """

    def __init__(self, top_n: int = 10):
        self.top_n = top_n
        self.config = get_config()
        self.running = False

        # Core components (initialized in start())
        self.broker: FyersBroker | None = None
        self.data_engine: MarketDataEngine | None = None
        self.risk_monitor: RiskMonitor | None = None
        self.exec_engine: ExecutionEngine | None = None
        self.order_manager: OrderManager | None = None
        self.stop_engine: StopLossEngine | None = None

        # State
        self.symbols: list[str] = []
        self.ws_thread: threading.Thread | None = None

    def initialize_broker(self) -> bool:
        """Initialize and authenticate Fyers broker."""
        client_id = os.environ.get("FYERS_APP_ID")
        secret_key = os.environ.get("FYERS_SECRET_KEY")

        if not client_id or not secret_key:
            logger.error("Missing FYERS_APP_ID or FYERS_SECRET_KEY in environment.")
            return False

        logger.info("Initializing Fyers Broker Client...")
        self.broker = FyersBroker(client_id=client_id, secret_key=secret_key)

        if not self.broker.authenticate():
            logger.warning(
                "Fyers authentication failed or token missing. Attempting auto-login..."
            )
            auto_login_path = os.path.join(
                os.path.dirname(__file__), "fyers_auto_login.py"
            )
            try:
                result = subprocess.run(
                    [sys.executable, auto_login_path], capture_output=True, text=True
                )
                if "LOGIN SUCCESSFUL" in result.stdout:
                    logger.info(
                        "Auto-login successful! Reloading environment variables..."
                    )
                    load_dotenv(override=True)
                    self.broker = FyersBroker(
                        client_id=client_id, secret_key=secret_key
                    )
                    if not self.broker.authenticate():
                        logger.error(
                            "Still failed to authenticate after successful auto-login script."
                        )
                        return False
                else:
                    logger.error(
                        f"Auto-login failed. Output: {result.stdout}\n{result.stderr}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Exception while running auto-login script: {e}")
                return False

        logger.info("Broker authenticated successfully.")
        return True

    def run_screener(self) -> bool:
        """Run pre-market screener to get active symbols."""
        logger.info("Running Dynamic Pre-Market Screener...")
        screener = DynamicScreener(top_n=self.top_n)
        self.symbols = screener.scan_pre_market()

        if not self.symbols:
            logger.error("Screener returned no symbols. Exiting.")
            return False

        logger.info(f"Today's active watchlist: {self.symbols}")
        return True

    def initialize_engines(self) -> bool:
        """Initialize all trading engines."""
        try:
            # 1. Market Data Engine
            self.data_engine = MarketDataEngine(broker_client=self.broker)

            # 2. Risk Monitor
            self.risk_monitor = RiskMonitor(initial_capital=self.config.max_capital)

            # 3. Stop Loss Engine
            self.stop_engine = StopLossEngine(config=self.config)

            # 4. Execution Engine
            self.exec_engine = ExecutionEngine(
                api_key=self.config.zerodha_api_key,
                api_secret=self.config.zerodha_api_secret,
                paper_trading=getattr(self.config, "paper_trading", True),
            )
            # Override with Fyers broker for actual order placement
            self.exec_engine.fyers_broker = self.broker

            # 5. Order Manager (connects everything)
            self.order_manager = OrderManager(
                execution_engine=self.exec_engine,
                risk_monitor=self.risk_monitor,
                stop_engine=self.stop_engine,
                config=(
                    self.config.model_dump()
                    if hasattr(self.config, "model_dump")
                    else self.config.__dict__
                ),
            )

            logger.info("All engines initialized successfully.")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}")
            return False

    def start_websocket(self):
        """Start Fyers L2 WebSocket in background thread."""
        if not self.broker or not self.symbols:
            return

        def ws_runner():
            try:
                self.broker.connect_websocket(self.symbols)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

        self.ws_thread = threading.Thread(target=ws_runner, daemon=True)
        self.ws_thread.start()
        logger.info(f"Started L2 WebSocket for {len(self.symbols)} symbols")

    def run_tick_loop(self):
        """Main trading loop - processes WebSocket updates and manages positions."""
        logger.info("Entering live tick loop...")
        self.running = True

        # Start WebSocket
        self.start_websocket()

        # Give WebSocket time to connect
        time.sleep(3)

        try:
            while self.running:
                now = datetime.now()
                current_time = now.time()

                # Check market hours
                if current_time < dt_time(9, 15):
                    logger.debug(f"Pre-market: {current_time}, waiting...")
                    time.sleep(10)
                    continue

                if current_time >= dt_time(15, 15):
                    logger.info("Market close. Exiting.")
                    break

                # Hard exit time
                if current_time >= dt_time(15, 10):
                    logger.info("Hard exit time reached. Liquidating all positions.")
                    self.order_manager.liquidate_all(self.get_current_prices())
                    break

                # Check risk monitor
                can_trade, reason = self.risk_monitor.can_trade()
                if not can_trade:
                    logger.warning(f"Risk check failed: {reason}")
                    if self.risk_monitor.trading_halted:
                        logger.critical("Kill switch active. Exiting.")
                        break
                    time.sleep(5)
                    continue

                # Get current prices from L2 cache
                current_prices = self.get_current_prices()
                if not current_prices:
                    time.sleep(1)
                    continue

                # Update risk monitor with current capital (from broker)
                self.update_risk_monitor()

                # Manage open positions (stops, trailing, exits)
                self.order_manager.manage_open_positions(current_prices)

                # Process new signals (if any) - would come from ML pipeline
                # signals = self.generate_signals(current_prices)
                # self.order_manager.process_signals(signals, current_prices, ...)

                # Log status every minute
                if now.second < 5:
                    status = self.risk_monitor.get_status()
                    logger.info(
                        f"Status: Capital={status['current_capital']:,.0f} "
                        f"DD={status['max_drawdown_pct']:.2%} "
                        f"DailyPnL={status['daily_pnl_pct']:.2%} "
                        f"Exposure={status['exposure_multiplier']:.2f} "
                        f"Halted={status['trading_halted']}"
                    )

                time.sleep(0.5)  # 2Hz loop

        except KeyboardInterrupt:
            logger.info("Interrupted by user. Shutting down...")
        finally:
            self.shutdown()

    def get_current_prices(self) -> dict[str, float]:
        """Get current LTP from broker's L2 cache."""
        prices = {}
        if self.broker and hasattr(self.broker, "l2_cache"):
            for symbol, cache in self.broker.l2_cache.items():
                ltp = cache.get("ltp", 0)
                if ltp > 0:
                    prices[symbol] = ltp
        return prices

    def update_risk_monitor(self):
        """Update risk monitor with current capital from broker positions."""
        try:
            positions = self.broker.get_positions()
            if not positions.empty:
                # Calculate current portfolio value
                total_value = self.config.max_capital
                for _, pos in positions.iterrows():
                    qty = pos["quantity"]
                    avg_price = pos["average_price"]
                    if qty != 0:
                        # Add unrealized P&L
                        pass  # Would need current price
                self.risk_monitor.update_capital(total_value)
        except Exception as e:
            logger.debug(f"Risk monitor update failed: {e}")

    def shutdown(self):
        """Graceful shutdown."""
        self.running = False
        logger.info("Shutting down live trader...")
        if self.order_manager:
            self.order_manager.liquidate_all(self.get_current_prices())
        if self.broker:
            self.broker.cancel_all_orders()


def main():
    parser = argparse.ArgumentParser(description="Live Intraday Trading Daemon")
    parser.add_argument(
        "--top-n", type=int, default=10, help="Number of stocks to track"
    )
    parser.add_argument(
        "--paper", action="store_true", help="Run in paper trading mode"
    )
    args = parser.parse_args()

    load_dotenv()

    # Override config for paper trading
    if args.paper:
        os.environ["PAPER_TRADING"] = "true"
        logger.info("PAPER TRADING MODE ENABLED")

    trader = LiveTrader(top_n=args.top_n)

    # Initialize
    if not trader.initialize_broker():
        return

    if not trader.run_screener():
        return

    if not trader.initialize_engines():
        return

    # Run main loop
    trader.run_tick_loop()


if __name__ == "__main__":
    main()
