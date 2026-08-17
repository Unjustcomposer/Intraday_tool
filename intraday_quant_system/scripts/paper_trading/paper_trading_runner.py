"""
Paper Trading Runner
====================

Runs the full intraday quant system in paper trading mode.
Integrates all components: data, features, models, signals, execution, risk.
"""

import sys
import os
import argparse
import logging
import time
import json
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intraday_quant_system.data.fyers_client import FyersBroker
from intraday_quant_system.data.market_data import MarketDataEngine
from intraday_quant_system.data.screener import DynamicScreener
from intraday_quant_system.features.feature_store import FeatureStore
from intraday_quant_system.models.lgbm_model import LGBMAlphaModel
from intraday_quant_system.models.catboost_meta_labeler import MetaLabeler
from intraday_quant_system.signals.ensemble import EnsembleScorer
from intraday_quant_system.signals.call_generator import CallGenerator
from intraday_quant_system.execution.order_manager import OrderManager
from intraday_quant_system.execution.execution_engine import ExecutionEngine
from intraday_quant_system.execution.stop_loss import StopLossEngine
from intraday_quant_system.risk.risk_monitor import RiskMonitor
from intraday_quant_system.regime.hmm_regime import RegimeDetector
from intraday_quant_system.monitoring import (
    get_logger, set_correlation_id, get_metrics_exporter,
    MonteCarloStressTester, calculate_dsr,
    PnLAttribution, AttributionResult
)
from intraday_quant_system.deployment.config import get_config

logger = logging.getLogger(__name__)
class PaperTrade:
    """Record of a paper trade"""
    """Record of a paper trade"""
    timestamp: str
    symbol: str
    side: str  # BUY/SELL
    quantity: int
    price: float
    order_type: str
    order_id: str
    parent_order_id: str = ""
    status: str = "PENDING"
    filled_qty: int = 0
    avg_fill_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    regime: str = "unknown"
    confidence: float = 0.0
    pnl: float = 0.0
    regime_at_entry: str = ""
    signal_confidence: float = 0.0

@dataclass
@dataclass
class PaperTradingState:
    """Complete state of paper trading session"""
    start_date: str
    end_date: str
    symbols: List[str]
    initial_capital: float
    current_capital: float
    trades: List[Any] = field(default_factory=list)
    open_positions: Dict[str, Dict] = field(default_factory=dict)
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    max_drawdown: float = 0.0
    peak_capital: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
class PaperTradingEngine:
    """
    Paper trading engine that runs the full system in simulation mode.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 1_000_000.0,
        config_path: str = "config.yaml",
        paper_trading: bool = True,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.config = get_config(config_path)
        self.paper_trading = paper_trading

        # State
        self.state = PaperTradingState(
            start_date=datetime.now().strftime("%Y-%m-%d"),
            end_date="",
            symbols=symbols,
            initial_capital=initial_capital,
            current_capital=initial_capital,
            peak_capital=initial_capital,
        )

        # Components (initialized in initialize())
        self.broker = None
        self.market_data = None
        self.screener = None
        self.feature_store = None
        self.lgbm_model = None
        self.meta_labeler = None
        self.ensemble = None
        self.call_generator = None
        self.exec_engine = None
        self.order_manager = None
        self.stop_engine = None
        self.risk_monitor = None
        self.regime_detector = None
        self.metrics = None
        self.attributor = None

        # Models loaded flag
        self.models_loaded = False

        logger.info(f"PaperTradingEngine initialized: {len(symbols)} symbols, "
                    f"capital={initial_capital:,.0f}, paper_trading={paper_trading}")

    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("Initializing paper trading components...")

            # 1. Broker
            self.broker = FyersBroker(
                client_id=os.environ.get("FYERS_APP_ID", ""),
                secret_key=os.environ.get("FYERS_SECRET_KEY", "")
            )

            # For paper trading, we don't need real auth
            if self.paper_trading:
                logger.info("Paper trading mode: skipping broker authentication")
                from unittest.mock import MagicMock
                self.broker = MagicMock()
                self.broker.authenticate.return_value = True
                self.broker.get_historical_data = self._mock_historical_data
                self.broker.place_order = self._mock_place_order
                self.broker.cancel_order = self._mock_cancel_order
                self.broker.cancel_all_orders = self._mock_cancel_all_orders
                self.broker.get_positions = self._mock_get_positions
                self.broker.connect_websocket = self._mock_connect_websocket
                self.broker.l2_cache = {}
            else:
        else:
                if not self.broker.authenticate():
                    logger.error("Failed to authenticate with Fyers")
                    return False

            # 2. Market Data Engine
            self.market_data = MarketDataEngine(broker_client=self.broker)

            # 3. Screener
            self.screener = DynamicScreener(top_n=len(self.symbols))

            # 3. Feature Store
            self.feature_store = FeatureStore(bars_per_day=self.config.intraday.bars_per_day)

            # 4. Risk Monitor
            self.risk_monitor = RiskMonitor(initial_capital=self.initial_capital)

            # 5. Stop Loss Engine
            self.stop_engine = StopLossEngine(config=self.config.model_dump() if hasattr(self.config, 'model_dump') else dict(self.config))

            # 6. Execution Engine
            self.exec_engine = ExecutionEngine(
                api_key="",
                api_secret="",
                paper_trading=True
            )

            # 7. Order Manager
            self.order_manager = OrderManager(
                execution_engine=self.exec_engine,
                risk_monitor=self.risk_monitor,
                stop_engine=self.stop_engine,
                config=self.config.model_dump() if hasattr(self.config, 'model_dump') else dict(self.config)
            )

            # 8. Models
            self.lgbm_model = LGBMAlphaModel(config={})
            self.meta_labeler = MetaLabeler(config={})
            self.ensemble = EnsembleScorer()

            # 10. Call Generator
            self.call_generator = CallGenerator(
                min_risk_reward=2.0,
                min_confidence=0.60,
                max_net_exposure=3,
                max_total_calls=5
            )

            # 11. Regime Detector
            self.regime_detector = RegimeDetector(n_regimes=3)

            # 12. Monitoring
            self.metrics = get_metrics_exporter()

            logger.info("Paper trading components initialized successfully")
            return True

    except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False

    def _mock_historical_data(self, symbol, timeframe, start, end):
        """Mock historical data for paper trading"""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start, end, freq='15min')
        n = len(dates)
        # Generate realistic OHLCV data
        base_price = 2500.0
        returns = np.random.normal(0, 0.01, n)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n),
        })
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        return df

    def _mock_place_order(self, symbol, quantity, side, order_type, price):
        order_id = f"PAPER_{int(time.time() * 1000)}"
        logger.info(f"Paper order placed: {side} {quantity} {symbol} @ {price} ({order_id})")
        return order_id

    def _mock_cancel_order(self, order_id):
        logger.info(f"Paper order cancelled: {order_id}")
        return True

    def _mock_cancel_all_orders(self, symbol=None):
        logger.info(f"All paper orders cancelled for {symbol or 'all'}")
        return True

    def _mock_get_positions(self):
        import pandas as pd
        return pd.DataFrame()


    def load_models(self) -> bool:
        """Load pre-trained models from disk."""
        try:
            model_dir = "./data/models"
            if not os.path.exists(model_dir):
                logger.warning(f"Model directory {model_dir} not found, will train new models")
                return self.train_models()
                        # Load LGBM
            lgbm_path = os.path.join(model_dir, "lgbm_latest.txt")
            if os.path.exists(lgbm_path):
                self.lgbm_model.load(lgbm_path)
                logger.info(f"Loaded LGBM from {lgbm_path}")
                        # Load Meta-Labeler
            meta_path = os.path.join(model_dir, "meta_latest.cbm")
            if os.path.exists(meta_path):
                self.meta_labeler.load(meta_path)
                logger.info(f"Loaded Meta-Labeler from {meta_path}")
                        # Load Regime Detector
            regime_path = os.path.join(model_dir, "regime_latest.pkl")
            if os.path.exists(regime_path):
                self.regime_detector.load(regime_path)
                logger.info(f"Loaded Regime Detector from {regime_path}")
                        self.models_loaded = True
            logger.info("Models loaded successfully")
            return True
                except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    def train_models(self) -> bool:
        """Train all models on historical data."""
        try:
            logger.info("Training models on historical data...")
                        # Fetch historical data for all symbols
            end_date = datetime.now()
            start_date = end_date - timedelta(days=180)
                        all_data = {}
            for symbol in self.symbols:
                logger.info(f"Fetching data for {symbol}...")
                df = self.market_data.fetch_fyers_historical_data(symbol, start_date, end_date)
                if not df.empty:
                    all_data[symbol] = df
                        if not all_data:
                logger.error("No historical data fetched")
                return False
                        # Compute features for all symbols
            all_features = {}
            for symbol, df in all_data.items():
                logger.info(f"Computing features for {symbol}...")
                features_df = self.feature_store.compute_all(symbol, df)
                all_features[symbol] = features_df
                        # Train models for each symbol
            for symbol, features_df in all_features.items():
                logger.info(f"Training models for {symbol}...")

                # Generate labels
                labels = LGBMAlphaModel.make_labels(features_df)
                features_df['label'] = labels
                features_df = features_df.dropna()

                if len(features_df) < 200:
                    logger.warning(f"Insufficient data for {symbol}: {len(features_df)} samples")
                    continue

                feature_cols = self.feature_store.get_feature_columns()
                feature_cols = [c for c in feature_cols if c in features_df.columns]

                X = features_df[feature_cols]
                y = features_df['label']

                # Train LGBM
                self.lgbm_model.train(X, y)

                # Train Meta-Labeler
                # Split: 70% primary, 30% meta
                split_idx = int(len(X) * 0.7)
                X_primary = X.iloc[:split_idx]
                y_primary = y.iloc[:split_idx]
                X_meta = X.iloc[split_idx:]
                y_meta = y.iloc[split_idx:]

                primary_preds = self.lgbm_model.predict(X_meta)
                y_meta_outcome = (primary_preds == y_meta).astype(int)

                self.meta_labeler.train(primary_preds, X_meta, y_meta_outcome)

                # Fit regime detector
                self.regime_detector.fit(features_df)
                        # Save models
            os.makedirs("./data/models", exist_ok=True)
            self.lgbm_model.save("./data/models/lgbm_latest.txt")
            self.meta_labeler.save("./data/models/meta_latest.cbm")
            self.regime_detector.save("./data/models/regime_latest.pkl")
                        self.models_loaded = True
            logger.info("Models trained and saved successfully")
            return True
                except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return False

    def run_paper_trading(self, days: int = 10) -> bool:
        """
        Run paper trading for specified number of days.
        """
        logger.info(f"Starting paper trading for {days} days...")

        if not self.models_loaded:
            logger.error("Models not loaded. Call load_models() or train_models() first.")
            return False

        # Fetch initial data for feature computation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        all_data = {}
        for symbol in self.symbols:
            logger.info(f"Fetching initial data for {symbol}...")
            df = self.market_data.fetch_fyers_historical_data(symbol, start_date, end_date)
            if not df.empty:
                all_data[symbol] = df

        if not all_data:
            logger.error("No initial data fetched")
            return False

        # Compute initial features
        all_features = {}
        for symbol, df in all_data.items():
            features_df = self.feature_store.compute_all(symbol, df)
            all_features[symbol] = features_df

        # Fit regime detector on combined data
        combined_features = pd.concat(all_features.values())
        self.regime_detector.fit(combined_features)

        # Main trading loop
        logger.info("Starting paper trading loop...")
        self.state.start_date = datetime.now().strftime("%Y-%m-%d")
        self.state.end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            for day in range(days):
                current_date = datetime.now() + timedelta(days=day)

                # Skip weekends
                if current_date.weekday() >= 5:
                    logger.info(f"Skipping weekend: {current_date.strftime('%Y-%m-%d')}")
                    continue

                logger.info(f"Trading day {day+1}/{days}: {current_date.strftime('%Y-%m-%d')}")

                # Run single trading day
                success = self._run_trading_day(current_date)

                if not success:
                    logger.error(f"Trading day failed for {current_date}")
                    break

                # Save daily state
                self._save_daily_state(current_date)

                # Small delay between days
                if day < days - 1:
                    time.sleep(1)
                        self.state.end_date = datetime.now().strftime("%Y-%m-%d")
            logger.info("Paper trading completed successfully")
            return True
                except Exception as e:
            logger.error(f"Paper trading failed: {e}")
            return False
    def _run_trading_day(self, current_date: datetime) -> bool:
        """Run a single trading day."""
        try:
            # Market hours: 9:15 - 15:15
            market_open = dt_time(9, 15)
            market_close = dt_time(15, 15)
                        logger.info(f"Market open at {market_open}")
                        # Pre-market screening (9:00 - 9:15)
            logger.info("Running pre-market screener...")
            if not self._run_pre_market_screening():
                logger.warning("Pre-market screening failed")
                        # Main trading loop (9:15 - 15:15)
            # In paper trading, we simulate the day using historical data
            return self._simulate_trading_day()
                except Exception as e:
            logger.error(f"Trading day failed: {e}")
            return False
    def _run_pre_market_screening(self) -> bool:
        """Run pre-market screener to find stocks in play."""
        try:
            # Use screener to find active symbols
            self.symbols = self.screener.scan_pre_market()
            logger.info(f"Pre-market screener found {len(self.symbols)} symbols: {self.symbols}")
            return True
    except Exception as e:
            logger.error(f"Pre-market screening failed: {e}")
            return False
    def _simulate_trading_day(self) -> bool:
        """
        Simulate a full trading day using historical data.
        In paper trading, we use the day's historical data to simulate trades.
        """
        try:
            # For each symbol, simulate the day's trading
            for symbol in self.symbols:
                # Get day's data (in real trading, this would be real-time)
                # For paper trading, we use the day's historical data
                pass
                        # Process signals through order manager
            # This is where the actual trading logic runs
            self._process_trading_signals()
                        return True
    except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    def _run_pre_market_screening(self) -> bool:
        """Run pre-market screener to find stocks in play."""
        try:
            # Use screener to find active symbols
            self.symbols = self.screener.scan_pre_market()
            logger.info(f"Pre-market screener found {len(self.symbols)} symbols: {self.symbols}")
            return True
    except Exception as e:
            logger.error(f"Pre-market screening failed: {e}")
            return False
    def _simulate_trading_day(self) -> bool:
        """
        Simulate a full trading day using historical data.
        In paper trading, we use the day's historical data to simulate trades.
        """
        try:
            # For each symbol, simulate the day's trading
            for symbol in self.symbols:
                # Get day's data (in real trading, this would be real-time)
                # For paper trading, we use the day's historical data
                pass
                        # Process signals through order manager
            # This is where the actual trading logic runs
            self._process_trading_signals()
                        return True
    except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    def _run_pre_market_screening(self) -> bool:
        """Run pre-market screener to find stocks in play."""
        try:
            # Use screener to find active symbols
            self.symbols = self.screener.scan_pre_market()
            logger.info(f"Pre-market screener found {len(self.symbols)} symbols: {self.symbols}")
            return True
    except Exception as e:
            logger.error(f"Pre-market screening failed: {e}")
            return False
    def _simulate_trading_day(self) -> bool:
        """
        Simulate a full trading day using historical data.
        In paper trading, we use the day's historical data to simulate trades.
        """
        try:
            # For each symbol, simulate the day's trading
            for symbol in self.symbols:
                # Get day's data (in real trading, this would be real-time)
                # For paper trading, we use the day's historical data
                pass
                        # Process signals through order manager
            # This is where the actual trading logic runs
            self._process_trading_signals()
                        return True
    except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return False
    def _process_trading_signals(self):
        """Process trading signals through the full pipeline."""
        try:
            # Get current market data
            current_prices = {}
            for symbol in self.symbols:
                # In paper trading, use mock prices
                current_prices[symbol] = 2500.0 + np.random.normal(0, 10)
                        # Get features for current state
            # In real trading, this would be computed from live data
            features_data = {}
                        # Generate signals
            signals = self._generate_signals()
                        # Process through order manager
            if signals:
                self.order_manager.process_signals(
                    signals_df=signals,
                    current_prices=current_prices,
                    market_data=None,
                    features_data=None
                )
                        # Manage open positions
            self.order_manager.manage_open_positions(current_prices)
                except Exception as e:
            logger.error(f"Signal processing failed: {e}")
    def _generate_signals(self):
        """Generate trading signals from models."""
        # This would use the trained models to generate signals
        # For paper trading, return empty for now
        return None

    def _save_daily_state(self, current_date: datetime):
        """Save daily state to disk."""
        try:
            state_file = f"./data/paper_trading/state_{current_date.strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2, default=str)
            logger.info(f"Daily state saved to {state_file}")
    except Exception as e:
            logger.error(f"Failed to save daily state: {e}")
    def generate_validation_report(self) -> dict:
        """Generate comprehensive validation report."""
        logger.info("Generating validation report...")

        # Calculate final metrics
        trades = self.state.trades
        if not trades:
            return {"error": "No trades executed"}

        # Basic metrics
        total_trades = len(trades)
        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning) / total_trades if total_trades > 0 else 0
        total_pnl = sum(t.pnl for t in trades)
        avg_win = np.mean([t.pnl for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl for t in losing]) if losing else 0
        profit_factor = abs(sum(t.pnl for t in winning) / sum(t.pnl for t in losing)) if losing else float('inf')

        # Calculate Sharpe
        returns = [t.pnl for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Max drawdown
        equity_curve = np.cumsum([0] + [t.pnl for t in trades])
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Monte Carlo stress test
        mc = MonteCarloStressTester(n_simulations=5000)
        mc_result = mc.run(pd.DataFrame({'return': [t.pnl for t in trades]}))

        # DSR
        returns = np.array([t.pnl for t in trades])
        dsr = calculate_dsr(
            np.array([t.pnl for t in trades]),
            n_trials=1000,
            variance_of_sharpes=0.5,
            annualization_factor=252
        )

        report = {
            "period": {
                "start": self.state.start_date,
                "end": self.state.end_date,
                "days": (datetime.strptime(self.state.end_date, "%Y-%m-%d") - 
                        datetime.strptime(self.state.start_date, "%Y-%m-%d")).days
            },
            "capital": {
                "initial": self.state.initial_capital,
                "final": self.state.current_capital,
                "total_pnl": self.state.total_pnl,
                "return_pct": (self.state.current_capital - self.state.initial_capital) / self.state.initial_capital * 100
            },
            "trading": {
                "total_trades": self.state.total_trades,
                "winning_trades": self.state.winning_trades,
                "losing_trades": self.state.losing_trades,
                "win_rate": self.state.win_rate,
                "profit_factor": self.state.profit_factor,
                "avg_win": self.state.avg_win,
                "avg_loss": self.state.avg_loss,
                "max_consecutive_losses": self.state.max_consecutive_losses,
            },
            "risk": {
                "max_drawdown": self.state.max_drawdown,
                "max_drawdown_pct": self.state.max_drawdown_pct,
                "sharpe_ratio": self.state.sharpe_ratio,
                "win_rate": self.state.win_rate,
            },
            "monte_carlo": mc_result,
            "deflated_sharpe": dsr.dsr,
            "go_no_go": self._make_go_no_go_decision()
        }

        return report
    def _make_go_no_go_decision(self) -> dict:
        """Make go/no-go decision based on validation criteria."""
        criteria = {
            "sharpe_ratio": {
                "value": self.state.sharpe_ratio,
                "threshold": 1.0,
                "pass": self.state.sharpe_ratio >= 1.0
            },
            "win_rate": {
                "value": self.state.win_rate,
                "threshold": 0.50,
                "pass": self.state.win_rate >= 0.50
            },
            "profit_factor": {
                "value": self.state.profit_factor,
                "threshold": 1.3,
                "pass": self.state.profit_factor >= 1.3
            },
            "max_drawdown": {
                "value": self.state.max_drawdown_pct,
                "threshold": 0.10,
                "pass": self.state.max_drawdown_pct <= 0.10
            },
            "min_trades": {
                "value": self.state.total_trades,
                "threshold": 30,
                "pass": self.state.total_trades >= 30
            }
        }

        all_pass = all(c["pass"] for c in criteria.values())

        return {
            "decision": "GO" if all_pass else "NO-GO",
            "criteria": criteria,
            "reason": "All criteria met" if all_pass else "Some criteria not met"
        }
    def save_report(self, report: dict, filepath: str = None):
        """Save validation report to file."""
        if filepath is None:
            filepath = f"./data/paper_trading/report_{datetime.now().strftime('%Y%m%d')}.json"

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {filepath}")
    def run_validation(self, days: int = 10) -> dict:
        """Run complete paper trading validation."""
        logger.info("Starting paper trading validation...")

        # Initialize
        if not self.initialize():
            return {"error": "Initialization failed"}

        # Load or train models
        if not self.load_models():
            if not self.train_models():
                return {"error": "Failed to load or train models"}

        # Run paper trading
        if not self.run_paper_trading(days=10):
            return {"error": "Paper trading failed"}

        # Generate report
        report = self.generate_validation_report()

        # Save report
        self.save_report(report)

        return report


    def main():
    parser = argparse.ArgumentParser(description="Paper Trading Validation Runner")
    parser.add_argument("--symbols", nargs="+", default=["RELIANCE", "HDFCBANK", "TCS", "INFY"], 
                        help="Symbols to trade")
    parser.add_argument("--days", type=int, default=10, help="Number of days to run")
    parser.add_argument("--capital", type=float, default=1000000.0, help="Initial capital")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--train", action="store_true", help="Train models before running")
    args = parser.parse_args()
        # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
        # Create engine
    engine = PaperTradingEngine(
        symbols=args.symbols,
        initial_capital=args.capital,
        config_path=args.config,
        paper_trading=True
    )
        if args.train:
        logger.info("Training models...")
        if not engine.train_models():
            logger.error("Model training failed")
            return 1
        # Run validation
    report = engine.run_validation(days=args.days)
        if "error" in report:
        logger.error(f"Validation failed: {report['error']}")
        return 1
        # Print summary
    print("\n" + "="*60)
    print("PAPER TRADING VALIDATION REPORT")
    print("="*60)
    print(f"Decision: {report['go_no_go']['decision']}")
    print(f"Reason: {report['go_no_go']['reason']}")
    print(f"Period: {report['period']['start']} to {report['period']['end']}")
    print(f"Total PnL: {report['capital']['total_pnl']:.2f} ({report['capital']['return_pct']:.2f}%)")
    print(f"Total Trades: {report['trading']['total_trades']}")
    print(f"Win Rate: {report['trading']['win_rate']:.2%}")
    print(f"Profit Factor: {report['trading']['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {report['risk']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {report['risk']['max_drawdown_pct']:.2%}")
    print(f"Monte Carlo P(Profit): {report['monte_carlo']['probability_of_profit']:.2%}")
    print(f"DSR: {report['deflated_sharpe']:.4f}")
    print("="*60)
        return 0 if report['go_no_go']['decision'] == 'GO' else 1


if __name__ == "__main__":
    sys.exit(main())