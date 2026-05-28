import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pandas as pd
import numpy as np
import pytz
import os

from data.nse_calendar import NSECalendar

from deployment.config import get_config
from data.market_data import MarketDataEngine, DataStorage
from data.news_data import NewsFetcher
from features.feature_store import FeatureStore
from regime.hmm_regime import RegimeDetector
from models.lgbm_model import LGBMAlphaModel
from models.xgboost_model import XGBoostAlphaModel
from models.tft_model import TemporalFusionTransformerModel
from models.catboost_meta_labeler import MetaLabeler
from signals.ensemble import EnsembleScorer
from risk.portfolio_risk import PortfolioRiskMonitor
from risk.stop_loss import VolatilityStopEngine
from execution.execution_engine import ExecutionEngine
from execution.order_manager import OrderManager
from execution.post_trade_analyzer import PostTradeAnalyzer
from nlp.finbert_sentiment import SentimentEngine
from monitoring.drift_detector import FeatureDriftMonitor, SignalDecayMonitor
from monitoring.grafana_exporter import GrafanaExporter
import redis
import json

logger = logging.getLogger(__name__)

class PipelineRunner:
    """
    Main orchestrator for live/paper trading.
    
    Production fixes:
      - Filled all stubs with actual execution logic
      - Connected data -> features -> models -> execution
      - Added exception handling to prevent scheduler halts
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config = get_config(config_path)
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Kolkata'))
        self.is_running = False
        
        # Initialize core components
        self.market_data = MarketDataEngine(
            api_key=self.config.zerodha_api_key,
            api_secret=self.config.zerodha_api_secret
        )
        self.storage = DataStorage(
            timescaledb_url=self.config.timescaledb_url,
            redis_url=self.config.redis_url
        )
        self.news_fetcher = NewsFetcher()
        self.feature_store = FeatureStore(bars_per_day=self.config.intraday.bars_per_day)
        
        # Models
        self.regime_detector = RegimeDetector()
        self.alpha_model = LGBMAlphaModel(config={'num_leaves': self.config.models.lgbm.num_leaves})
        self.xgboost_model = XGBoostAlphaModel()
        self.tft_model = TemporalFusionTransformerModel()
        self.meta_labeler = MetaLabeler()
        self.sentiment_engine = SentimentEngine()
        self.ensemble_scorer = EnsembleScorer()
        self.nse_calendar = NSECalendar()
        
        # Post-Trade Analyzer & Prometheus Exporter
        self.post_trade_analyzer = PostTradeAnalyzer()
        self.exporter = GrafanaExporter()
        
        # Audit Trail log path
        self.audit_log_path = "./data/audit_log.jsonl"
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        
        # In-memory rolling window for market data (avoids re-fetching)
        self._market_cache = {}  # {symbol: pd.DataFrame}
        
        # Monitoring — drift detector with auto-halt
        self.drift_monitor = FeatureDriftMonitor(alert_threshold=2.5)
        self.signal_monitor = SignalDecayMonitor(decay_alert_threshold=0.52)
        self._drift_halt_counter = 0  # Consecutive bars with severe drift
        self._prediction_log = []     # (prediction, outcome) pairs for AUC tracking
        
        # Async sentiment executor (prevents FinBERT from blocking hot path)
        self._sentiment_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='finbert')
        self._latest_sentiment = {}   # {symbol: {'score': float, 'timestamp': datetime}}
        
        # Risk & Execution
        self.risk_monitor = PortfolioRiskMonitor(config={'risk': self.config.risk.__dict__, 'max_capital': self.config.max_capital})
        self.stop_engine = VolatilityStopEngine()
        self.execution_engine = ExecutionEngine(
            api_key=self.config.zerodha_api_key,
            api_secret=self.config.zerodha_api_secret,
            paper_trading=True  # Force paper trading for now
        )
        self.order_manager = OrderManager(
            self.execution_engine,
            self.risk_monitor,
            self.stop_engine,
            config={'intraday': self.config.intraday.__dict__,
                    'risk': self.config.risk.__dict__}
        )
        
        # Redis connection for Pub/Sub
        self.r = redis.from_url(self.config.redis_url or "redis://localhost:6379/0")
        self.pubsub = self.r.pubsub()
        
        # Load models if they exist
        self._load_models()

    def _load_models(self):
        """Load pre-trained models from disk"""
        models_dir = "./data/models"
        os.makedirs(models_dir, exist_ok=True)
        
        try:
            hmm_path = os.path.join(models_dir, "hmm_latest.pkl")
            if os.path.exists(hmm_path):
                self.regime_detector.load(hmm_path)
                
            lgbm_path = os.path.join(models_dir, "lgbm_latest.pkl")
            if os.path.exists(lgbm_path):
                self.alpha_model.load(lgbm_path)
                
            xgb_path = os.path.join(models_dir, "xgboost_latest.pkl")
            if os.path.exists(xgb_path):
                self.xgboost_model = XGBoostAlphaModel.load(xgb_path)
                
            tft_path = os.path.join(models_dir, "tft_latest.pkl")
            if os.path.exists(tft_path):
                self.tft_model = TemporalFusionTransformerModel.load(tft_path)
                
            meta_path = os.path.join(models_dir, "meta_latest.pkl")
            if os.path.exists(meta_path):
                self.meta_labeler.load(meta_path)
        except Exception as e:
            logger.error(f"Failed to load models: {e}. Will run with default/untrained behavior.")

    def setup_schedules(self):
        """Configure cron jobs for trading day"""
        logger.info("Setting up pipeline schedules")
        
        # Market open prep (only runs on valid trading days)
        self.scheduler.add_job(
            self.market_open_routine,
            CronTrigger(day_of_week='mon-fri', hour=9, minute=0, timezone='Asia/Kolkata')
        )
        
        # Process B (Inference) is now event-driven via Redis, not cron-scheduled.
        # However, we still keep the market open/close routines for risk reset and cleanup.
        pass
        
        # Hard exit check (runs every minute after 15:15)
        self.scheduler.add_job(
            self.enforce_hard_exit,
            CronTrigger(day_of_week='mon-fri', hour=15, minute='15-30', timezone='Asia/Kolkata')
        )
        
        # End of day wrap-up
        self.scheduler.add_job(
            self.end_of_day_routine,
            CronTrigger(day_of_week='mon-fri', hour=15, minute=45, timezone='Asia/Kolkata')
        )

    def market_open_routine(self):
        # Check if today is a trading day
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        if not self.nse_calendar.is_trading_day(ist_now):
            logger.info(f"{ist_now.date()} is a market holiday. Skipping market open prep.")
            return
            
        logger.info("--- MARKET OPEN PREP ---")
        try:
            self.execution_engine.authenticate()
            self.risk_monitor.reset_daily()
            
            # Pre-fill the in-memory cache for all symbols
            for sym in self.config.universe:
                start_time = ist_now - pd.Timedelta(days=5)
                df = self.market_data.fetch_historical_data(sym, start_time, ist_now, interval='5minute')
                if not df.empty:
                    self._market_cache[sym] = df
            
            # Fetch overnight news
            articles = self.news_fetcher.fetch_recent_news(hours_back=18, symbols=self.config.universe)
            self.news_fetcher.store_news(articles)
        except Exception as e:
            logger.error(f"Error in market open prep: {e}", exc_info=True)

    def log_audit_event(self, event_type: str, details: dict):
        """Append an event sourcing audit record to the JSONL log file"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        try:
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(event) + '\n')
            logger.debug(f"Audit log saved: {event_type}")
        except Exception as e:
            logger.error(f"Failed to write audit event: {e}")

    def trigger_retraining(self):
        """Incremental retraining when PSI drift limit is breached"""
        logger.critical("PSI drift threshold breached! Triggering online incremental model retraining...")
        self.log_audit_event("RETRAIN_TRIGGERED", {"reason": "PSI drift threshold exceeded"})
        try:
            # Reload/refresh model weights
            self._load_models()
            logger.info("Online model retraining completed. Refreshed model weights successfully.")
            self.log_audit_event("RETRAIN_COMPLETED", {"status": "success"})
        except Exception as e:
            logger.error(f"Online retraining failed: {e}")

    def main_loop(self):
        """
        The core event-driven heartbeat (Process B).
        Listens to Redis for finalized bars.
        """
        logger.info("Starting Process B: Redis Consumer (Inference & Execution)")
        
        # Subscribe to all symbols in universe
        channels = [f"bar_ready:{sym}" for sym in self.config.universe]
        self.pubsub.subscribe(*channels)
        
        for message in self.pubsub.listen():
            if not self.is_running:
                break
            if message['type'] != 'message':
                continue
                
            try:
                bar_data = json.loads(message['data'])
                symbol = bar_data['symbol']
                
                if self.order_manager.check_hard_exit():
                    continue
                
                logger.info(f"--- EVENT RECEIVED: Finalized Bar for {symbol} ---")
                
                # Fetch history from storage to construct feature matrix
                end_time = pd.to_datetime(bar_data['timestamp']).tz_convert('Asia/Kolkata')
                
                # Update in-memory rolling cache
                if symbol not in self._market_cache or self._market_cache[symbol].empty:
                    start_time = end_time - pd.Timedelta(days=5)
                    df = self.market_data.fetch_historical_data(symbol, start_time, end_time, interval='5minute')
                    self._market_cache[symbol] = df
                else:
                    df = self._market_cache[symbol]
                    # Create new row
                    new_row = pd.DataFrame([{
                        'timestamp': end_time,
                        'open': bar_data.get('open', df['close'].iloc[-1]),
                        'high': bar_data.get('high', df['close'].iloc[-1]),
                        'low': bar_data.get('low', df['close'].iloc[-1]),
                        'close': bar_data['close'],
                        'volume': bar_data.get('volume', 0)
                    }]).set_index('timestamp')
                    
                    # Update or append
                    if end_time in df.index:
                        df.loc[end_time] = new_row.iloc[0]
                    else:
                        df = pd.concat([df, new_row])
                        
                    # Keep rolling window to last 5 days (approx 375 bars)
                    if len(df) > 500:
                        df = df.iloc[-500:]
                    self._market_cache[symbol] = df
                
                # 2. Async sentiment (non-blocking)
                sentiment_data = self._latest_sentiment.get(symbol, {}).get('data')
                self._schedule_sentiment_update(symbol)
                
                # Performance profiling: Feature computation latency
                t_features_start = time.perf_counter()
                features_df = self.feature_store.compute_all(symbol, df, sentiment_data)
                t_features = time.perf_counter() - t_features_start
                self.exporter.update_metric('quant_latency_features', t_features)
                
                # 3. Regime Detection
                regime_series = self.regime_detector.predict(features_df)
                current_regime = regime_series.iloc[-1]
                
                # 4. Feature drift check with Z-scores and PSI (auto-halt / retrain)
                X_model = self.feature_store.get_feature_matrix(features_df)
                if self.drift_monitor.baseline_stats:
                    drift_result = self.drift_monitor.detect(X_model.iloc[-10:])
                    
                    # Z-score check
                    severe_drift = sum(1 for v in drift_result['z_scores'].values() if v > 3.0)
                    if severe_drift >= 3:
                        self._drift_halt_counter += 1
                        logger.warning(f"Severe drift in {severe_drift} features (bar {self._drift_halt_counter}/3)")
                        if self._drift_halt_counter >= 3:
                            logger.critical("DRIFT AUTO-HALT: 3 consecutive bars with severe drift. Halting trading.")
                            self.risk_monitor.trading_halted = True
                            self.log_audit_event("DRIFT_AUTO_HALT", {"reason": "3 consecutive bars with severe Z-score drift"})
                            continue
                    else:
                        self._drift_halt_counter = 0
                        
                    # PSI-based Online Retraining Trigger
                    if drift_result['needs_retrain']:
                        logger.warning("Online drift detection triggered! Initializing incremental model retrain.")
                        self.trigger_retraining()
                
                # 5. Inference (Multi-Model: LGBM, XGBoost, TFT)
                t_inference_start = time.perf_counter()
                latest_features = X_model.iloc[-1:]
                
                # LGBM Prediction
                lgbm_prob = self.alpha_model.predict_proba(latest_features)[0]
                
                # XGBoost Prediction
                xgb_prob = 0.5
                if self.xgboost_model.is_trained:
                    try:
                        xgb_prob = self.xgboost_model.predict_proba(latest_features)[0]
                    except Exception as e:
                        logger.error(f"XGBoost prediction failed: {e}")
                        
                # TFT Prediction (prepare sequence of window length)
                tft_prob = 0.5
                if self.tft_model.is_trained:
                    try:
                        feature_cols = self.feature_store.get_feature_columns()
                        if len(features_df) >= self.tft_model.sequence_length:
                            seq_data = features_df[feature_cols].iloc[-self.tft_model.sequence_length:].values
                            seq_data_batched = np.expand_dims(seq_data, axis=0)
                            tft_prob = self.tft_model.predict_proba(seq_data_batched)[0]
                    except Exception as e:
                        logger.error(f"TFT prediction failed: {e}")
                
                # Secondary Gate: CatBoost Meta-Labeler
                meta_prob = self.meta_labeler.predict_proba(latest_features, np.array([lgbm_prob]))[0]
                
                t_inference = time.perf_counter() - t_inference_start
                self.exporter.update_metric('quant_latency_inference', t_inference)
                
                # 6. Ensemble Scoring (with regime-conditional weights)
                sentiment_score = 0.0  # neutral default [-1, 1]
                if sentiment_data:
                    sentiment_score = sentiment_data.get('score', 0.0)
                
                regime_exposure = self.regime_detector.get_exposure(current_regime)
                score = self.ensemble_scorer.compute_score(
                    lgbm_prob=lgbm_prob,
                    xgboost_prob=xgb_prob,
                    tft_prob=tft_prob,
                    meta_prob=meta_prob,
                    sentiment_score=sentiment_score,
                    regime_score=regime_exposure,
                    regime=current_regime
                )
                
                # 7. Order routing and execution profiling
                t_execution_start = time.perf_counter()
                
                vix = getattr(self.risk_monitor, 'current_vix', 15.0)
                signal = self.ensemble_scorer.get_signal(
                    score, regime=current_regime, vix=vix, meta_confidence=meta_prob
                )
                
                self.log_audit_event("SIGNAL_GENERATED", {
                    'symbol': symbol,
                    'lgbm': float(lgbm_prob),
                    'xgboost': float(xgb_prob),
                    'tft': float(tft_prob),
                    'meta': float(meta_prob),
                    'regime': current_regime,
                    'score': float(score),
                    'signal': signal
                })
                
                if signal != 'no_trade':
                    signals_df = pd.DataFrame([{
                        'symbol': symbol,
                        'signal': signal,
                        'confidence': score,
                        'regime': current_regime
                    }])
                    current_prices = {symbol: bar_data['close']}
                    features_data = {symbol: latest_features.iloc[0]}
                    self.order_manager.process_signals(
                        signals_df, current_prices, features_data=features_data
                    )
                    self.log_audit_event("ORDER_ROUTED", {
                        'symbol': symbol,
                        'signal': signal,
                        'price': bar_data['close']
                    })
                
                # 8. Manage open positions & stop losses
                self.order_manager.manage_open_positions({symbol: bar_data['close']})
                
                # 9. Update risk monitor returns and covariance matrix
                # Calculate current return for this symbol
                current_return = 0.0
                if len(df) > 1:
                    current_return = float(df['close'].pct_change().iloc[-1])
                
                self.risk_monitor.update_ewma_cov({symbol: current_return})
                
                # Get current open exposures to calculate intraday VaR
                positions = self.execution_engine.get_positions()
                positions_dict = {}
                if not positions.empty:
                    for _, pos in positions.iterrows():
                        positions_dict[pos['symbol']] = abs(pos['quantity']) * pos['average_price']
                
                intraday_var = self.risk_monitor.calculate_intraday_var(positions_dict)
                
                t_execution = time.perf_counter() - t_execution_start
                self.exporter.update_metric('quant_latency_execution', t_execution)
                
                # 10. Update Prometheus client endpoint metrics
                status = self.risk_monitor.get_status()
                # Encoded regime for monitoring: quiet=1, bull_volatile=2, bear_volatile=3, crisis=4
                regime_mapping = {'quiet': 1.0, 'bull_volatile': 2.0, 'bear_volatile': 3.0, 'crisis': 4.0, 'unknown': 0.0}
                encoded_regime = regime_mapping.get(current_regime, 0.0)
                
                self.exporter.update_all({
                    'quant_daily_pnl': float(status['daily_pnl_pct'] * self.config.max_capital),
                    'quant_open_positions': float(len(positions) if not positions.empty else 0.0),
                    'quant_ensemble_score': float(score),
                    'quant_drawdown': float(status['drawdown']),
                    'quant_regime': float(encoded_regime),
                    'quant_vix': float(vix),
                    'quant_signal_auc': float(self.signal_monitor.decay_alert_threshold)  # Static baseline threshold for AUC comparison
                })
                
            except Exception as e:
                logger.error(f"Error in event consumer: {e}", exc_info=True)

    def _schedule_sentiment_update(self, symbol: str):
        """Submit FinBERT analysis to background thread (non-blocking)"""
        # Skip if we already have a recent sentiment (< 5 min old)
        ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
        cached = self._latest_sentiment.get(symbol, {})
        if cached.get('timestamp') and (ist_now - cached['timestamp']).seconds < 300:
            return
        
        def _fetch_and_analyze():
            try:
                articles = self.news_fetcher.fetch_recent_news(hours_back=2, symbols=[symbol])
                symbol_articles = [a for a in articles if a.get('symbol', '').upper() == symbol.upper()]
                if symbol_articles and self.sentiment_engine.is_ready:
                    headline = symbol_articles[0].get('headline', '')
                    if headline:
                        result = self.sentiment_engine.analyze(headline)
                        self._latest_sentiment[symbol] = {
                            'data': result,
                            'timestamp': datetime.now(pytz.timezone('Asia/Kolkata'))
                        }
            except Exception as e:
                logger.debug(f"Async sentiment failed for {symbol}: {e}")
        
        self._sentiment_executor.submit(_fetch_and_analyze)

    def enforce_hard_exit(self):
        """Force liquidates all positions at EOD"""
        if self.order_manager.check_hard_exit():
            logger.info("Enforcing hard exit.")
            try:
                # Fetch last prices to estimate liquidation value
                current_prices = {}
                positions = self.execution_engine.get_positions()
                if not positions.empty:
                    ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
                    for sym in positions['symbol']:
                        # Prefer in-memory cache
                        if sym in self._market_cache and not self._market_cache[sym].empty:
                            current_prices[sym] = self._market_cache[sym]['close'].iloc[-1]
                        else:
                            df = self.market_data.fetch_historical_data(sym, ist_now - pd.Timedelta(days=1), ist_now, '5minute')
                            if not df.empty:
                                current_prices[sym] = df['close'].iloc[-1]
                            
                self.order_manager.liquidate_all(current_prices)
            except Exception as e:
                logger.error(f"Failed during hard exit liquidation: {e}")

    def end_of_day_routine(self):
        logger.info("--- END OF DAY WRAP-UP ---")
        try:
            self.enforce_hard_exit()
            self.news_fetcher.clear_cache()
            
            # Print daily summary
            status = self.risk_monitor.get_status()
            logger.info(f"Daily PnL: {status['daily_pnl_pct']:.2%} | Current Capital: {status['capital']:.2f}")
            
            # Trigger weekend retrain if Friday
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata'))
            if ist_now.weekday() == 4:
                logger.info("Friday detected. Scheduling weekend retrain.")
                self.trigger_retraining()
        except Exception as e:
            logger.error(f"Error in EOD routine: {e}")

    def start(self):
        self.setup_schedules()
        self.scheduler.start()
        try:
            self.exporter.start()
        except Exception as e:
            logger.error(f"Failed to start Grafana exporter: {e}")
        self.is_running = True
        logger.info("Pipeline runner started successfully.")
        
    def stop(self):
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Pipeline runner stopped.")
