from prometheus_client import start_http_server, Gauge
import logging
import threading
import time

logger = logging.getLogger(__name__)

class GrafanaExporter:
    """
    Expose Prometheus metrics endpoint (/metrics) for Grafana scraping
    Metrics to export:
      quant_daily_pnl, quant_open_positions, quant_ensemble_score,
      quant_drawdown, quant_regime, quant_vix, quant_signal_auc
    """
    def __init__(self, port: int = 8000):
        self.port = port
        self.is_running = False
        
        # Define Gauges
        self.metrics = {
            'quant_daily_pnl': Gauge('quant_daily_pnl', 'Daily PnL in INR'),
            'quant_open_positions': Gauge('quant_open_positions', 'Number of open positions'),
            'quant_ensemble_score': Gauge('quant_ensemble_score', 'Latest ensemble score'),
            'quant_drawdown': Gauge('quant_drawdown', 'Current drawdown percentage'),
            'quant_regime': Gauge('quant_regime', 'Current market regime (encoded)'),
            'quant_vix': Gauge('quant_vix', 'Current VIX level'),
            'quant_signal_auc': Gauge('quant_signal_auc', 'Rolling Signal AUC'),
            'quant_latency_features': Gauge('quant_latency_features', 'Feature computation latency in seconds'),
            'quant_latency_inference': Gauge('quant_latency_inference', 'Model inference latency in seconds'),
            'quant_latency_execution': Gauge('quant_latency_execution', 'Order execution routing latency in seconds')
        }
        
    def start(self):
        if not self.is_running:
            start_http_server(self.port)
            self.is_running = True
            logger.info(f"Started Prometheus metrics server on port {self.port}")
            
    def update_metric(self, metric_name: str, value: float):
        if metric_name in self.metrics:
            self.metrics[metric_name].set(value)
        else:
            logger.warning(f"Metric {metric_name} not found")
            
    def update_all(self, state: dict):
        for k, v in state.items():
            self.update_metric(k, v)
