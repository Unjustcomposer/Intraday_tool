"""
Structured JSON Logging + Prometheus Metrics Exporter
======================================================

Provides:
1. Structured JSON logging with consistent fields for log aggregation
2. Prometheus metrics exporter for monitoring system health & performance
3. Context-aware logging with correlation IDs for distributed tracing
"""

import json
import logging
import sys
import time
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Dict, Optional, Callable
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway

# Context variable for correlation ID propagation
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class LogLevel(Enum):
    """Structured log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class StructuredLogRecord:
    """Structured log record with consistent fields"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    correlation_id: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.__dict__, default=str)


class StructuredJSONFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs"""
    
    def __init__(self, service_name: str = "intraday-quant", include_extra: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        # Get correlation ID from context
        correlation_id = correlation_id_var.get()
        
        # Build structured log record
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "correlation_id": correlation_id,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename", "funcName",
                    "levelname", "levelno", "lineno", "module", "msecs",
                    "message", "name", "pathname", "process", "processName",
                    "relativeCreated", "thread", "threadName", "exc_info",
                    "exc_text", "stack_info", "getMessage"
                }:
                    extra_fields[key] = value
            if extra_fields:
                log_entry["extra"] = extra_fields
        
        return json.dumps(log_entry, default=str)


class CorrelationIdFilter(logging.Filter):
    """Filter to inject correlation ID into log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get()
        return True


def get_logger(name: str, service_name: str = "intraday-quant") -> logging.Logger:
    """Get a logger configured with structured JSON formatting"""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter(service_name=service_name))
        handler.addFilter(CorrelationIdFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    
    return logger


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID in context variable, generate if not provided"""
    if correlation_id is None:
        correlation_id = f"{int(time.time() * 1000)}-{threading.current_thread().ident}"
    correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id():
    """Clear correlation ID from context"""
    correlation_id_var.set(None)


def with_correlation_id(func: Callable) -> Callable:
    """Decorator to automatically set/clear correlation ID for function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        corr_id = set_correlation_id()
        try:
            return func(*args, **kwargs)
        finally:
            clear_correlation_id()
    return wrapper


@dataclass
class MetricLabels:
    """Standard labels for Prometheus metrics"""
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    regime: Optional[str] = None
    side: Optional[str] = None
    status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class MetricsExporter:
    """
    Prometheus metrics exporter for intraday trading system.
    Exposes metrics for: latency, throughput, errors, PnL, risk metrics.
    """
    
    def __init__(
        self, 
        registry: Optional[CollectorRegistry] = None,
        push_gateway_url: Optional[str] = None,
        job_name: str = "intraday_quant",
        instance: Optional[str] = None
    ):
        self.registry = registry or CollectorRegistry()
        self.push_gateway_url = push_gateway_url
        self.job_name = job_name
        self.instance = instance or f"{socket.gethostname()}:{os.getpid()}"
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics"""
        # Latency metrics
        self.order_latency = Histogram(
            "order_latency_seconds",
            "Order placement latency in seconds",
            ["symbol", "side", "order_type", "status"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.feature_computation_latency = Histogram(
            "feature_computation_latency_seconds",
            "Feature computation latency in seconds",
            ["symbol"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.model_inference_latency = Histogram(
            "model_inference_latency_seconds",
            "Model inference latency in seconds",
            ["model", "symbol"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # Throughput metrics
        self.orders_total = Counter(
            "orders_total",
            "Total number of orders placed",
            ["symbol", "side", "order_type", "status"],
            registry=self.registry
        )
        
        self.trades_total = Counter(
            "trades_total",
            "Total number of trades executed",
            ["symbol", "side", "strategy", "regime"],
            registry=self.registry
        )
        
        self.signals_generated = Counter(
            "signals_generated_total",
            "Total number of signals generated",
            ["symbol", "signal", "regime"],
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            "errors_total",
            "Total number of errors",
            ["component", "error_type", "symbol"],
            registry=self.registry
        )
        
        # Risk metrics
        self.current_drawdown = Gauge(
            "current_drawdown_pct",
            "Current drawdown percentage",
            ["symbol"],
            registry=self.registry
        )
        
        self.current_positions = Gauge(
            "current_positions",
            "Current open positions",
            ["symbol", "side"],
            registry=self.registry
        )
        
        self.portfolio_value = Gauge(
            "portfolio_value",
            "Current portfolio value in base currency",
            registry=self.registry
        )
        
        self.daily_pnl = Gauge(
            "daily_pnl",
            "Daily PnL in base currency",
            ["strategy"],
            registry=self.registry
        )
        
        self.current_exposure = Gauge(
            "current_exposure_pct",
            "Current portfolio exposure percentage",
            registry=self.registry
        )
        
        self.risk_limit_breaches = Counter(
            "risk_limit_breaches_total",
            "Total number of risk limit breaches",
            ["limit_type", "symbol"],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            "model_accuracy",
            "Model prediction accuracy",
            ["model", "symbol", "horizon"],
            registry=self.registry
        )
        
        self.model_predictions = Counter(
            "model_predictions_total",
            "Total model predictions",
            ["model", "symbol", "prediction"],
            registry=self.registry
        )
        
        # System health
        self.uptime_seconds = Gauge(
            "uptime_seconds",
            "Service uptime in seconds",
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry
        )
    
    def record_order_latency(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        status: str, 
        latency_seconds: float
    ):
        """Record order placement latency"""
        self.order_latency.labels(
            symbol=symbol, side=side, order_type=order_type, status=status
        ).observe(latency_seconds)
    
    def record_order(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        status: str
    ):
        """Record an order placement"""
        self.orders_total.labels(
            symbol=symbol, side=side, order_type=order_type, status=status
        ).inc()
    
    def record_trade(
        self, 
        symbol: str, 
        side: str, 
        strategy: str, 
        regime: str
    ):
        """Record a trade execution"""
        self.trades_total.labels(
            symbol=symbol, side=side, strategy=strategy, regime=regime
        ).inc()
    
    def record_signal(
        self, 
        symbol: str, 
        signal: str, 
        regime: str
    ):
        """Record a signal generation"""
        self.signals_generated.labels(
            symbol=symbol, signal=signal, regime=regime
        ).inc()
    
    def record_error(
        self, 
        component: str, 
        error_type: str, 
        symbol: Optional[str] = None
    ):
        """Record an error occurrence"""
        self.errors_total.labels(
            component=component, error_type=error_type, symbol=symbol or "unknown"
        ).inc()
    
    def set_drawdown(self, symbol: str, drawdown_pct: float):
        """Set current drawdown for a symbol"""
        self.current_drawdown.labels(symbol=symbol).set(drawdown_pct)
    
    def set_position(self, symbol: str, side: str, quantity: float):
        """Set current position for a symbol"""
        self.current_positions.labels(symbol=symbol, side=side).set(quantity)
    
    def set_portfolio_value(self, value: float):
        """Set current portfolio value"""
        self.portfolio_value.set(value)
    
    def set_daily_pnl(self, strategy: str, pnl: float):
        """Set daily PnL for a strategy"""
        self.daily_pnl.labels(strategy=strategy).set(pnl)
    
    def set_exposure(self, exposure_pct: float):
        """Set current portfolio exposure"""
        self.current_exposure.set(exposure_pct)
    
    def record_risk_breach(self, limit_type: str, symbol: str):
        """Record a risk limit breach"""
        self.risk_limit_breaches.labels(limit_type=limit_type, symbol=symbol).inc()
    
    def set_model_accuracy(self, model: str, symbol: str, horizon: str, accuracy: float):
        """Set model prediction accuracy"""
        self.model_accuracy.labels(model=model, symbol=symbol, horizon=horizon).set(accuracy)
    
    def record_model_prediction(self, model: str, symbol: str, prediction: str):
        """Record a model prediction"""
        self.model_predictions.labels(model=model, symbol=symbol, prediction=prediction).inc()
    
    def set_system_health(self, uptime: float, memory_bytes: float, cpu_percent: float):
        """Set system health metrics"""
        self.uptime_seconds.set(uptime)
        self.memory_usage_bytes.set(memory_bytes)
        self.cpu_usage_percent.set(cpu_percent)
    
    def push_to_gateway(self):
        """Push metrics to Prometheus Pushgateway"""
        if self.push_gateway_url:
            push_to_gateway(
                self.push_gateway_url,
                job=self.job_name,
                registry=self.registry,
                grouping_key={"instance": self.instance}
            )
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus text format"""
        from prometheus_client import generate_latest
        return generate_latest(self.registry).decode("utf-8")


# Global metrics instance
_metrics_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter(
    push_gateway_url: Optional[str] = None,
    job_name: str = "intraday_quant"
) -> MetricsExporter:
    """Get or create global metrics exporter instance"""
    global _metrics_exporter
    if _metrics_exporter is None:
        _metrics_exporter = MetricsExporter(
            push_gateway_url=push_gateway_url,
            job_name=job_name
        )
    return _metrics_exporter


# Context manager for timing operations
class Timer:
    """Context manager for timing operations"""
    
    def __init__(
        self, 
        metrics: MetricsExporter, 
        metric_name: str, 
        labels: Optional[Dict[str, str]] = None
    ):
        self.metrics = metrics
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        # Record to appropriate histogram based on metric name
        if "order" in self.metric_name:
            self.metrics.record_order_latency(
                symbol=self.labels.get("symbol", "unknown"),
                side=self.labels.get("side", "unknown"),
                order_type=self.labels.get("order_type", "unknown"),
                status=self.labels.get("status", "unknown" if exc_type else "success"),
                latency_seconds=elapsed
            )


# Decorator for timing functions
def timed(metrics: MetricsExporter, metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(metrics, metric_name, labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


import os
import socket