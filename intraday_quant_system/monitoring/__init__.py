"""
Monitoring Package
==================

Provides structured logging, Prometheus metrics, Monte Carlo stress testing,
Deflated Sharpe Ratio, and PnL attribution utilities.
"""

from .structured_logging import (
    get_logger,
    set_correlation_id,
    clear_correlation_id,
    with_correlation_id,
    LogLevel,
    StructuredLogRecord,
    StructuredJSONFormatter,
    CorrelationIdFilter,
    MetricsExporter,
    MetricLabels,
    Timer,
    timed,
    get_metrics_exporter,
    set_correlation_id,
    clear_correlation_id,
    with_correlation_id,
    correlation_id_var,
)

from .monte_carlo import (
    MonteCarloStressTester,
    MonteCarloResult,
)

from .dsr import (
    calculate_dsr,
    calculate_psr,
    calculate_min_track_record,
    sharpe_ratio_confidence_interval,
    DSRResult,
    DeflatedSharpeRatio,
)

from .attribution import (
    PnLAttribution,
    AttributionResult,
    run_attribution_backtest,
    compute_factor_returns,
)

__all__ = [
    # Structured logging
    "get_logger",
    "set_correlation_id",
    "clear_correlation_id",
    "with_correlation_id",
    "LogLevel",
    "StructuredLogRecord",
    "StructuredJSONFormatter",
    "CorrelationIdFilter",
    "MetricsExporter",
    "MetricLabels",
    "Timer",
    "timed",
    "get_metrics_exporter",
    "set_correlation_id",
    "clear_correlation_id",
    "with_correlation_id",
    "correlation_id_var",
    
    # Monte Carlo
    "MonteCarloStressTester",
    "MonteCarloResult",
    
    # Deflated Sharpe Ratio
    "calculate_dsr",
    "calculate_psr",
    "calculate_min_track_record",
    "sharpe_ratio_confidence_interval",
    "DSRResult",
    "DeflatedSharpeRatio",
    
    # Attribution
    "PnLAttribution",
    "AttributionResult",
    "run_attribution_backtest",
    "compute_factor_returns",
]