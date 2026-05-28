import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from deployment.pipeline_runner import PipelineRunner
from execution.execution_engine import ExecutionEngine
from execution.order_manager import OrderManager
from backtesting.backtest_engine import BacktestEngine, BacktestResult
from backtesting.walk_forward import WalkForwardValidator

def test_pipeline_initialization():
    runner = PipelineRunner()
    
    assert runner.execution_engine is not None
    assert runner.order_manager is not None
    assert not runner.is_running
    
def test_walk_forward_validator():
    validator = WalkForwardValidator(training_window=100, validation_window=20, step_size=10)
    
    # Mock data
    dates = pd.date_range('2023-01-01', '2023-08-01')
    df = pd.DataFrame({'close': np.random.randn(len(dates))}, index=dates)
    
    # Mock factory
    def model_factory(train_df, val_df):
        return {
            'sharpe': 1.2,
            'win_rate': 0.55
        }
        
    results = validator.run(df, model_factory)
    
    assert len(results) > 0
    
    agg = validator.aggregate_results(results)
    assert 'avg_sharpe' in agg
