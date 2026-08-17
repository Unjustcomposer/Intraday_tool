import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from intraday_quant_system.backtesting.walk_forward import (
    PurgedWalkForwardValidator as WalkForwardValidator,
)


def test_walk_forward_validator():
    validator = WalkForwardValidator(
        n_splits=3, train_size=0.7, val_size=0.15, purge_bars=10, embargo_bars=10
    )

    # Mock data
    dates = pd.date_range("2023-01-01", periods=500, freq="15min")
    df = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(len(dates)) * 0.5),
            "label": np.random.randint(0, 2, len(dates)),
        },
        index=dates,
    )

    # Mock factory
    def model_factory(train_df, val_df):
        return {"sharpe": 1.2, "win_rate": 0.55}

    results = validator.run(df, model_factory)

    assert len(results) > 0

    agg = validator.aggregate_results(results)
    assert "avg_sharpe" in agg
