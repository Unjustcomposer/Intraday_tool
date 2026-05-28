import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class WalkForwardValidator:
    """
    Purged Walk-Forward Cross Validation.
    
    Production features:
      - Actual rolling window evaluation (not mock data)
      - Purged windows to prevent leakage
    """
    def __init__(self, training_window: int = 120, validation_window: int = 20,
                 step_size: int = 20, purge_bars: int = 10, embargo_bars: int = 10):
        """
        Windows specified in number of days.
        purge_bars: Number of bars to drop from END of training set
        embargo_bars: Number of bars to drop from START of validation set
                      (prevents label horizon leakage from training price action)
        """
        self.training_window = training_window
        self.validation_window = validation_window
        self.step_size = step_size
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.results = []

    def get_splits(self, df: pd.DataFrame):
        """Generator yielding (train_idx, val_idx) for each split."""
        # Assuming df has datetime index or timestamp column
        if 'timestamp' in df.columns:
            dates = pd.to_datetime(df['timestamp']).dt.date.unique()
        elif hasattr(df.index, 'date'):
            dates = np.unique(df.index.date)
        else:
            raise ValueError("Dataframe must have datetime info for walk-forward splits")
            
        dates = sorted(dates)
        
        n_splits = 0
        for i in range(0, len(dates) - self.training_window - self.validation_window, self.step_size):
            train_dates = dates[i : i + self.training_window]
            val_dates = dates[i + self.training_window : i + self.training_window + self.validation_window]
            
            if 'timestamp' in df.columns:
                ts_date = pd.to_datetime(df['timestamp']).dt.date
                train_mask = ts_date.isin(train_dates)
                val_mask = ts_date.isin(val_dates)
                
                # Purge: drop last N bars of train (prevent label leakage forward)
                train_indices = np.where(train_mask)[0]
                if len(train_indices) > self.purge_bars:
                    train_indices = train_indices[:-self.purge_bars]
                
                # Embargo: drop first N bars of validation (prevent label leakage backward)
                val_indices = np.where(val_mask)[0]
                if len(val_indices) > self.embargo_bars:
                    val_indices = val_indices[self.embargo_bars:]
            else:
                ts_date = df.index.date
                train_mask = np.isin(ts_date, train_dates)
                val_mask = np.isin(ts_date, val_dates)
                
                train_indices = np.where(train_mask)[0]
                if len(train_indices) > self.purge_bars:
                    train_indices = train_indices[:-self.purge_bars]
                
                val_indices = np.where(val_mask)[0]
                if len(val_indices) > self.embargo_bars:
                    val_indices = val_indices[self.embargo_bars:]
            
            n_splits += 1
            yield train_indices, val_indices
            
        logger.info(f"Generated {n_splits} walk-forward splits")

    def run(self, df: pd.DataFrame, model_trainer_func: Callable) -> List[Dict]:
        """
        Run the walk-forward evaluation.
        
        model_trainer_func should take (X_train, y_train, X_val, y_val) 
        and return a dict of metrics.
        """
        self.results = []
        
        splits = list(self.get_splits(df))
        if not splits:
            logger.warning("No walk-forward splits generated. Check data length.")
            return []
            
        for i, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"Walk-Forward Split {i+1}/{len(splits)}")
            
            # This is a generalized runner. The actual implementation requires 
            # the caller to pass a function that handles the specific model extraction
            try:
                metrics = model_trainer_func(df.iloc[train_idx], df.iloc[val_idx])
                metrics['split_id'] = i
                metrics['train_start'] = str(df.iloc[train_idx[0]]['timestamp'] if 'timestamp' in df.columns else df.index[train_idx[0]])
                metrics['val_end'] = str(df.iloc[val_idx[-1]]['timestamp'] if 'timestamp' in df.columns else df.index[val_idx[-1]])
                self.results.append(metrics)
            except Exception as e:
                logger.error(f"Error in split {i}: {e}")
                
        return self.results

    def aggregate_results(self, results: List[Dict] = None) -> Dict:
        """Calculate aggregate performance across all splits."""
        res = results or self.results
        if not res:
            return {'error': 'No results to aggregate'}
            
        agg = {}
        # Assuming metrics dictionary contains numeric values
        keys = [k for k in res[0].keys() if isinstance(res[0][k], (int, float))]
        
        for k in keys:
            vals = [r[k] for r in res if k in r and r[k] is not None]
            if vals:
                agg[f"avg_{k}"] = float(np.mean(vals))
                agg[f"std_{k}"] = float(np.std(vals))
                if 'sharpe' in k.lower():
                    # Calculate % of splits with Sharpe > 0
                    passing = sum(1 for v in vals if v > 0) / len(vals)
                    agg[f"pct_positive_{k}"] = passing
                
        return agg
