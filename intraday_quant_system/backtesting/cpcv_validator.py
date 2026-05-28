import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)

class CPCVValidator:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Implements the Lopez de Prado method for backtesting time-series models
    by creating multiple combinations of train/test paths and purging
    overlap to prevent leakage.
    """
    def __init__(self, n_groups: int = 6, k_test_groups: int = 2):
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        
    def get_splits(self, df: pd.DataFrame):
        """
        Generates indices for train/test splits.
        """
        n_obs = len(df)
        indices = np.arange(n_obs)
        group_size = n_obs // self.n_groups
        
        # Define group boundaries
        groups = [indices[i*group_size : (i+1)*group_size] for i in range(self.n_groups)]
        
        # All possible combinations of k test groups from n total groups
        test_combos = list(combinations(range(self.n_groups), self.k_test_groups))
        
        for test_indices in test_combos:
            test_mask = np.isin(range(self.n_groups), test_indices)
            
            # Test set
            test_indices_flat = np.concatenate([groups[i] for i in test_indices])
            
            # Train set (remaining groups)
            train_indices = [i for i in range(self.n_groups) if i not in test_indices]
            train_indices_flat = np.concatenate([groups[i] for i in train_indices])
            
            # Purging: Remove observations from train set that are too close to test set
            # (To prevent leakage via serial correlation in features)
            purged_train = self._purge(train_indices_flat, test_indices_flat, n_obs)
            
            yield purged_train, test_indices_flat

    def _purge(self, train_indices, test_indices, n_obs, buffer_size=100):
        """
        Removes indices from train set that are within buffer_size of ANY test observation.
        
        For CPCV with non-contiguous test groups (e.g., groups 1 and 4), we must
        purge around each test group boundary independently, not just the global
        min/max which would incorrectly remove valid training data between groups.
        """
        # Build a set of all indices that are "too close" to any test index
        # For efficiency, check against test group boundaries rather than every index
        test_set = set(test_indices)
        test_sorted = np.sort(test_indices)
        
        # Find contiguous test group boundaries
        boundaries = []
        group_start = test_sorted[0]
        for i in range(1, len(test_sorted)):
            if test_sorted[i] - test_sorted[i-1] > 1:
                # Gap detected — end of one test group, start of another
                boundaries.append((group_start, test_sorted[i-1]))
                group_start = test_sorted[i]
        boundaries.append((group_start, test_sorted[-1]))
        
        # Purge train indices within buffer of any test group boundary
        mask = np.ones(len(train_indices), dtype=bool)
        for i, idx in enumerate(train_indices):
            for group_min, group_max in boundaries:
                if (idx >= group_min - buffer_size) and (idx <= group_max + buffer_size):
                    mask[i] = False
                    break
                    
        return train_indices[mask]

    def run_backtest(self, model, df: pd.DataFrame, features: list, target: str):
        """
        Runs the model across all CPCV paths and returns combined metrics.
        """
        all_results = []
        
        for train_idx, test_idx in self.get_splits(df):
            if len(train_idx) < 100: # Ensure minimum training data
                logger.warning(f"Skipping CPCV split: Insufficient training data ({len(train_idx)} samples)")
                continue
                
            X_train, y_train = df.iloc[train_idx][features], df.iloc[train_idx][target]
            X_test, y_test = df.iloc[test_idx][features], df.iloc[test_idx][target]
            
            model.train(X_train, y_train)
            probs = model.predict_proba(X_test)
            
            # Store results for this path
            all_results.append({
                'y_true': y_test.values,
                'y_prob': probs # LGBMAlphaModel returns 1D array of probs
            })
            
        return all_results
