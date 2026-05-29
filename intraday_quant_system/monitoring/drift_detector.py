import pandas as pd
import numpy as np
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class FeatureDriftMonitor:
    def __init__(self, alert_threshold: float = 2.5, psi_threshold: float = 0.25):
        self.alert_threshold = alert_threshold
        self.psi_threshold = psi_threshold
        self.baseline_stats = {}

    def fit_baseline(self, X_train: pd.DataFrame) -> 'FeatureDriftMonitor':
        """Store training distributions and compute decile bin boundaries"""
        logger.info(f"Fitting baseline statistics for {X_train.shape[1]} features")
        for col in X_train.select_dtypes(include=[np.number]).columns:
            mean = X_train[col].mean()
            std = X_train[col].std()
            
            # Compute decile bin edges
            try:
                # Add tiny noise to avoid unique-bin errors on constant features
                col_data = X_train[col].dropna()
                if len(col_data) > 10:
                    bin_edges = np.percentile(col_data, np.linspace(0, 100, 11))
                    # Ensure bin edges are strictly increasing
                    bin_edges = np.unique(bin_edges)
                    if len(bin_edges) < 2:
                        bin_edges = None
                else:
                    bin_edges = None
            except Exception as e:
                logger.warning(f"Could not compute bin edges for {col}: {e}")
                bin_edges = None
                
            self.baseline_stats[col] = {
                'mean': mean,
                'std': std,
                'bin_edges': bin_edges
            }
        return self

    @staticmethod
    def _calculate_psi(expected_pct: np.ndarray, actual_pct: np.ndarray) -> float:
        """Helper to calculate PSI with epsilon smoothing"""
        # Epsilon smoothing to prevent div-by-zero or log of zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

    def detect(self, X_live: pd.DataFrame) -> dict:
        """
        Detect feature drift using:
          1. Z-score shift in means
          2. Population Stability Index (PSI)
        
        Returns:
            Dict containing {'z_scores': dict, 'psi_scores': dict, 'needs_retrain': bool}
        """
        z_scores = {}
        psi_scores = {}
        needs_retrain = False
        
        if X_live.empty or not self.baseline_stats:
            return {'z_scores': z_scores, 'psi_scores': psi_scores, 'needs_retrain': False}
            
        for col in X_live.select_dtypes(include=[np.number]).columns:
            if col in self.baseline_stats:
                baseline_mean = self.baseline_stats[col]['mean']
                baseline_std = self.baseline_stats[col]['std']
                bin_edges = self.baseline_stats[col]['bin_edges']
                
                live_mean = X_live[col].mean()
                live_std = X_live[col].std()
                
                # 1. Compute Z-score of the shift in means
                # Use standard error (σ/√n) for proper hypothesis testing of mean shift
                n_live = len(X_live[col].dropna())
                if baseline_std > 0 and n_live > 0:
                    standard_error = baseline_std / np.sqrt(n_live)
                    z_score = abs(live_mean - baseline_mean) / standard_error
                else:
                    z_score = 0.0
                z_scores[col] = z_score
                
                if z_score > self.alert_threshold:
                    logger.warning(f"Feature drift (Z-score) detected for {col}! Z-score: {z_score:.2f}")
                
                # 2. Compute Population Stability Index (PSI)
                if bin_edges is not None and len(X_live) >= 10:
                    try:
                        # expected bucket counts (uniform 10% per bin)
                        n_bins = len(bin_edges) - 1
                        expected_pct = np.ones(n_bins) / n_bins
                        
                        # actual bucket counts
                        actual_counts, _ = np.histogram(X_live[col].dropna(), bins=bin_edges)
                        actual_pct = actual_counts / len(X_live[col].dropna())
                        
                        psi = self._calculate_psi(expected_pct, actual_pct)
                        psi_scores[col] = psi
                        
                        if psi >= self.psi_threshold:
                            logger.warning(f"Feature drift (PSI) detected for {col}! PSI: {psi:.4f} >= {self.psi_threshold}")
                            needs_retrain = True
                    except Exception as e:
                        logger.error(f"Error computing PSI for {col}: {e}")
                        psi_scores[col] = 0.0
                else:
                    psi_scores[col] = 0.0
                    
        return {
            'z_scores': z_scores,
            'psi_scores': psi_scores,
            'needs_retrain': needs_retrain
        }

class SignalDecayMonitor:
    def __init__(self, decay_alert_threshold: float = 0.52):
        self.decay_alert_threshold = decay_alert_threshold
        self.latest_auc = 0.0  # Tracks the most recently computed rolling AUC

    def monitor(self, predictions: np.ndarray, outcomes: np.ndarray, window: int = 63) -> float:
        """rolling AUC"""
        from sklearn.metrics import roc_auc_score
        
        if len(predictions) < 10 or len(np.unique(outcomes)) < 2:
            return 0.5 # Default AUC
            
        # Keep only recent window if applicable
        if len(predictions) > window:
            predictions = predictions[-window:]
            outcomes = outcomes[-window:]
            
        if len(np.unique(outcomes)) < 2:
            return 0.5
            
        try:
            auc = roc_auc_score(outcomes, predictions)
            self.latest_auc = auc
            if auc < self.decay_alert_threshold:
                logger.warning(f"Signal decay detected! AUC dropped to {auc:.3f}")
            return auc
        except Exception as e:
            logger.error(f"Error calculating AUC: {e}")
            return 0.5
