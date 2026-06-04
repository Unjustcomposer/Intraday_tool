import logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from features.volatility_features import atr

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regimes using a Gaussian Mixture Model (GMM).
    
    Outputs regime labels matching the ensemble's expected vocabulary:
      - 'quiet'     : Low volatility, small returns
      - 'trending'  : Moderate volatility, directional movement
      - 'volatile'  : High volatility
      - 'crisis'    : Extreme volatility (>2x median rolling std)
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.gmm = None
        self.is_fitted = False
        self.vol_percentiles = None  # For crisis detection
        
    def fit(self, df: pd.DataFrame):
        """Fit the GMM model on volatility and returns."""
        if len(df) < 50:
            logger.warning("Insufficient data to fit GMM regime model.")
            return
            
        features = self._extract_features(df)
        features = features.dropna()
        if len(features) < 20:
            return
            
        logger.info(f"Fitting GaussianMixture ({self.n_regimes} components) on {len(features)} samples")
        self.gmm = GaussianMixture(
            n_components=self.n_regimes, 
            covariance_type='full', 
            random_state=42,
            n_init=3
        )
        self.gmm.fit(features.values)
        self.is_fitted = True
        
        # Store volatility percentiles for crisis detection
        vol_col = features.iloc[:, 1]  # ATR column
        self.vol_percentiles = {
            'p50': float(vol_col.median()),
            'p90': float(vol_col.quantile(0.90)),
            'p95': float(vol_col.quantile(0.95)),
        }
        
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regimes for the entire dataframe."""
        if not self.is_fitted:
            return self._fallback_regime(df)
            
        features = self._extract_features(df)
        preds = pd.Series('unknown', index=df.index, dtype='object')
        
        valid_idx = features.dropna().index
        if len(valid_idx) == 0:
            return self._fallback_regime(df)
            
        state_preds = self.gmm.predict(features.loc[valid_idx].values)
        
        # Classify GMM states by volatility level
        means = self.gmm.means_
        vol_means = means[:, 1]  # ATR column
        sorted_states = np.argsort(vol_means)  # Low vol → High vol
        
        # Map GMM states to regime labels
        state_to_label = {}
        if self.n_regimes >= 3:
            state_to_label[sorted_states[0]] = 'quiet'
            state_to_label[sorted_states[1]] = 'trending'
            state_to_label[sorted_states[2]] = 'volatile'
        elif self.n_regimes == 2:
            state_to_label[sorted_states[0]] = 'quiet'
            state_to_label[sorted_states[1]] = 'volatile'
        else:
            state_to_label[sorted_states[0]] = 'unknown'
            
        # Apply base GMM labels
        state_labels = [state_to_label.get(s, 'unknown') for s in state_preds]
        preds.loc[valid_idx] = state_labels
        
        # Override to 'crisis' if current volatility exceeds 95th percentile of training data
        if self.vol_percentiles:
            atr_vals = features.loc[valid_idx].iloc[:, 1]
            crisis_mask = atr_vals > self.vol_percentiles['p95']
            crisis_indices = valid_idx[crisis_mask.values]
            if len(crisis_indices) > 0:
                preds.loc[crisis_indices] = 'crisis'
        
        return preds.fillna('unknown')
        
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        features['atr'] = atr(df)
        return features

    def _fallback_regime(self, df: pd.DataFrame) -> pd.Series:
        """Fallback regime based on rolling standard deviation of returns."""
        rets = np.log(df['close'] / df['close'].shift(1))
        rolling_std = rets.rolling(20).std()
        median_std = rolling_std.median()
        
        regimes = pd.Series('unknown', index=df.index, dtype='object')
        
        if median_std is None or np.isnan(median_std) or median_std == 0:
            return regimes
        
        regimes[rolling_std <= median_std * 0.7] = 'quiet'
        regimes[(rolling_std > median_std * 0.7) & (rolling_std <= median_std * 1.3)] = 'trending'
        regimes[(rolling_std > median_std * 1.3) & (rolling_std <= median_std * 2.0)] = 'volatile'
        regimes[rolling_std > median_std * 2.0] = 'crisis'
        return regimes
