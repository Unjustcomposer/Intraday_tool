import pandas as pd
import numpy as np
import logging
import pickle
from collections import Counter

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logging.warning("hmmlearn not found. RegimeDetector will use simple volatility fallback.")

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Market regime detection using GaussianHMM.
    Input features: vix, breadth, atr_percentile, correlation_dispersion, index_slope, market_volume
    
    Production features:
      - Analytical regime mapping based on cluster statistics
      - Stability controls (smoothing) to prevent intraday whipsaw
    """
    def __init__(self, n_components: int = 3, random_state: int = 42):
        self.n_components = n_components
        if HMM_AVAILABLE:
            self.model = GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=random_state)
        else:
            self.model = None
        self.is_fitted = False
        self.state_to_regime_map = {}
        
        # Exposure map based on regime (Kelly sizing fraction)
        self.exposure_map = {
            'quiet': 1.00,           # Max exposure in quiet/predictable regimes
            'bull_volatile': 0.50,   # Half exposure in volatile uptrends
            'bear_volatile': 0.25,   # Quarter exposure in volatile downtrends
            'unknown': 0.25
        }

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and normalize required features for HMM"""
        required_features = ['vix', 'breadth', 'atr', 'market_volume']
        X_df = pd.DataFrame(index=df.index)
        
        for f in required_features:
            if f in df.columns:
                X_df[f] = df[f]
            else:
                X_df[f] = 0.0
                
        # Z-score normalization for HMM stability
        for col in X_df.columns:
            std = X_df[col].std()
            if std > 0:
                X_df[col] = (X_df[col] - X_df[col].mean()) / std
                
        X_df = X_df.ffill().fillna(0)
        return X_df.values

    def _map_states_analytically(self, X: np.ndarray, states: np.ndarray):
        """
        Dynamically map HMM hidden states to regime names based on cluster statistics.
        Assuming features are [vix, breadth, atr, market_volume]
        """
        df = pd.DataFrame(X, columns=['vix', 'breadth', 'atr', 'volume'])
        df['state'] = states
        
        stats = df.groupby('state').mean()
        
        # Define logic to assign regimes based on relative characteristics (3 states)
        for state in range(self.n_components):
            if state not in stats.index:
                self.state_to_regime_map[state] = 'unknown'
                continue
                
            volatility = stats.loc[state, 'atr'] + stats.loc[state, 'vix']
            trend = stats.loc[state, 'breadth']
            
            # Simple 3-state heuristic mapping
            if volatility < 0: 
                # Lower than average volatility -> Quiet
                self.state_to_regime_map[state] = 'quiet'
            else:
                # Higher than average volatility -> Volatile
                if trend > 0:
                    self.state_to_regime_map[state] = 'bull_volatile'
                else:
                    self.state_to_regime_map[state] = 'bear_volatile'
                
        logger.info(f"HMM Regime Mapping: {self.state_to_regime_map}")

    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """Fit the HMM model on historical data"""
        logger.info(f"Fitting GaussianHMM on {len(df)} samples")
        X = self._prepare_features(df)
        self.model.fit(X)
        self.is_fitted = True
        
        # Determine analytical mapping
        states = self.model.predict(X)
        self._map_states_analytically(X, states)
        
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regime labels for new data with smoothing to prevent whipsaw"""
        if not HMM_AVAILABLE:
            return self._predict_fallback(df)
            
        if not self.is_fitted:
            logger.warning("Model not fitted. Returning default regime.")
            return pd.Series('unknown', index=df.index)
            
        X = self._prepare_features(df)
        try:
            states = self.model.predict(X)
            # Map state integers to string labels
            labels = [self.state_to_regime_map.get(state, 'unknown') for state in states]
            
            # Apply mode filter (smoothing) to prevent rapid intraday whipsawing
            # Takes the most common regime over the last 6 bars (30 mins)
            smoothed = self._smooth_regime_labels(labels, window=6)
            
            return pd.Series(smoothed, index=df.index)
            
        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return pd.Series('unknown', index=df.index)

    @staticmethod
    def _smooth_regime_labels(labels: list, window: int = 6) -> list:
        """
        Mode filter for string labels. Pandas rolling cannot handle strings,
        so we implement a manual sliding window with Counter.
        """
        smoothed = []
        for i in range(len(labels)):
            start = max(0, i - window + 1)
            window_labels = labels[start:i + 1]
            counts = Counter(window_labels)
            # Most common label in the window
            smoothed.append(counts.most_common(1)[0][0])
        return smoothed

    def _predict_fallback(self, df: pd.DataFrame) -> pd.Series:
        """
        Simple volatility-based regime detection as fallback.
        """
        if 'close' not in df.columns:
            return pd.Series('unknown', index=df.index)
            
        # Calculate annualized volatility of returns
        vol = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252 * 75)
        
        regime = pd.Series('quiet', index=df.index)
        regime[vol > 0.25] = 'bull_volatile' # High vol threshold
        return regime
            
    def get_exposure(self, regime: str) -> float:
        """Get the risk exposure multiplier for a given regime"""
        return self.exposure_map.get(regime, 0.5)

    def save(self, path: str):
        if self.is_fitted:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'map': self.state_to_regime_map
                }, f)
            
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.state_to_regime_map = data['map']
        self.is_fitted = True
