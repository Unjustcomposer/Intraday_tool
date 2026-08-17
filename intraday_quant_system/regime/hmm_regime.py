import logging
import os
import joblib

import numpy as np
import pandas as pd
from intraday_quant_system.features.volatility_features import atr

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None
    logging.warning("hmmlearn not installed. Please install it to use HMM regime detection.")

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detects market regimes using a Hidden Markov Model (HMM).

    Outputs regime labels matching the ensemble's expected vocabulary:
      - 'quiet'     : Low volatility, small returns
      - 'trending'  : Moderate volatility, directional movement
      - 'volatile'  : High volatility
      - 'crisis'    : Extreme volatility (>2x median rolling std)
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.hmm = None
        self.is_fitted = False
        self.vol_percentiles = None  # For crisis detection

    def fit(self, df: pd.DataFrame):
        """Fit the HMM model on volatility and returns."""
        if GaussianHMM is None:
            logger.warning("hmmlearn not available. Falling back to simple regime detection.")
            return

        if len(df) < 50:
            logger.warning("Insufficient data to fit HMM regime model.")
            return

        features = self._extract_features(df)
        features = features.dropna()
        if len(features) < 20:
            return

        logger.info(
            f"Fitting GaussianHMM ({self.n_regimes} components) on {len(features)} samples"
        )
        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=42,
            n_iter=100,
        )
        self.hmm.fit(features.values)
        self.is_fitted = True

        # Store volatility percentiles for crisis detection
        vol_col = features["atr"]  # ATR column
        rv5_col = features["realized_vol_5bar"].dropna()
        self.vol_percentiles = {
            "p50": float(vol_col.median()),
            "p90": float(vol_col.quantile(0.90)),
            "p95": float(vol_col.quantile(0.95)),
            "rv5_p90": float(rv5_col.quantile(0.90)) if len(rv5_col) > 0 else 0.0,
        }

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regimes for the entire dataframe."""
        if not self.is_fitted or self.hmm is None:
            return self._fallback_regime(df)

        features = self._extract_features(df)
        preds = pd.Series("unknown", index=df.index, dtype="object")

        valid_idx = features.dropna().index
        if len(valid_idx) == 0:
            return self._fallback_regime(df)

        state_preds = self.hmm.predict(features.loc[valid_idx].values)

        # Classify HMM states by volatility level
        means = self.hmm.means_
        vol_means = means[:, 1]  # ATR column
        sorted_states = np.argsort(vol_means)  # Low vol → High vol

        # Map HMM states to regime labels
        state_to_label = {}
        if self.n_regimes >= 3:
            state_to_label[sorted_states[0]] = "quiet"
            state_to_label[sorted_states[1]] = "trending"
            state_to_label[sorted_states[2]] = "volatile"
        elif self.n_regimes == 2:
            state_to_label[sorted_states[0]] = "quiet"
            state_to_label[sorted_states[1]] = "volatile"
        else:
            state_to_label[sorted_states[0]] = "unknown"

        # Apply base HMM labels
        state_labels = [state_to_label.get(s, "unknown") for s in state_preds]
        preds.loc[valid_idx] = state_labels

        # Override to 'crisis' if current volatility exceeds 95th percentile of training data
        if self.vol_percentiles:
            atr_vals = features.loc[valid_idx]["atr"]
            crisis_mask = atr_vals > self.vol_percentiles["p95"]

            # Fix #14: Instantaneous 5-bar realized volatility override
            if "rv5_p90" in self.vol_percentiles:
                rv5_vals = features.loc[valid_idx]["realized_vol_5bar"]
                instant_crisis_mask = rv5_vals > self.vol_percentiles["rv5_p90"]
                crisis_mask = crisis_mask | instant_crisis_mask

            crisis_indices = valid_idx[crisis_mask.values]
            if len(crisis_indices) > 0:
                preds.loc[crisis_indices] = "crisis"

        return preds.fillna("unknown")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features["log_ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)
        features["atr"] = atr(df)
        # Fix #14: Instantaneous 5-bar realized volatility to bypass HMM lag
        features["realized_vol_5bar"] = features["log_ret"].rolling(5).std().fillna(0)
        return features

    def _fallback_regime(self, df: pd.DataFrame) -> pd.Series:
        """Fallback regime based on rolling standard deviation of returns."""
        rets = np.log(df["close"] / df["close"].shift(1))
        rolling_std = rets.rolling(20).std()
        median_std = rolling_std.median()

        regimes = pd.Series("unknown", index=df.index, dtype="object")

        if median_std is None or np.isnan(median_std) or median_std == 0:
            return regimes

        regimes[rolling_std <= median_std * 0.7] = "quiet"
        regimes[
            (rolling_std > median_std * 0.7) & (rolling_std <= median_std * 1.3)
        ] = "trending"
        regimes[
            (rolling_std > median_std * 1.3) & (rolling_std <= median_std * 2.0)
        ] = "volatile"
        regimes[rolling_std > median_std * 2.0] = "crisis"
        return regimes

    def save(self, path: str):
        """Save fitted HMM model and parameters to disk."""
        if not self.is_fitted:
            logger.warning("RegimeDetector not fitted, nothing to save.")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_dict = {
            "hmm": self.hmm,
            "n_regimes": self.n_regimes,
            "is_fitted": self.is_fitted,
            "vol_percentiles": self.vol_percentiles,
        }
        
        joblib.dump(save_dict, path)
        logger.info(f"RegimeDetector saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        """Load fitted HMM model and parameters from disk."""
        if not os.path.exists(path):
            logger.error(f"RegimeDetector model not found at {path}")
            return cls()

        try:
            save_dict = joblib.load(path)
            
            detector = cls(n_regimes=save_dict.get("n_regimes", 3))
            detector.hmm = save_dict.get("hmm", save_dict.get("gmm")) # Support loading old GMM as fallback? Actually it would crash predict.
            detector.is_fitted = save_dict["is_fitted"]
            detector.vol_percentiles = save_dict.get("vol_percentiles")
            
            logger.info(f"RegimeDetector loaded from {path}")
            return detector
        except Exception as e:
            logger.error(f"Failed to load RegimeDetector: {e}")
            return cls()
