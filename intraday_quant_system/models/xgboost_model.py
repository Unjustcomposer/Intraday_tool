import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import logging
import os
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)

class XGBoostAlphaModel:
    """
    XGBoost Alpha Model for Intraday Trading
    Provides predict_proba(), predict(), and uses time-based split + Brier score calibration.
    """
    def __init__(self, config: dict = None):
        config = config or {}
        
        # Hyperparameters
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.05)
        self.n_estimators = config.get('n_estimators', 400)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        
        self.model = None
        self.calibrator = None
        self.feature_names = None
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series, val_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the XGBoost model using a time-based train/val split.
        Applies Isotonic Regression for probability calibration.
        """
        logger.info(f"Training XGBoost model on {len(X)} samples")
        self.feature_names = list(X.columns)
        
        # Time-based split (assuming X is ordered chronologically)
        split_idx = int(len(X) * (1 - val_size))
        
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # XGBoost handles NaNs internally, but warn if there are too many
        missing_pct = X_train.isna().mean().max()
        if missing_pct > 0.3:
            logger.warning(f"High missing values in training data: {missing_pct:.1%}")
            
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            eval_metric='auc',
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Calibrate probabilities using Isotonic Regression
        raw_probs = self.model.predict_proba(X_val)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(raw_probs, y_val)
        
        self.is_trained = True
        
        # Collect metrics
        metrics = {
            'best_iteration': self.model.best_iteration,
            'train_auc': float(self.model.evals_result()['validation_0']['auc'][self.model.best_iteration]),
            'val_auc': float(self.model.evals_result()['validation_1']['auc'][self.model.best_iteration])
        }
        
        logger.info(f"XGBoost training complete. Val AUC: {metrics['val_auc']:.4f}")
        return metrics
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict calibrated probabilities. Returns array of shape (n_samples,).
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
            
        # Ensure column order matches training
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
            
        raw_probs = self.model.predict_proba(X)[:, 1]
        calibrated_probs = self.calibrator.predict(raw_probs)
        return calibrated_probs
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels based on threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Return DataFrame of feature importances"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
            
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        return df
        
    def save(self, filepath: str):
        """Save model securely using joblib"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        state = {
            'model': self.model,
            'calibrator': self.calibrator,
            'feature_names': self.feature_names,
            'config': {
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators
            },
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(state, filepath)
        logger.info(f"Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> "XGBoostAlphaModel":
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        state = joblib.load(filepath)
        
        instance = cls(state.get('config', {}))
        instance.model = state['model']
        instance.calibrator = state['calibrator']
        instance.feature_names = state['feature_names']
        instance.is_trained = True
        
        logger.info(f"Loaded XGBoost model (trained at {state.get('timestamp', 'unknown')})")
        return instance

    @staticmethod
    def make_labels(df: pd.DataFrame, atr_mult_up: float = 2.0, atr_mult_down: float = 1.0, horizon_minutes: int = 45) -> pd.Series:
        """
        Creates triple-barrier labels. 1 for hit upper barrier, 0 otherwise.
        Reusing logic pattern from LGBM model.
        """
        if not all(col in df.columns for col in ['close', 'atr']):
            raise ValueError("DataFrame must contain 'close' and 'atr' columns to make labels.")
            
        # NOTE: horizon_minutes is a misnomer — it represents horizon_bars
        # (number of bars to look forward), matching LGBM's make_labels behavior.
        bars_forward = horizon_minutes
        
        labels = pd.Series(np.nan, index=df.index)
        closes = df['close'].values
        atrs = df['atr'].values
        
        for i in range(len(df) - bars_forward):
            current_close = closes[i]
            current_atr = atrs[i]
            
            upper_barrier = current_close + (current_atr * atr_mult_up)
            lower_barrier = current_close - (current_atr * atr_mult_down)
            
            path = closes[i+1 : i+bars_forward+1]
            
            hit_upper = False
            hit_lower = False
            
            for price in path:
                if price >= upper_barrier:
                    hit_upper = True
                    break
                elif price <= lower_barrier:
                    hit_lower = True
                    break
                    
            if hit_upper and not hit_lower:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = 0
                
        return labels
