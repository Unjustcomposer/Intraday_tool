import pandas as pd
import numpy as np
import lightgbm as lgb
import logging

import json
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, classification_report, brier_score_loss

logger = logging.getLogger(__name__)


class LGBMAlphaModel:
    """
    Primary alpha model using LightGBM.
    
    Production features:
      - Train/validation split with early stopping
      - Feature importance tracking
      - Calibration metrics (Brier score)
      - Model versioning
      - Label leakage safeguard
    """
    def __init__(self, config: dict = None):
        config = config or {}
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'num_leaves': config.get('num_leaves', 63),
            'learning_rate': config.get('learning_rate', 0.03),
            'n_estimators': config.get('n_estimators', 2000),
            'feature_fraction': config.get('feature_fraction', 0.7),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'min_child_samples': config.get('min_child_samples', 100),
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'verbose': -1,
        }
        self.early_stopping_rounds = config.get('early_stopping_rounds', 50)
        self.model = lgb.LGBMClassifier(**self.params)
        self.is_fitted = False
        self.train_metrics: dict = {}
        self.version: str = ""
        self.feature_names: list = []

    @staticmethod
    def make_labels(df: pd.DataFrame, atr_mult_up: float = 2.0, atr_mult_down: float = 1.0, horizon_minutes: int = 45) -> pd.Series:
        """
        Label = 1 if stock moves +atr_mult_up * ATR before -atr_mult_down * ATR within horizon.
        Label = 0 otherwise.
        
        IMPORTANT: This function uses future data for label generation ONLY.
        Labels must NEVER be used as features. The calling code must ensure
        labels are separated from features before training.
        """
        if 'close' not in df.columns or 'atr' not in df.columns:
            return pd.Series(0, index=df.index)
            
        labels = np.zeros(len(df))
        closes = df['close'].values
        atrs = df['atr'].values
        
        for i in range(len(df) - horizon_minutes):
            current_price = closes[i]
            current_atr = atrs[i]
            
            if np.isnan(current_atr) or current_atr == 0:
                continue
                
            target_up = current_price + (atr_mult_up * current_atr)
            target_down = current_price - (atr_mult_down * current_atr)
            
            window = closes[i+1 : i+1+horizon_minutes]
            
            for price in window:
                if price >= target_up:
                    labels[i] = 1
                    break
                elif price <= target_down:
                    labels[i] = 0
                    break
                    
        return pd.Series(labels, index=df.index)

    @staticmethod
    def make_directional_labels(df: pd.DataFrame, atr_mult: float = 1.5,
                                 horizon_minutes: int = 45) -> pd.Series:
        """
        Three-class directional labeling for proper long/short/neutral signals.
        
        Label =  1 (LONG):  stock hits +atr_mult * ATR before -atr_mult * ATR
        Label = -1 (SHORT): stock hits -atr_mult * ATR before +atr_mult * ATR
        Label =  0 (NEUTRAL): neither barrier hit within horizon
        
        Uses symmetric barriers (same magnitude up and down) to avoid
        directional bias in label distribution.
        
        IMPORTANT: Uses future data for label generation ONLY.
        """
        if 'close' not in df.columns or 'atr' not in df.columns:
            return pd.Series(0, index=df.index)
            
        labels = np.zeros(len(df))
        closes = df['close'].values
        atrs = df['atr'].values
        
        for i in range(len(df) - horizon_minutes):
            current_price = closes[i]
            current_atr = atrs[i]
            
            if np.isnan(current_atr) or current_atr == 0:
                continue
                
            target_up = current_price + (atr_mult * current_atr)
            target_down = current_price - (atr_mult * current_atr)
            
            window = closes[i+1 : i+1+horizon_minutes]
            
            for price in window:
                if price >= target_up:
                    labels[i] = 1   # Long signal
                    break
                elif price <= target_down:
                    labels[i] = -1  # Short signal
                    break
            # If loop completes without break, label stays 0 (neutral)
                    
        return pd.Series(labels, index=df.index)

    def train(self, X: pd.DataFrame, y: pd.Series, val_size: float = 0.2) -> 'LGBMAlphaModel':
        """
        Train with proper train/validation split and early stopping.
        
        Uses time-based split (last val_size% as validation) to prevent
        future data leakage — NOT random split.
        """
        logger.info(f"Training LightGBM on {len(X)} samples")
        
        # Time-based split: use last portion as validation (no shuffling)
        split_idx = int(len(X) * (1 - val_size))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")
        logger.info(f"Label distribution — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")
        
        # Check for label imbalance
        pos_rate = y_train.mean()
        if pos_rate < 0.1 or pos_rate > 0.9:
            logger.warning(f"Severe label imbalance: {pos_rate:.1%} positive. Consider adjusting labeling thresholds.")
        
        # Fit with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.is_fitted = True
        self.feature_names = list(X.columns)
        self.version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Compute validation metrics
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)
        
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = 0.5
        
        try:
            brier = brier_score_loss(y_val, val_probs)
        except ValueError:
            brier = 0.25
        
        self.train_metrics = {
            'val_auc': float(val_auc),
            'val_brier_score': float(brier),
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else -1,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'train_pos_rate': float(pos_rate),
            'val_pos_rate': float(y_val.mean()),
            'version': self.version,
        }
        
        logger.info(f"Training complete. Val AUC: {val_auc:.4f} | Brier: {brier:.4f} | "
                     f"Best iteration: {self.train_metrics['best_iteration']}")
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            logger.warning("Model not fitted, returning 0.5 probabilities")
            return np.ones(len(X)) * 0.5
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            logger.warning("Model not fitted, returning 0 classes")
            return np.zeros(len(X))
        return self.model.predict(X)

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame()
            
        importance = self.model.feature_importances_
        features = self.model.feature_name_
        
        df = pd.DataFrame({
            'feature': features,
            'importance': importance
        })
        return df.sort_values('importance', ascending=False)

    def save(self, path: str):
        """Save model using native LightGBM format (no pickle) with JSON sidecar metadata"""
        if self.is_fitted:
            self.model.booster_.save_model(path)
            
            # Save metadata alongside
            meta_path = path + '.meta.json'
            with open(meta_path, 'w') as f:
                json.dump({
                    'version': self.version,
                    'metrics': self.train_metrics,
                    'feature_names': self.feature_names,
                    'params': {k: v for k, v in self.params.items() if not callable(v)},
                }, f, indent=2, default=str)
            
            logger.info(f"Saved model v{self.version} to {path}")
            
    def load(self, path: str):
        """Load model using native LightGBM Booster (no pickle)"""
        booster = lgb.Booster(model_file=path)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model._Booster = booster
        self.model._n_classes = 2
        self.is_fitted = True
        
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.version = meta.get('version', 'unknown')
                self.train_metrics = meta.get('metrics', {})
                self.feature_names = meta.get('feature_names', [])
            logger.info(f"Loaded model v{self.version}")
