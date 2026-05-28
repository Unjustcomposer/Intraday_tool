import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import logging
import json
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class MetaLabeler:
    """
    MetaLabeler using CatBoostClassifier.
    
    Purpose: Given a primary model's signal, predict whether the trade will
    actually be profitable. This acts as a confidence filter.
    
    Production features:
      - Early stopping with eval set
      - Proper train/val split
      - Model versioning
    """
    def __init__(self, config: dict = None):
        config = config or {}
        self.model = CatBoostClassifier(
            iterations=config.get('iterations', 800),
            learning_rate=config.get('learning_rate', 0.05),
            depth=config.get('depth', 6),
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=False,
            random_seed=42,
            early_stopping_rounds=config.get('early_stopping_rounds', 50),
        )
        self.is_fitted = False
        self.version: str = ""
        self.train_metrics: dict = {}

    def train(self, X_primary_preds: np.ndarray, X_features: pd.DataFrame, y_trade_outcome: pd.Series, val_size: float = 0.2):
        """
        Train meta-labeler with proper time-based validation split.
        
        Input: primary model predictions + all features
        Label: 1 if the primary model's trade was actually profitable
        """
        logger.info(f"Training MetaLabeler on {len(X_features)} samples")
        
        # Combine primary predictions with features
        X_combined = X_features.copy()
        X_combined['primary_pred'] = X_primary_preds
        
        # Time-based split
        split_idx = int(len(X_combined) * (1 - val_size))
        X_train, X_val = X_combined.iloc[:split_idx], X_combined.iloc[split_idx:]
        y_train, y_val = y_trade_outcome.iloc[:split_idx], y_trade_outcome.iloc[split_idx:]
        
        # Identify categorical features for CatBoost
        cat_features = []
        for col in X_combined.columns:
            if X_combined[col].dtype == 'object' or X_combined[col].dtype.name == 'category':
                cat_features.append(col)
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features if cat_features else None)
        val_pool = Pool(X_val, y_val, cat_features=cat_features if cat_features else None)
        
        self.model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        self.is_fitted = True
        self.version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Compute metrics
        val_probs = self.model.predict_proba(X_val)[:, 1]
        try:
            val_auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auc = 0.5
        
        self.train_metrics = {
            'val_auc': float(val_auc),
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else -1,
            'n_train': len(X_train),
            'n_val': len(X_val),
            'version': self.version,
        }
        
        logger.info(f"MetaLabeler training complete. Val AUC: {val_auc:.4f}")

    def predict_proba(self, X: pd.DataFrame, primary_preds: np.ndarray = None) -> np.ndarray:
        """Confidence that trade is worth taking"""
        if not self.is_fitted:
            logger.warning("MetaLabeler not fitted, returning 0.5 probabilities")
            return np.ones(len(X)) * 0.5
            
        X_combined = X.copy()
        if primary_preds is not None:
            X_combined['primary_pred'] = primary_preds
            
        return self.model.predict_proba(X_combined)[:, 1]

    def save(self, path: str):
        if self.is_fitted:
            self.model.save_model(path)
            meta_path = path + '.meta.json'
            with open(meta_path, 'w') as f:
                json.dump({'version': self.version, 'metrics': self.train_metrics}, f, indent=2, default=str)
            
    def load(self, path: str):
        self.model.load_model(path)
        self.is_fitted = True
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.version = meta.get('version', 'unknown')
                self.train_metrics = meta.get('metrics', {})
