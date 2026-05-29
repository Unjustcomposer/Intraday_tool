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
            depth=min(max(config.get('depth', 5), 4), 6),
            l2_leaf_reg=config.get('l2_leaf_reg', 10.0),
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=False,
            random_seed=42,
            early_stopping_rounds=config.get('early_stopping_rounds', 50),
        )
        self.is_fitted = False
        self.version: str = ""
        self.train_metrics: dict = {}
        self.conformal_threshold: float = 0.50

    def train(self, X_primary_preds: np.ndarray, X_features: pd.DataFrame, y_trade_outcome: pd.Series, val_size: float = 0.2):
        """
        Train meta-labeler with strict Purged Walk-Forward Cross Validation.
        
        Input: primary model predictions + all features
        Label: 1 if the primary model's trade was actually profitable
        """
        logger.info(f"Training MetaLabeler on {len(X_features)} samples using Purged Walk-Forward CV")
        
        # Combine primary predictions with features
        X_combined = X_features.copy()
        X_combined['primary_pred'] = X_primary_preds
        
        # Identify categorical features for CatBoost
        cat_features = []
        for col in X_combined.columns:
            if X_combined[col].dtype == 'object' or X_combined[col].dtype.name == 'category':
                cat_features.append(col)
        
        total_len = len(X_combined)
        purge_bars = 10
        embargo_bars = 10
        
        # Define expanding window splits dynamically based on dataset size
        if total_len >= 300:
            splits = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        elif total_len >= 150:
            splits = [(0.5, 0.75), (0.75, 1.0)]
        else:
            splits = [(1 - val_size, 1.0)]
            
        best_iterations = []
        val_aucs = []
        
        for fold, (train_end_pct, val_end_pct) in enumerate(splits):
            train_idx_end = int(total_len * train_end_pct)
            val_idx_end = int(total_len * val_end_pct)
            
            # Apply purging (drop last N bars of train to prevent forward label leakage)
            train_end_purged = max(0, train_idx_end - purge_bars)
            
            # Apply embargo (drop first N bars of validation to prevent backward leakage)
            val_start_embargoed = min(total_len, train_idx_end + embargo_bars)
            
            if train_end_purged < 40 or (val_idx_end - val_start_embargoed) < 10:
                logger.warning(f"Fold {fold+1} skipped: insufficient samples (train: {train_end_purged}, val: {val_idx_end - val_start_embargoed})")
                continue
                
            X_tr = X_combined.iloc[:train_end_purged]
            y_tr = y_trade_outcome.iloc[:train_end_purged]
            X_va = X_combined.iloc[val_start_embargoed:val_idx_end]
            y_va = y_trade_outcome.iloc[val_start_embargoed:val_idx_end]
            
            # Create clone of the classifier configuration
            fold_model = CatBoostClassifier(
                iterations=self.model.get_params().get('iterations', 800),
                learning_rate=self.model.get_params().get('learning_rate', 0.05),
                depth=self.model.get_params().get('depth', 5),
                l2_leaf_reg=self.model.get_params().get('l2_leaf_reg', 10.0),
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                random_seed=42 + fold,
                early_stopping_rounds=self.model.get_params().get('early_stopping_rounds', 50),
            )
            
            tr_pool = Pool(X_tr, y_tr, cat_features=cat_features if cat_features else None)
            va_pool = Pool(X_va, y_va, cat_features=cat_features if cat_features else None)
            
            try:
                fold_model.fit(tr_pool, eval_set=va_pool, use_best_model=True)
                best_iter = fold_model.best_iteration_ if hasattr(fold_model, 'best_iteration_') else fold_model.get_params().get('iterations', 800)
                best_iterations.append(best_iter)
                
                # Validation AUC
                va_probs = fold_model.predict_proba(X_va)[:, 1]
                try:
                    fold_auc = roc_auc_score(y_va, va_probs)
                except ValueError:
                    fold_auc = 0.5
                val_aucs.append(fold_auc)
                logger.info(f"Fold {fold+1} complete. Best Iteration: {best_iter}, Val AUC: {fold_auc:.4f}")
                last_successful_model = fold_model
                last_fold_train_size = train_end_purged
                
                # Conformal Prediction Calibration: Find tau such that precision >= 60%
                sorted_probs = np.sort(va_probs)
                best_tau = 0.50
                for tau in sorted_probs:
                    preds = (va_probs > tau).astype(int)
                    if preds.sum() == 0:
                        break
                    precision = y_va[preds == 1].mean()
                    if precision >= 0.60:
                        best_tau = tau
                        break
                last_fold_conformal = float(best_tau)
                
            except Exception as e:
                logger.error(f"Failed to train Fold {fold+1}: {e}")
                
        # Use last fold's model as production model to prevent label contamination.
        # Refitting on ALL data would let the model see labels it will predict on.
        if best_iterations:
            avg_val_auc = float(np.mean(val_aucs))
            avg_best_iter = max(10, int(np.mean(best_iterations)))
            logger.info(f"Walk-Forward CV complete. Avg Best Iteration: {avg_best_iter}, Avg Val AUC: {avg_val_auc:.4f}")
            
            # Use the last fold's already-fitted model directly (no refit)
            self.model = last_successful_model
            n_train = last_fold_train_size
            self.conformal_threshold = last_fold_conformal
            logger.info(f"Conformal Calibration Threshold: {self.conformal_threshold:.4f}")
        else:
            # Fallback: no folds succeeded, fit on first 80% only (never all data)
            logger.warning("No walk-forward folds succeeded. Falling back to 80/20 train split.")
            avg_val_auc = 0.5
            avg_best_iter = self.model.get_params().get('iterations', 800)
            
            fallback_split = int(total_len * 0.8)
            X_fb_train = X_combined.iloc[:fallback_split]
            y_fb_train = y_trade_outcome.iloc[:fallback_split]
            X_fb_val = X_combined.iloc[fallback_split:]
            y_fb_val = y_trade_outcome.iloc[fallback_split:]
            
            fb_pool = Pool(X_fb_train, y_fb_train, cat_features=cat_features if cat_features else None)
            fb_val_pool = Pool(X_fb_val, y_fb_val, cat_features=cat_features if cat_features else None)
            self.model = CatBoostClassifier(
                iterations=avg_best_iter,
                learning_rate=0.05,
                depth=5,
                l2_leaf_reg=10.0,
                loss_function='Logloss',
                eval_metric='AUC',
                verbose=False,
                random_seed=42,
                early_stopping_rounds=50,
            )
            self.model.fit(fb_pool, eval_set=fb_val_pool, use_best_model=True, verbose=False)
            n_train = fallback_split
            self.conformal_threshold = 0.50
        
        self.is_fitted = True
        self.version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self.metadata = {
            'val_auc': avg_val_auc,
            'best_iteration': avg_best_iter,
            'n_train': n_train,
            'n_splits_cv': len(best_iterations),
            'conformal_threshold': self.conformal_threshold,
            'version': self.version,
        }
        
        logger.info(f"MetaLabeler training complete. Version: {self.version} (used last fold model, no refit)")

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
                json.dump({'version': self.version, 'conformal_threshold': self.conformal_threshold, 'metrics': self.train_metrics}, f, indent=2, default=str)
            
    def load(self, path: str):
        self.model.load_model(path)
        self.is_fitted = True
        meta_path = path + '.meta.json'
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.version = meta.get('version', 'unknown')
                self.conformal_threshold = meta.get('conformal_threshold', 0.50)
                self.train_metrics = meta.get('metrics', {})
