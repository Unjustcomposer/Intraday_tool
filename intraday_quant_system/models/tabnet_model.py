import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

class TabNetModel:
    """
    TabNet Wrapper to replace XGBoost.
    Uses pytorch-tabnet for attention-based tabular learning.
    """
    def __init__(self, model_type: str = 'classifier', params: Optional[Dict[str, Any]] = None):
        self.model_type = model_type
        
        default_params = {
            'n_d': 8,
            'n_a': 8,
            'n_steps': 3,
            'gamma': 1.3,
            'n_independent': 2,
            'n_shared': 2,
            'momentum': 0.02,
            'clip_value': 1.,
            'lambda_sparse': 1e-3,
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=2e-2),
            'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
            'scheduler_params': {"mode":'min', "patience":10, "min_lr":1e-5, "factor":0.5},
            'mask_type': 'entmax',
            'verbose': 1
        }
        
        if params:
            default_params.update(params)
            
        self.params = default_params
        
        if self.model_type == 'classifier':
            self.model = TabNetClassifier(**self.params)
        elif self.model_type == 'regressor':
            self.model = TabNetRegressor(**self.params)
        else:
            raise ValueError("model_type must be either 'classifier' or 'regressor'")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_valid: Optional[np.ndarray] = None, y_valid: Optional[np.ndarray] = None,
            max_epochs: int = 100, patience: int = 15, batch_size: int = 1024, 
            virtual_batch_size: int = 128):
        
        eval_set = []
        if X_valid is not None and y_valid is not None:
            eval_set = [(X_train, y_train), (X_valid, y_valid)]
            eval_name = ['train', 'valid']
            eval_metric = ['auc'] if self.model_type == 'classifier' else ['rmse']
        else:
            eval_set = [(X_train, y_train)]
            eval_name = ['train']
            eval_metric = ['auc'] if self.model_type == 'classifier' else ['rmse']

        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=eval_metric,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            num_workers=0,
            drop_last=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_type != 'classifier':
            raise ValueError("predict_proba only available for classifier")
        return self.model.predict_proba(X)
        
    def save_model(self, path: str):
        self.model.save_model(path)
        
    def load_model(self, path: str):
        if not path.endswith('.zip'):
            path = path + '.zip'
        self.model.load_model(path)
