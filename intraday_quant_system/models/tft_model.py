import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

class TFTNetwork(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head (predicting binary outcome)
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        
        # Generate causal mask to prevent attending to future positions (data leakage)
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transformer expects (batch_size, seq_len, embed_dim) with batch_first=True
        out = self.transformer(x, mask=mask)
        
        # Use the last time step for prediction
        last_step_out = out[:, -1, :]
        
        prob = self.output_head(last_step_out)
        return prob.squeeze(-1)

class TemporalFusionTransformerModel:
    """
    Temporal Fusion Transformer wrapper for Intraday Trading
    """
    def __init__(self, config: dict = None):
        config = config or {}
        
        self.embed_dim = config.get('embed_dim', 64)
        self.nhead = config.get('nhead', 4)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.sequence_length = config.get('sequence_length', 30)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"TFT Model initialized on {self.device}")
        
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    @staticmethod
    def prepare_sequences(df: pd.DataFrame, feature_cols: List[str], seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converts flat DataFrame into sliding window sequences.
        Returns: X_seq (N, seq_len, features), y_arr (N,), valid_indices (N,)
        """
        # Ensure target is present for y
        has_target = 'target' in df.columns
        
        data = df[feature_cols].values
        if has_target:
            targets = df['target'].values
        else:
            targets = np.zeros(len(data))
            
        N = len(data) - seq_len + 1
        if N <= 0:
            raise ValueError(f"DataFrame length ({len(data)}) must be >= sequence length ({seq_len})")
            
        X_seq = np.zeros((N, seq_len, len(feature_cols)))
        y_arr = np.zeros(N)
        valid_indices = np.zeros(N, dtype=int)
        
        for i in range(N):
            X_seq[i] = data[i:i+seq_len]
            # Target aligns with the end of the sequence
            y_arr[i] = targets[i+seq_len-1]
            valid_indices[i] = i+seq_len-1
            
        return X_seq, y_arr, valid_indices
        
    def train(self, X_seq: np.ndarray, y: np.ndarray, val_size: float = 0.2, epochs: int = 50, lr: float = 0.001) -> Dict[str, Any]:
        """
        Train TFT using early stopping.
        X_seq: (N, seq_len, num_features)
        """
        logger.info(f"Training TFT model on {len(X_seq)} sequences")
        
        input_dim = X_seq.shape[-1]
        self.model = TFTNetwork(
            input_dim=input_dim,
            embed_dim=self.embed_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        split_idx = int(len(X_seq) * (1 - val_size))
        
        X_train, y_train = X_seq[:split_idx], y[:split_idx]
        X_val, y_val = X_seq[split_idx:], y[split_idx:]
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        batch_size = 256
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            # Sequential batching to preserve temporal order (no random shuffling)
            indices_order = torch.arange(X_train_t.size()[0])
            for i in range(0, X_train_t.size()[0], batch_size):
                indices = indices_order[i:i+batch_size]
                batch_x, batch_y = X_train_t[indices], y_train_t[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                
            train_loss /= X_train_t.size()[0]
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        # Load best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        self.is_trained = True
        return {'val_loss': best_val_loss, 'epochs_trained': epoch}

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Predict probability for positive class"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
            
        self.model.eval()
        X_t = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            probs = self.model(X_t).cpu().numpy()
            
        return probs

    def save(self, filepath: str):
        """Save PyTorch model and config"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
            
        state = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_projection.in_features,
            'config': {
                'embed_dim': self.embed_dim,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length
            },
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(state, filepath)
        logger.info(f"TFT Model saved to {filepath}")
        
    @classmethod
    def load(cls, filepath: str) -> "TemporalFusionTransformerModel":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        state = torch.load(filepath, map_location=torch.device('cpu'), weights_only=True)
        instance = cls(state.get('config', {}))
        
        instance.model = TFTNetwork(
            input_dim=state['input_dim'],
            embed_dim=instance.embed_dim,
            nhead=instance.nhead,
            num_layers=instance.num_layers,
            dropout=instance.dropout
        ).to(instance.device)
        
        instance.model.load_state_dict(state['model_state_dict'])
        instance.feature_names = state.get('feature_names')
        instance.is_trained = True
        
        return instance
