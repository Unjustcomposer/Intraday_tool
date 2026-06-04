import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# Conditionally import PyG
try:
    from torch_geometric.nn import GATConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

logger = logging.getLogger(__name__)

class MomentumSpilloverGNN(nn.Module):
    """
    Phase 5: Graph Neural Network for Sector Momentum Spillover.
    Uses Graph Attention Networks (GAT) to predict momentum cascades
    across Indian equities based on supply-chain and sector-correlation adjacency matrices.
    """
    def __init__(self, num_node_features: int, hidden_dim: int = 64, num_classes: int = 1):
        super(MomentumSpilloverGNN, self).__init__()
        self.enabled = PYG_AVAILABLE
        
        if self.enabled:
            # 2-layer Graph Attention Network
            self.conv1 = GATConv(num_node_features, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)
            self.fc = nn.Linear(hidden_dim, num_classes)
        else:
            logger.warning("torch_geometric not installed. GNN model is disabled.")
            
    def forward(self, x, edge_index, batch):
        if not self.enabled:
            raise NotImplementedError("PyG required")
            
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Graph pooling (if predicting market-wide regime) or node-level (for stock predictions)
        # Here we do node-level predictions for individual stock momentum
        out = self.fc(x)
        return torch.sigmoid(out)

    def train_epoch(self, dataloader, optimizer, criterion):
        """Standard PyG training loop stub"""
        if not self.enabled:
            return 0.0
            
        self.train()
        total_loss = 0
        for data in dataloader:
            optimizer.zero_grad()
            out = self(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
