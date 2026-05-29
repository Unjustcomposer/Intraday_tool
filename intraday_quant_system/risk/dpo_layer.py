import torch
import torch.nn as nn
try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    CVX_AVAILABLE = True
except ImportError:
    CVX_AVAILABLE = False
import numpy as np

class DPOLayer(nn.Module):
    """
    Differentiable Portfolio Optimization (DPO) layer.
    Uses cvxpylayers to formulate a Mean-Variance portfolio optimization problem 
    whose solution can be backpropagated through.
    """
    def __init__(self, n_assets: int, gamma: float = 1.0):
        """
        Args:
            n_assets: Number of assets in the portfolio
            gamma: Risk aversion parameter
        """
        super(DPOLayer, self).__init__()
        self.n_assets = n_assets
        self.gamma = gamma
        
        if not CVX_AVAILABLE:
            raise RuntimeError("DPOLayer requires cvxpy and cvxpylayers to be installed. Please compile them or install pre-built packages.")
            
        # Define the cvxpy problem
        w = cp.Variable(n_assets)
        mu = cp.Parameter(n_assets)
        Sigma = cp.Parameter((n_assets, n_assets))
        
        # Mean-variance objective: maximize mu^T w - gamma * w^T Sigma w
        # which is equivalent to minimizing gamma * w^T Sigma w - mu^T w
        objective = cp.Minimize(self.gamma * cp.quad_form(w, Sigma) - mu.T @ w)
        
        # Constraints: fully invested, long-only
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        problem = cp.Problem(objective, constraints)
        
        # Create the differentiable layer
        self.layer = CvxpyLayer(problem, parameters=[mu, Sigma], variables=[w])

    def forward(self, mu: torch.Tensor, Sigma: torch.Tensor) -> torch.Tensor:
        """
        Solve the portfolio optimization problem for a batch of inputs.
        
        Args:
            mu: Expected returns tensor of shape (batch_size, n_assets)
            Sigma: Covariance matrices tensor of shape (batch_size, n_assets, n_assets)
            
        Returns:
            Optimal portfolio weights tensor of shape (batch_size, n_assets)
        """
        # cvxpylayers requires batch dimensions, or handles them if inputs have batch dims
        # The layer expects inputs to be passed as arguments in the order of `parameters`
        weights, = self.layer(mu, Sigma)
        return weights
