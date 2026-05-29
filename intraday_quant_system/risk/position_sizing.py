import math
import logging
import pandas as pd
from typing import Dict, List
import torch
import numpy as np
from risk.dpo_layer import DPOLayer

logger = logging.getLogger(__name__)

def dpo_position_sizes(returns_df: pd.DataFrame, signals: Dict[str, str], capital: float, gamma: float = 1.0) -> Dict[str, float]:
    """
    Calculate position sizes using Differentiable Portfolio Optimization (DPO).
    Replaces static Kelly/VaR sizing with Mean-Variance optimization via cvxpylayers.
    
    Args:
        returns_df: DataFrame of rolling asset returns (columns are symbols)
        signals: Dict mapping symbol -> 'buy' or 'sell' (or 'no_trade')
        capital: Total capital available
        gamma: Risk aversion parameter for DPO
    """
    symbols = list(signals.keys())
    available_syms = [s for s in symbols if s in returns_df.columns]
    
    if len(available_syms) < 2:
        sizes = {}
        for sym in available_syms:
            if signals[sym] in ['buy', 'sell']:
                sizes[sym] = capital / len(available_syms)
            else:
                sizes[sym] = 0.0
        return sizes

    # Use historical mean as proxy for expected returns, adjusted by signal direction
    mu_hist = returns_df[available_syms].mean().values
    mu_adjusted = np.zeros_like(mu_hist)
    
    for i, sym in enumerate(available_syms):
        if signals[sym] == 'buy':
            mu_adjusted[i] = abs(mu_hist[i]) if mu_hist[i] != 0 else 0.001
        elif signals[sym] == 'sell':
            # Note: The DPO layer is long-only. For short positions, we would 
            # ideally modify constraints. Here we provide a negative expected return 
            # which will result in 0 weight under long-only constraints.
            mu_adjusted[i] = -abs(mu_hist[i]) if mu_hist[i] != 0 else -0.001
        else:
            mu_adjusted[i] = 0.0
            
    cov_matrix = returns_df[available_syms].cov().values
    # Regularize covariance to ensure positive semi-definiteness
    cov_matrix += np.eye(len(available_syms)) * 1e-6
    
    mu_tensor = torch.tensor(mu_adjusted, dtype=torch.float32).unsqueeze(0)
    Sigma_tensor = torch.tensor(cov_matrix, dtype=torch.float32).unsqueeze(0)
    
    try:
        dpo = DPOLayer(n_assets=len(available_syms), gamma=gamma)
        weights_batch = dpo(mu_tensor, Sigma_tensor)
        weights = weights_batch.squeeze(0).detach().numpy()
    except Exception as e:
        logger.error(f"DPO optimization failed: {e}")
        weights = np.ones(len(available_syms)) / len(available_syms)
        
    sizes = {}
    for i, sym in enumerate(available_syms):
        w = max(0.0, float(weights[i]))
        # Note: If we want to allow shorting, sizes should just be capital * w * direction
        # Here we follow the weight magnitude.
        if signals[sym] in ['buy', 'sell'] and w > 0:
            sizes[sym] = capital * w
        else:
            sizes[sym] = 0.0
            
    return sizes

class PortfolioLimits:
    """Enforces absolute portfolio limits"""
    
    @staticmethod
    def check_trade(
        capital: float,
        current_positions: int,
        proposed_risk: float,
        current_sector_exposure: float,
        current_portfolio_exposure: float,
        config: dict = None
    ) -> bool:
        """
        Validates if a proposed trade violates any portfolio limits.
        """
        config = config or {}
        risk_params = config.get('risk', {})
        
        max_positions = risk_params.get('max_open_positions', 5)
        max_risk_pct = risk_params.get('max_risk_per_trade', 0.02)
        max_sector_pct = risk_params.get('max_sector_exposure', 0.25)
        max_portfolio_pct = risk_params.get('max_portfolio_exposure', 0.70)
        
        if current_positions >= max_positions:
            logger.warning(f"Trade rejected: Max positions reached ({max_positions})")
            return False
            
        risk_pct = proposed_risk / capital
        if risk_pct > max_risk_pct:
            logger.warning(f"Trade rejected: Risk too high ({risk_pct:.2%} > {max_risk_pct:.2%})")
            return False
            
        if current_sector_exposure > max_sector_pct:
            logger.warning(f"Trade rejected: Sector exposure limit breached ({current_sector_exposure:.2%} > {max_sector_pct:.2%})")
            return False
            
        if current_portfolio_exposure > max_portfolio_pct:
            logger.warning(f"Trade rejected: Portfolio exposure limit breached ({current_portfolio_exposure:.2%} > {max_portfolio_pct:.2%})")
            return False
            
        return True

def apply_correlation_discounts(proposed_sizes: Dict[str, float], returns_df: pd.DataFrame,
                              signals: Dict[str, str]) -> Dict[str, float]:
    """
    Scale down proposed position sizes if assets are highly correlated and bet in the same direction.
    
    Args:
        proposed_sizes: Dict mapping symbol -> proposed size in currency units
        returns_df: DataFrame of rolling asset returns (columns are symbols)
        signals: Dict mapping symbol -> 'buy' or 'sell'
    """
    if len(proposed_sizes) < 2 or returns_df.empty:
        return proposed_sizes
        
    symbols = list(proposed_sizes.keys())
    # Intersect with returns columns
    available_syms = [s for s in symbols if s in returns_df.columns]
    if len(available_syms) < 2:
        return proposed_sizes
        
    # Calculate correlation matrix
    corr_matrix = returns_df[available_syms].corr()
    
    adjusted_sizes = proposed_sizes.copy()
    
    for i in range(len(available_syms)):
        sym_i = available_syms[i]
        sig_i = signals.get(sym_i, 'no_trade')
        if sig_i not in ['buy', 'sell']:
            continue
            
        discount = 0.0
        n_correlated = 0
        
        for j in range(len(available_syms)):
            if i == j:
                continue
            sym_j = available_syms[j]
            sig_j = signals.get(sym_j, 'no_trade')
            
            # Only discount if they are on the same side
            if sig_j == sig_i:
                correlation = corr_matrix.loc[sym_i, sym_j]
                if pd.notna(correlation) and correlation > 0.5:
                    # Discount increases with correlation
                    discount += (correlation - 0.5)
                    n_correlated += 1
                    
        if n_correlated > 0:
            # Average discount across correlated peers, capped at 50% max reduction
            avg_discount = min(0.5, discount / n_correlated)
            scale_factor = 1.0 - avg_discount
            adjusted_sizes[sym_i] = proposed_sizes[sym_i] * scale_factor
            logger.info(f"Correlation sizing discount for {sym_i}: scaled by {scale_factor:.2f} due to {n_correlated} correlated bets.")
            
    return adjusted_sizes

def kelly_fraction(win_prob: float, win_loss_ratio: float, fraction: float = 0.5) -> float:
    """
    Calculate the Kelly fraction.
    By default, uses half-Kelly (fraction=0.5) to avoid excessive drawdowns.
    """
    if win_loss_ratio <= 0:
        return 0.0
    f = win_prob - (1.0 - win_prob) / win_loss_ratio
    return max(0.0, f * fraction)

def volatility_adjusted_size(target_vol: float, asset_vol: float, capital: float) -> float:
    """
    Calculate position size adjusted by target volatility vs asset volatility.
    """
    if asset_vol <= 0:
        return 0.0
    return capital * (target_vol / asset_vol)

