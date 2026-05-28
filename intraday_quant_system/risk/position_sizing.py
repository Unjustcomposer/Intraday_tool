import math
import logging

logger = logging.getLogger(__name__)

def kelly_fraction(win_prob: float, win_loss_ratio: float, fraction: float = 0.5) -> float:
    """
    Calculate Kelly criterion position size.
    
    Args:
        win_prob: Estimated probability of a winning trade
        win_loss_ratio: Ratio of average win magnitude to average loss magnitude
        fraction: Kelly fraction to apply (typically 0.5 for Half-Kelly to reduce volatility)
    """
    if win_loss_ratio <= 0:
        return 0.0
        
    kelly_pct = win_prob - ((1 - win_prob) / win_loss_ratio)
    
    # Bound between 0 and 1, and apply fractional modifier
    size = max(0.0, min(1.0, kelly_pct)) * fraction
    return size

def historical_kelly_fraction(trade_history: list, regime: str = 'quiet', base_fraction: float = 0.5) -> float:
    """Calculate Kelly fraction from actual trade history with Regime-Conditional scaling"""
    
    # Apply regime modifiers (Phase 2 Redesign)
    regime_scalars = {
        'quiet': 1.0,           # Max exposure
        'bull_volatile': 0.5,   # Half exposure
        'bear_volatile': 0.5,   # Half exposure (0.25 Kelly)
        'unknown': 0.5
    }
    
    fraction = base_fraction * regime_scalars.get(regime, 0.5)
    
    if len(trade_history) < 30:
        return 0.1 * fraction  # Default safe size if insufficient history
        
    wins = [t for t in trade_history if t > 0]
    losses = [t for t in trade_history if t < 0]
    
    if not losses:
        return 1.0 * fraction
        
    win_prob = len(wins) / len(trade_history)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = abs(sum(losses) / len(losses))
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    return kelly_fraction(win_prob, win_loss_ratio, fraction)

def volatility_adjusted_size(target_vol: float, asset_vol: float, capital: float) -> float:
    """
    Calculate position size to maintain constant risk based on asset volatility.
    
    Args:
        target_vol: Target annualized volatility for the portfolio (e.g., 0.15)
        asset_vol: Annualized volatility of the asset
        capital: Total capital available
    """
    if asset_vol <= 0:
        return 0.0
        
    vol_scalar = target_vol / asset_vol
    # Cap leverage at 2x for safety
    vol_scalar = min(vol_scalar, 2.0)
    
    return capital * vol_scalar

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

