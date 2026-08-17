import logging
from typing import Any

logger = logging.getLogger(__name__)


def kelly_fraction(win_rate: float, risk_reward: float) -> float:
    """
    Calculate Kelly Criterion fraction.
    
    Kelly = W - (1 - W) / R
    where W = win rate, R = risk:reward ratio
    
    Args:
        win_rate: Win probability (0-1)
        risk_reward: Risk:Reward ratio (e.g., 2.0 for 1:2)
        
    Returns:
        Kelly fraction (can be negative if edge is negative)
    """
    if risk_reward <= 0:
        return 0.0
    return win_rate - ((1.0 - win_rate) / risk_reward)


def volatility_adjusted_size(
    target_volatility: float,
    asset_volatility: float,
    capital: float,
    max_leverage: float = 1.0
) -> float:
    """
    Calculate position size based on volatility targeting.
    
    Position = (target_vol / asset_vol) * capital
    
    Args:
        target_volatility: Target portfolio volatility (e.g., 0.15 for 15%)
        asset_volatility: Asset's annualized volatility
        capital: Available capital
        max_leverage: Maximum leverage allowed
        
    Returns:
        Position value in currency units
    """
    if asset_volatility <= 0:
        return capital * 0.1  # 10% fallback

    size = (target_volatility / asset_volatility) * capital
    max_size = capital * max_leverage

    return min(size, max_size)


class PortfolioLimits:
    """Portfolio-level risk limits."""

    @staticmethod
    def check_trade(
        capital: float,
        current_positions: int,
        proposed_risk: float,
        current_sector_exposure: float,
        current_portfolio_exposure: float,
        config: dict[str, Any]
    ) -> bool:
        """
        Check if a proposed trade passes all portfolio limits.
        
        Returns:
            True if trade is allowed, False otherwise
        """
        risk_config = config.get('risk', {})

        # Max positions
        max_positions = risk_config.get('max_open_positions', 5)
        if current_positions >= max_positions:
            logger.warning(f"Max positions limit reached: {current_positions} >= {max_positions}")
            return False

        # Max sector exposure
        max_sector = risk_config.get('max_sector_exposure', 0.25)
        if current_sector_exposure >= max_sector:
            logger.warning(f"Max sector exposure reached: {current_sector_exposure:.2%} >= {max_sector:.2%}")
            return False

        # Max portfolio exposure
        max_portfolio = risk_config.get('max_portfolio_exposure', 0.70)
        if current_portfolio_exposure >= max_portfolio:
            logger.warning(f"Max portfolio exposure reached: {current_portfolio_exposure:.2%} >= {max_portfolio:.2%}")
            return False

        # Min cash reserve
        min_cash = risk_config.get('min_cash_reserve', 0.30)
        if current_portfolio_exposure > (1 - min_cash):
            logger.warning(f"Min cash reserve violated: {current_portfolio_exposure:.2%} > {1 - min_cash:.2%}")
            return False

        return True

    @staticmethod
    def get_max_position_value(capital: float, config: dict[str, Any]) -> float:
        """Get maximum position value based on risk per trade."""
        max_risk = config.get('risk', {}).get('max_risk_per_trade', 0.02)
        return capital * max_risk
