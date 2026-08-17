import numpy as np


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Array of period returns (decimal, e.g., 0.01 = 1%)
        periods_per_year: Number of periods per year (252 for daily, trades_per_year for trade-level)

    Returns:
        Annualized Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0.0

    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray, periods_per_year: float, mar: float = 0.0
) -> float:
    """
    Calculate annualized Sortino Ratio (downside deviation).

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods per year
        mar: Minimum Acceptable Return (default 0)

    Returns:
        Annualized Sortino Ratio
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - mar
    downside = excess[excess < 0]

    if len(downside) == 0:
        return float("inf") if np.mean(excess) > 0 else 0.0

    downside_dev = np.sqrt(np.mean(downside**2))

    if downside_dev == 0:
        return 0.0

    return float(np.mean(excess) / downside_dev * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Array of equity values (not returns)

    Returns:
        Maximum drawdown as decimal (e.g., 0.15 = 15%)
    """
    if len(equity_curve) < 2:
        return 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max

    return float(np.max(drawdowns))


def profit_factor(wins: np.ndarray, losses: np.ndarray) -> float:
    """
    Calculate Profit Factor = Gross Profits / Gross Losses.

    Args:
        wins: Array of winning trade returns (positive)
        losses: Array of losing trade returns (negative or absolute)

    Returns:
        Profit Factor (inf if no losses)
    """
    gross_profit = np.sum(wins[wins > 0]) if len(wins) > 0 else 0
    gross_loss = np.sum(np.abs(losses[losses < 0])) if len(losses) > 0 else 0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def calmar_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    """
    Calculate Calmar Ratio = Annualized Return / Max Drawdown.

    Args:
        returns: Array of period returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar Ratio
    """
    if len(returns) < 2:
        return 0.0

    # Compound returns to equity curve
    equity = np.cumprod(1 + returns)
    max_dd = max_drawdown(equity)

    if max_dd == 0:
        return float("inf")

    annual_return = np.mean(returns) * periods_per_year
    return float(annual_return / max_dd)


def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate Omega Ratio = Probability-weighted gains / losses above threshold.

    Args:
        returns: Array of period returns
        threshold: Return threshold (default 0)

    Returns:
        Omega Ratio
    """
    if len(returns) < 2:
        return 0.0

    excess = returns - threshold
    gains = excess[excess > 0]
    losses = excess[excess < 0]

    if len(losses) == 0:
        return float("inf") if len(gains) > 0 else 0.0

    return float(np.sum(gains) / np.abs(np.sum(losses)))


def annualized_return(returns: np.ndarray, periods_per_year: float) -> float:
    """Calculate annualized return from period returns."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) * periods_per_year)


def annualized_volatility(returns: np.ndarray, periods_per_year: float) -> float:
    """Calculate annualized volatility from period returns."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def win_rate(returns: np.ndarray) -> float:
    """Calculate win rate from trade returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def avg_win_loss(returns: np.ndarray) -> tuple:
    """Calculate average win and average loss."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    return avg_win, avg_loss


def expectancy(returns: np.ndarray) -> float:
    """Calculate expectancy (average return per trade)."""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))
