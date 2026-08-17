import numpy as np
from scipy import stats


def calculate_dsr(
    returns: np.ndarray,
    n_trials: int,
    variance_of_sharpes: float,
    annualization_factor: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR) - Probabilistic Sharpe Ratio with multiple testing correction.

    Based on Bailey & López de Prado (2014) "The Deflated Sharpe Ratio: correcting for selection bias"

    The DSR adjusts the Sharpe Ratio for:
    1. Non-normal returns (skewness, kurtosis)
    2. Multiple testing (number of trials/configurations tried)
    3. Sample length (finite sample correction)

    Args:
        returns: Array of trade-level returns (decimal, e.g., 0.01 = 1%)
        n_trials: Number of independent strategy trials/configurations tested
        variance_of_sharpes: Cross-sectional variance of Sharpe ratios across trials
        annualization_factor: Number of periods per year (e.g., 252 for daily, trades_per_year for trade-level)
        benchmark_sharpe: Minimum acceptable Sharpe ratio (default 0.0)

    Returns:
        DSR value: Probability that true Sharpe > benchmark_sharpe (0 to 1)
        DSR >= 0.95 means 95% confidence strategy has positive edge after selection bias
    """
    if len(returns) < 2:
        return 0.0

    # Sample statistics
    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0.0

    # Observed Sharpe Ratio (annualized)
    sr_observed = mean_ret / std_ret * np.sqrt(annualization_factor)

    # Sample skewness and kurtosis
    skew = stats.skew(returns, bias=False)
    kurt = stats.kurtosis(returns, bias=False, fisher=True)  # Excess kurtosis

    # Standard Error of Sharpe Ratio (Lo 2002, Mertens 2002)
    # Accounts for non-normality
    se_sr = np.sqrt(
        (
            1
            + 0.5 * sr_observed**2
            - skew * sr_observed
            + (kurt - 3) / 4 * sr_observed**2
        )
        / (n - 1)
    )

    # Annualize SE
    se_sr_annual = se_sr * np.sqrt(annualization_factor)

    if se_sr_annual == 0:
        return 1.0 if sr_observed > benchmark_sharpe else 0.0

    # Probabilistic Sharpe Ratio (PSR) - no multiple testing correction
    # PSR = Φ[(SR_obs - SR_bench) / SE(SR)]
    z_psr = (sr_observed - benchmark_sharpe) / se_sr_annual
    stats.norm.cdf(z_psr)

    # Deflated Sharpe Ratio - corrects for multiple trials
    # DSR = Φ[(SR_obs - SR_0) / SE(SR)] where SR_0 is expected max Sharpe under null

    # Expected maximum Sharpe under null (no skill) across n_trials
    # Using approximation: E[max] ≈ μ + σ * Φ^(-1)(1 - 1/n_trials)
    # Under null: μ = benchmark_sharpe, σ = sqrt(variance_of_sharpes)

    if variance_of_sharpes <= 0:
        variance_of_sharpes = 1e-6

    # Expected maximum Sharpe under null (Embrechts et al. approximation)
    expected_max_sr = benchmark_sharpe + np.sqrt(variance_of_sharpes) * stats.norm.ppf(
        1 - 1 / n_trials
    )

    # DSR: probability observed SR exceeds expected max under null
    z_dsr = (sr_observed - expected_max_sr) / se_sr_annual
    dsr = stats.norm.cdf(z_dsr)

    return float(np.clip(dsr, 0.0, 1.0))


def calculate_psr(
    returns: np.ndarray,
    annualization_factor: float,
    benchmark_sharpe: float = 0.0,
) -> float:
    """
    Calculate Probabilistic Sharpe Ratio (PSR) without multiple testing correction.

    Simpler version: just tests if Sharpe > benchmark given sample statistics.
    """
    if len(returns) < 2:
        return 0.0

    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return 0.0

    sr_observed = mean_ret / std_ret * np.sqrt(annualization_factor)

    skew = stats.skew(returns, bias=False)
    kurt = stats.kurtosis(returns, bias=False, fisher=True)

    se_sr = np.sqrt(
        (
            1
            + 0.5 * sr_observed**2
            - skew * sr_observed
            + (kurt - 3) / 4 * sr_observed**2
        )
        / (n - 1)
    )

    se_sr_annual = se_sr * np.sqrt(annualization_factor)

    if se_sr_annual == 0:
        return 1.0 if sr_observed > benchmark_sharpe else 0.0

    z = (sr_observed - benchmark_sharpe) / se_sr_annual
    return float(stats.norm.cdf(z))


def calculate_min_track_record(
    target_sr: float,
    target_psr: float = 0.95,
    skew: float = 0.0,
    kurt: float = 3.0,
    annualization_factor: float = 252,
) -> int:
    """
    Calculate minimum track record length (in periods) to achieve target PSR.

    Inverts the PSR formula to solve for n.

    Args:
        target_sr: Target Sharpe Ratio
        target_psr: Target Probabilistic Sharpe Ratio (e.g., 0.95)
        skew: Expected skewness of returns
        kurt: Expected kurtosis of returns (excess kurtosis + 3)
        annualization_factor: Periods per year

    Returns:
        Minimum number of periods (trades/days) needed
    """
    if target_psr <= 0.5:
        return 1

    z = stats.norm.ppf(target_psr)

    # From PSR formula: z = (SR - 0) / SE(SR)
    # SE(SR) = sqrt((1 + 0.5*SR^2 - skew*SR + (kurt-3)/4*SR^2) / (n-1))
    # Solve for n:

    numerator = (
        1 + 0.5 * target_sr**2 - skew * target_sr + (kurt - 3) / 4 * target_sr**2
    )
    n = numerator / (z**2 / annualization_factor) + 1

    return int(np.ceil(n))


def sharpe_ratio_confidence_interval(
    returns: np.ndarray,
    annualization_factor: float,
    confidence: float = 0.95,
) -> tuple:
    """
    Calculate confidence interval for Sharpe Ratio.

    Returns:
        (lower_bound, upper_bound) annualized Sharpe
    """
    if len(returns) < 2:
        return (0.0, 0.0)

    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    if std_ret == 0:
        return (0.0, 0.0)

    sr_observed = mean_ret / std_ret * np.sqrt(annualization_factor)

    skew = stats.skew(returns, bias=False)
    kurt = stats.kurtosis(returns, bias=False, fisher=True)

    se_sr = np.sqrt(
        (
            1
            + 0.5 * sr_observed**2
            - skew * sr_observed
            + (kurt - 3) / 4 * sr_observed**2
        )
        / (n - 1)
    )

    se_sr_annual = se_sr * np.sqrt(annualization_factor)
    z = stats.norm.ppf((1 + confidence) / 2)

    margin = z * se_sr_annual
    return (sr_observed - margin, sr_observed + margin)


# Backward compatibility alias
DeflatedSharpeRatio = calculate_dsr
