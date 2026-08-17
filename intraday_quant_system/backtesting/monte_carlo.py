import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloStressTester:
    """
    Monte Carlo stress tester for trade-level returns.

    Performs bootstrap resampling of historical trades to simulate
    alternative equity curves and estimate tail risk.
    """

    def __init__(
        self,
        n_simulations: int = 5000,
        confidence_levels: list[float] | None = None,
        initial_capital: float = 1_000_000.0,
        risk_per_trade: float = 0.02,
        random_seed: int = 42,
    ):
        """
        Args:
            n_simulations: Number of Monte Carlo paths
            confidence_levels: Percentiles to report (e.g., [0.05, 0.25, 0.5, 0.75, 0.95])
            initial_capital: Starting capital for simulation
            risk_per_trade: Fraction of capital risked per trade (for position sizing)
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.confidence_levels = confidence_levels or [0.05, 0.25, 0.5, 0.75, 0.95]
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.random_seed = random_seed

        np.random.seed(random_seed)

    def run(
        self, trades_df: pd.DataFrame, initial_capital: float | None = None
    ) -> dict:
        """
        Run Monte Carlo simulation on historical trades.

        Args:
            trades_df: DataFrame with columns:
                - 'return': Net return per trade (decimal, e.g., 0.02 = 2%)
                - 'duration_bars': Holding period in bars (optional, for compounding)
                - 'side': 'long' or 'short' (optional)
            initial_capital: Override default initial capital

        Returns:
            Dict with simulation results including percentiles, ruin probability, etc.
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital

        if trades_df.empty:
            logger.warning("Empty trades DataFrame provided")
            return self._empty_result()

        if "return" not in trades_df.columns:
            logger.error("trades_df must contain 'return' column")
            return self._empty_result()

        returns = trades_df["return"].values
        n_trades = len(returns)

        if n_trades < 10:
            logger.warning(f"Only {n_trades} trades - results may be unreliable")

        # Run simulations
        terminal_capitals = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # Block Bootstrapping to preserve volatility clustering
            block_size = max(5, n_trades // 20)
            n_blocks = int(np.ceil(n_trades / block_size))
            sampled_returns = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, max(1, n_trades - block_size + 1))
                sampled_returns.extend(returns[start_idx:start_idx + block_size])
            sampled_returns = np.array(sampled_returns)[:n_trades]

            # Simulate equity curve
            equity = self.initial_capital
            peak = self.initial_capital
            max_dd = 0.0

            for ret in sampled_returns:
                equity *= 1 + ret
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

            terminal_capitals[i] = equity
            max_drawdowns[i] = max_dd

        # Calculate statistics
        profit_prob = (terminal_capitals > self.initial_capital).mean()
        ruin_50_prob = (max_drawdowns > 0.5).mean()
        ruin_25_prob = (max_drawdowns > 0.25).mean()

        percentiles = np.percentile(
            terminal_capitals, [c * 100 for c in self.confidence_levels]
        )
        dd_percentiles = np.percentile(
            max_drawdowns, [c * 100 for c in self.confidence_levels]
        )

        # VaR / CVaR
        var_95 = self.initial_capital - np.percentile(terminal_capitals, 5)
        cvar_95 = (
            self.initial_capital
            - terminal_capitals[
                terminal_capitals <= np.percentile(terminal_capitals, 5)
            ].mean()
        )

        return {
            "n_trades_simulated": n_trades,
            "n_simulations": self.n_simulations,
            "initial_capital": self.initial_capital,
            "median_terminal_capital": float(np.median(terminal_capitals)),
            "mean_terminal_capital": float(np.mean(terminal_capitals)),
            "std_terminal_capital": float(np.std(terminal_capitals)),
            "worst_5pct_capital": float(percentiles[0]),
            "best_5pct_capital": float(percentiles[-1]),
            "percentiles": {
                f"p{int(c*100)}": float(p)
                for c, p in zip(self.confidence_levels, percentiles, strict=False)
            },
            "probability_of_profit": float(profit_prob),
            "probability_of_ruin_50pct": float(ruin_50_prob),
            "probability_of_ruin_25pct": float(ruin_25_prob),
            "median_max_drawdown": float(np.median(max_drawdowns)),
            "mean_max_drawdown": float(np.mean(max_drawdowns)),
            "max_drawdown_percentiles": {
                f"p{int(c*100)}": float(p)
                for c, p in zip(self.confidence_levels, dd_percentiles, strict=False)
            },
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "sharpe_ratio": float(self._compute_sharpe(terminal_capitals)),
        }

    def _compute_sharpe(self, terminal_capitals: np.ndarray) -> float:
        """Compute Sharpe ratio of terminal returns."""
        returns = (terminal_capitals - self.initial_capital) / self.initial_capital
        if np.std(returns) > 0:
            return float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        return 0.0

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "n_trades_simulated": 0,
            "n_simulations": self.n_simulations,
            "initial_capital": self.initial_capital,
            "median_terminal_capital": self.initial_capital,
            "mean_terminal_capital": self.initial_capital,
            "std_terminal_capital": 0.0,
            "worst_5pct_capital": self.initial_capital,
            "best_5pct_capital": self.initial_capital,
            "percentiles": {
                f"p{int(c*100)}": self.initial_capital for c in self.confidence_levels
            },
            "probability_of_profit": 0.0,
            "probability_of_ruin_50pct": 1.0,
            "probability_of_ruin_25pct": 1.0,
            "median_max_drawdown": 0.0,
            "mean_max_drawdown": 0.0,
            "max_drawdown_percentiles": {
                f"p{int(c*100)}": 0.0 for c in self.confidence_levels
            },
            "var_95": 0.0,
            "cvar_95": 0.0,
            "sharpe_ratio": 0.0,
        }

    def simulate_path(
        self, trades_df: pd.DataFrame, n_paths: int = 100
    ) -> pd.DataFrame:
        """
        Generate equity curve paths for visualization.

        Returns:
            DataFrame with columns: path_id, step, equity
        """
        if trades_df.empty or "return" not in trades_df.columns:
            return pd.DataFrame()

        returns = trades_df["return"].values
        n_trades = len(returns)

        paths = []
        for path_id in range(min(n_paths, self.n_simulations)):
            # Block Bootstrapping to preserve volatility clustering
            block_size = max(5, n_trades // 20)
            n_blocks = int(np.ceil(n_trades / block_size))
            sampled_returns = []
            for _ in range(n_blocks):
                start_idx = np.random.randint(0, max(1, n_trades - block_size + 1))
                sampled_returns.extend(returns[start_idx:start_idx + block_size])
            sampled = np.array(sampled_returns)[:n_trades]
            equity = self.initial_capital
            for step, ret in enumerate(sampled):
                equity *= 1 + ret
                paths.append({"path_id": path_id, "step": step, "equity": equity})

        return pd.DataFrame(paths)
