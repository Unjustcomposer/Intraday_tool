import logging
from collections.abc import Callable, Generator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PurgedWalkForwardValidator:
    """
    Purged Walk-Forward Cross-Validator for financial time series.

    Implements purge/embargo logic to prevent data leakage:
    - Purge: Remove bars at end of training set (forward leakage)
    - Embargo: Remove bars at start of validation set (backward leakage)

    Based on López de Prado (2018) "Advances in Financial Machine Learning"
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: float = 0.7,
        val_size: float = 0.15,
        purge_bars: int = 10,
        embargo_bars: int = 10,
        step_size: int | None = None,
        min_train_size: int = 100,
        min_val_size: int = 30,
    ):
        """
        Args:
            n_splits: Number of walk-forward splits
            train_size: Proportion of data for training (expanding window)
            val_size: Proportion of data for validation
            purge_bars: Number of bars to purge between train and val
            embargo_bars: Number of bars to embargo at validation start
            step_size: If set, use sliding window; else expanding window
            min_train_size: Minimum training samples required
            min_val_size: Minimum validation samples required
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.purge_bars = purge_bars
        self.embargo_bars = embargo_bars
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.min_val_size = min_val_size

        if train_size + val_size > 1.0:
            raise ValueError("train_size + val_size must be <= 1.0")
        if purge_bars < 0 or embargo_bars < 0:
            raise ValueError("purge_bars and embargo_bars must be >= 0")

    def split(
        self, X: pd.DataFrame
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/validation indices with purge and embargo.

        Yields:
            (train_idx, val_idx) - integer indices for iloc indexing
        """
        n_samples = len(X)
        if n_samples < self.min_train_size + self.min_val_size:
            logger.warning(
                f"Insufficient data: {n_samples} samples, need at least {self.min_train_size + self.min_val_size}"
            )
            return

        # Calculate base sizes
        if self.step_size is not None:
            # Sliding window: each split moves by step_size
            total_window = int(n_samples * (self.train_size + self.val_size))
            max_start = n_samples - total_window
            if max_start <= 0:
                logger.warning("Data too small for sliding window")
                return

            for i in range(min(self.n_splits, max_start // max(1, self.step_size) + 1)):
                start = i * self.step_size
                train_end = start + int(n_samples * self.train_size)
                val_end = min(start + total_window, n_samples)

                train_idx = np.arange(start, max(start, train_end - self.purge_bars))
                val_start = train_end + self.embargo_bars
                val_idx = np.arange(val_start, val_end)

                if (
                    len(train_idx) >= self.min_train_size
                    and len(val_idx) >= self.min_val_size
                ):
                    yield train_idx, val_idx
        else:
            # Expanding window: training grows, validation moves forward
            val_start_base = int(n_samples * self.train_size)
            val_step = (
                int(n_samples * self.val_size / max(1, self.n_splits - 1))
                if self.n_splits > 1
                else 0
            )

            for i in range(self.n_splits):
                train_end = val_start_base + i * val_step
                val_end = min(train_end + int(n_samples * self.val_size), n_samples)

                # Apply purge to training end
                train_end_purged = max(0, train_end - self.purge_bars)

                # Apply embargo to validation start
                val_start_embargoed = min(n_samples, train_end + self.embargo_bars)

                train_idx = np.arange(0, train_end_purged)
                val_idx = np.arange(val_start_embargoed, val_end)

                if (
                    len(train_idx) >= self.min_train_size
                    and len(val_idx) >= self.min_val_size
                ):
                    yield train_idx, val_idx
                else:
                    logger.debug(
                        f"Split {i} skipped: train={len(train_idx)}, val={len(val_idx)}"
                    )

    def run(
        self,
        df: pd.DataFrame,
        train_fn: Callable[[pd.DataFrame, pd.DataFrame], dict],
        feature_cols: list[str] | None = None,
        label_col: str = "label",
    ) -> list[dict]:
        """
        Run walk-forward validation with a user-provided training function.

        Args:
            df: Full DataFrame with features and labels
            train_fn: Function(train_df, val_df) -> dict with metrics
            feature_cols: List of feature column names (if None, inferred)
            label_col: Name of label column

        Returns:
            List of result dicts, one per split
        """
        results = []

        for split_id, (train_idx, val_idx) in enumerate(self.split(df)):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            logger.info(
                f"Split {split_id + 1}/{self.n_splits}: train={len(train_df)}, val={len(val_df)}"
            )

            try:
                result = train_fn(train_df, val_df)
                result["split_id"] = split_id
                result["train_size"] = len(train_df)
                result["val_size"] = len(val_df)
                results.append(result)
            except Exception as e:
                logger.error(f"Split {split_id} failed: {e}")
                results.append(
                    {
                        "split_id": split_id,
                        "train_size": len(train_df),
                        "val_size": len(val_df),
                        "error": str(e),
                    }
                )

        return results

    def aggregate_results(self, results: list[dict]) -> dict:
        """
        Aggregate metrics across all splits.

        Returns:
            Dict with mean, std, min, max, and % positive for each metric
        """
        if not results:
            return {}

        # Collect all metric keys (excluding metadata)
        metric_keys = set()
        for r in results:
            for k, v in r.items():
                if k not in (
                    "split_id",
                    "train_size",
                    "val_size",
                    "error",
                ) and isinstance(v, int | float):
                    metric_keys.add(k)

        aggregated = {}
        for key in metric_keys:
            values = [
                r[key] for r in results if key in r and isinstance(r[key], int | float)
            ]
            if not values:
                continue

            values_arr = np.array(values)
            positive_pct = (
                (values_arr > 0).mean()
                if key in ("sharpe", "profit_factor", "net_return", "sortino")
                else 0.0
            )

            aggregated[f"avg_{key}"] = float(np.mean(values_arr))
            aggregated[f"std_{key}"] = float(np.std(values_arr))
            aggregated[f"min_{key}"] = float(np.min(values_arr))
            aggregated[f"max_{key}"] = float(np.max(values_arr))
            aggregated[f"pct_positive_{key}"] = float(positive_pct)

        # Overall stats
        aggregated["n_splits"] = len(results)
        aggregated["n_successful"] = sum(1 for r in results if "error" not in r)

        return aggregated
