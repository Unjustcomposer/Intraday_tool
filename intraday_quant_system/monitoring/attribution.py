"""
Daily PnL Attribution
======================

Decomposes daily PnL into:
- Alpha (model-driven returns)
- Beta (market exposure returns)  
- Sector allocation
- Factor exposures (size, value, momentum, etc.)
- Residual/idiosyncratic returns

This is essential for understanding WHERE profits come from
and detecting strategy decay or regime changes.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Result of PnL attribution analysis"""
    date: str
    total_pnl: float
    alpha: float                    # Model-driven returns (residual after factor exposure)
    beta_pnl: float                 # Market beta exposure PnL
    sector_allocation_pnl: float    # Sector allocation effect
    factor_pnl: Dict[str, float]    # Factor exposures (size, value, momentum, etc.)
    idiosyncratic_pnl: float        # Unexplained/idiosyncratic
    r_squared: float                # Model fit quality
    
    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "total_pnl": self.total_pnl,
            "alpha": self.alpha,
            "beta_pnl": self.beta_pnl,
            "sector_allocation_pnl": self.sector_allocation_pnl,
            "factor_pnl": self.factor_pnl,
            "idiosyncratic_pnl": self.idiosyncratic_pnl,
            "r_squared": self.r_squared,
        }


class PnLAttribution:
    """
    Daily PnL attribution using multi-factor regression.
    
    Decomposes portfolio returns into:
    - Market beta (systematic risk)
    - Sector allocation (active sector bets)
    - Style factors (size, value, momentum, quality, low vol)
    - Alpha (residual after factor exposure)
    
    Based on Brinson-Hood-Beebower attribution framework extended
    with multi-factor models (Fama-French, Barra-style).
    """
    
    def __init__(
        self,
        factor_data: Optional[pd.DataFrame] = None,
        sector_map: Optional[Dict[str, str]] = None,
        benchmark_symbol: str = "NIFTY50",
        risk_free_rate: float = 0.06,
    ):
        """
        Initialize attribution engine.
        
        Args:
            factor_data: DataFrame with daily factor returns
                         Columns: ['market', 'size', 'value', 'momentum', 'quality', 'low_vol', ...]
                         Index: datetime
            sector_map: Dict mapping symbol -> sector
            benchmark_symbol: Benchmark symbol for beta calculation
            risk_free_rate: Annual risk-free rate (decimal)
        """
        self.factor_data = factor_data
        self.sector_map = sector_map or {}
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        
        # Default factor names if not provided
        self.factor_names = list(factor_data.columns) if factor_data is not None else [
            'market', 'size', 'value', 'momentum', 'quality', 'low_vol'
        ]
        
        # Cache for factor model
        self._factor_model = None
        self._factor_loadings = {}
    
    def fit_factor_model(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        lookback_days: int = 252,
        min_observations: int = 60,
    ) -> Dict[str, float]:
        """
        Fit multi-factor regression to estimate factor loadings.
        
        Uses rolling OLS regression with lookback window.
        
        Returns:
            Dict of factor loadings (betas) for the latest period
        """
        # Align data
        common_index = portfolio_returns.index.intersection(factor_returns.index)
        if len(common_index) < min_observations:
            logger.warning(f"Insufficient observations for factor model: {len(common_index)}")
            return {}
        
        y = portfolio_returns.loc[common_index].values
        X = factor_returns.loc[common_index, self.factor_names].values
        
        # Rolling regression - use last window
        if len(common_index) > lookback_days:
            y = y[-lookback_days:]
            X = X[-lookback_days:]
        
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS: (X'X)^-1 X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            loadings = {
                'alpha': beta[0],
                **{name: beta[i+1] for i, name in enumerate(self.factor_names)}
            }
            
            self._factor_loadings = loadings
            self._factor_model = {
                'beta': beta,
                'r_squared': self._compute_r2(y, X, beta),
                'n_obs': len(y)
            }
            
            return loadings
            
        except Exception as e:
            logger.error(f"Factor model fitting failed: {e}")
            return {}
    
    def _compute_r2(self, y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> float:
        """Compute R-squared of regression"""
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    def attribute_daily_pnl(
        self,
        date: str,
        portfolio_returns: pd.Series,
        positions: pd.DataFrame,
        factor_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> AttributionResult:
        """
        Attribute daily PnL to various sources.
        
        Args:
            date: Attribution date
            portfolio_returns: Series of position returns (index=symbol)
            positions: DataFrame with columns ['symbol', 'quantity', 'price', 'sector']
            factor_returns: DataFrame of daily factor returns
            benchmark_returns: Series of benchmark returns (index=symbol or single value)
            
        Returns:
            AttributionResult with PnL decomposition
        """
        # Total portfolio PnL
        total_pnl = portfolio_returns.sum()
        
        # If we have factor data, run attribution
        if factor_returns is not None and len(self.factor_names) > 0:
            loadings = self.fit_factor_model(
                portfolio_returns, factor_returns
            )
            
            if loadings:
                # Factor PnL contributions
                factor_pnl = {}
                for factor in self.factor_names:
                    if factor in factor_returns.columns and factor in loadings:
                        # Factor PnL = loading * factor_return * portfolio_value
                        factor_ret = factor_returns[factor].iloc[-1]
                        loading = loadings[factor]
                        # Approximate: factor PnL = loading * factor_return * capital
                        factor_pnl[factor] = loading * factor_ret
                
                # Beta PnL (market exposure)
                beta_pnl = 0.0
                if 'market' in loadings and 'market' in factor_returns.columns:
                    beta_pnl = loadings['market'] * factor_returns['market'].iloc[-1]
                
                # Alpha = residual
                total_factor_pnl = sum(factor_pnl.values())
                alpha = total_pnl - total_factor_pnl
            else:
                # Fallback: simple beta attribution
                factor_pnl = {}
                beta_pnl = 0.0
                alpha = total_pnl
            
            # Sector attribution
            sector_allocation_pnl = self._attribute_sector_pnl(
                positions, benchmark_returns
            )
            
            # Idiosyncratic = residual after all factors
            idiosyncratic_pnl = total_pnl - sum(factor_pnl.values()) - sector_allocation_pnl
            
            return AttributionResult(
                date=date,
                total_pnl=total_pnl,
                alpha=alpha,
                beta_pnl=beta_pnl,
                sector_allocation_pnl=sector_allocation_pnl,
                factor_pnl=factor_pnl,
                idiosyncratic_pnl=idiosyncratic_pnl,
                r_squared=self._factor_model.get('r_squared', 0.0) if self._factor_model else 0.0,
            )
    
    def _attribute_sector_pnl(
        self,
        positions: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> float:
        """Calculate sector allocation PnL using Brinson framework"""
        if positions.empty or 'sector' not in positions.columns:
            return 0.0
        
        # Portfolio sector weights
        total_value = (positions['quantity'] * positions['price']).sum()
        if total_value == 0:
            return 0.0
        
        positions['value'] = positions['quantity'] * positions['price']
        positions['weight'] = positions['value'] / total_value
        
        portfolio_sector_weights = positions.groupby('sector')['weight'].sum()
        
        # Benchmark sector weights (would need benchmark composition data)
        # For now, assume equal weight or use benchmark proxy
        benchmark_sector_weights = portfolio_sector_weights * 0.5  # Placeholder
        
        # Sector returns from benchmark
        sector_returns = {}
        for sector in portfolio_sector_weights.index:
            # Get sector ETF or proxy returns
            sector_returns[sector] = 0.0  # Placeholder
        
        # Brinson: Allocation effect = sum((wp - wb) * (rb - rp))
        # where wp = portfolio weight, wb = benchmark weight
        # rb = sector return, rp = portfolio return
        
        allocation_effect = 0.0
        for sector in portfolio_sector_weights.index:
            wp = portfolio_sector_weights.get(sector, 0)
            wb = benchmark_sector_weights.get(sector, 0)
            rb = sector_returns.get(sector, 0)
            rp = 0.0  # Portfolio return placeholder
            
            allocation_effect += (wp - wb) * (rb - rp)
        
        return allocation_effect
    
    def get_factor_loadings(self) -> Dict[str, float]:
        """Get latest factor loadings"""
        return self._factor_loadings.copy()
    
    def get_model_diagnostics(self) -> Dict:
        """Get factor model diagnostics"""
        return self._factor_model.copy() if self._factor_model else {}


def run_attribution_backtest(
    portfolio_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    positions_history: Dict[str, pd.DataFrame],
    benchmark_returns: pd.Series,
    sector_map: Dict[str, str],
    lookback_days: int = 252,
) -> pd.DataFrame:
    """
    Run full attribution backtest over historical period.
    
    Args:
        portfolio_returns: DataFrame (date x symbol) of daily returns
        factor_returns: DataFrame (date x factor) of daily factor returns
        positions_history: Dict mapping date -> positions DataFrame
        benchmark_returns: Series of benchmark daily returns
        sector_map: Dict mapping symbol -> sector
        lookback_days: Lookback window for factor model
        
    Returns:
        DataFrame with daily attribution results
    """
    attributor = PnLAttribution(
        factor_data=factor_returns,
        sector_map=sector_map,
    )
    
    results = []
    dates = portfolio_returns.index
    
    for date in dates:
        try:
            daily_returns = portfolio_returns.loc[date]
            positions = positions_history.get(str(date.date()), pd.DataFrame())
            factor_rets = factor_returns.loc[:date]
            bench_ret = benchmark_returns.loc[:date]
            
            result = attributor.attribute_daily_pnl(
                date=str(date.date()),
                portfolio_returns=daily_returns,
                positions=positions,
                factor_returns=factor_rets,
                benchmark_returns=bench_ret,
            )
            results.append(result.to_dict())
        except Exception as e:
            logger.warning(f"Attribution failed for {date}: {e}")
    
    return pd.DataFrame(results)


def compute_factor_returns(
    price_data: pd.DataFrame,
    factor_definitions: Dict[str, List[str]],
    method: str = "long_short",
) -> pd.DataFrame:
    """
    Compute factor returns from price data.
    
    Args:
        price_data: DataFrame (date x symbol) of prices
        factor_definitions: Dict mapping factor_name -> list of symbols (long leg)
                           or Dict with 'long' and 'short' keys
        method: 'long_short' or 'long_only'
        
    Returns:
        DataFrame (date x factor) of daily factor returns
    """
    returns = price_data.pct_change()
    factor_returns = pd.DataFrame(index=returns.index)
    
    for factor_name, symbols in factor_definitions.items():
        if isinstance(symbols, dict):
            long_symbols = symbols.get('long', [])
            short_symbols = symbols.get('short', [])
        else:
            long_symbols = symbols
            short_symbols = []
        
        # Filter valid symbols
        long_symbols = [s for s in long_symbols if s in returns.columns]
        short_symbols = [s for s in short_symbols if s in returns.columns]
        
        if not long_symbols:
            continue
        
        # Long leg
        long_rets = returns[long_symbols].mean(axis=1)
        
        if short_symbols:
            short_rets = returns[short_symbols].mean(axis=1)
            if method == "long_short":
                factor_ret = long_rets - short_rets
            else:
                factor_ret = long_rets
        else:
            factor_ret = long_rets
        
        factor_returns[factor_name] = factor_ret
    
    return factor_returns.dropna()