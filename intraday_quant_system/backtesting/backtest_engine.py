import pandas as pd
import numpy as np
import vectorbt as vbt
import logging
from dataclasses import dataclass
from deployment.config import get_config, TransactionCosts

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    sharpe: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    trade_count: int
    stats: pd.Series
    trades: pd.DataFrame
    equity_curve: pd.Series
    regime_breakdown: dict

class BacktestEngine:
    """
    VectorBT-based Backtesting Engine.
    
    Production fixes:
      - Uses rigorous transaction cost model matching NSE intraday
      - Explicitly enforces `.shift(1)` to guarantee no lookahead bias
    """
    def __init__(self, config: dict = None):
        self.config = config or {}
        # Load transaction costs from central config
        tc_config = get_config().transaction_costs
        self.transaction_costs = tc_config
        
        # We model costs as a flat percentage of turnover for VectorBT simplicity,
        # calculated assuming a round trip.
        self.estimated_slippage = tc_config.estimated_slippage_pct
        self.round_trip_cost = tc_config.total_round_trip_pct()

    def run(self, df: pd.DataFrame, signals: pd.Series, initial_capital: float = 100000.0) -> BacktestResult:
        """
        Run backtest on a single symbol.
        
        signals: +1 (buy), -1 (sell short), 0 (flat)
        """
        if df.empty or signals.empty:
            logger.warning("Empty data provided for backtest")
            return None
            
        # CRITICAL: Shift signals by 1 to prevent lookahead bias!
        # If signal is generated at close of bar N, we enter at open of bar N+1
        execution_signals = signals.shift(1).fillna(0)
        
        # Convert to boolean entries/exits for vectorbt
        entries = execution_signals == 1
        exits = execution_signals == 0  # Close long when signal is 0 or -1
        short_entries = execution_signals == -1
        short_exits = execution_signals == 0 # Close short when signal is 0 or 1
        
        # Use open price for execution since signal was generated at previous close
        execution_price = df['open']
        
        logger.info(f"Running backtest with {self.round_trip_cost:.4%} assumed round-trip costs")
        
        try:
            # Build portfolio
            pf = vbt.Portfolio.from_signals(
                execution_price,
                entries,
                exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=initial_capital,
                fees=self.round_trip_cost / 2,  # vbt applies fee per trade leg
                slippage=self.estimated_slippage,
                freq='5min'  # Assume 5-min bars
            )
            
            stats = pf.stats()
            trades = pf.trades.records_readable
            
            # Calculate win rate and profit factor
            win_rate = 0.0
            profit_factor = 0.0
            
            if not trades.empty and 'PnL' in trades.columns:
                wins = trades[trades['PnL'] > 0]
                losses = trades[trades['PnL'] < 0]
                
                win_rate = len(wins) / len(trades)
                
                gross_profit = wins['PnL'].sum() if not wins.empty else 0
                gross_loss = abs(losses['PnL'].sum()) if not losses.empty else 0
                
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
            # Collect regime breakdown if regime data exists
            regime_breakdown = {}
            if 'regime' in df.columns and not trades.empty:
                # Map trades to regime at entry time
                try:
                    trades_with_regime = trades.copy()
                    # entry_idx maps to df index
                    regimes = []
                    for idx in trades_with_regime['Entry Index']:
                        if hasattr(df.index, 'get_loc') and idx in df.index:
                            regimes.append(df.loc[idx, 'regime'])
                        else:
                            regimes.append('unknown')
                            
                    trades_with_regime['regime'] = regimes
                    
                    for regime in trades_with_regime['regime'].unique():
                        regime_trades = trades_with_regime[trades_with_regime['regime'] == regime]
                        r_wins = regime_trades[regime_trades['PnL'] > 0]
                        
                        regime_breakdown[regime] = {
                            'trade_count': len(regime_trades),
                            'win_rate': len(r_wins) / len(regime_trades) if len(regime_trades) > 0 else 0,
                            'pnl': regime_trades['PnL'].sum()
                        }
                except Exception as e:
                    logger.error(f"Failed to calculate regime breakdown: {e}")
            
            result = BacktestResult(
                sharpe=stats.get('Sharpe Ratio', 0.0),
                max_drawdown=stats.get('Max Drawdown [%]', 0.0) / 100.0,
                total_return=stats.get('Total Return [%]', 0.0) / 100.0,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trade_count=len(trades),
                stats=stats,
                trades=trades,
                equity_curve=pf.value(),
                regime_breakdown=regime_breakdown
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None
