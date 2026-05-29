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
        exits = execution_signals <= 0      # Close long when signal is 0 (flat) or -1 (short)
        short_entries = execution_signals == -1
        short_exits = execution_signals >= 0  # Close short when signal is 0 (flat) or +1 (long)
        
        # Use VWAP for execution to better model intra-bar execution. Fallback to open if missing.
        execution_price = df['vwap'] if 'vwap' in df.columns else df['open']
        
        logger.info(f"Running backtest with {self.round_trip_cost:.4%} assumed round-trip costs")
        
        try:
            # Pre-calculate asymmetric Almgren-Chriss slippage array
            # ADV is approximated via a rolling 20-period volume.
            # k=0.1, asymmetry=1.5 for long, 0.8 for short.
            adv = df['volume'].rolling(window=20, min_periods=1).mean().fillna(1000)
            order_size = initial_capital / df['close'] # Approximation of share size
            participation_rate = (order_size / adv).clip(upper=1.0)
            
            # Slippage for buys (long entries and short exits)
            buy_slippage = (0.1 * np.sqrt(participation_rate) * 1.5).clip(upper=0.01)
            # Slippage for sells (short entries and long exits)
            sell_slippage = (0.1 * np.sqrt(participation_rate) * 0.8).clip(upper=0.01)
            
            # Institutional Upgrade: Adverse Selection (Queue Position) Penalty
            # Simulates paying the spread crossing + getting filled at the back of the queue
            queue_penalty = 0.0005 # 5 bps static penalty for queue disadvantage
            
            # Since vectorbt takes a single slippage parameter, we use the average, 
            # combined with the static queue penalty for extreme realism.
            combined_slippage = ((buy_slippage + sell_slippage) / 2.0) + queue_penalty
            
            # Build portfolio
            pf = vbt.Portfolio.from_signals(
                execution_price,
                entries,
                exits,
                short_entries=short_entries,
                short_exits=short_exits,
                init_cash=initial_capital,
                fees=self.round_trip_cost / 2,  # vbt applies fee per trade leg
                slippage=combined_slippage,
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
