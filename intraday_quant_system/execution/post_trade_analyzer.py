import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PostTradeAnalyzer:
    """
    PostTradeAnalyzer: Analyzes execution quality.
    Computes Implementation Shortfall (IS), actual vs. predicted slippage,
    and fill statistics by order type.
    """
    def __init__(self):
        self.trades: List[Dict[str, Any]] = []

    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: int,
        decision_price: float,
        execution_price: float,
        predicted_slippage: float,
        order_type: str,
        filled_status: str  # 'FILLED_LIMIT', 'CONVERTED_MARKET', 'CANCELLED'
    ):
        """
        Record a trade for post-execution analysis.
        """
        if execution_price <= 0 or decision_price <= 0:
            return
            
        # Implementation Shortfall (bps)
        # IS_bps = side_multiplier * (exec_price - decision_price) / decision_price * 10000
        side_mult = 1.0 if side.upper() == "BUY" else -1.0
        is_val = side_mult * (execution_price - decision_price)
        is_bps = (is_val / decision_price) * 10000.0
        
        # Realized Slippage (bps)
        actual_slippage = abs(execution_price - decision_price) / decision_price
        actual_slippage_bps = actual_slippage * 10000.0
        pred_slippage_bps = predicted_slippage * 10000.0
        
        trade_record = {
            'symbol': symbol,
            'side': side.upper(),
            'qty': qty,
            'decision_price': decision_price,
            'execution_price': execution_price,
            'implementation_shortfall_bps': is_bps,
            'actual_slippage_bps': actual_slippage_bps,
            'predicted_slippage_bps': pred_slippage_bps,
            'order_type': order_type.upper(),
            'filled_status': filled_status.upper()
        }
        
        self.trades.append(trade_record)
        logger.info(f"Logged trade for {symbol}: IS={is_bps:.1f} bps, Slippage={actual_slippage_bps:.1f} bps (vs Pred={pred_slippage_bps:.1f} bps), Fill={filled_status}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics across all logged trades.
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'mean_is_bps': 0.0,
                'mean_slippage_bps': 0.0,
                'mean_predicted_slippage_bps': 0.0,
                'fill_rates': {}
            }
            
        df = pd.DataFrame(self.trades)
        
        # Compute fill rates by status
        total_orders = len(df)
        fill_counts = df['filled_status'].value_counts()
        fill_rates = {status: float(count / total_orders) for status, count in fill_counts.items()}
        
        # Compute statistics
        stats = {
            'total_trades': total_orders,
            'mean_is_bps': float(df['implementation_shortfall_bps'].mean()),
            'mean_slippage_bps': float(df['actual_slippage_bps'].mean()),
            'mean_predicted_slippage_bps': float(df['predicted_slippage_bps'].mean()),
            'slippage_bias_bps': float((df['actual_slippage_bps'] - df['predicted_slippage_bps']).mean()),
            'fill_rates': fill_rates
        }
        
        logger.info(f"Post-Trade Performance Summary: {stats['total_trades']} trades, Mean IS: {stats['mean_is_bps']:.1f} bps, Mean Slip: {stats['mean_slippage_bps']:.1f} bps")
        return stats
        
    def save_reports(self, filepath: str):
        """Save trade log to CSV/Parquet"""
        if not self.trades:
            return
        df = pd.DataFrame(self.trades)
        try:
            df.to_csv(filepath, index=False)
            logger.info(f"Saved post-trade performance reports to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save post-trade report: {e}")
