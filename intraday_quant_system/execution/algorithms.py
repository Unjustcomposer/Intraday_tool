import logging
import math
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExecutionAlgorithm:
    def __init__(self, symbol: str, total_quantity: int, side: str, start_time: datetime, end_time: datetime, **kwargs):
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.side = side
        self.start_time = start_time
        self.end_time = end_time
        self.executed_quantity = 0
        self.is_active = True
        self.params = kwargs

    def get_next_slice(self, current_time: datetime, market_data: Dict[str, Any] = None) -> int:
        raise NotImplementedError
        
class TWAP(ExecutionAlgorithm):
    """
    Time-Weighted Average Price algorithm.
    Executes trades in equal slices over the specified time window.
    """
    def __init__(self, symbol: str, total_quantity: int, side: str, start_time: datetime, end_time: datetime, slices: int = 10, **kwargs):
        super().__init__(symbol, total_quantity, side, start_time, end_time, **kwargs)
        self.slices = slices
        self.slice_quantity = total_quantity // slices
        self.interval = (end_time - start_time) / slices
        self.next_execution = start_time
        
    def get_next_slice(self, current_time: datetime, market_data: Dict[str, Any] = None) -> int:
        if not self.is_active or current_time > self.end_time:
            self.is_active = False
            remaining = self.total_quantity - self.executed_quantity
            if remaining > 0:
                self.executed_quantity += remaining
                return remaining
            return 0
            
        if current_time >= self.next_execution:
            self.next_execution += self.interval
            qty = self.slice_quantity
            remaining = self.total_quantity - self.executed_quantity
            if qty > remaining:
                qty = remaining
            self.executed_quantity += qty
            if self.executed_quantity >= self.total_quantity:
                self.is_active = False
            return qty
        return 0

class VWAP(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price algorithm.
    Executes trades in proportion to historical or predicted volume profiles.
    """
    def __init__(self, symbol: str, total_quantity: int, side: str, start_time: datetime, end_time: datetime, volume_profile: List[float] = None, **kwargs):
        super().__init__(symbol, total_quantity, side, start_time, end_time, **kwargs)
        # Default flat volume profile if none provided (degenerates to TWAP)
        self.volume_profile = volume_profile or [1.0/10] * 10
        self.total_intervals = len(self.volume_profile)
        self.interval_duration = (end_time - start_time) / self.total_intervals
        self.current_interval = 0
        self.next_execution = start_time
        
    def get_next_slice(self, current_time: datetime, market_data: Dict[str, Any] = None) -> int:
        if not self.is_active or current_time > self.end_time or self.current_interval >= self.total_intervals:
            self.is_active = False
            remaining = self.total_quantity - self.executed_quantity
            if remaining > 0:
                self.executed_quantity += remaining
                return remaining
            return 0
            
        if current_time >= self.next_execution:
            fraction = self.volume_profile[self.current_interval]
            qty = int(self.total_quantity * fraction)
            self.next_execution += self.interval_duration
            self.current_interval += 1
            
            remaining = self.total_quantity - self.executed_quantity
            if qty > remaining:
                qty = remaining
            self.executed_quantity += qty
            if self.executed_quantity >= self.total_quantity:
                self.is_active = False
            return qty
        return 0

class AlmgrenChriss(ExecutionAlgorithm):
    """
    Almgren-Chriss optimal execution algorithm.
    Balances market impact and variance (shortfall risk).
    """
    def __init__(self, symbol: str, total_quantity: int, side: str, start_time: datetime, end_time: datetime, risk_aversion: float = 1e-6, intervals: int = 10, **kwargs):
        super().__init__(symbol, total_quantity, side, start_time, end_time, **kwargs)
        self.risk_aversion = risk_aversion
        self.intervals = intervals
        self.interval_duration = (end_time - start_time) / intervals
        self.current_interval = 0
        self.next_execution = start_time
        self.trajectory = self._calculate_trajectory()

    def _calculate_trajectory(self):
        kappa = math.sqrt(self.risk_aversion * 1e4)
        if kappa == 0:
            return [self.total_quantity / self.intervals] * self.intervals
            
        times = [i / self.intervals for i in range(self.intervals + 1)]
        T = 1.0
        trajectory = []
        for i in range(1, self.intervals + 1):
            t_prev = times[i-1]
            t = times[i]
            x_prev = self.total_quantity * math.sinh(kappa * (T - t_prev)) / math.sinh(kappa * T)
            x_curr = self.total_quantity * math.sinh(kappa * (T - t)) / math.sinh(kappa * T)
            trajectory.append(x_prev - x_curr)
        return trajectory

    def get_next_slice(self, current_time: datetime, market_data: Dict[str, Any] = None) -> int:
        if not self.is_active or current_time > self.end_time or self.current_interval >= self.intervals:
            self.is_active = False
            remaining = self.total_quantity - self.executed_quantity
            if remaining > 0:
                self.executed_quantity += remaining
                return remaining
            return 0
            
        if current_time >= self.next_execution:
            qty = int(self.trajectory[self.current_interval])
            self.next_execution += self.interval_duration
            self.current_interval += 1
            
            remaining = self.total_quantity - self.executed_quantity
            if qty > remaining:
                qty = remaining
            self.executed_quantity += qty
            if self.executed_quantity >= self.total_quantity:
                self.is_active = False
            return qty
        return 0

class Iceberg(ExecutionAlgorithm):
    """
    Iceberg slicing strategy. Hides total size by showing only a peak size.
    """
    def __init__(self, symbol: str, total_quantity: int, side: str, start_time: datetime, end_time: datetime, peak_size: int = 100, interval_seconds: int = 30, **kwargs):
        super().__init__(symbol, total_quantity, side, start_time, end_time, **kwargs)
        self.peak_size = peak_size
        self.interval = timedelta(seconds=interval_seconds)
        self.next_execution = start_time
        
    def get_next_slice(self, current_time: datetime, market_data: Dict[str, Any] = None) -> int:
        if not self.is_active or current_time > self.end_time:
            self.is_active = False
            remaining = self.total_quantity - self.executed_quantity
            if remaining > 0:
                self.executed_quantity += remaining
                return remaining
            return 0
            
        if current_time >= self.next_execution:
            self.next_execution += self.interval
            remaining = self.total_quantity - self.executed_quantity
            qty = min(self.peak_size, remaining)
            self.executed_quantity += qty
            if self.executed_quantity >= self.total_quantity:
                self.is_active = False
            return qty
        return 0
