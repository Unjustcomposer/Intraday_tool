from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

class BrokerClient(ABC):
    """
    Abstract Base Class for all Broker implementations (Fyers, Dhan, Zerodha, etc.).
    Ensures the execution engine and market data engine can swap brokers seamlessly.
    """
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the broker API."""
        pass
        
    @abstractmethod
    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Fetch historical candle data.
        Must return a DataFrame with columns: [open, high, low, close, volume]
        """
        pass
        
    @abstractmethod
    def place_order(self, symbol: str, quantity: int, side: str, order_type: str, price: float = 0.0) -> Optional[str]:
        """
        Place an order.
        Returns the broker's order_id if successful, None otherwise.
        """
        pass
        
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID."""
        pass
        
    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        """
        Get current open positions.
        Must return a DataFrame with columns: [symbol, quantity, average_price]
        """
        pass
        
    @abstractmethod
    def get_order_book(self) -> pd.DataFrame:
        """Get list of today's orders and their statuses."""
        pass
