# Base strategy class for trading strategies
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, List
import pandas as pd

@dataclass
class Signal:
    """Trading signal"""
    timestamp: pd.Timestamp
    action: str  # 'BUY', 'SELL', 'HOLD'
    quantity: Optional[float] = None  # None = full position, otherwise specific quantity
    price: Optional[float] = None
    symbol: str = 'ASSET'  # Symbol to trade
    reason: Optional[str] = None
    metadata: Dict = None

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, params: Dict):
        self.params = params
        self.initialized = False
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on data
        
        Args:
            data: OHLCV dataframe with indicators
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate required indicators
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        pass
    
    def warmup_period(self) -> int:
        """Return minimum bars needed before generating signals"""
        return 0
    
    def validate_params(self) -> bool:
        """Validate strategy parameters"""
        return True