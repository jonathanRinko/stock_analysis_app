# Simulated broker for backtesting with realistic order execution
import pandas as pd
import numpy as np

class SimulatedBroker:
    """Simulates realistic order execution"""
    
    def __init__(self, commission: float = 0.001, slippage: float = 0.0005):
        self.commission_rate = commission
        self.slippage_rate = slippage
        
    def simulate_fill(
        self, 
        signal_price: float, 
        side: str,
        bar: pd.Series
    ) -> float:
        """
        Simulate realistic order fill
        
        Args:
            signal_price: Desired execution price
            side: 'buy' or 'sell'
            bar: Current OHLCV bar
            
        Returns:
            Actual fill price
        """
        # Add slippage
        if side == 'buy':
            slippage = signal_price * self.slippage_rate
            fill_price = signal_price + slippage
            
            # Can't buy below the low
            fill_price = max(fill_price, bar['low'])
            # Can't buy above the high
            fill_price = min(fill_price, bar['high'])
            
        else:  # sell
            slippage = signal_price * self.slippage_rate
            fill_price = signal_price - slippage
            
            # Can't sell below the low
            fill_price = max(fill_price, bar['low'])
            # Can't sell above the high
            fill_price = min(fill_price, bar['high'])
        
        return fill_price
    
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade"""
        return trade_value * self.commission_rate