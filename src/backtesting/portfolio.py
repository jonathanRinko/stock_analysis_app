# Portfolio management for backtesting trading strategies
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd

@dataclass
class Position:
    """Open position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: pd.Timestamp
    entry_commission: float

@dataclass
class Trade:
    """Closed trade"""
    symbol: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    return_pct: float
    commission: float
    
class Portfolio:
    """Track portfolio state"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.positions_value = 0
        self.total_value = initial_capital
        
    def open_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp,
        commission: float
    ):
        """Open a new position"""
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=timestamp,
            entry_commission=commission
        )
        
        self.cash -= (quantity * price + commission)
        
    def close_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: pd.Timestamp,
        commission: float
    ):
        """Close a position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Calculate P&L
        proceeds = quantity * price
        cost = quantity * position.entry_price
        total_commission = position.entry_commission + commission
        pnl = proceeds - cost - total_commission
        return_pct = (proceeds - cost) / cost
        
        # Record trade
        trade = Trade(
            symbol=symbol,
            quantity=quantity,
            entry_price=position.entry_price,
            exit_price=price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=pnl,
            return_pct=return_pct,
            commission=total_commission
        )
        self.closed_trades.append(trade)
        
        # Update cash
        self.cash += (proceeds - commission)
        
        # Remove or reduce position
        if quantity >= position.quantity:
            del self.positions[symbol]
        else:
            position.quantity -= quantity
            
    def update_market_value(self, current_price: float, timestamp: pd.Timestamp):
        """Update portfolio market value"""
        self.positions_value = sum(
            pos.quantity * current_price 
            for pos in self.positions.values()
        )
        self.total_value = self.cash + self.positions_value
        
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in symbol"""
        return symbol in self.positions
    
    def get_position_size(self, symbol: str) -> float:
        """Get quantity of position"""
        return self.positions.get(symbol).quantity if symbol in self.positions else 0