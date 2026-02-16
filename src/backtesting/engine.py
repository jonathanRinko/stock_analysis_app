# Backtesting engine for simulating trading strategies
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from ..strategies.base import BaseStrategy, Signal
from .broker import SimulatedBroker
from .portfolio import Portfolio
from .performance import PerformanceAnalyzer

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005   # 0.05%
    position_size: float = 1.0  # Fraction of capital per trade
    max_positions: int = 10
    
@dataclass
class BacktestResult:
    """Backtesting results"""
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: Dict
    signals: List[Signal]
    portfolio_history: pd.DataFrame

class BacktestEngine:
    """Core backtesting engine"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.broker = SimulatedBroker(
            commission=config.commission,
            slippage=config.slippage
        )
        self.portfolio = Portfolio(initial_capital=config.initial_capital)
        self.performance = PerformanceAnalyzer()
        
    def run(
        self, 
        strategy: BaseStrategy, 
        data: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run backtest
        
        Args:
            strategy: Trading strategy instance
            data: Historical OHLCV data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            BacktestResult object
        """
        # Filter data by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Calculate indicators
        data = strategy.calculate_indicators(data)
        
        # Skip warmup period
        warmup = strategy.warmup_period()
        if warmup > 0:
            data = data.iloc[warmup:]
        
        # Initialize tracking
        all_signals = []
        equity_curve = []
        portfolio_history = []
        
        # Main backtest loop
        for i in range(len(data)):
            current_data = data.iloc[:i+1]
            current_bar = data.iloc[i]
            timestamp = current_bar.name
            
            # Generate signals
            signals = strategy.generate_signals(current_data)
            all_signals.extend(signals)
            
            # Process signals
            for signal in signals:
                if signal.action == 'BUY':
                    self._execute_buy(signal, current_bar)
                elif signal.action == 'SELL':
                    self._execute_sell(signal, current_bar)
            
            # Update portfolio value
            self.portfolio.update_market_value(
                current_bar['close'],
                timestamp
            )
            
            # Record state
            equity_curve.append({
                'timestamp': timestamp,
                'equity': self.portfolio.total_value,
                'cash': self.portfolio.cash,
                'positions_value': self.portfolio.positions_value
            })
            
            portfolio_history.append(self._get_portfolio_snapshot(timestamp))
        
        # Compile results
        result = BacktestResult(
            trades=pd.DataFrame(self.portfolio.closed_trades),
            equity_curve=pd.DataFrame(equity_curve),
            signals=all_signals,
            portfolio_history=pd.DataFrame(portfolio_history),
            metrics=self._calculate_metrics(pd.DataFrame(equity_curve))
        )
        
        return result
    
    def _execute_buy(self, signal: Signal, bar: pd.Series):
        """Execute buy signal"""
        price = self.broker.simulate_fill(
            signal.price or bar['close'],
            'buy',
            bar
        )

        # Calculate position size
        # If quantity is None, use all available capital based on position_size config
        # Must account for commission in the calculation
        if signal.quantity is None:
            # Calculate max quantity we can afford including commission
            # Formula: quantity * price * (1 + commission_rate) <= available_cash
            # Therefore: quantity <= available_cash / (price * (1 + commission_rate))
            available_cash = self.portfolio.cash * self.config.position_size
            quantity = available_cash / (price * (1 + self.broker.commission_rate))
        else:
            quantity = signal.quantity

        commission = self.broker.calculate_commission(quantity * price)

        # Check if we can afford it (should always be true now with proper calculation)
        total_cost = quantity * price + commission
        if total_cost <= self.portfolio.cash:
            self.portfolio.open_position(
                symbol=signal.symbol,
                quantity=quantity,
                price=price,
                timestamp=signal.timestamp,
                commission=commission
            )
    
    def _execute_sell(self, signal: Signal, bar: pd.Series):
        """Execute sell signal"""
        symbol = signal.symbol

        if not self.portfolio.has_position(symbol):
            return

        price = self.broker.simulate_fill(
            signal.price or bar['close'],
            'sell',
            bar
        )

        # If quantity is None, close the full position
        # Otherwise use the specified quantity
        if signal.quantity is None:
            quantity = self.portfolio.get_position_size(symbol)
        else:
            quantity = signal.quantity

        commission = self.broker.calculate_commission(quantity * price)

        self.portfolio.close_position(
            symbol=symbol,
            quantity=quantity,
            price=price,
            timestamp=signal.timestamp,
            commission=commission
        )
    
    def _get_portfolio_snapshot(self, timestamp: pd.Timestamp) -> Dict:
        """Get current portfolio state"""
        return {
            'timestamp': timestamp,
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'num_positions': len(self.portfolio.positions),
            'positions': dict(self.portfolio.positions)
        }
    
    def _calculate_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate performance metrics"""
        return self.performance.calculate_metrics(equity_curve)