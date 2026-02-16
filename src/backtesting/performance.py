# Performance analysis
import pandas as pd
import numpy as np
from typing import Dict

class PerformanceAnalyzer:
    """Calculate performance metrics"""
    
    def calculate_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics
        
        Args:
            equity_curve: DataFrame with timestamp and equity columns
            
        Returns:
            Dictionary of metrics
        """
        equity = equity_curve['equity'].values
        returns = pd.Series(equity).pct_change().dropna()
        
        metrics = {
            # Returns
            'total_return': (equity[-1] / equity[0] - 1) * 100,
            'annualized_return': self._annualized_return(returns),
            'avg_return': returns.mean() * 100,
            
            # Risk
            'volatility': returns.std() * np.sqrt(252) * 100,
            'max_drawdown': self._max_drawdown(equity),
            'sharpe_ratio': self._sharpe_ratio(returns),
            'sortino_ratio': self._sortino_ratio(returns),
            'calmar_ratio': self._calmar_ratio(returns, equity),
            
            # Trade statistics
            'num_trades': len(returns),
            'win_rate': (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0,
            'avg_win': returns[returns > 0].mean() * 100 if (returns > 0).any() else 0,
            'avg_loss': returns[returns < 0].mean() * 100 if (returns < 0).any() else 0,
            'profit_factor': self._profit_factor(returns),
            
            # Time-based
            'best_day': returns.max() * 100,
            'worst_day': returns.min() * 100,
            
            # Final values
            'final_equity': equity[-1],
            'starting_equity': equity[0]
        }
        
        return metrics
    
    def _annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        total_days = len(returns)
        if total_days == 0:
            return 0
        total_return = (1 + returns).prod()
        years = total_days / 252
        return (total_return ** (1/years) - 1) * 100 if years > 0 else 0
    
    def _max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return drawdown.min() * 100
    
    def _sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        if excess_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calmar_ratio(self, returns: pd.Series, equity: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        ann_return = self._annualized_return(returns)
        max_dd = abs(self._max_drawdown(equity))
        return ann_return / max_dd if max_dd != 0 else 0
    
    def _profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return wins / losses if losses != 0 else 0