# Walk-forward optimization implementation
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .engine import BacktestEngine, BacktestConfig

class WalkForwardOptimizer:
    """Implement walk-forward optimization"""
    
    def __init__(
        self,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,    # 3 months
        step_days: int = 63            # Move forward 3 months
    ):
        self.train_period = train_period_days
        self.test_period = test_period_days
        self.step = step_days
        
    def optimize(
        self,
        strategy_class,
        data: pd.DataFrame,
        param_grid: Dict,
        config: BacktestConfig
    ) -> pd.DataFrame:
        """
        Perform walk-forward optimization
        
        Args:
            strategy_class: Strategy class to test
            data: Historical data
            param_grid: Parameters to optimize
            config: Backtest configuration
            
        Returns:
            DataFrame with results for each window
        """
        results = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date
        
        while current_date + timedelta(days=self.train_period + self.test_period) <= end_date:
            # Define periods
            train_start = current_date
            train_end = current_date + timedelta(days=self.train_period)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period)
            
            # Optimize on training period
            best_params = self._optimize_period(
                strategy_class,
                data,
                param_grid,
                config,
                train_start,
                train_end
            )
            
            # Test on test period
            strategy = strategy_class(best_params)
            engine = BacktestEngine(config)
            test_result = engine.run(
                strategy,
                data,
                start_date=test_start,
                end_date=test_end
            )
            
            results.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'best_params': best_params,
                'test_return': test_result.metrics['total_return'],
                'test_sharpe': test_result.metrics['sharpe_ratio'],
                'test_max_dd': test_result.metrics['max_drawdown']
            })
            
            # Move to next window
            current_date += timedelta(days=self.step)
        
        return pd.DataFrame(results)
    
    def _optimize_period(
        self,
        strategy_class,
        data: pd.DataFrame,
        param_grid: Dict,
        config: BacktestConfig,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Optimize parameters for a single period"""
        best_sharpe = -np.inf
        best_params = None
        
        # Grid search
        param_combinations = self._generate_param_combinations(param_grid)
        
        for params in param_combinations:
            strategy = strategy_class(params)
            engine = BacktestEngine(config)
            result = engine.run(strategy, data, start_date, end_date)
            
            if result.metrics['sharpe_ratio'] > best_sharpe:
                best_sharpe = result.metrics['sharpe_ratio']
                best_params = params
        
        return best_params
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all parameter combinations"""
        from itertools import product
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = []
        
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations