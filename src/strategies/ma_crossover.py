# Moving Average Crossover Strategy
from typing import Dict, List
import pandas as pd
import numpy as np
from .base import BaseStrategy, Signal

class MovingAverageCrossover(BaseStrategy):
    """Simple MA crossover strategy"""

    def __init__(self, params: Dict):
        """
        Args:
            params: Dictionary with 'fast_period' and 'slow_period'
        """
        super().__init__(params)
        self.fast_period = params.get('fast_period', 10)
        self.slow_period = params.get('slow_period', 30)
        self.last_position_state = None  # Track position state
        self.last_processed_timestamp = None  # Track last processed timestamp

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        df = data.copy()
        df['ma_fast'] = df['close'].rolling(self.fast_period).mean()
        df['ma_slow'] = df['close'].rolling(self.slow_period).mean()
        return df

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate crossover signals for the latest bar only"""
        # Need at least 2 bars and indicators must be calculated
        if len(data) < 2:
            return []

        current = data.iloc[-1]

        # Only process if we haven't already processed this timestamp
        if self.last_processed_timestamp == current.name:
            return []

        previous = data.iloc[-2]
        signals = []

        # Skip if indicators aren't ready
        if pd.isna(current['ma_fast']) or pd.isna(current['ma_slow']):
            self.last_processed_timestamp = current.name
            return []
        if pd.isna(previous['ma_fast']) or pd.isna(previous['ma_slow']):
            self.last_processed_timestamp = current.name
            return []

        # Check for bullish crossover (Fast MA crosses above Slow MA)
        if (previous['ma_fast'] <= previous['ma_slow'] and
            current['ma_fast'] > current['ma_slow']):

            # Only generate BUY signal if we're not already long
            if self.last_position_state != 'LONG':
                signals.append(Signal(
                    timestamp=current.name,
                    action='BUY',
                    quantity=None,  # Use all available capital
                    price=current['close'],
                    symbol='ASSET',
                    reason='Fast MA crossed above Slow MA'
                ))
                self.last_position_state = 'LONG'

        # Check for bearish crossover (Fast MA crosses below Slow MA)
        elif (previous['ma_fast'] >= previous['ma_slow'] and
              current['ma_fast'] < current['ma_slow']):

            # Only generate SELL signal if we're currently long
            if self.last_position_state == 'LONG':
                signals.append(Signal(
                    timestamp=current.name,
                    action='SELL',
                    quantity=None,  # Close full position
                    price=current['close'],
                    symbol='ASSET',
                    reason='Fast MA crossed below Slow MA'
                ))
                self.last_position_state = None

        # Update last processed timestamp
        self.last_processed_timestamp = current.name

        return signals

    def warmup_period(self) -> int:
        """Need slow_period bars before generating signals"""
        return self.slow_period