"""
Trading Strategies Module
"""
from .base import BaseStrategy, Signal
from .ma_crossover import MovingAverageCrossover

__all__ = ['BaseStrategy', 'Signal', 'MovingAverageCrossover']
