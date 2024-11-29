"""
Moving Average Crossover Strategy
"""
import pandas as pd
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    def __init__(self, short_period, long_period):
        super().__init__()
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self, data):
        signals = pd.Series(index=data.index, data=0)
        signals[data[f'MA_{self.short_period}'] > data[f'MA_{self.long_period}']] = 1
        signals[data[f'MA_{self.short_period}'] < data[f'MA_{self.long_period}']] = -1
        return signals