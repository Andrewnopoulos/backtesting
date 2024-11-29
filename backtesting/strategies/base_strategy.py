"""
Base class for trading strategies
"""
class BaseStrategy:
    def __init__(self):
        self.position = 0
        self.positions = []
        
    def generate_signals(self, data):
        raise NotImplementedError("Subclass must implement generate_signals()")