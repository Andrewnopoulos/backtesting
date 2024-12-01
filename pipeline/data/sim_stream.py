import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import random

class SimulatedDataGenerator:
    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "AMZN"]
        self.prices = {symbol: 100.0 for symbol in self.symbols}  # Starting prices
        self.volatility = 0.002  # 0.2% volatility per tick
        
    def generate_trade(self) -> Dict:
        """Generate a single simulated trade"""
        symbol = random.choice(self.symbols)
        
        # Random walk price movement
        price_change = np.random.normal(0, self.volatility) * self.prices[symbol]
        self.prices[symbol] += price_change
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "price": round(self.prices[symbol], 2),
            "size": round(random.lognormvariate(3, 1)),  # Random trade size
            "trade_id": str(random.randint(1000000, 9999999)),
            "type": "trade"
        }

class SimulatedMarketStream:
    def __init__(self, 
                 symbols: List[str] = None,
                 data_frequency: float = 1.0,  # Data frequency in seconds
                 random_delay: bool = True):
        self.generator = SimulatedDataGenerator(symbols)
        self.data_frequency = data_frequency
        self.random_delay = random_delay
        self.running = False
        self.trade_handler = None
        
    def subscribe_trade_updates(self, handler):
        """Store the trade update handler"""
        self.trade_handler = handler
        
    async def _run_forever(self):
        """Run the simulated data stream"""
        self.running = True
        
        while self.running:
            try:
                # Generate and send trade data
                trade = self.generator.generate_trade()
                
                if self.trade_handler:
                    await self.trade_handler(trade)
                
                # Add random delay if enabled
                if self.random_delay:
                    delay = self.data_frequency * (0.5 + random.random())
                else:
                    delay = self.data_frequency
                    
                await asyncio.sleep(delay)
                
            except Exception as e:
                print(f"Error in simulated stream: {str(e)}")
                await asyncio.sleep(1)
                
    async def stop_ws(self):
        """Stop the simulated stream"""
        self.running = False