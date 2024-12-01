from typing import Dict, Optional
from .position import Position, Trade
from datetime import datetime

class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.initial_cash = initial_cash
        
    def add_trade(self, symbol: str, quantity: float, price: float, 
                  side: str, timestamp: Optional[datetime] = None) -> bool:
        """Process a new trade and update portfolio state."""
        if timestamp is None:
            timestamp = datetime.now()
            
        trade_value = quantity * price
        
        # Check if we have enough cash for buys
        if side == 'BUY' and trade_value > self.cash:
            return False
            
        # Create or get position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
            
        # Update cash
        if side == 'BUY':
            self.cash -= trade_value
        else:
            self.cash += trade_value
            
        # Add trade to position
        trade = Trade(timestamp, quantity, price, side)
        self.positions[symbol].add_trade(trade)
        
        # Remove position if quantity is 0
        if self.positions[symbol].quantity == 0:
            del self.positions[symbol]
            
        return True
        
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol if it exists."""
        return self.positions.get(symbol)
        
    def total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions."""
        position_value = sum(
            pos.quantity * current_prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value
        
    def realized_pnl(self) -> float:
        """Calculate total realized P&L across all positions."""
        return sum(pos.realized_pnl() for pos in self.positions.values())
        
    def unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L across all positions."""
        return sum(
            pos.unrealized_pnl(current_prices[symbol])
            for symbol, pos in self.positions.items()
        )