from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class Trade:
    timestamp: datetime
    quantity: float
    price: float
    side: str  # 'BUY' or 'SELL'
    fees: float = 0.0

class Position:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.trades: List[Trade] = []
        self.average_entry = 0.0
        
    def add_trade(self, trade: Trade) -> None:
        """Add a trade and update position details."""
        self.trades.append(trade)
        
        if trade.side == 'BUY':
            new_quantity = self.quantity + trade.quantity
            # Update average entry price
            self.average_entry = (
                (self.average_entry * self.quantity + trade.price * trade.quantity) 
                / new_quantity if new_quantity > 0 else 0
            )
            self.quantity = new_quantity
        else:  # SELL
            self.quantity -= trade.quantity
            
    def realized_pnl(self) -> float:
        """Calculate realized P&L for the position."""
        pnl = 0.0
        running_quantity = 0
        for trade in self.trades:
            if trade.side == 'BUY':
                running_quantity += trade.quantity
            else:  # SELL
                pnl += (trade.price - self.average_entry) * trade.quantity
                running_quantity -= trade.quantity
        return pnl
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L based on current market price."""
        return (current_price - self.average_entry) * self.quantity