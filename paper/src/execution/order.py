from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    symbol: str
    order_type: OrderType
    side: str  # 'BUY' or 'SELL'
    quantity: float
    timestamp: datetime
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None
    
    def fill(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Mark the order as filled at specified price and time."""
        self.status = OrderStatus.FILLED
        self.filled_price = price
        self.filled_timestamp = timestamp or datetime.now()
        
    def cancel(self) -> None:
        """Cancel the order if it's still pending."""
        if self.status == OrderStatus.PENDING:
            self.status = OrderStatus.CANCELLED