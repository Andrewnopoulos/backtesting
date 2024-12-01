# src/execution/executor.py
from datetime import datetime
from typing import Dict, List, Optional
from .order import Order, OrderType, OrderStatus
from ..portfolio.portfolio import Portfolio
from ..risk.risk_manager import RiskManager
import logging

logger = logging.getLogger(__name__)

class OrderExecutor:
    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.pending_orders: List[Order] = []
        self.filled_orders: List[Order] = []
        self.cancelled_orders: List[Order] = []
        
    def place_market_order(
        self, 
        symbol: str, 
        quantity: float, 
        side: str,
        current_prices: Dict[str, float]  # Add this parameter
    ) -> Optional[Order]:
        """
        Place a market order for immediate execution.
        Returns the created order if successful, None if rejected.
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            timestamp=datetime.now()
        )
        
        # Check risk limits with current prices
        if not self.risk_manager.check_order_risk(order, current_prices):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected due to risk limits: {order}")
            return None
            
        self.pending_orders.append(order)
        return order
        
    def place_limit_order(
        self, 
        symbol: str, 
        quantity: float, 
        side: str, 
        limit_price: float,
        current_prices: Dict[str, float]  # Add this parameter
    ) -> Optional[Order]:
        """
        Place a limit order to be executed when price conditions are met.
        Returns the created order if successful, None if rejected.
        """
        order = Order(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            side=side,
            quantity=quantity,
            timestamp=datetime.now(),
            limit_price=limit_price
        )
        
        # Check risk limits
        if not self.risk_manager.check_order_risk(order, current_prices):
            order.status = OrderStatus.REJECTED
            logger.warning(f"Order rejected due to risk limits: {order}")
            return None
            
        self.pending_orders.append(order)
        return order
        
    def cancel_order(self, order: Order) -> bool:
        """
        Cancel a pending order.
        Returns True if cancelled successfully, False otherwise.
        """
        if order not in self.pending_orders:
            return False
            
        order.cancel()
        self.pending_orders.remove(order)
        self.cancelled_orders.append(order)
        logger.info(f"Order cancelled: {order}")
        return True
        
    def process_market_data(self, market_data: Dict[str, float]) -> None:
        """
        Process current market data and handle order execution.
        market_data: Dictionary of symbol -> current price
        """
        for order in self.pending_orders[:]:  # Create copy to allow modification during iteration
            if self._should_execute_order(order, market_data):
                self._execute_order(order, market_data[order.symbol])
                
    def _should_execute_order(self, order: Order, market_data: Dict[str, float]) -> bool:
        """Determine if an order should be executed based on its type and current market data."""
        if order.status != OrderStatus.PENDING:
            return False
            
        current_price = market_data.get(order.symbol)
        if current_price is None:
            return False
            
        if order.order_type == OrderType.MARKET:
            return True
            
        # For limit orders, check if price conditions are met
        if order.order_type == OrderType.LIMIT:
            if order.side == 'BUY':
                return current_price <= order.limit_price
            else:  # SELL
                return current_price >= order.limit_price
                
        return False
        
    def _execute_order(self, order: Order, execution_price: float) -> None:
        """Execute an order at the specified price."""
        # Try to add the trade to the portfolio
        success = self.portfolio.add_trade(
            symbol=order.symbol,
            quantity=order.quantity,
            price=execution_price,
            side=order.side,
            timestamp=datetime.now()
        )
        
        if success:
            order.fill(execution_price)
            self.pending_orders.remove(order)
            self.filled_orders.append(order)
            logger.info(f"Order executed: {order} at price {execution_price}")
        else:
            order.status = OrderStatus.REJECTED
            self.pending_orders.remove(order)
            logger.warning(f"Order execution failed: {order}")
            
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all pending orders, optionally filtered by symbol."""
        if symbol is None:
            return self.pending_orders
        return [order for order in self.pending_orders if order.symbol == symbol]
        
    def get_order_history(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all filled and cancelled orders, optionally filtered by symbol."""
        orders = self.filled_orders + self.cancelled_orders
        if symbol is None:
            return orders
        return [order for order in orders if order.symbol == symbol]