# src/risk/risk_manager.py
from dataclasses import dataclass
from typing import Dict, Optional
from ..portfolio.portfolio import Portfolio
from ..execution.order import Order
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    max_position_size: float  # Maximum size of any single position
    max_portfolio_value: float  # Maximum total portfolio value
    max_drawdown: float  # Maximum allowed drawdown percentage
    position_value_limit: float  # Maximum value for any single position
    max_leverage: float  # Maximum allowed leverage
    concentration_limit: float  # Maximum percentage of portfolio in single position

class RiskManager:
    def __init__(
        self,
        portfolio: Portfolio,
        risk_limits: Optional[RiskLimits] = None
    ):
        self.portfolio = portfolio
        self.risk_limits = risk_limits or RiskLimits(
            max_position_size=1000000.0,
            max_portfolio_value=10000000.0,
            max_drawdown=0.20,  # 20% max drawdown
            position_value_limit=1000000.0,
            max_leverage=1.0,  # No leverage by default
            concentration_limit=0.25  # Max 25% in single position
        )
        self.high_water_mark = portfolio.total_value({})  # Initialize with current portfolio value


    def check_order_risk(self, order: Order, current_prices: Dict[str, float]) -> bool:
        """
        Check if an order complies with all risk limits.
        Returns True if order is acceptable, False otherwise.
        """
        current_price = current_prices.get(order.symbol)
        if current_price is None:
            logger.warning(f"No price available for {order.symbol}")
            return False
            
        # Calculate order value
        order_value = order.quantity * current_price
        
        # Check position size limits
        if not self._check_position_size(order, current_price):
            return False
            
        # Check portfolio value limits
        if not self._check_portfolio_value(order_value, current_prices):
            return False
            
        # Check concentration limits
        if not self._check_concentration(order, current_price):
            return False
            
        # Check drawdown limits
        if not self._check_drawdown(order_value):
            return False
            
        # Check leverage limits - Now passing current_prices
        if not self._check_leverage(order_value, current_prices):
            return False
            
        return True
        
    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        volatility: float
    ) -> float:
        """
        Calculate the recommended position size based on volatility and risk limits.
        Uses position sizing based on volatility and portfolio value.
        """
        portfolio_value = self.portfolio.total_value({})
        
        # Calculate position size based on volatility
        # Higher volatility = smaller position size
        base_position_size = (portfolio_value * self.risk_limits.concentration_limit)
        volatility_adjustment = 1.0 / (1.0 + volatility)
        
        # Adjust for current portfolio concentration
        current_position = self.portfolio.get_position(symbol)
        current_exposure = (
            (current_position.quantity * current_price)
            if current_position else 0.0
        )
        
        max_additional_exposure = (
            portfolio_value * self.risk_limits.concentration_limit
            - current_exposure
        )
        
        # Calculate final position size
        position_value = min(
            base_position_size * volatility_adjustment,
            max_additional_exposure,
            self.risk_limits.position_value_limit
        )
        
        return position_value / current_price if current_price > 0 else 0.0
        
    def _check_position_size(self, order: Order, current_price: float) -> bool:
        """Check if the order would violate position size limits."""
        position = self.portfolio.get_position(order.symbol)
        new_position_size = order.quantity
        if position:
            if order.side == 'BUY':
                new_position_size += position.quantity
            else:
                new_position_size = position.quantity - order.quantity
                
        position_value = abs(new_position_size * current_price)
        
        if position_value > self.risk_limits.position_value_limit:
            logger.warning(f"Position size limit exceeded for {order.symbol}")
            return False
            
        return True

    def _check_portfolio_value(self, order_value: float, current_prices: Dict[str, float]) -> bool:
        """Check if the order would violate portfolio value limits."""
        current_value = self.portfolio.total_value(current_prices)
        if current_value + order_value > self.risk_limits.max_portfolio_value:
            logger.warning("Portfolio value limit exceeded")
            return False
        return True
        
    def _check_concentration(self, order: Order, current_price: float) -> bool:
        """Check if the order would violate concentration limits."""
        portfolio_value = self.portfolio.total_value({})
        position_value = order.quantity * current_price
        
        if position_value / portfolio_value > self.risk_limits.concentration_limit:
            logger.warning(f"Concentration limit exceeded for {order.symbol}")
            return False
        return True
        
    def _check_drawdown(self, order_value: float) -> bool:
        """Check if the order would violate drawdown limits."""
        current_value = self.portfolio.total_value({})
        drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        
        if drawdown > self.risk_limits.max_drawdown:
            logger.warning("Drawdown limit exceeded")
            return False
            
        # Update high water mark if we have a new maximum
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value
            
        return True
        
    def _check_leverage(self, order_value: float, current_prices: Dict[str, float]) -> bool:
        """
        Check if the order would violate leverage limits.
        
        Args:
            order_value: Value of the new order
            current_prices: Dictionary of symbol -> current price
        """
        portfolio_value = self.portfolio.total_value(current_prices)
        total_exposure = sum(
            pos.quantity * current_prices[symbol]
            for symbol, pos in self.portfolio.positions.items()
            if symbol in current_prices
        )
        
        leverage = (total_exposure + order_value) / portfolio_value if portfolio_value > 0 else float('inf')
        
        if leverage > self.risk_limits.max_leverage:
            logger.warning(f"Leverage limit exceeded: {leverage:.2f}x > {self.risk_limits.max_leverage}x")
            return False
        return True