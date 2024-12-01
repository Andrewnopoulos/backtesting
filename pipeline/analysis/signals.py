import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class Signal:
    symbol: str
    timestamp: pd.Timestamp
    signal_type: str
    strength: float
    metadata: Dict

class SignalDetector:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        
    def detect_momentum_signal(self, data: pd.DataFrame, threshold: float = 0.02) -> Optional[Signal]:
        """
        Detect momentum signals based on price movement.
        
        Args:
            data (pd.DataFrame): Price data with columns ['timestamp', 'close']
            threshold (float): Minimum price change threshold for signal generation
            
        Returns:
            Optional[Signal]: Detected signal or None
        """
        if len(data) < self.lookback_period:
            return None
            
        # Calculate returns
        returns = data['close'].pct_change(self.lookback_period)
        current_return = returns.iloc[-1]
        
        if abs(current_return) >= threshold:
            return Signal(
                symbol=data['symbol'].iloc[-1],
                timestamp=data.index[-1],
                signal_type='MOMENTUM_LONG' if current_return > 0 else 'MOMENTUM_SHORT',
                strength=abs(current_return),
                metadata={'lookback_period': self.lookback_period}
            )
        return None
        
    def detect_volatility_breakout(self, data: pd.DataFrame, std_multiplier: float = 2.0) -> Optional[Signal]:
        """
        Detect volatility breakout signals.
        
        Args:
            data (pd.DataFrame): Price data with columns ['timestamp', 'close', 'high', 'low']
            std_multiplier (float): Standard deviation multiplier for breakout detection
            
        Returns:
            Optional[Signal]: Detected signal or None
        """
        if len(data) < self.lookback_period:
            return None
            
        # Calculate volatility
        volatility = data['close'].rolling(window=self.lookback_period).std()
        current_volatility = volatility.iloc[-1]
        
        # Calculate price change
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        if abs(price_change) > std_multiplier * current_volatility:
            return Signal(
                symbol=data['symbol'].iloc[-1],
                timestamp=data.index[-1],
                signal_type='VOLATILITY_BREAKOUT',
                strength=abs(price_change) / current_volatility,
                metadata={
                    'volatility': current_volatility,
                    'price_change': price_change
                }
            )
        return None
    
    def detect_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Run all signal detection algorithms on the data.
        
        Args:
            data (pd.DataFrame): Price and volume data
            
        Returns:
            List[Signal]: List of detected signals
        """
        signals = []
        
        try:
            # Detect momentum signals
            momentum_signal = self.detect_momentum_signal(data)
            if momentum_signal:
                signals.append(momentum_signal)
            
            # Detect volatility breakout
            volatility_signal = self.detect_volatility_breakout(data)
            if volatility_signal:
                signals.append(volatility_signal)
                
        except Exception as e:
            self.logger.error(f"Error detecting signals: {str(e)}")
            
        return signals