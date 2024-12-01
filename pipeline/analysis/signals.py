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

    def __str__(self):
        """String representation of the signal for logging"""
        return (f"{self.signal_type} signal for {self.symbol} "
                f"(strength: {self.strength:.2f}, time: {self.timestamp})")

class SignalDetector:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        self.logger = logging.getLogger(__name__)
        
    def detect_momentum_signal(self, data: pd.DataFrame, threshold: float = 0.002) -> Optional[Signal]:
        """
        Detect momentum signals based on price movement.
        
        Args:
            data (pd.DataFrame): Price data with columns ['timestamp', 'close']
            threshold (float): Minimum price change threshold for signal generation (0.2% default)
        """
        if len(data) < self.lookback_period:
            return None
            
        # Calculate returns
        returns = data['close'].pct_change(self.lookback_period)
        current_return = returns.iloc[-1]
        
        if abs(current_return) >= threshold:
            signal = Signal(
                symbol=data['symbol'].iloc[-1],
                timestamp=data.index[-1],
                signal_type='MOMENTUM_LONG' if current_return > 0 else 'MOMENTUM_SHORT',
                strength=abs(current_return),
                metadata={
                    'lookback_period': self.lookback_period,
                    'price_change': f"{current_return:.2%}"
                }
            )
            self.logger.info(f"Momentum signal detected: {signal}")
            return signal
        return None
        
    def detect_volatility_breakout(self, data: pd.DataFrame, std_multiplier: float = 2.0) -> Optional[Signal]:
        """
        Detect volatility breakout signals.
        """
        if len(data) < self.lookback_period:
            return None
            
        # Calculate volatility
        volatility = data['close'].rolling(window=self.lookback_period).std()
        current_volatility = volatility.iloc[-1]
        
        # Calculate price change
        price_change = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
        
        if abs(price_change) > std_multiplier * current_volatility:
            signal = Signal(
                symbol=data['symbol'].iloc[-1],
                timestamp=data.index[-1],
                signal_type='VOLATILITY_BREAKOUT',
                strength=abs(price_change) / current_volatility,
                metadata={
                    'volatility': f"{current_volatility:.4f}",
                    'price_change': f"{price_change:.2%}"
                }
            )
            self.logger.info(f"Volatility breakout detected: {signal}")
            return signal
        return None

    def detect_volume_spike(self, data: pd.DataFrame, volume_multiplier: float = 3.0) -> Optional[Signal]:
        """
        Detect unusual volume spikes.
        """
        if len(data) < self.lookback_period:
            return None

        # Calculate volume moving average
        volume_ma = data['volume'].rolling(window=self.lookback_period).mean()
        current_volume = data['volume'].iloc[-1]
        
        if current_volume > volume_multiplier * volume_ma.iloc[-1]:
            signal = Signal(
                symbol=data['symbol'].iloc[-1],
                timestamp=data.index[-1],
                signal_type='VOLUME_SPIKE',
                strength=current_volume / volume_ma.iloc[-1],
                metadata={
                    'current_volume': current_volume,
                    'average_volume': f"{volume_ma.iloc[-1]:.0f}"
                }
            )
            self.logger.info(f"Volume spike detected: {signal}")
            return signal
        return None
    
    def detect_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Run all signal detection algorithms on the data.
        """
        signals = []
        
        try:
            # Convert timestamp to index if it's not already
            if not isinstance(data.index, pd.DatetimeIndex):
                data = data.set_index('timestamp')
            
            # Sort data by timestamp
            data = data.sort_index()
            
            # Detect momentum signals
            momentum_signal = self.detect_momentum_signal(data)
            if momentum_signal:
                signals.append(momentum_signal)
            
            # Detect volatility breakout
            volatility_signal = self.detect_volatility_breakout(data)
            if volatility_signal:
                signals.append(volatility_signal)
                
            # Detect volume spikes
            volume_signal = self.detect_volume_spike(data)
            if volume_signal:
                signals.append(volume_signal)
                
        except Exception as e:
            self.logger.error(f"Error detecting signals: {str(e)}")
            
        return signals