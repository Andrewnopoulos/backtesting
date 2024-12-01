import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from utils.logger import get_market_logger
from utils.validators import MarketDataValidator
from analysis.signals import SignalDetector, Signal
from dataclasses import dataclass

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from utils.logger import get_market_logger
from utils.validators import MarketDataValidator
from analysis.signals import SignalDetector, Signal
from dataclasses import dataclass

@dataclass
class ProcessedData:
    symbol: str
    timestamp: pd.Timestamp
    metrics: Dict
    signals: List[Signal]

    def __str__(self):
        """String representation for logging"""
        signal_summary = [str(signal) for signal in self.signals]
        return (f"ProcessedData for {self.symbol} at {self.timestamp}\n"
                f"Metrics: {self.metrics}\n"
                f"Signals: {signal_summary if signal_summary else 'None'}")

class DataProcessor:
    def __init__(self, lookback_period: int = 20):
        self.logger = get_market_logger()
        self.validator = MarketDataValidator()
        self.signal_detector = SignalDetector(lookback_period=lookback_period)
        self.last_signal_time = {}  # Track last signal time per symbol
        
    def process_data(self, data: pd.DataFrame) -> Optional[ProcessedData]:
        """
        Process market data and detect signals.
        """
        # Validate input data
        validation_result = self.validator.validate_price_data(data)
        if not validation_result.is_valid:
            self.logger.warning(
                f"Invalid market data received: {validation_result.errors}"
            )
            return None
            
        try:
            # Group data by symbol
            grouped = data.groupby('symbol')
            processed_results = []
            
            for symbol, symbol_data in grouped:
                # Calculate metrics
                metrics = self.calculate_metrics(symbol_data)
                
                # Detect signals
                signals = self.signal_detector.detect_signals(symbol_data)
                
                if signals:
                    # Filter out signals too close to previous ones
                    signals = self.filter_recent_signals(symbol, signals)
                    
                    if signals:
                        processed = ProcessedData(
                            symbol=symbol,
                            timestamp=symbol_data.index[-1],
                            metrics=metrics,
                            signals=signals
                        )
                        processed_results.append(processed)
                        self.logger.info(f"Processed data with signals: {processed}")
            
            return processed_results if processed_results else None
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            return None

    def filter_recent_signals(self, symbol: str, signals: List[Signal], min_interval_seconds: int = 60) -> List[Signal]:
        """Filter out signals that are too close to previous ones"""
        current_time = pd.Timestamp.now()
        if symbol in self.last_signal_time:
            time_since_last = (current_time - self.last_signal_time[symbol]).total_seconds()
            if time_since_last < min_interval_seconds:
                return []
        
        if signals:
            self.last_signal_time[symbol] = current_time
        return signals
            
    def calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate various market metrics.
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics['vwap'] = self.calculate_vwap(data)
            metrics['volatility'] = self.calculate_volatility(data)
            metrics['rsi'] = self.calculate_rsi(data)
            
            # Volume metrics
            metrics['volume_ma'] = data['volume'].rolling(window=20).mean().iloc[-1]
            metrics['volume_std'] = data['volume'].rolling(window=20).std().iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            
        return metrics
        
    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        try:
            return (data['close'] * data['volume']).sum() / data['volume'].sum()
        except Exception:
            return 0.0
        
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, window: int = 20) -> float:
        """Calculate price volatility"""
        try:
            return data['close'].pct_change().rolling(window=window).std().iloc[-1]
        except Exception:
            return 0.0
        
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14) -> float:
        """Calculate Relative Strength Index"""
        try:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs.iloc[-1]))
        except Exception:
            return 50.0  # Return neutral RSI on error