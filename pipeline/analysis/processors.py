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

class DataProcessor:
    def __init__(self, lookback_period: int = 20):
        self.logger = get_market_logger()
        self.validator = MarketDataValidator()
        self.signal_detector = SignalDetector(lookback_period=lookback_period)
        
    def process_data(self, data: pd.DataFrame) -> Optional[ProcessedData]:
        """
        Process market data and detect signals.
        
        Args:
            data (pd.DataFrame): Market data to process
            
        Returns:
            Optional[ProcessedData]: Processed data and signals if valid
        """
        # Validate input data
        validation_result = self.validator.validate_price_data(data)
        if not validation_result.is_valid:
            self.logger.warning(
                "Invalid market data received",
                extra={'errors': str(validation_result.errors)}
            )
            return None
            
        try:
            # Calculate metrics
            metrics = self.calculate_metrics(data)
            
            # Detect signals
            signals = self.signal_detector.detect_signals(data)
            
            return ProcessedData(
                symbol=data['symbol'].iloc[-1],
                timestamp=data.index[-1],
                metrics=metrics,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {str(e)}")
            return None
            
    def calculate_metrics(self, data: pd.DataFrame) -> Dict:
        """
        Calculate various market metrics.
        
        Args:
            data (pd.DataFrame): Market data
            
        Returns:
            Dict: Calculated metrics
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
        return (data['price'] * data['size']).sum() / data['size'].sum()
        
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, window: int = 20) -> float:
        """Calculate price volatility"""
        return data['price'].pct_change().rolling(window=window).std().iloc[-1]
        
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, window: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))