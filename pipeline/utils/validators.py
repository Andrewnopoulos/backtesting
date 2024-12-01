from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]

class MarketDataValidator:
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> ValidationResult:
        """
        Validate price data structure and values.
        
        Args:
            data (pd.DataFrame): Price data to validate
            
        Returns:
            ValidationResult: Validation results with any errors found
        """
        errors = []
        
        # Check required columns
        required_columns = ['timestamp', 'symbol', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
            
        # Check for null values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
            
        # Validate price values
        if 'close' in data.columns:
            if (data['close'] <= 0).any():
                errors.append("Invalid price values found (prices must be positive)")
                
        # Validate volume values
        if 'volume' in data.columns:
            if (data['volume'] < 0).any():
                errors.append("Invalid volume values found (volume cannot be negative)")
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
    
    @staticmethod
    def validate_trade_update(trade_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate trade update data from the stream.
        
        Args:
            trade_data (Dict[str, Any]): Trade update data to validate
            
        Returns:
            ValidationResult: Validation results with any errors found
        """
        errors = []
        
        # Check required fields
        required_fields = ['symbol', 'price', 'timestamp']
        missing_fields = [field for field in required_fields if field not in trade_data]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            
        # Validate price
        if 'price' in trade_data:
            try:
                price = float(trade_data['price'])
                if price <= 0:
                    errors.append("Invalid price value (must be positive)")
            except (ValueError, TypeError):
                errors.append("Invalid price format")
                
        # Validate timestamp
        if 'timestamp' in trade_data:
            try:
                datetime.fromisoformat(trade_data['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                errors.append("Invalid timestamp format")
                
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )
        
    @staticmethod
    def validate_signal_data(signal_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate signal data before processing.
        
        Args:
            signal_data (Dict[str, Any]): Signal data to validate
            
        Returns:
            ValidationResult: Validation results with any errors found
        """
        errors = []
        
        # Check required fields
        required_fields = ['symbol', 'signal_type', 'timestamp', 'strength']
        missing_fields = [field for field in required_fields if field not in signal_data]
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            
        # Validate signal strength
        if 'strength' in signal_data:
            try:
                strength = float(signal_data['strength'])
                if not (0 <= strength <= 1):
                    errors.append("Signal strength must be between 0 and 1")
            except (ValueError, TypeError):
                errors.append("Invalid signal strength format")
                
        # Validate signal type
        valid_signal_types = ['MOMENTUM_LONG', 'MOMENTUM_SHORT', 'VOLATILITY_BREAKOUT']
        if 'signal_type' in signal_data and signal_data['signal_type'] not in valid_signal_types:
            errors.append(f"Invalid signal type. Must be one of: {valid_signal_types}")
            
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )