"""
Collection of technical indicators
"""
import pandas as pd
import numpy as np

class TechnicalIndicators:
    @staticmethod
    def add_moving_average(df, period, column='close', prefix='MA'):
        df[f'{prefix}_{period}'] = df[column].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_rsi(df, period=14, column='close'):
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_macd(df, column='close', fast=12, slow=26, signal=9):
        fast_ma = df[column].ewm(span=fast, adjust=False).mean()
        slow_ma = df[column].ewm(span=slow, adjust=False).mean()
        df['MACD'] = fast_ma - slow_ma
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        return df