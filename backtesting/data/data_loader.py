"""
Handles data downloading and preprocessing
"""
from alpha_vantage.timeseries import TimeSeries

class DataLoader:
    def __init__(self, api_key):
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        
    def get_daily_data(self, symbol, outputsize='full'):
        # Using get_daily() instead of get_daily_adjusted() for free tier
        data, _ = self.ts.get_daily(symbol=symbol, outputsize=outputsize)
        return self._preprocess_data(data)
    
    def _preprocess_data(self, df):
        # Basic preprocessing
        df = df.sort_index()
        # Rename columns to lowercase without numbers
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        return df