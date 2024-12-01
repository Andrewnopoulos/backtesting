from alpaca.trading.stream import TradingStream
from typing import Dict, Optional
from utils.logger import get_market_logger
from utils.validators import MarketDataValidator
import pandas as pd
import asyncio

class MarketDataStream:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.stream = TradingStream(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper
        )
        self.logger = get_market_logger()
        self.validator = MarketDataValidator()
        self.data_buffer = pd.DataFrame()
        
    async def handle_trade_update(self, data: Dict):
        """Handle incoming trade data with validation and logging."""
        validation_result = self.validator.validate_trade_update(data)
        
        if validation_result.is_valid:
            self.logger.info(f"Received trade update: {data}")
            await self.process_trade_data(data)
        else:
            self.logger.warning(f"Received invalid trade data: {validation_result.errors}")

    async def process_trade_data(self, data: Dict):
        """Process validated trade data and update the buffer."""
        try:
            # Convert trade data to DataFrame row
            trade_df = pd.DataFrame([{
                'timestamp': pd.Timestamp(data['timestamp']),
                'symbol': data['symbol'],
                'price': float(data['price']),
                'size': float(data.get('size', 0)),
            }])
            
            # Update buffer
            self.data_buffer = pd.concat([self.data_buffer, trade_df])
            
            # Keep only recent data (e.g., last 1000 trades)
            self.data_buffer = self.data_buffer.tail(1000)
            
        except Exception as e:
            self.logger.error(f"Error processing trade data: {str(e)}")

    def get_current_data(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Get current data from the buffer, optionally filtered by symbol."""
        if symbol:
            return self.data_buffer[self.data_buffer['symbol'] == symbol].copy()
        return self.data_buffer.copy()

    async def start_ws(self):
        """Start the websocket connection asynchronously"""
        try:
            self.logger.info("Starting market data stream...")
            self.stream.subscribe_trade_updates(self.handle_trade_update)
            await self.stream._run_forever()  # Use the internal async method
        except Exception as e:
            self.logger.error(f"Error in market data stream: {str(e)}")
            raise

    async def stop(self):
        """Stop the websocket connection gracefully"""
        try:
            self.logger.info("Stopping market data stream...")
            await self.stream.stop_ws()
        except Exception as e:
            self.logger.error(f"Error stopping market data stream: {str(e)}")
            raise