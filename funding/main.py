import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import logging
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

class ExchangeInterface:
    def __init__(self, exchange_configs: Dict):
        """
        Initialize exchange connections with API keys
        
        exchange_configs = {
            'binance': {
                'api_key': 'your_api_key',
                'secret_key': 'your_secret_key',
                'testnet': False
            },
            'ftx': {
                'api_key': 'your_api_key',
                'secret_key': 'your_secret_key',
                'subaccount': 'optional_subaccount_name'
            }
        }
        """
        self.exchanges = {}
        self.logger = self._setup_logger()
        
        for exchange_name, config in exchange_configs.items():
            try:
                if exchange_name == 'binance':
                    self.exchanges[exchange_name] = ccxt.binance({
                        'apiKey': config['api_key'],
                        'secret': config['secret_key'],
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future',
                            'adjustForTimeDifference': True,
                            'testnet': config.get('testnet', False)
                        }
                    })
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {str(e)}")
                raise

    def _setup_logger(self):
        logger = logging.getLogger('ExchangeArbitrage')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    async def get_funding_rates(self, symbol: str) -> Dict[str, float]:
        """Fetch current funding rates from all connected exchanges"""
        funding_rates = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                if exchange_name == 'binance':
                    markets = exchange.fetch_markets()
                    # markets = await asyncio.gather(*tasks)
                    future_market = next(m for m in markets if m['id'] == f"{symbol}USDT")
                    funding_info = exchange.fetch_funding_rate(future_market['symbol'])
                    funding_rates[exchange_name] = float(funding_info['fundingRate'])
                
            except Exception as e:
                self.logger.error(f"Error fetching funding rate from {exchange_name}: {str(e)}")
                
        return funding_rates

    async def get_market_prices(self, symbol: str) -> Dict[str, Dict]:
        """Fetch spot and perpetual prices from exchanges"""
        prices = {}
        
        for exchange_name, exchange in self.exchanges.items():
            try:
                # Fetch spot price
                spot_ticker = await exchange.fetch_ticker(f"{symbol}/USDT")
                perp_ticker = await exchange.fetch_ticker(f"{symbol}/USDT:USDT")
                
                prices[exchange_name] = {
                    'spot': float(spot_ticker['last']),
                    'perpetual': float(perp_ticker['last']),
                    'spot_volume': float(spot_ticker['quoteVolume']),
                    'perp_volume': float(perp_ticker['quoteVolume'])
                }
                
            except Exception as e:
                self.logger.error(f"Error fetching prices from {exchange_name}: {str(e)}")
                
        return prices

    async def execute_arbitrage_positions(self, 
                                       exchange: str, 
                                       symbol: str, 
                                       size: float,
                                       prices: Dict) -> Tuple[bool, Dict]:
        """Execute spot-perpetual arbitrage positions"""
        try:
            exchange_instance = self.exchanges[exchange]
            
            # Calculate precise amounts based on exchange minimums
            market_info = await exchange_instance.fetch_market(f"{symbol}/USDT:USDT")
            min_amount = market_info.get('limits', {}).get('amount', {}).get('min', 0)
            precision = market_info.get('precision', {}).get('amount', 8)
            
            # Round size to meet exchange requirements
            adjusted_size = round(size, precision)
            if adjusted_size < min_amount:
                return False, {"error": f"Size {size} below minimum {min_amount}"}
            
            # Execute spots and futures simultaneously
            tasks = [
                # Spot market buy
                exchange_instance.create_order(
                    symbol=f"{symbol}/USDT",
                    type='market',
                    side='buy',
                    amount=adjusted_size
                ),
                # Perpetual futures sell
                exchange_instance.create_order(
                    symbol=f"{symbol}/USDT:USDT",
                    type='market',
                    side='sell',
                    amount=adjusted_size
                )
            ]
            
            spot_order, perp_order = await asyncio.gather(*tasks)
            
            return True, {
                'spot_order': spot_order,
                'perp_order': perp_order,
                'timestamp': datetime.now().isoformat(),
                'size': adjusted_size,
                'exchange': exchange
            }
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage: {str(e)}")
            return False, {"error": str(e)}

    async def monitor_positions(self, 
                              positions: List[Dict], 
                              min_spread: float = 0.01) -> List[Dict]:
        """Monitor active arbitrage positions and funding payments"""
        position_updates = []
        
        for position in positions:
            exchange = position['exchange']
            symbol = position['symbol']
            
            try:
                # Get current rates and prices
                current_funding = await self.get_funding_rates(symbol)
                current_prices = await self.get_market_prices(symbol)
                
                # Calculate current spread and P&L
                spread = current_funding[exchange]
                spot_price = current_prices[exchange]['spot']
                perp_price = current_prices[exchange]['perpetual']
                
                unrealized_pnl = (spot_price - perp_price) * position['size']
                
                position_updates.append({
                    'position_id': position['position_id'],
                    'current_spread': spread,
                    'unrealized_pnl': unrealized_pnl,
                    'should_close': spread < min_spread / 2,
                    'current_prices': current_prices[exchange],
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Error monitoring position: {str(e)}")
                
        return position_updates

    async def close_positions(self, position: Dict) -> Tuple[bool, Dict]:
        """Close both spot and perpetual positions"""
        try:
            exchange = self.exchanges[position['exchange']]
            
            # Close both positions simultaneously
            tasks = [
                # Spot market sell
                exchange.create_order(
                    symbol=f"{position['symbol']}/USDT",
                    type='market',
                    side='sell',
                    amount=position['size']
                ),
                # Perpetual futures buy
                exchange.create_order(
                    symbol=f"{position['symbol']}/USDT:USDT",
                    type='market',
                    side='buy',
                    amount=position['size']
                )
            ]
            
            spot_close, perp_close = await asyncio.gather(*tasks)
            
            return True, {
                'spot_close': spot_close,
                'perp_close': perp_close,
                'timestamp': datetime.now().isoformat(),
                'position_id': position['position_id']
            }
            
        except Exception as e:
            self.logger.error(f"Error closing positions: {str(e)}")
            return False, {"error": str(e)}

# Example usage
async def main():

    load_dotenv()

    # Initialize with your API keys
    exchange_configs = {
        'binance': {
            'api_key': os.getenv('BINANCE_API_KEY'),
            'secret_key': os.getenv('BINANCE_SECRET_KEY'),
            'testnet': True  # Use testnet for testing
        }
    }
    
    exchange_interface = ExchangeInterface(exchange_configs)
    
    # Fetch current market data
    symbol = 'BTC'
    funding_rates = await exchange_interface.get_funding_rates(symbol)
    market_prices = await exchange_interface.get_market_prices(symbol)
    
    # Example position size in USD
    position_size = 1000 / market_prices['binance']['spot']  # Convert USD to crypto amount
    
    # Execute arbitrage if conditions are met
    if funding_rates['binance'] > 0.01:  # 1% funding rate threshold
        success, position = await exchange_interface.execute_arbitrage_positions(
            exchange='binance',
            symbol=symbol,
            size=position_size,
            prices=market_prices
        )
        
        if success:
            print(f"Successfully opened arbitrage position: {position}")
            
            # Monitor position
            while True:
                updates = await exchange_interface.monitor_positions([position])
                
                if updates[0]['should_close']:
                    success, close_result = await exchange_interface.close_positions(position)
                    if success:
                        print(f"Successfully closed position: {close_result}")
                        break
                
                await asyncio.sleep(60)  # Check every minute

if __name__ == "__main__":
    asyncio.run(main())