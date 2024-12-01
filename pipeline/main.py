import asyncio
from data.stream import MarketDataStream
from analysis.processors import DataProcessor
from utils.logger import get_market_logger
import os
from dotenv import load_dotenv

class MarketDataApp:
    def __init__(self, simulation_mode: bool = False):
        # Load environment variables
        load_dotenv()
        
        # Initialize logger
        self.logger = get_market_logger()
        
        # Initialize components
        self.stream = MarketDataStream(
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            paper=True,
            simulation_mode=simulation_mode,
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN"]  # Example symbols
        )
        self.processor = DataProcessor()
        
    async def process_data_loop(self):
        """Background task to process accumulated data"""
        while True:
            try:
                # Get current data from stream buffer
                data = self.stream.get_current_data()
                
                if not data.empty:
                    # Process data and detect signals
                    processed_data = self.processor.process_data(data)
                    
                    if processed_data and processed_data.signals:
                        self.logger.info(
                            f"Signals detected for {processed_data.symbol}: {processed_data.signals}"
                        )
                
                # Wait before next processing
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                await asyncio.sleep(5)  # Wait longer on error
                
    async def run(self):
        """Main application loop"""
        try:
            # Start background processing task
            self.processing_task = asyncio.create_task(self.process_data_loop())
            
            # Start market data stream
            self.logger.info("Starting market data application...")
            await self.stream.start_ws()  # Changed to use async start
            
            # Keep the application running
            while True:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Error in application: {str(e)}")
        finally:
            await self.shutdown()
            
    async def shutdown(self):
        """Shutdown all application components"""
        try:
            # Stop processing task
            if hasattr(self, 'processing_task'):
                self.processing_task.cancel()
                try:
                    await self.processing_task
                except asyncio.CancelledError:
                    pass
                
            # Stop market data stream
            await self.stream.stop()
            
            self.logger.info("Application shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")

def main():
    """Main entry point"""
    app = MarketDataApp(simulation_mode=True)
    
    try:
        asyncio.run(app.run())
    except KeyboardInterrupt:
        print("\nShutdown initiated...")

if __name__ == "__main__":
    main()