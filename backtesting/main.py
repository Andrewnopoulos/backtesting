"""
Main execution script
"""
from config import *
from data.data_loader import DataLoader
from indicators.technical_indicators import TechnicalIndicators
from strategies.ma_crossover import MACrossoverStrategy
from backtesting.backtest import Backtester
import matplotlib.pyplot as plt

def main():
    # Initialize components
    data_loader = DataLoader(ALPHA_VANTAGE_API_KEY)
    
    # Get data
    data = data_loader.get_daily_data(DEFAULT_SYMBOL)
    
    # Add indicators
    indicators = TechnicalIndicators()
    data = indicators.add_moving_average(data, MA_SHORT)
    data = indicators.add_moving_average(data, MA_LONG)
    data = indicators.add_rsi(data, RSI_PERIOD)
    data = indicators.add_macd(data)
    
    # Create and run strategy
    strategy = MACrossoverStrategy(MA_SHORT, MA_LONG)
    backtester = Backtester(data, strategy)
    returns, metrics = backtester.run()

    # Plot the full strategy
    # backtester.plot_strategy()

    # Plot for a specific date range
    backtester.plot_strategy(start_date='2023-01-01', end_date='2024-10-31')

    # # Plot returns distribution
    # backtester.plot_returns_distribution()

    plt.savefig('backtest_results.png')
    
    # Print results
    print("Backtest Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2%}")

if __name__ == "__main__":
    main()