# main.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict

from src.portfolio.portfolio import Portfolio
from src.execution.executor import OrderExecutor
from src.risk.risk_manager import RiskManager, RiskLimits
from src.analytics.performance import PerformanceTracker
from src.analytics.visualization import PerformanceVisualizer

def generate_test_data(days: int = 252) -> Dict[str, List]:
    """Generate test market data and signals."""
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    # Generate price data for two symbols
    np.random.seed(42)
    aapl_prices = 150 + np.random.randn(days).cumsum()
    googl_prices = 2800 + np.random.randn(days).cumsum() * 2
    
    # Ensure prices don't go negative
    aapl_prices = np.maximum(aapl_prices, 50)
    googl_prices = np.maximum(googl_prices, 1000)
    
    return {
        'dates': dates,
        'AAPL': aapl_prices.tolist(),
        'GOOGL': googl_prices.tolist()
    }

def simulate_trading():
    """Run trading simulation with all components."""
    # Initialize components
    initial_cash = 1000000.0  # $1M starting capital
    portfolio = Portfolio(initial_cash=initial_cash)
    
    risk_limits = RiskLimits(
        max_position_size=100000.0,
        max_portfolio_value=2000000.0,
        max_drawdown=0.20,
        position_value_limit=500000.0,
        max_leverage=1.0,
        concentration_limit=0.25
    )
    
    risk_manager = RiskManager(portfolio, risk_limits)
    executor = OrderExecutor(portfolio, risk_manager)
    performance_tracker = PerformanceTracker(portfolio)
    visualizer = PerformanceVisualizer(performance_tracker)
    
    # Generate test data
    market_data = generate_test_data(252)  # One year of daily data
    
    print("Starting simulation...")
    
    # Simulate trading for each day
    for i in range(len(market_data['dates'])):
        current_date = market_data['dates'][i]
        current_prices = {
            'AAPL': market_data['AAPL'][i],
            'GOOGL': market_data['GOOGL'][i]
        }
        
        # Simple trading logic for demonstration
        # Buy AAPL if price is less than previous day
        if i > 0 and market_data['AAPL'][i] < market_data['AAPL'][i-1]:
            executor.place_market_order('AAPL', 100, 'BUY', current_prices)
            
        # Buy GOOGL if price is less than previous day
        if i > 0 and market_data['GOOGL'][i] < market_data['GOOGL'][i-1]:
            executor.place_market_order('GOOGL', 10, 'BUY', current_prices)
            
        # Sell positions if price is 5% higher than previous day
        for symbol in ['AAPL', 'GOOGL']:
            if i > 0:
                price_change = (market_data[symbol][i] / market_data[symbol][i-1]) - 1
                if price_change > 0.05:
                    position = portfolio.get_position(symbol)
                    if position and position.quantity > 0:
                        executor.place_market_order(symbol, position.quantity, 'SELL', current_prices)
        
        # Process orders with current market data
        executor.process_market_data(current_prices)
        
        # Update performance tracking
        performance_tracker.update(current_prices, current_date)
        
    print("Simulation completed.")
    
    # Calculate final metrics
    final_metrics = performance_tracker.calculate_metrics()
    
    # Generate visualizations
    print("\nGenerating performance visualizations...")
    
    # Create and save plots
    plots = {
        'equity_curve': visualizer.plot_equity_curve(),
        'returns_dist': visualizer.plot_returns_distribution(),
        'rolling_metrics': visualizer.plot_rolling_metrics(),
        'underwater': visualizer.plot_underwater_chart(),
        'trade_analysis': visualizer.plot_trade_analysis(),
        'metrics_dashboard': visualizer.plot_metrics_dashboard(final_metrics)
    }
    
    # Save plots
    for name, fig in plots.items():
        fig.savefig(f'output_{name}.png')
        plt.close(fig)
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Final Portfolio Value: ${portfolio.total_value(current_prices):,.2f}")
    print(f"Total Return: {(portfolio.total_value(current_prices) / initial_cash - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {final_metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {final_metrics.max_drawdown * 100:.2f}%")
    print(f"Win Rate: {final_metrics.win_rate * 100:.2f}%")
    print(f"Total Trades: {final_metrics.total_trades}")

if __name__ == "__main__":
    simulate_trading()