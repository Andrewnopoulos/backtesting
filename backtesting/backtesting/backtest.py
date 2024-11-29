"""
Backtesting engine with visualization capabilities
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Backtester:
    def __init__(self, data, strategy, initial_capital=100000):
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.signals = None
        self.returns = None
        self.equity_curve = None
        
    def run(self):
        self.signals = self.strategy.generate_signals(self.data)
        self.returns = self.calculate_returns(self.signals)
        self.equity_curve = self.calculate_equity_curve()
        metrics = self.calculate_metrics(self.returns)
        return self.returns, metrics
    
    def calculate_returns(self, signals):
        # Calculate position returns
        price_returns = self.data['close'].pct_change()
        strategy_returns = signals.shift(1) * price_returns
        return strategy_returns
    
    def calculate_equity_curve(self):
        return self.initial_capital * (1 + self.returns).cumprod()
    
    def calculate_metrics(self, returns):
        metrics = {
            'Total Return': (1 + returns).prod() - 1,
            'Annual Return': returns.mean() * 252,
            'Annual Volatility': returns.std() * np.sqrt(252),
            'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
            'Max Drawdown': self.calculate_max_drawdown(),
            'Win Rate': (returns > 0).mean()
        }
        return metrics
    
    def calculate_max_drawdown(self):
        equity_curve = self.calculate_equity_curve()
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        return drawdown.min()
    
    def plot_strategy(self, start_date=None, end_date=None):
        """
        Plot the trading strategy results including price data, signals, and performance
        """
        # Create a figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: Price and MA lines
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        data_to_plot = self.data[start_date:end_date] if start_date else self.data
        
        # Plot price
        ax1.plot(data_to_plot.index, data_to_plot['close'], label='Price', alpha=0.7)
        
        # Plot moving averages if they exist
        for col in self.data.columns:
            if col.startswith('MA_'):
                ax1.plot(data_to_plot.index, data_to_plot[col], 
                        label=col, alpha=0.7)
        
        # Plot buy/sell signals
        signals_to_plot = self.signals[start_date:end_date] if start_date else self.signals
        buy_signals = signals_to_plot[signals_to_plot == 1].index
        sell_signals = signals_to_plot[signals_to_plot == -1].index
        
        ax1.scatter(buy_signals, data_to_plot.loc[buy_signals, 'close'], 
                   marker='^', color='g', label='Buy Signal', alpha=1)
        ax1.scatter(sell_signals, data_to_plot.loc[sell_signals, 'close'], 
                   marker='v', color='r', label='Sell Signal', alpha=1)
        
        ax1.set_title('Trading Signals')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Equity Curve
        ax2 = plt.subplot2grid((3, 1), (2, 0))
        equity_curve = self.equity_curve[start_date:end_date] if start_date else self.equity_curve
        ax2.plot(equity_curve.index, equity_curve, label='Portfolio Value', color='blue')
        ax2.set_title('Portfolio Value Over Time')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(self):
        """
        Plot the distribution of returns
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Returns distribution
        sns.histplot(self.returns.dropna(), bins=50, ax=ax1)
        ax1.set_title('Distribution of Returns')
        ax1.set_xlabel('Return')
        ax1.set_ylabel('Frequency')
        
        # QQ plot
        from scipy import stats
        stats.probplot(self.returns.dropna(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Returns')
        
        plt.tight_layout()
        return fig