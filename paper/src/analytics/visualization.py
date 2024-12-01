# src/analytics/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Optional
from datetime import datetime
from .metrics import TradeMetrics
from .performance import PerformanceTracker

class PerformanceVisualizer:
    def __init__(self, performance_tracker: PerformanceTracker):
        self.performance_tracker = performance_tracker
        # Set style for all plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_equity_curve(self, figsize=(12, 6)) -> plt.Figure:
        """Plot equity curve with drawdown overlay."""
        performance_data = self.performance_tracker.generate_report()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True)
        
        # Plot equity curve
        ax1.plot(performance_data['timestamp'], performance_data['equity'], 
                label='Portfolio Value', color='blue')
        ax1.set_title('Equity Curve and Drawdown')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        
        # Plot drawdown
        ax2.fill_between(performance_data['timestamp'], 
                        performance_data['drawdown'] * 100, 
                        0, 
                        color='red', 
                        alpha=0.3, 
                        label='Drawdown %')
        ax2.set_ylabel('Drawdown %')
        ax2.set_xlabel('Date')
        ax2.legend()
        
        plt.tight_layout()
        return fig
        
    def plot_returns_distribution(self, figsize=(12, 6)) -> plt.Figure:
        """Plot distribution of returns with normal distribution overlay."""
        returns = pd.Series(self.performance_tracker.daily_returns)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot histogram of returns
        sns.histplot(returns, stat='density', alpha=0.5, label='Returns', ax=ax)
        
        # Add normal distribution overlay
        x = np.linspace(returns.min(), returns.max(), 100)
        y = stats.norm.pdf(x, returns.mean(), returns.std())
        ax.plot(x, y, 'r--', label='Normal Distribution')
        
        # Add vertical lines for mean and standard deviations
        ax.axvline(returns.mean(), color='g', linestyle='--', label='Mean')
        ax.axvline(returns.mean() + returns.std(), color='r', linestyle=':', 
                  label='+1 Std Dev')
        ax.axvline(returns.mean() - returns.std(), color='r', linestyle=':',
                  label='-1 Std Dev')
        
        ax.set_title('Distribution of Daily Returns')
        ax.set_xlabel('Daily Returns')
        ax.set_ylabel('Density')
        ax.legend()
        
        return fig
        
    def plot_rolling_metrics(self, window: int = 30, figsize=(12, 8)) -> plt.Figure:
        """Plot rolling Sharpe ratio, volatility, and returns."""
        performance_data = self.performance_tracker.generate_report()
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Plot rolling returns
        performance_data['rolling_returns'] = (
            performance_data['returns'].rolling(window=window).mean() * 252
        )
        ax1.plot(performance_data['timestamp'], 
                performance_data['rolling_returns'],
                label=f'{window}-day Rolling Returns (Ann.)')
        ax1.set_title(f'{window}-day Rolling Metrics')
        ax1.set_ylabel('Annual Returns')
        ax1.legend()
        
        # Plot rolling Sharpe ratio
        ax2.plot(performance_data['timestamp'],
                performance_data['rolling_sharpe'],
                label=f'{window}-day Rolling Sharpe',
                color='green')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.legend()
        
        # Plot rolling volatility
        ax3.plot(performance_data['timestamp'],
                performance_data['rolling_volatility'],
                label=f'{window}-day Rolling Volatility (Ann.)',
                color='red')
        ax3.set_ylabel('Annual Volatility')
        ax3.set_xlabel('Date')
        ax3.legend()
        
        plt.tight_layout()
        return fig
        
    def plot_underwater_chart(self, figsize=(12, 6)) -> plt.Figure:
        """Plot underwater chart showing drawdowns over time."""
        performance_data = self.performance_tracker.generate_report()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.fill_between(performance_data['timestamp'],
                       performance_data['drawdown'] * 100,
                       0,
                       color='red',
                       alpha=0.3)
        
        ax.set_title('Underwater Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown %')
        ax.grid(True)
        
        # Add horizontal lines for significant drawdown levels
        ax.axhline(y=-5, color='yellow', linestyle='--', alpha=0.5, label='-5%')
        ax.axhline(y=-10, color='orange', linestyle='--', alpha=0.5, label='-10%')
        ax.axhline(y=-20, color='red', linestyle='--', alpha=0.5, label='-20%')
        
        ax.legend()
        return fig
        
    def plot_trade_analysis(self, figsize=(15, 10)) -> plt.Figure:
        """Plot trade analysis dashboard."""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2)
        
        # Trade PnL distribution
        ax1 = fig.add_subplot(gs[0, 0])
        trade_pnls = [trade.pnl for trade in self.performance_tracker.trades]
        sns.histplot(trade_pnls, ax=ax1)
        ax1.set_title('Trade P&L Distribution')
        ax1.set_xlabel('P&L ($)')
        
        # Cumulative trades
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_pnl = np.cumsum(trade_pnls)
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl)
        ax2.set_title('Cumulative P&L')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        
        # Trade duration histogram
        ax3 = fig.add_subplot(gs[1, 0])
        trade_durations = [trade.hold_time_minutes for trade in self.performance_tracker.trades]
        sns.histplot(trade_durations, ax=ax3)
        ax3.set_title('Trade Duration Distribution')
        ax3.set_xlabel('Duration (minutes)')
        
        # Win rate by month
        ax4 = fig.add_subplot(gs[1, 1])
        # Create DataFrame with datetime index
        trade_data = pd.DataFrame({
            'date': [pd.to_datetime(trade.exit_time) for trade in self.performance_tracker.trades],
            'result': [1 if trade.pnl > 0 else 0 for trade in self.performance_tracker.trades]
        })
        trade_data.set_index('date', inplace=True)
        
        # Calculate monthly win rate
        if not trade_data.empty:
            monthly_winrate = trade_data['result'].resample('M').mean()
            monthly_winrate.plot(kind='bar', ax=ax4)
            ax4.set_title('Monthly Win Rate')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Win Rate')
            plt.xticks(rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No trade data available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        return fig


    def plot_metrics_dashboard(self, metrics: TradeMetrics, figsize=(15, 10)) -> plt.Figure:
        """Create a dashboard of key performance metrics."""
        fig = plt.figure(figsize=figsize)
        
        # Convert metrics to dictionary
        metrics_dict = {
            'Basic Metrics': {
                'Win Rate': f"{metrics.win_rate:.2%}",
                'Profit Factor': f"{metrics.profit_factor:.2f}",
                'Sharpe Ratio': f"{metrics.sharpe_ratio:.2f}",
                'Max Drawdown': f"{metrics.max_drawdown:.2%}"
            },
            'Risk Metrics': {
                'Value at Risk': f"{metrics.value_at_risk:.2%}",
                'Expected Shortfall': f"{metrics.expected_shortfall:.2%}",
                'Ulcer Index': f"{metrics.ulcer_index:.2f}",
                'Tail Ratio': f"{metrics.tail_ratio:.2f}"
            },
            'Advanced Metrics': {
                'Calmar Ratio': f"{metrics.calmar_ratio:.2f}",
                'Information Ratio': f"{metrics.information_ratio:.2f}",
                'Omega Ratio': f"{metrics.omega_ratio:.2f}",
                'Kappa Three': f"{metrics.kappa_three:.2f}"
            }
        }
        
        for i, (section, section_metrics) in enumerate(metrics_dict.items()):
            ax = fig.add_subplot(3, 1, i+1)
            y_pos = np.arange(len(section_metrics))
            
            # Create horizontal bar chart
            bars = ax.barh(y_pos, 
                          [float(str(v).rstrip('%')) for v in section_metrics.values()])
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       list(section_metrics.values())[i],
                       ha='left', va='center', fontweight='bold')
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(section_metrics.keys()))
            ax.set_title(section)
            ax.invert_yaxis()
        
        plt.tight_layout()
        return fig