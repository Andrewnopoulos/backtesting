import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, t

class TradingStrategyAnalysis:
    def __init__(self, returns):
        """
        Initialize with a pandas Series of returns
        """
        self.returns = returns
        self.alpha = 0.05  # significance level
        plt.style.use('seaborn-v0_8')  # Using seaborn style for better-looking plots
        
    def analyze_distribution(self):
        """
        Analyze return distribution and test for normality
        """
        # Basic statistics
        stats_dict = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'skew': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns)
        }
        
        # Shapiro-Wilk test for normality
        _, p_value = stats.shapiro(self.returns)
        is_normal = p_value > self.alpha
        
        # Fit Student's t-distribution
        t_params = stats.t.fit(self.returns)
        
        return {
            'statistics': stats_dict,
            'is_normal': is_normal,
            'p_value': p_value,
            't_params': t_params
        }
    
    def plot_return_distribution(self, figsize=(12, 8)):
        """
        Plot return distribution with normal and t-distribution fits
        """
        plt.figure(figsize=figsize)
        
        # Plot histogram of returns
        sns.histplot(self.returns, stat='density', alpha=0.6, label='Returns')
        
        # Fit normal distribution
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        mu, sigma = norm.fit(self.returns)
        plt.plot(x, norm.pdf(x, mu, sigma), 'r-', label='Normal fit')
        
        # Fit t-distribution
        t_params = stats.t.fit(self.returns)
        plt.plot(x, t.pdf(x, *t_params), 'g-', label='Student-t fit')
        
        plt.title('Return Distribution with Fitted Distributions')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig('return-distribution.png')
        
    def plot_qq(self, figsize=(10, 6)):
        """
        Create Q-Q plot to assess normality
        """
        plt.figure(figsize=figsize)
        stats.probplot(self.returns, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Returns')
        plt.grid(True, alpha=0.3)
        plt.savefig('qq.png')
        
    def plot_time_series(self, figsize=(12, 8)):
        """
        Plot time series of returns and cumulative returns
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Returns
        ax1.plot(self.returns.index, self.returns)
        ax1.set_title('Returns Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Returns')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        cum_returns = (1 + self.returns).cumprod()
        ax2.plot(cum_returns.index, cum_returns)
        ax2.set_title('Cumulative Returns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Return')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('time-series.png')
        
    def plot_autocorrelation(self, lags=40, figsize=(12, 8)):
        """
        Plot ACF and PACF
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        plot_acf(self.returns, lags=lags, ax=ax1)
        ax1.set_title('Autocorrelation Function')
        
        plot_pacf(self.returns, lags=lags, ax=ax2)
        ax2.set_title('Partial Autocorrelation Function')
        
        plt.tight_layout()
        plt.savefig('autocorrelation.png')
        
    def plot_monte_carlo_var(self, n_simulations=10000, confidence_level=0.95, figsize=(10, 6)):
        """
        Plot Monte Carlo VaR simulation results
        """
        # Fit t-distribution and generate simulations
        t_params = stats.t.fit(self.returns)
        simulated_returns = stats.t.rvs(*t_params, size=n_simulations)
        
        # Calculate VaR
        var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        plt.figure(figsize=figsize)
        sns.histplot(simulated_returns, stat='density', alpha=0.6)
        plt.axvline(var, color='r', linestyle='--', 
                   label=f'{confidence_level*100}% VaR: {var:.4f}')
        
        plt.title('Monte Carlo Simulation of Returns with VaR')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('monte-carlo.png')
        
    def plot_rolling_statistics(self, window=252, figsize=(12, 8)):
        """
        Plot rolling mean, std, and Sharpe ratio
        """
        rolling_mean = self.returns.rolling(window=window).mean()
        rolling_std = self.returns.rolling(window=window).std()
        rolling_sharpe = np.sqrt(252) * (rolling_mean / rolling_std)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)
        
        # Rolling mean
        ax1.plot(rolling_mean.index, rolling_mean)
        ax1.set_title(f'Rolling Mean ({window} days)')
        ax1.grid(True, alpha=0.3)
        
        # Rolling std
        ax2.plot(rolling_std.index, rolling_std)
        ax2.set_title(f'Rolling Std ({window} days)')
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe
        ax3.plot(rolling_sharpe.index, rolling_sharpe)
        ax3.set_title(f'Rolling Sharpe Ratio ({window} days)')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()

        plt.savefig('rolling.png')

# Example usage
if __name__ == "__main__":
    # Generate sample returns (with some non-normal characteristics)
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000) + 
                       np.random.standard_t(df=3, size=1000) * 0.01,
                       index=dates)
    
    # Create analyzer instance
    analyzer = TradingStrategyAnalysis(returns)
    
    # Generate all plots
    analyzer.plot_return_distribution()
    analyzer.plot_qq()
    analyzer.plot_time_series()
    analyzer.plot_autocorrelation()
    analyzer.plot_monte_carlo_var()
    analyzer.plot_rolling_statistics()