from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from ..portfolio.portfolio import Portfolio
from .metrics import TradeMetrics, Trade

class PerformanceTracker:
    def __init__(self, portfolio: Portfolio, benchmark_returns: Optional[List[float]] = None):
        self.portfolio = portfolio
        self.trades: List[Trade] = []
        self.daily_returns: List[float] = []
        self.benchmark_returns = benchmark_returns or []
        self.equity_curve: List[float] = []
        self.timestamps: List[datetime] = []
        self.high_water_mark: float = portfolio.initial_cash
        self.current_drawdown: float = 0.0
        self.max_drawdown: float = 0.0
        
    def calculate_metrics(self, risk_free_rate: float = 0.02) -> TradeMetrics:
        """Calculate comprehensive trading metrics including advanced measures."""
        if not self.trades:
            return self._empty_metrics()
            
        returns_array = np.array(self.daily_returns)
        
        # Basic metrics calculation (from previous implementation)
        basic_metrics = self._calculate_basic_metrics(returns_array, risk_free_rate)
        
        # Advanced metrics calculation
        advanced_metrics = self._calculate_advanced_metrics(returns_array, risk_free_rate)
        
        return TradeMetrics(
            **basic_metrics,
            **advanced_metrics
        )
        
    def _calculate_basic_metrics(self, returns_array: np.ndarray, risk_free_rate: float) -> dict:
        """Calculate basic trading metrics."""
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
        
        gross_profits = sum(t.pnl for t in winning_trades)
        gross_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        if len(returns_array) >= 2:
            excess_returns = returns_array - (risk_free_rate / 252)
            sharpe_ratio = self._calculate_sharpe_ratio(excess_returns)
            sortino_ratio = self._calculate_sortino_ratio(excess_returns)
        else:
            sharpe_ratio = sortino_ratio = 0.0
            
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades)
        }
        
    def _calculate_advanced_metrics(self, returns_array: np.ndarray, risk_free_rate: float) -> dict:
        """Calculate advanced trading metrics."""
        if len(returns_array) < 2:
            return self._empty_advanced_metrics()
            
        annualized_return = np.mean(returns_array) * 252
        annualized_vol = np.std(returns_array, ddof=1) * np.sqrt(252)
        
        # Calmar Ratio (annualized return / max drawdown)
        calmar_ratio = annualized_return / self.max_drawdown if self.max_drawdown != 0 else 0
        
        # MAR Ratio (annualized return / max drawdown) with minimum 3-year window
        mar_ratio = calmar_ratio if len(returns_array) >= 756 else 0  # 756 trading days ≈ 3 years
        
        # Omega Ratio (probability weighted ratio of gains versus losses)
        threshold = risk_free_rate / 252
        omega_ratio = self._calculate_omega_ratio(returns_array, threshold)
        
        # Tail Ratio (95th percentile / 5th percentile in absolute terms)
        tail_ratio = self._calculate_tail_ratio(returns_array)
        
        # Value at Risk (95% confidence)
        var_95 = self._calculate_var(returns_array, confidence_level=0.95)
        
        # Expected Shortfall (Average loss beyond VaR)
        expected_shortfall = self._calculate_expected_shortfall(returns_array, confidence_level=0.95)
        
        # Gain to Pain Ratio
        gain_to_pain = self._calculate_gain_to_pain_ratio(returns_array)
        
        # Ulcer Index (measure of drawdown severity)
        ulcer_index = self._calculate_ulcer_index()
        
        # Information Ratio (if benchmark returns available)
        information_ratio = self._calculate_information_ratio(returns_array)
        
        # Treynor Ratio (return over beta)
        treynor_ratio = self._calculate_treynor_ratio(returns_array, risk_free_rate)
        
        # Kappa Three (similar to Sortino but with cubic root of downside deviation)
        kappa_three = self._calculate_kappa_three(returns_array, risk_free_rate)
        
        # Trade Expectancy (average win * win rate - average loss * loss rate)
        trade_expectancy = self._calculate_trade_expectancy()
        
        return {
            'calmar_ratio': calmar_ratio,
            'mar_ratio': mar_ratio,
            'omega_ratio': omega_ratio,
            'tail_ratio': tail_ratio,
            'value_at_risk': var_95,
            'expected_shortfall': expected_shortfall,
            'gain_to_pain_ratio': gain_to_pain,
            'ulcer_index': ulcer_index,
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'kappa_three': kappa_three,
            'trade_expectancy': trade_expectancy
        }
        
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float) -> float:
        """Calculate Omega ratio."""
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns < threshold].sum())
        return gains / losses if losses != 0 else float('inf')
        
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio."""
        if len(returns) < 20:  # Need sufficient data for percentiles
            return 0.0
        return abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))
        
    def _calculate_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) < 2:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence_level) * 100))
        
    def _calculate_expected_shortfall(self, returns: np.ndarray, confidence_level: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) < 2:
            return 0.0
        var = self._calculate_var(returns, confidence_level)
        return abs(np.mean(returns[returns <= -var]))
        
    def _calculate_gain_to_pain_ratio(self, returns: np.ndarray) -> float:
        """Calculate Gain to Pain ratio."""
        if len(returns) < 1:
            return 0.0
        return np.sum(returns) / np.sum(np.abs(returns[returns < 0]))
        
    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index."""
        if not self.equity_curve:
            return 0.0
        prices = np.array(self.equity_curve)
        drawdowns = np.maximum.accumulate(prices) - prices
        squared_drawdowns = (drawdowns / prices) ** 2
        return np.sqrt(np.mean(squared_drawdowns))
        
    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate Information Ratio against benchmark."""
        if not self.benchmark_returns or len(returns) != len(self.benchmark_returns):
            return 0.0
        excess_returns = returns - np.array(self.benchmark_returns)
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) if len(excess_returns) > 1 else 0
        
    def _calculate_treynor_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Treynor Ratio."""
        if not self.benchmark_returns or len(returns) < 2:
            return 0.0
        beta = self._calculate_beta(returns)
        excess_return = np.mean(returns) - (risk_free_rate / 252)
        return excess_return / beta if beta != 0 else 0
        
    def _calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate beta against benchmark."""
        if not self.benchmark_returns or len(returns) != len(self.benchmark_returns):
            return 0.0
        benchmark_returns = np.array(self.benchmark_returns)
        covariance = np.cov(returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)
        return covariance / benchmark_variance if benchmark_variance != 0 else 0
        
    def _calculate_kappa_three(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate Kappa Three ratio."""
        if len(returns) < 2:
            return 0.0
        threshold = risk_free_rate / 252
        excess_returns = returns - threshold
        below_threshold = excess_returns[excess_returns < 0]
        if len(below_threshold) < 1:
            return 0.0
        lpm = np.mean(below_threshold ** 3) ** (1/3)
        return np.mean(excess_returns) / lpm if lpm != 0 else 0
        
    def _calculate_trade_expectancy(self) -> float:
        """Calculate trade expectancy."""
        if not self.trades:
            return 0.0
        win_rate = sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if any(t.pnl > 0 for t in self.trades) else 0
        avg_loss = abs(np.mean([t.pnl for t in self.trades if t.pnl <= 0])) if any(t.pnl <= 0 for t in self.trades) else 0
        return (avg_win * win_rate) - (avg_loss * (1 - win_rate))
        
    def _empty_advanced_metrics(self) -> dict:
        """Return empty advanced metrics."""
        return {
            'calmar_ratio': 0.0,
            'mar_ratio': 0.0,
            'omega_ratio': 0.0,
            'tail_ratio': 0.0,
            'value_at_risk': 0.0,
            'expected_shortfall': 0.0,
            'gain_to_pain_ratio': 0.0,
            'ulcer_index': 0.0,
            'information_ratio': 0.0,
            'treynor_ratio': 0.0,
            'kappa_three': 0.0,
            'trade_expectancy': 0.0
        }

    def _empty_metrics(self) -> TradeMetrics:
        """Return empty metrics when no trades are available."""
        return TradeMetrics(
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            max_drawdown=self.max_drawdown,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            calmar_ratio=0.0,
            mar_ratio=0.0,
            omega_ratio=0.0,
            tail_ratio=0.0,
            value_at_risk=0.0,
            expected_shortfall=0.0,
            gain_to_pain_ratio=0.0,
            ulcer_index=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            kappa_three=0.0,
            trade_expectancy=0.0
        )

    def _calculate_sharpe_ratio(self, excess_returns: np.ndarray) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            excess_returns: Array of returns minus risk-free rate
            
        Returns:
            Annualized Sharpe ratio
        """
        if len(excess_returns) < 2:
            return 0.0
            
        # Calculate annualized Sharpe ratio
        # Multiply by sqrt(252) to annualize (assuming daily returns)
        return (
            np.mean(excess_returns) / 
            np.std(excess_returns, ddof=1) * 
            np.sqrt(252)
        )

    def _calculate_sortino_ratio(self, excess_returns: np.ndarray) -> float:
        """
        Calculate annualized Sortino ratio using downside deviation.
        
        Args:
            excess_returns: Array of returns minus risk-free rate
            
        Returns:
            Annualized Sortino ratio
        """
        if len(excess_returns) < 2:
            return 0.0
            
        # Calculate downside returns (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        # If no downside returns, return 0 to avoid division by zero
        if len(downside_returns) < 2:
            return 0.0
            
        # Calculate downside deviation
        downside_std = np.std(downside_returns, ddof=1)
        
        # Avoid division by zero
        if downside_std == 0:
            return 0.0
            
        # Calculate annualized Sortino ratio
        # Multiply by sqrt(252) to annualize (assuming daily returns)
        return (
            np.mean(excess_returns) / 
            downside_std * 
            np.sqrt(252)
        )
    
    def update(self, current_prices: Dict[str, float], timestamp: Optional[datetime] = None) -> None:
        """
        Update performance metrics with current market data.
        
        Args:
            current_prices: Dictionary mapping symbols to their current prices
            timestamp: Optional timestamp for the update (defaults to now)
        """
        timestamp = timestamp or datetime.now()
        current_value = self.portfolio.total_value(current_prices)
        
        # Calculate return since last update if we have previous data
        if self.equity_curve:
            daily_return = (current_value - self.equity_curve[-1]) / self.equity_curve[-1]
            self.daily_returns.append(daily_return)
        
        # Update equity curve
        self.equity_curve.append(current_value)
        self.timestamps.append(timestamp)
        
        # Update drawdown metrics
        self._update_drawdown(current_value)
        
        # Update high water mark if necessary
        if current_value > self.high_water_mark:
            self.high_water_mark = current_value

    def generate_report(self) -> pd.DataFrame:
        """
        Generate a detailed performance report as a DataFrame.
        
        Returns:
            DataFrame containing performance metrics over time
        """
        if not self.equity_curve:
            return pd.DataFrame()
            
        # Create base DataFrame
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'equity': self.equity_curve,
            'returns': [0.0] + self.daily_returns  # Add 0 for first day
        })
        
        # Calculate cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Calculate drawdowns
        df['drawdown'] = (df['equity'].cummax() - df['equity']) / df['equity'].cummax()
        
        # Calculate rolling metrics (30-day window)
        window = 30
        df['rolling_returns'] = df['returns'].rolling(window=window).mean() * 252  # Annualized
        df['rolling_volatility'] = df['returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
        df['rolling_sharpe'] = (
            df['rolling_returns'] / df['rolling_volatility']
            if any(df['rolling_volatility'] != 0)
            else 0.0
        )
        
        # Add rolling drawdown
        df['rolling_max_drawdown'] = df['drawdown'].rolling(window=window).max()
        
        # Add trade information if available
        if self.trades:
            trade_df = pd.DataFrame([{
                'timestamp': trade.exit_time,
                'trade_pnl': trade.pnl,
                'trade_duration': trade.hold_time_minutes
            } for trade in self.trades])
            
            # Merge trade information if we have trades
            df = pd.merge_asof(
                df,
                trade_df,
                on='timestamp',
                direction='backward'
            )
            
            # Calculate cumulative trade metrics
            df['cumulative_trade_pnl'] = df['trade_pnl'].cumsum()
        
        return df

    def _update_drawdown(self, current_value: float) -> None:
        """
        Update drawdown calculations based on current portfolio value.
        
        Args:
            current_value: Current total portfolio value
        """
        # Update high water mark if we have a new maximum
        self.high_water_mark = max(self.high_water_mark, current_value)
        
        # Calculate current drawdown as a percentage
        if self.high_water_mark > 0:  # Avoid division by zero
            self.current_drawdown = (self.high_water_mark - current_value) / self.high_water_mark
        else:
            self.current_drawdown = 0.0
            
        # Update maximum drawdown if current drawdown is larger
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
