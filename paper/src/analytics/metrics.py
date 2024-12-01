from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class TradeMetrics:
    # Basic metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Advanced metrics
    calmar_ratio: float
    mar_ratio: float
    omega_ratio: float
    tail_ratio: float
    value_at_risk: float
    expected_shortfall: float
    gain_to_pain_ratio: float
    ulcer_index: float
    information_ratio: float
    treynor_ratio: float
    kappa_three: float
    trade_expectancy: float

@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    hold_time_minutes: float