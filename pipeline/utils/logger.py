import logging
import sys
from typing import Optional
from pathlib import Path

def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger instance with both console and file handlers.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_market_logger(log_file: Optional[str] = "logs/market_data.log") -> logging.Logger:
    """
    Get a pre-configured logger for market data operations.
    """
    return setup_logger(
        name="market_data",
        log_level=logging.INFO,
        log_file=log_file
    )