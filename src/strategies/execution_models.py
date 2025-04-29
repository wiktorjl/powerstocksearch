
import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)

class BaseExecutionModel(ABC):
    """Abstract base class for strategy execution models."""

    @abstractmethod
    def execute(self, signals_df: pd.DataFrame, strategy_name: str = "Strategy") -> List[Dict[str, Any]]:
        """
        Executes the trading logic based on the provided signals.

        Args:
            signals_df (pd.DataFrame): DataFrame containing price, date, and action signals.
            strategy_name (str): Name of the strategy for logging purposes.

        Returns:
            List[Dict[str, Any]]: A list of completed trades.
        """
        pass

class SimpleLongOnlyExecution(BaseExecutionModel):
    """
    A simple execution model that goes long one lot on 'OPEN' signals
    and closes the position on 'CLOSE' signals. Does not go short.
    """
    def execute(self, signals_df: pd.DataFrame, strategy_name: str = "Strategy") -> List[Dict[str, Any]]:
        """
        Executes the simple long-only trading logic.
        """
        if signals_df.empty:
            logger.warning(f"[{strategy_name}] No signals data provided to execution model. Returning no trades.")
            return []

        position = 0  # 0 = flat, 1 = long 1 lot
        entry_price = 0.0
        entry_date = None
        trades = []

        logger.info(f"[{strategy_name}] Starting execution with SimpleLongOnlyExecution model...")
        for index, row in signals_df.iterrows():
            price = row.get('PRICE')
            date = row.get('DATE')
            action = row.get('ACTION')

            if price is None or date is None or action is None:
                logger.warning(f"[{strategy_name}] Skipping row due to missing data in execution: {row}")
                continue

            if action == 'OPEN' and position == 0:
                position = 1
                entry_price = price
                entry_date = date
                logger.debug(f"[{strategy_name}] {date}: Execution - Entered LONG at {price}")
            elif action == 'CLOSE' and position == 1:
                exit_price = price
                exit_date = date
                pnl = exit_price - entry_price
                trade_duration = exit_date - entry_date if entry_date else None

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "duration": trade_duration
                })
                logger.debug(f"[{strategy_name}] {date}: Execution - Exited LONG at {exit_price}. PnL: {pnl:.2f}")
                position = 0
                entry_price = 0.0
                entry_date = None

        logger.info(f"[{strategy_name}] Execution finished. Generated {len(trades)} trades.")
        return trades
