import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import the base class
from .base_strategy import BaseStrategy
# Keep original imports for config fallback if base_strategy fails (though unlikely)
# and for the __main__ block example
try:
    from ..config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
except ImportError:
    try:
        from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    except ImportError:
        # Base class already warns, maybe just pass here
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD = (None,) * 5


class BasicDBStrategy(BaseStrategy):
    """
    Simulates a trading strategy based on signals from the
    provided database view. Inherits from BaseStrategy.

    The strategy buys 1 lot on 'OPEN' signals and sells 1 lot on 'CLOSE' signals.
    It does not go short; if a 'CLOSE' signal appears while the position is flat,
    it is ignored.
    """

    def __init__(self, strategy_name, strategy_title, db_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the HighLow strategy simulator.

        Args:
            db_config (Dict[str, Any], optional): Database connection parameters.
                                                 Defaults to config file values if available.
        """
        # Call the parent constructor with the specific strategy name
        self.strategy_name = strategy_name
        self.strategy_title = strategy_title
        self.view_name = db_config.get("view_name", "v_strategy_sma_150") if db_config else "v_strategy_sma_150"
        super().__init__(db_config=db_config, strategy_name=strategy_title)

    def get_data_query(self) -> str:
        """
        Returns the SQL query specific to the HighLow 250 strategy.
        """
        # self.strategy_name = "v_strategy_sma_150"

        query = f"""
            SELECT "PRICE", "DATE", "ACTION"
            FROM public."{self.view_name}" -- Use validated table name
            WHERE "DATE" >= %(start_date)s::date -- Cast parameter to date
            ORDER BY "DATE";
        """
        return query

    def simulate(self, start_date: str) -> List[Dict[str, Any]]:
        """
        Runs the backtest simulation based on fetched data for the HighLow strategy.

        Args:
            start_date (str): The start date for the simulation in 'YYYY-MM-DD'.

        Returns:
            List[Dict[str, Any]]: A list of completed trades.
        """
        # Fetch data using the base class method which uses get_data_query()
        signals_df = self._fetch_data(start_date)

        # Data cleaning (like dropping NaN prices) is handled in base _fetch_data

        if signals_df.empty:
            logging.warning(f"[{self.strategy_name}] No valid signals data found after cleaning or error fetching data. Simulation cannot proceed.")
            return []

        position = 0  # 0 = flat, 1 = long 1 lot
        entry_price = 0.0
        entry_date = None
        trades = []

        logging.info(f"[{self.strategy_name}] Starting simulation...")
        for index, row in signals_df.iterrows():
            # Ensure columns exist before accessing
            price = row.get('PRICE')
            date = row.get('DATE')
            action = row.get('ACTION')

            # Skip row if essential data is missing
            if price is None or date is None or action is None:
                logging.warning(f"[{self.strategy_name}] Skipping row due to missing data: {row}")
                continue

            if action == 'OPEN' and position == 0:
                position = 1
                entry_price = price
                entry_date = date
                logging.debug(f"[{self.strategy_name}] {date}: Entered LONG at {price}")
            elif action == 'CLOSE' and position == 1:
                exit_price = price
                exit_date = date
                pnl = exit_price - entry_price  # PnL per lot
                trade_duration = exit_date - entry_date if entry_date else None

                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "duration": trade_duration
                })
                logging.debug(f"[{self.strategy_name}] {date}: Exited LONG at {exit_price}. PnL: {pnl:.2f}")
                # Reset position for next trade
                position = 0
                entry_price = 0.0
                entry_date = None
            # Ignore OPEN if already long
            # Ignore CLOSE if flat (position == 0)

        logging.info(f"[{self.strategy_name}] Simulation finished. Completed {len(trades)} trades.")
        return trades

    # _calculate_statistics, run_simulation_and_print_stats, _connect_db,
    # close_connection are now handled by the BaseStrategy class.


# Example Usage (remains similar, but now uses the refactored class)
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Check if config was loaded, otherwise prompt or use defaults
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER]):
         print("Warning: Database configuration not fully loaded from src.config.")
         print("Attempting to run with potentially missing configuration.")
         # Optionally, you could exit or prompt for input here
         # exit(1)

    start_simulation_date = "2020-01-01"

    # Instantiate the specific strategy
    strategy_runner = BasicDBStrategy(strategy_name="a", strategy_title="b") # db_config will be handled by BaseStrategy

    # Example with explicit config (if needed):
    # db_conf = {"host": "your_host", "port": 5432, "database": "your_db", "user": "your_user", "password": "your_password"}
    # strategy_runner = HighLowStrategy(db_config=db_conf)

    # Run simulation and print stats using the method from BaseStrategy
    # Check connection status before running
    if strategy_runner.connection:
        strategy_runner.run_simulation_and_print_stats(start_simulation_date)
        strategy_runner.close_connection()
    else:
        # Error message already printed by BaseStrategy's run method or __init__
        print("Exiting due to database connection issues.")