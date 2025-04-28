import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Assuming the project structure allows this import
# Need to adjust path if running from root or specific location
try:
    # Use relative import if part of the same package
    from ..database.connection import get_db_connection
    from ..config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
except ImportError:
    # Fallback for different execution contexts or if structure differs
    try:
        from src.database.connection import get_db_connection
        from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    except ImportError:
        logging.warning("Could not import database/config modules. Ensure PYTHONPATH is set or run from project root.")
        # Provide fallback or raise a more specific error if needed
        DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD = (None,) * 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategy simulations.

    Handles database connection, data fetching framework, statistics calculation,
    and results printing. Subclasses must implement the specific trading logic
    in `simulate` and define the data source via `get_data_query`.
    """

    def __init__(self, db_config: Optional[Dict[str, Any]] = None, strategy_name: str = "BaseStrategy"):
        """
        Initializes the base strategy simulator.

        Args:
            db_config (Dict[str, Any], optional): Database connection parameters.
                                                 Defaults to config file values if available.
            strategy_name (str): The name of the specific strategy.
        """
        self.strategy_name = strategy_name
        if db_config is None:
            # Check if config values were imported successfully
            if all([DB_HOST, DB_PORT, DB_NAME, DB_USER]): # Password can be None/empty
                 self.db_config = {
                    "host": DB_HOST,
                    "port": DB_PORT,
                    "database": DB_NAME,
                    "user": DB_USER,
                    "password": DB_PASSWORD
                }
            else:
                logging.error(f"[{self.strategy_name}] Database configuration not found in src.config or provided.")
                self.db_config = {} # Set empty config
        else:
            self.db_config = db_config

        self.connection = None
        if self.db_config: # Only attempt connection if config is present
            self._connect_db()

    def _connect_db(self):
        """Establishes the database connection."""
        if not self.db_config:
             logging.error(f"[{self.strategy_name}] Cannot connect to DB: Configuration is missing.")
             self.connection = None
             return
        try:
            # get_db_connection likely reads config directly from src.config or environment
            self.connection = get_db_connection() # Call without config argument
            logging.info(f"[{self.strategy_name}] Database connection established successfully.")
        except Exception as e:
            logging.error(f"[{self.strategy_name}] Failed to connect to the database: {e}")
            self.connection = None # Ensure connection is None if failed

    @abstractmethod
    def get_data_query(self) -> str:
        """
        Abstract method to get the SQL query for fetching strategy data.
        Subclasses must implement this to define their data source.

        Returns:
            str: The SQL query string. Should include placeholders like %(start_date)s.
        """
        pass

    def _fetch_data(self, start_date: str) -> pd.DataFrame:
        """
        Fetches trading signals from the database using the query provided by the subclass.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.

        Returns:
            pd.DataFrame: DataFrame containing the data needed for the strategy,
                          ordered by DATE. Returns empty DataFrame on error.
                          Expects columns like 'PRICE', 'DATE', 'ACTION' or similar.
        """
        if not self.connection:
            logging.error(f"[{self.strategy_name}] No database connection available to fetch data.")
            return pd.DataFrame()

        # Ensure start_date is valid before querying
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
        except ValueError:
            logging.error(f"[{self.strategy_name}] Invalid start_date format: {start_date}. Please use YYYY-MM-DD.")
            return pd.DataFrame()

        query = self.get_data_query()
        if not query:
             logging.error(f"[{self.strategy_name}] No data query defined by the strategy.")
             return pd.DataFrame()

        try:
            logging.info(f"[{self.strategy_name}] Fetching data starting from {start_date}")
            df = pd.read_sql(query, self.connection, params={"start_date": start_date})
            logging.info(f"[{self.strategy_name}] Fetched {len(df)} data points.")
            # Basic data cleaning - subclasses might need more specific cleaning
            if not df.empty:
                if 'DATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['DATE'])
                if 'PRICE' in df.columns:
                    df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')
                    df.dropna(subset=['PRICE'], inplace=True) # Drop rows with invalid prices
            return df
        except Exception as e:
            logging.error(f"[{self.strategy_name}] Error fetching data from database: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    @abstractmethod
    def simulate(self, start_date: str) -> List[Dict[str, Any]]:
        """
        Abstract method to run the backtest simulation based on fetched data.
        Subclasses must implement their specific trading logic here.

        Args:
            start_date (str): The start date for the simulation in 'YYYY-MM-DD'.

        Returns:
            List[Dict[str, Any]]: A list of completed trades, each represented
                                  as a dictionary. Returns empty list if no
                                  data or error occurs.
        """
        pass

    def _calculate_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates performance statistics from a list of trades.
        (Copied from original HighLowStrategy, potentially reusable)
        """
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "breakeven_trades": 0,
                "win_rate_excl_breakeven": "0.00%",
                "total_pnl": "0.00",
                "average_pnl_per_trade": "0.00",
                "average_win": "0.00",
                "average_loss": "0.00",
                "profit_factor": "N/A",
                "average_trade_duration": "N/A"
            }

        total_trades = len(trades)
        pnl_values = [trade['pnl'] for trade in trades]
        total_pnl = sum(pnl_values)

        winning_trades_pnl = [pnl for pnl in pnl_values if pnl > 0]
        losing_trades_pnl = [pnl for pnl in pnl_values if pnl < 0]
        breakeven_trades = [pnl for pnl in pnl_values if pnl == 0]

        num_winning_trades = len(winning_trades_pnl)
        num_losing_trades = len(losing_trades_pnl)
        num_breakeven_trades = len(breakeven_trades)

        # Win rate excluding breakeven trades
        total_win_loss_trades = num_winning_trades + num_losing_trades
        win_rate_excl_breakeven = (num_winning_trades / total_win_loss_trades * 100) \
                                  if total_win_loss_trades > 0 else 0.0

        total_profit = sum(winning_trades_pnl)
        total_loss = abs(sum(losing_trades_pnl)) # Loss is negative, take absolute

        average_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
        average_win = total_profit / num_winning_trades if num_winning_trades > 0 else 0.0
        average_loss = total_loss / num_losing_trades if num_losing_trades > 0 else 0.0 # Use absolute loss total

        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') # Handle zero loss case

        # Basic duration calculation
        durations = [trade['duration'] for trade in trades if trade.get('duration') is not None] # Use .get for safety
        average_trade_duration = sum(durations, pd.Timedelta(0)) / len(durations) if durations else None

        # Format results for printing
        return {
            "total_trades": total_trades,
            "winning_trades": num_winning_trades,
            "losing_trades": num_losing_trades,
            "breakeven_trades": num_breakeven_trades,
            "win_rate_excl_breakeven": f"{win_rate_excl_breakeven:.2f}%",
            "total_pnl": f"{total_pnl:.2f}",
            "average_pnl_per_trade": f"{average_pnl_per_trade:.2f}",
            "average_win": f"{average_win:.2f}",
            "average_loss": f"{average_loss:.2f}",
            "profit_factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinite",
            "average_trade_duration": str(average_trade_duration) if average_trade_duration else "N/A"
        }

    def run_simulation_and_print_stats(self, start_date: str):
        """
        Runs the simulation using the subclass's implementation and prints the calculated statistics.
        """
        # Ensure DB connection is attempted if not already connected
        if not self.connection and self.db_config:
            self._connect_db()

        if not self.connection:
             logging.error(f"[{self.strategy_name}] Cannot run simulation without a valid database connection.")
             print("\n--- Simulation Results ---")
             print(f"Strategy: {self.strategy_name}")
             print("ERROR: Could not connect to the database.")
             print("--------------------------\n")
             return # Exit if no connection

        trades = self.simulate(start_date)
        stats = self._calculate_statistics(trades)

        print("\n--- Simulation Results ---")
        if not trades:
            print(f"Strategy: {self.strategy_name} | Start: {start_date} | Result: No trades executed.")
        else:
            print(f"Strategy: {self.strategy_name} | Start: {start_date} | Trades: {stats['total_trades']} | WinRate: {stats['win_rate_excl_breakeven']} | PnL: {stats['total_pnl']} | Avg PnL: {stats['average_pnl_per_trade']} | PF: {stats['profit_factor']}")
        print("--------------------------\n")


    def close_connection(self):
        """Closes the database connection if it's open."""
        if self.connection:
            try:
                self.connection.close()
                logging.info(f"[{self.strategy_name}] Database connection closed.")
                self.connection = None
            except Exception as e:
                logging.error(f"[{self.strategy_name}] Error closing database connection: {e}")