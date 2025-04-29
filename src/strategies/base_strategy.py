import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type # Add Type
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

# Import the new execution model base class
from .execution_models import BaseExecutionModel, SimpleLongOnlyExecution

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategy simulations.

    Handles database connection, data fetching framework, statistics calculation,
    and results printing. Subclasses must implement the specific trading logic
    in `simulate` and define the data source via `get_data_query`.
    """

    def __init__(self, db_config: Optional[Dict[str, Any]] = None, 
                 strategy_name: str = "BaseStrategy",
                 execution_model: Optional[BaseExecutionModel] = None):
        """
        Initializes the base strategy simulator.

        Args:
            db_config (Dict[str, Any], optional): Database connection parameters.
            strategy_name (str): The name of the specific strategy.
            execution_model (BaseExecutionModel, optional): The execution model to use.
                                                        Defaults to SimpleLongOnlyExecution.
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

        # Assign or default the execution model
        self.execution_model = execution_model if execution_model is not None else SimpleLongOnlyExecution()
        logging.info(f"[{self.strategy_name}] Using execution model: {self.execution_model.__class__.__name__}")

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

    # REMOVE the abstract simulate method
    # @abstractmethod
    # def simulate(self, start_date: str) -> List[Dict[str, Any]]:
    #     """
    #     Abstract method to run the backtest simulation based on fetched data.
    #     Subclasses must implement their specific trading logic here.
    #     ... (rest of docstring) ...
    #     """
    #     pass

    def _calculate_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates performance statistics from a list of trades.
        Includes maximum drawdown calculation.
        Returns a dictionary containing both raw numerical values and formatted strings.
        """
        if not trades:
            # Return structure with Nones/Zeros for raw values and N/A or 0.00 for formatted
            return {
                "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "breakeven_trades": 0,
                "win_rate_raw": 0.0, "total_pnl_raw": 0.0, "average_pnl_per_trade_raw": 0.0,
                "average_win_raw": 0.0, "average_loss_raw": 0.0, "profit_factor_raw": 0.0, # Use 0 for PF when no trades/loss
                "average_trade_duration_raw": None,
                "max_drawdown_raw": 0.0, # Add max drawdown
                # Formatted values
                "win_rate_formatted": "0.00%", "total_pnl_formatted": "0.00",
                "average_pnl_per_trade_formatted": "0.00", "average_win_formatted": "0.00",
                "average_loss_formatted": "0.00", "profit_factor_formatted": "N/A",
                "average_trade_duration_formatted": "N/A",
                "max_drawdown_formatted": "0.00" # Add formatted max drawdown
            }

        # Sort trades by exit date to calculate equity curve correctly
        trades.sort(key=lambda x: x['exit_date'])

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

        # Handle profit factor calculation carefully
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        elif total_profit > 0 and total_loss == 0:
             profit_factor = float('inf') # Infinite profit factor
        else: # No profit and no loss (or only breakeven trades)
             profit_factor = 0.0 # Or 1.0 depending on definition, let's use 0 for simplicity

        # Basic duration calculation
        durations = [trade['duration'] for trade in trades if trade.get('duration') is not None and isinstance(trade['duration'], timedelta)] # Check type
        average_trade_duration = sum(durations, timedelta(0)) / len(durations) if durations else None

        # --- Maximum Drawdown Calculation ---
        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_drawdown = 0.0
        for pnl in pnl_values:
            cumulative_pnl += pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            drawdown = peak_pnl - cumulative_pnl # Drawdown is positive value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        # --- End Maximum Drawdown Calculation ---


        # Return both raw and formatted values
        return {
            "total_trades": total_trades,
            "winning_trades": num_winning_trades,
            "losing_trades": num_losing_trades,
            "breakeven_trades": num_breakeven_trades,
            # Raw values
            "win_rate_raw": win_rate_excl_breakeven,
            "total_pnl_raw": total_pnl,
            "average_pnl_per_trade_raw": average_pnl_per_trade,
            "average_win_raw": average_win,
            "average_loss_raw": average_loss, # Note: This is positive value of loss
            "profit_factor_raw": profit_factor,
            "average_trade_duration_raw": average_trade_duration,
            "max_drawdown_raw": max_drawdown, # Add raw max drawdown
            # Formatted values for display
            "win_rate_formatted": f"{win_rate_excl_breakeven:.2f}%",
            "total_pnl_formatted": f"{total_pnl:.2f}",
            "average_pnl_per_trade_formatted": f"{average_pnl_per_trade:.2f}",
            "average_win_formatted": f"{average_win:.2f}",
            "average_loss_formatted": f"{average_loss:.2f}",
            "profit_factor_formatted": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinite",
            "average_trade_duration_formatted": str(average_trade_duration) if average_trade_duration else "N/A",
            "max_drawdown_formatted": f"{max_drawdown:.2f}" # Add formatted max drawdown
        }

    def run_simulation_and_get_stats(self, start_date: str) -> Optional[Dict[str, Any]]:
        """
        Runs the simulation by fetching data and using the assigned execution model,
        then returns the calculated statistics.
        Returns None if the simulation cannot be run (e.g., DB connection issue or fetch error).
        """
        # Ensure DB connection is attempted if not already connected
        if not self.connection and self.db_config:
            self._connect_db()

        if not self.connection:
             logging.error(f"[{self.strategy_name}] Cannot run simulation without a valid database connection.")
             return None

        # Fetch data first
        signals_df = self._fetch_data(start_date)

        if signals_df.empty:
            logging.warning(f"[{self.strategy_name}] No data fetched for start date {start_date}. Cannot run execution.")
            # Return stats for zero trades
            return self._calculate_statistics([])

        # Use the execution model to get trades
        if not self.execution_model:
            logging.error(f"[{self.strategy_name}] No execution model assigned. Cannot run simulation.")
            return None # Or return zero stats? Error seems more appropriate

        logging.info(f"[{self.strategy_name}] Running execution model {self.execution_model.__class__.__name__}...")
        trades = self.execution_model.execute(signals_df, self.strategy_name)

        # Calculate statistics based on the trades from the execution model
        stats = self._calculate_statistics(trades)

        logging.info(f"[{self.strategy_name}] Simulation complete for start date {start_date}. Trades: {stats.get('total_trades', 0)}.")

        return stats

    def close_connection(self):
        """Closes the database connection if it's open."""
        if self.connection:
            try:
                self.connection.close()
                logging.info(f"[{self.strategy_name}] Database connection closed.")
                self.connection = None
            except Exception as e:
                logging.error(f"[{self.strategy_name}] Error closing database connection: {e}")