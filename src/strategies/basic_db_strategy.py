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

    def __init__(self, strategy_name, strategy_title, view_name, db_config: Optional[Dict[str, Any]] = None):
        """
        Initializes the HighLow strategy simulator.

        Args:
            db_config (Dict[str, Any], optional): Database connection parameters.
                                                 Defaults to config file values if available.
        """
        # Call the parent constructor with the specific strategy name
        self.strategy_name = strategy_name
        self.strategy_title = strategy_title
        if view_name:
            self.view_name = view_name
        elif db_config and "view_name" in db_config:
            self.view_name = db_config.get("view_name", "v_strategy_sma_150")
        else:
            raise ValueError("view_name must be provided in db_config or as an argument.")
        
        super().__init__(db_config=db_config, strategy_name=strategy_title)

    def get_data_query(self) -> str:
        """
        Returns the SQL query specific to the HighLow 250 strategy.
        """

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



strategies = {
    "HighLow250":{"title": "HighLow Strategy with 250 days", "view": "v_strategy_highlow_250"},
    "SMA150": {"title" :"SMA Strategy with 150 days", "view": "v_strategy_sma_150"},
    "Low250_SMA150": {"title" :"Buy on 250 day low / Sell when below 150SMA", "view": "v_strategy_high250_sma150"}
}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER]):
         print("Warning: Database configuration not fully loaded from src.config. Exiting.")
         exit(1)

    start_simulation_date = "2020-01-01"
    all_results = {} # Dictionary to store results {strategy_key: stats_dict}

    print(f"\nRunning simulations starting from {start_simulation_date}...\n")

    # Ensure pandas is imported if not already at the top
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas library is required for comparative statistics. Please install it (`pip install pandas`).")
        exit(1)


    for key, strategy_definition in strategies.items():
        print(f"--- Running Strategy: {key} ({strategy_definition['title']}) ---")
        # Instantiate the strategy runner
        strategy_runner = BasicDBStrategy(
            strategy_name=strategy_definition["title"], # Name used in BaseStrategy logging
            strategy_title=strategy_definition["title"], # Specific title if needed elsewhere
            view_name=strategy_definition["view"]
            # db_config is implicitly handled by BaseStrategy using src.config
        )

        if strategy_runner.connection:
            # Run simulation and get stats
            stats = strategy_runner.run_simulation_and_get_stats(start_simulation_date)
            if stats:
                all_results[key] = stats # Store results using the strategy key
                # Optionally print individual summary here if desired
                # print(f"  Completed: Trades={stats['total_trades']}, PnL={stats['total_pnl_formatted']}, PF={stats['profit_factor_formatted']}")
            else:
                 print(f"  Strategy {key} failed to produce results (check logs).")
            # Close connection after each strategy run
            strategy_runner.close_connection()
            print(f"--- Finished Strategy: {key} ---")
        else:
            print(f"Could not connect to DB for strategy {key}. Skipping.")
            # No need to call close_connection if it never connected

    # --- Comparative Statistics ---
    print("\n--- Comparative Statistics ---")
    if not all_results:
        print("No strategy results were collected.")
    else:
        # Convert results to DataFrame for better display
        # Select relevant columns for comparison (using formatted strings for clarity)
        comparison_data = {}
        for key, stats in all_results.items():
             comparison_data[key] = {
                 "Total Trades": stats.get("total_trades", 0),
                 "Win Rate (%)": stats.get("win_rate_formatted", "N/A"),
                 "Total PnL": stats.get("total_pnl_formatted", "N/A"),
                 "Avg PnL/Trade": stats.get("average_pnl_per_trade_formatted", "N/A"),
                 "Profit Factor": stats.get("profit_factor_formatted", "N/A"),
                 "Avg Win": stats.get("average_win_formatted", "N/A"),
                 "Avg Loss": stats.get("average_loss_formatted", "N/A"),
                 # Store raw PnL for sorting
                 "_Total PnL (Raw)": stats.get("total_pnl_raw", 0.0)
             }

        results_df = pd.DataFrame.from_dict(comparison_data, orient='index')

        # Sort by raw Total PnL (descending)
        if "_Total PnL (Raw)" in results_df.columns:
             results_df = results_df.sort_values(by="_Total PnL (Raw)", ascending=False)
             results_df = results_df.drop(columns=["_Total PnL (Raw)"]) # Remove the raw column after sorting

        # Print using pandas default formatting (often better than to_string for wide tables)
        print(results_df)
        # Or use to_string if you prefer that format:
        # print(results_df.to_string())

    print("\n--- Simulation Run Complete ---")