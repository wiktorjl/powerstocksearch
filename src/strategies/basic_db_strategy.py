import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import the base class and execution models
from .base_strategy import BaseStrategy
from .execution_models import BaseExecutionModel, SimpleLongOnlyExecution # Import execution models
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

    def __init__(self, strategy_name, strategy_title, symbol_name, view_name, 
                 db_config: Optional[Dict[str, Any]] = None,
                 execution_model: Optional[BaseExecutionModel] = None): # Add execution_model param
        """
        Initializes the BasicDB strategy simulator.

        Args:
            strategy_name (str): Name for logging/identification.
            strategy_title (str): Display title for the strategy.
            view_name (str): The database view containing signals.
            db_config (Dict[str, Any], optional): Database connection parameters.
            execution_model (BaseExecutionModel, optional): The execution model to use.
                                                        Defaults to SimpleLongOnlyExecution.
        """
        self.strategy_name = strategy_name # Keep this for potential specific logging if needed
        self.strategy_title = strategy_title
        self.symbol_name = symbol_name
        if view_name:
            self.view_name = view_name
        elif db_config and "view_name" in db_config:
            self.view_name = db_config.get("view_name", "v_strategy_sma_150")
        else:
            raise ValueError("view_name must be provided in db_config or as an argument.")

        # Pass execution_model up to the BaseStrategy constructor
        super().__init__(db_config=db_config, 
                         strategy_name=strategy_title, # Use title for BaseStrategy's name
                         execution_model=execution_model)

    def get_data_query(self) -> str:
        """
        Returns the SQL query specific to the HighLow 250 strategy.
        """

        query = f"""
            SELECT "PRICE", "DATE", "ACTION"
            FROM public."{self.view_name}" -- Use validated table name
            WHERE "DATE" >= %(start_date)s::date -- Cast parameter to date
            AND "SYMBOL" = '{self.symbol_name}'
            ORDER BY "DATE";
        """
        return query

    # REMOVE the simulate method entirely
    # def simulate(self, start_date: str) -> List[Dict[str, Any]]:
    #     """
    #     Runs the backtest simulation based on fetched data for the HighLow strategy.
    #     ... (rest of docstring) ...
    #     """
    #     # ... (old implementation removed) ...
    #     pass

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

    # Define which execution model to use for this run
    # You could make this configurable later
    execution_model_to_use = SimpleLongOnlyExecution()
    print(f"Using Execution Model: {execution_model_to_use.__class__.__name__}\n")

    for symbol in ["AMZN", "AAPL"]:
        for key, strategy_definition in strategies.items():
            print(f"--- Running Strategy: {key} ({strategy_definition['title']}) ---")
            # Instantiate the strategy runner, passing the execution model
            strategy_runner = BasicDBStrategy(
                strategy_name=key, # Use the key for internal logging maybe
                strategy_title=strategy_definition["title"],
                view_name=strategy_definition["view"],
                symbol_name=symbol,  # Example symbol, adjust as needed``
                execution_model=execution_model_to_use # Pass the chosen model
            )

            if strategy_runner.connection:
                # Run simulation and get stats (uses the execution model internally now)
                stats = strategy_runner.run_simulation_and_get_stats(start_simulation_date)
                if stats:
                    all_results[key+symbol] = (symbol, stats) # Store results using the strategy key
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
        for key, symbol_stats in all_results.items():
             symbol = symbol_stats[0]
             stats = symbol_stats[1]
             comparison_data[key] = {
                 "Symbol": symbol,
                 "Total Trades": stats.get("total_trades", 0),
                 "Win Rate (%)": stats.get("win_rate_formatted", "N/A"),
                 "Total PnL": stats.get("total_pnl_formatted", "N/A"),
                 "Avg PnL/Trade": stats.get("average_pnl_per_trade_formatted", "N/A"),
                 "Profit Factor": stats.get("profit_factor_formatted", "N/A"),
                 "Max Drawdown": stats.get("max_drawdown_formatted", "N/A"), # Add Max Drawdown
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