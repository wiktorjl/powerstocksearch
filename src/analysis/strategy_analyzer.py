import logging
import inspect
import pandas as pd
from datetime import datetime, date
from typing import Any, Dict, List, Type, Callable, Generator, Union, Iterable
from itertools import product

# Assuming BaseStrategy is the common ancestor for strategies
from ..strategies.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class StrategyAnalyzer:
    """
    Analyzes a trading strategy by running its simulation method
    with varying input parameters based on specified ranges or generators.
    """

    def __init__(self, strategy_class: Type[BaseStrategy], db_config: Dict[str, Any] = None):
        """
        Initializes the StrategyAnalyzer.

        Args:
            strategy_class (Type[BaseStrategy]): The strategy class to analyze (e.g., HighLowStrategy).
                                                 Must inherit from BaseStrategy.
            db_config (Dict[str, Any], optional): Database configuration to pass to the strategy.
                                                  Defaults to None, letting the strategy handle defaults.
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise TypeError("strategy_class must be a subclass of BaseStrategy")

        self.strategy_class = strategy_class
        self.db_config = db_config
        self.strategy_name = getattr(strategy_class, '__name__', 'UnknownStrategy')
        logger.info(f"Initialized StrategyAnalyzer for strategy: {self.strategy_name}")

    def _get_simulation_params(self) -> inspect.Signature:
        """Inspects the strategy's simulate method to get its parameters."""
        if not hasattr(self.strategy_class, 'simulate') or not callable(getattr(self.strategy_class, 'simulate')):
            raise AttributeError(f"Strategy class {self.strategy_name} does not have a callable 'simulate' method.")

        sig = inspect.signature(self.strategy_class.simulate)
        # Remove 'self' parameter
        params = {name: param for name, param in sig.parameters.items() if name != 'self'}
        return inspect.Signature(parameters=list(params.values()), return_annotation=sig.return_annotation)

    def _generate_param_combinations(self, analysis_params: Dict[str, Union[Any, Iterable, Generator]]) -> Generator[Dict[str, Any], None, None]:
        """
        Generates combinations of parameters based on the analysis configuration.

        Args:
            analysis_params: A dictionary where keys are parameter names of the
                             strategy's simulate method, and values are either:
                             - A single value for the parameter.
                             - An iterable (list, tuple, range) for multiple discrete values.
                             - A pandas date_range for date parameters.
                             - A generator for string or other types.
                             - A tuple (start, end, step) for numeric ranges (inclusive).

        Yields:
            Dict[str, Any]: A dictionary representing one combination of parameters
                            to be passed to the simulate method.
        """
        sim_sig = self._get_simulation_params()
        param_names = list(sim_sig.parameters.keys())
        processed_iterables = []

        for name in param_names:
            if name not in analysis_params:
                # Use default value if available, otherwise raise error
                if sim_sig.parameters[name].default is inspect.Parameter.empty:
                    raise ValueError(f"Missing required parameter '{name}' in analysis_params and no default value in simulate method.")
                logger.debug(f"Using default value for parameter '{name}': {sim_sig.parameters[name].default}")
                processed_iterables.append([sim_sig.parameters[name].default]) # Wrap default in list for product
                continue

            value = analysis_params[name]
            param_type = sim_sig.parameters[name].annotation

            if isinstance(value, pd.DatetimeIndex): # Date range
                 # Convert Timestamps to strings if the simulate method expects strings
                if param_type == str:
                     processed_iterables.append([d.strftime('%Y-%m-%d') for d in value])
                elif param_type == date or param_type == datetime:
                     processed_iterables.append([d.date() for d in value]) # Assuming date object is preferred
                else:
                     processed_iterables.append(list(value)) # Keep as Timestamps if type is Any or datetime
            elif isinstance(value, tuple) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value): # Numeric range (start, end, step)
                start, end, step = value
                # Simple range for now, could use np.arange or np.linspace for floats
                num_range = list(range(start, end + 1, step)) if isinstance(step, int) else [start + i * step for i in range(int((end - start) / step) + 1)] # Basic float range
                processed_iterables.append(num_range)
            elif isinstance(value, (list, tuple, range, Generator)): # Iterable or generator
                processed_iterables.append(list(value)) # Convert generators to list for product
            else: # Single value
                processed_iterables.append([value])

        # Generate all combinations
        for combo_values in product(*processed_iterables):
            yield dict(zip(param_names, combo_values))


    def run_analysis(self, analysis_params: Dict[str, Any]) -> pd.DataFrame:
        """
        Runs the strategy simulation for multiple parameter combinations.

        Args:
            analysis_params (Dict[str, Any]): Dictionary specifying the parameters
                                              and their ranges/generators for the
                                              strategy's 'simulate' method. See
                                              _generate_param_combinations docstring.

        Returns:
            pd.DataFrame: A DataFrame containing the results of all simulations,
                          indexed by the parameter combinations used.
        """
        logger.info(f"Starting analysis for strategy: {self.strategy_name}")
        simulation_results = []
        param_combinations = []

        strategy_instance = self.strategy_class(db_config=self.db_config, strategy_name=self.strategy_name, strategy_title=self.strategy_name)
        if not strategy_instance.connection:
             logger.error(f"[{self.strategy_name}] Failed to establish database connection. Analysis aborted.")
             return pd.DataFrame() # Return empty DataFrame

        try:
            param_generator = self._generate_param_combinations(analysis_params)

            for i, params_combo in enumerate(param_generator):
                logger.info(f"Running simulation {i+1} with parameters: {params_combo}")
                param_combinations.append(params_combo)
                try:
                    # Ensure connection is alive (optional, depends on strategy implementation)
                    # if not strategy_instance.connection or strategy_instance.connection.closed != 0:
                    #    logger.warning("Re-establishing DB connection for new simulation run.")
                    #    strategy_instance.close_connection() # Close if open but broken
                    #    strategy_instance = self.strategy_class(db_config=self.db_config) # Re-instantiate might be safer
                    #    if not strategy_instance.connection:
                    #        logger.error("Failed to re-establish DB connection. Skipping run.")
                    #        simulation_results.append({'error': 'DB Connection Failed', 'results': []})
                    #        continue

                    # Run the simulation
                    result = strategy_instance.simulate(**params_combo)
                    # Store results along with parameters
                    # We might want to calculate summary stats here instead of raw trades
                    # For now, just store the list of trades (or whatever simulate returns)
                    simulation_results.append({'params': params_combo, 'results': result})
                    logger.debug(f"Simulation {i+1} finished. Result count: {len(result) if isinstance(result, list) else 'N/A'}")

                except Exception as e:
                    logger.error(f"Exception during simulation run with params {params_combo}: {e}", exc_info=True)
                    simulation_results.append({'params': params_combo, 'error': str(e), 'results': []})

        finally:
            # Ensure the connection is closed after all simulations
            strategy_instance.close_connection()
            logger.info("Database connection closed after analysis.")


        # --- Result Aggregation (Example: Flatten trades into a DataFrame) ---
        all_trades = []
        for run_result in simulation_results:
            params = run_result['params']
            trades = run_result.get('results', [])
            error = run_result.get('error')

            if error:
                 # Create a placeholder row indicating the error for these params
                 error_row = params.copy()
                 error_row['error'] = error
                 all_trades.append(error_row)
            elif isinstance(trades, list) and trades:
                for trade in trades:
                    # Combine parameters with trade details
                    trade_row = params.copy()
                    trade_row.update(trade)
                    all_trades.append(trade_row)
            elif isinstance(trades, list) and not trades:
                 # No trades for this run, add params row with maybe PnL=0 or similar
                 no_trade_row = params.copy()
                 no_trade_row['pnl'] = 0 # Example placeholder
                 no_trade_row['trade_count'] = 0
                 all_trades.append(no_trade_row)
            # Handle other potential return types from simulate if necessary

        results_df = pd.DataFrame(all_trades)
        logger.info(f"Analysis complete. Generated {len(results_df)} total rows in results DataFrame.")

        return results_df

# Example Usage (Conceptual - requires strategy and config)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # This example assumes HighLowStrategy and a working config/DB setup
    try:
        from ..strategies.basic_db_strategy import BasicDBStrategy
        # from ..config import DB_CONFIG # Assuming DB_CONFIG is a dict like {'host': ..., 'port': ...}
        # Mock config if not available
        # DB_CONFIG = None # Replace with actual config loading if possible
        DB_CONFIG = {"view_name": "v_strategy_highlow_250"}
        analyzer = StrategyAnalyzer(strategy_class=BasicDBStrategy, db_config=DB_CONFIG)

        # Define analysis parameters
        # HighLowStrategy.simulate(start_date: str)
        analysis_parameters = {
            'start_date': pd.date_range(start='2020-01-01', end='2024-04-01', freq='D')
            # Example if simulate had a numeric param 'window_size'
            # 'window_size': range(10, 31, 10), # Test windows 10, 20, 30
            # Example if simulate had a string param 'market_condition'
            # 'market_condition': (mc for mc in ['bull', 'bear', 'sideways']) # Generator
        }

        results = analyzer.run_analysis(analysis_parameters)

        if not results.empty:
            print("\nAnalysis Results:")
            print(results)

            # Example: Calculate average PnL only for start_dates where the first trade entry changes
            if 'pnl' in results.columns and 'start_date' in results.columns and 'entry_date' in results.columns:
                # Ensure start_date is sorted for correct shift comparison
                results = results.sort_values(by='start_date').reset_index(drop=True)

                # Find the first entry date for each start date
                # Handle potential NaT if no trade occurred for a start_date
                first_entry_dates = results.groupby('start_date')['entry_date'].min()

                # Identify start_dates where the first entry_date is different from the previous one
                changed_mask = first_entry_dates != first_entry_dates.shift()
                changed_mask.iloc[0] = True # Always include the first start_date

                # Get the start_dates where the first trade entry changed
                relevant_start_dates = first_entry_dates[changed_mask].index

                # Filter the original results to keep only these relevant start_dates
                filtered_results = results[results['start_date'].isin(relevant_start_dates)]

                # Sort by p&l descending
                filtered_results = filtered_results.sort_values(by='pnl', ascending=False)
                print("\nTop 10 Results (where first trade entry changed, Descending):")
                print(filtered_results.head(10))

                if not filtered_results.empty:
                    # Calculate average PnL on the filtered results
                    avg_pnl = filtered_results.groupby('start_date')['pnl'].mean()
                    print("\nTop 10 Average PnL per Start Date (where first trade entry changed, Descending):")
                    # Sort by PnL descending and take top 10
                    print(avg_pnl.nlargest(10))
                else:
                    print("\nNo changes detected in first trade entry dates across the analyzed start dates.")
        else:
            print("Analysis did not produce results (check logs for errors).")

    except ImportError:
        logger.error("Could not import HighLowStrategy or config. Run this script from the project root or ensure paths are correct.")
    except Exception as e:
        logger.error(f"An error occurred during the example execution: {e}", exc_info=True)