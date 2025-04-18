import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Adjust path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import get_db_connection
# Import the efficient multi-symbol fetch function and list_symbols
from database.data_provider import list_symbols_db, fetch_ohlc_data_for_symbols_db
from screening.reversal_screener import screen_stocks, DEFAULT_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Lookback period for fetching data (adjust as needed)
FETCH_LOOKBACK_DAYS = 300 # Match the original Flask app logic initially

def run_scan_and_update_db():
    """
    Runs the reversal stock screener, fetches necessary data,
    and updates the reversal_scan_results table in the database.
    """
    logger.info("Starting reversal scan background job.")
    conn = None
    try:
        # 1. Get all symbols
        logger.info("Fetching list of all symbols...")
        all_symbols = list_symbols_db()
        if all_symbols is None:
            logger.error("Failed to fetch symbol list. Aborting job.")
            return
        if not all_symbols:
            logger.warning("No symbols found in the database. Aborting job.")
            return
        logger.info(f"Found {len(all_symbols)} symbols.")

        # 2. Fetch OHLCV data efficiently for all symbols
        logger.info(f"Fetching OHLCV data efficiently for {len(all_symbols)} symbols (lookback: {FETCH_LOOKBACK_DAYS} days)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=FETCH_LOOKBACK_DAYS)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        fetch_start_time = datetime.now()
        historical_data_map = fetch_ohlc_data_for_symbols_db(
            all_symbols,
            start_date=start_date_str,
            end_date=end_date_str
        )
        fetch_end_time = datetime.now()

        if historical_data_map is None:
            logger.error("Failed to fetch multi-symbol OHLCV data (fetch_ohlc_data_for_symbols_db returned None). Aborting job.")
            return
        elif not historical_data_map:
             logger.warning("No historical data returned for any symbols by fetch_ohlc_data_for_symbols_db. Aborting job.")
             return

        logger.info(f"Finished fetching and processing data for {len(historical_data_map)} symbols in {fetch_end_time - fetch_start_time}.")

        if not historical_data_map:
            logger.warning("No historical data fetched for any symbols. Cannot run screen. Aborting job.")
            return

        # 3. Run the screening function
        logger.info(f"Running reversal screen on {len(historical_data_map)} symbols with data...")
        screen_start_time = datetime.now()
        results = screen_stocks(list(historical_data_map.keys()), historical_data_map, DEFAULT_CONFIG)
        screen_end_time = datetime.now()
        logger.info(f"Screening completed in {screen_end_time - screen_start_time}. Found {len(results)} qualified stocks.")

        # 4. Update the database
        if not results:
            logger.info("No stocks qualified. Clearing results table.")
            # Still clear the table even if no results
            conn = get_db_connection()
            if conn:
                with conn.cursor() as cur:
                    #cur.execute("TRUNCATE TABLE reversal_scan_results;")
                    logger.info("Cleared reversal_scan_results table.")
                conn.commit()
            else:
                logger.error("Failed to get DB connection to clear results table.")
            return # Exit early

        logger.info(f"Updating reversal_scan_results table with {len(results)} results...")
        update_start_time = datetime.now()
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to get DB connection to update results. Aborting.")
            return

        with conn.cursor() as cur:
            # Clear previous results
            cur.execute("TRUNCATE TABLE reversal_scan_results;")
            logger.info("Cleared previous results from reversal_scan_results.")

            # Prepare insert statement
            insert_sql = """
                INSERT INTO reversal_scan_results (
                    symbol, last_close, last_volume, sma150,
                    sma150_slope_norm, rsi14, last_date, scan_timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            scan_time = datetime.now() # Use a consistent timestamp for all rows in this batch

            # Prepare data tuples for insertion
            data_to_insert = [
                (
                    res['symbol'],
                    res.get('last_close'),
                    res.get('last_volume'),
                    res.get('sma150'),
                    res.get('sma150_slope_norm'),
                    res.get('rsi14'),
                    res.get('last_date'),
                    scan_time
                ) for res in results
            ]

            # Execute batch insert
            cur.executemany(insert_sql, data_to_insert)
            conn.commit()
            update_end_time = datetime.now()
            logger.info(f"Successfully inserted {len(results)} results into reversal_scan_results in {update_end_time - update_start_time}.")

    except Exception as e:
        logger.exception(f"An error occurred during the reversal scan job: {e}")
        if conn:
            conn.rollback() # Rollback any partial changes on error
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")
        logger.info("Reversal scan background job finished.")

if __name__ == "__main__":
    run_scan_and_update_db()