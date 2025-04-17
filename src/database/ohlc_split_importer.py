import os
import json
import logging
from typing import Dict, Optional, Set, List, Any
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import pandas as pd # For parsing split date

import psycopg2
import psycopg2.extras

# Import project modules
from src import config
from src.database.connection import get_db_connection # Import shared connection function
from src.data_loading import local_file_loader as local_data


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Use logger instance

# Define directories from config
SPLIT_ADJUSTED_OHLC_DIR = Path(config.SPLIT_ADJUSTED_DIR)
SPLIT_DATA_DIR = Path(config.SPLIT_DATA_DIR) # Directory containing split JSON files

# --- Symbol Management ---
def get_or_create_symbol_id(cursor, symbol):
    """
    Retrieves the symbol_id for a given stock symbol.
    If the symbol doesn't exist, it inserts it into the symbols table.
    """
    try:
        cursor.execute("SELECT symbol_id FROM symbols WHERE symbol = %s;", (symbol,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            logger.info(f"Symbol '{symbol}' not found. Inserting...")
            cursor.execute("INSERT INTO symbols (symbol) VALUES (%s) RETURNING symbol_id;", (symbol,))
            new_id = cursor.fetchone()[0]
            logger.info(f"Symbol '{symbol}' inserted with id {new_id}.")
            # No commit here, handled by the main transaction
            return new_id
    except psycopg2.Error as e:
        logger.error(f"Error getting or creating symbol_id for '{symbol}': {e}")
        raise  # Re-raise to allow transaction rollback

# --- Data Reading ---
def read_split_data(symbol: str, directory: Path) -> Optional[List[Dict[str, Any]]]:
    """Reads split data from a JSON file for a given symbol."""
    # Sanitize symbol for filename matching
    safe_symbol = symbol.replace('.', '_').replace('-', '_').strip()
    split_filepath = directory / f"{safe_symbol}_splits.json" # Expecting SYMBOL_splits.json

    if not split_filepath.is_file():
        # It's common for stocks to have no splits, so INFO level is appropriate
        logger.info(f"No split data file found for ticker {symbol} at {split_filepath}")
        return None # Return None to indicate no splits found, not an error

    try:
        with open(split_filepath, 'r') as f:
            splits_data = json.load(f)

        # Basic validation: Check if it's a list
        if not isinstance(splits_data, list):
            logger.error(f"Invalid format in split file {split_filepath}: Expected a JSON list.")
            return None # Indicate error by returning None

        # Optional: Validate structure of list items (e.g., contain 'date' and 'split')
        validated_splits = []
        for item in splits_data:
            if isinstance(item, dict) and 'date' in item and 'split' in item:
                 # Further validation can be added here (e.g., date format, split format)
                 validated_splits.append(item)
            else:
                logger.warning(f"Skipping invalid split entry in {split_filepath}: {item}")

        logger.info(f"Read {len(validated_splits)} split records from {split_filepath}")
        return validated_splits if validated_splits else None # Return None if list is empty after validation

    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from split file: {split_filepath}")
        return None
    except Exception as e:
        logger.error(f"Error reading split file {split_filepath}: {e}")
        return None

# --- Data Parsing ---
def parse_ohlc_data(json_data, symbol_id):
    """Parses OHLCV data from JSON and prepares it for insertion."""
    parsed_data = []
    required_keys = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'} # Use capitalized keys

    for entry in json_data:
        if not required_keys.issubset(entry.keys()):
            logger.warning(f"Skipping OHLC entry due to missing keys: {entry}")
            continue
        try:
            # Ensure timestamp includes timezone information (assuming UTC if not specified)
            ts = datetime.fromisoformat(entry['Date']).replace(tzinfo=timezone.utc) # Assume UTC
            open_price = float(entry['Open']) if entry['Open'] is not None else None
            high_price = float(entry['High']) if entry['High'] is not None else None
            low_price = float(entry['Low']) if entry['Low'] is not None else None
            close_price = float(entry['Close']) if entry['Close'] is not None else None
            volume = int(entry['Volume']) if entry['Volume'] is not None else None

            parsed_data.append((
                ts,
                symbol_id,
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ))
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping OHLC entry due to parsing error ({e}): {entry}")
            continue
        except KeyError as e:
             logger.warning(f"Skipping OHLC entry due to missing key '{e}': {entry}")
             continue

    return parsed_data

def parse_split_ratio(split_str: str) -> Optional[Decimal]:
    """Parses a split ratio string 'New/Old' (e.g., '2/1') into an adjustment factor (Old/New)."""
    try:
        # EODHD uses '/' as the delimiter
        new, old = map(float, split_str.split('/'))
        if old == 0 or new == 0: # Avoid division by zero in ratio or inverse
            logger.error(f"Invalid split ratio: zero found in numerator or denominator ('{split_str}')")
            return None
        # Adjustment factor = Old / New. Prices before split are multiplied by this.
        # Store as Decimal for precision in DB
        return Decimal(old) / Decimal(new)
    except ValueError:
        logger.error(f"Could not parse split ratio: '{split_str}'")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing split ratio '{split_str}': {e}")
        return None

def parse_split_data_for_db(splits_json: List[Dict[str, Any]], symbol_id: int) -> List[tuple]:
    """Parses split data from JSON and prepares it for database insertion."""
    parsed_splits = []
    for split_info in splits_json:
        split_date_str = split_info.get('date')
        split_ratio_str = split_info.get('split')

        if not split_date_str or not split_ratio_str:
            logger.warning(f"Missing date or split string in split data for symbol_id {symbol_id}: {split_info}")
            continue

        # Parse date (assuming YYYY-MM-DD format)
        try:
            split_date = pd.to_datetime(split_date_str).date() # Store as date
        except ValueError:
            logger.warning(f"Invalid date format '{split_date_str}' for symbol_id {symbol_id}. Skipping split.")
            continue

        # Parse ratio
        ratio = parse_split_ratio(split_ratio_str)
        if ratio is None:
            logger.warning(f"Invalid split ratio '{split_ratio_str}' for symbol_id {symbol_id} on {split_date}. Skipping split.")
            continue

        parsed_splits.append((symbol_id, split_date, ratio))

    return parsed_splits


# --- Database Insertion ---
def insert_ohlc_batch(cursor, data):
    """Inserts a batch of OHLCV data using execute_values for efficiency."""
    if not data:
        return 0 # Nothing to insert

    query = """
        INSERT INTO ohlc_data (timestamp, symbol_id, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (symbol_id, timestamp) DO NOTHING;
    """
    try:
        psycopg2.extras.execute_values(
            cursor,
            query,
            data,
            template="(%s, %s, %s, %s, %s, %s, %s)",
            page_size=1000 # Adjust page_size based on performance testing
        )
        return cursor.rowcount # Returns the number of rows affected by the last command
    except psycopg2.Error as e:
        logger.error(f"Error during OHLC batch insert: {e}")
        raise # Re-raise to allow transaction rollback

def insert_splits_batch(cursor, data):
    """Inserts a batch of split data using execute_values."""
    if not data:
        return 0

    query = """
        INSERT INTO splits (symbol_id, "split_date", ratio)
        VALUES %s
        ON CONFLICT (symbol_id, "split_date") DO NOTHING;
    """
    try:
        psycopg2.extras.execute_values(
            cursor,
            query,
            data,
            template="(%s, %s, %s)",
            page_size=500 # Splits are less frequent, smaller page size is fine
        )
        return cursor.rowcount
    except psycopg2.Error as e:
        logger.error(f"Error during splits batch insert: {e}")
        raise # Re-raise to allow transaction rollback


def insert_symbol_details(cursor, symbol_id, details):
    """
    Inserts or updates symbol details (name, sector, etc.) into the database.
    """
    try:
        cursor.execute("""
            INSERT INTO symbols_details (symbol_id, name, sector, subsector)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (symbol_id) DO UPDATE
            SET name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                subsector = EXCLUDED.subsector;
        """, (symbol_id, details.get('company'), details.get('sector'), details.get('subsector'))) # Use .get for safety
    except psycopg2.Error as e:
        logger.error(f"Error inserting symbol details for ID {symbol_id}: {e}")
        raise # Re-raise to allow transaction rollback

# --- Main Execution ---
def main():
    """Main function to orchestrate the OHLC and Split data loading process."""
    conn = None # Initialize conn to None

    # Check if directories exist
    if not SPLIT_ADJUSTED_OHLC_DIR.is_dir():
        logger.error(f"Adjusted OHLC data directory '{SPLIT_ADJUSTED_OHLC_DIR}' not found.")
        return
    if not SPLIT_DATA_DIR.is_dir():
        logger.error(f"Split data directory '{SPLIT_DATA_DIR}' not found.")
        return

    try:
        conn = get_db_connection()
        if conn is None:
             logger.error("Failed to establish database connection.")
             return

        with conn.cursor() as cursor:
            # Find OHLC files first as the primary source - looking for CSV files now
            ohlc_files = [f for f in SPLIT_ADJUSTED_OHLC_DIR.glob('*.csv')] # Changed glob pattern
            logger.info(f"Found {len(ohlc_files)} adjusted OHLC CSV files in '{SPLIT_ADJUSTED_OHLC_DIR}'.") # Updated log message

            total_ohlc_inserted = 0
            total_splits_inserted = 0
            processed_files = 0
            error_files = 0

            # Load all symbols details from configured sources
            local_data.ensure_data_dir_exists() # Ensure base data dir exists
            all_tickers_details: Dict[str, Dict[str, Optional[str]]] = local_data.load_tickers_and_data_from_sources(config.TICKER_SOURCES)
            logger.info(f"Loaded details for {len(all_tickers_details)} tickers from sources.")

            for ohlc_filepath in ohlc_files:
                filename = ohlc_filepath.name
                # Expecting filename like 'AMZN.csv', so the stem is the symbol
                symbol_base = ohlc_filepath.stem
                if not symbol_base: # Handle potential empty filenames or edge cases
                    logger.warning(f"Skipping file with empty stem: '{filename}'")
                    error_files += 1
                    continue

                symbol = symbol_base.upper() # Standardize to uppercase for DB operations

                logger.info(f"--- Processing symbol: {symbol} (File: {filename}) ---")

                try:
                    # 1. Get Symbol ID (Create if needed) using the standardized symbol
                    symbol_id = get_or_create_symbol_id(cursor, symbol)

                    # 2. Process and Insert Splits (if split file exists)
                    splits_inserted_count = 0
                    # Use the base symbol (e.g., 'AMZN') to find the corresponding split file
                    splits_json = read_split_data(symbol_base, SPLIT_DATA_DIR)
                    if splits_json:
                        splits_to_insert = parse_split_data_for_db(splits_json, symbol_id)
                        if splits_to_insert:
                            splits_inserted_count = insert_splits_batch(cursor, splits_to_insert)
                            logger.info(f"Inserted {splits_inserted_count} split records for symbol '{symbol}'.")
                            total_splits_inserted += splits_inserted_count
                        else:
                            logger.info(f"No valid splits parsed from file for symbol '{symbol}'.")
                    else:
                        logger.info(f"No split data file found or read for symbol '{symbol}'.")


                    # 3. Process and Insert OHLC Data from CSV
                    ohlc_inserted_count = 0
                    try:
                        # Read CSV using pandas
                        # Assuming standard columns: Date, Open, High, Low, Close, Volume
                        # Adjust column names if they differ in your CSVs
                        df_ohlc = pd.read_csv(ohlc_filepath, parse_dates=['Date'])

                        # Check for required columns (case-insensitive check)
                        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
                        if not required_cols.issubset({col.lower() for col in df_ohlc.columns}):
                             missing = required_cols - {col.lower() for col in df_ohlc.columns}
                             logger.error(f"Missing required columns in {filename}: {missing}. Skipping OHLC insert.")
                             raise ValueError("Missing required columns in CSV")

                        # Rename columns to match expected keys in parse_ohlc_data (Date, Open, etc.)
                        df_ohlc.columns = [col.capitalize() for col in df_ohlc.columns]

                        # Convert DataFrame to list of dictionaries (records)
                        # Handle potential NaNs or missing values appropriately before conversion if needed
                        # Convert 'Date' column back to ISO string format expected by parse_ohlc_data
                        df_ohlc['Date'] = df_ohlc['Date'].dt.strftime('%Y-%m-%dT%H:%M:%S') # Assuming T00:00:00 if only date
                        ohlc_data_list = df_ohlc.to_dict('records')

                    except pd.errors.EmptyDataError:
                        logger.warning(f"OHLC file {filename} is empty. Skipping.")
                        ohlc_data_list = [] # Ensure it's an empty list
                    except FileNotFoundError:
                         logger.error(f"OHLC file {filename} not found during read attempt (should not happen if glob worked). Skipping.")
                         raise # Treat as critical error
                    except Exception as e: # Catch other pandas or file reading errors
                        logger.error(f"Error reading or processing OHLC CSV {filename}: {e}")
                        raise # Treat as critical error for this file

                    ohlc_data_to_insert = parse_ohlc_data(ohlc_data_list, symbol_id)

                    if ohlc_data_to_insert:
                        ohlc_inserted_count = insert_ohlc_batch(cursor, ohlc_data_to_insert)
                        logger.info(f"Inserted {ohlc_inserted_count} OHLC records for symbol '{symbol}'.")
                        total_ohlc_inserted += ohlc_inserted_count
                    else:
                        logger.info(f"No valid OHLC data to insert for symbol '{symbol}'.")

                    # 4. Insert/Update Symbol Details - Use standardized symbol for lookup
                    if symbol in all_tickers_details:
                        details = all_tickers_details[symbol]
                        insert_symbol_details(cursor, symbol_id, details) # Use correct symbol_id
                        logger.info(f"Inserted/Updated details for symbol '{symbol}'.")
                    # Check using the original base symbol if the uppercase wasn't found
                    elif symbol_base in all_tickers_details:
                        logger.warning(f"Details found using original case symbol '{symbol_base}' but not standardized uppercase '{symbol}'. Using original case key.")
                        details = all_tickers_details[symbol_base]
                        insert_symbol_details(cursor, symbol_id, details) # Still use the correct symbol_id
                    else:
                        logger.warning(f"No details found for symbol '{symbol}' or original case '{symbol_base}' in ticker sources.")

                    processed_files += 1

                except (IOError, psycopg2.Error, ValueError, TypeError, KeyError, pd.errors.ParserError) as e: # Added pandas ParserError
                    logger.error(f"Failed to process file {filename} for symbol {symbol}: {e}")
                    error_files += 1
                    conn.rollback() # Rollback changes for this file/symbol
                    # Continue to the next file after rollback
                    logger.info(f"Rolled back transaction for symbol {symbol}. Continuing...")
                    continue # Explicitly continue to next file
                except Exception as e: # Catch any other unexpected errors
                    logger.error(f"An unexpected error occurred processing {filename} for symbol {symbol}: {e}", exc_info=True)
                    error_files += 1
                    conn.rollback()
                    # Decide whether to stop or continue. For robustness, let's continue.
                    logger.info(f"Rolled back transaction for symbol {symbol} due to unexpected error. Continuing...")
                    continue

            # Commit transaction only if all files processed successfully (or errors handled by rollback+continue)
            if error_files == 0:
                conn.commit()
                logger.info("--- Import Process Finished Successfully ---")
            else:
                # If we continued after errors, we might still want to commit the successful ones.
                # The current logic rolls back *per file* on error, so a final commit saves the good ones.
                conn.commit()
                logger.warning(f"--- Import Process Finished with {error_files} Errors ---")
                logger.warning("Changes for successfully processed symbols have been committed.")


            logger.info(f"Successfully processed: {processed_files} symbols")
            logger.info(f"Total OHLC records inserted/updated: {total_ohlc_inserted}")
            logger.info(f"Total Split records inserted/updated: {total_splits_inserted}")
            logger.info(f"Symbols with processing errors: {error_files}")


    except Exception as e:
        logger.error(f"A critical error occurred during the main process: {e}", exc_info=True)
        if conn:
            conn.rollback() # Ensure rollback on any top-level error
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    main()