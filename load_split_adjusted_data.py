import os
import json
import psycopg2
import psycopg2.extras
import logging
from datetime import datetime, timezone
from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Database Connection ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logging.info("Database connection established successfully.")
        return conn
    except psycopg2.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        raise

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
            logging.info(f"Symbol '{symbol}' not found. Inserting...")
            cursor.execute("INSERT INTO symbols (symbol) VALUES (%s) RETURNING symbol_id;", (symbol,))
            new_id = cursor.fetchone()[0]
            logging.info(f"Symbol '{symbol}' inserted with id {new_id}.")
            # No commit here, handled by the main transaction
            return new_id
    except psycopg2.Error as e:
        logging.error(f"Error getting or creating symbol_id for '{symbol}': {e}")
        raise  # Re-raise to allow transaction rollback

# --- Data Processing ---
def parse_ohlc_data(json_data, symbol_id):
    """Parses OHLCV data from JSON and prepares it for insertion."""
    parsed_data = []
    required_keys = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'} # Use capitalized keys

    for entry in json_data:
        if not required_keys.issubset(entry.keys()):
            logging.warning(f"Skipping entry due to missing keys: {entry}")
            continue
        try:
            # Ensure timestamp includes timezone information (assuming UTC if not specified)
            # Adjust parsing if your 'Date' format differs or includes timezone
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
            logging.warning(f"Skipping entry due to parsing error ({e}): {entry}")
            continue
        except KeyError as e:
             logging.warning(f"Skipping entry due to missing key '{e}': {entry}")
             continue

    return parsed_data

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
        logging.error(f"Error during batch insert: {e}")
        raise # Re-raise to allow transaction rollback

# --- Main Execution ---
def main():
    """Main function to orchestrate the data loading process."""
    data_dir = "splitadjusted"
    conn = None # Initialize conn to None

    if not os.path.isdir(data_dir):
        logging.error(f"Data directory '{data_dir}' not found.")
        return

    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
            logging.info(f"Found {len(json_files)} JSON files in '{data_dir}'.")

            total_inserted_count = 0
            processed_files = 0

            for filename in json_files:
                file_path = os.path.join(data_dir, filename)
                symbol = os.path.splitext(filename)[0] # Extract symbol from filename

                logging.info(f"Processing file: {filename} for symbol: {symbol}")

                try:
                    symbol_id = get_or_create_symbol_id(cursor, symbol)

                    with open(file_path, 'r') as f:
                        try:
                            json_data = json.load(f)
                        except json.JSONDecodeError as e:
                            logging.error(f"Error decoding JSON from {filename}: {e}")
                            continue # Skip this file

                    if not isinstance(json_data, list):
                         logging.error(f"Expected a list of records in {filename}, got {type(json_data)}. Skipping.")
                         continue

                    ohlc_data_to_insert = parse_ohlc_data(json_data, symbol_id)

                    if ohlc_data_to_insert:
                        inserted_count = insert_ohlc_batch(cursor, ohlc_data_to_insert)
                        logging.info(f"Inserted {inserted_count} records for symbol '{symbol}'.")
                        total_inserted_count += inserted_count
                    else:
                        logging.info(f"No valid data to insert for symbol '{symbol}'.")

                    processed_files += 1

                except (IOError, psycopg2.Error, ValueError, TypeError) as e:
                    logging.error(f"Failed to process file {filename}: {e}")
                    conn.rollback() # Rollback changes for this file
                    # Decide if you want to continue with other files or stop
                    # continue
                    # For now, let's re-raise to stop the whole process on error
                    raise
                except Exception as e: # Catch any other unexpected errors
                    logging.error(f"An unexpected error occurred processing {filename}: {e}")
                    conn.rollback()
                    raise # Stop the process

            # Commit transaction only if all files processed successfully (or handled errors appropriately)
            conn.commit()
            logging.info(f"Successfully processed {processed_files} files. Total records inserted/updated: {total_inserted_count}.")

    except Exception as e:
        logging.error(f"An error occurred during the main process: {e}")
        if conn:
            conn.rollback() # Ensure rollback on any top-level error
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()