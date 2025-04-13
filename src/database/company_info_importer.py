import os
import json
import logging
import psycopg2
from psycopg2 import sql
import src.config as config
from src.database.connection import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use the directory defined in the central configuration
DATA_DIR = config.COMPANY_INFO_DIR
TABLE_NAME = 'company_profile'

def import_company_info(data_dir=DATA_DIR): # Keep parameter for potential flexibility, but default to config
    """
    Imports company information from JSON files into the database.

    Args:
        data_dir (str): The directory containing the JSON files.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Failed to get database connection. Aborting import.")
        return

    cursor = conn.cursor()
    logger.info(f"Starting import process from directory: {data_dir}")
    imported_count = 0
    error_count = 0

    # Ensure the data directory exists
    if not os.path.isdir(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        conn.close()
        return

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(data_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Map JSON keys to database columns (snake_case)
                # Handle potential missing keys or None values gracefully
                company_data = {
                    'country': data.get('country'),
                    'currency': data.get('currency'),
                    'estimate_currency': data.get('estimateCurrency'),
                    'exchange': data.get('exchange'),
                    'finnhub_industry': data.get('finnhubIndustry'),
                    'ipo': data.get('ipo') if data.get('ipo') else None, # Ensure empty strings become NULL
                    'logo': data.get('logo'),
                    'market_capitalization': data.get('marketCapitalization'),
                    'name': data.get('name'),
                    'phone': data.get('phone'),
                    'share_outstanding': data.get('shareOutstanding'),
                    'ticker': data.get('ticker'),
                    'weburl': data.get('weburl')
                }

                # Basic validation: Ensure ticker exists as it's likely a primary identifier
                if not company_data['ticker']:
                    logger.warning(f"Skipping file {filename}: Missing 'ticker'.")
                    continue

                # Store ticker for logging before potential deletion
                ticker = company_data.get('ticker')
                symbol_id = None # Initialize symbol_id

                # Look up symbol_id from the symbols table
                try:
                    cursor.execute("SELECT symbol_id FROM symbols WHERE symbol = %s", (ticker,))
                    result = cursor.fetchone()
                    if result:
                        symbol_id = result[0]
                    else:
                        logger.warning(f"Skipping file {filename}: Ticker '{ticker}' not found in symbols table.")
                        error_count += 1
                        continue # Skip to the next file
                except psycopg2.Error as db_err:
                    logger.error(f"Database error looking up symbol_id for ticker {ticker} in {filename}: {db_err}")
                    conn.rollback() # Rollback potentially needed if transaction started implicitly
                    error_count += 1
                    continue # Skip to the next file

                # Prepare data for insertion/update
                # Add symbol_id and remove ticker (assuming stock_data uses symbol_id as FK)
                company_data['symbol_id'] = symbol_id
                # if 'ticker' in company_data: # Only delete if it exists
                #     del company_data['ticker']

                # Construct the INSERT statement dynamically
                columns = list(company_data.keys())
                values = [company_data[col] for col in columns]

                # Use ON CONFLICT to handle existing symbol_ids (UPSERT)
                # Assumes 'symbol_id' is a unique constraint or primary key in stock_data
                insert_statement = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(TABLE_NAME),
                    sql.SQL(', ').join(map(sql.Identifier, columns)),
                    sql.SQL(', ').join(sql.Placeholder() * len(values))
                )

                # Execute the insert statement
                cursor.execute(insert_statement, values)
                imported_count += 1
                # logger.debug(f"Successfully imported data for symbol_id: {symbol_id} (ticker: {ticker})")

            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from file: {filename}")
                error_count += 1
            except psycopg2.Error as db_err:
                logger.error(f"Database error importing data from {filename} for ticker {ticker} (symbol_id: {symbol_id}): {db_err}")
                conn.rollback() # Rollback the transaction on error for this file
                error_count += 1
            except Exception as e:
                logger.error(f"Unexpected error processing file {filename}: {e}")
                conn.rollback()
                error_count += 1
        else:
            logger.debug(f"Skipping non-JSON file: {filename}")


    # Commit changes if no errors occurred during the loop for any file (or handle partial commits)
    # For simplicity, we commit at the end. Consider transaction management per file if needed.
    if error_count == 0:
        conn.commit()
        logger.info(f"Import completed successfully. Imported {imported_count} records.")
    else:
        # Decide if you want to commit partial data or rollback all
        # conn.rollback() # Uncomment this to rollback everything if any error occurs
        conn.commit() # Commit successful imports even if some errors occurred
        logger.warning(f"Import completed with {error_count} errors. Imported {imported_count} records successfully (committed).")


    cursor.close()
    conn.close()
    logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info("Company Info Importer script started.")
    # Example usage: You might want to pass the data directory via command-line arguments in a real application
    # For now, it uses the default DATA_DIR
    # Call the function, it will use the default DATA_DIR from config
    import_company_info()
    logger.info("Company Info Importer script finished.")
