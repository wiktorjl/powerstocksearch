# src/database/indicator_calculator.py
import logging
import psycopg2
from pathlib import Path
from src.database.connection import get_db_connection # Reuse the shared connection logic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load SQL for calculating indicators
sql_file = Path(__file__).resolve().parents[2] / 'sql' / 'indicators_calculator.sql'
try:
    CALCULATE_SMA_SQL = sql_file.read_text()
except Exception as e:
    logger.error(f"Failed to load SQL from {sql_file}: {e}")
    CALCULATE_SMA_SQL = ""

def calculate_and_insert_indicators():
    """Connects to the database and executes the SMA calculation SQL."""
    conn = None
    try:
        conn = get_db_connection()
        if conn is None:
            logger.error("Failed to establish database connection.")
            return False

        with conn.cursor() as cursor:
            logger.info("Starting SMA indicator calculation and insertion...")
            cursor.execute(CALCULATE_SMA_SQL)
            inserted_count = cursor.rowcount # Get affected rows for the last INSERT part
            logger.info(f"Finished SMA calculation. Affected rows in final insert step: {inserted_count}. Committing changes.")
            conn.commit()
            logger.info("Changes committed successfully.")
            return True

    except psycopg2.Error as e:
        logger.error(f"Database error during indicator calculation: {e}")
        if conn:
            conn.rollback()
            logger.warning("Transaction rolled back due to error.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    logger.info("--- Starting Indicator Calculation Script ---")
    success = calculate_and_insert_indicators()
    if success:
        logger.info("--- Indicator Calculation Script Finished Successfully ---")
    else:
        logger.error("--- Indicator Calculation Script Finished with Errors ---")
        exit(1) # Exit with error code if failed