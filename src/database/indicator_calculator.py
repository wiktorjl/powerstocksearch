# src/database/indicator_calculator.py
import logging
import psycopg2
from src.database.connection import get_db_connection # Reuse the shared connection logic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# SQL statements to calculate and insert SMAs
CALCULATE_SMA_SQL = """
-- Ensure indicator definitions exist
INSERT INTO indicator_definitions (name, description)
VALUES
('SMA_5', '5-day Simple Moving Average'),
('SMA_10', '10-day Simple Moving Average'),
('SMA_100', '100-day Simple Moving Average'),
('SMA_150', '150-day Simple Moving Average')
ON CONFLICT (name) DO NOTHING; -- Avoid errors if they already exist

-- Calculate SMAs using window functions and insert into indicators table
WITH sma_data AS (
    SELECT
        timestamp,
        symbol_id,
        AVG(close) OVER w5 AS sma_5,
        AVG(close) OVER w10 AS sma_10,
        AVG(close) OVER w100 AS sma_100,
        AVG(close) OVER w150 AS sma_150
    FROM
        ohlc_data -- Assumes ohlc_data is populated by the importer
    WINDOW
        w5 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
        w10 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
        w100 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),
        w150 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 149 PRECEDING AND CURRENT ROW)
),
indicator_ids AS (
    SELECT indicator_id, name FROM indicator_definitions WHERE name IN ('SMA_5', 'SMA_10', 'SMA_100', 'SMA_150')
)
-- Insert calculated indicators, avoiding duplicates
INSERT INTO indicators (timestamp, symbol_id, indicator_id, value)
SELECT s.timestamp, s.symbol_id, i.indicator_id, s.sma_5
FROM sma_data s JOIN indicator_ids i ON i.name = 'SMA_5'
WHERE s.sma_5 IS NOT NULL
UNION ALL
SELECT s.timestamp, s.symbol_id, i.indicator_id, s.sma_10
FROM sma_data s JOIN indicator_ids i ON i.name = 'SMA_10'
WHERE s.sma_10 IS NOT NULL
UNION ALL
SELECT s.timestamp, s.symbol_id, i.indicator_id, s.sma_100
FROM sma_data s JOIN indicator_ids i ON i.name = 'SMA_100'
WHERE s.sma_100 IS NOT NULL
UNION ALL
SELECT s.timestamp, s.symbol_id, i.indicator_id, s.sma_150
FROM sma_data s JOIN indicator_ids i ON i.name = 'SMA_150'
WHERE s.sma_150 IS NOT NULL
ON CONFLICT (timestamp, symbol_id, indicator_id) DO NOTHING; -- Apply conflict resolution once at the end
"""

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