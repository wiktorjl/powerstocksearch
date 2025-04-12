#!/usr/bin/env python3
"""
Data Provider Module

Provides functions to fetch financial data, initially focusing on OHLC data
from a PostgreSQL database.
"""

import logging
import psycopg2
import pandas as pd
from src.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT # Adjusted import
from src.database.connection import get_db_connection # Import shared connection function

# Configure logging for this module
logger = logging.getLogger(__name__)

# Removed local get_db_connection - using shared one from src.database.connection

def fetch_ohlc_data_db(symbol, start_date=None, end_date=None):
    """
    Fetch OHLC data for a given symbol from the database with optional date range.

    Args:
        symbol (str): The stock ticker symbol
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format

    Returns:
        pandas.DataFrame: DataFrame containing OHLC data or None if fetch fails
    """
    connection = get_db_connection()
    if not connection:
        return None

    # Base query
    query = """
    SELECT ohlc_data.timestamp,
           ohlc_data.open,
           ohlc_data.high,
           ohlc_data.low,
           ohlc_data.close,
           ohlc_data.volume
    FROM ohlc_data
    JOIN symbols ON ohlc_data.symbol_id = symbols.symbol_id
    WHERE symbols.symbol = %s
    """

    params = [symbol]

    # Add date filters if provided
    if start_date:
        query += " AND ohlc_data.timestamp >= %s"
        params.append(start_date)

    if end_date:
        query += " AND ohlc_data.timestamp <= %s"
        params.append(end_date)

    query += " ORDER BY ohlc_data.timestamp;"

    try:
        logger.info(f"Fetching data for {symbol}" +
                   (f" from {start_date}" if start_date else "") +
                   (f" to {end_date}" if end_date else ""))

        df = pd.read_sql_query(query, connection, params=params)
        connection.close()

        if df.empty:
            logger.warning(f"No data found for symbol {symbol}")
            return None

        logger.info(f"Retrieved {len(df)} data points")
        return df

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        if connection:
            connection.close()
        return None

# Potential future expansion:
# def fetch_ohlc_data_api(symbol, start_date=None, end_date=None):
#     # Placeholder for fetching data from an API
#     pass

# def fetch_ohlc_data_file(filepath):
#     # Placeholder for fetching data from a file
#     pass