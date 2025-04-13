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


def list_symbols_db(prefix=None):
    """
    Fetch a list of all symbols from the database, optionally filtered by prefix.

    Args:
        prefix (str, optional): A prefix to filter symbols by. Defaults to None (no filter).

    Returns:
        list: A list of symbol strings, or None if fetch fails.
    """
    connection = get_db_connection()
    if not connection:
        return None

    try:
        cursor = connection.cursor()
        if prefix:
            query = "SELECT symbol FROM symbols WHERE symbol ILIKE %s ORDER BY symbol;"
            # Use ILIKE for case-insensitive matching and add wildcard
            cursor.execute(query, (prefix + '%',))
            logger.info(f"Fetching symbols starting with prefix: {prefix}")
        else:
            query = "SELECT symbol FROM symbols ORDER BY symbol;"
            cursor.execute(query)
            logger.info("Fetching all symbols")

        symbols = [item[0] for item in cursor.fetchall()]
        cursor.close()
        connection.close()

        if not symbols:
            logger.warning(f"No symbols found" + (f" with prefix {prefix}" if prefix else ""))
            return [] # Return empty list instead of None for no results

        logger.info(f"Retrieved {len(symbols)} symbols")
        return symbols

    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        if connection:
            connection.close()
        return None

def fetch_symbol_details_db(symbol):
    """
    Fetch details for a specific symbol from the database.
    (Currently fetches basic info like symbol_id, name, exchange)

    Args:
        symbol (str): The stock ticker symbol.

    Returns:
        dict: A dictionary containing symbol details, or None if not found or error.
    """
    connection = get_db_connection()
    if not connection:
        return None

    query = """
    SELECT symbol_id, symbol, sector, subsector, name, "timestamp", open, high, low, close, volume
	FROM public.symbol_info_basic 
    WHERE symbol = %s;
    """

    try:
        cursor = connection.cursor()
        logger.info(f"Fetching details for symbol: {symbol}")
        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        cursor.close()
        connection.close()

        if result:
            # Convert row to dictionary
            details = {
                'symbol_id': result[0],
                'symbol': result[1],
                'sector': result[2],
                'subsector': result[3],
                'name': result[4],
                'timestamp': result[5],
                'open': result[6],
                'high': result[7],
                'low': result[8],
                'close': result[9],
                'volume': result[10]
            }
            logger.info(f"Retrieved details for {symbol}")
            return details
        else:
            logger.warning(f"No details found for symbol {symbol}")
            return None

    except Exception as e:
        logger.error(f"Error fetching symbol details for {symbol}: {e}")
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