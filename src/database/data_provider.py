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
    SELECT symbol_id, symbol, sector, subsector, name, "timestamp", open, high, low, close, volume, logo, weburl
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
                'volume': result[10],
                'logo': result[11],      # Use the actual column name 'logo'
                'weburl': result[12]     # Use the actual column name 'weburl'
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

def fetch_latest_indicators_db(symbol_id):
    """
    Fetch the latest indicator values for a given symbol_id from the database.

    Args:
        symbol_id (int): The ID of the symbol.

    Returns:
        list: A list of dictionaries, each containing 'description' and 'value'
              for the latest indicators. Returns None if error or no indicators found.
    """
    connection = get_db_connection()
    if not connection:
        return None

    # Query to get the latest timestamp for the symbol in the indicators table
    latest_timestamp_query = """
    SELECT MAX(timestamp)
    FROM indicators
    WHERE symbol_id = %s;
    """

    # Query to get indicator descriptions and values for the symbol at the latest timestamp
    indicators_query = """
    SELECT idf.description, i.value
    FROM indicators i
    JOIN indicator_definitions idf ON i.indicator_id = idf.indicator_id
    WHERE i.symbol_id = %s AND i.timestamp = %s;
    """

    try:
        cursor = connection.cursor()
        logger.info(f"Fetching latest timestamp for indicators for symbol_id: {symbol_id}")
        cursor.execute(latest_timestamp_query, (symbol_id,))
        latest_timestamp_result = cursor.fetchone()

        if not latest_timestamp_result or not latest_timestamp_result[0]:
            logger.warning(f"No indicator timestamp found for symbol_id: {symbol_id}")
            cursor.close()
            connection.close()
            return None # Or maybe an empty dict {}? Let's return None for now.

        latest_timestamp = latest_timestamp_result[0]
        logger.info(f"Latest indicator timestamp for symbol_id {symbol_id} is {latest_timestamp}")

        logger.info(f"Fetching indicators for symbol_id {symbol_id} at timestamp {latest_timestamp}")
        cursor.execute(indicators_query, (symbol_id, latest_timestamp))
        indicator_results = cursor.fetchall()
        cursor.close()
        connection.close()

        if not indicator_results:
            logger.warning(f"No indicators found for symbol_id {symbol_id} at timestamp {latest_timestamp}")
            return None # Or empty dict {}?

        # Convert list of tuples to list of dictionaries
        # Ensure value is converted from Decimal to float for JSON serialization if needed
        indicators_list = [
            {'description': row[0], 'value': float(row[1]) if row[1] is not None else None}
            for row in indicator_results
        ]
        logger.info(f"Retrieved {len(indicators_list)} indicators for symbol_id {symbol_id}")
        return indicators_list

    except Exception as e:
        logger.error(f"Error fetching latest indicators for symbol_id {symbol_id}: {e}")
        if connection:
            connection.close()
        return None




def get_unique_sectors():
    """
    Fetch a list of unique industry/sector values from the company_info table.

    Returns:
        list: A list of unique sector strings, or None if fetch fails.
    """
    connection = get_db_connection()
    if not connection:
        return None

    query = "SELECT DISTINCT finnhub_industry FROM company_profile WHERE finnhub_industry IS NOT NULL AND finnhub_industry != '' ORDER BY finnhub_industry;"

    try:
        cursor = connection.cursor()
        logger.info("Fetching unique sectors (finnhub_industry from company_profile)")
        cursor.execute(query)
        sectors = [item[0] for item in cursor.fetchall()]
        cursor.close()
        connection.close()

        logger.info(f"Retrieved {len(sectors)} unique sectors")
        return sectors

    except Exception as e:
        logger.error(f"Error fetching unique sectors: {e}")
        if connection:
            connection.close()
        return None

def scan_stocks(sector=None, close_op=None, close_val=None, vol_op=None, vol_val=None, page=1, per_page=20):
    """
    Scan stocks based on sector, latest close price, and latest volume criteria,
    with pagination.

    Args:
        sector (str, optional): Filter by sector (industry).
        close_op (str, optional): Operator for close price (e.g., '>', '<', '=', '>=', '<=').
        close_val (float, optional): Value for close price comparison.
        vol_op (str, optional): Operator for volume (e.g., '>', '<', '=', '>=', '<=').
        vol_val (int, optional): Value for volume comparison.
        page (int): The current page number (1-based).
        per_page (int): Number of results per page.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries representing the matching stocks
                    (keys: symbol, name, sector).
            - int: The total number of matching stocks (before pagination).
            - str: An error message string if an error occurred, otherwise None.
        Returns (None, 0, "Database connection failed") if connection fails initially.
        Returns (None, 0, error_message) if any other error occurs during query execution.
    """
    connection = get_db_connection()
    if not connection:
        return None, 0, "Database connection failed"

    # Explicitly join symbols, company_profile, and latest ohlc_data
    # Subquery to get the latest timestamp for each symbol_id in ohlc_data
    latest_ohlc_subquery = """
    SELECT DISTINCT ON (symbol_id)
           symbol_id, "timestamp", "close", volume
    FROM ohlc_data
    ORDER BY symbol_id, "timestamp" DESC
    """

    base_query = f"""
    FROM symbols s
    JOIN company_profile cp ON s.symbol_id = cp.symbol_id
    JOIN ({latest_ohlc_subquery}) latest_dd ON s.symbol_id = latest_dd.symbol_id
    WHERE 1=1
    """
    filters = []
    params = []

    # Validate and map operators safely
    valid_operators = {'>': '>', '<': '<', '=': '=', '>=': '>=', '<=': '<='}

    if sector:
        filters.append("cp.finnhub_industry = %s") # Filter on the actual table column
        params.append(sector)

    if close_op and close_val is not None:
        if close_op in valid_operators:
            try:
                # Ensure close_val is a float
                close_val_float = float(close_val)
                filters.append(f"latest_dd.close {valid_operators[close_op]} %s")
                params.append(close_val_float)
            except ValueError:
                logger.error(f"Invalid close value provided: {close_val}")
                connection.close()
                return None, 0, f"Invalid close value: {close_val}. Must be a number."
        else:
            logger.warning(f"Invalid close operator provided: {close_op}")
            connection.close()
            return None, 0, f"Invalid close operator: {close_op}"


    if vol_op and vol_val is not None:
        if vol_op in valid_operators:
            try:
                 # Ensure vol_val is an integer
                vol_val_int = int(vol_val)
                filters.append(f"latest_dd.volume {valid_operators[vol_op]} %s")
                params.append(vol_val_int)
            except ValueError:
                logger.error(f"Invalid volume value provided: {vol_val}")
                connection.close()
                return None, 0, f"Invalid volume value: {vol_val}. Must be an integer."
        else:
            logger.warning(f"Invalid volume operator provided: {vol_op}")
            connection.close()
            return None, 0, f"Invalid volume operator: {vol_op}"

    where_clause = " AND ".join(filters) if filters else ""
    if where_clause:
        base_query += " AND " + where_clause

    count_query = f"SELECT COUNT(*) {base_query}"
    results_query = f'SELECT s.symbol, cp.name, cp.finnhub_industry AS sector, latest_dd.close, latest_dd.volume {base_query} ORDER BY s.symbol LIMIT %s OFFSET %s'

    offset = (page - 1) * per_page
    results_params = params + [per_page, offset]

    try:
        cursor = connection.cursor()

        # Get total count
        logger.debug(f"Executing count query: {count_query} with params: {params}")
        cursor.execute(count_query, tuple(params)) # Pass params as tuple
        total_count = cursor.fetchone()[0]
        logger.info(f"Total matching stocks found: {total_count}")

        # Get paginated results
        logger.debug(f"Executing results query: {results_query} with params: {results_params}")
        cursor.execute(results_query, tuple(results_params)) # Pass params as tuple
        results = cursor.fetchall()

        # Convert results to list of dicts
        columns = [desc[0] for desc in cursor.description]
        stock_list = [dict(zip(columns, row)) for row in results]

        cursor.close()
        connection.close()

        logger.info(f"Retrieved {len(stock_list)} stocks for page {page}")
        return stock_list, total_count, None # No error

    except Exception as e:
        logger.error(f"Error scanning stocks: {e}")
        logger.error(f"Failed Query (Count): {count_query} with params: {params}")
        logger.error(f"Failed Query (Results): {results_query} with params: {results_params}")
        if connection:
            connection.close()
        return None, 0, f"Error scanning stocks: {e}"

# Potential future expansion:
# def fetch_ohlc_data_api(symbol, start_date=None, end_date=None):
#     # Placeholder for fetching data from an API
#     pass

# def fetch_ohlc_data_file(filepath):
#     # Placeholder for fetching data from a file
#     pass