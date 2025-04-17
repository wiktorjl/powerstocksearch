#!/usr/bin/env python3
"""
Data Provider Module

Provides functions to fetch financial data, initially focusing on OHLC data
from a PostgreSQL database.
"""

import logging
import psycopg2
import pandas as pd
from decimal import Decimal # Import Decimal for type checking if needed
from src.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT # Adjusted import
from src.database.connection import get_db_connection # Import shared connection function

# Configure logging for this module
logger = logging.getLogger(__name__)

# Removed local get_db_connection - using shared one from src.database.connection

def fetch_ohlc_data_db(symbol, start_date=None, end_date=None):
    """
    Fetch **split-adjusted** OHLC data for a given symbol from the database
    with optional date range. Adjustment uses the 'splits' table.

    Args:
        symbol (str): The stock ticker symbol
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format

    Returns:
        pandas.DataFrame: DataFrame containing split-adjusted OHLC data or None if fetch fails
    """
    connection = get_db_connection()
    if not connection:
        return None

    # Base query
    query = """
    SELECT o.timestamp,
           o.open,
           o.high,
           o.low,
           o.close,
           o.volume,
           s.symbol_id -- Include symbol_id for split lookup
    FROM ohlc_data o
    JOIN symbols s ON o.symbol_id = s.symbol_id
    WHERE s.symbol = %s
    """

    params = [symbol]

    # Add date filters if provided
    if start_date:
        query += " AND o.timestamp >= %s"
        params.append(start_date)

    if end_date:
        query += " AND o.timestamp <= %s"
        params.append(end_date)

    query += " ORDER BY o.timestamp;"

    try:
        logger.info(f"Fetching data for {symbol}" +
                   (f" from {start_date}" if start_date else "") +
                   (f" to {end_date}" if end_date else ""))

        # Fetch raw data first
        df = pd.read_sql_query(query, connection, params=params)

        if df.empty:
            logger.warning(f"No raw OHLC data found for symbol {symbol}")
            connection.close()
            return None

        logger.info(f"Retrieved {len(df)} raw OHLC data points for {symbol}. Checking for splits...")

        # Ensure timestamp is datetime type for comparison
        # Ensure timestamp is datetime type and convert to UTC for consistent comparison
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Get symbol_id from the first row (it's the same for all rows in this query)
        symbol_id = df['symbol_id'].iloc[0]

        # Fetch splits using the standard column name
        splits_query = """
        SELECT split_date, ratio
        FROM splits
        WHERE symbol_id = %s
        ORDER BY split_date DESC;
        """
        # Convert numpy.int64 symbol_id to standard Python int for psycopg2 compatibility
        splits_df = pd.read_sql_query(splits_query, connection, params=[int(symbol_id)])

        connection.close() # Close connection after all DB queries

        if not splits_df.empty:
            logger.info(f"Found {len(splits_df)} splits for symbol {symbol} (ID: {symbol_id}). Applying adjustments...")
            # Ensure split_date is datetime type
            # Ensure split_date is datetime type and convert to UTC for consistent comparison
            # Ensure split_date is datetime type (using correct column name) and convert to UTC
            # Ensure split_date is datetime type and convert to UTC
            splits_df['split_date'] = pd.to_datetime(splits_df['split_date'], utc=True)

            # Apply splits using backward propagation
            for index, split in splits_df.iterrows():
                # Access the column using the correct name with the leading space
                # Access the column using the standard name
                split_date = split['split_date']
                # Ratio from DB is the adjustment factor (Old/New)
                # Ensure ratio is float, not Decimal, for pandas operations
                adj_factor = float(split['ratio'])

                if adj_factor <= 0:
                    logger.warning(f"Skipping split on {split_date.date()} for {symbol} due to invalid ratio: {adj_factor}")
                    continue

                # Apply adjustment to rows *before* the split date
                mask = df['timestamp'] < split_date
                if mask.any():
                    # Adjust Price columns (O, H, L, C)
                    price_cols = ['open', 'high', 'low', 'close']
                    df.loc[mask, price_cols] = df.loc[mask, price_cols] * adj_factor

                    # Adjust Volume (inverse factor)
                    # Volume factor = New / Old = 1 / adj_factor
                    if adj_factor != 0:
                        volume_factor = 1.0 / adj_factor
                        # Calculate new volume, round, and convert to int before assignment
                        # Ensure volume is numeric first, handle potential non-numeric data if necessary
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0) # Handle potential errors/NaN
                        new_volume = (df.loc[mask, 'volume'] * volume_factor).round().astype(int)
                        df.loc[mask, 'volume'] = new_volume
                    else:
                         logger.warning(f"Cannot adjust volume for split on {split_date.date()} for {symbol} due to zero adjustment factor.")

                    logger.info(f"Applied split ratio {adj_factor:.4f} on {split_date.date()} for {symbol}, adjusted {mask.sum()} records.")
        else:
            logger.info(f"No splits found in database for symbol {symbol}. Returning unadjusted data.")

        # Drop the symbol_id column before returning, as it's not part of the original OHLC schema expected
        df = df.drop(columns=['symbol_id'])

        logger.info(f"Returning {len(df)} split-adjusted data points for {symbol}")
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
    Fetch details for a specific symbol by joining 'symbols', 'symbols_details',
    and 'company_profile' tables.

    Args:
        symbol (str): The stock ticker symbol.

    Returns:
        dict: A dictionary containing available symbol details from company_profile,
              or None if not found or error.
    """
    connection = get_db_connection()
    if not connection:
        return None

    # Query joins symbols, symbols_details, and company_profile
    # using INNER JOINs to ensure all details exist.
    query = """
    SELECT
        s.symbol_id,
        s.symbol,
        sd.sector,          -- From symbols_details
        sd.subsector,       -- From symbols_details
        sd.name AS details_name, -- Alias to distinguish from cp.name if needed, though schema shows name only in sd
        cp.country,         -- From company_profile
        cp.currency,        -- From company_profile
        cp.exchange,        -- From company_profile
        cp.finnhub_industry,-- From company_profile
        cp.ipo,             -- From company_profile
        cp.logo,            -- From company_profile
        cp.market_capitalization, -- From company_profile
        cp.name AS profile_name, -- Alias to distinguish from sd.name
        cp.phone,           -- From company_profile
        cp.share_outstanding, -- From company_profile
        cp.ticker,          -- From company_profile
        cp.weburl           -- From company_profile
    FROM symbols s
    INNER JOIN symbols_details sd ON s.symbol_id = sd.symbol_id
    INNER JOIN company_profile cp ON s.symbol_id = cp.symbol_id
    WHERE s.symbol = %s;
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
            # Map results to dictionary based on the columns selected in the corrected query
            # Map results based on the new query structure
            details = {
                'symbol_id': result[0],
                'symbol': result[1],
                'sector': result[2],         # From symbols_details
                'subsector': result[3],      # From symbols_details
                'name': result[4] or result[12], # Use name from symbols_details, fallback to company_profile if needed (though schema shows it only in sd)
                'country': result[5],        # From company_profile
                'currency': result[6],       # From company_profile
                'exchange': result[7],       # From company_profile
                'finnhub_industry': result[8],# From company_profile
                'ipo': result[9],            # From company_profile
                'logo': result[10],           # From company_profile
                'market_capitalization': result[11], # From company_profile
                # 'profile_name': result[12], # Included if needed, but 'name' (result[4]) is likely primary
                'phone': result[13],          # From company_profile
                'share_outstanding': result[14],# From company_profile
                'ticker': result[15],         # From company_profile
                'weburl': result[16]          # From company_profile
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