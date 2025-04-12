#!/usr/bin/env python3
"""
OHLC Plotter - A tool for creating professional candlestick charts from database data.

This script fetches OHLC (Open, High, Low, Close) data for a specified stock symbol
from a PostgreSQL database and renders a professional-looking candlestick chart,
optionally including calculated support and resistance levels.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, AlgorithmConfig
# Import the support/resistance calculation function
from support_resistance import identify_support_resistance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Establish a connection to the PostgreSQL database.

    Returns:
        psycopg2.connection: Database connection object or None if connection fails
    """
    try:
        connection = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logger.info("Successfully connected to the database")
        return connection
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        return None

def fetch_ohlc_data(symbol, start_date=None, end_date=None):
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

def resample_dataframe(df, timeframe='daily'):
    """
    Resample the dataframe to the specified timeframe.

    Args:
        df (pandas.DataFrame): DataFrame with datetime index
        timeframe (str): Timeframe to resample to ('daily', 'weekly', or 'monthly')

    Returns:
        pandas.DataFrame: Resampled dataframe
    """
    if timeframe == 'daily' or df is None or df.empty:
        return df

    # Define the rule for resampling
    if timeframe == 'weekly':
        rule = 'W'  # Weekly resampling
    elif timeframe == 'monthly':
        rule = 'M'  # Monthly resampling
    else:
        logger.warning(f"Unknown timeframe '{timeframe}', using daily")
        return df

    # Resample the dataframe
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else None
    })

    # Drop any rows with NaN values
    resampled.dropna(inplace=True)

    return resampled

def process_dataframe(df, timeframe='daily'):
    """
    Process the DataFrame to prepare it for plotting.

    Args:
        df (pandas.DataFrame): Raw DataFrame from database
        timeframe (str): Timeframe to resample to ('daily', 'weekly', or 'monthly')

    Returns:
        pandas.DataFrame: Processed DataFrame ready for mplfinance
    """
    if df is None or df.empty:
        return None

    # Make a copy to avoid modifying the original
    df_plot = df.copy()

    # Ensure timestamp column is datetime and set as index
    # Convert timestamp to UTC naive datetime64 to handle potential timezone awareness
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], utc=True)
    df_plot.set_index('timestamp', inplace=True)

    # Ensure all necessary columns are numeric
    for col in ['open', 'high', 'low', 'close']:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

    # Handle volume if present
    if 'volume' in df_plot.columns:
        df_plot['volume'] = pd.to_numeric(df_plot['volume'], errors='coerce')

    # Check for missing data
    if df_plot.isnull().any().any():
        logger.warning("Some data points contain NaN values and will be dropped")
        df_plot.dropna(inplace=True)

    # Resample the dataframe if needed
    if timeframe != 'daily':
        df_plot = resample_dataframe(df_plot, timeframe)
        if df_plot is None or df_plot.empty:
            logger.error(f"Failed to resample data to {timeframe} timeframe")
            return None
        logger.info(f"Resampled data to {timeframe} timeframe, resulting in {len(df_plot)} data points")

    return df_plot

# Function signature updated to accept combined sr_levels
def plot_candlestick(df_plot, symbol, output_file=None, timeframe='daily', title=None, sr_levels=None):
    """
    Plot a professional candlestick chart using mplfinance, manually drawing S/R lines.

    Args:
        df_plot (pandas.DataFrame): DataFrame containing OHLC data for the desired plot period (processed and filtered).
        symbol (str): Stock ticker symbol
        output_file (str, optional): Path to save the plot image
        timeframe (str): Timeframe for the chart ('daily', 'weekly', or 'monthly')
        title (str, optional): Custom title for the plot
        sr_levels (list, optional): List of combined top S/R levels to potentially plot.

    Returns:
        bool: True if plot was successfully created, False otherwise
    """
    if df_plot is None or df_plot.empty:
        logger.error("No data available to plot")
        return False

    # Define custom style
    mc = mpf.make_marketcolors(
        up='green',
        down='red',
        edge='black',
        wick={'up': 'green', 'down': 'red'},
        volume={'up': 'green', 'down': 'red'},
    )

    s = mpf.make_mpf_style(
        marketcolors=mc,
        gridstyle='--',
        y_on_right=False,
        facecolor='white',
        edgecolor='black',
        figcolor='white',
        gridcolor='gray',
        rc={'font.size': 10}
    )

    # S/R lines will be drawn manually after the main plot

    # Create figure and primary axis
    try:
        # Plot without hlines initially
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style=s,
            title=title or f'{timeframe.capitalize()} Candlestick Chart for {symbol}',
            ylabel='Price',
            volume=True if 'volume' in df_plot.columns else False,
            show_nontrading=False,
            datetime_format='%Y-%m-%d',
            xrotation=45,
            returnfig=True # Crucial to get axes object
            # No hlines argument here
        )

        # --- Manually draw S/R lines ---
        if sr_levels and axes and hasattr(axes[0], 'axhline'): # Check if axes exist and have axhline method
            try:
                # Get current price from the last data point being plotted
                current_price = df_plot['close'].iloc[-1]
                max_lines_per_type = AlgorithmConfig.MAX_SR_LINES

                # Separate, sort, and limit
                support = sorted([lvl for lvl in sr_levels if isinstance(lvl, (int, float)) and lvl <= current_price], reverse=True) # Highest support first
                resistance = sorted([lvl for lvl in sr_levels if isinstance(lvl, (int, float)) and lvl > current_price]) # Lowest resistance first

                final_support = support[:max_lines_per_type]
                final_resistance = resistance[:max_lines_per_type]

                # Draw support lines (green) using the first axes panel (price panel)
                price_ax = axes[0] # Typically the main price panel
                for level in final_support:
                    price_ax.axhline(y=level, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
                    logger.debug(f"Drawing support line at {level:.2f}")

                # Draw resistance lines (red)
                for level in final_resistance:
                    price_ax.axhline(y=level, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
                    logger.debug(f"Drawing resistance line at {level:.2f}")

                if final_support or final_resistance:
                    logger.info(f"Manually drew {len(final_support)} support and {len(final_resistance)} resistance lines.")
                else:
                    logger.info("No S/R lines within the top threshold to draw for the plotted period.")

            except IndexError:
                 logger.error("Could not access axes[0] to draw S/R lines. Plot might be empty or structure unexpected.")
            except Exception as draw_err:
                 logger.error(f"Error manually drawing S/R lines: {draw_err}")
        elif sr_levels:
             logger.warning("S/R levels provided, but could not get valid axes object to draw on.")
        # --- End manual drawing ---

    except Exception as e:
        logger.error(f"Error during mplfinance plot generation: {e}")
        # Attempt to show plot even if saving fails later
        try: plt.show()
        except: pass
        return False


    # Add grid to price panel (axes[0] is the main price panel)
    if axes and len(axes) > 0:
        axes[0].grid(True, linestyle='--', alpha=0.3)

    # Customize title (mpf.plot handles the main title, this sets the figure suptitle)
    if title:
        fig.suptitle(title, fontsize=14, color='black')
    else:
        # Generate date range string safely
        date_range_str = ""
        if not df_plot.empty:
            try:
                start_dt_str = df_plot.index[0].strftime('%Y-%m-%d')
                end_dt_str = df_plot.index[-1].strftime('%Y-%m-%d')
                date_range_str = f"({start_dt_str} to {end_dt_str})"
            except Exception as e:
                logger.warning(f"Could not format date range for title: {e}")
        fig.suptitle(f'{timeframe.capitalize()} Candlestick Chart for {symbol} {date_range_str}', fontsize=14, color='black')

    # Save to file if output_file is provided
    if output_file:
        try:
            # Ensure directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Chart saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save chart to file '{output_file}': {e}")
            plt.show() # Show plot if saving failed
            return False
    else:
        plt.show()

    plt.close(fig) # Close the figure after showing or saving
    return True

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate professional candlestick charts from database data')

    parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    parser.add_argument('--start', '-s', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', '-d', type=int, help='Number of most recent days/periods to plot')
    parser.add_argument('--timeframe', '-tf', type=str, choices=['daily', 'weekly', 'monthly'],
                        default='daily', help='Timeframe for the chart (daily, weekly, monthly)')
    parser.add_argument('--output', '-o', type=str, help='Output file path to save the chart')
    parser.add_argument('--title', '-t', type=str, help='Custom title for the chart')
    # Add argument for support/resistance
    parser.add_argument('--sr', action='store_true', help='Calculate and plot support/resistance levels')

    return parser.parse_args()

def main():
    """Main function to orchestrate the script execution."""
    # Parse command line arguments
    try:
        args = parse_arguments()
    except SystemExit:
        # This happens when --help is called or there's an error in arguments
        return

    # Assign args to variables
    symbol = args.symbol.strip().upper()
    start_date = args.start
    end_date = args.end
    days = args.days
    timeframe = args.timeframe
    output_file = args.output
    title = args.title
    plot_sr = args.sr # Get the boolean flag for S/R

    # Fetch data
    raw_df = fetch_ohlc_data(symbol, start_date, end_date)

    if raw_df is None or raw_df.empty:
        logger.error(f"No data available for {symbol}")
        return # Exit if no data

    # Process dataframe for plotting (sets index, resamples if needed)
    df_processed = process_dataframe(raw_df, timeframe)

    if df_processed is None or df_processed.empty:
        logger.error(f"Data processing failed for {symbol}")
        return

    # Filter for the requested number of days/periods *before* S/R calculation
    df_plot_final = df_processed.copy()
    if days and days > 0:
        if len(df_plot_final) > days:
            df_plot_final = df_plot_final.iloc[-days:]
            logger.info(f"Using most recent {days} periods for plot and S/R calculation")
        else:
             logger.info(f"Using all available {len(df_plot_final)} periods (less than requested {days}) for plot and S/R calculation")
    else:
        logger.info(f"Using all available {len(df_plot_final)} periods for plot and S/R calculation")


    # Calculate S/R levels if requested, using ONLY the data being plotted
    # identify_support_resistance returns a single list of combined top levels
    sr_levels = None
    if plot_sr:
        if not df_plot_final.empty:
            logger.info("Calculating top Support/Resistance levels on plotted data...")
            # Pass df with DatetimeIndex for potential future use in identify_support_resistance
            # but reset_index().copy() is needed if identify_support_resistance expects 'timestamp' column
            sr_levels = identify_support_resistance(df_plot_final.reset_index().copy())
            # Logging is now done inside identify_support_resistance
            if not sr_levels:
                 logger.warning("Could not identify any S/R levels for the plotted period.")
        else:
            logger.warning("Cannot calculate S/R levels, no data to plot.")


    # Create the plot using the final filtered data and relevant S/R levels
    if not df_plot_final.empty:
        # Pass the combined list of levels to the plot function for manual drawing
        plot_candlestick(df_plot_final, symbol, output_file, timeframe, title,
                         sr_levels=sr_levels)
    else:
        # This case should ideally be caught earlier, but double-check
        logger.error(f"Data processing failed for {symbol}")


if __name__ == "__main__":
    main()