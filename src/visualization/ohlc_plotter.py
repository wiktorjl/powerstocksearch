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
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from src.config import AlgorithmConfig # Updated import path
# Import the support/resistance calculation function
from src.feature_engineering.support_resistance import identify_support_resistance # Updated import path
# Import the data fetching function
from src.database.data_provider import fetch_ohlc_data_db # Updated import path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection and data fetching moved to data_provider.py
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

def plot_candlestick(df_plot, symbol, output_file=None, timeframe='daily', title=None, support_levels=None, resistance_levels=None):
    """
    Plot a professional candlestick chart using mplfinance, manually drawing pre-calculated S/R lines.

    Args:
        df_plot (pandas.DataFrame): DataFrame containing OHLC data for the desired plot period (processed and filtered).
                                    Must have a DatetimeIndex.
        symbol (str): Stock ticker symbol
        output_file (str, optional): Path to save the plot image
        timeframe (str): Timeframe for the chart ('daily', 'weekly', or 'monthly')
        title (str, optional): Custom title for the plot
        support_levels (list, optional): List of pre-calculated support levels to plot.
        resistance_levels (list, optional): List of pre-calculated resistance levels to plot.

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
        if (support_levels or resistance_levels) and axes and hasattr(axes[0], 'axhline'): # Check if axes exist and have axhline method
            try:
                price_ax = axes[0] # Typically the main price panel
                drawn_support = 0
                drawn_resistance = 0

                # Draw support lines (green)
                if support_levels:
                    for level in support_levels:
                        if isinstance(level, (int, float)):
                            price_ax.axhline(y=level, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
                            logger.debug(f"Drawing support line at {level:.2f}")
                            drawn_support += 1

                # Draw resistance lines (red)
                if resistance_levels:
                    for level in resistance_levels:
                         if isinstance(level, (int, float)):
                            price_ax.axhline(y=level, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
                            logger.debug(f"Drawing resistance line at {level:.2f}")
                            drawn_resistance += 1

                if drawn_support > 0 or drawn_resistance > 0:
                    logger.info(f"Manually drew {drawn_support} support and {drawn_resistance} resistance lines.")
                else:
                    logger.info("No valid S/R levels provided or drawable.")

            except IndexError:
                 logger.error("Could not access axes[0] to draw S/R lines. Plot might be empty or structure unexpected.")
            except Exception as draw_err:
                 logger.error(f"Error manually drawing S/R lines: {draw_err}")
        elif support_levels or resistance_levels:
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

    # Fetch data using the imported function
    raw_df = fetch_ohlc_data_db(symbol, start_date, end_date)

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
    support_levels_to_plot = None
    resistance_levels_to_plot = None
    if plot_sr:
        if not df_plot_final.empty:
            logger.info("Calculating top Support/Resistance levels on plotted data...")
            # Instantiate the config for S/R calculation
            sr_config = AlgorithmConfig()
            # Pass the DataFrame and the config object to identify_support_resistance
            all_sr_levels = identify_support_resistance(df_plot_final.reset_index().copy(), sr_config)

            if all_sr_levels:
                # Filter and limit the S/R levels here before passing to plot function
                try:
                    current_price = df_plot_final['close'].iloc[-1]
                    max_lines_per_type = AlgorithmConfig.MAX_SR_LINES

                    # Separate, sort, and limit
                    support = sorted([lvl for lvl in all_sr_levels if isinstance(lvl, (int, float)) and lvl <= current_price], reverse=True) # Highest support first
                    resistance = sorted([lvl for lvl in all_sr_levels if isinstance(lvl, (int, float)) and lvl > current_price]) # Lowest resistance first

                    support_levels_to_plot = support[:max_lines_per_type]
                    resistance_levels_to_plot = resistance[:max_lines_per_type]

                    logger.info(f"Identified {len(support_levels_to_plot)} support and {len(resistance_levels_to_plot)} resistance levels to plot.")

                except IndexError:
                    logger.error("Could not get current price to filter S/R levels.")
                except Exception as filter_err:
                    logger.error(f"Error filtering S/R levels: {filter_err}")
            else:
                 logger.warning("Could not identify any S/R levels for the plotted period.")
        else:
            logger.warning("Cannot calculate S/R levels, no data to plot.")
    # Create the plot using the final filtered data and relevant S/R levels
    if not df_plot_final.empty:
        # Pass the pre-filtered lists of support and resistance levels
        plot_candlestick(df_plot_final, symbol, output_file, timeframe, title,
                         support_levels=support_levels_to_plot,
                         resistance_levels=resistance_levels_to_plot)
    else:
        # This case should ideally be caught earlier, but double-check
        logger.error(f"Data processing failed for {symbol}")


if __name__ == "__main__":
    main()