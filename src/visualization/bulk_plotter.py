#!/usr/bin/env python3
"""
Bulk OHLC Plotter - Generates candlestick charts for multiple symbols.

This script fetches a list of symbols from the database, then iterates through
each symbol, fetching its OHLC data and generating a candlestick chart similar
to the single ohlc_plotter.py script.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Assuming standard project structure for imports
# Add project root to sys.path if necessary, or ensure PYTHONPATH is set
# project_root = Path(__file__).resolve().parents[2]
# sys.path.append(str(project_root))

try:
    from src.config import AlgorithmConfig
    from src.feature_engineering.support_resistance import identify_support_resistance
    from src.database.data_provider import fetch_ohlc_data_db, list_symbols_db
    from src.visualization.ohlc_plotter import process_dataframe, plot_candlestick
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure the script is run from the project root or PYTHONPATH is configured correctly.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate candlestick charts for multiple symbols')

    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Directory to save the generated plot images')
    parser.add_argument('--days', '-d', type=int,
                        help='Number of most recent days/periods to plot for each symbol')
    parser.add_argument('--timeframe', '-tf', type=str, choices=['daily', 'weekly', 'monthly'],
                        default='daily', help='Timeframe for the charts (daily, weekly, monthly)')
    parser.add_argument('--sr', action='store_true',
                        help='Calculate and plot support/resistance levels for each chart')
    parser.add_argument('--prefix', '-p', type=str,
                        help='Only plot symbols starting with this prefix (case-insensitive)')
    parser.add_argument('--limit', '-l', type=int,
                        help='Limit the number of symbols to process (for testing)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing plot files in the output directory')

    return parser.parse_args()

def main():
    """Main function to orchestrate the bulk plotting process."""
    args = parse_arguments()

    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: {output_dir.resolve()}")
    except OSError as e:
        logger.error(f"Failed to create output directory '{output_dir}': {e}")
        sys.exit(1)

    # Fetch symbols
    logger.info(f"Fetching symbols from database" + (f" with prefix '{args.prefix}'" if args.prefix else ""))
    symbols = list_symbols_db(prefix=args.prefix)

    if symbols is None:
        logger.error("Failed to fetch symbols from the database. Exiting.")
        sys.exit(1)
    elif not symbols:
        logger.warning("No symbols found matching the criteria. Exiting.")
        sys.exit(0)

    logger.info(f"Found {len(symbols)} symbols.")

    # Apply limit if specified
    if args.limit and args.limit > 0:
        symbols = symbols[:args.limit]
        logger.info(f"Processing limited to the first {len(symbols)} symbols.")

    processed_count = 0
    skipped_count = 0
    error_count = 0
    sr_config = AlgorithmConfig() if args.sr else None # Instantiate config only if needed

    # Process each symbol
    for i, symbol in enumerate(symbols):
        logger.info(f"--- Processing symbol {i+1}/{len(symbols)}: {symbol} ---")
        output_filename = output_dir / f"{symbol}_{args.timeframe}.png" # Use PNG for better quality

        if not args.overwrite and output_filename.exists():
            logger.info(f"Skipping {symbol}: Plot file already exists at {output_filename}")
            skipped_count += 1
            continue

        # 1. Fetch data
        raw_df = fetch_ohlc_data_db(symbol) # Fetch all available data initially
        if raw_df is None or raw_df.empty:
            logger.warning(f"No data found for symbol {symbol}. Skipping.")
            skipped_count += 1
            continue

        # 2. Process dataframe (resample, set index)
        df_processed = process_dataframe(raw_df, args.timeframe)
        if df_processed is None or df_processed.empty:
            logger.error(f"Data processing failed for {symbol}. Skipping.")
            error_count += 1
            continue

        # 3. Filter for the requested number of days/periods *before* S/R calculation
        df_plot_final = df_processed.copy()
        if args.days and args.days > 0:
            if len(df_plot_final) > args.days:
                df_plot_final = df_plot_final.iloc[-args.days:]
                logger.info(f"Using most recent {args.days} {args.timeframe} periods for plot")
            else:
                 logger.info(f"Using all available {len(df_plot_final)} {args.timeframe} periods (less than requested {args.days})")
        else:
            logger.info(f"Using all available {len(df_plot_final)} {args.timeframe} periods")

        if df_plot_final.empty:
             logger.warning(f"No data left for {symbol} after filtering/processing. Skipping.")
             skipped_count += 1
             continue

        # 4. Calculate S/R levels if requested
        support_levels_to_plot = None
        resistance_levels_to_plot = None
        if args.sr:
            logger.info(f"Calculating S/R levels for {symbol}...")
            try:
                # Pass the DataFrame and the config object
                # Ensure identify_support_resistance expects a DataFrame with a 'timestamp' column if reset_index() is used
                all_sr_levels = identify_support_resistance(df_plot_final.reset_index().copy(), sr_config)

                if all_sr_levels:
                    current_price = df_plot_final['close'].iloc[-1]
                    max_lines_per_type = AlgorithmConfig.MAX_SR_LINES

                    support = sorted([lvl for lvl in all_sr_levels if isinstance(lvl, (int, float)) and lvl <= current_price], reverse=True)
                    resistance = sorted([lvl for lvl in all_sr_levels if isinstance(lvl, (int, float)) and lvl > current_price])

                    support_levels_to_plot = support[:max_lines_per_type]
                    resistance_levels_to_plot = resistance[:max_lines_per_type]
                    logger.info(f"Identified {len(support_levels_to_plot)} support and {len(resistance_levels_to_plot)} resistance levels for {symbol}.")
                else:
                    logger.warning(f"Could not identify any S/R levels for {symbol}.")
            except IndexError:
                 logger.error(f"Could not get current price for {symbol} to filter S/R levels.")
            except Exception as sr_err:
                 logger.error(f"Error calculating S/R levels for {symbol}: {sr_err}")
                 # Continue without S/R levels for this symbol

        # 5. Plot candlestick chart
        logger.info(f"Generating plot for {symbol}...")
        try:
            plot_success = plot_candlestick(
                df_plot=df_plot_final,
                symbol=symbol,
                output_file=str(output_filename), # Pass as string
                timeframe=args.timeframe,
                title=f'{args.timeframe.capitalize()} Chart for {symbol}', # Simple title
                support_levels=support_levels_to_plot,
                resistance_levels=resistance_levels_to_plot
            )
            if plot_success:
                logger.info(f"Successfully saved plot for {symbol} to {output_filename}")
                processed_count += 1
            else:
                logger.error(f"Plot generation failed for {symbol}.")
                error_count += 1
        except Exception as plot_err:
            logger.error(f"Unhandled error during plot generation for {symbol}: {plot_err}")
            error_count += 1

    logger.info("--- Bulk Plotting Summary ---")
    logger.info(f"Total symbols found: {len(symbols) if symbols else 0}")
    if args.limit: logger.info(f"Symbols processed (due to limit): {len(symbols)}")
    logger.info(f"Plots successfully generated: {processed_count}")
    logger.info(f"Symbols skipped (no data/already exists): {skipped_count}")
    logger.info(f"Errors encountered: {error_count}")
    logger.info("-----------------------------")

if __name__ == "__main__":
    main()