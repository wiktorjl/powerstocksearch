from src import config
from src.data_acquisition import eod_downloader as eod_api # Renamed and moved
from src.data_loading import local_file_loader as local_data # Renamed and moved
from datetime import date, timedelta, datetime
import logging
import sys
from typing import Set
# import pandas as pd # No longer needed directly here

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Exchanges to fetch tickers from. Adjust as needed.
# Common examples: 'US' (US Stocks), 'LSE' (London), 'TSE' (Toronto),
# 'INDX' (World Indices), 'COMM' (Commodities), 'FOREX'
# Note: Some 'exchanges' like GSPC (S&P 500) list components, not the index itself.
# Use '.INDX' suffix for the index, e.g., 'GSPC.INDX'
# Default start date if no local data exists for a ticker
DEFAULT_START_DATE = date(2000, 1, 1)
# --- End Configuration ---

def run_downloader():
    """
    Main function to orchestrate the EOD data download process.
    """
    logging.info("--- Starting EOD Data Downloader ---")

    # 1. Load API Key
    if not config.EODHD_API_KEY:
        logging.error("EODHD_API_KEY not found in .env file. Exiting.")
        sys.exit(1) # Exit if API key is missing
    logging.info("API Key loaded successfully.")

    # 2. Initialize API Client
    try:
        client = eod_api.EodApiClient(config.EODHD_API_KEY)
    except ValueError as e:
        logging.error(f"Failed to initialize API client: {e}")
        sys.exit(1)

    # 3. Ensure Data Directory Exists
    local_data.ensure_data_dir_exists()

    # 4. Load Tickers from Configured Sources
    logging.info("Loading tickers from sources specified in config.TICKER_SOURCES...")
    try:
        all_tickers: Set[str] = local_data.load_tickers_from_sources(config.TICKER_SOURCES)
        if not all_tickers:
            # The load_tickers_from_sources function logs details, just add a summary here
            logging.error("No tickers were loaded from any configured source. Exiting.")
            sys.exit(1)
        # Logging of total count is now handled within load_tickers_from_sources
        # logging.info(f"Successfully loaded {len(all_tickers)} unique tickers.") # Optional: Keep if you want summary here too
    except Exception as e:
        # Catch potential unexpected errors during the loading process itself
        logging.error(f"An unexpected error occurred during ticker loading: {e}")
        sys.exit(1)

    # Keep the logging info about total tickers
    logging.info(f"Total unique tickers to process: {len(all_tickers)}")

    # Save tickers to a file for reference (optional)
    ticker_file_path = local_data.DATA_DIR / "tickers.txt"
    with open(ticker_file_path, 'w') as f:
        for ticker in sorted(all_tickers):
            f.write(f"{ticker}\n")
    logging.info(f"Tickers saved to {ticker_file_path}")

    # 5. Process Each Ticker
    today_date = date.today()
    today_date_str = today_date.strftime("%Y-%m-%d")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for ticker in sorted(list(all_tickers)): # Process in alphabetical order
        logging.info(f"--- Processing ticker: {ticker} ---")

        # Determine start date for fetching
        latest_local_date = local_data.get_latest_date_for_ticker(ticker)

        if latest_local_date:
            # Start fetching from the day AFTER the latest local date
            fetch_start_date = latest_local_date + timedelta(days=1)
            logging.info(f"Latest local data ends on {latest_local_date}. Fetching from {fetch_start_date}.")
        else:
            # No local data, use the default start date
            fetch_start_date = DEFAULT_START_DATE
            logging.info(f"No local data found. Fetching from default start date: {fetch_start_date}.")

        fetch_start_date_str = fetch_start_date.strftime("%Y-%m-%d")

        # Check if the start date is after the end date (today)
        if fetch_start_date > today_date:
            logging.info(f"Ticker {ticker} is already up-to-date (local data ends {latest_local_date}). Skipping.")
            skipped_count += 1
            continue

        # Fetch EOD data
        logging.info(f"Fetching EOD data for {ticker} from {fetch_start_date_str} to {today_date_str}")
        try:
            eod_data_df = client.get_eod_data(ticker, fetch_start_date_str, today_date_str)

            if eod_data_df is not None and not eod_data_df.empty:
                # Generate filename and save
                # Use the actual start/end dates for the filename, which are fetch_start_date and today_date
                filename = local_data.generate_filename(ticker, fetch_start_date, today_date)
                local_data.save_data_to_csv(eod_data_df, filename)
                processed_count += 1
            elif eod_data_df is None:
                 # Error logged within get_eod_data or _make_request
                 logging.warning(f"No data frame returned for {ticker} (API/parsing issue or no data available for range).")
                 error_count += 1 # Count as error if fetch failed or returned None unexpectedly
            else: # Empty dataframe returned
                 logging.info(f"No new EOD data found for {ticker} in the requested range ({fetch_start_date_str} to {today_date_str}).")
                 # Not necessarily an error, could just be no trading days/data available
                 skipped_count += 1


        except Exception as e:
            logging.error(f"An unexpected error occurred processing ticker {ticker}: {e}")
            error_count += 1
            # Decide if you want to continue with the next ticker or stop

    logging.info("--- Download Process Finished ---")
    logging.info(f"Summary: Processed={processed_count}, Skipped (up-to-date/no new data)={skipped_count}, Errors={error_count}")
    logging.info(f"Total tickers checked: {len(all_tickers)}")

if __name__ == "__main__":
    run_downloader()