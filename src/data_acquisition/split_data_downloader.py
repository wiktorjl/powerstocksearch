import logging
import sys
from datetime import date, timedelta
from typing import Set, List, Dict, Any

from src import config
from src.data_acquisition import eod_downloader as eod_api
from src.data_loading import local_file_loader as local_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_split_downloader():
    """
    Main function to orchestrate the stock split data download process.
    Fetches split data incrementally based on the latest date found locally.
    """
    logging.info("--- Starting Split Data Downloader ---")

    # 1. Load API Key
    if not config.EODHD_API_KEY:
        logging.error("EODHD_API_KEY not found in .env file. Exiting.")
        sys.exit(1)
    logging.info("API Key loaded successfully.")

    # 2. Initialize API Client
    try:
        client = eod_api.EodApiClient(config.EODHD_API_KEY)
    except ValueError as e:
        logging.error(f"Failed to initialize API client: {e}")
        sys.exit(1)

    # 3. Ensure Split Data Directory Exists
    local_data.ensure_split_data_dir_exists() # Use the new function

    # 4. Load Tickers from Configured Sources
    logging.info("Loading tickers from sources specified in config.TICKER_SOURCES...")
    try:
        # Use the simpler ticker loader for now, assuming we don't need company/sector info here
        all_tickers: Set[str] = local_data.load_tickers_from_sources(config.TICKER_SOURCES)
        if not all_tickers:
            logging.error("No tickers were loaded from any configured source. Exiting.")
            sys.exit(1)
        logging.info(f"Total unique tickers to process for splits: {len(all_tickers)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during ticker loading: {e}")
        sys.exit(1)

    # 5. Process Each Ticker for Splits
    processed_count = 0
    updated_count = 0
    no_new_data_count = 0
    error_count = 0
    today_date = date.today() # Used for logging range, not strictly necessary for API call

    for ticker in sorted(list(all_tickers)): # Process in alphabetical order
        logging.info(f"--- Processing splits for ticker: {ticker} ---")

        # Determine start date for fetching new splits
        latest_local_split_date = local_data.get_latest_split_date(ticker)
        fetch_start_date: Optional[date] = None

        if latest_local_split_date:
            # Start fetching from the day AFTER the latest known split
            # EODHD split endpoint includes the 'from' date, so fetch from the exact date
            # to potentially re-fetch the latest one if needed, or simply start after.
            # Let's fetch starting the day *after* to avoid re-fetching the known latest.
            fetch_start_date = latest_local_split_date + timedelta(days=1)
            logging.info(f"Latest local split date is {latest_local_split_date}. Fetching new splits from {fetch_start_date}.")
        else:
            # No local data, fetch all historical splits (API default)
            logging.info("No local split data found. Fetching all historical splits.")
            # No need to set fetch_start_date, API call without 'from' fetches all

        fetch_start_date_str = fetch_start_date.strftime("%Y-%m-%d") if fetch_start_date else None

        # Fetch split data from API
        try:
            # Pass fetch_start_date_str which might be None (fetches all) or a date string
            new_splits: Optional[List[Dict[str, Any]]] = client.get_splits_data(ticker, start_date=fetch_start_date_str)

            processed_count += 1

            # Load existing splits
            existing_splits: List[Dict[str, Any]] = local_data.load_split_data(ticker)
            existing_split_dates = {s['date'] for s in existing_splits}

            if new_splits:
                # Filter out any potential duplicates fetched (e.g., if API returns overlapping dates)
                truly_new_splits = [s for s in new_splits if s['date'] not in existing_split_dates]

                if truly_new_splits:
                    logging.info(f"Fetched {len(truly_new_splits)} new split records for {ticker}.")
                    combined_splits = existing_splits + truly_new_splits
                    local_data.save_split_data(ticker, combined_splits)
                    updated_count += 1
                else:
                    logging.info(f"Fetched {len(new_splits)} split records, but they already exist locally for {ticker}. No update needed.")
                    no_new_data_count += 1
            elif new_splits is None and latest_local_split_date:
                 # API returned None (could be 404 or error), and we had local data
                 # If 404, it means no *new* splits since fetch_start_date
                 logging.info(f"No new split data found for {ticker} since {fetch_start_date_str}.")
                 no_new_data_count += 1
            elif new_splits is None and not latest_local_split_date:
                 # API returned None (could be 404 or error), and we had no local data
                 # This means the ticker likely has no splits at all
                 logging.info(f"No split data found for {ticker} (checked entire history).")
                 no_new_data_count += 1
                 # Optionally save an empty list to mark as checked?
                 # local_data.save_split_data(ticker, [])
            # else: # new_splits is an empty list [] - API returned empty list explicitly
            #     logging.info(f"API returned an empty list for splits for {ticker} in the requested range.")
            #     no_new_data_count += 1


        except Exception as e:
            logging.error(f"An unexpected error occurred processing splits for ticker {ticker}: {e}")
            error_count += 1
            # Decide if you want to continue with the next ticker or stop

    logging.info("--- Split Download Process Finished ---")
    logging.info(f"Summary: Tickers Processed={processed_count}, Tickers Updated={updated_count}, No New Data/No Splits={no_new_data_count}, Errors={error_count}")
    logging.info(f"Total unique tickers checked: {len(all_tickers)}")

if __name__ == "__main__":
    run_split_downloader()