import os
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

# Import project modules
from src import config
from src.data_acquisition.eod_downloader import EodApiClient # Updated import path
from src.data_loading import local_file_loader as local_data # Updated import path and alias

# --- Configuration ---
# Keep INFO level for required messages, warnings, and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories using pathlib for better path handling
# Define paths relative to the project root (assuming this script is in src/feature_engineering/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INCOMING_DIR = Path(config.INCOMING_DATA_DIR)
OUTPUT_DIR = Path(config.SPLIT_ADJUSTED_DIR)
# TICKER_FILE = PROJECT_ROOT / "data" / "metadata" / "russel.csv" # No longer needed, loaded via config

# --- Helper Functions ---

# Removed read_tickers function, using centralized loader now.
def find_ohlc_file(symbol: str, directory: Path) -> Optional[Path]:
    """Finds the OHLC CSV file for a given symbol in the specified directory."""
    # Sanitize symbol for filename matching (consistent with local_data.py)
    safe_symbol = symbol.replace('.', '_').replace('-', '_').strip()
    try:
        # Look for files matching the pattern TICKER_YYYYMMDD_YYYYMMDD.csv
        matching_files = list(directory.glob(f"{safe_symbol}_*.csv"))
        if not matching_files:
            # Keep this warning as it indicates a potential data issue when splits *are* found
            logging.warning(f"No OHLC file found for ticker {symbol} in {directory} (expected pattern: {safe_symbol}_*.csv)")
            return None
        if len(matching_files) > 1:
            # Keep this warning
            logging.warning(f"Multiple OHLC files found for ticker {symbol} in {directory}. Using {matching_files[0].name}")
        return matching_files[0]
    except Exception as e:
        logging.error(f"Error searching for OHLC file for ticker {symbol} in {directory}: {e}")
        return None

def read_ohlc_data(filepath: Path) -> Optional[pd.DataFrame]:
    """Reads OHLC data from a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Date']) # Ensure 'Date' is parsed
        # Basic validation - check for required columns
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logging.error(f"Missing required columns in {filepath}. Expected: {required_cols}")
            return None
        # Convert price/volume columns to numeric, coercing errors
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=required_cols) # Drop rows where essential data is missing/invalid
        df = df.sort_values(by='Date', ascending=True).reset_index(drop=True) # Ensure chronological order
        # logging.info(f"Read {len(df)} OHLC records from {filepath}") # Reduced verbosity
        return df
    except FileNotFoundError:
        logging.error(f"OHLC file not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading OHLC file {filepath}: {e}")
        return None

def parse_split_ratio(split_str: str) -> Optional[float]:
    """Parses a split ratio string 'New/Old' (e.g., '2/1') into an adjustment factor."""
    try:
        # EODHD uses '/' as the delimiter
        new, old = map(float, split_str.split('/'))
        if old == 0: # Avoid division by zero
            logging.error(f"Invalid split ratio: denominator cannot be zero ('{split_str}')")
            return None
        # Adjustment factor = Old / New. Prices before split are multiplied by this.
        # Example: 2:1 split means 1 old share becomes 2 new shares. Price is halved. Factor = 1/2 = 0.5
        # Example: 1:10 reverse split means 10 old shares become 1 new share. Price x10. Factor = 10/1 = 10.0
        return old / new
    except ValueError:
        logging.error(f"Could not parse split ratio: '{split_str}'")
        return None
    except Exception as e:
        logging.error(f"Unexpected error parsing split ratio '{split_str}': {e}")
        return None

def adjust_ohlc_data(ohlc_df: pd.DataFrame, splits: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Adjusts OHLC data for stock splits using backward propagation.

    Args:
        ohlc_df: DataFrame with OHLC data, sorted by Date ascending.
        splits: List of split dictionaries [{'date': 'YYYY-MM-DD', 'split': 'New/Old'}].

    Returns:
        A new DataFrame with adjusted OHLC data.
    """
    if not splits:
        # This case is handled in the main loop now, no need to log here
        return ohlc_df.copy() # Return a copy to avoid modifying original

    adjusted_df = ohlc_df.copy()
    adjusted_df['Date'] = pd.to_datetime(adjusted_df['Date']) # Ensure Date is datetime

    # Sort splits by date descending to apply backward propagation correctly
    splits_sorted = sorted(splits, key=lambda x: pd.to_datetime(x['date']), reverse=True)

    # logging.info(f"Applying {len(splits_sorted)} splits (backward propagation)...") # Reduced verbosity

    for split in splits_sorted:
        try:
            split_date = pd.to_datetime(split['date'])
            split_ratio_str = split['split']
            adj_factor = parse_split_ratio(split_ratio_str)

            if adj_factor is None:
                # Keep this warning
                logging.warning(f"Skipping split on {split_date.date()} due to invalid ratio: '{split_ratio_str}'")
                continue

            # Apply adjustment to rows *before* the split date
            mask = adjusted_df['Date'] < split_date
            if mask.any():
                # Adjust Price columns (O, H, L, C)
                for col in ['Open', 'High', 'Low', 'Close']:
                    adjusted_df.loc[mask, col] *= adj_factor

                # Adjust Volume (inverse factor)
                # Volume factor = New / Old = 1 / adj_factor
                if adj_factor != 0: # Avoid division by zero if adj_factor was somehow 0
                    volume_factor = 1.0 / adj_factor
                    # Calculate new volume, round, and convert to int before assignment
                    new_volume = (adjusted_df.loc[mask, 'Volume'] * volume_factor).round().astype(int)
                    adjusted_df.loc[mask, 'Volume'] = new_volume
                else:
                     # Keep this warning
                     logging.warning(f"Cannot adjust volume for split on {split_date.date()} due to zero adjustment factor.")
                # Log the successful application of this split - KEEP THIS
                logging.info(f"Applied split {split_ratio_str} on {split_date.date()}, adjusted {mask.sum()} records.")
            # else: # No need to log if no rows were adjusted before the date

        except Exception as e:
            logging.error(f"Error processing split {split}: {e}")
            # Decide whether to continue or stop on error

    return adjusted_df

def save_adjusted_data(ticker: str, data: pd.DataFrame, directory: Path):
    """Saves the adjusted data DataFrame to a JSON file."""
    if data is None or data.empty:
        logging.warning(f"No adjusted data to save for ticker {ticker}.")
        return

    # Ensure output directory exists
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {directory}: {e}")
        return

    output_path = directory / f"{ticker}.json"

    try:
        # Convert DataFrame to list of dictionaries (JSON records format)
        # Convert Timestamp to ISO 8601 string format for JSON compatibility
        data_dict = data.copy()
        data_dict['Date'] = data_dict['Date'].dt.strftime('%Y-%m-%d')
        result_json = data_dict.to_dict(orient='records')

        with open(output_path, 'w') as f:
            json.dump(result_json, f, indent=4)
        # logging.info(f"Successfully saved adjusted data for {ticker} to {output_path}") # Reduced verbosity

    except Exception as e:
        logging.error(f"Failed to save adjusted data for {ticker} to {output_path}: {e}")


# --- Main Execution ---
def main():
    """Main function to orchestrate the split adjustment process."""
    logging.info("Starting stock split adjustment process...") # Keep start message

    # --- Initialization ---
    # Ensure API key is loaded
    if not config.EODHD_API_KEY:
        logging.error("EODHD_API_KEY not found. Please set it in the .env file.")
        return # Exit if no API key

    # Initialize EOD API client
    api_client = EodApiClient(config.EODHD_API_KEY)

    # Ensure output directory exists
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # logging.info(f"Ensured output directory exists: {OUTPUT_DIR}") # Reduced verbosity
    except OSError as e:
        logging.error(f"Could not create output directory {OUTPUT_DIR}: {e}. Exiting.")
        return

    # --- Data Loading ---
    # Load tickers using the centralized function and configuration
    logging.info("Loading tickers from sources specified in config.TICKER_SOURCES...")
    tickers_set: Set[str] = local_data.load_tickers_from_sources(config.TICKER_SOURCES)
    tickers: List[str] = sorted(list(tickers_set)) # Convert set to sorted list for processing

    if not tickers:
        logging.error("No tickers loaded. Exiting.")
        return

    # --- Processing Loop ---
    processed_count = 0
    error_count = 0
    logging.info(f"Processing {len(tickers)} tickers...") # Add a start count

    for ticker in tickers:
        # logging.info(f"--- Processing ticker: {ticker} ---") # Reduced verbosity
        try:
            # 1. Fetch Split Data
            # Assuming tickers in russel.csv might not have .US suffix, add it if needed
            ticker_eod = ticker if '.' in ticker else f"{ticker}.US"
            splits = api_client.get_splits_data(ticker_eod)

            if not splits:
                # No splits found, copy original data
                logging.info(f"No splits found for {ticker}. Copying original OHLC data.")
                ohlc_filepath = find_ohlc_file(ticker, INCOMING_DIR)
                if not ohlc_filepath:
                    logging.warning(f"Skipping {ticker}: No splits found, and original OHLC file not found in {INCOMING_DIR}.")
                    error_count += 1
                    continue # Skip to next ticker if original file not found

                ohlc_data = read_ohlc_data(ohlc_filepath)
                if ohlc_data is None or ohlc_data.empty:
                    logging.warning(f"Skipping {ticker}: No splits found, and failed to read or empty original OHLC data from {ohlc_filepath.name}.")
                    error_count += 1
                    continue # Skip to next ticker if original data is bad

                # Save the original, unadjusted data
                save_adjusted_data(ticker, ohlc_data, OUTPUT_DIR)
                processed_count += 1
            else:
                # Splits found, proceed with adjustment logic
                # logging.info(f"Splits found for {ticker}. Proceeding to read local OHLC data.") # Reduced verbosity
                ohlc_filepath = find_ohlc_file(ticker, INCOMING_DIR)
                if not ohlc_filepath:
                    # Keep this warning
                    logging.warning(f"Skipping {ticker}: Splits found, but OHLC file not found in {INCOMING_DIR}.")
                    error_count += 1
                    continue

                ohlc_data = read_ohlc_data(ohlc_filepath)
                if ohlc_data is None or ohlc_data.empty:
                     # Keep this warning
                    logging.warning(f"Skipping {ticker}: Splits found, but failed to read or empty OHLC data from {ohlc_filepath.name}.")
                    error_count += 1
                    continue

                # 3. Adjust Data (We know splits exist at this point)
                adjusted_data = adjust_ohlc_data(ohlc_data, splits)

                # 4. Save Adjusted Data
                save_adjusted_data(ticker, adjusted_data, OUTPUT_DIR)
                processed_count += 1

        except Exception as e:
            # Keep error logging for individual ticker failures
            logging.error(f"Failed to process ticker {ticker}: {e}")
            error_count += 1
            # Continue to the next ticker

    # Keep final summary logs
    logging.info("--- Split Adjustment Process Finished ---")
    logging.info(f"Successfully processed: {processed_count} tickers")
    logging.info(f"Errors/Skipped: {error_count} tickers")

if __name__ == "__main__":
    main()