import os
import pandas as pd
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Import project modules
from src import config
from src.data_loading import local_file_loader as local_data # Updated import path and alias

# Removed EodApiClient import as we'll read splits locally
# --- Configuration ---
# Keep INFO level for required messages, warnings, and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories using pathlib for better path handling
# Define paths relative to the project root (assuming this script is in src/feature_engineering/)
INCOMING_DIR = Path(config.INCOMING_DATA_DIR)
OUTPUT_DIR = Path(config.SPLIT_ADJUSTED_DIR)
SPLIT_DATA_DIR = Path(config.SPLIT_DATA_DIR) # Directory containing split JSON files
# TICKER_FILE = PROJECT_ROOT / "data" / "metadata" / "russel.csv" # No longer needed, loaded via config

# --- Helper Functions ---

# Removed read_tickers function, using centralized loader now.
def find_ohlc_files(symbol: str, directory: Path) -> List[Path]:
    """Finds all OHLC CSV files for a given symbol, sorted alphabetically."""
    # Sanitize symbol for filename matching
    safe_symbol = symbol.replace('.', '_').replace('-', '_').strip()
    try:
        # Look for files matching the pattern TICKER_*.csv
        matching_files = sorted(list(directory.glob(f"{safe_symbol}_*.csv"))) # Sort ensures chronological order if dates are in filename
        if not matching_files:
            # Log info level, as it might be expected for some tickers
            logging.info(f"No OHLC files found for ticker {symbol} in {directory} (pattern: {safe_symbol}_*.csv)")
        # else: # Log if multiple files are found for clarity
            # if len(matching_files) > 1:
                # logging.info(f"Found {len(matching_files)} OHLC files for ticker {symbol}.")
        return matching_files # Return the list of paths
    except Exception as e:
        logging.error(f"Error searching for OHLC files for ticker {symbol} in {directory}: {e}")
        return [] # Return empty list on error

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

def read_split_data(symbol: str, directory: Path) -> Optional[List[Dict[str, Any]]]:
    """Reads split data from a JSON file for a given symbol."""
    # Sanitize symbol for filename matching
    safe_symbol = symbol.replace('.', '_').replace('-', '_').strip()
    split_filepath = directory / f"{safe_symbol}_splits.json" # Expecting SYMBOL_splits.json

    if not split_filepath.is_file():
        # It's common for stocks to have no splits, so INFO level is appropriate
        logging.info(f"No split data file found for ticker {symbol} at {split_filepath}")
        return None # Return None to indicate no splits found, not an error

    try:
        with open(split_filepath, 'r') as f:
            splits_data = json.load(f)

        # Basic validation: Check if it's a list
        if not isinstance(splits_data, list):
            logging.error(f"Invalid format in split file {split_filepath}: Expected a JSON list.")
            return None # Indicate error by returning None

        # Optional: Validate structure of list items (e.g., contain 'date' and 'split')
        validated_splits = []
        for item in splits_data:
            if isinstance(item, dict) and 'date' in item and 'split' in item:
                 # Further validation can be added here (e.g., date format, split format)
                 validated_splits.append(item)
            else:
                logging.warning(f"Skipping invalid split entry in {split_filepath}: {item}")

        # logging.info(f"Read {len(validated_splits)} split records from {split_filepath}") # Reduced verbosity
        return validated_splits if validated_splits else None # Return None if list is empty after validation

    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from split file: {split_filepath}")
        return None
    except Exception as e:
        logging.error(f"Error reading split file {split_filepath}: {e}")
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
    """Saves the adjusted data DataFrame to a CSV file."""
    if data is None or data.empty:
        logging.warning(f"No adjusted data to save for ticker {ticker}.")
        return

    # Ensure output directory exists
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create output directory {directory}: {e}")
        return

    # Save as CSV, using the ticker as the filename stem
    output_path = directory / f"{ticker}.csv"

    try:
        # Ensure 'Date' column is in 'YYYY-MM-DD' format for consistency
        data_to_save = data.copy()
        data_to_save['Date'] = pd.to_datetime(data_to_save['Date']).dt.strftime('%Y-%m-%d')

        # Save to CSV, excluding the DataFrame index
        data_to_save.to_csv(output_path, index=False, date_format='%Y-%m-%d')
        logging.info(f"Successfully saved combined and adjusted data for {ticker} to {output_path}")

    except Exception as e:
        logging.error(f"Failed to save adjusted data for {ticker} to {output_path}: {e}")


# --- Main Execution ---
def main():
    """Main function to orchestrate the split adjustment process."""
    logging.info("Starting stock split adjustment process...") # Keep start message

    # --- Initialization ---
    # Removed API key check and EOD client initialization

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
            # 1. Read Local Split Data
            splits = read_split_data(ticker, SPLIT_DATA_DIR)

            # 1b. Find all OHLC files for the ticker
            ohlc_filepaths = find_ohlc_files(ticker, INCOMING_DIR)

            if not ohlc_filepaths:
                # If no OHLC files are found at all, log and skip
                logging.warning(f"Skipping {ticker}: No OHLC files found in {INCOMING_DIR}.")
                error_count += 1
                continue

            # 2. Read and Combine OHLC Data from all found files
            all_ohlc_data = []
            for ohlc_filepath in ohlc_filepaths:
                df_part = read_ohlc_data(ohlc_filepath)
                if df_part is not None and not df_part.empty:
                    all_ohlc_data.append(df_part)
                else:
                    logging.warning(f"Could not read or empty data from {ohlc_filepath.name} for ticker {ticker}.")
                    # Optionally, consider this an error and skip the ticker
                    # error_count += 1
                    # continue # This would skip the ticker if any file fails

            if not all_ohlc_data:
                logging.warning(f"Skipping {ticker}: No valid OHLC data could be read from any found files.")
                error_count += 1
                continue

            # Concatenate, sort by date, and remove duplicates (keeping the last entry for a given date)
            combined_ohlc_data = pd.concat(all_ohlc_data, ignore_index=True)
            combined_ohlc_data = combined_ohlc_data.sort_values(by='Date', ascending=True)
            combined_ohlc_data = combined_ohlc_data.drop_duplicates(subset=['Date'], keep='last')
            combined_ohlc_data = combined_ohlc_data.reset_index(drop=True)
            logging.info(f"Combined {len(combined_ohlc_data)} unique date records for {ticker} from {len(ohlc_filepaths)} file(s).")


            # 3. Adjust Data (using combined data)
            if not splits:
                # No splits found, use the combined data directly
                logging.info(f"No split data found locally for {ticker}. Using combined OHLC data.")
                adjusted_data = combined_ohlc_data # Already a copy due to concat/drop_duplicates
            else:
                # Splits found, proceed with adjustment logic on combined data
                logging.info(f"Local split data found for {ticker}. Adjusting combined OHLC data.")
                adjusted_data = adjust_ohlc_data(combined_ohlc_data, splits)

            # 4. Save Adjusted Data (combined and potentially adjusted)
            save_adjusted_data(ticker, adjusted_data, OUTPUT_DIR)
            processed_count += 1
            # The logic for handling cases with splits is now integrated into the main flow
            # before this point (adjusting 'combined_ohlc_data' if splits exist).
            # No separate 'else' block is required here anymore.

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