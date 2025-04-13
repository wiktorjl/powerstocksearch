import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional, Union, List, Dict, Set
from typing import Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the directory for storing incoming data relative to this script's location
# Using Path for better cross-platform compatibility
# Point to the 'incoming' directory at the project root, relative to this file's location
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "incoming"

# Regex to parse filenames like TICKER_YYYYMMDD_YYYYMMDD.csv
# It captures the ticker, start date, and end date.
# Making ticker capture more robust (allowing dots, hyphens)
FILENAME_REGEX = re.compile(r"^([A-Z0-9\.\-]+)_(\d{8})_(\d{8})\.csv$")

def ensure_data_dir_exists():
    """
    Creates the data directory if it doesn't already exist.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured data directory exists: {DATA_DIR.resolve()}")
    except OSError as e:
        logging.error(f"Failed to create data directory {DATA_DIR}: {e}")
        # Depending on the application's needs, you might want to raise the exception
        # raise

def get_latest_date_for_ticker(ticker: str) -> Optional[date]:
    """
    Scans the data directory for files matching the ticker and finds the
    latest end date from the filenames.

    Args:
        ticker: The stock ticker symbol (e.g., 'AAPL.US').

    Returns:
        The latest end date found as a datetime.date object, or None if no
        data files for the ticker are found or filenames are malformed.
    """
    latest_date: Optional[date] = None
    ticker_safe = ticker.replace('.', '_').replace('-', '_') # Make ticker safe for regex if needed, though current regex handles '.' and '-'

    if not DATA_DIR.is_dir():
        logging.warning(f"Data directory {DATA_DIR} does not exist. Cannot find latest date.")
        return None

    try:
        for filepath in DATA_DIR.glob(f"{ticker_safe}*.csv"): # More efficient glob pattern
            match = FILENAME_REGEX.match(filepath.name)
            if match:
                # Extract the end date string (YYYYMMDD)
                end_date_str = match.group(3)
                try:
                    # Convert string to date object
                    current_end_date = datetime.strptime(end_date_str, "%Y%m%d").date()
                    # Update latest_date if this file's end date is newer
                    if latest_date is None or current_end_date > latest_date:
                        latest_date = current_end_date
                except ValueError:
                    logging.warning(f"Could not parse date from filename: {filepath.name}")
                    continue # Skip malformed filenames

        if latest_date:
            logging.info(f"Latest data found for {ticker} ends on {latest_date.strftime('%Y-%m-%d')}")
        else:
            logging.info(f"No existing data files found for {ticker} in {DATA_DIR}")

        return latest_date

    except Exception as e:
        logging.error(f"Error scanning directory {DATA_DIR} for ticker {ticker}: {e}")
        return None


def generate_filename(ticker: str, start_date: Union[str, date], end_date: Union[str, date]) -> Path:
    """
    Generates the standard filename for saving EOD data.

    Args:
        ticker: The stock ticker symbol.
        start_date: The start date of the data (string 'YYYY-MM-DD' or date object).
        end_date: The end date of the data (string 'YYYY-MM-DD' or date object).

    Returns:
        A Path object representing the full path to the CSV file.
    """
    # Ensure dates are strings in YYYYMMDD format
    start_date_str = start_date.strftime("%Y%m%d") if isinstance(start_date, date) else datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y%m%d")
    end_date_str = end_date.strftime("%Y%m%d") if isinstance(end_date, date) else datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y%m%d")

    # Sanitize ticker for filename (replace characters unsafe for filenames if necessary)
    # For simplicity, replacing '.' and '-' common in tickers. Adjust if other chars appear.
    safe_ticker = ticker.replace('.', '_').replace('-', '_')

    filename = f"{safe_ticker}_{start_date_str}_{end_date_str}.csv"
    return DATA_DIR / filename

def save_data_to_csv(dataframe: pd.DataFrame, filename: Path):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        dataframe: The pandas DataFrame to save.
        filename: The Path object representing the target CSV file path.
    """
    if dataframe is None or dataframe.empty:
        logging.warning(f"Attempted to save an empty or None DataFrame to {filename}. Skipping.")
        return

    try:
        # Ensure the directory exists (though ensure_data_dir_exists should handle this)
        filename.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(filename, index=False)
        logging.info(f"Successfully saved data to {filename}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {filename}: {e}")
        # Consider re-raising or handling more specifically


def load_tickers_from_sources(sources: List[Dict[str, str]]) -> Set[str]:
    """
    Loads ticker symbols from a list of specified source files (CSV/TSV).

    Args:
        sources: A list of dictionaries. Each dictionary must contain:
                 'path': The relative path to the CSV/TSV file from the project root.
                 'symbol_column': The name of the column containing the ticker symbols.
                 Optionally, 'separator': The delimiter used in the file (defaults to ',').

    Returns:
        A set of unique ticker symbols loaded from all specified files.
        Returns an empty set if no files are processed or errors occur.
    """
    all_tickers: Set[str] = set()
    project_root = Path(__file__).resolve().parent.parent.parent

    if not sources:
        logging.warning("No ticker sources provided in the configuration.")
        return all_tickers

    for source_info in sources:
        file_path_str = source_info.get('path')
        symbol_column = source_info.get('symbol_column')
        separator = source_info.get('separator', ',') # Default to comma

        if not file_path_str or not symbol_column:
            logging.warning(f"Skipping invalid source entry in configuration: {source_info}. Missing 'path' or 'symbol_column'.")
            continue

        file_path = project_root / file_path_str
        logging.info(f"Loading tickers from {file_path} (Column: '{symbol_column}', Separator: '{separator}')")

        try:
            if not file_path.is_file():
                logging.error(f"Ticker source file not found: {file_path}")
                continue

            df = pd.read_csv(file_path, sep=separator)

            if symbol_column not in df.columns:
                logging.error(f"Column '{symbol_column}' not found in {file_path}. Available columns: {df.columns.tolist()}")
                continue

            # Extract tickers, handle NaN, convert to string, strip whitespace, and filter out empty strings
            tickers_in_file = set()
            if not df.empty and symbol_column in df.columns:
                valid_tickers = df[symbol_column].dropna().astype(str).str.strip()
                tickers_in_file = set(valid_tickers[valid_tickers != ''])

            logging.info(f"Loaded {len(tickers_in_file)} unique tickers from {file_path}")
            all_tickers.update(tickers_in_file)

        except pd.errors.EmptyDataError:
            logging.warning(f"Ticker source file is empty: {file_path}")
        except FileNotFoundError:
             logging.error(f"Ticker source file not found (double check): {file_path}")
        except Exception as e:
            logging.error(f"Error reading ticker file {file_path}: {e}")

    logging.info(f"Total unique tickers loaded from all sources: {len(all_tickers)}")
    return all_tickers


# Placeholder for the old function if needed, or remove if load_tickers_and_data_from_sources replaces it entirely.
# def load_tickers_from_sources(sources: List[Dict[str, str]]) -> Set[str]:
#     pass

def load_tickers_and_data_from_sources(sources: List[Dict[str, str]]) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Loads ticker symbols and associated data (sector, subsector) from specified source files.

    Args:
        sources: A list of dictionaries. Each dictionary must contain:
                 'path': Relative path to the CSV/TSV file from the project root.
                 'symbol_column': Name of the column with ticker symbols.
                 Optionally:
                   'sector_column': Name of the column with sector information.
                   'subsector_column': Name of the column with subsector information.
                   'separator': Delimiter used in the file (defaults to ',').

    Returns:
        A dictionary where keys are unique ticker symbols (str) and values are
        dictionaries containing 'sector' and 'subsector' (Optional[str]).
        Returns an empty dictionary if no files are processed or errors occur.
    """
    all_ticker_data: Dict[str, Dict[str, Optional[str]]] = {}
    project_root = Path(__file__).resolve().parent.parent.parent

    if not sources:
        logging.warning("No ticker sources provided in the configuration.")
        return all_ticker_data

    for source_info in sources:
        file_path_str = source_info.get('path')
        symbol_column = source_info.get('symbol_column')
        sector_column = source_info.get('sector_column') # Optional
        subsector_column = source_info.get('subsector_column') # Optional
        company_column = source_info.get('name_column') # Optional

        separator = source_info.get('separator', ',') # Default to comma

        if not file_path_str or not symbol_column:
            logging.warning(f"Skipping invalid source entry in configuration: {source_info}. Missing 'path' or 'symbol_column'.")
            continue

        file_path = project_root / file_path_str
        logging.info(f"Loading tickers from {file_path} (Column: '{symbol_column}', Separator: '{separator}')")

        try:
            # Check if file exists before attempting to read
            if not file_path.is_file():
                logging.error(f"Ticker source file not found: {file_path}")
                continue # Skip this file and try the next one

            df = pd.read_csv(file_path, sep=separator)

            # --- Column Validation ---
            required_columns = [symbol_column]
            optional_columns = {'sector': sector_column, 'subsector': subsector_column, 'company': company_column}
            missing_required = [col for col in required_columns if col and col not in df.columns] # Check if col is not None/empty before checking existence
            available_optional = {key: col for key, col in optional_columns.items() if col and col in df.columns}
            missing_optional = {key: col for key, col in optional_columns.items() if col and col not in df.columns}

            if missing_required:
                logging.error(f"Required column(s) '{', '.join(missing_required)}' not found in {file_path}. Available: {df.columns.tolist()}")
                continue # Skip this file

            if missing_optional:
                 logging.warning(f"Optional column(s) '{', '.join(missing_optional.values())}' specified but not found in {file_path}. Proceeding without them.")

            # --- Data Extraction ---
            count = 0
            for index, row in df.iterrows():
                # Ensure row[symbol_column] is not NaN before converting to str
                if pd.isna(row[symbol_column]):
                    continue
                ticker = str(row[symbol_column]).strip()
                if not ticker: # Skip empty tickers
                    continue

                sector = str(row[available_optional['sector']]).strip() if 'sector' in available_optional and pd.notna(row[available_optional['sector']]) else None
                subsector = str(row[available_optional['subsector']]).strip() if 'subsector' in available_optional and pd.notna(row[available_optional['subsector']]) else None
                company = str(row[available_optional['company']]).strip() if 'company' in available_optional and pd.notna(row[available_optional['company']]) else None
                # Add or update ticker data, potentially overwriting from previous sources if duplicates exist
                count += 1


                if subsector is None or subsector == "" or len(subsector) < 1:
                    subsector = "NONE"

                all_ticker_data[ticker] = {'sector': sector, 'subsector': subsector, 'company': company}


            logging.info(f"Processed {count} tickers with data from {file_path}")

        except pd.errors.EmptyDataError:
            logging.warning(f"Ticker source file is empty: {file_path}")
        except FileNotFoundError: # Should be caught by is_file() check, but good practice
             logging.error(f"Ticker source file not found (double check): {file_path}")
        except Exception as e:
            logging.error(f"Error reading ticker file {file_path}: {e}")
            # Decide if you want to stop processing other files on error
            # For robustness, we continue to the next file by default

    logging.info(f"Total unique tickers loaded with data from all sources: {len(all_ticker_data)}")
    return all_ticker_data


# Example Usage (Optional - for testing)
if __name__ == '__main__':
    ensure_data_dir_exists()

    # Test filename generation
    test_ticker = "AAPL.US"
    start = date(2023, 1, 1)
    end = date(2023, 1, 31)
    fname = generate_filename(test_ticker, start, end)
    print(f"Generated filename: {fname}")

    # Create a dummy file to test date finding
    dummy_fname_1 = generate_filename(test_ticker, date(2022, 12, 1), date(2022, 12, 31))
    dummy_fname_2 = generate_filename(test_ticker, date(2023, 1, 1), date(2023, 1, 15)) # This one is later
    dummy_fname_3 = generate_filename("MSFT-US", date(2023, 1, 1), date(2023, 1, 20)) # Different ticker
    dummy_fname_4 = DATA_DIR / "AAPL_US_badformat.csv" # Malformed name

    # Create dummy files (ensure directory exists first)
    if not DATA_DIR.exists(): DATA_DIR.mkdir()
    dummy_fname_1.touch()
    dummy_fname_2.touch()
    dummy_fname_3.touch()
    dummy_fname_4.touch()
    print(f"Created dummy files: {dummy_fname_1.name}, {dummy_fname_2.name}, {dummy_fname_3.name}, {dummy_fname_4.name}")


    # Test finding the latest date
    latest_aapl_date = get_latest_date_for_ticker(test_ticker)
    print(f"Latest date found for {test_ticker}: {latest_aapl_date}") # Should be 2023-01-15

    latest_msft_date = get_latest_date_for_ticker("MSFT-US")
    print(f"Latest date found for MSFT-US: {latest_msft_date}") # Should be 2023-01-20

    latest_goog_date = get_latest_date_for_ticker("GOOG.US")
    print(f"Latest date found for GOOG.US: {latest_goog_date}") # Should be None

    # Test saving data (create a dummy DataFrame)
    dummy_data = {'Date': ['2023-01-16', '2023-01-17'], 'Open': [150, 151], 'Close': [151, 152]}
    df_to_save = pd.DataFrame(dummy_data)
    save_filename = generate_filename(test_ticker, date(2023, 1, 16), date(2023, 1, 17))
    save_data_to_csv(df_to_save, save_filename)

    # Clean up dummy files
    # dummy_fname_1.unlink()
    # dummy_fname_2.unlink()
    # dummy_fname_3.unlink()
    # dummy_fname_4.unlink()
    # save_filename.unlink()
    # print("Cleaned up dummy files.")
    # Optionally remove the directory if it was created empty
    # try:
    #     DATA_DIR.rmdir() # Only removes if empty
    # except OSError:
    #     pass # Directory not empty, which is fine if save_data_to_csv ran