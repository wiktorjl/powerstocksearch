import finnhub
import os
import time
import json
import argparse

# --- Configuration ---
# Read API key from environment variable
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
SYMBOLS_FILE = "symbols.txt"
OUTPUT_DIR = "data"
RATE_LIMIT_PER_MINUTE = 58
SLEEP_INTERVAL = 60 / RATE_LIMIT_PER_MINUTE # ~1.03 seconds

# --- Functions ---

def get_company_profile(symbol: str, client: finnhub.Client):
    """
    Fetches the company profile for a given stock symbol using the Finnhub API.

    Args:
        symbol: The stock symbol (e.g., "AAPL").
        client: An initialized Finnhub client instance.

    Returns:
        A dictionary containing the company profile information, or None if
        an error occurs.
    """
    try:
        profile = client.company_profile2(symbol=symbol)
        return profile
    except finnhub.FinnhubAPIException as e:
        print(f"Finnhub API error fetching profile for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching profile for {symbol}: {e}")
        return None

def read_symbols(filepath: str) -> list[str]:
    """Reads symbols from a file, one per line."""
    try:
        with open(filepath, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        print(f"Read {len(symbols)} symbols from {filepath}")
        return symbols
    except FileNotFoundError:
        print(f"Error: Symbols file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading symbols file {filepath}: {e}")
        return []

def save_profile(symbol: str, profile_data: dict, output_dir: str):
    """Saves the profile data to a JSON file."""
    filepath = os.path.join(output_dir, f"{symbol}.json")
    try:
        os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=4)
        print(f"Successfully saved profile for {symbol} to {filepath}")
    except Exception as e:
        print(f"Error saving profile for {symbol} to {filepath}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Finnhub company profiles for symbols listed in a file.")
    parser.add_argument(
        "--symbols-file",
        default=SYMBOLS_FILE,
        help=f"Path to the file containing stock symbols (one per line). Default: {SYMBOLS_FILE}"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Directory to save the output JSON files. Default: {OUTPUT_DIR}"
    )
    args = parser.parse_args()

    print("--- Finnhub Company Profile Downloader ---")

    if not FINNHUB_API_KEY:
        print("Error: FINNHUB_API_KEY environment variable is not set.")
        print("Please set the FINNHUB_API_KEY environment variable and try again.")
        exit(1) # Use exit(1) for errors

    symbols_to_process = read_symbols(args.symbols_file)
    if not symbols_to_process:
        print("No symbols to process. Exiting.")
        exit(0) # Use exit(0) for normal exit

    # Initialize Finnhub client once
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
        # Test connection with a simple call (optional, but good practice)
        # finnhub_client.quote('AAPL')
        print("Finnhub client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Finnhub client: {e}")
        exit(1)

    print(f"Output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True) # Ensure directory exists before loop

    processed_count = 0
    start_time = time.time()

    for i, symbol in enumerate(symbols_to_process):
        print(f"\n[{i+1}/{len(symbols_to_process)}] Processing symbol: {symbol}")

        # Check if output file already exists
        output_filepath = os.path.join(args.output_dir, f"{symbol}.json")
        if os.path.exists(output_filepath):
            print(f"Output file {output_filepath} already exists. Skipping API call.")
            continue # Move to the next symbol, skipping API call and sleep

        # File doesn't exist, proceed with API call
        profile_data = get_company_profile(symbol, finnhub_client)

        if profile_data:
            save_profile(symbol, profile_data, args.output_dir)
            processed_count += 1
        else:
            print(f"Skipping {symbol} due to previous error.")

        # Rate limiting: Sleep after each request
        print(f"Sleeping for {SLEEP_INTERVAL:.2f} seconds...")
        time.sleep(SLEEP_INTERVAL)

    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Download Complete ---")
    print(f"Processed {processed_count} symbols out of {len(symbols_to_process)}.")
    print(f"Total time taken: {duration:.2f} seconds.")
    print(f"Results saved in: {args.output_dir}")
    print("-------------------------")