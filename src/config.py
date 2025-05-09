import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
from typing import Optional


# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
EODHD_API_KEY: Optional[str] = os.getenv("EODHD_API_KEY")

FINNHUB_API_KEY: Optional[str] = os.getenv("FINNHUB_API_KEY")

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
# Retrieve Database connection details
DB_NAME: Optional[str] = os.getenv("DB_NAME")
DB_USER: Optional[str] = os.getenv("DB_USER")
DB_PASSWORD: Optional[str] = os.getenv("DB_PASSWORD")
DB_HOST: Optional[str] = os.getenv("DB_HOST")
DB_PORT: Optional[str] = os.getenv("DB_PORT")

# Basic validation for database variables (optional but recommended)
if not all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
    print("Error: One or more database configuration variables (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT) not found in .env file.")
    # Consider raising an exception or exiting
    # exit(1)

if not EODHD_API_KEY:
    print("Error: EODHD_API_KEY not found in .env file.")
    # Consider raising an exception or exiting depending on desired behavior
    # raise ValueError("EODHD_API_KEY not found in .env file.")
    # exit(1)


if not FINNHUB_API_KEY:
    print("Warning: FINNHUB_API_KEY not found in .env file. Finnhub features will be unavailable.")
    # Depending on requirements, you might want to raise an error or exit
    # raise ValueError("FINNHUB_API_KEY not found in .env file.")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in .env file. Company descriptions will be unavailable.")
    # Depending on requirements, you might want to raise an error or exit
    # raise ValueError("OPENAI_API_KEY not found in .env file.")




# --- Data Directory Configuration ---
# Base directory for external stock data (can be overridden by env var)
STOCK_DATA_BASE_DIR = os.getenv("STOCK_DATA_BASE_DIR", "/home/user/stockdata")

# Specific data directories
COMPANY_INFO_DIR = os.path.join(STOCK_DATA_BASE_DIR, "company_info")
SPLIT_ADJUSTED_DIR = os.path.join(STOCK_DATA_BASE_DIR, "splitadjusted")
INCOMING_DATA_DIR = os.path.join(STOCK_DATA_BASE_DIR, "incoming")
SPLIT_DATA_DIR = os.path.join(STOCK_DATA_BASE_DIR, "splits") # Directory for split data JSON files

# --- Ticker Loading Configuration ---
# List of dictionaries, each specifying a CSV/TSV file and the column containing tickers.
# You can add more sources here or potentially load this from an environment variable.
TICKER_SOURCES: List[Dict[str, str]] = [
    {"path": "data/metadata/russel.csv", "symbol_column": "Symbol", "name_column": "Company", "sector_column": "GICS Sector", "subsector_column": "GICS Sub-Industry"},
    # Example for another file (e.g., TSV):
    {"path": "data/metadata/sp500.csv", "symbol_column": "Symbol", "name_column": "Company", "sector_column": "GICS Sector", "subsector_column": "GICS Sub-Industry"},
]

# You could potentially override TICKER_SOURCES via an environment variable
# e.g., TICKER_SOURCES_JSON = os.getenv("TICKER_SOURCES_JSON")
# if TICKER_SOURCES_JSON:
#     import json
#     try:
#         TICKER_SOURCES = json.loads(TICKER_SOURCES_JSON)
#         print(f"Loaded ticker sources from environment variable.")
#     except json.JSONDecodeError:
#         print("Warning: Invalid JSON in TICKER_SOURCES_JSON environment variable. Using default.")


# Parameters for Support/Resistance Algorithm
class AlgorithmConfig:
    # Key price point detection
    SWING_WINDOW = 30              # window (days) for swing high/lows (Increased from 20)
    ZIGZAG_THRESHOLD = 0.10        # 10% reversal threshold (Increased from 0.05)

    # Smoothing parameters for Savitzky-Golay filter
    SMOOTHING_WINDOW = 21          # window length for smoothing (must be odd)
    SMOOTHING_POLYORDER = 3        # polynomial order

    # Clustering: distance threshold as percentage of average price
    CLUSTER_DISTANCE_THRESHOLD_PERCENT = 0.05 # Increased further from 0.025

    # Time weighting (exponential decay factor)
    DECAY_FACTOR = 0.01

    # Maximum number of S/R lines to return
    MAX_SR_LINES = 3