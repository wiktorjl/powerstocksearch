import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
EODHD_API_KEY: Optional[str] = os.getenv("EODHD_API_KEY")


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