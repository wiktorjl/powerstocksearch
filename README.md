# Power Stock Search - Data Pipeline

This project provides a data pipeline for downloading, processing, and storing end-of-day (EOD) stock market data. It fetches data from EODHD, adjusts for stock splits, calculates technical indicators (like support/resistance), and stores the results in a PostgreSQL database.

## Project Overview

The core workflow consists of several distinct steps, executed sequentially:

1.  **Download Raw Data**: Fetch historical OHLCV data from the EODHD API.
2.  **Adjust for Splits**: Adjust historical prices and volumes based on stock split events.
3.  **Load to Database**: Import the split-adjusted data into a PostgreSQL database.
4.  **Analyze Data**: Retrieve data from the database and apply feature engineering techniques (e.g., calculate support/resistance levels).

## Setup

### 1. Environment Variables

Create a `.env` file in the project root directory with the following variables:

```dotenv
# EODHD API Key (Required for downloading data and splits)
EODHD_API_KEY="YOUR_EODHD_API_KEY"

# PostgreSQL Database Credentials (Required for storing data)
DB_NAME="your_db_name"
DB_USER="your_db_user"
DB_PASSWORD="your_db_password"
DB_HOST="your_db_host" # e.g., localhost
DB_PORT="your_db_port" # e.g., 5432
```

Replace the placeholder values with your actual credentials.

### 2. Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 1: Download Raw Data (`src/main.py`)

This script downloads raw EOD data for tickers specified in `src/config.py` (under `TICKER_SOURCES`). It checks the `incoming/` directory for existing data and only fetches new data since the last recorded date for each ticker.

**Configuration:**
*   Modify `TICKER_SOURCES` in `src/config.py` to point to your CSV/TSV files containing ticker symbols.
*   Set `DEFAULT_START_DATE` in `src/main.py` for the initial download if no local data exists.

**Execution:**

```bash
python -m src.main
```

**Output:**
*   Raw OHLCV data saved as CSV files in the `incoming/` directory (e.g., `incoming/AAPL_20000101_20250412.csv`).
*   A `tickers.txt` file listing all unique tickers processed.

### Step 2: Adjust for Splits (`src/feature_engineering/split_adjuster.py`)

This script reads the raw CSV files from `incoming/`, fetches split data from EODHD for each ticker, adjusts the 'Open', 'High', 'Low', 'Close', and 'Volume' columns based on the splits (using backward propagation), and saves the adjusted data.

**Execution:**

```bash
python -m src.feature_engineering.split_adjuster
```

**Output:**
*   Split-adjusted OHLCV data saved as JSON files in the `splitadjusted/` directory (e.g., `splitadjusted/AAPL.json`).

### Step 3: Prep the database

Potentially clean up the old database:
docker compose down --volumes
python3 -m src.database.init_db


### Step 4: Load Data into Database (`src/database/ohlc_split_importer.py`)

python3 -m src.database.ohlc_split_importer

### Step 5: Load company info

python3 -m src.database.company_info_importer


### Step 6: Calculate indicators

python3 -m src.database.indicator_calculator