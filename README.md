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

### 3. Database Schema

Ensure your PostgreSQL database has the necessary tables. You can use a migration tool or execute the following SQL (adjust types/constraints as needed):

```sql
-- symbols table stores unique stock tickers
CREATE TABLE IF NOT EXISTS symbols (
    symbol_id SERIAL PRIMARY KEY,
    symbol VARCHAR(16) UNIQUE NOT NULL,
    -- Add other metadata if needed (e.g., name, exchange)
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ohlc_data table stores the split-adjusted time series data
CREATE TABLE IF NOT EXISTS ohlc_data (
    ohlc_id BIGSERIAL PRIMARY KEY,
    symbol_id INTEGER NOT NULL REFERENCES symbols(symbol_id),
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(19, 4),
    high NUMERIC(19, 4),
    low NUMERIC(19, 4),
    close NUMERIC(19, 4),
    volume BIGINT,
    -- Add other indicators or adjusted prices if needed
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (symbol_id, timestamp) -- Ensure data uniqueness per symbol per timestamp
);

-- Optional: Create indexes for faster querying
CREATE INDEX IF NOT EXISTS idx_ohlc_data_symbol_id_timestamp ON ohlc_data (symbol_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols (symbol);
```

## Workflow & Usage

Execute the following scripts in order:

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

### Step 3: Load Data into Database (`src/database/split_adjusted_importer.py`)

This script reads the adjusted JSON files from `splitadjusted/`, connects to the PostgreSQL database (using credentials from `.env`), and inserts the data into the `symbols` and `ohlc_data` tables. It uses `ON CONFLICT DO NOTHING` to avoid duplicates based on `(symbol_id, timestamp)`.

**Execution:**

```bash
python -m src.database.split_adjusted_importer
```

**Output:**
*   Data inserted into the `symbols` and `ohlc_data` tables in your PostgreSQL database.

### Step 4: Accessing and Analyzing Data

Once the data is in the database, you can use the provided modules to access and analyze it.

**Example: Fetching Data**

Use the `fetch_ohlc_data_db` function from `src.database.data_provider` to get data for a specific symbol:

```python
import logging
from src.database.data_provider import fetch_ohlc_data_db
from src.feature_engineering.support_resistance import identify_support_resistance
from src.config import AlgorithmConfig # Import the config class

# Configure logging if running standalone
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

symbol = "AAPL"
start_date = "2024-01-01"
end_date = "2024-12-31"

# Fetch data from the database
ohlc_df = fetch_ohlc_data_db(symbol, start_date, end_date)

if ohlc_df is not None:
    print(f"Fetched {len(ohlc_df)} records for {symbol}")
    print(ohlc_df.head())

    # Example: Calculate Support/Resistance
    # Ensure you have the AlgorithmConfig defined or imported
    sr_config = AlgorithmConfig() # Use default config values
    sr_levels = identify_support_resistance(ohlc_df, sr_config)

    if sr_levels:
        print(f"\nIdentified S/R Levels for {symbol}:")
        print([round(level, 2) for level in sr_levels])
    else:
        print(f"\nCould not identify S/R levels for {symbol} (check data/config).")

else:
    print(f"Failed to fetch data for {symbol}")

```

**Example: Calculating Support/Resistance**

Use the `identify_support_resistance` function from `src.feature_engineering.support_resistance` along with the configuration from `src.config.AlgorithmConfig`:

```python
# (Continuing from the previous example where ohlc_df is loaded)
from src.feature_engineering.support_resistance import identify_support_resistance
from src.config import AlgorithmConfig

if ohlc_df is not None:
    # Load algorithm configuration
    sr_config = AlgorithmConfig()

    # Calculate levels
    support_resistance_levels = identify_support_resistance(ohlc_df, sr_config)

    if support_resistance_levels:
        print(f"Support/Resistance Levels for {symbol}: {support_resistance_levels}")
    else:
        print(f"Could not calculate S/R levels for {symbol}.")
```

## Configuration (`src/config.py`)

This file centralizes key configurations:

*   **`EODHD_API_KEY`**: Loaded from `.env`.
*   **Database Credentials**: `DB_NAME`, `DB_USER`, etc., loaded from `.env`.
*   **`TICKER_SOURCES`**: A list of dictionaries specifying paths to CSV/TSV files and the column containing ticker symbols. Used by `src/main.py` and `src/feature_engineering/split_adjuster.py`.
*   **`AlgorithmConfig`**: A class containing parameters for the support/resistance calculation algorithm used by `src.feature_engineering.support_resistance`.

Modify this file to adjust data sources or algorithm behavior.