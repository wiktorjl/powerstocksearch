-- Create the symbols table
CREATE TABLE symbols (
    symbol_id SERIAL PRIMARY KEY,
    symbol TEXT UNIQUE NOT NULL
);

-- Create the ohlc_data table
CREATE TABLE ohlc_data (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol_id INTEGER REFERENCES symbols(symbol_id),
    open DECIMAL(18,6),
    high DECIMAL(18,6),
    low DECIMAL(18,6),
    close DECIMAL(18,6),
    volume BIGINT,
    UNIQUE (timestamp, symbol_id)
);

-- -- Convert ohlc_data to a hypertable, partitioned by timestamp
-- SELECT create_hypertable('ohlc_data', 'timestamp');

-- Create the table to define indicators
CREATE TABLE indicator_definitions (
    indicator_id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    description TEXT
);

-- Create the table to store indicator values
CREATE TABLE indicators (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol_id INTEGER REFERENCES symbols(symbol_id),
    indicator_id INTEGER REFERENCES indicator_definitions(indicator_id),
    value DECIMAL(18,6),
    UNIQUE (timestamp, symbol_id, indicator_id)
);

-- If using TimescaleDB, turn indicators into a hypertable for time-series efficiency
-- SELECT create_hypertable('indicators', 'timestamp');

CREATE TABLE company_profile (
	symbol_id INTEGER REFERENCES symbols(symbol_id),
    country VARCHAR(2),
    currency VARCHAR(3),
    estimate_currency VARCHAR(3),
    exchange VARCHAR(100),
    finnhub_industry VARCHAR(100),
    ipo DATE,
    logo VARCHAR(255),
    market_capitalization DECIMAL(15, 2),
    name VARCHAR(100),
    phone VARCHAR(20),
    share_outstanding DECIMAL(10, 2),
    ticker VARCHAR(10),
    weburl VARCHAR(255)
);

-- Create the splits table
CREATE TABLE splits
(
    symbol_id integer NOT NULL REFERENCES symbols(symbol_id),
    split_date timestamp with time zone NOT NULL,
    ratio numeric NOT NULL,
    CONSTRAINT splits_pkey PRIMARY KEY (symbol_id, split_date)
);

-- Create table to store the latest reversal scan results
CREATE TABLE reversal_scan_results (
    symbol TEXT PRIMARY KEY REFERENCES symbols(symbol), -- Use symbol directly as PK and FK
    last_close DECIMAL(18,6),
    last_volume BIGINT,
    sma150 DECIMAL(18,6),
    sma150_slope_norm DECIMAL(18,8), -- Increased precision for slope
    rsi14 DECIMAL(18,6),
    last_date DATE,
    scan_timestamp TIMESTAMPTZ NOT NULL -- Timestamp of when the scan was run
);

commit;