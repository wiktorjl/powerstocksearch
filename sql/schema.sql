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

commit;