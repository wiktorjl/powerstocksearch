-- delete from company_profile;
-- delete from ohlc_data;
delete from indicator_definitions;
delete from indicators;
-- delete from symbols;


INSERT INTO indicator_definitions (name, description)
VALUES
('SMA_5', '5-day Simple Moving Average'),
('SMA_10', '10-day Simple Moving Average'),
('SMA_100', '100-day Simple Moving Average'),
('SMA_150', '150-day Simple Moving Average');

INSERT INTO indicator_definitions (name, description)
VALUES
('HIGH_100', '100-day High'),
('HIGH_150', '150-day High'),
('HIGH_200', '200-day High'),
('HIGH_250', '250-day High');

WITH sma_data AS (
    SELECT
        timestamp,
        symbol_id,
        AVG(close) OVER w5 AS sma_5,
        AVG(close) OVER w10 AS sma_10,
        AVG(close) OVER w100 AS sma_100,
        AVG(close) OVER w150 AS sma_150
    FROM
        ohlc_data
    WINDOW
        w5 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
        w10 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
        w100 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),
        w150 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 149 PRECEDING AND CURRENT ROW)
)
INSERT INTO indicators (timestamp, symbol_id, indicator_id, value)
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'SMA_5'), sma_5 FROM sma_data WHERE sma_5 IS NOT NULL
UNION ALL
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'SMA_10'), sma_10 FROM sma_data WHERE sma_10 IS NOT NULL
UNION ALL
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'SMA_100'), sma_100 FROM sma_data WHERE sma_100 IS NOT NULL
UNION ALL
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'SMA_150'), sma_150 FROM sma_data WHERE sma_150 IS NOT NULL;

commit;


WITH high_data AS (
    SELECT
        timestamp,
        symbol_id,
        MAX(high) OVER w100 AS high_100,
        MAX(high) OVER w150 AS high_150,
        MAX(high) OVER w200 AS high_200,
        MAX(high) OVER w250 AS high_250
    FROM
        ohlc_data
    WINDOW
        w100 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 99 PRECEDING AND CURRENT ROW),
        w150 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 149 PRECEDING AND CURRENT ROW),
        w200 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 199 PRECEDING AND CURRENT ROW),
        w250 AS (PARTITION BY symbol_id ORDER BY timestamp ROWS BETWEEN 249 PRECEDING AND CURRENT ROW)
)
INSERT INTO indicators (timestamp, symbol_id, indicator_id, value)
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'HIGH_100'), high_100 FROM high_data WHERE high_100 IS NOT NULL
UNION ALL
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'HIGH_150'), high_150 FROM high_data WHERE high_150 IS NOT NULL
UNION ALL
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'HIGH_200'), high_200 FROM high_data WHERE high_200 IS NOT NULL
UNION ALL
SELECT timestamp, symbol_id, (SELECT indicator_id FROM indicator_definitions WHERE name = 'HIGH_250'), high_250 FROM high_data WHERE high_250 IS NOT NULL;

COMMIT;