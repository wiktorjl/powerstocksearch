delete from company_profile;
delete from ohlc_data;
delete from indicators;
delete from symbols;
delete from indicator_definitions;


INSERT INTO indicator_definitions (name, description)
VALUES
('SMA_5', '5-day Simple Moving Average'),
('SMA_10', '10-day Simple Moving Average'),
('SMA_100', '100-day Simple Moving Average'),
('SMA_150', '150-day Simple Moving Average');

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
