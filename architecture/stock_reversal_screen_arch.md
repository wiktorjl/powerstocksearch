# Specification: Stock Reversal Screen

## 1. Objective

To identify stocks listed in the system that exhibit a specific bottoming and potential reversal pattern. This pattern consists of:
1.  A significant downtrend over approximately the past year.
2.  A flattening of the 150-day Simple Moving Average (SMA).
3.  A subsequent upward turn of the 150-day SMA.
4.  Confirmation of initial price strength by the closing price remaining above the 150-day SMA for a defined period.
5.  (Optional but Recommended) Confirmation via secondary indicators like volume and momentum.

The screen aims to filter a large universe of stocks down to a manageable list of candidates potentially transitioning from a long-term decline (Stage 4) into a basing phase (Stage 1) and showing early signs of entering a new uptrend (Stage 2), based on the principles outlined in `docs/reversal_strategy.txt`.

## 2. Inputs

The screening process requires the following data for each stock symbol to be analyzed:

*   **Stock Symbol:** Identifier for the stock (e.g., 'AAPL', 'MSFT').
*   **Historical OHLCV Data:** Daily Open, High, Low, Close, and Volume data.
    *   **Minimum Lookback Period:** Sufficient data to calculate all required indicators. This includes:
        *   ~252 trading days for the 1-year downtrend check.
        *   150 trading days for the 150-day SMA.
        *   Additional days for the SMA slope calculation lookback (`N`).
        *   Additional days for the 200-day SMA calculation.
        *   Total recommended minimum: ~475 trading days (~252 + 150 + 20 + buffer).
        *   // TDD: test_insufficient_data_handling - Ensure stocks with less than the minimum required data are skipped or handled gracefully.
*   **Configuration Parameters:** (These should be configurable, not hard-coded)
    *   `DOWNTREND_LOOKBACK`: 252 (days, ~1 year)
    *   `DOWNTREND_PERFORMANCE_THRESHOLD`: -0.20 (-20%)
    *   `DOWNTREND_PRICE_MA_PERIOD`: 200 (days)
    *   `DOWNTREND_PRICE_MA_PCT_BELOW`: 0.75 (75%)
    *   `PRIMARY_MA_PERIOD`: 150 (days)
    *   `SLOPE_MA_PERIOD`: 150 (days, same as primary for this spec)
    *   `SLOPE_LOOKBACK_N`: 20 (days for LinReg calculation)
    *   `SLOPE_FLAT_THRESHOLD`: 0.0005 (Normalized slope value)
    *   `SLOPE_RISING_THRESHOLD`: 0.001 (Normalized slope value)
    *   `SLOPE_FLAT_LOOKBACK_M`: 10 (days to check for prior flatness)
    *   `PRICE_CONFIRMATION_PERIOD`: 5 (consecutive days)
    *   `VOLUME_AVG_PERIOD`: 50 (days)
    *   `VOLUME_SPIKE_MULTIPLIER`: 1.5
    *   `RSI_PERIOD`: 14 (days)
    *   `RSI_CONFIRMATION_THRESHOLD`: 50

## 3. Processing Logic

The core logic involves iterating through each stock symbol with sufficient historical data and applying a series of filters based on the defined criteria.

```pseudocode
FUNCTION screen_stocks(stock_symbols, historical_data_map, config):
  // TDD: test_screen_stocks_empty_input - Handle empty stock list or data map.
  qualified_stocks = []

  FOR symbol IN stock_symbols:
    // TDD: test_screen_stocks_data_retrieval - Ensure correct data is fetched for symbol.
    ohlcv_data = historical_data_map.get(symbol)

    // Check for sufficient data
    IF ohlcv_data IS NULL OR length(ohlcv_data) < config.MIN_DATA_LENGTH: // MIN_DATA_LENGTH ~475
      CONTINUE // Skip stock if insufficient data

    // --- Calculate Indicators ---
    // TDD: test_indicator_calculation - Verify all indicators are calculated correctly.
    sma150 = calculate_sma(ohlcv_data.Close, config.PRIMARY_MA_PERIOD)
    sma200 = calculate_sma(ohlcv_data.Close, config.DOWNTREND_PRICE_MA_PERIOD)
    sma150_slope_normalized = calculate_normalized_sma_slope(
                                sma150,
                                config.SLOPE_LOOKBACK_N,
                                config.PRIMARY_MA_PERIOD // Pass SMA value for normalization
                              )
    rsi14 = calculate_rsi(ohlcv_data.Close, config.RSI_PERIOD)
    avg_volume_50 = calculate_sma(ohlcv_data.Volume, config.VOLUME_AVG_PERIOD)

    // --- Apply Filters ---
    // TDD: test_filter_logic_combinations - Test various combinations of passing/failing filters.

    // Filter 1: Preceding Downtrend
    // TDD: test_downtrend_filter
    is_downtrend = check_preceding_downtrend(
                      ohlcv_data.Close,
                      sma200,
                      config.DOWNTREND_LOOKBACK,
                      config.DOWNTREND_PERFORMANCE_THRESHOLD,
                      config.DOWNTREND_PRICE_MA_PCT_BELOW
                    )
    IF NOT is_downtrend:
      CONTINUE

    // Filter 2: SMA Flattening followed by Rising
    // TDD: test_sma_slope_sequence
    is_sma_turning_up = check_sma_slope_sequence(
                          sma150_slope_normalized,
                          config.SLOPE_FLAT_THRESHOLD,
                          config.SLOPE_RISING_THRESHOLD,
                          config.SLOPE_FLAT_LOOKBACK_M
                        )
    IF NOT is_sma_turning_up:
      CONTINUE

    // Filter 3: Price Strength Confirmation
    // TDD: test_price_strength_confirmation
    is_price_confirmed = check_price_confirmation(
                            ohlcv_data.Close,
                            sma150,
                            config.PRICE_CONFIRMATION_PERIOD
                          )
    IF NOT is_price_confirmed:
      CONTINUE

    // Filter 4: Optional Confirmation Indicators
    // TDD: test_confirmation_indicators
    passes_confirmation = check_confirmation_indicators(
                              ohlcv_data.Volume[-1], // Most recent volume
                              avg_volume_50[-1],     // Most recent avg volume
                              rsi14[-1],             // Most recent RSI
                              config.VOLUME_SPIKE_MULTIPLIER,
                              config.RSI_CONFIRMATION_THRESHOLD
                            )
    IF NOT passes_confirmation:
      CONTINUE

    // If all filters pass, add stock to the list
    add_stock_details(qualified_stocks, symbol, ohlcv_data, sma150, sma150_slope_normalized, rsi14) // Add relevant data for output

  RETURN qualified_stocks
END FUNCTION

// --- Helper Functions ---

FUNCTION calculate_sma(data_series, period):
  // Calculates Simple Moving Average
  // TDD: test_calculate_sma
  // Requires at least 'period' data points. Returns series of SMA values.
  // Implementation: e.g., pandas rolling mean
  RETURN sma_series
END FUNCTION

FUNCTION calculate_normalized_sma_slope(sma_series, lookback_n, sma_value_series):
  // Calculates normalized slope of the SMA using Linear Regression
  // TDD: test_calculate_normalized_sma_slope_positive
  // TDD: test_calculate_normalized_sma_slope_negative
  // TDD: test_calculate_normalized_sma_slope_flat
  // TDD: test_calculate_normalized_sma_slope_edge_cases (e.g., start of series)
  slopes = []
  FOR i FROM lookback_n TO length(sma_series):
    y = sma_series[i-lookback_n : i]
    x = range(lookback_n)
    // Fit linear regression: y = mx + c
    raw_slope = calculate_linear_regression_slope(x, y) // e.g., using numpy.polyfit or scipy.stats.linregress

    // Normalize slope by current SMA value (Slope / Price)
    // Avoid division by zero or very small SMA values
    current_sma = sma_value_series[i-1]
    IF abs(current_sma) > 0.0001: // Small tolerance
        normalized_slope = raw_slope / current_sma
    ELSE:
        normalized_slope = 0 // Or handle as NaN/error

    append normalized_slope to slopes
  RETURN slopes // Series of normalized slope values
END FUNCTION

FUNCTION calculate_rsi(data_series, period):
  // Calculates Relative Strength Index (RSI)
  // TDD: test_calculate_rsi
  // Requires at least 'period' + 1 data points. Returns series of RSI values.
  // Implementation: Standard RSI calculation using average gains/losses
  RETURN rsi_series
END FUNCTION

FUNCTION check_preceding_downtrend(close_prices, sma200, lookback, perf_threshold, pct_below_threshold):
  // Checks for ~1 year downtrend based on performance and price vs SMA200
  // TDD: test_check_preceding_downtrend_pass
  // TDD: test_check_preceding_downtrend_fail_performance
  // TDD: test_check_preceding_downtrend_fail_price_ma
  // Requires at least 'lookback' data points for close_prices and sma200 aligned.

  // Performance Check
  current_price = close_prices[-1]
  price_lookback_ago = close_prices[-lookback]
  performance = (current_price / price_lookback_ago) - 1
  IF performance >= perf_threshold:
    RETURN FALSE

  // Price vs MA Check
  prices_in_lookback = close_prices[-lookback:]
  sma200_in_lookback = sma200[-lookback:]
  days_below_sma = 0
  FOR i FROM 0 TO lookback-1:
    IF prices_in_lookback[i] < sma200_in_lookback[i]:
      days_below_sma += 1
  pct_below_sma = days_below_sma / lookback
  IF pct_below_sma < pct_below_threshold:
    RETURN FALSE

  RETURN TRUE
END FUNCTION

FUNCTION check_sma_slope_sequence(normalized_slope_series, flat_thresh, rising_thresh, flat_lookback_m):
  // Checks if SMA slope was recently flat and is now rising
  // TDD: test_check_sma_slope_sequence_pass
  // TDD: test_check_sma_slope_sequence_fail_not_rising
  // TDD: test_check_sma_slope_sequence_fail_not_recently_flat
  // Requires at least 'flat_lookback_m' slope values.

  current_slope = normalized_slope_series[-1]
  IF current_slope <= rising_thresh:
    RETURN FALSE // Not currently rising enough

  // Check if it was flat recently
  was_flat_recently = FALSE
  recent_slopes = normalized_slope_series[-flat_lookback_m : -1] // Slopes before the current one
  FOR slope IN recent_slopes:
    IF abs(slope) < flat_thresh:
      was_flat_recently = TRUE
      BREAK

  RETURN was_flat_recently
END FUNCTION

FUNCTION check_price_confirmation(close_prices, sma150, confirmation_period):
  // Checks if Close > SMA150 for the last 'confirmation_period' days
  // TDD: test_check_price_confirmation_pass
  // TDD: test_check_price_confirmation_fail
  // Requires at least 'confirmation_period' data points for close_prices and sma150 aligned.

  FOR i FROM 1 TO confirmation_period:
    IF close_prices[-i] <= sma150[-i]:
      RETURN FALSE
  RETURN TRUE
END FUNCTION

FUNCTION check_confirmation_indicators(current_volume, current_avg_volume, current_rsi, vol_multiplier, rsi_threshold):
  // Checks optional Volume Spike and RSI level
  // TDD: test_check_confirmation_indicators_pass_both
  // TDD: test_check_confirmation_indicators_pass_one (if logic allows OR)
  // TDD: test_check_confirmation_indicators_fail_both

  volume_ok = current_volume > (current_avg_volume * vol_multiplier)
  rsi_ok = current_rsi > rsi_threshold

  // Initial implementation requires BOTH confirmations
  // TDD: test_confirmation_logic_AND_vs_OR (Decide if AND or OR logic is desired)
  RETURN volume_ok AND rsi_ok
END FUNCTION

FUNCTION add_stock_details(qualified_stocks_list, symbol, ohlcv, sma150, slope, rsi):
    // Adds relevant details for the qualified stock to the output list
    // TDD: test_add_stock_details_format
    details = {
        "symbol": symbol,
        "last_close": ohlcv.Close[-1],
        "last_volume": ohlcv.Volume[-1],
        "sma150": sma150[-1],
        "sma150_slope_norm": slope[-1],
        "rsi14": rsi[-1]
        // Add other relevant data points as needed for display
    }
    append details to qualified_stocks_list
END FUNCTION

```

**Assumptions Made:**

*   **Downtrend Criteria:** Using Performance < -20% (1yr) AND Price < 200 SMA for > 75% of the last year.
*   **SMA Slope Calculation:** Using Linear Regression over `N=20` days.
*   **Slope Normalization:** Dividing raw slope by the current SMA150 value.
*   **Slope Thresholds:** `Threshold_flat = 0.0005`, `Threshold_rising = 0.001`. These are initial estimates and **require backtesting and tuning**.
*   **Slope Sequence:** Requiring the slope to be below `Threshold_flat` within the last `M=10` days before exceeding `Threshold_rising`.
*   **Confirmation Indicators:** Initially requiring **both** Volume Spike (>1.5x 50-day Avg) AND RSI(14) > 50. This can be adjusted (e.g., to require only one).
*   **Data Availability:** Assumes clean, daily OHLCV data is available via `historical_data_map`.

## 4. Outputs

The primary output of the screen will be a list of stock symbols that meet all the specified criteria. For each qualified stock, the output should ideally include relevant data points to facilitate further analysis or display in a UI table:

*   Stock Symbol
*   Last Closing Price
*   Last Volume
*   Current 150-day SMA Value
*   Current Normalized 150-day SMA Slope
*   Current RSI(14) Value
*   (Optional) Date the criteria were met.
*   (Optional) Link to a detailed chart page for the stock.

The output format should be structured (e.g., a list of dictionaries or objects) suitable for consumption by a web frontend or further processing.

## 5. Integration

*   **Backend (Flask):**
    *   A new Flask route (e.g., `/scan/reversal`) needs to be created in `src/web/flaskapp.py`.
    *   This route will trigger the `screen_stocks` function.
    *   It will fetch the necessary stock symbols and historical data (likely interacting with `src/database/data_provider.py`).
    *   It will pass the results (the `qualified_stocks` list) to a new HTML template.
    *   // TDD: test_flask_reversal_route_success - Test route returns 200 and expected data format.
    *   // TDD: test_flask_reversal_route_no_data - Test route handles cases where no stocks qualify.
*   **Frontend (HTML/Navbar):**
    *   A new HTML template (e.g., `templates/reversal_scan_results.html`) needs to be created to display the results in a user-friendly table format.
    *   A link to the new `/scan/reversal` route should be added to the main navigation bar (likely in a base template like `templates/base.html` or similar).
    *   // TDD: test_navbar_link_present - Ensure the link appears correctly in the UI.
    *   // TDD: test_reversal_results_table_render - Ensure the results table displays correctly with data.