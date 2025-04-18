import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Parameters (Defaults based on spec, ideally load from config file) ---
# These should match the values in specifications/stock_reversal_screen.md
# Consider moving these to src/config.py or a dedicated config mechanism
DEFAULT_CONFIG = {
    "MIN_DATA_LENGTH": 475,
    "DOWNTREND_LOOKBACK": 252,
    "DOWNTREND_PERFORMANCE_THRESHOLD": -0.20,
    "DOWNTREND_PRICE_MA_PERIOD": 200,
    "DOWNTREND_PRICE_MA_PCT_BELOW": 0.75,
    "PRIMARY_MA_PERIOD": 150,
    "SLOPE_MA_PERIOD": 150, # Same as PRIMARY_MA_PERIOD in spec
    "SLOPE_LOOKBACK_N": 20,
    "SLOPE_FLAT_THRESHOLD": 0.0005, # Normalized slope
    "SLOPE_RISING_THRESHOLD": 0.001, # Normalized slope
    "SLOPE_FLAT_LOOKBACK_M": 10,
    "PRICE_CONFIRMATION_PERIOD": 5,
    "VOLUME_AVG_PERIOD": 50,
    "VOLUME_SPIKE_MULTIPLIER": 1.5,
    "RSI_PERIOD": 14,
    "RSI_CONFIRMATION_THRESHOLD": 50,
}

# --- Helper Functions ---

def calculate_sma(data_series: pd.Series, period: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    if len(data_series) < period:
        return pd.Series(index=data_series.index, dtype=float) # Return empty series aligned with index
    return data_series.rolling(window=period).mean()

def calculate_rsi(data_series: pd.Series, period: int) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    if len(data_series) <= period:
        return pd.Series(index=data_series.index, dtype=float)

    delta = data_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_normalized_sma_slope(sma_series: pd.Series, lookback_n: int, sma_value_series: pd.Series) -> pd.Series:
    """
    Calculates normalized slope of the SMA using Linear Regression over lookback_n periods.
    Normalization is done by dividing the raw slope by the current SMA value.
    """
    if len(sma_series) < lookback_n:
        return pd.Series(index=sma_series.index, dtype=float)

    slopes = pd.Series(index=sma_series.index, dtype=float)
    x = np.arange(lookback_n)

    for i in range(lookback_n -1, len(sma_series)):
        y_window = sma_series.iloc[i - lookback_n + 1 : i + 1]

        # Ensure no NaN values in the window for regression
        if y_window.isnull().any():
            slopes.iloc[i] = np.nan
            continue

        # Fit linear regression
        try:
            # Using scipy.stats.linregress for slope calculation
            result = stats.linregress(x, y_window)
            raw_slope = result.slope
        except ValueError as e:
            logging.warning(f"Linregress failed at index {i}: {e}. Setting slope to NaN.")
            slopes.iloc[i] = np.nan
            continue


        # Normalize slope by current SMA value
        current_sma = sma_value_series.iloc[i]
        if pd.notna(current_sma) and abs(current_sma) > 1e-6: # Avoid division by zero/very small numbers
            normalized_slope = raw_slope / current_sma
        else:
            normalized_slope = 0.0 # Assign 0 if SMA is zero or NaN

        slopes.iloc[i] = normalized_slope

    return slopes


def check_preceding_downtrend(close_prices: pd.Series, sma200: pd.Series, lookback: int, perf_threshold: float, pct_below_threshold: float) -> bool:
    """Checks for ~1 year downtrend based on performance and price vs SMA200."""
    if len(close_prices) < lookback or len(sma200) < lookback:
        return False

    # Performance Check
    current_price = close_prices.iloc[-1]
    price_lookback_ago = close_prices.iloc[-lookback]
    if price_lookback_ago == 0: return False # Avoid division by zero
    performance = (current_price / price_lookback_ago) - 1
    if performance >= perf_threshold:
        return False

    # Price vs MA Check
    prices_in_lookback = close_prices.iloc[-lookback:]
    sma200_in_lookback = sma200.iloc[-lookback:]
    days_below_sma = (prices_in_lookback < sma200_in_lookback).sum()
    pct_below_sma = days_below_sma / lookback
    if pct_below_sma < pct_below_threshold:
        return False

    return True

def check_sma_slope_sequence(normalized_slope_series: pd.Series, flat_thresh: float, rising_thresh: float, flat_lookback_m: int) -> bool:
    """Checks if SMA slope was recently flat and is now rising."""
    if len(normalized_slope_series) < flat_lookback_m + 1: # Need enough data for lookback + current
        return False

    current_slope = normalized_slope_series.iloc[-1]
    if pd.isna(current_slope) or current_slope <= rising_thresh:
        return False # Not currently rising enough or NaN

    # Check if it was flat recently (in the M days *before* the current day)
    was_flat_recently = False
    recent_slopes = normalized_slope_series.iloc[-flat_lookback_m-1 : -1]
    if len(recent_slopes) == 0: # Should not happen with initial check, but safety first
        return False

    for slope in recent_slopes:
        if pd.notna(slope) and abs(slope) < flat_thresh:
            was_flat_recently = True
            break

    return was_flat_recently

def check_price_confirmation(close_prices: pd.Series, sma150: pd.Series, confirmation_period: int) -> bool:
    """Checks if Close > SMA150 for the last 'confirmation_period' days."""
    if len(close_prices) < confirmation_period or len(sma150) < confirmation_period:
        return False

    # Check last 'confirmation_period' days (indices -1 down to -confirmation_period)
    for i in range(1, confirmation_period + 1):
        if pd.isna(close_prices.iloc[-i]) or pd.isna(sma150.iloc[-i]) or close_prices.iloc[-i] <= sma150.iloc[-i]:
            return False
    return True

def check_confirmation_indicators(current_volume: float, current_avg_volume: float, current_rsi: float, vol_multiplier: float, rsi_threshold: float) -> bool:
    """Checks optional Volume Spike and RSI level."""
    if pd.isna(current_volume) or pd.isna(current_avg_volume) or pd.isna(current_rsi):
        return False

    volume_ok = current_volume > (current_avg_volume * vol_multiplier) if current_avg_volume > 0 else False
    rsi_ok = current_rsi > rsi_threshold

    # Requires BOTH confirmations as per spec
    return volume_ok and rsi_ok

def add_stock_details(qualified_stocks_list: List[Dict], symbol: str, ohlcv: pd.DataFrame, sma150: pd.Series, slope: pd.Series, rsi: pd.Series):
    """Adds relevant details for the qualified stock to the output list."""
    try:
        details = {
            "symbol": symbol,
            "last_close": ohlcv['Close'].iloc[-1],
            "last_volume": ohlcv['Volume'].iloc[-1],
            "sma150": sma150.iloc[-1],
            "sma150_slope_norm": slope.iloc[-1],
            "rsi14": rsi.iloc[-1],
            "last_date": ohlcv.index[-1].strftime('%Y-%m-%d') # Assuming index is DateTimeIndex
        }
        qualified_stocks_list.append(details)
    except IndexError:
        logging.warning(f"Could not extract details for {symbol} due to index error.")
    except Exception as e:
        logging.error(f"Error adding details for {symbol}: {e}")


# --- Main Screening Function ---

def screen_stocks(stock_symbols: List[str], historical_data_map: Dict[str, pd.DataFrame], config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Screens a list of stock symbols based on the reversal strategy criteria.

    Args:
        stock_symbols: A list of stock symbols to screen.
        historical_data_map: A dictionary mapping symbols to their OHLCV DataFrames.
                             DataFrames must have a DateTimeIndex and columns 'Open', 'High', 'Low', 'Close', 'Volume'.
        config: An optional dictionary with configuration parameters. Uses defaults if None.

    Returns:
        A list of dictionaries, where each dictionary contains details of a stock
        that met all the screening criteria.
    """
    if not stock_symbols or not historical_data_map:
        logging.warning("Screening called with empty stock list or data map.")
        return []

    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config) # Override defaults with provided config

    qualified_stocks = []
    logging.info(f"Starting reversal screen for {len(stock_symbols)} symbols...")

    processed_count = 0
    qualified_count = 0

    for symbol in stock_symbols:
        ohlcv_data = historical_data_map.get(symbol)

        if ohlcv_data is None or ohlcv_data.empty:
            logging.debug(f"Skipping {symbol}: No data found.")
            continue

        if len(ohlcv_data) < cfg["MIN_DATA_LENGTH"]:
            logging.debug(f"Skipping {symbol}: Insufficient data ({len(ohlcv_data)} < {cfg['MIN_DATA_LENGTH']}).")
            continue

        # Ensure data is sorted by date (index)
        ohlcv_data = ohlcv_data.sort_index()

        # --- Calculate Indicators ---
        try:
            sma150 = calculate_sma(ohlcv_data['Close'], cfg["PRIMARY_MA_PERIOD"])
            sma200 = calculate_sma(ohlcv_data['Close'], cfg["DOWNTREND_PRICE_MA_PERIOD"])
            # Pass sma150 itself for normalization reference
            sma150_slope_normalized = calculate_normalized_sma_slope(sma150, cfg["SLOPE_LOOKBACK_N"], sma150)
            rsi14 = calculate_rsi(ohlcv_data['Close'], cfg["RSI_PERIOD"])
            avg_volume_50 = calculate_sma(ohlcv_data['Volume'], cfg["VOLUME_AVG_PERIOD"])

            # Check if essential indicators could be calculated for the latest period
            if sma150.isnull().iloc[-1] or sma200.isnull().iloc[-1] or sma150_slope_normalized.isnull().iloc[-1] or rsi14.isnull().iloc[-1] or avg_volume_50.isnull().iloc[-1]:
                 logging.debug(f"Skipping {symbol}: Could not calculate all required indicators for the latest date.")
                 continue

        except Exception as e:
            logging.error(f"Error calculating indicators for {symbol}: {e}")
            continue # Skip stock if indicators fail

        # --- Apply Filters ---
        try:
            # Filter 1: Preceding Downtrend
            is_downtrend = check_preceding_downtrend(
                ohlcv_data['Close'], sma200, cfg["DOWNTREND_LOOKBACK"],
                cfg["DOWNTREND_PERFORMANCE_THRESHOLD"], cfg["DOWNTREND_PRICE_MA_PCT_BELOW"]
            )
            if not is_downtrend:
                logging.debug(f"Skipping {symbol}: Failed Downtrend filter.")
                continue

            # Filter 2: SMA Flattening followed by Rising
            is_sma_turning_up = check_sma_slope_sequence(
                sma150_slope_normalized, cfg["SLOPE_FLAT_THRESHOLD"],
                cfg["SLOPE_RISING_THRESHOLD"], cfg["SLOPE_FLAT_LOOKBACK_M"]
            )
            if not is_sma_turning_up:
                logging.debug(f"Skipping {symbol}: Failed SMA Slope Sequence filter.")
                continue

            # Filter 3: Price Strength Confirmation
            is_price_confirmed = check_price_confirmation(
                ohlcv_data['Close'], sma150, cfg["PRICE_CONFIRMATION_PERIOD"]
            )
            if not is_price_confirmed:
                logging.debug(f"Skipping {symbol}: Failed Price Confirmation filter.")
                continue

            # Filter 4: Optional Confirmation Indicators
            passes_confirmation = check_confirmation_indicators(
                ohlcv_data['Volume'].iloc[-1], avg_volume_50.iloc[-1], rsi14.iloc[-1],
                cfg["VOLUME_SPIKE_MULTIPLIER"], cfg["RSI_CONFIRMATION_THRESHOLD"]
            )
            if not passes_confirmation:
                logging.debug(f"Skipping {symbol}: Failed Confirmation Indicators filter.")
                continue

            # If all filters pass, add stock to the list
            logging.info(f"Stock {symbol} qualified for reversal pattern.")
            add_stock_details(qualified_stocks, symbol, ohlcv_data, sma150, sma150_slope_normalized, rsi14)
            qualified_count += 1

        except IndexError:
             logging.warning(f"Skipping {symbol} due to IndexError during filtering (likely insufficient data near end).")
             continue
        except Exception as e:
            logging.error(f"Error applying filters for {symbol}: {e}")
            continue # Skip stock if filters fail

        processed_count += 1
        if processed_count % 100 == 0:
            logging.info(f"Processed {processed_count}/{len(stock_symbols)} symbols...")

    logging.info(f"Reversal screen finished. Found {qualified_count} qualified stocks out of {processed_count} processed.")
    return qualified_stocks