# support_resistance.py
import pandas as pd
import numpy as np
from datetime import datetime

# Import helper modules (config import removed from top level)
# from config import AlgorithmConfig # Removed - will be passed as argument
from .technical_indicators import find_swing_highs_lows, apply_zigzag_indicator # Renamed and relative import
from .smoothing import smooth_prices, calculate_derivative, find_sign_changes # Relative import
from .clustering_analysis import cluster_points, calculate_cluster_centroids # Renamed and relative import
from .weighting_schemes import adjust_levels_with_weights # Renamed and relative import
import logging # Import logging

logger = logging.getLogger(__name__) # Setup logger for this module

def clean_and_prepare_sr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare OHLC data specifically for SR analysis.
    Assumes df has 'timestamp', 'open', 'high', 'low', 'close' columns.
    Converts 'timestamp' to timezone-naive datetime, sorts, and handles basic cleaning.
    """
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None.")
        return pd.DataFrame()

    df_clean = df.copy()

    # --- Robust Timestamp Handling ---
    timestamp_col = 'timestamp'
    if timestamp_col not in df_clean.columns:
        logger.error(f"'{timestamp_col}' column not found in DataFrame.")
        return pd.DataFrame()

    # 1. Attempt conversion to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_clean[timestamp_col]):
        logger.info(f"'{timestamp_col}' column is not datetime type. Attempting conversion.")
        try:
            # Try parsing with UTC assumption first, handles tz-aware strings
            df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], utc=True, errors='coerce')
        except Exception as e1:
            logger.warning(f"Conversion with utc=True failed ({e1}). Trying without UTC.")
            try:
                # Fallback for naive strings or other formats
                df_clean[timestamp_col] = pd.to_datetime(df_clean[timestamp_col], errors='coerce')
            except Exception as e2:
                logger.error(f"Failed to convert '{timestamp_col}' to datetime: {e2}")
                return pd.DataFrame() # Cannot proceed without valid timestamps

    # 2. Handle Timezones: Convert to UTC then remove timezone info
    if pd.api.types.is_datetime64_any_dtype(df_clean[timestamp_col]) and df_clean[timestamp_col].dt.tz is not None:
        logger.info(f"'{timestamp_col}' is timezone-aware. Converting to UTC and making naive.")
        try:
            df_clean[timestamp_col] = df_clean[timestamp_col].dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception as e:
            logger.error(f"Error converting timezone for '{timestamp_col}': {e}. Cannot proceed reliably.")
            # Depending on requirements, could try tz_localize(None) directly, but UTC conversion is safer
            return pd.DataFrame()
    # --- End Timestamp Handling ---

    # Drop rows where timestamp conversion might have failed (resulted in NaT)
    initial_rows = len(df_clean)
    df_clean.dropna(subset=[timestamp_col], inplace=True)
    if len(df_clean) < initial_rows:
        logger.warning(f"Dropped {initial_rows - len(df_clean)} rows due to NaT timestamps after conversion.")

    if df_clean.empty:
        logger.warning("DataFrame is empty after timestamp handling.")
        return pd.DataFrame()

    # Sort by timestamp
    df_clean = df_clean.sort_values(timestamp_col).reset_index(drop=True)

    # Use 'close' price for calculations
    price_col = 'close'
    if price_col not in df_clean.columns:
        logger.error(f"Required column '{price_col}' not found.")
        return pd.DataFrame()
    df_clean['price'] = pd.to_numeric(df_clean[price_col], errors='coerce')

    # Fill missing 'price' values forward (simple approach)
    df_clean['price'] = df_clean['price'].fillna(method='ffill')

    # Remove any remaining NaNs (e.g., at the start or if 'close' conversion failed)
    initial_rows = len(df_clean)
    df_clean.dropna(subset=['price'], inplace=True)
    if len(df_clean) < initial_rows:
         logger.warning(f"Dropped {initial_rows - len(df_clean)} rows due to NaN prices after ffill.")

    if df_clean.empty:
        logger.warning("DataFrame is empty after price handling.")
        return pd.DataFrame()

    # Basic outlier removal (optional, adjust as needed)
    # threshold = df_clean['price'].mean() * 3
    # df_clean = df_clean[df_clean['price'] < threshold]

    logger.info(f"Data cleaning complete. Resulting shape: {df_clean.shape}")
    return df_clean

def identify_support_resistance(df: pd.DataFrame, config):
    """
    Main function to identify support and resistance levels from OHLC data.

    Args:
        df (pd.DataFrame): DataFrame containing OHLC data with a 'timestamp' column.
                           Expected columns: 'timestamp', 'open', 'high', 'low', 'close'.
        config: An object or dictionary containing algorithm parameters like
                SWING_WINDOW, ZIGZAG_THRESHOLD, SMOOTHING_WINDOW, SMOOTHING_POLYORDER,
                CLUSTER_DISTANCE_THRESHOLD_PERCENT, DECAY_FACTOR, MAX_SR_LINES.

    Returns:
        list: A list of the top N strongest support/resistance levels (prices),
              where N is defined by config.MAX_SR_LINES * 2 (potentially).
              Returns an empty list if data is insufficient or an error occurs.
    """
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    if df is None or not all(col in df.columns for col in required_cols):
         logger.error(f"Input DataFrame is missing required columns ({required_cols}).")
         return []

    # Use passed config object
    min_required_len = max(config.SWING_WINDOW, config.SMOOTHING_WINDOW, 2) # Need at least 2 for some calcs
    if len(df) < min_required_len:
        logger.warning(f"Insufficient data ({len(df)} rows) for support/resistance calculation (min required: {min_required_len} based on config).")
        return []

    # Step 1: Data Preparation specific for SR
    clean_data = clean_and_prepare_sr_data(df)
    if clean_data.empty:
        logger.warning("Data cleaning resulted in empty DataFrame. Cannot calculate S/R levels.")
        return []

    # Ensure we have enough data after cleaning
    if len(clean_data) < min_required_len:
        logger.warning(f"Insufficient data ({len(clean_data)} rows) after cleaning for S/R calculation (min required: {min_required_len} based on config).")
        return []

    # Step 2: Identify Key Price Points using passed config
    swing_points = find_swing_highs_lows(clean_data, config.SWING_WINDOW)
    zigzag_points = apply_zigzag_indicator(clean_data, config.ZIGZAG_THRESHOLD)

    # Step 3: Smoothing and Derivative Detection
    derivative_points = []
    # Use passed config for smoothing parameters
    if len(clean_data['price']) >= config.SMOOTHING_WINDOW and config.SMOOTHING_WINDOW > config.SMOOTHING_POLYORDER:
        # Ensure smoothing window is not larger than data length and is odd
        smoothing_window = min(config.SMOOTHING_WINDOW, len(clean_data['price']))
        if smoothing_window % 2 == 0: smoothing_window -= 1

        if smoothing_window > config.SMOOTHING_POLYORDER:
            smoothed_prices = smooth_prices(clean_data['price'].values,
                                            smoothing_window,
                                            config.SMOOTHING_POLYORDER)
            if smoothed_prices is not None and len(smoothed_prices) > 1:
                derivatives = calculate_derivative(smoothed_prices)
                if derivatives is not None and len(derivatives) > 1:
                    sign_change_indices = find_sign_changes(derivatives)
                    derivative_points = [(clean_data.iloc[i]['timestamp'], clean_data.iloc[i]['price'], 'derivative')
                                         for i in sign_change_indices if i < len(clean_data)]
                else: logger.warning("Derivative calculation failed or resulted in insufficient data.")
            else: logger.warning("Smoothing failed or resulted in insufficient data.")
        else: logger.warning(f"Adjusted smoothing window ({smoothing_window}) not > polyorder ({config.SMOOTHING_POLYORDER}). Skipping derivative points.")
    else: logger.warning("Insufficient data or invalid parameters (from config) for smoothing. Skipping derivative points.")


    # Combine all points from different methods
    all_points = swing_points + zigzag_points + derivative_points
    if not all_points:
        logger.warning("No key price points identified from any method.")
        return []
    logger.info(f"Identified {len(all_points)} total key points (Swing: {len(swing_points)}, ZigZag: {len(zigzag_points)}, Derivative: {len(derivative_points)}).")

    # Step 4: Clustering for level identification
    avg_price = clean_data['price'].mean()
    # Use passed config for clustering
    clusters = cluster_points(all_points,
                              config.CLUSTER_DISTANCE_THRESHOLD_PERCENT,
                              avg_price)
    if not clusters:
        logger.warning("Clustering did not produce any levels.")
        return []
    levels = calculate_cluster_centroids(clusters)
    if not levels:
        logger.warning("Cluster centroid calculation did not produce any levels.")
        return []
    logger.info(f"Calculated {len(levels)} raw levels from clusters.")


    # Step 5: Time Weighting
    # Use the last date from the cleaned (naive) dataframe
    current_date = clean_data['timestamp'].iloc[-1]

    # Use passed config for weighting
    weighted_levels = adjust_levels_with_weights(levels, all_points, config.DECAY_FACTOR, current_date)
    if not weighted_levels:
         logger.warning("Time weighting did not produce any levels. Returning unweighted levels.")
         weighted_levels = levels # Fallback to unweighted

    # Remove duplicates and sort
    # Round levels slightly to merge very close ones before making unique
    rounded_levels = [round(lvl, 2) for lvl in weighted_levels] # Adjust rounding precision if needed
    unique_levels = sorted(list(set(rounded_levels)))

    # --- Start: Separate, Calculate Strength, and Filter Top N Support & Resistance ---
    if not unique_levels:
        logger.warning("No unique levels identified after weighting and rounding.")
        return []

    logger.info(f"Identified {len(unique_levels)} unique potential S/R levels before strength filtering.")

    # Get the most recent price for classification
    if clean_data.empty or 'price' not in clean_data.columns:
         logger.error("Clean data is empty or missing 'price' column before final filtering.")
         return []
    current_price = clean_data['price'].iloc[-1]

    # Separate into potential support and resistance
    potential_support = [lvl for lvl in unique_levels if lvl <= current_price]
    potential_resistance = [lvl for lvl in unique_levels if lvl > current_price]
    logger.info(f"Separated into {len(potential_support)} potential support and {len(potential_resistance)} potential resistance levels.")

    # Calculate strength (number of confirmations) for each level
    # Define proximity threshold for counting confirmations
    # Ensure avg_price is available here. It was calculated around line 169.
    # If avg_price calculation was moved or removed, recalculate: avg_price = clean_data['price'].mean()
    # Use passed config for proximity threshold calculation
    proximity_threshold = (config.CLUSTER_DISTANCE_THRESHOLD_PERCENT * avg_price) / 2.0
    all_point_prices = [p[1] for p in all_points if isinstance(p[1], (int, float))]

    def calculate_strengths(levels_to_check):
        level_strengths = []
        for level in levels_to_check:
            strength = 0
            for point_price in all_point_prices:
                if abs(point_price - level) <= proximity_threshold:
                    strength += 1
            if strength > 0: # Only consider levels with at least one confirmation
                 level_strengths.append({'level': level, 'strength': strength})
        # Sort by strength (descending)
        return sorted(level_strengths, key=lambda x: x['strength'], reverse=True)

    # Calculate strengths and sort for each group
    support_strengths = calculate_strengths(potential_support)
    resistance_strengths = calculate_strengths(potential_resistance)

    # Select the top N for each category using passed config
    top_n = config.MAX_SR_LINES
    top_support = [item['level'] for item in support_strengths[:top_n]]
    top_resistance = [item['level'] for item in resistance_strengths[:top_n]]

    # Combine and sort final levels
    final_levels = sorted(list(set(top_support + top_resistance))) # Use set to avoid duplicates

    logger.info(f"Selected top {len(top_support)} support and {len(top_resistance)} resistance levels (max requested per category: {top_n}). Final unique levels: {len(final_levels)}")
    logger.debug(f"Final S/R levels: {[round(l, 2) for l in final_levels]}")

    return final_levels # Already sorted by price
    # --- End: Separate, Calculate Strength, and Filter Top N Support & Resistance ---