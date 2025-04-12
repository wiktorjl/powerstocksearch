# indicators.py
import numpy as np
import pandas as pd

def find_swing_highs_lows(df: pd.DataFrame, window: int):
    """
    Identify swing highs (local maxima) and swing lows (local minima)
    using a sliding window on the 'price' column.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'price' columns.
        window (int): The number of periods for the sliding window.

    Returns:
        list: A list of tuples: (timestamp, price, 'resistance'/'support').
              Returns an empty list if the DataFrame is too small for the window.
    """
    swing_points = []
    if len(df) < window:
        print(f"Warning: Data length ({len(df)}) is less than swing window ({window}). Cannot find swing points.")
        return swing_points

    half_window = window // 2
    prices = df['price'].values
    dates = df['timestamp'].values # Use the timestamp column

    # Ensure indices are within bounds
    start_index = half_window
    end_index = len(df) - half_window

    for i in range(start_index, end_index):
        window_data = prices[i - half_window : i + half_window + 1]
        current_price = prices[i]
        current_date = dates[i]

        # Check if current price is a local maximum or minimum within the window
        if current_price == np.max(window_data):
            swing_points.append((current_date, current_price, 'resistance'))
        elif current_price == np.min(window_data):
            swing_points.append((current_date, current_price, 'support'))

    return swing_points

def apply_zigzag_indicator(df: pd.DataFrame, threshold: float):
    """
    Implements a simple zig-zag indicator based on percentage change
    from the last pivot point using the 'price' column.

    Args:
        df (pd.DataFrame): DataFrame with 'timestamp' and 'price' columns.
        threshold (float): The minimum percentage change required to form a new pivot.

    Returns:
        list: A list of pivot tuples: (timestamp, price, 'pivot').
              Returns an empty list if the DataFrame is empty.
    """
    zigzag_points = []
    if df.empty:
        return zigzag_points

    prices = df['price'].values
    dates = df['timestamp'].values # Use the timestamp column

    if len(prices) < 2:
        return zigzag_points # Need at least two points

    # Initialize with the first point
    last_pivot_price = prices[0]
    last_pivot_date = dates[0]
    trend = None  # Initial trend direction ('up' or 'down')
    pivot_points = [(last_pivot_date, last_pivot_price, 'pivot')] # Start with the first point as a pivot

    potential_pivot_price = last_pivot_price
    potential_pivot_date = last_pivot_date

    for i in range(1, len(prices)):
        current_price = prices[i]
        current_date = dates[i]
        price_change = (current_price - last_pivot_price) / last_pivot_price if last_pivot_price != 0 else 0

        if trend is None: # Determine initial trend
            if abs(price_change) >= threshold:
                trend = 'up' if price_change > 0 else 'down'
                # The point causing the trend change is the new pivot
                potential_pivot_price = current_price
                potential_pivot_date = current_date
                pivot_points.append((potential_pivot_date, potential_pivot_price, 'pivot'))
                last_pivot_price = potential_pivot_price
                last_pivot_date = potential_pivot_date
        elif trend == 'up':
            if current_price > potential_pivot_price: # Price continues upward, update potential pivot
                potential_pivot_price = current_price
                potential_pivot_date = current_date
            # Check for reversal: price drops below threshold from the potential high pivot
            elif (potential_pivot_price - current_price) / potential_pivot_price >= threshold:
                # Confirm the last potential high as a pivot
                if potential_pivot_date != last_pivot_date: # Avoid duplicate pivots
                     pivot_points.append((potential_pivot_date, potential_pivot_price, 'pivot'))
                     last_pivot_price = potential_pivot_price
                     last_pivot_date = potential_pivot_date

                # Start new downward trend, current point is the new potential low pivot
                trend = 'down'
                potential_pivot_price = current_price
                potential_pivot_date = current_date
        elif trend == 'down':
            if current_price < potential_pivot_price: # Price continues downward, update potential pivot
                potential_pivot_price = current_price
                potential_pivot_date = current_date
            # Check for reversal: price rises above threshold from the potential low pivot
            elif (current_price - potential_pivot_price) / potential_pivot_price >= threshold:
                 # Confirm the last potential low as a pivot
                if potential_pivot_date != last_pivot_date: # Avoid duplicate pivots
                    pivot_points.append((potential_pivot_date, potential_pivot_price, 'pivot'))
                    last_pivot_price = potential_pivot_price
                    last_pivot_date = potential_pivot_date

                # Start new upward trend, current point is the new potential high pivot
                trend = 'up'
                potential_pivot_price = current_price
                potential_pivot_date = current_date

    # Add the last potential pivot if it's different from the last confirmed pivot
    if potential_pivot_date != last_pivot_date:
         pivot_points.append((potential_pivot_date, potential_pivot_price, 'pivot'))

    # Return unique pivots (based on date and price)
    unique_pivots = []
    seen = set()
    for p in pivot_points:
        key = (p[0], p[1])
        if key not in seen:
            unique_pivots.append(p)
            seen.add(key)

    return unique_pivots