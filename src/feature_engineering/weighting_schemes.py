# weighting.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def adjust_levels_with_weights(levels: list, points: list, decay_factor: float, current_date):
    """
    Adjust cluster centroids (levels) based on time-weighted average of nearby points.
    Points closer in time to the 'current_date' receive higher weights.

    Args:
        levels (list): A list of preliminary support/resistance levels (cluster centroids).
        points (list): The list of all identified key points (timestamp, price, type).
                       Timestamps should be timezone-naive datetime objects or pandas Timestamps.
        decay_factor (float): The exponential decay factor (lambda) for time weighting.
                              Higher values give more weight to recent points.
        current_date (datetime or pd.Timestamp): The reference date (usually the last date
                                                 in the dataset) for calculating time elapsed.
                                                 Should be timezone-naive.

    Returns:
        list: A list of time-weighted support/resistance levels.
    """
    if not levels:
        return []
    if not points:
        print("Warning: No points provided for weighting. Returning original levels.")
        return levels

    # Ensure current_date is a timezone-naive pandas Timestamp for reliable comparison
    if isinstance(current_date, datetime):
        current_date = pd.Timestamp(current_date).tz_localize(None)
    elif isinstance(current_date, pd.Timestamp) and current_date.tz is not None:
        current_date = current_date.tz_localize(None)
    elif not isinstance(current_date, pd.Timestamp):
         print(f"Warning: Invalid current_date type ({type(current_date)}). Cannot perform weighting.")
         return levels # Or raise error

    weighted_levels = []
    for level in levels:
        # Find points close to the current level (e.g., within 1% range)
        # This threshold helps associate points with a specific level before weighting
        nearby_points = []
        for p in points:
            try:
                price = float(p[1])
                if abs(price - level) <= 0.01 * level: # Check if point price is within 1% of the level
                    # Ensure point timestamp is timezone-naive pandas Timestamp
                    point_date = p[0]
                    if isinstance(point_date, datetime):
                        point_date = pd.Timestamp(point_date).tz_localize(None)
                    elif isinstance(point_date, pd.Timestamp) and point_date.tz is not None:
                        point_date = point_date.tz_localize(None)
                    elif not isinstance(point_date, pd.Timestamp):
                         print(f"Warning: Skipping point with invalid date type ({type(point_date)}) during weighting.")
                         continue

                    # Check if point_date is valid and not NaT
                    if pd.isna(point_date):
                        print(f"Warning: Skipping point with NaT date during weighting.")
                        continue

                    nearby_points.append({'date': point_date, 'price': price})

            except (ValueError, TypeError, IndexError) as e:
                print(f"Warning: Skipping invalid point during weighting check: {p}. Error: {e}")
                continue

        if not nearby_points:
            # If no points are near this level, keep the original level
            weighted_levels.append(level)
            continue

        # Calculate weights and weighted average
        total_weight = 0.0
        weighted_price_sum = 0.0
        valid_points_count = 0

        for p_data in nearby_points:
            point_date = p_data['date']
            price = p_data['price']

            # Calculate time elapsed in days
            time_elapsed_delta = current_date - point_date
            # Handle potential negative timedelta if point date is somehow after current_date
            time_elapsed_days = max(0, time_elapsed_delta.days)

            # Calculate weight using exponential decay
            # Add a small epsilon to avoid issues with log(0) or division by zero if needed elsewhere,
            # but exp(-lambda * t) handles t=0 correctly (weight=1).
            weight = np.exp(-decay_factor * time_elapsed_days)

            if weight > 0: # Ensure weight is valid
                weighted_price_sum += price * weight
                total_weight += weight
                valid_points_count += 1

        # Calculate the weighted average price for the level
        if total_weight > 0:
            weighted_average_price = weighted_price_sum / total_weight
            weighted_levels.append(weighted_average_price)
            # print(f"Level {level:.2f} adjusted to {weighted_average_price:.2f} using {valid_points_count} points.")
        else:
            # Fallback: if total weight is zero (e.g., all points too old or decay factor too high),
            # keep the original level or use simple average of nearby points.
            print(f"Warning: Total weight is zero for level {level:.2f}. Keeping original level.")
            weighted_levels.append(level)

    return sorted(weighted_levels) # Return sorted weighted levels