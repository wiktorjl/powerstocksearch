# smoothing.py
import numpy as np
from scipy.signal import savgol_filter

def smooth_prices(prices: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter to smooth price data.

    Args:
        prices (np.ndarray): Array of prices to smooth.
        window (int): The length of the filter window (must be odd and > polyorder).
        polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
        np.ndarray: The smoothed price array. Returns original if smoothing fails.
    """
    # Basic validation
    if not isinstance(prices, np.ndarray) or prices.ndim != 1:
        raise ValueError("Input 'prices' must be a 1D numpy array.")
    if not isinstance(window, int) or not isinstance(polyorder, int):
        raise ValueError("'window' and 'polyorder' must be integers.")
    if len(prices) < window:
        print(f"Warning: Data length ({len(prices)}) is less than smoothing window ({window}). Cannot smooth.")
        return prices # Return original if not enough data
    if window <= polyorder:
        print(f"Warning: Smoothing window ({window}) must be greater than polyorder ({polyorder}). Cannot smooth.")
        return prices # Return original if window is too small
    if window % 2 == 0:
        print(f"Warning: Smoothing window ({window}) must be odd. Adjusting to {window - 1}.")
        window -= 1
        if window <= polyorder:
             print(f"Warning: Adjusted window ({window}) is not greater than polyorder ({polyorder}). Cannot smooth.")
             return prices

    try:
        smoothed = savgol_filter(prices, window_length=window, polyorder=polyorder)
        return smoothed
    except Exception as e:
        print(f"Error during Savitzky-Golay filtering: {e}. Returning original prices.")
        return prices


def calculate_derivative(smoothed_prices: np.ndarray) -> np.ndarray:
    """
    Calculate the discrete derivative (first order difference) of the smoothed prices.

    Args:
        smoothed_prices (np.ndarray): Array of smoothed prices.

    Returns:
        np.ndarray: Array containing the derivative values. Returns empty array if input is invalid.
    """
    if not isinstance(smoothed_prices, np.ndarray) or smoothed_prices.ndim != 1:
        print("Warning: Invalid input for derivative calculation. Returning empty array.")
        return np.array([])
    if len(smoothed_prices) < 2:
        print("Warning: Need at least two points to calculate derivative. Returning empty array.")
        return np.array([])

    try:
        # Using np.gradient for potentially better handling of boundaries
        derivative = np.gradient(smoothed_prices)
        return derivative
    except Exception as e:
        print(f"Error calculating derivative: {e}. Returning empty array.")
        return np.array([])


def find_sign_changes(derivatives: np.ndarray):
    """
    Identify indices where the derivative changes sign (crosses zero).
    These indices represent potential local extrema (peaks and troughs).

    Args:
        derivatives (np.ndarray): Array of derivative values.

    Returns:
        np.ndarray: Array of indices where sign changes occur. Returns empty array if input is invalid.
    """
    if not isinstance(derivatives, np.ndarray) or derivatives.ndim != 1:
         print("Warning: Invalid input for finding sign changes. Returning empty array.")
         return np.array([], dtype=int)
    if len(derivatives) < 2:
        print("Warning: Need at least two derivative points to find sign changes. Returning empty array.")
        return np.array([], dtype=int)

    try:
        # Find where the sign of the difference between consecutive elements changes
        # np.sign returns -1, 0, 1. We look for transitions between non-zero signs.
        signs = np.sign(derivatives)
        # Find indices where sign is different from the previous non-zero sign
        sign_changes = np.where(np.diff(signs) != 0)[0]

        # Filter out changes involving zero, unless it's a peak/trough (e.g., +1 -> 0 -> -1)
        # A simple approach is to take the index *after* the change.
        # Example: [1, 1, 0, -1, -1] -> diff -> [0, -1, -1, 0]. np.where -> [1, 2]. Indices are 1, 2.
        # We want the index where the derivative is zero or crosses zero, so index+1 is appropriate.
        valid_indices = sign_changes + 1

        # Ensure indices are within the bounds of the original price array length
        # (assuming derivatives array is same length or length-1 of prices)
        # No direct length check here, rely on caller context.

        return valid_indices
    except Exception as e:
        print(f"Error finding sign changes: {e}. Returning empty array.")
        return np.array([], dtype=int)