# src/visualization/rrg.py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
import matplotlib.colors as mcolors
import logging

logger = logging.getLogger(__name__)

# Basic colors for plotting if needed, can be customized
BASIC_COLORS = [
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black",
    "gray", "cyan", "magenta", "lime", "teal", "indigo", "maroon", "navy", "olive", "silver",
    "aqua", "fuchsia", "lightblue", "darkgreen", "coral", "gold", "ivory", "khaki", "orchid", "plum", "salmon"
]

def _get_colors_for_stocks(stocks):
    """Generates a color mapping for a list of stocks."""
    return {stock: BASIC_COLORS[index % len(BASIC_COLORS)] for index, stock in enumerate(stocks)}

def _set_background_colors(ax, x_min, x_max, y_min, y_max):
    """Sets the colored quadrant backgrounds for the RRG plot based on dynamic limits."""
    lightred = mcolors.to_rgba('red', alpha=0.1)
    lightblue = mcolors.to_rgba('blue', alpha=0.1)
    lightgreen = mcolors.to_rgba('green', alpha=0.1)
    lightyellow = mcolors.to_rgba('yellow', alpha=0.2)

    center_x, center_y = 100, 100

    # Calculate widths and heights based on dynamic limits and center
    width_left = center_x - x_min
    width_right = x_max - center_x
    height_bottom = center_y - y_min
    height_top = y_max - center_y

    # Ensure widths/heights are non-negative
    width_left = max(0, width_left)
    width_right = max(0, width_right)
    height_bottom = max(0, height_bottom)
    height_top = max(0, height_top)

    # Improving (Top-Left Quadrant: x_min to center_x, center_y to y_max)
    ax.add_patch(patches.Rectangle((x_min, center_y), width_left, height_top, facecolor=lightblue, zorder=0))
    # Leading (Top-Right Quadrant: center_x to x_max, center_y to y_max)
    ax.add_patch(patches.Rectangle((center_x, center_y), width_right, height_top, facecolor=lightgreen, zorder=0))
    # Lagging (Bottom-Left Quadrant: x_min to center_x, y_min to center_y)
    ax.add_patch(patches.Rectangle((x_min, y_min), width_left, height_bottom, facecolor=lightred, zorder=0))
    # Weakening (Bottom-Right Quadrant: center_x to x_max, y_min to center_y)
    ax.add_patch(patches.Rectangle((center_x, y_min), width_right, height_bottom, facecolor=lightyellow, zorder=0))

    # Add text labels, positioned relative to the plot limits and center
    # Use a small offset from the edges/center for better placement
    x_offset = (x_max - x_min) * 0.02 # 2% offset
    y_offset = (y_max - y_min) * 0.02 # 2% offset

    ax.text(x_min + x_offset, y_max - y_offset, 'Improving', fontsize=12, color='black', ha='left', va='top')
    ax.text(center_x + x_offset, y_max - y_offset, 'Leading', fontsize=12, color='black', ha='left', va='top')
    ax.text(x_min + x_offset, y_min + y_offset, 'Lagging', fontsize=12, color='black', ha='left', va='bottom')
    ax.text(center_x + x_offset, y_min + y_offset, 'Weakening', fontsize=12, color='black', ha='left', va='bottom')

    # Add center lines
    ax.axhline(center_y, color='grey', linestyle='--', linewidth=0.8, zorder=0)
    ax.axvline(center_x, color='grey', linestyle='--', linewidth=0.8, zorder=0)


def _draw_rrg_paths(ax, data_points, colors):
    """Draws the smoothed paths and points for each stock."""
    logger.info(f"Drawing RRG paths for {len(data_points)} stocks.")
    for label, coords_list in data_points.items():
        if not coords_list:
            logger.warning(f"No data points found for stock: {label}. Skipping path drawing.")
            continue
        if len(coords_list) > 1:
            x_values, y_values = zip(*coords_list)

            # Ensure values are numeric, coerce errors to NaN and drop them
            x_values = pd.to_numeric(x_values, errors='coerce')
            y_values = pd.to_numeric(y_values, errors='coerce')
            valid_indices = ~np.isnan(x_values) & ~np.isnan(y_values)
            x_values = x_values[valid_indices]
            y_values = y_values[valid_indices]

            if len(x_values) < 2:
                 logger.warning(f"Not enough valid data points ({len(x_values)}) for stock: {label} after cleaning. Skipping path drawing.")
                 continue

            # Smoothing requires at least polyorder + 1 points, and window must be odd and <= len(data)
            min_points_for_savgol = 3 + 1 # polyorder=3
            if len(x_values) >= min_points_for_savgol:
                # Ensure window_length is odd and not larger than the number of points
                window_length = min(len(x_values), 9) # Use a smaller fixed window or adapt
                if window_length % 2 == 0:
                    window_length -= 1
                window_length = max(5, window_length) # Ensure window is at least 5 (polyorder 3 + 2)

                if window_length > len(x_values):
                     logger.warning(f"Window length ({window_length}) too large for data length ({len(x_values)}) for {label}. Skipping smoothing.")
                     x_smooth_values = x_values
                     y_smooth_values = y_values
                else:
                    try:
                        x_smooth_values = savgol_filter(x_values, window_length, 3)
                        y_smooth_values = savgol_filter(y_values, window_length, 3)
                    except Exception as e:
                        logger.error(f"Error during Savgol filter for {label}: {e}. Using raw data.")
                        x_smooth_values = x_values
                        y_smooth_values = y_values

                # Interpolation (optional, can use smoothed points directly)
                # Using linear interpolation (k=1) as cubic might overshoot with noisy data
                t = np.linspace(0, 1, len(x_smooth_values))
                t_smooth = np.linspace(0, 1, 100) # Fewer points for interpolation line
                try:
                    spline_x = make_interp_spline(t, x_smooth_values, k=1)
                    spline_y = make_interp_spline(t, y_smooth_values, k=1)
                    x_plot = spline_x(t_smooth)
                    y_plot = spline_y(t_smooth)
                    ax.plot(x_plot, y_plot, color=colors.get(label, 'black'), alpha=0.7, zorder=1)
                except Exception as e:
                    logger.error(f"Error during spline interpolation for {label}: {e}. Plotting smoothed points directly.")
                    ax.plot(x_smooth_values, y_smooth_values, color=colors.get(label, 'black'), alpha=0.7, zorder=1, linestyle='--') # Dashed line if interpolation fails

            else:
                 logger.warning(f"Not enough data points ({len(x_values)}) for Savgol smoothing for {label}. Using raw data.")
                 x_smooth_values = x_values
                 y_smooth_values = y_values
                 ax.plot(x_smooth_values, y_smooth_values, color=colors.get(label, 'black'), alpha=0.7, zorder=1, linestyle=':') # Dotted line for raw data plot


            # Plot markers - use the smoothed values
            # Use a colormap for the tail points (e.g., fading alpha or color gradient)
            num_points = len(x_smooth_values)
            alphas = np.linspace(0.3, 1.0, num_points) # Fade in alpha

            for i, coords in enumerate(zip(x_smooth_values, y_smooth_values)):
                point_color = colors.get(label, 'black')
                alpha = alphas[i]
                size = 30 # Default size

                if i == num_points - 1: # Last point
                    size = 80 # Make last point larger
                    marker = 'o' # Circle marker for last point
                    edge_color = 'black' # Add edge color to last point
                    ax.scatter(*coords, c=point_color, s=size, alpha=1.0, marker=marker, edgecolors=edge_color, linewidths=1, zorder=2)
                    # Annotate last point
                    ax.annotate(label, (coords[0] + 0.1, coords[1] + 0.1), fontsize=10, zorder=3)
                else: # Tail points
                    marker = '.' # Smaller marker for tail
                    ax.scatter(*coords, c=point_color, s=size, alpha=alpha, marker=marker, zorder=2)

        elif len(coords_list) == 1:
            # Only one point, plot it directly
            coords = coords_list[0]
            if pd.notna(coords[0]) and pd.notna(coords[1]):
                ax.scatter(*coords, c=colors.get(label, 'black'), s=80, marker='o', edgecolors='black', linewidths=1, zorder=2)
                ax.annotate(label, (coords[0] + 0.1, coords[1] + 0.1), fontsize=10, zorder=3)
            else:
                 logger.warning(f"Single data point for {label} contains NaN. Skipping scatter plot.")
        else:
             logger.warning(f"Unexpected number of points ({len(coords_list)}) for {label}. Skipping.")


def calculate_rrg_data(prices_df: pd.DataFrame, stocks: list, benchmark: str, momentum_period: int = 5):
    """
    Calculates Relative Strength Ratio (JdK RS-Ratio) and Momentum (JdK RS-Momentum)
    for a list of stocks against a benchmark.

    Args:
        prices_df (pd.DataFrame): DataFrame with dates as index and closing prices
                                  for stocks and benchmark as columns.
        stocks (list): List of stock symbols to analyze (must be columns in prices_df).
        benchmark (str): Benchmark symbol (must be a column in prices_df).
        momentum_period (int): Lookback period for momentum calculation.

    Returns:
        tuple: (rss_df, mom_df)
            - rss_df (pd.DataFrame): DataFrame of RS-Ratio values (scaled 0-100).
            - mom_df (pd.DataFrame): DataFrame of RS-Momentum values (scaled 0-100).
          Returns (None, None) if benchmark or any stock is missing or on error.
    """
    logger.info(f"Calculating RRG data for stocks: {stocks} against benchmark: {benchmark}")

    if benchmark not in prices_df.columns:
        logger.error(f"Benchmark '{benchmark}' not found in input DataFrame columns.")
        return None, None

    missing_stocks = [s for s in stocks if s not in prices_df.columns]
    if missing_stocks:
        logger.error(f"Stocks not found in input DataFrame columns: {missing_stocks}")
        return None, None

    # Use only the necessary columns (stocks + benchmark)
    relevant_symbols = stocks + [benchmark]
    stock_data = prices_df[relevant_symbols].copy()

    # Resample to weekly frequency, taking the last price of the week
    # This aligns with typical RRG usage.
    stock_data = stock_data.resample('W').last()
    stock_data = stock_data.dropna(how='all') # Drop weeks where all prices are NaN

    if stock_data.empty:
        logger.error("DataFrame is empty after weekly resampling.")
        return None, None

    rsss = {}
    moms = {}

    # --- RS-Ratio Calculation ---
    for column in stocks:
        # Relative Strength: price of stock / price of benchmark
        rs = stock_data[column] / stock_data[benchmark]

        # JdK RS-Ratio: Normalized ratio of RS to benchmark RS (100 * RS / Benchmark_RS)
        # Since benchmark RS relative to itself is 1, this simplifies to 100 * RS
        # However, the common RRG implementation normalizes this ratio itself.
        # Let's follow the typical approach: Ratio = 100 * (RS / SimpleMovingAverage(RS, period))
        # Using EMA instead of SMA as in the original code snippet provided
        ema10_rs = rs.ewm(span=10, adjust=False).mean()
        ema30_rs = rs.ewm(span=30, adjust=False).mean()

        # RS Ratio - Scaled 0-100 around the benchmark's average ratio (which is 100)
        # A common scaling method: 100 + ((ratio - mean_ratio) / std_dev_ratio) * 10
        # Simpler approach from original code: (ema_fast / ema_slow) * 100
        # Let's stick to the original code's method for consistency with the visual quadrants
        rs_ratio = (ema10_rs / ema30_rs) * 100
        rsss[column] = rs_ratio

    rss_df = pd.DataFrame(rsss)

    # --- RS-Momentum Calculation ---
    for column in stocks:
        # Momentum of the RS-Ratio (change over momentum_period)
        # Original code used diff * 100, let's keep that scaling for now
        mom = rss_df[column].diff(periods=momentum_period) * 100

        # Normalize momentum to 90-110 range (as per original code)
        # This normalization seems arbitrary and might distort interpretation.
        # A more standard approach is scaling around 100 based on benchmark momentum.
        # Let's implement the original normalization first.
        min_val = mom.min()
        max_val = mom.max()

        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            # Handle cases with NaNs or constant momentum
             logger.warning(f"Could not normalize momentum for {column} due to NaN or constant values. Setting to 100.")
             normalized_mom = pd.Series(100, index=mom.index) # Set to center (100)
        else:
            normalized_mom = 100 + ((mom - mom.mean()) / mom.std()) * 10 # More standard scaling around 100
            # Original scaling: 20 * (mom - min_val) / (max_val - min_val) + 90

        moms[column] = normalized_mom # Use standard scaling

    mom_df = pd.DataFrame(moms)

    # Drop initial NaNs created by EMA and diff calculations
    min_periods_ema = 30 # Longest EMA window
    min_periods_mom = momentum_period
    total_min_periods = min_periods_ema + min_periods_mom

    rss_df = rss_df.iloc[total_min_periods:]
    mom_df = mom_df.iloc[total_min_periods:]

    # Align indices after potential NaN removal from normalization
    common_index = rss_df.index.intersection(mom_df.index)
    rss_df = rss_df.loc[common_index]
    mom_df = mom_df.loc[common_index]


    if rss_df.empty or mom_df.empty:
        logger.error("RRG calculation resulted in empty DataFrames after processing.")
        return None, None

    logger.info(f"Finished RRG data calculation. Ratio shape: {rss_df.shape}, Momentum shape: {mom_df.shape}")
    return rss_df, mom_df


def create_rrg_plot(data_points, start_date, end_date, filename='rrg_plot.png'):
    """
    Creates and saves the RRG plot visualization.

    Args:
        data_points (dict): Dictionary where keys are stock symbols and values are
                            lists of (rs_ratio, rs_momentum) tuples for the tail.
        start_date (datetime): The start date of the data used for the plot tail.
        end_date (datetime): The end date of the data used for the plot tail.
        filename (str): The path/filename to save the plot image.

    Returns:
        bool: True if plot creation was successful, False otherwise.
    """
    logger.info(f"Creating RRG plot. Data points for {len(data_points)} stocks. Filename: {filename}")
    if not data_points:
        logger.error("Cannot create RRG plot: No data points provided.")
        return False

    try:
        fig, ax = plt.subplots(figsize=(12, 12)) # Adjusted size

        # Determine plot limits dynamically based on data, with padding
        all_x = [p[0] for points in data_points.values() for p in points if pd.notna(p[0])]
        all_y = [p[1] for points in data_points.values() for p in points if pd.notna(p[1])]

        if not all_x or not all_y:
             logger.warning("No valid numeric data points found to determine plot limits. Using defaults.")
             x_min, x_max = 90, 110
             y_min, y_max = 90, 110
        else:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            padding_x = (x_max - x_min) * 0.10 # 10% padding
            padding_y = (y_max - y_min) * 0.10 # 10% padding

            # Ensure center (100,100) is roughly included
            x_min = min(95, x_min - padding_x)
            x_max = max(105, x_max + padding_x)
            y_min = min(95, y_min - padding_y)
            y_max = max(105, y_max + padding_y)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # ax.set_xticks(np.arange(90, 110.1, 2)) # Adjust ticks based on limits
        # ax.set_yticks(np.arange(90, 110.1, 2)) # Adjust ticks based on limits
        ax.grid(True, linestyle='--', alpha=0.6)


        ax.set_xlabel('JdK RS-Ratio (Relative Strength vs Benchmark)', fontsize=12)
        ax.set_ylabel('JdK RS-Momentum (Momentum of RS-Ratio)', fontsize=12)
        ax.set_title('Relative Rotation Graph (RRG)', fontsize=16, loc='center', pad=20)

        # Format dates for subtitle
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else 'N/A'
        end_date_str = end_date.strftime('%Y-%m-%d') if end_date else 'N/A'
        ax.text(0.98, 1.01, f'Tail Period: {start_date_str} to {end_date_str}',
                transform=ax.transAxes, fontsize=10, color='gray', ha='right')


        _set_background_colors(ax, x_min, x_max, y_min, y_max) # Pass dynamic limits

        # Get colors for the stocks being plotted
        stock_symbols = list(data_points.keys())
        colors = _get_colors_for_stocks(stock_symbols)

        _draw_rrg_paths(ax, data_points, colors) # Draw the paths and points

        plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to prevent title overlap
        plt.savefig(filename, dpi=150) # Use slightly lower DPI for web
        plt.close(fig) # Close the figure to free memory
        logger.info(f"RRG plot saved successfully to {filename}")
        return True
    except Exception as e:
        logger.exception(f"Error creating RRG plot: {e}")
        # Ensure figure is closed even if error occurs during plotting/saving
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        return False


def generate_rrg_plot(prices_df: pd.DataFrame, stocks: list, benchmark: str, output_filename: str, tail_length: int = 10):
    """
    Orchestrates the calculation and plotting of the RRG.

    Args:
        prices_df (pd.DataFrame): DataFrame with dates as index and closing prices
                                  for stocks and benchmark as columns.
        stocks (list): List of stock symbols to analyze.
        benchmark (str): Benchmark symbol.
        output_filename (str): Path to save the generated plot.
        tail_length (int): Number of trailing data points to plot for each stock.

    Returns:
        str: The output filename if successful, None otherwise.
    """
    logger.info(f"Generating RRG plot: Stocks={stocks}, Benchmark={benchmark}, File={output_filename}, Tail={tail_length}")

    if prices_df is None or prices_df.empty:
        logger.error("Input prices DataFrame is None or empty. Cannot generate RRG plot.")
        return None

    # 1. Calculate RRG Data (Ratio and Momentum)
    rss_df, mom_df = calculate_rrg_data(prices_df, stocks, benchmark)

    if rss_df is None or mom_df is None or rss_df.empty or mom_df.empty:
        logger.error("Failed to calculate RRG data (Ratio or Momentum).")
        return None

    # 2. Prepare data points for plotting (the tail)
    data_points_for_plot = {}
    try:
        for stock in stocks:
            if stock in rss_df.columns and stock in mom_df.columns:
                # Combine ratio and momentum, take the tail
                stock_points = list(zip(rss_df[stock], mom_df[stock]))
                # Ensure tail_length doesn't exceed available data
                actual_tail_length = min(tail_length, len(stock_points))
                if actual_tail_length > 0:
                     data_points_for_plot[stock] = stock_points[-actual_tail_length:]
                else:
                     logger.warning(f"No data points available for stock {stock} after calculation.")
                     data_points_for_plot[stock] = [] # Add empty list if no points
            else:
                logger.warning(f"Stock {stock} missing from RRG calculation results. Skipping.")
                data_points_for_plot[stock] = [] # Add empty list if missing

        if not any(data_points_for_plot.values()): # Check if all lists are empty
             logger.error("No valid data points found for any stock after RRG calculation and tail selection.")
             return None

        # Determine date range from the index of the data used for the tail
        # Use the index of the momentum df as it's likely the shortest after diff()
        if not mom_df.empty:
             # Ensure actual_tail_length is positive before slicing
             if actual_tail_length > 0:
                 tail_indices = mom_df.index[-actual_tail_length:]
                 start_date = tail_indices.min().to_pydatetime() if not tail_indices.empty else None
                 end_date = tail_indices.max().to_pydatetime() if not tail_indices.empty else None
             else:
                 start_date, end_date = None, None # No tail data
        else:
             start_date, end_date = None, None


    except Exception as e:
        logger.exception(f"Error preparing data points for plotting: {e}")
        return None

    # 3. Create the plot
    success = create_rrg_plot(data_points_for_plot, start_date, end_date, filename=output_filename)

    if success:
        return output_filename
    else:
        logger.error("Failed to create the RRG plot image.")
        return None
