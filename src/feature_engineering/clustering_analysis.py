# clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

def cluster_points(points: list, distance_threshold_percent: float, avg_price: float):
    """
    Cluster identified price points using Agglomerative Clustering.
    The distance threshold is calculated based on a percentage of the average price.

    Args:
        points (list): A list of tuples, where each tuple is (timestamp, price, type).
                       'timestamp' should be pandas Timestamp or similar datetime object.
                       'price' is the numeric price value.
                       'type' is a string indicating the source ('swing', 'zigzag', 'derivative').
        distance_threshold_percent (float): The clustering distance threshold as a
                                            percentage of the average price (e.g., 0.01 for 1%).
        avg_price (float): The average price of the stock over the period, used to
                           calculate the absolute distance threshold.

    Returns:
        dict: A dictionary mapping cluster labels to lists of points belonging to that cluster.
              Returns an empty dictionary if no points are provided or clustering fails.
    """
    if not points:
        print("Warning: No points provided for clustering.")
        return {}

    # Extract price values and ensure they are numeric
    try:
        prices = np.array([float(p[1]) for p in points]).reshape(-1, 1)
    except (ValueError, TypeError) as e:
        print(f"Error extracting numeric prices from points: {e}. Points: {points[:5]}") # Log first few points for debugging
        return {}

    if prices.size == 0:
         print("Warning: No valid numeric prices extracted for clustering.")
         return {}

    # Calculate the absolute distance threshold
    if avg_price <= 0:
        print("Warning: Average price is non-positive. Cannot calculate absolute threshold.")
        # Fallback: use a small absolute value or handle as error
        # For now, return empty dict
        return {}
    abs_threshold = distance_threshold_percent * avg_price

    if abs_threshold <= 0:
        print(f"Warning: Calculated absolute distance threshold ({abs_threshold}) is non-positive. Clustering might behave unexpectedly.")
        # Decide on handling: proceed with caution or return {}
        # Let's proceed for now, AgglomerativeClustering might handle it.

    try:
        # n_clusters=None and distance_threshold are used together
        # linkage='ward' requires euclidean affinity, which is default
        clustering = AgglomerativeClustering(n_clusters=None,
                                           distance_threshold=abs_threshold,
                                           linkage='ward') # 'ward' minimizes variance within clusters
        labels = clustering.fit_predict(prices)
    except Exception as e:
        print(f"Error during Agglomerative Clustering: {e}")
        return {}

    # Group points by cluster label
    clusters = defaultdict(list)
    for label, point in zip(labels, points):
        clusters[label].append(point)

    print(f"Clustering resulted in {len(clusters)} levels from {len(points)} points with threshold {abs_threshold:.2f}.")
    return dict(clusters)


def calculate_cluster_centroids(clusters: dict):
    """
    Calculate the representative price level for each cluster.
    Uses the median price of the points within each cluster.

    Args:
        clusters (dict): A dictionary mapping cluster labels to lists of points.

    Returns:
        list: A list of calculated support/resistance levels (median prices).
              Returns an empty list if the input dictionary is empty.
    """
    if not clusters:
        return []

    levels = []
    for label, cluster_points in clusters.items():
        if not cluster_points:
            print(f"Warning: Cluster {label} is empty.")
            continue

        try:
            # Extract prices, ensuring they are float
            prices = [float(p[1]) for p in cluster_points]
            if not prices:
                 print(f"Warning: No valid prices in cluster {label}.")
                 continue
            # Use median as it's less sensitive to outliers within a cluster
            level = np.median(prices)
            levels.append(level)
        except (ValueError, TypeError) as e:
            print(f"Error calculating median for cluster {label}: {e}. Points: {cluster_points[:5]}")
            continue # Skip this cluster

    return sorted(levels) # Return sorted levels