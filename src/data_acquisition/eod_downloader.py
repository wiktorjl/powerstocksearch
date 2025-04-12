import requests
import pandas as pd
from typing import Optional, List, Dict, Any
import logging
from io import StringIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EodApiClient:
    """
    A client for interacting with the EOD Historical Data API (eodhd.com).
    """
    BASE_URL = "https://eodhd.com/api"
# https://eodhd.com/api/eod/MCD.US?api_token=demo&fmt=json
    def __init__(self, api_key: str):
        """
        Initializes the EodApiClient.

        Args:
            api_key: Your EOD Historical Data API key.
        """
        if not api_key:
            raise ValueError("API key is required.")
        self.api_key = api_key
        self.session = requests.Session() # Use a session for potential performance benefits

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[requests.Response]:
        """
        Makes a GET request to a specified EODHD API endpoint.

        Args:
            endpoint: The API endpoint path (e.g., '/exchanges-list/').
            params: A dictionary of query parameters for the request.

        Returns:
            A requests.Response object. The caller is responsible for checking the
            status code (e.g., 200 for success, 404 for not found). Returns None
            only if a connection-level error occurs.
        """
        url = f"{self.BASE_URL}{endpoint}"
        request_params = params.copy() if params else {} # Use copy to avoid modifying original dict
        request_params['api_token'] = self.api_key
        if 'fmt' not in request_params: # Only set default fmt if not already specified
            request_params['fmt'] = 'json'

        try:
            # Use the modified request_params dict containing the potentially overridden fmt
            response = self.session.get(url, params=request_params, timeout=30) # Added timeout
            # Check status code before raising an error, allow 404 to be handled by caller
            if response.status_code == 200:
                 log_params = request_params.copy()
                 log_params['api_token'] = '***REDACTED***' # Avoid logging the key
                 logging.info(f"Successfully made request to {response.url}") # Log the final URL used by requests
                 return response
            elif response.status_code == 404:
                 logging.info(f"Request to {response.url} returned 404 Not Found. Passing response to caller.")
                 return response # Allow caller (e.g., get_splits_data) to handle 404
            else:
                 # For other errors (4xx, 5xx), raise the exception
                 logging.warning(f"Request to {response.url} failed with status {response.status_code}. Raising exception.")
                 response.raise_for_status() # This will now raise for non-200/404 errors
                 return None # Should not be reached if raise_for_status works
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {url} with params {params}: {e}")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred during API request to {url}: {e}")
            return None

    def get_exchange_tickers(self, exchange_code: str) -> List[str]:
        """
        Fetches the list of tickers for a given exchange code.

        Args:
            exchange_code: The code of the exchange (e.g., 'US', 'LSE', 'GSPC', 'INDX').

        Returns:
            A list of ticker symbols for the specified exchange, or an empty list on failure.
        """
        endpoint = f"/exchange-symbol-list/{exchange_code}"
        response = self._make_request(endpoint)

        if response and response.status_code == 200:
            try:
                data = response.json()
                # Assuming the JSON structure is a list of dicts, each with a 'Code' key
                tickers = [item['Code'] for item in data if 'Code' in item]
                logging.info(f"Fetched {len(tickers)} tickers for exchange {exchange_code}")
                return tickers
            except (ValueError, KeyError, TypeError) as e:
                logging.error(f"Failed to parse ticker list JSON for {exchange_code}: {e}")
                return []
        else:
            logging.warning(f"Failed to fetch tickers for exchange {exchange_code}. Status: {response.status_code if response else 'No Response'}")
            return []

    def get_eod_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetches End-of-Day (EOD) OHLCV data for a specific ticker and date range.

        Args:
            ticker: The ticker symbol (e.g., 'AAPL.US').
            start_date: The start date in 'YYYY-MM-DD' format.
            end_date: The end date in 'YYYY-MM-DD' format.

        Returns:
            A pandas DataFrame containing the EOD data, or None if the request fails
            or no data is returned.
        """
        endpoint = f"/eod/{ticker}"
        params = {
            'from': start_date,
            'to': end_date,
            'period': 'd', # Daily period
            'fmt': 'csv'   # Request CSV format for easier parsing into DataFrame
        }
        response = self._make_request(endpoint, params=params)

        if response and response.status_code == 200:
            if response.text.strip(): # Check if response body is not empty
                try:
                    # Use StringIO to read the CSV content directly into pandas
                    csv_data = StringIO(response.text)
                    df = pd.read_csv(csv_data)
                    if not df.empty:
                        logging.info(f"Fetched {len(df)} EOD records for {ticker} from {start_date} to {end_date}")
                        # Ensure 'Date' column is parsed correctly if needed later
                        # df['Date'] = pd.to_datetime(df['Date'])
                        return df
                    else:
                        logging.info(f"No EOD data returned for {ticker} from {start_date} to {end_date}")
                        return None
                except pd.errors.EmptyDataError:
                     logging.warning(f"Received empty CSV data for {ticker} from {start_date} to {end_date}.")
                     return None
                except Exception as e:
                    logging.error(f"Failed to parse EOD CSV data for {ticker}: {e}")
                    return None
            else:
                logging.info(f"Received empty response body for EOD data request for {ticker} from {start_date} to {end_date}")
                return None
        else:
            logging.warning(f"Failed to fetch EOD data for {ticker}. Status: {response.status_code if response else 'No Response'}")
            return None

    def get_splits_data(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches stock split data for a specific ticker.

        Args:
            ticker: The ticker symbol (e.g., 'AAPL.US').
            start_date: Optional start date in 'YYYY-MM-DD' format to filter splits.
            end_date: Optional end date in 'YYYY-MM-DD' format to filter splits.

        Returns:
            A list of dictionaries, where each dictionary represents a split
            (e.g., {'date': 'YYYY-MM-DD', 'split': 'New:Old'}), or None if the
            request fails or no splits are found.
        """
        endpoint = f"/splits/{ticker}"
        params = {'fmt': 'json'} # Ensure JSON format
        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date

        response = self._make_request(endpoint, params=params)

        if response and response.status_code == 200:
            try:
                splits_data = response.json()
                if isinstance(splits_data, list):
                    # Basic validation of structure (can be enhanced)
                    valid_splits = []
                    for item in splits_data:
                        if isinstance(item, dict) and 'date' in item and 'split' in item:
                            valid_splits.append(item)
                        else:
                             logging.warning(f"Skipping invalid split item for {ticker}: {item}")
                    if valid_splits:
                        logging.info(f"Fetched {len(valid_splits)} split records for {ticker}")
                        return valid_splits
                    else:
                        logging.info(f"No valid split data found for {ticker} in the response.")
                        return None # Return None if list is empty or contains no valid splits
                else:
                    logging.warning(f"Unexpected format for splits data for {ticker}. Expected list, got {type(splits_data)}")
                    return None
            except ValueError as e:
                logging.error(f"Failed to parse splits JSON data for {ticker}: {e}")
                return None
        else:
            status = response.status_code if response else 'No Response'
            # EODHD might return 404 if no splits exist, treat this as "no splits" not an error
            if response and response.status_code == 404:
                 logging.info(f"No split data found for {ticker} (API returned 404).")
                 return None # No splits found is not an error in this context
            else:
                logging.warning(f"Failed to fetch splits data for {ticker}. Status: {status}")
                return None
# Example Usage (Optional - for testing)
if __name__ == '__main__':
    from src import config # Adjusted import for direct execution context

    if config.EODHD_API_KEY:
        client = EodApiClient(config.EODHD_API_KEY)

        # Test fetching tickers
        # us_tickers = client.get_exchange_tickers('US')
        # print(f"First 10 US Tickers: {us_tickers[:10]}")

        # gspc_tickers = client.get_exchange_tickers('GSPC') # Example for an index
        # print(f"GSPC Tickers: {gspc_tickers}")

        # Test fetching EOD data
        # Note: Ensure the ticker format includes the exchange, e.g., AAPL.US
        # Adjust dates as needed
        # eod_df = client.get_eod_data('AAPL.US', '2023-01-01', '2023-01-10')
        # if eod_df is not None:
        #     print("\nEOD Data for AAPL.US:")
        #     print(eod_df.head())
        # else:
        #     print("\nFailed to fetch EOD data for AAPL.US")

        # Test fetching data for an index component (ensure correct ticker format)
        # Example: Fetch data for a component of GSPC (S&P 500)
        # Need to know a valid ticker within GSPC, e.g., 'AAPL.US' is also part of it
        # index_comp_df = client.get_eod_data('AAPL.US', '2024-01-01', '2024-01-05')
        # if index_comp_df is not None:
        #     print("\nEOD Data for Index Component (AAPL.US):")
        #     print(index_comp_df)
        # else:
        #     print("\nFailed to fetch EOD data for index component.")

        # Test fetching data for an index itself (ensure correct ticker format, often ends with .INDX or similar)
        # Example: Fetch data for S&P 500 index itself
        index_df = client.get_eod_data('GSPC.INDX', '2024-01-01', '2024-01-10')
        if index_df is not None:
            print("\nEOD Data for GSPC.INDX:")
            print(index_df)
        else:
             print("\nFailed to fetch EOD data for GSPC.INDX")

    else:
        print("Please set your EODHD_API_KEY in the .env file to run tests.")