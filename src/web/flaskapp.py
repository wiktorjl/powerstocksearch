import logging
import os
# Consolidate Flask imports
from flask import Flask, jsonify, request, render_template, flash, session, url_for
import requests
import math
from datetime import datetime, timedelta
import pandas as pd
from src.database.data_provider import (
    list_symbols_db,
    fetch_symbol_details_db,
    fetch_ohlc_data_db,
    fetch_ohlc_data_for_symbols_db, # Added import for multi-symbol fetch
    fetch_latest_indicators_db,
    get_unique_sectors,
    scan_stocks
)
# Import the chart generation function
from src.visualization.ohlc_plotter import generate_and_save_chart
# Import S/R calculation and config
from src.visualization.rrg import generate_rrg_plot # Import RRG plotter
from src.feature_engineering.support_resistance import identify_support_resistance
from src.config import AlgorithmConfig
from src.services.openai_service import get_company_summary, get_economic_variables, get_economic_analysis # Import the new services
from src.analysis.economic_influence_analyzer import EconomicInfluenceAnalyzer # Import the analyzer
from src.screening.reversal_screener import DEFAULT_CONFIG # Screener function no longer called directly here
# Import the new function to fetch results from the DB
from src.database.data_provider import fetch_reversal_scan_results_db

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../../templates', static_folder='../../static')

# TODO: Set a proper secret key, preferably from config/environment
app.secret_key = 'dev-secret-key' # Add a temporary key for flash


# --- RRG Plot Configuration ---
RRG_STOCKS = ['GOOGL', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'META'] # Example stocks for RRG
RRG_BENCHMARK = 'GE'
RRG_PLOT_FILENAME = 'rrg_plot.png'
RRG_PLOT_DIR = os.path.join(app.static_folder, 'plots') # Use app's static folder
RRG_PLOT_PATH = os.path.join(RRG_PLOT_DIR, RRG_PLOT_FILENAME)
# RRG_PLOT_URL removed from global scope


# --- Custom Jinja Filters ---
def format_currency(value):
    """Formats a value as currency (e.g., $ 1,234.56)."""
    if value is None:
        return '' # Return empty string for None
    try:
        # Attempt conversion to float first for flexibility
        num = float(value)
        # Format with dollar sign, comma separators, and 2 decimal places
        # Using locale might be better for internationalization, but this is simpler for now.
        return '${:,.2f}'.format(num)
    except (ValueError, TypeError):
        # Return original value as string if conversion fails
        return str(value)

def format_large_number(value):
    """Formats a large number with thousands separators (e.g., 1,234,567)."""
    if value is None:
        return '' # Return empty string for None
    try:
        # Attempt conversion to float first (to handle potential float strings), then int
        num = int(float(value))
        # Format with comma separators
        return '{:,}'.format(num)
    except (ValueError, TypeError):
        # Return original value as string if conversion fails
        return str(value)

def format_date(value, fmt='%Y-%m-%d'):
    """Formats a date string or datetime object."""
    if value is None:
        return 'N/A' # Return N/A for None
    try:
        # If it's already a datetime object
        if isinstance(value, datetime):
            return value.strftime(fmt)
        # If it's a string, try parsing it using pandas for robustness
        elif isinstance(value, str):
             # Handle empty strings explicitly
             if not value.strip():
                 return 'N/A'
             dt_obj = pd.to_datetime(value).to_pydatetime()
             return dt_obj.strftime(fmt)
        else:
            # Handle other types if necessary, or return original as string
            return str(value)
    except (ValueError, TypeError, pd.errors.ParserError):
        # Return original value as string if parsing/formatting fails
        logger.warning(f"Could not format date value: {value}", exc_info=False) # Log warning
        return str(value) # Return original string representation

def format_indicator(indicator_dict):
    """
    Formats an indicator value. Assumes currency unless description contains
    keywords indicating otherwise (e.g., ratio, index, %, score, change).
    """
    if not indicator_dict or 'value' not in indicator_dict or indicator_dict['value'] is None:
        return 'N/A'

    value = indicator_dict['value']
    description = indicator_dict.get('description', '').lower() # Get description, default to empty, lowercase

    # Keywords suggesting NON-currency values
    non_currency_keywords = ['ratio', 'index', '%', 'score', 'change', 'rsi', 'roc', 'momentum', 'beta'] # Add more as needed

    try:
        num = float(value) # Ensure value is numeric

        # Check if description contains any NON-currency keywords
        is_non_currency = any(keyword in description for keyword in non_currency_keywords)

        if is_non_currency:
            # Format as standard number (e.g., ratios, points)
            # Format to 2-4 decimal places depending on magnitude, add commas
            if abs(num) < 0.01 and num != 0:
                 return '{:,.4f}'.format(num) # More precision for small numbers
            else:
                 return '{:,.2f}'.format(num) # Standard 2 decimal places
        else:
            # Assume currency formatting for anything else (SMA, EMA, Price, Value, Cap etc.)
            return format_currency(num)

    except (ValueError, TypeError):
        # Return original value as string if conversion or formatting fails
        return str(value)

# --- Register Filters with Jinja Environment ---
app.jinja_env.filters['currency'] = format_currency
app.jinja_env.filters['large_number'] = format_large_number
app.jinja_env.filters['date'] = format_date
app.jinja_env.filters['indicator'] = format_indicator


@app.route('/')
def hello_world():
    """ Basic welcome route. """
    return 'Welcome to the Power Stock Search API!'

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """
    Lists all available stock symbols.
    Accepts an optional 'prefix' query parameter to filter symbols.
    e.g., /symbols?prefix=AA
    """
    prefix = request.args.get('prefix')
    logger.info(f"Received request for symbols" + (f" with prefix '{prefix}'" if prefix else ""))
    try:
        symbols = list_symbols_db(prefix=prefix)
        if symbols is None:
            logger.error("Failed to fetch symbols from database.")
            return jsonify({"error": "Database error fetching symbols"}), 500
        elif not symbols:
             # Return empty list if no symbols match (even without prefix)
            logger.info("No symbols found" + (f" matching prefix '{prefix}'" if prefix else ""))
            return jsonify([])
        else:
            logger.info(f"Returning {len(symbols)} symbols.")
            return jsonify(symbols)
    except Exception as e:
        logger.exception(f"Unexpected error fetching symbols: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/symbols/<string:symbol>', methods=['GET'])
def get_symbol_details(symbol):
    """
    Returns details for a specific stock symbol, including recent OHLC data and a chart URL.
    Accepts an optional 'theme' query parameter ('light' or 'dark').
    e.g., /symbols/AAPL?theme=dark
    """
    logger.info(f"Received request for details for symbol: {symbol}")
    symbol_upper = symbol.upper() # Ensure symbol is uppercase
    # Prioritize theme from session, then request args, then default
    session_theme = session.get('theme')
    request_theme = request.args.get('theme')
    theme = session_theme or request_theme or 'light'
    logger.info(f"[/symbols/{symbol_upper}] Theme check: Session='{session_theme}', Request='{request_theme}', Final='{theme}'")

    try:
        details = fetch_symbol_details_db(symbol_upper)

        if details:
            logger.info(f"Found basic details for {symbol_upper}. Fetching OHLC and chart.")
            # --- Fetch Indicators ---
            indicators_data = None
            if 'symbol_id' in details:
                symbol_id = details['symbol_id']
                logger.info(f"Fetching latest indicators for symbol_id: {symbol_id}")
                try:
                    indicators_data = fetch_latest_indicators_db(symbol_id)
                    if indicators_data:
                        logger.info(f"Successfully fetched {len(indicators_data)} indicators for symbol_id: {symbol_id}")
                        details['indicators'] = indicators_data # Add indicators to the response
                    else:
                        logger.warning(f"Could not fetch indicators for symbol_id: {symbol_id}")
                        details['indicators'] = None # Explicitly set to None if not found
                except Exception as ind_err:
                    logger.error(f"Error fetching indicators for symbol_id {symbol_id}: {ind_err}")
                    details['indicators'] = None # Set to None on error
            else:
                logger.warning(f"Could not fetch indicators because symbol_id is missing in company_data for {symbol_upper}")
                details['indicators'] = None


            # --- Fetch OHLC Data (last 90 days) ---
            ohlc_data_list = None
            ohlc_df = None # Initialize ohlc_df
            try:
                end_date = datetime.now()
                # Fetch 1 year of data for S/R calculation
                start_date_sr = end_date - timedelta(days=365)
                ohlc_df = fetch_ohlc_data_db(
                    symbol_upper,
                    start_date=start_date_sr.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                if ohlc_df is not None and not ohlc_df.empty:
                    # Convert timestamp to string for JSON compatibility
                    ohlc_df['timestamp'] = ohlc_df['timestamp'].astype(str)
                    ohlc_data_list = ohlc_df.to_dict(orient='records')
                    logger.info(f"Successfully fetched {len(ohlc_data_list)} OHLC data points for {symbol_upper}")

                    # Extract latest data point and add to details
                    if ohlc_data_list:
                        latest_data = ohlc_data_list[-1] # Get the last item
                        details['latest_timestamp'] = latest_data.get('timestamp')
                        details['latest_open'] = latest_data.get('open')
                        details['latest_high'] = latest_data.get('high')
                        details['latest_low'] = latest_data.get('low')
                        details['latest_close'] = latest_data.get('close')
                        details['latest_volume'] = latest_data.get('volume')
                        logger.info(f"Extracted latest OHLC data for {symbol_upper}")
                    else:
                         logger.warning(f"OHLC data list was empty for {symbol_upper}, cannot extract latest.")

                else:
                    logger.warning(f"No OHLC data found for {symbol_upper} in the last 90 days.")
            except Exception as ohlc_err:
                logger.error(f"Error fetching/processing OHLC data for {symbol_upper}: {ohlc_err}")
                # Continue without OHLC data

            # --- Calculate Support/Resistance ---
            sr_levels = []
            if ohlc_df is not None and not ohlc_df.empty:
                try:
                    logger.info(f"Calculating S/R levels for {symbol_upper} using {len(ohlc_df)} data points.")
                    config = AlgorithmConfig() # Instantiate config
                    sr_levels = identify_support_resistance(ohlc_df, config)
                    if sr_levels:
                        logger.info(f"Calculated {len(sr_levels)} S/R levels for {symbol_upper}: {sr_levels}")
                    else:
                        logger.warning(f"S/R calculation returned no levels for {symbol_upper}.")
                except Exception as sr_err:
                    logger.error(f"Error calculating S/R levels for {symbol_upper}: {sr_err}")
            else:
                logger.warning(f"Skipping S/R calculation due to missing or empty OHLC data for {symbol_upper}.")
            details['sr_levels'] = sr_levels # Add S/R levels to the response

            # --- Generate Chart (last 90 days) ---
            chart_url = None
            try:
                # Pass calculated sr_levels to the chart generator
                # The 'theme' variable now prioritizes session, then request args
                logger.info(f"Calling generate_and_save_chart for {symbol_upper} with theme: '{theme}' (Source: {'session' if 'theme' in session else 'request/default'})") # Log theme being passed and its source
                chart_url = generate_and_save_chart(symbol=symbol_upper, days=90, theme=theme, sr_levels=sr_levels) # Pass theme and sr_levels
                if chart_url:
                    logger.info(f"Chart generated successfully for {symbol_upper}: {chart_url}") # Log the returned URL
                else:
                    logger.warning(f"Chart generation failed for {symbol_upper}, generate_and_save_chart returned None") # Log failure
            except Exception as chart_err:
                logger.error(f"Error generating chart for {symbol_upper}: {chart_err}")
                # Continue without chart URL

            # Add chart URL
            details['chart_url'] = chart_url

            logger.info(f"Returning combined details for {symbol_upper}")
            logger.debug(f"Final details dictionary being returned for {symbol_upper}: {details}") # DEBUG LOG
            return jsonify(details)
        else:
            # Symbol not found in the database
            logger.warning(f"Symbol not found: {symbol_upper}")
            return jsonify({"error": f"Symbol '{symbol_upper}' not found"}), 404

    except Exception as e:
        logger.exception(f"Unexpected error fetching details for {symbol}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/symbols/<string:symbol>/chart', methods=['GET'])
def get_symbol_chart(symbol):
    """
    Returns OHLC chart data for a specific stock symbol.
    Accepts optional 'start_date' and 'end_date' query parameters (YYYY-MM-DD).
    e.g., /symbols/AAPL/chart?start_date=2024-01-01&end_date=2024-12-31
    """
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    logger.info(f"Received request for chart data for symbol: {symbol}" +
                (f" from {start_date}" if start_date else "") +
                (f" to {end_date}" if end_date else ""))

    try:
        ohlc_data = fetch_ohlc_data_db(symbol.upper(), start_date=start_date, end_date=end_date) # Ensure symbol is uppercase

        if ohlc_data is None or ohlc_data.empty: # Check for None or empty DataFrame
            logger.warning(f"No chart data found for symbol {symbol}" +
                           (f" from {start_date}" if start_date else "") +
                           (f" to {end_date}" if end_date else ""))
            return jsonify({"error": f"No chart data found for symbol '{symbol}'" +
                                     (f" in the specified date range" if start_date or end_date else "")}), 404

        # Convert DataFrame to JSON format suitable for charting libraries
        ohlc_data['timestamp'] = ohlc_data['timestamp'].astype(str)
        chart_data = ohlc_data.to_dict(orient='records')

        logger.info(f"Returning {len(chart_data)} data points for {symbol}")
        return jsonify(chart_data)

    except Exception as e:
        logger.exception(f"Unexpected error fetching chart data for {symbol}: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/search', methods=['GET'])
def search_page():
    """
    Renders the stock search page.
    If 'symbol' is provided in query args, fetches and displays its data.
    Reads 'theme' query parameter to request the correct chart theme.
    including paginated historical prices.
    """
    symbol = request.args.get('symbol') # Check for symbol in query args
    page = request.args.get('page', 1, type=int) # Get page number for pagination
    # Prioritize theme from session, then request args, then default
    theme = session.get('theme', request.args.get('theme', 'light'))
    per_page = 10
    if page < 1: page = 1

    logger.info(f"Rendering stock search page. Symbol: {symbol}, Page: {page}, Theme: {theme} (Source: {'session' if 'theme' in session else 'request/default'})")

    company_data = None
    indicators_data = None
    chart_url = None
    historical_data = None # Added for historical prices
    pagination = None      # Added for pagination
    error = None
    search_term = symbol # Use symbol from args as the search term
    company_summary = None # Initialize company summary
    economic_variables = None # Initialize economic variables
    economic_analysis = None # Initialize economic analysis
    economic_influences = None # Initialize economic influences
    trend = None # Initialize trend

    if symbol:
        symbol_upper = symbol.upper()
        logger.info(f"Fetching data for symbol: {symbol_upper}, page: {page}")

        # --- Fetch Company Details, Indicators, Chart URL (using internal API) ---
        api_base_url = request.host_url.strip('/')
        # Internal API call still needs theme, but it's now reliably sourced
        details_url = f"{api_base_url}/symbols/{symbol_upper}?theme={theme}"
        logger.info(f"Fetching details from internal API: {details_url}")
        try:
            response = requests.get(details_url, timeout=15)
            response.raise_for_status()
            api_response_data = response.json()
            company_data = api_response_data
            indicators_data = api_response_data.get('indicators')
            chart_url = api_response_data.get('chart_url')
            logger.info(f"Successfully fetched base data for {symbol_upper} from API")

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                error = f"Symbol '{symbol_upper}' not found."
                logger.warning(f"Symbol not found via internal API: {symbol_upper}")
                # Don't proceed to fetch historical if symbol not found
                return render_template('stock_search.html', error=error, search_term=search_term)
            else:
                error = f"API error fetching details: {response.status_code} - {response.reason}"
                logger.error(f"HTTP error fetching details for {symbol_upper}: {http_err}")
                # Continue, but historical might fail or be empty
        except requests.exceptions.RequestException as req_err:
            error = f"Error connecting to API: {req_err}"
            logger.exception(f"Request exception fetching details for {symbol_upper}: {req_err}")
            # Continue, but historical might fail or be empty
        except Exception as e:
            error = "An unexpected error occurred fetching details."
            logger.exception(f"Unexpected error fetching details for {symbol_upper}: {e}")
            # Continue, but historical might fail or be empty

        # --- Fetch Company Summary (if details were fetched) ---
        if company_data and company_data.get('name'):
            try:
                logger.info(f"Fetching company summary for: {company_data['name']}")
                company_summary = get_company_summary(company_data['name'])
                if company_summary:
                    logger.info(f"Successfully fetched company summary for {company_data['name']}")
                else:
                    logger.warning(f"Could not fetch company summary for {company_data['name']}")
            except Exception as summary_err:
                logger.error(f"Error fetching company summary for {company_data['name']}: {summary_err}")
                # Continue without summary, don't overwrite primary error
            # --- Fetch Economic Variables ---
            try:
                logger.info(f"Fetching economic variables for: {company_data['name']} ({symbol_upper})")
                economic_variables = get_economic_variables(company_data['name'], symbol_upper)
                if economic_variables:
                    logger.info(f"Successfully fetched {len(economic_variables)} economic variables for {symbol_upper}")
                else:
                    logger.warning(f"Could not fetch economic variables for {symbol_upper}")
            except Exception as econ_err:
                logger.error(f"Error fetching economic variables for {symbol_upper}: {econ_err}")
                # Continue without variables, don't overwrite primary error
            # --- Fetch Economic Analysis ---
            try:
                logger.info(f"Fetching economic analysis for: {company_data['name']} ({symbol_upper})")
                economic_analysis = get_economic_analysis(
                    company_name=company_data.get('name'),
                    symbol=symbol_upper,
                    sector=company_data.get('sector'),
                    industry=company_data.get('finnhub_industry')
                )
                if economic_analysis:
                    logger.info(f"Successfully fetched economic analysis for {symbol_upper}")
                else:
                    logger.warning(f"Could not fetch economic analysis for {symbol_upper}")
            except Exception as econ_an_err:
                logger.error(f"Error fetching economic analysis for {symbol_upper}: {econ_an_err}")
                # Continue without analysis

            # --- Analyze Economic Influences ---
            try:
                logger.info(f"Analyzing economic influences for: {symbol_upper}")
                analyzer = EconomicInfluenceAnalyzer()
                # TODO: Pass actual economic_data if available/needed
                economic_influences = analyzer.AnalyzeEconomicInfluences(symbol_upper)
                if economic_influences:
                    logger.info(f"Successfully analyzed economic influences for {symbol_upper}")
                else:
                    logger.warning(f"Economic influence analysis returned None for {symbol_upper}")
            except Exception as influence_err:
                logger.error(f"Error analyzing economic influences for {symbol_upper}: {influence_err}")
                # Continue without influences

        # --- Calculate 90-Day Trend ---
        trend = None
        try:
            # Fetch last 90 days specifically for trend calculation
            end_date_trend = datetime.now()
            start_date_trend = end_date_trend - timedelta(days=90)
            ohlc_df_trend = fetch_ohlc_data_db(
                symbol_upper,
                start_date=start_date_trend.strftime('%Y-%m-%d'),
                end_date=end_date_trend.strftime('%Y-%m-%d')
            )
            if ohlc_df_trend is not None and not ohlc_df_trend.empty and len(ohlc_df_trend) > 1:
                # Ensure data is sorted by date if not already guaranteed by fetch_ohlc_data_db
                ohlc_df_trend = ohlc_df_trend.sort_values(by='timestamp')
                first_close = ohlc_df_trend.iloc[0]['close']
                last_close = ohlc_df_trend.iloc[-1]['close']
                if last_close > first_close:
                    trend = 'up'
                elif last_close < first_close:
                    trend = 'down'
                else:
                    trend = 'flat'
                logger.info(f"Calculated 90-day trend for {symbol_upper}: {trend} (Start: {first_close}, End: {last_close})")
            else:
                logger.warning(f"Could not calculate trend for {symbol_upper} due to insufficient data (found {len(ohlc_df_trend) if ohlc_df_trend is not None else 'None'} points).")
        except Exception as trend_err:
            logger.error(f"Error calculating trend for {symbol_upper}: {trend_err}")

        # --- Fetch and Paginate Historical Data (only if company_data was found or no fatal API error) ---
        if company_data or not error: # Proceed if we have company data or the error wasn't fatal
            try:
                ohlc_df = fetch_ohlc_data_db(symbol_upper) # Fetch all historical data
                if ohlc_df is not None and not ohlc_df.empty:
                    logger.info(f"Fetched {len(ohlc_df)} total historical OHLC records for {symbol_upper}")
                    ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp']).dt.strftime('%Y-%m-%d')

                    total_count = len(ohlc_df)
                    total_pages = math.ceil(total_count / per_page)
                    start_index = (page - 1) * per_page
                    end_index = start_index + per_page
                    # Sort by timestamp descending before slicing for pagination
                    ohlc_df_sorted = ohlc_df.sort_values(by='timestamp', ascending=False)
                    ohlc_page_df = ohlc_df_sorted.iloc[start_index:end_index]

                    historical_data = ohlc_page_df.to_dict(orient='records')

                    pagination = {
                        'page': page, 'per_page': per_page, 'total_count': total_count,
                        'total_pages': total_pages, 'has_prev': page > 1, 'has_next': page < total_pages,
                        'prev_num': page - 1 if page > 1 else None,
                        'next_num': page + 1 if page < total_pages else None,
                        'symbol': symbol_upper # Pass symbol for pagination links
                    }
                    logger.info(f"Displaying historical page {page}/{total_pages} for {symbol_upper}")
                else:
                    logger.warning(f"No historical OHLC data found for {symbol_upper}")
            except Exception as hist_e:
                 logger.exception(f"Error fetching/paginating historical data for {symbol_upper}: {hist_e}")
                 if not error: # Avoid overwriting primary error
                     error = "Error fetching historical data."
            # --- End Historical Data Fetch ---

    # Render the template with all fetched data
    return render_template(
        'stock_search.html',
        company_data=company_data,
        indicators=indicators_data,
        chart_url=chart_url,
        historical_data=historical_data, # Pass historical data
        pagination=pagination,          # Pass pagination object
        error=error,
        search_term=search_term, # Pass symbol to prefill search box
        company_summary=company_summary, # Pass company summary
        economic_variables=economic_variables, # Pass economic variables
        economic_analysis=economic_analysis, # Pass economic analysis
        economic_influences=economic_influences, # Pass economic influences
        trend=trend, # Pass calculated trend
        active_page='search' # Indicate current page
    )


@app.route('/search', methods=['POST'])
def search_symbol():
    """
    Handles the stock symbol search form submission.
    Fetches data (details, indicators, chart, historical) and re-renders the search page.
    Reads 'theme' form parameter to request the correct chart theme.
    """
    symbol = request.form.get('symbol')
    page = 1 # Default to page 1 for POST search
    per_page = 10
    # For POST search, prioritize theme from the submitted form, then session, then default
    form_theme_post = request.form.get('theme') # Read from hidden input
    session_theme_post = session.get('theme')
    theme = form_theme_post or session_theme_post or 'light'
    logger.info(f"[/search POST] Theme check: Form='{form_theme_post}', Session='{session_theme_post}', Final='{theme}'")
    logger.info(f"Received search POST request for symbol: {symbol}, Theme: {theme}")

    company_data = None
    indicators_data = None
    chart_url = None
    historical_data = None
    pagination = None
    error = None
    company_summary = None # Initialize company summary
    economic_variables = None # Initialize economic variables
    economic_analysis = None # Initialize economic analysis
    economic_influences = None # Initialize economic influences

    if not symbol:
        error = "Please enter a stock symbol."
    else:
        symbol_upper = symbol.upper()
        logger.info(f"Fetching data for symbol: {symbol_upper}, page: {page}")

        # --- Fetch Company Details, Indicators, Chart URL (using internal API) ---
        api_base_url = request.host_url.strip('/')
        # Internal API call still needs theme, but it's now reliably sourced
        details_url = f"{api_base_url}/symbols/{symbol_upper}?theme={theme}"
        logger.info(f"Fetching details from internal API: {details_url}")
        try:
            response = requests.get(details_url, timeout=15)
            response.raise_for_status()
            api_response_data = response.json()
            company_data = api_response_data
            indicators_data = api_response_data.get('indicators')
            chart_url = api_response_data.get('chart_url')
            logger.info(f"Successfully fetched base data for {symbol_upper} from API")

        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404:
                error = f"Symbol '{symbol_upper}' not found."
                logger.warning(f"Symbol not found via internal API: {symbol_upper}")
                # Don't proceed to fetch historical if symbol not found
                return render_template('stock_search.html', error=error, search_term=symbol_upper)
            else:
                error = f"API error fetching details: {response.status_code} - {response.reason}"
                logger.error(f"HTTP error fetching details for {symbol_upper}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            error = f"Error connecting to API: {req_err}"
            logger.exception(f"Request exception fetching details for {symbol_upper}: {req_err}")
        except Exception as e:
            error = "An unexpected error occurred fetching details."
            logger.exception(f"Unexpected error fetching details for {symbol_upper}: {e}")

        # --- Fetch Company Summary (if details were fetched) ---
        if company_data and company_data.get('name'):
            try:
                logger.info(f"Fetching company summary for: {company_data['name']}")
                company_summary = get_company_summary(company_data['name'])
                if company_summary:
                    logger.info(f"Successfully fetched company summary for {company_data['name']}")
                else:
                    logger.warning(f"Could not fetch company summary for {company_data['name']}")
            except Exception as summary_err:
                logger.error(f"Error fetching company summary for {company_data['name']}: {summary_err}")
                # Continue without summary, don't overwrite primary error
            # --- Fetch Economic Variables ---
            try:
                logger.info(f"Fetching economic variables for: {company_data['name']} ({symbol_upper})")
                economic_variables = get_economic_variables(company_data['name'], symbol_upper)
                if economic_variables:
                    logger.info(f"Successfully fetched {len(economic_variables)} economic variables for {symbol_upper}")
                else:
                    logger.warning(f"Could not fetch economic variables for {symbol_upper}")
            except Exception as econ_err:
                logger.error(f"Error fetching economic variables for {symbol_upper}: {econ_err}")
                # Continue without variables, don't overwrite primary error
            # --- Fetch Economic Analysis ---
            try:
                logger.info(f"Fetching economic analysis for: {company_data['name']} ({symbol_upper})")
                economic_analysis = get_economic_analysis(
                    company_name=company_data.get('name'),
                    symbol=symbol_upper,
                    sector=company_data.get('sector'),
                    industry=company_data.get('finnhub_industry')
                )
                if economic_analysis:
                    logger.info(f"Successfully fetched economic analysis for {symbol_upper}")
                else:
                    logger.warning(f"Could not fetch economic analysis for {symbol_upper}")
            except Exception as econ_an_err:
                logger.error(f"Error fetching economic analysis for {symbol_upper}: {econ_an_err}")
                # Continue without analysis

            # --- Analyze Economic Influences ---
            try:
                logger.info(f"Analyzing economic influences for: {symbol_upper}")
                analyzer = EconomicInfluenceAnalyzer()
                # TODO: Pass actual economic_data if available/needed
                economic_influences = analyzer.AnalyzeEconomicInfluences(symbol_upper)
                if economic_influences:
                    logger.info(f"Successfully analyzed economic influences for {symbol_upper}")
                else:
                    logger.warning(f"Economic influence analysis returned None for {symbol_upper}")
            except Exception as influence_err:
                logger.error(f"Error analyzing economic influences for {symbol_upper}: {influence_err}")
                # Continue without influences
        # --- Calculate 90-Day Trend ---
        trend = None
        try:
            # Fetch last 90 days specifically for trend calculation
            end_date_trend = datetime.now()
            start_date_trend = end_date_trend - timedelta(days=90)
            ohlc_df_trend = fetch_ohlc_data_db(
                symbol_upper,
                start_date=start_date_trend.strftime('%Y-%m-%d'),
                end_date=end_date_trend.strftime('%Y-%m-%d')
            )
            if ohlc_df_trend is not None and not ohlc_df_trend.empty and len(ohlc_df_trend) > 1:
                 # Ensure data is sorted by date if not already guaranteed by fetch_ohlc_data_db
                ohlc_df_trend = ohlc_df_trend.sort_values(by='timestamp')
                first_close = ohlc_df_trend.iloc[0]['close']
                last_close = ohlc_df_trend.iloc[-1]['close']
                if last_close > first_close:
                    trend = 'up'
                elif last_close < first_close:
                    trend = 'down'
                else:
                    trend = 'flat'
                logger.info(f"Calculated 90-day trend for {symbol_upper}: {trend} (Start: {first_close}, End: {last_close})")
            else:
                logger.warning(f"Could not calculate trend for {symbol_upper} due to insufficient data (found {len(ohlc_df_trend) if ohlc_df_trend is not None else 'None'} points).")
        except Exception as trend_err:
            logger.error(f"Error calculating trend for {symbol_upper}: {trend_err}")

        # --- Fetch and Paginate Historical Data (only if company_data was found or no fatal API error) ---
        if company_data or not error:
            try:
                ohlc_df = fetch_ohlc_data_db(symbol_upper) # Fetch all historical data
                if ohlc_df is not None and not ohlc_df.empty:
                    logger.info(f"Fetched {len(ohlc_df)} total historical OHLC records for {symbol_upper}")
                    ohlc_df['timestamp'] = pd.to_datetime(ohlc_df['timestamp']).dt.strftime('%Y-%m-%d')

                    total_count = len(ohlc_df)
                    total_pages = math.ceil(total_count / per_page)
                    start_index = (page - 1) * per_page
                    end_index = start_index + per_page
                    # Sort by timestamp descending before slicing for pagination
                    ohlc_df_sorted = ohlc_df.sort_values(by='timestamp', ascending=False)
                    ohlc_page_df = ohlc_df_sorted.iloc[start_index:end_index]

                    historical_data = ohlc_page_df.to_dict(orient='records')

                    pagination = {
                        'page': page, 'per_page': per_page, 'total_count': total_count,
                        'total_pages': total_pages, 'has_prev': page > 1, 'has_next': page < total_pages,
                        'prev_num': page - 1 if page > 1 else None,
                        'next_num': page + 1 if page < total_pages else None,
                        'symbol': symbol_upper # Pass symbol for pagination links
                    }
                    logger.info(f"Displaying historical page {page}/{total_pages} for {symbol_upper}")
                else:
                    logger.warning(f"No historical OHLC data found for {symbol_upper}")
            except Exception as hist_e:
                 logger.exception(f"Error fetching/paginating historical data for {symbol_upper}: {hist_e}")
                 if not error: # Avoid overwriting primary error
                     error = "Error fetching historical data."
            # --- End Historical Data Fetch ---

    # Re-render the same template with all results
    return render_template(
        'stock_search.html',
        company_data=company_data,
        indicators=indicators_data,
        chart_url=chart_url,
        historical_data=historical_data,
        pagination=pagination,
        error=error,
        search_term=symbol_upper if symbol else None, # Pass uppercase symbol back
        company_summary=company_summary, # Pass company summary
        economic_variables=economic_variables, # Pass economic variables
        economic_analysis=economic_analysis, # Pass economic analysis
        economic_influences=economic_influences, # Pass economic influences
        trend=trend, # Pass calculated trend
        active_page='search' # Indicate current page
    )


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    """
    Provides symbol suggestions based on a prefix.
    Calls the internal /symbols API endpoint.
    """
    prefix = request.args.get('prefix', '')
    logger.info(f"Received autocomplete request for prefix: '{prefix}'")

    if not prefix:
        return jsonify([]) # Return empty list if no prefix

    # Construct the internal API URL dynamically
    api_base_url = request.host_url.strip('/') # Ensure no trailing slash
    symbols_url = f"{api_base_url}/symbols"
    params = {'prefix': prefix}
    logger.info(f"Fetching symbols from internal API: {symbols_url} with params: {params}")

    try:
        response = requests.get(symbols_url, params=params, timeout=3) # Shorter timeout for autocomplete
        response.raise_for_status()
        suggestions = response.json()
        logger.info(f"Returning {len(suggestions)} suggestions for prefix '{prefix}'")
        return jsonify(suggestions)

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Error calling internal /symbols endpoint for autocomplete (prefix: '{prefix}'): {req_err}")
        # Don't expose internal errors directly to the autocomplete frontend
        return jsonify({"error": "Failed to fetch suggestions"}), 500
    except Exception as e:
        logger.exception(f"Unexpected error during autocomplete for prefix '{prefix}': {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/scan', methods=['GET', 'POST'])
def scan_page():
    """ Renders the stock scanner page and handles scan submissions. """
    logger.info(f"Scan page request. Method: {request.method}, Args: {request.args}, Form: {request.form}")

    # Fetch sectors for the dropdown
    sectors = get_unique_sectors()
    if sectors is None:
        sectors = []
        # Use flash for user-facing errors that don't stop rendering
        flash("Error fetching sector list from database.", "danger")
        logger.error("Failed to fetch sectors from database for scan page.")


    results = None
    error = None # For errors during scan execution
    pagination = None
    form_data = {} # To store current filter values (from args or form)
    page = request.args.get('page', 1, type=int)
    if page < 1: # Ensure page is at least 1
        page = 1
    per_page = 20 # Define results per page

    # Read filter criteria from request.values (combines args and form, args take precedence)
    # This allows pagination links (GET requests with args) to preserve filters
    form_data['sector'] = request.values.get('sector', '').strip()
    form_data['close_op'] = request.values.get('close_op', '').strip()
    form_data['close_val'] = request.values.get('close_val', '').strip()
    form_data['vol_op'] = request.values.get('vol_op', '').strip()
    form_data['vol_val'] = request.values.get('vol_val', '').strip()

    # Only perform the scan if it's a POST request OR if filter parameters are present in GET (e.g., from pagination click)
    # Check if any filter value is non-empty after stripping whitespace
    has_filters = any(v for v in form_data.values())
    perform_scan = request.method == 'POST' or (request.method == 'GET' and has_filters)

    if perform_scan:
        logger.info(f"Performing scan with filters: {form_data}, page: {page}")
        try:
            # Pass None if value is empty string after stripping
            results, total_count, db_error = scan_stocks(
                sector=form_data['sector'] if form_data['sector'] else None,
                close_op=form_data['close_op'] if form_data['close_op'] else None,
                close_val=form_data['close_val'] if form_data['close_val'] else None,
                vol_op=form_data['vol_op'] if form_data['vol_op'] else None,
                vol_val=form_data['vol_val'] if form_data['vol_val'] else None,
                page=page,
                per_page=per_page
            )

            if db_error:
                error = f"Database Error: {db_error}" # Show specific DB error if available
                results = None
                total_count = 0
                logger.error(f"Scan failed with DB error: {db_error}")
            elif results is not None:
                 # Calculate pagination details only if results were fetched successfully
                if total_count > 0:
                    total_pages = math.ceil(total_count / per_page)
                    pagination = {
                        'page': page,
                        'per_page': per_page,
                        'total_count': total_count,
                        'total_pages': total_pages,
                        'has_prev': page > 1,
                        'has_next': page < total_pages,
                        'prev_num': page - 1 if page > 1 else None,
                        'next_num': page + 1 if page < total_pages else None,
                        # Pass current non-empty filters to pagination links
                        'filters': {k: v for k, v in form_data.items() if v}
                    }
                    logger.info(f"Scan successful. Page: {page}, Results: {len(results)}, Total: {total_count}")
                else:
                    logger.info("Scan successful but found 0 matching results.")
                    # No pagination needed if no results
            else:
                # This case might indicate an unexpected issue in scan_stocks if db_error was None
                error = "Scan returned no results or an unexpected error occurred."
                total_count = 0
                logger.warning("scan_stocks returned None results without a db_error message.")


        except ValueError as ve:
             # Catch potential float/int conversion errors passed up from scan_stocks (though handled there too)
             error = f"Invalid input value: {ve}"
             logger.error(f"ValueError during scan processing: {ve}")
             results = None
             total_count = 0
        except Exception as e:
            logger.exception(f"Error processing scan request: {e}")
            error = "An unexpected server error occurred during the scan."
            results = None
            total_count = 0 # Ensure total_count is defined even on exception

    # Render the template, passing necessary context
    return render_template(
        'scan.html',
        sectors=sectors,
        results=results,
        pagination=pagination,
        error=error,
        form_data=form_data, # Pass current filters back to form
        active_page='scan' # Indicate current page
    )
# Removed the stock_detail_page route as historical data is now shown on the search page


@app.route('/scan/reversal', methods=['GET'])
def reversal_scan_page():
    """ Displays the latest pre-calculated reversal scan results from the database. """
    logger.info("Received request for Reversal Scan results page.")
    results = None
    error = None
    scan_timestamp = None

    try:
        # Fetch results directly from the dedicated table
        results = fetch_reversal_scan_results_db()

        if results is None:
            error = "Could not retrieve reversal scan results from the database. The background job might have failed or not run yet."
            logger.error("fetch_reversal_scan_results_db returned None.")
        elif not results:
            error = "No stocks matched the reversal criteria in the latest scan."
            logger.info("No results found in reversal_scan_results table.")
            # Attempt to get the timestamp even if results are empty (table might exist but be empty)
            # This requires a separate query or modification to fetch_reversal_scan_results_db
            # For simplicity, we'll only show timestamp if results exist.
        else:
            # Extract the timestamp from the first result (assuming all rows have the same)
            scan_timestamp = results[0].get('scan_timestamp')
            logger.info(f"Successfully retrieved {len(results)} reversal scan results.")

    except Exception as e:
        logger.exception(f"Error fetching or processing reversal scan results: {e}")
        error = "An unexpected error occurred while retrieving scan results."
        results = None # Ensure results are None on error

    # Render the results template
    return render_template(
        'reversal_scan_results.html',
        results=results,
        error=error,
        scan_timestamp=scan_timestamp, # Pass timestamp to template
        active_page='reversal_scan' # For navbar highlighting
    )



@app.route('/set_theme', methods=['POST'])
def set_theme():
    """ Stores the selected theme ('light' or 'dark') in the session. """
    data = request.get_json()
    theme = data.get('theme')
    if theme in ['light', 'dark']:
        session['theme'] = theme
        session.modified = True # Explicitly mark session as modified
        logger.info(f"Theme set in session: {theme}")
        return jsonify({"success": True, "theme": theme}), 200
    else:
        logger.warning(f"Invalid theme value received: {theme}")
        return jsonify({"error": "Invalid theme value"}), 400




@app.route('/rrg')
def rrg_page():
    """ Renders the Relative Rotation Graph (RRG) page. """
    logger.info("Received request for RRG page.")
    rrg_plot_url = None
    error = None

    try:
        # Ensure the plot directory exists
        os.makedirs(RRG_PLOT_DIR, exist_ok=True)

        # 1. Fetch data for RRG stocks and benchmark
        symbols_to_fetch = RRG_STOCKS + [RRG_BENCHMARK]
        # Fetch data for the last 2 years for better EMA/momentum calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        logger.info(f"Fetching data for RRG: {symbols_to_fetch} from {start_date.date()} to {end_date.date()}")

        prices_df = fetch_ohlc_data_for_symbols_db(
            symbols_to_fetch,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if prices_df is None or prices_df.empty:
            logger.error("Failed to fetch necessary data for RRG plot.")
            error = "Could not retrieve the data needed to generate the RRG plot."
        else:
            logger.info(f"Successfully fetched data for {len(prices_df.columns)} symbols for RRG.")
            # 2. Generate the RRG plot
            # Use the defined constants and pass the fetched data
            generated_path = generate_rrg_plot(
                prices_df=prices_df,
                stocks=RRG_STOCKS,
                benchmark=RRG_BENCHMARK,
                output_filename=RRG_PLOT_PATH,
                tail_length=10 # Default tail length
            )

            if generated_path:
                # Generate URL within the request context using url_for
                rrg_plot_url = url_for('static', filename=f'plots/{RRG_PLOT_FILENAME}')
                logger.info(f"RRG plot generated successfully: {rrg_plot_url}")
                # Add cache-busting query parameter
                rrg_plot_url += f"?v={datetime.now().timestamp()}"
            else:
                logger.error("RRG plot generation function failed.")
                error = "Failed to generate the RRG plot image."

    except Exception as e:
        logger.exception(f"Error generating RRG page: {e}")
        error = "An unexpected error occurred while generating the RRG page."

    return render_template(
        'rrg.html',
        rrg_plot_url=rrg_plot_url,
        error=error,
        rrg_benchmark=RRG_BENCHMARK, # Pass benchmark to template
        rrg_stocks=RRG_STOCKS,       # Pass stocks list to template
        active_page='rrg' # For navbar highlighting
    )


if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on the network
    app.run(host='0.0.0.0', port=5000, debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)