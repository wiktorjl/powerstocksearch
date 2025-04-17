https://logo.clearbit.com/wp.pl

Order of scripts:

# Download data (daily operation)
# Pull stock price and split history from EODHD
python3 -m src.data_acquisition.stock_data_downloader
python3 -m src.data_acquisition.split_data_downloader


# Adjust prices for splits
python -m src.feature_engineering.split_adjuster

# Move from json with split adjusted data to DB
python -m src.database.ohlc_spit_importer 

# Upload company profile info
python -m src.database.company_info_importer

# Indicators
python -m src.database.indicator_calculator 



# Generate graphs
python -m src.visualization.ohlc_plotter --output /home/user/stockdata/graphs/AAPL.jpg --days 200 --sr AAP


## Economic Influence Analysis Summary

| Feature/Capability        | Assessment | Description                                                                                                |
| :------------------------ | :--------- | :--------------------------------------------------------------------------------------------------------- |
| Factor Breakdown          | (+)        | Identifies key economic indicators (interest rates, inflation, etc.) and assesses their likely impact.     |
| Sentiment Analysis        | (+)        | Incorporates sentiment from news/social media related to the stock or sector.                              |
| Influence Score/Summary | (+)        | Generates an overall score or summary indicating the net economic influence.                               |
| Implementation Details    | (Info)     | Core logic in `src/analysis/economic_influence_analyzer.py`, integrated into `src/web/flaskapp.py`. |

### Placeholder Economic Factors Used in Analysis
| Category         | Factor                  | Condition for Positive Influence | Condition for Negative Influence |
| :--------------- | :---------------------- | :------------------------------- | :------------------------------- |
| **Macroeconomic**| Interest Rate           | `< 0.03`                         | `>= 0.03`                        |
|                  | GDP Growth              | `> 0.02`                         | `<= 0.02`                        |
|                  | Inflation Rate          | `<= 0.04`                        | `> 0.04`                         |
| **Company-Specific**| Company Earnings Growth | `> 0.08`                         | `<= 0.08`                        |
|                  | Industry Competition    | `!= "High"`                      | `== "High"`                      |