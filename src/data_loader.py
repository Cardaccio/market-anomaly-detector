# Load market data using Yahoo Finance
import yfinance as yf
import os
import pandas as pd

# Define tickers and output directory
tickers = ['AAPL', 'SPY', 'TSLA', 'JPM', 'BTC-USD']
output_dir = '../data/raw'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize an empty DataFrame to store all ticker's data
all_data = pd.DataFrame()

# Download data for each ticker and append to the DataFrame
for ticker in tickers:
    data = yf.download(ticker, period='2y', interval='1d')
    data.columns=['Close','High','Low','Open','Volume']
    data.reset_index(inplace=True)
    data['Ticker'] = ticker  # Add a column to identify the ticker
    all_data = pd.concat([all_data, data])

# Save the combined data to a single CSV file
csv_path = os.path.join(output_dir, "all_tickers.csv")
all_data.to_csv(csv_path)

