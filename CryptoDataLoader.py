import pandas as pd
from binance.client import Client
from sklearn.preprocessing import StandardScaler
import requests
import numpy as np


class CryptoDataLoader:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret, requests_params={'timeout': 20})

    def get_historical_data(self, symbol, interval, start, end):
        """
        Fetch historical price data from Binance for a given symbol, interval, and time period.

        :param symbol: Trading pair symbol, e.g., 'BTCUSDT'
        :param interval: Interval for candlestick data, e.g., '1h' for one hour
        :param start: Start time for the data retrieval
        :param end: End time for the data retrieval
        :return: DataFrame containing the historical price data
        """
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                   'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                   'Taker buy quote asset volume', 'Ignore']

        klines = self.client.get_historical_klines(symbol, interval, start, end)
        df = pd.DataFrame(klines, columns=columns)
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df.set_index('Open time', inplace=True)

        # Convert columns to appropriate data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(float)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]

    def get_binance_historical_data(self, symbol, interval, start, end):
        """
        Fetch historical candlestick data from Binance for a given symbol and interval.

        :param symbol: Symbol to fetch data for (e.g., 'BTCUSDT')
        :param interval: Interval for candlestick data (e.g., '1h' for one hour, '1d' for one day)
        :param start: Start time in milliseconds since Epoch
        :param end: End time in milliseconds since Epoch
        :return: DataFrame containing the historical data
        """
        url = f'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start,
            'endTime': end,
            'limit': 1000  # Adjust based on how much data you want (max 1000)
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Convert the API response into a pandas DataFrame
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                   'Close time', 'Quote asset volume', 'Number of trades',
                   'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        df = pd.DataFrame(data, columns=columns)

        # Convert timestamp to datetime and numeric columns to floats
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                           'Quote asset volume', 'Taker buy base asset volume',
                           'Taker buy quote asset volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, axis=1)

        return df

    def preprocess_data(self, df):
        # Calculate daily percentage change
        df = df.pct_change().dropna()

        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Additional feature engineering could go here

        # Normalize or standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

        return df_scaled
