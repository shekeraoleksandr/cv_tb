import pandas as pd
from binance.client import Client
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import requests
import numpy as np


class CryptoDataLoader:

    def __init__(self, api_key, secret_key):
        self.client = None
        self.api_key = api_key
        self.secret_key = secret_key

    def initialize_client(self):
        try:
            self.client = Client(self.api_key, self.secret_key, requests_params={'timeout': 20})
            self.client.ping()
            print("Successfully connected to Binance API.")
            return True
        except requests.exceptions.ConnectionError:
            print("Connection error. Unable to connect to Binance API.")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

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

    @staticmethod
    def get_binance_historical_data(symbol, interval, start, end):
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

    def save_historical_data_to_csv(self, symbol, interval, start, end, file_path):
        """
        Fetch historical data and save it to a CSV file.

        :param symbol: Trading pair symbol, e.g., 'BTCUSDT'
        :param interval: Interval for candlestick data, e.g., '1h'
        :param start: Start date string
        :param end: End date string
        :param file_path: File path where the CSV should be saved
        """
        df = self.get_historical_data(symbol, interval, start, end)
        df = self.preprocess_data(df)
        df.to_csv(file_path)
        print(f"Data saved to {file_path}")

    @staticmethod
    def load_data_from_csv(filepath):
        try:
            df = pd.read_csv(filepath)
            # Ensure the 'Open time' column is treated as datetime type
            if 'Open time' in df.columns:
                df['Open time'] = pd.to_datetime(df['Open time'])
                df.set_index('Open time', inplace=True)
            return df
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return None

    @staticmethod
    def preprocess_data(df):
        # Calculate daily percentage change
        df = df.pct_change().dropna()

        # Handle infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Normalize features using Min-Max scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

        return df_scaled
