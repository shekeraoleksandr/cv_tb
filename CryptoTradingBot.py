from CryptoDataLoader import CryptoDataLoader
from CryptoTradingModel import CryptoTradingModel
from dotenv import load_dotenv
import h5py
import os


class CryptoTradingBot:
    def __init__(self, api_key, api_secret, symbol, interval, start, end, mode):
        self.loader = CryptoDataLoader(api_key, api_secret)
        self.model = CryptoTradingModel()
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end
        self.model_path = 'models/my_trading_model.h5'
        self.mode = mode

    def run(self):
        if os.path.exists(self.model_path) and self.mode != 1:
            print("Loading existing model...")
            self.model.load_model(self.model_path)
        else:
            print("Training new model...")
            data = self.loader.get_historical_data(self.symbol, self.interval, self.start, self.end)
            preprocessed_data = self.loader.preprocess_data(data)

            # Assume we have a way to generate labels for the data
            preprocessed_data, labels = self.generate_labels(preprocessed_data)

            # Split the data
            X_train, X_test, y_train, y_test = self.model.train_test_split(preprocessed_data, labels)

            # Build and train the model
            self.model.build_model(input_shape=(X_train.shape[1],))
            self.model.train_model(X_train, y_train)
            self.model.save_model(self.model_path)

            # Evaluate the model
            evaluation_results = self.model.evaluate_model(X_test, y_test)
            print(evaluation_results)

    def generate_labels(self, data):
        # Example label generation based on price increase/decrease
        labels = (data['Close'].diff() > 0).astype(int)
        # Align labels with the data by shifting and dropping the last NaN value
        labels = labels.shift(-1).dropna()
        # Also, drop the last row from data to match the labels' length
        data = data.iloc[:-1]
        return data, labels

    def inspect_h5_file(self):
        if os.path.exists(self.model_path):
            with h5py.File(self.model_path, 'r') as file:
                self._print_file_structure(file)
        else:
            print(f"Model file {self.model_path} not found.")

    def _print_file_structure(self, file, indent=0):
        for key in file.keys():
            item = file[key]
            print('  ' * indent + f'Key: {key}')
            if isinstance(item, h5py.Dataset):
                print('  ' * indent + f'  Dataset with shape: {item.shape}, type: {item.dtype}')
            else:
                print('  ' * indent + '  Group')
                self._print_file_structure(item, indent + 1)


# Example usage (commented out to prevent accidental execution)
if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("API_KEY")
    secret_key = os.getenv("SECRET_KEY")
    bot = CryptoTradingBot(api_key, secret_key,
                           'BTCUSDT', '1d',
                           '1 Jan, 2021', '1 Jan, 2022', 1)
    bot.run()
    # bot.inspect_h5_file()
    # cdl = CryptoDataLoader(api_key, secret_key)
    # history = cdl.get_historical_data('BTCUSDT', '1d', '1 Jan, 2021', '1 Jan, 2022')
    # cdl.preprocess_data(history)
