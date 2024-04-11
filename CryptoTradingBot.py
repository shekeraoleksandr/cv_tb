from CryptoDataLoader import CryptoDataLoader
from CryptoTradingModel import CryptoTradingModel
from sklearn.model_selection import KFold
from dotenv import load_dotenv
import h5py
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


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
            logger.info("Loading existing model...")
            self.model.load_model(self.model_path)
        else:
            logger.info("Training new model...")
            # data = self.loader.get_historical_data(self.symbol, self.interval, self.start, self.end)
            data = self.loader.get_binance_historical_data(self.symbol, self.interval, self.start, self.end)
            preprocessed_data = self.loader.preprocess_data(data)

            preprocessed_data, labels = self.generate_labels(preprocessed_data)

            kf = KFold(n_splits=5)
            scores = []

            for train_index, test_index in kf.split(preprocessed_data):
                X_train, X_test = preprocessed_data.iloc[train_index], preprocessed_data.iloc[test_index]
                y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

                self.model.build_model(input_shape=(X_train.shape[1],))
                self.model.train_model(X_train, y_train)

                score = self.model.evaluate_model(X_test, y_test)
                scores.append(score)
                logger.info(f"Fold score: {score}")

            avg_score = np.mean(scores, axis=0)
            logger.info(f"Average model score: {avg_score}")

            self.model.save_model(self.model_path)

    def generate_labels(self, data):
        labels = (data['Close'].diff() > 0).astype(int)
        labels = labels.shift(-1).dropna()
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
