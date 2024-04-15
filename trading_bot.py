import json
import os
import pandas as pd
from sklearn.model_selection import KFold
from numpy import mean
from trading_strategy import TradingStrategy
from backtester import Backtester
from crypto_data_loader import CryptoDataLoader
from crypto_trading_model import CryptoTradingModel


def load_config():
    with open('config.json', 'r') as file:
        return json.load(file)


class CryptoTradingBot:
    def __init__(self, config):
        self.config = config
        self.loader = CryptoDataLoader(config['binance']['api_key'], config['binance']['api_secret'])
        self.model = CryptoTradingModel(config['backtesting']['input_shape'])
        self.csv_path = config['backtesting']['csv_path']
        self.model_path = config['backtesting']['model_path']
        self.mode = config['mode']  # Mode is set in the config for what action to perform

    def run(self):
        if os.path.exists(self.model_path) and self.mode == 0:
            print("Loading existing model...")
            self.model.load_model(self.model_path)
        elif self.mode == 1:
            print("Training new model...")
            data = self.loader.get_historical_data(self.config['backtesting']['symbol'],
                                                   self.config['backtesting']['interval'],
                                                   self.config['backtesting']['start_date'],
                                                   self.config['backtesting']['end_date'])
            preprocessed_data = self.loader.preprocess_data(data)
            preprocessed_data, labels = self.generate_labels(preprocessed_data)

            kf = KFold(n_splits=5)
            accuracies, precisions, recalls, f1_scores = [], [], [], []

            for train_index, test_index in kf.split(preprocessed_data):
                X_train, X_test = preprocessed_data.iloc[train_index], preprocessed_data.iloc[test_index]
                y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

                self.model.train_model(X_train, y_train)
                eval_results = self.model.evaluate_model(X_test, y_test)
                accuracies.append(eval_results['accuracy'])
                precisions.append(eval_results['precision'])
                recalls.append(eval_results['recall'])
                f1_scores.append(eval_results['f1_score'])

            print(f"Average Accuracy: {mean(accuracies)}")
            print(f"Average Precision: {mean(precisions)}")
            print(f"Average Recall: {mean(recalls)}")
            print(f"Average F1 Score: {mean(f1_scores)}")

            self.model.save_model(self.model_path)
        elif self.mode == 2:
            print("Running backtesting...")
            if not os.path.exists(self.csv_path):
                print("Downloading historical data...")
                self.loader.save_historical_data_to_csv(
                    self.config['backtesting']['symbol'],
                    self.config['backtesting']['interval'],
                    self.config['backtesting']['start_date'],
                    self.config['backtesting']['end_date'],
                    self.csv_path
                )

            data = pd.read_csv(self.csv_path)
            backtester = Backtester(
                initial_capital=self.config['backtesting']['initial_capital'],
                transaction_cost_percent=self.config['backtesting']['transaction_cost'],
                model_path=self.config['backtesting']['model_path']
            )
            results = backtester.simulate_trading(data)
            print("Backtesting results:", results)

    @staticmethod
    def generate_labels(data):
        labels = (data['Close'].diff() > 0).astype(int)
        labels = labels.shift(-1).dropna()
        data = data.iloc[:-1]
        return data, labels


if __name__ == "__main__":
    config = load_config()
    bot = CryptoTradingBot(config)
    bot.run()
