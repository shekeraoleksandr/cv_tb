import tensorflow as tf
import numpy as np


class TradingStrategy:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def decide_trade(self, data):
        if data['Close'] <= 0:
            return 'hold'  # Avoid trading on invalid data

        features = self.prepare_features(data)
        prediction = self.model.predict(features, verbose=0)[0][0]

        if prediction > 0.6:
            return 'buy'
        elif prediction < 0.4:
            return 'sell'
        else:
            return 'hold'

    @staticmethod
    def prepare_features(data):
        # Assuming preprocessing that matches the model's training preprocessing
        features = np.array([[data['Open'], data['High'], data['Low'], data['Close'], data['Volume']]])
        return features
