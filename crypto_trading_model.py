import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


class CryptoTradingModel:
    def __init__(self, input_shape=(None, 5)):  # Default input shape, adjust as needed
        """
        Initialize the TensorFlow model.

        :param input_shape: Expected shape of the input data
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    @staticmethod
    def train_test_split(data, labels, test_size=0.3):
        return train_test_split(data, labels, test_size=test_size, random_state=42)

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    @tf.function  # Decorate the predict method to optimize it
    def predict(self, X):
        return self.model(X)

    def evaluate_model(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        predictions = tf.cast(predictions > 0.5, tf.int32)  # Cast boolean to int
        predictions = tf.reshape(predictions, [-1])  # Flatten the predictions

        accuracy = accuracy_score(y_test, predictions.numpy())
        precision = precision_score(y_test, predictions.numpy())
        recall = recall_score(y_test, predictions.numpy())
        f1 = f1_score(y_test, predictions.numpy())

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save_model(self, model_path='model.h5'):
        self.model.save(model_path)
        print(f'Model saved to {model_path}')

    def load_model(self, model_path='model.h5'):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f'Model loaded from {model_path}')
        else:
            print('Model file not found.')
