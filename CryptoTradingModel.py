from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
import os


class CryptoTradingModel:
    def __init__(self):
        self.model = None  # Placeholder for the TensorFlow model

    def build_model(self, input_shape):
        """
        Build and compile the TensorFlow model.

        :param input_shape: Shape of the input data
        """
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

    def train_test_split(self, data, labels, test_size=0.3):
        """
        Split the data into training and testing sets.

        :param data: DataFrame containing the features
        :param labels: Series containing the target variable
        :param test_size: Proportion of the dataset to include in the test split
        :return: Split data and labels into training and testing sets
        """
        return train_test_split(data, labels, test_size=test_size, random_state=42)

    def train_model(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the model on the provided data.

        :param X_train: Training data
        :param y_train: Labels for the training data
        :param epochs: Number of epochs to train for
        :param batch_size: Batch size for training
        """
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model's performance on the provided test data and labels.

        :param X_test: Test data
        :param y_test: Labels for the test data
        :return: Evaluation results
        """
        predictions = self.model.predict(X_test)
        predictions = (predictions > 0.5).astype(int).flatten()  # Converting probabilities to binary output

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def save_model(self, model_path='model.h5'):
        if self.model:
            self.model.save(model_path)
            print(f'Model saved to {model_path}')

    def load_model(self, model_path='model.h5'):
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f'Model loaded from {model_path}')
        else:
            print('Model file not found.')
