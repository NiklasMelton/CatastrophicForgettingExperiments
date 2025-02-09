import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD


class CNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 input_shape=(28, 28, 1),
                 num_classes=10,
                 conv_layers=[(32, (3, 3))],
                 pool_size=(2, 2),
                 dense_layers=[128],
                 dropout=0.5,
                 learning_rate=0.001,
                 epochs=10,
                 batch_size=32):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_layers = conv_layers
        self.pool_size = pool_size
        self.dense_layers = dense_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._initialized = False

    def _build_model(self):
        model = Sequential()

        # Add convolutional layers
        for filters, kernel_size in self.conv_layers:
            if len(model.layers) == 0:
                model.add(Conv2D(filters, kernel_size, activation='relu',
                                 input_shape=self.input_shape))
            else:
                model.add(Conv2D(filters, kernel_size, activation='relu'))
            model.add(MaxPooling2D(pool_size=self.pool_size))

        model.add(Flatten())

        # Add dense layers
        for units in self.dense_layers:
            model.add(Dense(units, activation='relu'))
            if self.dropout > 0:
                model.add(Dropout(self.dropout))

        # Add output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(optimizer=SGD(learning_rate=self.learning_rate, momentum=0.0),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def fit(self, X, y):
        # Preprocess labels
        y_categorical = to_categorical(y, num_classes=self.num_classes)

        # Build and train model
        if not self._initialized:
            self.model = self._build_model()
            self._initialized = True

        self.model.fit(X, y_categorical,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)
        return self

    def partial_fit(self, X, y, classes=None):
        # Preprocess labels
        y_categorical = to_categorical(y, num_classes=self.num_classes)

        # Build model if not initialized
        if not self._initialized:
            if classes is None:
                raise ValueError(
                    "`classes` must be specified for the first call to `partial_fit`.")
            self.model = self._build_model()
            self._initialized = True

        # Train on the batch without resetting the optimizer state
        self.model.fit(X, y_categorical,
                       epochs=1,
                       batch_size=self.batch_size,
                       verbose=1)
        return self

    def predict(self, X):
        if not self.model:
            raise ValueError(
                "The model has not been trained yet. Call `fit` or `partial_fit` first.")

        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        if not self.model:
            raise ValueError(
                "The model has not been trained yet. Call `fit` or `partial_fit` first.")
        return self.model.predict(X)