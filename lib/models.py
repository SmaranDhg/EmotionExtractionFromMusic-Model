# %%
import tensorflow as tf
import numpy as np
from .setup import EMBEDDING_SIZE, MODEL_ANNOT, MODEL, NUM_CLASSES


"""---------------------------Basic Models---------------------------"""

__models = ["classifier_seq", "regressor_seq", "classifier", "regressor"]
__models += []  # vision
__models += []  # time-series
__models += []  # nlp


class __Regressor(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.i = tf.keras.layers.Dense(300, activation="relu")
        self.h1 = tf.keras.layers.Dense(200, activation="relu")
        self.h2 = tf.keras.layers.Dense(200, activation="relu")
        self.o = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs):
        x = self.i(inputs)
        x = self.h1(x)
        x = self.h2(x)
        x = self.o(x)

        return x


class __Classifier(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.i = tf.keras.layers.Dense(300, activation="relu")
        self.h1 = tf.keras.layers.Dense(200, activation="relu")
        self.h2 = tf.keras.layers.Dense(200, activation="relu")
        self.o = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")

    def call(self, inputs):
        x = self.i(inputs)
        x = self.h1(x)
        x = self.h2(x)
        x = self.o(x)

        return x


class __Regressor_seq(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        input = tf.keras.layers.Input(
            shape=[None, EMBEDDING_SIZE], dtype=tf.float32, ragged=True
        )
        x = input

        x = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(10))(x)

        x = tf.keras.layers.Dense(300, activation="relu")(x)
        x = tf.keras.layers.Dense(200, activation="relu")(x)
        x = tf.keras.layers.Dense(200, activation="relu")(x)
        x = tf.keras.layers.Dense(1, activation="relu")(x)

        self.model = tf.keras.Model(input, x)
        plot_model(
            self.model,
            to_file=f"model{MODEL_ANNOT}.png",
            show_dtype=True,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
        )

    def call(self, inputs):
        x = self.model(inputs)
        return x


class __Classifier_seq(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        input = tf.keras.layers.Input(
            shape=[None, EMBEDDING_SIZE], dtype=tf.float32, ragged=True
        )
        x = input

        x = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(10))(x)

        x = tf.keras.layers.Dense(300, activation="relu")(x)
        x = tf.keras.layers.Dense(200, activation="relu")(x)
        x = tf.keras.layers.Dense(200, activation="relu")(x)
        x = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

        self.model = tf.keras.Model(input, x)
        plot_model(
            self.model,
            to_file=f"model{MODEL_ANNOT}.png",
            show_dtype=True,
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True,
        )

    def call(self, inputs):
        x = self.model(inputs)
        return x


"""---------------------------Vision Models---------------------------"""

"""---------------------------Time Series Models---------------------------"""

"""---------------------------NLP Models---------------------------"""


def get_model():
    try:
        from .model import Model

        return Model
    except:
        models = dict(
            zip(
                __models,
                [
                    __Classifier_seq,
                    __Regressor_seq,
                    __Classifier,
                    __Regressor,
                ],
            )
        )
        return models[MODEL]
