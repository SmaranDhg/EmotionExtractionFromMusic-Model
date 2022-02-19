# %%

import tensorflow_addons as tfa
import tensorflow as tf

from lib.setup import INPUT_SHAPE, OUT, OUT_1, OUT_2


class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, data_format="channels_last", name=None):
        super().__init__(name=name)
        self.conv_in = tf.keras.layers.Conv2D(
            filters, kernel_size, padding="same", data_format=data_format
        )

        self.conv_1 = tf.keras.layers.Conv2D(
            filters, kernel_size, padding="same", data_format=data_format
        )
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(
            filters, kernel_size, padding="same", data_format=data_format
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation("softsign")
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        input_tensor = self.conv_in(input_tensor)

        x = self.conv_1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv_2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)

        return x


class Model(tf.keras.models.Model):
    def __init__(self, in_shape=INPUT_SHAPE[1:]):
        super().__init__()

        self.in_shape = in_shape

        self.feature_layers = [
            tf.keras.layers.Conv2D(124, 4, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("softsign"),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Dropout(0.5),
            IdentityBlock(64, 4),
            IdentityBlock(64, 4),
            tf.keras.layers.Dropout(0.5),
            IdentityBlock(64, 4),
            tf.keras.layers.Dropout(0.4),
            IdentityBlock(64, 4),
            tf.keras.layers.Dropout(0.3),
            IdentityBlock(32, 3),
            tf.keras.layers.Dropout(0.3),
            IdentityBlock(32, 3),
            tf.keras.layers.Dropout(0.3),
            IdentityBlock(64, 3),
            tf.keras.layers.Dropout(0.2),
            IdentityBlock(64, 3),
            tf.keras.layers.Dropout(0.2),
            IdentityBlock(64, 3),
            tf.keras.layers.GlobalAveragePooling2D(),
        ]

        self.reg_layers = [
            tf.keras.layers.Dense(300, activation="softsign"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(200, activation="softsign"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(150, activation="softsign"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation="softsign"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(100, activation="softsign"),
            tf.keras.layers.Dense(1, name=OUT),
        ]
        self.segments = [
            self.feature_layers,
            self.reg_layers,
        ]

    def call(self, x):

        for l in self.feature_layers:
            x = l(x)

        for l in self.reg_layers:
            x = l(x)

        return x

    def model(self):
        x = tf.keras.layers.Input(shape=self.in_shape)
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
