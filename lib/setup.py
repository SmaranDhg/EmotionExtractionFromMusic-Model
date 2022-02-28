from math import ceil
import tensorflow as tf

import tensorflow_addons as tfa
import os

DIR_SAVE_MODEL = "saved_models"
DIR_LOGS = "logs"
DIR_ASSETS = "assets"
DIR_RUN = "runs"
DIR_PICKLE = f"{DIR_ASSETS}/pickles"

for p in [k for k in globals().keys() if k.startswith("DIR")]:
    os.makedirs(globals()[p], exist_ok=True)


# input
EMBEDDING_SIZE = 40

# classification
NUM_CLASSES = 2
OUT_SIZE = 1

# data Preprocessing input_shape=[MFB,TB]
MEL_FREQUENCY_BANDS = 64  # MFB
TEMPORAL_BANDS = 22  # TB


# model
MODEL = "classifier"
CALLBACK = "general"
MODEL_ANNOT = "exp_16.10"
INPUT_SHAPE = (None, MEL_FREQUENCY_BANDS, TEMPORAL_BANDS, 1)
LOSS_FUNC = tf.keras.losses.MeanSquaredError()
METRICS = [
    "mae",
    tfa.metrics.RSquare(dtype=tf.float32, y_shape=(OUT_SIZE,)),
]
OUT_1, OUT_2 = "arousal", "valence"
OUT = OUT_1


# pickle
MAX_PICKLE_SIZE = 500_000_000  # aka 500 mb
PICKLE_ANOT = f"data_chunk_list_500ms_{MEL_FREQUENCY_BANDS}x{TEMPORAL_BANDS}_single"


# train
DATA_POINTS = 744 * 90
BATCH_SIZE = 128
STEP_SIZE = 10  # number of batchs per step
NUM_STEP = ceil(DATA_POINTS / (BATCH_SIZE))  # number of steps per epoch

EPOCHS = 1000
EPOCH_START = 0
EPOCH_END = EPOCHS

TRAIN_STEP_START = 0


# model saving
SAVE_MODEL_EVERY_STEP = NUM_STEP * 20
SAVE_MODEL_EVERY_EPOCH = 10

SAVE_MODEL_ANNOT = f"saved_instance_{MODEL_ANNOT}_{OUT}"

# logging
MAX_LOGFILE_SIZE = 1_000_000  # 1 mb

# classification
NUM_CLASSES = 2
OUT_SIZE = 1


# optimizers
LEARNING_RATE = 5e-4
LR_FEATURE = LEARNING_RATE
LR_AROUSAL = LEARNING_RATE
LR_VALENCE = LEARNING_RATE

# tensorboard
RUN_ANNOT = f"{MODEL_ANNOT}.{EPOCH_START}.{LR_FEATURE}"

β_1 = 0.9  # momentum i.e avg of 10 of δw
β_2 = 0.999  # rmsprob i.e avg of 1000 of δw square
