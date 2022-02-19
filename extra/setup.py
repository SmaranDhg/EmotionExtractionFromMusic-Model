# import tensorflow as tf
import os


DIR_SAVE_MODEL = "saved_models"
DIR_LOGS = "logs"
DIR_ASSETS = "assets"
DIR_RUN = "runs"
DIR_PICKLE = f"{DIR_ASSETS}/pickles"

# input
EMBEDDING_SIZE = 40


# model
MODEL = "classifier"
CALLBACK = "general"
MODEL_ANNOT = ""
INPUT_SHAPE = None
# LOSS_FUNC = tf.keras.losses.SparseCategoricalCrossentropy()
METRICS = ["accuracy"]

# model saving
SAVE_MODEL_EVERY_STEP = 10
SAVE_MODEL_ANNOT = f"saved_instance_{MODEL_ANNOT}"

# pickle
MAX_PICKLE_SIZE = 500_000_000  # aka 500 mb
PICKLE_ANOT = "data_chunk"
PICKLE_ANOT = "data_chunk_list"
PICKLE_ANOT = "data_chunk_list_500ms"


# train
BATCH_SIZE = 512
STEP_SIZE = 10  # number of batchs per step
NUM_STEP = 10  # number of steps per epoch

EPOCHS = 100
EPOCH_START = 0
EPOCH_END = EPOCHS

TRAIN_STEP_START = 0

RUN_ANNOT = ""
SAVE_MODEL_ANNOT = f"{MODEL_ANNOT}_"

# logging
MAX_LOGFILE_SIZE = 10_000_000  # 10 mb

# classification
NUM_CLASSES = 2
OUT_SIZE = 2


# optimizers
LEARNING_RATE = 1e-4
MOMENTUM = 0.8
