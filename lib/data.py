# %%
import os
import pandas as pd
import numpy as np
from librosa import load, power_to_db
from librosa.feature import melspectrogram
import tensorflow as tf
import re
import sys

from .log import *
from .setup import (
    DIR_PICKLE,
    NUM_STEP,
    PICKLE_ANOT,
    STEP_SIZE,
    MAX_PICKLE_SIZE,
    BATCH_SIZE,
)
import time
import pickle
from .preprocess import *


pat = r".*/(\d+).mp3"

logger = get_logger(PREPROCESS, redirect_sysout=False)
logger.info("Data Preprocessing Session started!!")


# smaran datapaths
data_dir = f"{os.environ['data1']}/_classification/MusicEmotions"

anot_dir = f"{data_dir}/annotations"
data_45_dir = f"{data_dir}/clips_45sec/clips_45seconds"
default_feature_dir = f"{data_dir}/default_features"


# %%
ls_dir = lambda x: f"{x}/" + np.array(os.listdir(x), dtype=object)
anots = ls_dir(anot_dir)
clips_45 = dict(
    map(
        lambda x: (int(re.match(pat, x).group(1)), x),
        ls_dir(f"{data_45_dir}/test").tolist()
        + ls_dir(f"{data_45_dir}/train").tolist()
        + ls_dir(f"{data_45_dir}/val").tolist(),
    )
)
# %%

[
    "/media/smaran/Storage/_Datasets/_classification/MusicEmotions/annotations/arousal_cont_average.csv"
    "/media/smaran/Storage/_Datasets/_classification/MusicEmotions/annotations/arousal_cont_std.csv"
    "/media/smaran/Storage/_Datasets/_classification/MusicEmotions/annotations/songs_info.csv"
    "/media/smaran/Storage/_Datasets/_classification/MusicEmotions/annotations/static_annotations.csv"
    "/media/smaran/Storage/_Datasets/_classification/MusicEmotions/annotations/valence_cont_average.csv"
    "/media/smaran/Storage/_Datasets/_classification/MusicEmotions/annotations/valence_cont_std.csv"
]
avg_arousal_df = pd.read_csv(anots[0])
avg_valence_df = pd.read_csv(anots[-2])
std_arousal_df = pd.read_csv(anots[1])
std_valence_df = pd.read_csv(anots[-1])

DONE = 0


def song_to_tf(song_id):
    global DONE
    t = time.time()
    pat = r"(?:.*/)?(\d+).mp3"

    song_id = song_id.numpy()
    song_id_ = song_id.decode("utf-8")
    song_id = int(re.match(pat, song_id_).group(1))

    try:

        X, y1, y2 = pickle.load(
            open(f"{DIR_PICKLE}/{PICKLE_ANOT}_{song_id}.pickle", "rb")
        )
        logger.info(f"{song_id} Loaded from pickle.")
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            logger.exception(e)
        logger.info(f"{song_id} Prepocessing: {song_id_} ")

        song, _ = load(clips_45[song_id])
        valence = avg_valence_df[avg_valence_df.song_id == song_id]
        valence = valence.values[0, 1:]

        arousal = avg_arousal_df[avg_arousal_df.song_id == song_id]
        arousal = arousal.values[0, 1:]
        # logger.print(arousal, song_id)

        specs, a, v = break_clip_to(song, arousal, valence)
        # logger.print("shapes:", a.shape, v.shape)

        X, y1, y2 = (
            tf.stack(specs),
            tf.stack(a[:, np.newaxis]),
            tf.stack(v[:, np.newaxis]),
        )

        # logger.print(y1.shape, y2.shape)

        pickle.dump(
            [X, y1, y2], open(f"{DIR_PICKLE}/{PICKLE_ANOT}_{song_id}.pickle", "wb")
        )

    DONE += 90
    logger.info(f"{song_id} preprocessed: {time.time()-t} Done:{DONE}")
    return X, y1, y2


# using tensorflow dataset API
def pickle_to_tf(filepath):
    items = np.array(pickle.load(open(filepath.numpy(), "rb")))
    y = np.stack(items[:, 3])
    X, y1, y2 = (
        tf.stack(items[:, 2].tolist()),
        tf.convert_to_tensor(y[:, 0][:, np.newaxis], dtype=tf.float32),
        tf.convert_to_tensor(y[:, 1][:, np.newaxis], dtype=tf.float32),
    )
    return X, y1, y2


def train_dataset(**kwargs):
    if kwargs.get("from_pickle", True):
        pat = f"{os.path.abspath('.')}/{DIR_PICKLE}/{PICKLE_ANOT}_*train.pickle"
        preprocess = pickle_to_tf
    else:
        pat = f"{data_45_dir}/train/*.mp3"
        preprocess = song_to_tf
    return get_dataset(pat, preprocess, **kwargs)


def test_dataset(**kwargs):
    if kwargs.get("from_pickle", True):
        pat = f"{os.path.abspath('.')}/{DIR_PICKLE}/{PICKLE_ANOT}_*test.pickle"
        preprocess = pickle_to_tf
    else:
        pat = f"{data_45_dir}/test/*.mp3"
        preprocess = song_to_tf
    return get_dataset(pat, preprocess, **kwargs)


def val_dataset(**kwargs):
    if kwargs.get("from_pickle", True):
        pat = f"{os.path.abspath('.')}/{DIR_PICKLE}/{PICKLE_ANOT}_*val.pickle"
        preprocess = pickle_to_tf
    else:
        pat = f"{data_45_dir}/val/*.mp3"
        preprocess = song_to_tf
    return get_dataset(pat, preprocess, **kwargs)


def set_shape(X, y):
    logger.info("Setting Shape")
    X.set_shape(INPUT_SHAPE)
    y[OUT_1].set_shape([None, 1])
    y[OUT_2].set_shape([None, 1])
    return X, y


def select_out(X, y):
    logger.info(f"Selected Out: {OUT} {y[OUT].shape}")
    y = y[OUT]
    return X, y


def anot_exist(x):
    pat = r"(?:.*/)?(\d+).mp3"
    x = x.numpy()
    x = x.decode("utf-8")
    x = int(re.match(pat, x).group(1))
    if np.random.random() < 0.5:
        return False
    return (avg_valence_df.song_id == x).any()


def get_dataset(file_pat, preprocess, **kwargs):

    dataset = tf.data.Dataset.list_files(file_pat, shuffle=True)
    dataset = dataset.filter(lambda x: tf.py_function(anot_exist, [x], tf.bool))
    # print(list(dataset.as_numpy_iterator()))

    dataset = dataset.map(
        lambda x: tf.py_function(preprocess, [x], [tf.float32, tf.float32, tf.float32]),
        # num_parallel_calls=tf.data.AUTOTUNE,
        name="song_to_tensor",
    )
    dataset = dataset.flat_map(
        lambda X, y1, y2: tf.data.Dataset.from_tensor_slices(
            (X, {OUT_1: y1, OUT_2: y2})
        ),
    )
    dataset = dataset.shuffle(
        buffer_size=kwargs.get("shuffle_buffer_size", 6660),
        reshuffle_each_iteration=True,
    )
    dataset = dataset.batch(kwargs.get("batch_size", BATCH_SIZE))
    dataset = dataset.prefetch(64)  # 64 batch
    dataset = dataset.map(
        set_shape,
        # num_parallel_calls=20,
    )
    dataset = dataset.map(
        select_out,
        # num_parallel_calls=20,
    )
    dataset = dataset.cache()
    if kwargs.get("repeat", True):
        dataset = dataset.repeat()

    return dataset
