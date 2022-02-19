# %%
import os
import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from librosa import *
import re
from librosa.display import *
from librosa.feature import *
import tensorflow as tf
from setup import *


pat = r".*/(\d+).mp3"


# smaran datapaths
data_dir = f"{os.environ['data1']}/_classification/MusicEmotions"

anot_dir = f"{data_dir}/annotations"
data_45_dir = f"{data_dir}/clips_45sec/clips_45seconds"
default_feature_dir = f"{data_dir}/default_features"
# %%
ls_dir = lambda x: f"{x}/" + np.array(os.listdir(x), dtype=np.object)
anots = ls_dir(anot_dir)
clips_45 = dict(map(lambda x: (int(re.match(pat, x).group(1)), x), ls_dir(data_45_dir)))

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


# %%
import tensorflow_io as tfio

song_id = 3
song, _ = load(clips_45[song_id])
valence = avg_valence_df[avg_valence_df.song_id == song_id]
valence = valence[1:].values

arousal = avg_arousal_df[avg_arousal_df.song_id == song_id]
arousal = arousal[1:].values
# file.shape
# tf.contrib.ffmpeg.decode_audio(file, file_format='mp3', samples_per_second=44100, channel_count=1)
# %%
l = len(avg_arousal_df)

for i in range(l):
    row = avg_arousal_df.iloc[i]
    song_id, avg_arousal_v = int(row[0]), row[1:]
# %%
import sys
import pickle

avg_arousal_v.values.shape
sys.getsizeof(avg_arousal_v)

# %%
shapes = []
for i in range(1, l)[:10]:
    s, f = load(clips_45[i])
    s.shape
    spec = np.abs(melspectrogram(s, hop_length=512))
    shapes.append(spec.shape)

# %%
shapes
# %%
import tensorflow as tf

tf.image.resize(spec[:, :, np.newaxis], (128, 1941)).shape

# %%
step = 0
i = step * STEP_SIZE * BATCH_SIZE
pat = rf".*/{PICKLE_ANOT}_(\d+).pickle"

files = ls_dir(f"../{DIR_PICKLE}")
sample = np.array([int(re.match(pat, f).group(1)) for f in files if re.match(pat, f)])
s = sample[sample > i][-1]
s

# %%

df = pickle.load(open(f"../{DIR_PICKLE}/{PICKLE_ANOT}_{126}.pickle", "rb"))
# %%
df = np.array(df)

s_id = df[:, 0]
s = df[:, 1]
X = df[:, 2]
y = df[:, 3]
len(df)
# %%
tf.stack(X).shape
tf.stack(y).shape

# %%
s[0]
import sounddevice as sd

m = df[100]
print(m[0])
sd.play(m[1], 22054)
figure(figsize=(10, 2))
plot(m[1])
show()
figure(figsize=(10, 2))
plot(m[3])
show()
specshow((m[2][:, :, 0].numpy()), x_axis="time", y_axis="log")
show()
s, _ = load(clips_45[m[0]])
X = np.abs(melspectrogram(s))
X = power_to_db(X, ref=np.max)
specshow(X, x_axis="time", y_axis="log")


# %%
# %%
DIR_SAVE_MODEL = "saved_models"
DIR_LOGS = "logs"
DIR_ASSETS = "assets"
DIR_RUN = "runs"
DIR_PICKLE = f"{DIR_ASSETS}/pickles"
import pickle


pat = f"{os.path.abspath('.')}/{DIR_PICKLE}/{PICKLE_ANOT}_*.pickle"


train_files = tf.data.Dataset.list_files(pat, shuffle=True)

train_dataset = train_files.interleave(
    lambda x: train_files,
    cycle_length=4,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


def load(filepath):
    items = np.array(pickle.load(open(filepath.numpy(), "rb")))
    return tf.stack(items[:, 2].tolist()), tf.stack(items[:, 3].tolist())


train_dataset = train_dataset.map(
    lambda x: tf.py_function(load, [x], [tf.float32, tf.float32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
).flat_map(lambda X, y: tf.data.Dataset.from_tensor_slices((X, y)))
train_dataset = train_dataset.shuffle(buffer_size=512).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.cache()
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


def get_dataset(file_pat, preprocess):

    files = tf.data.Dataset.list_files(file_pat, shuffle=True)
    dataset = files.interleave(
        lambda x: files,
        cycle_length=4,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.map(
        lambda x: tf.py_function(preprocess, [x], [tf.float32, tf.float32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    ).flat_map(lambda X, y: tf.data.Dataset.from_tensor_slices((X, y)))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=512).batch(BATCH_SIZE).repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


# %%

dataset = train_dataset.as_numpy_iterator()

i = 0
# %%
for i in range(NUM_STEP * 3):
    i += 1
    X, y = dataset.__next__()
    X.shape, y.shape

    v = 7
    specshow(X[v, :, :, 0])
    show()
    print(y[v], i)

# %%
# %%
def preprocess_and_cache():
    logger.print(f"{'':-<70s}\nPreprocessing Session started!!")
    l = len(avg_arousal_df)
    i = 0
    chunck = 0
    tss = time.time()
    while i < l:
        ts = time.time()
        df = []
        size = 0
        while size < MAX_PICKLE_SIZE and i < l:
            t = time.time()
            row = avg_arousal_df.iloc[i]
            song_id, y = int(row[0]), row[1:].values
            y = tf.convert_to_tensor(y)
            s, _ = load(clips_45[song_id])
            X = np.abs(melspectrogram(s))
            X = power_to_db(X, ref=np.max)
            X = tf.image.resize(X[:, :, np.newaxis], (128, 1941))
            size += (
                sys.getsizeof(X)
                + sys.getsizeof(y)
                + sys.getsizeof(s)
                + sys.getsizeof(song_id)
            )
            df.append([song_id, s, X, y])
            i += 1
            logger.info(f"{i} preprocessed: {time.time()-t} total_size: {size}.")
        pickle.dump(df, open(f"{DIR_PICKLE}/{PICKLE_ANOT}_{i}.pickle", "wb"))
        logger.info(f"Cached {chunck}: {time.time()-ts}")

        chunck += 1
    logger.info(f"Finised Preprocessing and Caching: {time.time()-tss}")


def get_train(step=0, epoch=0, **kwargs):
    samples_per_step = STEP_SIZE * BATCH_SIZE
    l = len(avg_arousal_df)
    i = 0

    if step:
        step -= 1

    if kwargs.get("from_pickle", False):  # incase if data is already pickled
        get_train = ___get_train(step)
        while True:
            yield next(get_train)

    while True:
        logger.info(f"Epoch:  {epoch}")

        if epoch and kwargs.get(
            "from_pickle", True
        ):  # after one cycle just use the pickled data
            get_train = ___get_train(step)
            while True:

                yield next(get_train)

        i = step * int(samples_per_step / 90)
        song_ids, songs, X, y = [], [], [], []
        samples = 0

        logger.info(f"Preprocessing:  {epoch}")

        while i < l:  # for fresh preprocessing of data
            n = 0

            ts = time.time()

            while i < l and n < samples_per_step:
                t = time.time()

                arousal = avg_arousal_df.iloc[i]
                valence = avg_valence_df.iloc[i]

                song_id, arousal = int(arousal[0]), arousal[1:].values
                valence = valence[1:].values

                song, _ = load(clips_45[song_id])
                spec, av = break_clip_to(song, arousal, valence)

                n += len(spec)
                samples += len(spec)
                i += 1

                song_ids += [song_id] * len(spec)
                songs += [song] * len(spec)

                X += spec.tolist()
                y += av.tolist()
                logger.info(
                    f"{samples} preprocessed: {time.time()-t} total_songs: {i}."
                )

            _l = min(len(X), samples_per_step)

            pickle.dump(
                np.array(list(zip(song_ids, songs, X, y))).tolist(),
                open(f"{DIR_PICKLE}/{PICKLE_ANOT}_{samples}.pickle", "wb"),
            )
            logger.info(f"Cached {samples}: {time.time()-ts}")

            yield tf.stack(X[:_l]), tf.stack(y[:_l])

            if kwargs.get("song", False):
                yield song_ids, songs

            song_ids, songs, X, y = song_ids[_l:], songs[_l:], X[_l:], y[_l:]

        epoch += 1
        step = 0


def ___get_train(step=0, **kwargs):
    pat = rf".*/{PICKLE_ANOT}_(\d+).pickle"
    samples_per_step = STEP_SIZE * BATCH_SIZE
    files = ls_dir(DIR_PICKLE)
    # print(files)
    sample = np.array(
        [int(re.match(pat, f).group(1)) for f in files if re.match(pat, f)]
    )
    print(sample)
    sample.sort()
    if step:
        step -= 1
    while True:
        i = step * samples_per_step
        print(sample > i)
        song_id, song, X, y = [], [], [], []

        while (sample > i).any():
            n = 0
            s = sample[sample > i][0]
            print(i)
            t = time.time()
            while (sample > i).any() and n < samples_per_step:
                s = sample[sample > i][0]
                tt = time.time()
                items = pickle.load(
                    open(f"{DIR_PICKLE}/{PICKLE_ANOT}_{s}.pickle", "rb")
                )

                logger.info(f"{s} Depickled: {time.time()-tt}")
                items = np.array(items)
                n += len(items)

                song_id += items[:, 0].tolist()
                song += items[:, 1].tolist()

                X += items[:, 2].tolist()
                y += items[:, 3].tolist()

            _l = min(len(X), samples_per_step)

            logger.info(f"{n} Fetched: {time.time()-t}")
            yield tf.stack(X[:_l]), tf.stack(y[:_l])

            if kwargs.get("song", False):
                yield song_id[:_l], song[:_l]

            song_id, song, X, y = song_id[_l:], song[_l:], X[_l:], y[_l:]
        step = 0

