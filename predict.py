# %%
# from tensorboardX import SummaryWriter

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from lib.setup import *

from lib.models import get_model
from lib.callbacks import get_callback

from lib.data import train_dataset
from lib.log import *

import os
import pandas as pd
import numpy as np
from matplotlib.pyplot import *
from librosa import *
from librosa.display import *
from librosa.feature import *

import sounddevice as sd

# %%

logger = get_logger(TEST, redirect_sysout=False)
Model = get_model()
Callback = get_callback()

epoch = 0
checkpoint_path = f"{DIR_SAVE_MODEL}/{SAVE_MODEL_ANNOT}/epoch_{epoch}"
logger.info(f"Loading Saved Model: {checkpoint_path}!!")
model = tf.keras.models.load_model(checkpoint_path, compile=False)
logger.info("Saved Model Loaded!")


# %%

train_gen = iter(train_dataset(from_pickle=False))
# %%

X, y = next(train_gen)
X, y = X[:25], y[:25]
# song_ids, songs = next(train_gen)
y_pred = model.predict(X)


f = 22050
i = np.random.randint(0, len(y) - 1)

# s = songs[i]
arousal = y
arousal_pred = y_pred
# song_id = song_ids[i]
spec = X[i].numpy()
song_id = i

st = 0
ed = -1
# s = s[int(f * st / 2) : int(f * ed / 2)]
# arousal = arousal[st:ed]

figure(figsize=(25, 16))
# subplot2grid((5, 1), (0, 0))
# title(f"Plot song:{song_id}")
# plot(s)
# show()


subplot2grid((5, 1), (1, 0))

title(f"Arousal song: {song_id}")
plot(arousal)
# show()

subplot2grid((5, 1), (2, 0))


title(f"Predicted Arousal: {song_id}")
plot(arousal_pred)
# show()

subplot2grid((5, 1), (3, 0), rowspan=2)
title(f"Mel Spectrum: {song_id}")
specshow(spec[:, :, 0], sr=f, x_axis="time", y_axis="log")
# colorbar(format="%+2.0f dB")
path = f"{DIR_ASSETS}/predictions_{MODEL_ANNOT}"
os.path.exists(path) or os.makedirs(path)
savefig(
    f"{path}/prediction_{song_id}_M:{MODEL_ANNOT}_E:{epoch}.jpg",
)
# show()
# spec.__module__
# %%
arousal_pred
