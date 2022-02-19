import numpy as np
import pandas as pd
from .setup import *
from librosa.feature import *
from librosa import *


def one_hot_encoder(cats):
    def _en(val):
        ret = np.zeros(len(cats), dtype=np.float32)
        ret[cats == val] = 1
        return ret

    return np.frompyfunc(_en, 1, 1)


__to_mel_spec = np.frompyfunc(
    lambda x: melspectrogram(
        x,
        hop_length=np.power(2, int(np.ceil(np.log2(22050 / TEMPORAL_BANDS)))),
        n_mels=MEL_FREQUENCY_BANDS,
    ),
    1,
    1,
)
__to_power = np.frompyfunc(lambda x: power_to_db(x), 1, 1)

__normalize = np.frompyfunc(lambda x: x / np.linalg.norm(x), 1, 1)

__extend = np.frompyfunc(
    lambda x: tf.image.resize(
        x[:, :, np.newaxis], (MEL_FREQUENCY_BANDS, TEMPORAL_BANDS)
    ),
    1,
    1,
)


def break_clip_to(song, arousal, valence, i_sec=0.5, len_sec=45, f=22050):
    song = song[abs(len(song) % len_sec - 1) :]

    specs = np.array(np.array_split(song, len_sec / i_sec),dtype=object)
    f = len(specs[0]) / i_sec

    specs = __to_mel_spec(specs)
    specs = __to_power(specs)
    specs = __normalize(specs)
    specs = __extend(specs)

    specs = np.stack(specs)

    a_ = np.interp(np.arange(0, f * 30, f), [-f, f * 29], [0, arousal[0]])
    arousal = np.concatenate([a_, arousal[1:]])

    v_ = np.interp(np.arange(0, f * 30, f), [-f, f * 29], [0, valence[0]])
    valence = np.concatenate([v_, valence[1:]])

    return specs, arousal.astype("f4"), valence.astype("f4")
