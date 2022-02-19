# %%
# from tensorboardX import SummaryWriter
# import tensorflow_addons as tfa

from sympy import cycle_length
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from lib.models import get_model
from lib.callbacks import get_callback
from lib.data import train_dataset, val_dataset
from lib.log import *
from lib.setup import *
from tensorflow.keras.utils import plot_model
import tensorflow_addons as tfa

logger = get_logger(TRAIN)
Model = get_model()
Callback = get_callback()

checkpoint_path = f"{DIR_SAVE_MODEL}/{SAVE_MODEL_ANNOT}/epoch_{EPOCH_START}"
writer = tf.summary.create_file_writer(f"{DIR_RUN}/{RUN_ANNOT}/")

step = min(TRAIN_STEP_START, NUM_STEP)
callback = Callback(writer, logger, epoch=EPOCH_START, step=step)

optimizers = [
    tf.keras.optimizers.Adam(learning_rate=LR_FEATURE, beta_1=β_1, beta_2=β_2),
    tf.keras.optimizers.Adam(learning_rate=LR_AROUSAL, beta_1=β_1, beta_2=β_2),
    tf.keras.optimizers.Adam(learning_rate=LR_VALENCE, beta_1=β_1, beta_2=β_2),
]
loss_weights = [1, 0.0]
loss_weights = None
loss = {OUT_1: LOSS_FUNC, OUT_2: LOSS_FUNC}
loss = LOSS_FUNC

try:
    logger.info(f"Loading Saved Model: {checkpoint_path}!!")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={"Model": Model},
    )

    layer_optimizers = []
    for o, seg in zip(optimizers, model.segments):
        layer_optimizers += list(zip([o] * len(seg), seg))
    optimizer = tfa.optimizers.MultiOptimizer(layer_optimizers)

    model.compile(
        loss=loss,
        optimizer=optimizers[0],
        metrics=METRICS,
        loss_weights=loss_weights,
    )
    model.build(INPUT_SHAPE)
    logger.info("Saved Model Loaded!")

except Exception as e:
    logger.info("Creating new Model.")
    model = Model()
    layer_optimizers = []
    for o, seg in zip(optimizers, model.segments):
        layer_optimizers += list(zip([o] * len(seg), seg))
    optimizer = tfa.optimizers.MultiOptimizer(layer_optimizers)

    model.compile(
        loss=loss,
        optimizer=optimizers[0],
        metrics=METRICS,
        loss_weights=loss_weights,
    )
    model.build(INPUT_SHAPE)

logger.print(model.summary())
try:
    plot_model(
        model.model(),
        to_file=f"{DIR_ASSETS}/model_{MODEL_ANNOT}:D.png",  # detail plot
        show_shapes=True,
        show_dtype=True,
        expand_nested=True,
        show_layer_activations=True,
    )

    plot_model(
        model.model(),
        to_file=f"{DIR_ASSETS}/model_{MODEL_ANNOT}:A.png",  # Abstract plot
        show_shapes=True,
        show_layer_activations=True,
    )
except:
    pass

dataset_train = train_dataset(from_pickle=False)
dataset_val = val_dataset(from_pickle=False)
model.fit(
    dataset_train,
    callbacks=[callback],
    epochs=EPOCHS,
    initial_epoch=EPOCH_START,
    steps_per_epoch=ceil(65070 / BATCH_SIZE),
    use_multiprocessing=True,
    validation_data=dataset_val,
    validation_steps=ceil((DATA_POINTS - 65070) / BATCH_SIZE),
    validation_freq=1,
)


model.save(f"{DIR_SAVE_MODEL}/{SAVE_MODEL_ANNOT}/epoch_{callback.epoch}")
