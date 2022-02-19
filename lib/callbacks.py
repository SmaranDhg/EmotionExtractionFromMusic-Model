import tensorflow as tf
import numpy as np

from .setup import (
    CALLBACK,
    BATCH_SIZE,
    DATA_POINTS,
    EPOCHS,
    OUT,
    OUT_1,
    OUT_2,
    SAVE_MODEL_EVERY_EPOCH,
    STEP_SIZE,
    NUM_STEP,
    SAVE_MODEL_EVERY_STEP,
    DIR_SAVE_MODEL,
    SAVE_MODEL_ANNOT,
)

__callbacks = ["general"]


class __ModelCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, writer, logger, epoch=0, step=0):
        self.writer = writer
        self.epoch = epoch
        self.logger = logger
        self.pos = int(epoch * DATA_POINTS / BATCH_SIZE)
        pass

    def on_train_batch_end(self, batch, logs):
        self.pos += 1
        with self.writer.as_default():
            for lbl, out in zip([OUT.capitalize()], [OUT]):
                for k, v in logs.items():
                    tf.summary.scalar(
                        f"{lbl} {k.replace('_',' ').title()}",
                        v,
                        step=self.pos,
                    )

            pass

    def on_epoch_end(self, epoch, logs):
        self.epoch = epoch
        save_interval = max(
            5,
            int(SAVE_MODEL_EVERY_EPOCH - (EPOCHS - epoch) / int(EPOCHS * 0.95)),
        )

        if self.epoch and not epoch % save_interval:
            try:
                self.model.save(
                    f"{DIR_SAVE_MODEL}/{SAVE_MODEL_ANNOT}/epoch_{self.epoch}"
                )
                self.logger.info(
                    f"Model Saved Succesfully :{DIR_SAVE_MODEL}/{SAVE_MODEL_ANNOT}/epoch_{self.epoch}!"
                )
            except:
                pass

        with self.writer.as_default():

            for lbl, out in zip([OUT.capitalize()], [OUT]):
                for k, v in logs.items():
                    if "val_" in k:
                        tf.summary.scalar(
                            f"Validation {lbl} {k.replace('_',' ').title()}",
                            v,
                            step=self.pos,
                        )
            pass


def get_callback():
    callback = dict(zip(__callbacks, [__ModelCallbacks]))
    return callback[CALLBACK]
