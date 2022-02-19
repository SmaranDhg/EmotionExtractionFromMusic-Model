import logging
from .setup import DIR_LOGS, MAX_LOGFILE_SIZE, MODEL_ANNOT
import sys, os


logging.basicConfig(level=logging.DEBUG)


options = dict(
    PREPROCESS=f"{DIR_LOGS}/Preprocessing/M:{MODEL_ANNOT}",
    TRAIN=f"{DIR_LOGS}/Training/M:{MODEL_ANNOT}",
    TEST=f"{DIR_LOGS}/Testing/M:{MODEL_ANNOT}",
)
for k in options.keys():
    setattr(sys.modules[__name__], k, k)


def get_logger(for_=TRAIN, **kwargs):
    logger = logging.getLogger(for_)

    i = 0
    file = f"{options[for_]}_{i}.log"
    while (
        os.path.exists(file)
        or not os.makedirs(os.path.dirname(file), exist_ok=True)
        and open(file, "w")
    ) and os.path.getsize(file) > MAX_LOGFILE_SIZE:
        i += 1
        file = f"{options[for_]}_{i}.log"

    if kwargs.get("redirect_sysout", True):
        sys.stdout = open(file, "a")
    f_handler = logging.FileHandler(file)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    setattr(
        logger.__class__,
        "flush",
        lambda self: [h.flush() for h in self.handlers] + [sys.stdout.flush()],
    )
    setattr(
        logger.__class__,
        "print",
        lambda self, s, *a, **kwarg: print(s, *a, flush=True, **kwarg),
    )

    return logger
