import logging
import os.path
import sys
import datetime
from pathlib import Path

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(levelname)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

def set_logging():
    save_path = os.path.join(Path(__file__).parent.parent, 'logs')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "log_" + timestamp
    save_path = os.path.join(save_path, log_dirpath)
    os.mkdir(save_path)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    ch = logging.FileHandler(os.path.join(save_path, "log.txt"))
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)

    return logger, save_path
