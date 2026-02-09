import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: os.PathLike = '', level = logging.INFO) -> logging.Logger:
    # logging
    logger = logging.getLogger('easybfe')
    logger.propagate = False
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [PID:%(process)d] [%(name)s]: %(message)s"
    )

    # file
    if log_file is not None:
        handler = RotatingFileHandler(str(log_file), maxBytes=50 * 1024 * 1024, backupCount=5)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger
