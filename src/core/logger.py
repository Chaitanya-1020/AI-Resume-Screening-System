import logging
import sys
from .config import settings

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(settings.LOG_LEVEL)

    # File Handler
    fh = logging.FileHandler('app.log')
    fh.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
