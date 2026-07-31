import logging


def configure_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("finagent.api")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger