import logging

LOG_FORMAT = "[%(asctime)s][%(levelname)s]%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# TODO: support tensorboard.
def get_logger():
    return logging.getLogger()
