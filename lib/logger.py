from asyncio.log import logger
import logging
import coloredlogs
import os


def setup_log(name='train', log_dir=None, file_name=None):
    logger = None
    if not logger:
        logger = get_logger(
            name, log_dir, 0, filename=file_name)
    else:
        logger.warning('already exists logger')
    return logger

# todo remove color when logging in file
def get_logger(name='', save_dir=None, distributed_rank=0, filename="log.txt"):
    logger = logging.getLogger(name)
    coloredlogs.install(level='DEBUG', logger=logger)
    # logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s")

    # ch = logging.StreamHandler(stream=sys.stdout)
    # ch.setLevel(logging.DEBUG)
    # ch.setFormatter(formatter)
    # logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
