"""Help functions for logging"""

import logging
from logging import handlers
from time import time


def singleton(cls):
    """For a single run, only one logger is allowed"""
    _instance = {}

    def get_instance(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return get_instance


@singleton
class Logger(object):
    """Common logger for each module. Singleton class"""

    def __init__(self,
                 file_name: str = "",
                 log_level: str = "info",
                 fmt: str = '%(asctime)s - %(levelname)s: %(message)s'):
        """
        Args:
            file_name (str): name of the log file
            log_level (str): "debug", "info", "warn", "error"
            fmt (str): format string for python logging module
        """
        if file_name == "":
            # by default, for each run, we save all logs in a file
            # names `uihash_<timestamp>.log`
            file_name = f"../uihash_{time()}.log"
        self._logger = logging.getLogger(file_name)
        format_str = logging.Formatter(fmt)
        self._logger.setLevel(logging.getLevelName(log_level.upper()))

        # to screen
        self._sh = logging.StreamHandler()
        self._logger.addHandler(self._sh)
        self._sh.setFormatter(format_str)

        # to file
        self._th = handlers.TimedRotatingFileHandler(
            filename=file_name, backupCount=0, encoding='utf-8')
        self._th.setFormatter(format_str)
        self._logger.addHandler(self._th)

    @property
    def get_logger(self):
        return self._logger

    def __del__(self):
        # clean jobs
        self._logger.removeHandler(self._sh)
        self._logger.removeHandler(self._th)


if __name__ == '__main__':
    log = Logger()
    log.get_logger().debug('debug')
    log.get_logger.info('info')
