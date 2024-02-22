"""Help functions for the Android app feature extraction platform"""

import sys
import logging


def init_path(path: str) -> str:
    """Clean the input path to make it valid in both NT and POSIX"""
    return f"{path}/". \
        replace("//", "/").replace(r"\/", "/")


def clean_package_name(package_name: str) -> str:
    """In windows, dirs shart with certain prefixes are not allowed.
    This function rename the invalid names.

    Args:
        package_name (str): An input package name

    Returns:
        A valid package name (str)
    """
    for _invalid_prefix in ['aux', 'com1', 'com2', 'prn', 'con', 'nul']:
        if package_name.startswith(_invalid_prefix):
            return '_' + package_name
    return package_name


def get_logger(class_name: str, log_level: str) -> logging.Logger:
    """Logger for Android app feature extraction platform

    Args:
        class_name (str): The caller
        log_level (str): Log level. 'debug' for debug, and 'warn' for
          warn. In other cases, the level will be info

    Returns:

    """
    level = logging.INFO
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "warn":
        level = logging.WARNING
    logger = logging.getLogger(class_name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logger.addHandler(console_handler)
    return logger


def check_ok(dic: dict, key: str) -> bool:
    """Check if a feature is already successfully printed"""
    if key not in dic:
        return False
    if dic[key] != "ok":
        return False
    return True
